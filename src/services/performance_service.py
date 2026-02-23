"""Service for evaluating and monitoring model performance per output field."""

import logging
from datetime import datetime, timezone

import numpy as np
import mlflow
from mlflow.exceptions import MlflowException

from src.schemas.model import ArchitectureType, ModelConfig
from src.schemas.performance import FieldEvaluationResponse, ModelPerformance
from src.services.config_service import MLConfigService
from src.services.data_preparation import calculate_timestamps, prepare_last_sequence
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)

_TAG_RMSE = "rmse_for:{field}"
_TAG_EVAL_AT = "eval_at:{field}"
_TAG_BEST = "best_for:{field}"


def _rmse_key(field_name: str) -> str:
    return f"rmse_for:{field_name}"


def _eval_at_key(field_name: str) -> str:
    return f"eval_at:{field_name}"


def _best_key(field_name: str) -> str:
    return f"best_for:{field_name}"


class PerformanceService:
    """Service for RMSE-based model evaluation and best-model designation."""

    MAX_EVAL_CELLS = 3

    def __init__(
        self,
        mlflow_service: MLflowService,
        ml_config_service: MLConfigService,
        data_storage_client: DataStorageClient,
    ):
        self.mlflow_service = mlflow_service
        self.ml_config_service = ml_config_service
        self.data_storage_client = data_storage_client
        self.client = mlflow_service.client

    async def evaluate_field(self, field_name: str) -> FieldEvaluationResponse:
        """
        Score ALL trained models that predict field_name.

        Computes RMSE against live cell data, persists scores as MLflow tags,
        updates the best_for:{field_name} tag, and returns a ranked response.
        """
        db_configs = [
            c
            for c in self.ml_config_service.list_all()
            if field_name in c.output_fields
        ]

        if not db_configs:
            return FieldEvaluationResponse(
                field_name=field_name,
                models=[],
                best_model_id=None,
                last_evaluated_at=None,
            )

        cells = await self.data_storage_client.get_known_cells()
        logger.info(
            f"Evaluating {len(db_configs)} models for field '{field_name}' "
            f"using up to {self.MAX_EVAL_CELLS} reference cells "
            f"(from {len(cells)} known)"
        )

        evaluated_at = datetime.now(tz=timezone.utc)
        evaluated_at_str = evaluated_at.isoformat()

        performances: list[ModelPerformance] = []
        scored: list[tuple[str, float]] = []  # (model_id, rmse)

        for db_config in db_configs:
            model_detail = self.mlflow_service.get_model(db_config.model_id)

            base = ModelPerformance(
                model_id=db_config.model_id,
                model_name=db_config.name,
                architecture=ArchitectureType(db_config.architecture),
                latest_version=model_detail.latest_version,
                training_loss=model_detail.training_loss,
                last_trained_at=model_detail.last_trained_at,
            )

            if model_detail.latest_version is None:
                # Not trained — skip scoring
                performances.append(base)
                continue

            config = model_detail.config

            try:
                model = self._load_model(db_config.model_id, model_detail.latest_version, config)
                rmse = await self._compute_rmse(model, config, field_name, cells)
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate model {db_config.model_id} for field '{field_name}': {e}"
                )
                rmse = None

            if rmse is not None:
                scored.append((db_config.model_id, rmse))

            performances.append(
                base.model_copy(update={"rmse": rmse, "evaluated_at": evaluated_at})
            )

        # Determine best
        best_model_id: str | None = None
        if scored:
            best_model_id = min(scored, key=lambda t: t[1])[0]

        # Persist tags
        self._update_tags(
            field_name=field_name,
            scored_models=scored,
            best_model_id=best_model_id,
            evaluated_at_str=evaluated_at_str,
        )

        # Mark is_best
        for p in performances:
            p.is_best = p.model_id == best_model_id

        # Sort: scored asc by RMSE, then unscored
        sorted_performances = sorted(
            performances,
            key=lambda p: (p.rmse is None, p.rmse if p.rmse is not None else 0.0),
        )

        return FieldEvaluationResponse(
            field_name=field_name,
            models=sorted_performances,
            best_model_id=best_model_id,
            last_evaluated_at=evaluated_at,
        )

    def get_cached_evaluation(self, field_name: str) -> FieldEvaluationResponse:
        """
        Return the last evaluation result from stored MLflow tags.

        No computation is triggered. Models that have never been evaluated
        will have rmse=None and evaluated_at=None.
        """
        db_configs = [
            c
            for c in self.ml_config_service.list_all()
            if field_name in c.output_fields
        ]

        if not db_configs:
            return FieldEvaluationResponse(
                field_name=field_name,
                models=[],
                best_model_id=None,
                last_evaluated_at=None,
            )

        performances: list[ModelPerformance] = []
        best_model_id: str | None = None
        last_evaluated_at: datetime | None = None

        for db_config in db_configs:
            model_detail = self.mlflow_service.get_model(db_config.model_id)
            tags = self._get_registered_model_tags(db_config.model_id)

            rmse_str = tags.get(_rmse_key(field_name))
            eval_at_str = tags.get(_eval_at_key(field_name))
            is_best_str = tags.get(_best_key(field_name))

            rmse = float(rmse_str) if rmse_str else None
            evaluated_at = datetime.fromisoformat(eval_at_str) if eval_at_str else None
            is_best = is_best_str == "true"

            if is_best:
                best_model_id = db_config.model_id

            if evaluated_at and (
                last_evaluated_at is None or evaluated_at > last_evaluated_at
            ):
                last_evaluated_at = evaluated_at

            performances.append(
                ModelPerformance(
                    model_id=db_config.model_id,
                    model_name=db_config.name,
                    architecture=ArchitectureType(db_config.architecture),
                    latest_version=model_detail.latest_version,
                    training_loss=model_detail.training_loss,
                    rmse=rmse,
                    is_best=is_best,
                    last_trained_at=model_detail.last_trained_at,
                    evaluated_at=evaluated_at,
                )
            )

        sorted_performances = sorted(
            performances,
            key=lambda p: (p.rmse is None, p.rmse if p.rmse is not None else 0.0),
        )

        return FieldEvaluationResponse(
            field_name=field_name,
            models=sorted_performances,
            best_model_id=best_model_id,
            last_evaluated_at=last_evaluated_at,
        )

    def get_best_model(self, field_name: str) -> ModelPerformance | None:
        """
        Return the model currently tagged as best for field_name.

        Returns None if no evaluation has been run yet.
        """
        db_configs = [
            c
            for c in self.ml_config_service.list_all()
            if field_name in c.output_fields
        ]

        for db_config in db_configs:
            tags = self._get_registered_model_tags(db_config.model_id)
            if tags.get(_best_key(field_name)) == "true":
                model_detail = self.mlflow_service.get_model(db_config.model_id)
                rmse_str = tags.get(_rmse_key(field_name))
                eval_at_str = tags.get(_eval_at_key(field_name))

                return ModelPerformance(
                    model_id=db_config.model_id,
                    model_name=db_config.name,
                    architecture=ArchitectureType(db_config.architecture),
                    latest_version=model_detail.latest_version,
                    training_loss=model_detail.training_loss,
                    rmse=float(rmse_str) if rmse_str else None,
                    is_best=True,
                    last_trained_at=model_detail.last_trained_at,
                    evaluated_at=datetime.fromisoformat(eval_at_str) if eval_at_str else None,
                )

        return None

    async def monitor_best_model(self, field_name: str) -> ModelPerformance:
        """
        Re-evaluate only the current best model for field_name.

        Updates rmse_for and eval_at tags but does NOT change the best_for tag
        (designation only changes via a full evaluate call).

        Raises:
            ValueError: If no best model has been designated yet.
        """
        best = self.get_best_model(field_name)
        if best is None:
            raise ValueError(
                f"No best model designated for field '{field_name}'. "
                f"Run POST /performance/{field_name}/evaluate first."
            )

        db_config = self.ml_config_service.get_config(best.model_id)
        model_detail = self.mlflow_service.get_model(best.model_id)

        if model_detail.latest_version is None:
            raise ValueError(
                f"Best model '{best.model_id}' has no trained version. "
                f"Train the model before monitoring."
            )

        config = model_detail.config
        cells = await self.data_storage_client.get_known_cells()

        model = self._load_model(best.model_id, model_detail.latest_version, config)
        rmse = await self._compute_rmse(model, config, field_name, cells)

        evaluated_at = datetime.now(tz=timezone.utc)
        evaluated_at_str = evaluated_at.isoformat()

        # Update score tags only — best_for stays unchanged
        if rmse is not None:
            self.client.set_registered_model_tag(
                best.model_id, _rmse_key(field_name), str(rmse)
            )
        self.client.set_registered_model_tag(
            best.model_id, _eval_at_key(field_name), evaluated_at_str
        )

        logger.info(
            f"Monitor: updated RMSE for best model {best.model_id} "
            f"on field '{field_name}': {rmse}"
        )

        return ModelPerformance(
            model_id=best.model_id,
            model_name=best.model_name,
            architecture=best.architecture,
            latest_version=model_detail.latest_version,
            training_loss=model_detail.training_loss,
            rmse=rmse,
            is_best=True,
            last_trained_at=model_detail.last_trained_at,
            evaluated_at=evaluated_at,
        )

    def _load_model(self, model_id: str, version: int, config: ModelConfig):
        """Load a trained PyTorch model from the MLflow registry."""
        model_class = MODEL_REGISTRY.get(config.architecture)
        if not model_class:
            raise ValueError(f"Unsupported architecture: {config.architecture}")

        model_uri = f"models:/{model_id}/{version}"
        logger.info(f"Loading model for evaluation: {model_uri}")

        try:
            loaded_pytorch_model = mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_uri}: {e}")

        if config.architecture == ArchitectureType.LSTM:
            model = model_class(
                input_fields=config.input_fields,
                output_fields=config.output_fields,
                window_duration_seconds=config.window_duration_seconds,
                lookback_steps=config.lookback_steps,
                forecast_steps=config.forecast_steps,
                hidden_size=config.hidden_size,
                num_layers=2,
            )
        else:
            model = model_class(
                input_fields=config.input_fields,
                output_fields=config.output_fields,
                window_duration_seconds=config.window_duration_seconds,
                lookback_steps=config.lookback_steps,
                forecast_steps=config.forecast_steps,
                hidden_size=config.hidden_size,
            )

        model.model = loaded_pytorch_model
        return model

    async def _compute_rmse(
        self,
        model,
        config: ModelConfig,
        field_name: str,
        cells: list[int],
    ) -> float | None:
        """
        Compute aggregate RMSE for field_name over a set of reference cells.

        For each cell, takes the most recent (lookback_steps + forecast_steps) windows,
        uses the first lookback_steps as input and the last forecast_steps as ground truth.
        """
        min_windows = config.lookback_steps + config.forecast_steps
        # Fetch a much larger window so sparse cells still yield enough data.
        # We only need min_windows entries but data may not be available in the
        # last few minutes — search the last 24 hours and use the most recent windows.
        min_secs = min_windows * config.window_duration_seconds
        fetch_secs = max(min_secs, 86400)
        start_ts, end_ts = calculate_timestamps(fetch_secs)

        field_idx = config.output_fields.index(field_name)
        num_out = len(config.output_fields)
        all_squared_errors: list[float] = []
        valid_cells = 0

        for cell_index in cells:
            if valid_cells >= self.MAX_EVAL_CELLS:
                break
            try:
                data = await self.data_storage_client.fetch_cell_data(
                    cell_index=cell_index,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    window_duration_seconds=config.window_duration_seconds,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch data for cell {cell_index}: {e}")
                continue

            if not data or len(data) < min_windows:
                logger.warning(
                    f"Cell {cell_index}: insufficient data "
                    f"({len(data) if data else 0} windows, need {min_windows})"
                )
                continue

            data.sort(key=lambda x: x.get("window_start_time", 0))

            # Take exactly the last (lookback + forecast) windows
            eval_data = data[-min_windows:]
            x_windows = eval_data[: config.lookback_steps]
            truth_windows = eval_data[config.lookback_steps :]

            # Build input tensor: shape (1, lookback_steps, num_input_fields)
            X = prepare_last_sequence(x_windows, config.input_fields, config.lookback_steps)

            # Run prediction: shape (1, forecast_steps * num_output_fields)
            try:
                pred = model.predict(X)[0]
            except Exception as e:
                logger.warning(f"Prediction failed for cell {cell_index}: {e}")
                continue

            # Extract values for field_name only
            pred_vals = pred[field_idx::num_out]  # (forecast_steps,)
            truth_vals = np.array(
                [float(w.get(field_name, 0.0)) for w in truth_windows],
                dtype=np.float32,
            )

            squared_errors = (pred_vals - truth_vals) ** 2
            all_squared_errors.extend(squared_errors.tolist())
            valid_cells += 1

        if not all_squared_errors:
            logger.warning(
                f"No valid cells for RMSE computation on field '{field_name}'"
            )
            return None

        return float(np.sqrt(np.mean(all_squared_errors)))

    def _update_tags(
        self,
        field_name: str,
        scored_models: list[tuple[str, float]],
        best_model_id: str | None,
        evaluated_at_str: str,
    ) -> None:
        """
        Persist RMSE scores and best designation as MLflow registered model tags.

        Sets rmse_for:{field} and eval_at:{field} on scored models.
        Sets best_for:{field}="true" on the best; deletes the tag from others.
        """
        scored_ids = {model_id for model_id, _ in scored_models}

        for model_id, rmse in scored_models:
            self.client.set_registered_model_tag(model_id, _rmse_key(field_name), str(rmse))
            self.client.set_registered_model_tag(
                model_id, _eval_at_key(field_name), evaluated_at_str
            )

            if model_id == best_model_id:
                self.client.set_registered_model_tag(
                    model_id, _best_key(field_name), "true"
                )
            else:
                # Remove best designation if previously set
                try:
                    self.client.delete_registered_model_tag(
                        model_id, _best_key(field_name)
                    )
                except MlflowException:
                    pass

    def _get_registered_model_tags(self, model_id: str) -> dict[str, str]:
        """Fetch the tags dict for a registered model."""
        try:
            rm = self.client.get_registered_model(model_id)
            return dict(rm.tags) if rm.tags else {}
        except MlflowException:
            return {}
