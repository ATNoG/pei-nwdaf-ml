"""Service for evaluating and monitoring model performance per output field."""

import logging
from datetime import datetime, timezone

import numpy as np
import mlflow
from mlflow.exceptions import MlflowException
from sqlalchemy.orm import Session

from src.db.score_history import ScoreHistoryDB
from src.schemas.model import ArchitectureType, ModelConfig
from src.schemas.performance import (
    FieldEvaluationResponse,
    ModelPerformance,
    ScoreHistoryEntry,
    ScoreHistoryResponse,
)
from src.services.config_service import MLConfigService
from src.services.data_preparation import calculate_timestamps, prepare_last_sequence
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# Metrics where a higher value is better — affects best-election and degradation direction
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"r2"})


def _score_key(field_name: str) -> str:
    return f"score_for:{field_name}"


def _eval_at_key(field_name: str) -> str:
    return f"eval_at:{field_name}"


def _best_key(field_name: str) -> str:
    return f"best_for:{field_name}"


def _metric_key(field_name: str) -> str:
    return f"eval_metric:{field_name}"


def _baseline_key(field_name: str) -> str:
    return f"baseline_score:{field_name}"


class PerformanceService:
    """Service for metric-agnostic model evaluation and best-model designation."""

    MAX_EVAL_CELLS = 3

    def __init__(
        self,
        mlflow_service: MLflowService,
        ml_config_service: MLConfigService,
        data_storage_client: DataStorageClient,
        db: Session,
    ):
        self.mlflow_service = mlflow_service
        self.ml_config_service = ml_config_service
        self.data_storage_client = data_storage_client
        self.db = db
        self.client = mlflow_service.client

    async def evaluate_field(
        self, field_name: str, metric: str = "rmse"
    ) -> FieldEvaluationResponse:
        """
        Score ALL trained models that predict field_name using the given metric.

        Computes score against live cell data, persists scores as MLflow tags,
        updates the best_for:{field_name} tag, writes a history row per scored model,
        and returns a ranked response.
        """
        model_configs = [
            c
            for c in self.ml_config_service.list_all()
            if field_name in c.output_fields
        ]

        if not model_configs:
            return FieldEvaluationResponse(
                field_name=field_name,
                models=[],
                best_model_id=None,
                last_evaluated_at=None,
            )

        cells = await self.data_storage_client.get_known_cells()
        logger.info(
            f"Evaluating {len(model_configs)} models for field '{field_name}' "
            f"using metric '{metric}' and up to {self.MAX_EVAL_CELLS} reference cells "
            f"(from {len(cells)} known)"
        )

        # Probe once for a fallback timestamp in case the current-time window is empty
        fallback_start_ts = await self.data_storage_client.probe_data_timestamp(cells, model_configs[0].window_duration_seconds if model_configs else 60)
        if fallback_start_ts is not None:
            logger.info(
                "Evaluation fallback anchor: window_start_time=%d (data found in storage)",
                fallback_start_ts,)
        else:
            logger.warning("Evaluation: no data found in storage for any time range — scores will be None")

        evaluated_at = datetime.now(tz=timezone.utc)
        evaluated_at_str = evaluated_at.isoformat()

        performances: list[ModelPerformance] = []
        scored: list[tuple[str, float]] = []  # (model_id, score)

        for model_config in model_configs:
            model_detail = self.mlflow_service.get_model(model_config.model_id)

            base = ModelPerformance(
                model_id=model_config.model_id,
                model_name=model_config.name,
                architecture=ArchitectureType(model_config.architecture),
                latest_version=model_detail.latest_version,
                training_loss=model_detail.training_loss,
                last_trained_at=model_detail.last_trained_at,
                metric=metric,
            )

            if model_detail.latest_version is None:
                # Not trained — skip scoring
                performances.append(base)
                continue

            config = model_detail.config

            try:
                model = self._load_model(model_config.model_id, model_detail.latest_version, config)
                score = await self._compute_score_for_model(model, config, field_name, cells, metric, fallback_start_ts)
            except Exception as e:
                logger.warning(
                    f"Failed to evaluate model {model_config.model_id} for field '{field_name}': {e}"
                )
                score = None

            if score is not None:
                scored.append((model_config.model_id, score))
                self._write_history_row(model_config.model_id, field_name, score, metric, "evaluate")

            performances.append(
                base.model_copy(update={"score": score, "evaluated_at": evaluated_at})
            )

        # Determine best (direction-aware)
        best_model_id: str | None = None
        if scored:
            if metric in _HIGHER_IS_BETTER:
                best_model_id = max(scored, key=lambda t: t[1])[0]
            else:
                best_model_id = min(scored, key=lambda t: t[1])[0]

        # Persist tags + baseline on winner
        self._update_tags(
            field_name=field_name,
            scored_models=scored,
            best_model_id=best_model_id,
            evaluated_at_str=evaluated_at_str,
            metric=metric,
        )

        for p in performances:
            p.is_best = p.model_id == best_model_id

        return FieldEvaluationResponse(
            field_name=field_name,
            models=self._sort_by_score(performances, metric),
            best_model_id=best_model_id,
            last_evaluated_at=evaluated_at,
        )

    def get_cached_evaluation(self, field_name: str) -> FieldEvaluationResponse:
        """
        Return the last evaluation result from stored MLflow tags.

        No computation is triggered. Models that have never been evaluated
        will have score=None and evaluated_at=None.
        """
        model_configs = [
            c
            for c in self.ml_config_service.list_all()
            if field_name in c.output_fields
        ]

        if not model_configs:
            return FieldEvaluationResponse(
                field_name=field_name,
                models=[],
                best_model_id=None,
                last_evaluated_at=None,
            )

        performances: list[ModelPerformance] = []
        best_model_id: str | None = None
        last_evaluated_at: datetime | None = None
        field_metric: str = "rmse"  # fallback for sort direction if no tag found

        for model_config in model_configs:
            model_detail = self.mlflow_service.get_model(model_config.model_id)
            tags = self._get_registered_model_tags(model_config.model_id)

            score_str = tags.get(_score_key(field_name))
            eval_at_str = tags.get(_eval_at_key(field_name))
            is_best_str = tags.get(_best_key(field_name))
            metric_str = tags.get(_metric_key(field_name))

            score = float(score_str) if score_str else None
            evaluated_at = datetime.fromisoformat(eval_at_str) if eval_at_str else None
            is_best = is_best_str == "true"
            metric = metric_str or "rmse"

            if is_best:
                best_model_id = model_config.model_id
                field_metric = metric

            if evaluated_at and (
                last_evaluated_at is None or evaluated_at > last_evaluated_at
            ):
                last_evaluated_at = evaluated_at

            performances.append(
                ModelPerformance(
                    model_id=model_config.model_id,
                    model_name=model_config.name,
                    architecture=ArchitectureType(model_config.architecture),
                    latest_version=model_detail.latest_version,
                    training_loss=model_detail.training_loss,
                    score=score,
                    metric=metric,
                    is_best=is_best,
                    last_trained_at=model_detail.last_trained_at,
                    evaluated_at=evaluated_at,
                )
            )

        return FieldEvaluationResponse(
            field_name=field_name,
            models=self._sort_by_score(performances, field_metric),
            best_model_id=best_model_id,
            last_evaluated_at=last_evaluated_at,
        )

    def get_best_model(self, field_name: str) -> ModelPerformance | None:
        """
        Return the model currently tagged as best for field_name.

        Returns None if no evaluation has been run yet.
        """
        model_configs = [
            c
            for c in self.ml_config_service.list_all()
            if field_name in c.output_fields
        ]

        for model_config in model_configs:
            tags = self._get_registered_model_tags(model_config.model_id)
            if tags.get(_best_key(field_name)) == "true":
                model_detail = self.mlflow_service.get_model(model_config.model_id)
                score_str = tags.get(_score_key(field_name))
                eval_at_str = tags.get(_eval_at_key(field_name))
                metric_str = tags.get(_metric_key(field_name))

                return ModelPerformance(
                    model_id=model_config.model_id,
                    model_name=model_config.name,
                    architecture=ArchitectureType(model_config.architecture),
                    latest_version=model_detail.latest_version,
                    training_loss=model_detail.training_loss,
                    score=float(score_str) if score_str else None,
                    metric=metric_str or "rmse",
                    is_best=True,
                    last_trained_at=model_detail.last_trained_at,
                    evaluated_at=datetime.fromisoformat(eval_at_str) if eval_at_str else None,
                )

        return None

    async def monitor_best_model(
        self, field_name: str, trigger: str = "monitor"
    ) -> ModelPerformance:
        """
        Re-evaluate only the current best model for field_name.

        Reads eval_metric:{field} tag so monitoring always uses the same metric
        as the original election. Updates score_for and eval_at tags but does NOT
        change the best_for tag — designation only changes via a full /evaluate call.

        Args:
            trigger: Label written to history row ("monitor" for manual calls,
                     "auto_monitor" for background loop calls).

        Raises:
            ValueError: If no best model has been designated yet.
        """
        best = self.get_best_model(field_name)
        if best is None:
            raise ValueError(
                f"No best model designated for field '{field_name}'. "
                f"Run POST /performance/{field_name}/evaluate first."
            )

        # Always use the metric stored at election time
        tags = self._get_registered_model_tags(best.model_id)
        metric = tags.get(_metric_key(field_name), "rmse")

        model_detail = self.mlflow_service.get_model(best.model_id)

        if model_detail.latest_version is None:
            raise ValueError(
                f"Best model '{best.model_id}' has no trained version. "
                f"Train the model before monitoring."
            )

        config = model_detail.config
        cells = await self.data_storage_client.get_known_cells()

        fallback_start_ts = await self.data_storage_client.probe_data_timestamp(cells, config.window_duration_seconds)

        model = self._load_model(best.model_id, model_detail.latest_version, config)
        score = await self._compute_score_for_model(model, config, field_name, cells, metric, fallback_start_ts)

        evaluated_at = datetime.now(tz=timezone.utc)
        evaluated_at_str = evaluated_at.isoformat()

        # Update score + timestamp tags only — best_for and baseline_score stay unchanged
        if score is not None:
            self.client.set_registered_model_tag(
                best.model_id, _score_key(field_name), str(score)
            )
            self._write_history_row(best.model_id, field_name, score, metric, trigger)

        self.client.set_registered_model_tag(
            best.model_id, _eval_at_key(field_name), evaluated_at_str
        )

        logger.info(
            f"Monitor: updated {metric} for best model {best.model_id} "
            f"on field '{field_name}': {score}"
        )

        return ModelPerformance(
            model_id=best.model_id,
            model_name=best.model_name,
            architecture=best.architecture,
            latest_version=model_detail.latest_version,
            training_loss=model_detail.training_loss,
            score=score,
            metric=metric,
            is_best=True,
            last_trained_at=model_detail.last_trained_at,
            evaluated_at=evaluated_at,
        )

    def get_monitored_fields(self) -> list[str]:
        """Return all output fields that have a best_for:{field} tag set."""
        fields: set[str] = set()
        for config in self.ml_config_service.list_all():
            tags = self._get_registered_model_tags(config.model_id)
            for key in tags:
                if key.startswith("best_for:"):
                    fields.add(key.removeprefix("best_for:"))
        return list(fields)

    def get_baseline_score(self, field_name: str) -> float | None:
        """Return the baseline score recorded at election time for field_name."""
        best = self.get_best_model(field_name)
        if best is None:
            return None
        tags = self._get_registered_model_tags(best.model_id)
        raw = tags.get(_baseline_key(field_name))
        return float(raw) if raw else None

    def get_score_history(
        self, field_name: str, model_id: str | None = None
    ) -> ScoreHistoryResponse:
        """Return score measurement history for field_name, oldest first."""
        q = self.db.query(ScoreHistoryDB).filter(ScoreHistoryDB.field_name == field_name)
        if model_id:
            q = q.filter(ScoreHistoryDB.model_id == model_id)
        rows = q.order_by(ScoreHistoryDB.measured_at).all()
        return ScoreHistoryResponse(
            field_name=field_name,
            entries=[
                ScoreHistoryEntry(
                    model_id=r.model_id,
                    field_name=r.field_name,
                    score=r.score,
                    metric=r.metric,
                    measured_at=r.measured_at,
                    trigger=r.trigger,
                )
                for r in rows
            ],
        )

    def _score_is_worse(
        self, current: float, baseline: float, metric: str, factor: float
    ) -> bool:
        """Return True if current score has degraded past the factor threshold."""
        if metric in _HIGHER_IS_BETTER:
            return current < baseline / factor
        return current > baseline * factor

    def _sort_by_score(
        self, performances: list[ModelPerformance], metric: str
    ) -> list[ModelPerformance]:
        """Sort performances: best score first, unscored models last."""
        if metric in _HIGHER_IS_BETTER:
            return sorted(
                performances,
                key=lambda p: (p.score is None, -(p.score if p.score is not None else 0.0)),
            )
        return sorted(
            performances,
            key=lambda p: (p.score is None, p.score if p.score is not None else 0.0),
        )

    def _write_history_row(
        self,
        model_id: str,
        field_name: str,
        score: float,
        metric: str,
        trigger: str,
    ) -> None:
        """Persist a score measurement to the score_history table."""
        self.db.add(
            ScoreHistoryDB(
                model_id=model_id,
                field_name=field_name,
                score=score,
                metric=metric,
                measured_at=datetime.now(tz=timezone.utc),
                trigger=trigger,
            )
        )
        self.db.commit()

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

    async def _compute_score_for_model(
        self,
        model,
        config: ModelConfig,
        field_name: str,
        cells: list[int],
        metric: str,
        fallback_start_ts: int | None = None,
    ) -> float | None:
        """
        Collect raw predictions and ground-truth values across reference cells,
        then compute the requested metric in one shot.

        Raw values (not pre-squared) are accumulated so any metric can be
        applied at the end via _compute_score().

        If the current-time window has no data, the method retries each cell using fallback_start_ts
        as the window anchor.  fallback_start_ts is discovered once per evaluation
        call by probing the data storage with a wide time range.
        """
        min_windows = config.lookback_steps + config.forecast_steps
        min_secs = min_windows * config.window_duration_seconds
        fetch_secs = max(min_secs, 86400)
        start_ts, end_ts = calculate_timestamps(fetch_secs)

        field_idx = config.output_fields.index(field_name)
        num_out = len(config.output_fields)
        all_pred_vals: list[float] = []
        all_truth_vals: list[float] = []
        valid_cells = 0

        for cell_index in cells:
            if valid_cells >= self.MAX_EVAL_CELLS:
                break

            # Primary fetch: window anchored to now
            data: list[dict] | None = None
            try:
                data = await self.data_storage_client.fetch_cell_data(
                    cell_index=cell_index,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    window_duration_seconds=config.window_duration_seconds,
                )
            except Exception as e:
                logger.warning("Failed to fetch data for cell %d: %s", cell_index, e)

            # Fallback: retry anchored to probed data timestamp
            if (not data or len(data) < min_windows) and fallback_start_ts is not None:
                alt_start = fallback_start_ts
                alt_end = fallback_start_ts + fetch_secs
                logger.info(
                    "Cell %d: no current data, retrying with probed window [%d, %d]",
                    cell_index,
                    alt_start,
                    alt_end,
                )
                try:
                    data = await self.data_storage_client.fetch_cell_data(
                        cell_index=cell_index,
                        start_timestamp=alt_start,
                        end_timestamp=alt_end,
                        window_duration_seconds=config.window_duration_seconds,
                    )
                except Exception as e:
                    logger.warning("Fallback fetch failed for cell %d: %s", cell_index, e)
                    data = None

            if not data or len(data) < min_windows:
                logger.warning(
                    f"Cell {cell_index}: insufficient data "
                    f"({len(data) if data else 0} windows, need {min_windows})"
                )
                continue

            data.sort(key=lambda x: x.get("window_start_time", 0))

            eval_data = data[-min_windows:]
            x_windows = eval_data[: config.lookback_steps]
            truth_windows = eval_data[config.lookback_steps :]

            X = prepare_last_sequence(x_windows, config.input_fields, config.lookback_steps)

            try:
                pred = model.predict(X)[0]
            except Exception as e:
                logger.warning(f"Prediction failed for cell {cell_index}: {e}")
                continue

            pred_vals = pred[field_idx::num_out].tolist()
            truth_vals = [float(w.get(field_name, 0.0)) for w in truth_windows]

            all_pred_vals.extend(pred_vals)
            all_truth_vals.extend(truth_vals)
            valid_cells += 1

        if not all_pred_vals:
            logger.warning(
                f"No valid cells for score computation on field '{field_name}' (metric={metric})"
            )
            return None

        return self._compute_score(all_pred_vals, all_truth_vals, metric)

    def _compute_score(
        self,
        pred_vals: list[float],
        truth_vals: list[float],
        metric: str,
    ) -> float:
        """Dispatch to the appropriate scoring function."""
        p = np.array(pred_vals, dtype=np.float64)
        t = np.array(truth_vals, dtype=np.float64)
        errors = p - t

        if metric == "rmse":
            return float(np.sqrt(np.mean(errors ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(errors)))
        if metric == "mape":
            return float(np.mean(np.abs(errors / t)) * 100)
        if metric == "r2":
            ss_res = float(np.sum(errors ** 2))
            ss_tot = float(np.sum((t - np.mean(t)) ** 2))
            return (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        raise ValueError(f"Unknown metric: {metric!r}")

    def _update_tags(
        self,
        field_name: str,
        scored_models: list[tuple[str, float]],
        best_model_id: str | None,
        evaluated_at_str: str,
        metric: str,
    ) -> None:
        """
        Persist scores and best designation as MLflow registered model tags.

        For every scored model: sets score_for:{field} and eval_at:{field}.
        For the winner only: sets best_for:{field}, eval_metric:{field},
        and baseline_score:{field} (baseline is never overwritten by monitoring).
        For non-winners: removes best_for:{field} if previously set.
        """
        for model_id, score in scored_models:
            self.client.set_registered_model_tag(model_id, _score_key(field_name), str(score))
            self.client.set_registered_model_tag(
                model_id, _eval_at_key(field_name), evaluated_at_str
            )

            if model_id == best_model_id:
                self.client.set_registered_model_tag(model_id, _best_key(field_name), "true")
                self.client.set_registered_model_tag(model_id, _metric_key(field_name), metric)
                self.client.set_registered_model_tag(
                    model_id, _baseline_key(field_name), str(score)
                )
            else:
                try:
                    self.client.delete_registered_model_tag(model_id, _best_key(field_name))
                except MlflowException:
                    pass

    def _get_registered_model_tags(self, model_id: str) -> dict[str, str]:
        """Fetch the tags dict for a registered model."""
        try:
            rm = self.client.get_registered_model(model_id)
            return dict(rm.tags) if rm.tags else {}
        except MlflowException:
            return {}
