"""Service for evaluating and monitoring model performance per output field."""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import mlflow
from mlflow.exceptions import MlflowException
from sqlalchemy.orm import Session

from src.core.config import settings
from src.db.score_history import ScoreHistoryDB
from src.schemas.model import ModelConfig
from src.schemas.performance import (
    FeatureImportanceResponse,
    FieldEvaluationResponse,
    ModelPerformance,
    ScoreHistoryEntry,
    ScoreHistoryResponse,
)
from src.services.config_service import MLConfigService
from src.services.data_preparation import calculate_timestamps, prepare_last_sequence
from src.services.data_storage_client import DataStorageClient
from src.services.mlflow_service import MLflowService
from src.services.model_loader import load_trained_model
from src.services.safe_predict import sync_safe_predict

logger = logging.getLogger(__name__)

# Metrics where a higher value is better - affects best-election and degradation direction
_HIGHER_IS_BETTER: frozenset[str] = frozenset({"r2"})


def _score_key(field_name: str) -> str:
    return f"score_for:{field_name}"


def _eval_at_key(field_name: str) -> str:
    return f"eval_at:{field_name}"


def _best_key(event_type: str, field_name: str) -> str:
    return f"best_for:{event_type}:{field_name}"


def _metric_key(field_name: str) -> str:
    return f"eval_metric:{field_name}"


def _baseline_key(field_name: str) -> str:
    return f"baseline_score:{field_name}"


def _importance_key(field_name: str) -> str:
    return f"importance:{field_name}"


def _importance_at_key(field_name: str) -> str:
    return f"importance_at:{field_name}"


def _make_alibi_scorer(metric: str, score_fn):
    """
    Wrap _compute_score into an alibi-compatible scorer (higher = better).
    Error metrics (rmse, mae, mape) are negated so alibi's convention holds.
    Sets __name__ = metric so alibi keys the result dict by metric name.
    """
    def scorer(y_true, y_pred):
        if len(y_pred) == 0 or len(y_true) == 0:
            return 0.0
        val = score_fn(list(y_pred), list(y_true), metric)
        return val if metric == "r2" else -val

    return scorer


class _SequenceIndexBridge:
    """
    Bridges 2D permutation methods with 3D sequence models.

    Encodes (n_cells, lookback, n_fields) as (n_cells, n_fields) float indices
    pointing into a pool of real observed sequences.

    Pool layout:
        indices 0..n_cells-1  -> real cell sequences
    """

    def __init__(self, X_all: np.ndarray):
        self.pool = X_all.astype(np.float32)  # (n_cells, lookback, n_fields)
        self.n_cells = len(X_all)

    def encode(self, n_fields: int) -> np.ndarray:
        """Identity encoding: cell i uses its own sequences for every field."""
        return np.tile(np.arange(self.n_cells, dtype=np.float32)[:, np.newaxis], (1, n_fields))  # (n_cells, n_fields)

    def reconstruct(self, X_2d: np.ndarray) -> np.ndarray:
        """Look up real sequences from pool using float indices (rounded + clipped)."""
        n_samples, n_fields = X_2d.shape
        lookback = self.pool.shape[1]
        X_3d = np.zeros((n_samples, lookback, n_fields), dtype=np.float32)
        for s in range(n_samples):
            for i in range(n_fields):
                idx = int(round(np.clip(X_2d[s, i], 0, len(self.pool) - 1)))
                X_3d[s, :, i] = self.pool[idx, :, i]
        return X_3d


class PerformanceService:
    """Service for metric-agnostic model evaluation and best-model designation."""

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

        event_type = model_configs[0].event_type if model_configs else ""

        logger.info(
            f"Evaluating {len(model_configs)} models for field '{field_name}' "
            f"using metric '{metric}' and event_type '{event_type}'"
        )

        fallback_start_ts = await self.data_storage_client.probe_data_timestamp(
            model_configs[0].window_duration_seconds if model_configs else 60,
            event_type=event_type or None,
        )
        if fallback_start_ts is not None:
            logger.info("Evaluation fallback anchor: window_start_time=%d", fallback_start_ts)
        else:
            logger.warning("Evaluation: no data found in storage - scores will be None")

        evaluated_at = datetime.now(tz=timezone.utc)
        evaluated_at_str = evaluated_at.isoformat()

        performances: list[ModelPerformance] = []
        scored: list[tuple[str, float]] = []  # (model_id, score)
        loaded_models: dict[str, object] = {}  # to avoid loading the same model twice for importance calculation
        detail_map: dict[str, object] = {}

        for model_config in model_configs:
            model_detail = self.mlflow_service.get_model(model_config.model_id)

            base = ModelPerformance(
                model_id=model_config.model_id,
                model_name=model_config.name,
                architecture=model_config.architecture,
                latest_version=model_detail.latest_version,
                training_loss=model_detail.training_loss,
                last_trained_at=model_detail.last_trained_at,
                metric=metric,
            )

            if model_detail.latest_version is None:
                # Not trained - skip scoring
                performances.append(base)
                continue

            config = model_detail.config

            detail_map[model_config.model_id] = model_detail
            try:
                _model_uri = f"models:/{model_config.model_id}/{model_detail.latest_version}"
                loaded_models[model_config.model_id] = _model_uri
                score = await self._compute_score_for_model(_model_uri, config, field_name, event_type, metric, fallback_start_ts)
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
            event_type=event_type,
            field_name=field_name,
            scored_models=scored,
            best_model_id=best_model_id,
            evaluated_at_str=evaluated_at_str,
            metric=metric,
        )

        if best_model_id and best_model_id in loaded_models:
            try:
                importances = await self._compute_permutation_importance(
                    model_uri=loaded_models[best_model_id],
                    config=detail_map[best_model_id].config,
                    field_name=field_name,
                    event_type=event_type,
                    metric=metric,
                    fallback_start_ts=fallback_start_ts,
                )
                if importances is not None:
                    self.client.set_registered_model_tag(best_model_id, _importance_key(field_name), json.dumps(importances))
                    self.client.set_registered_model_tag(best_model_id, _importance_at_key(field_name), evaluated_at_str)
            except Exception as e:
                logger.warning("Permutation importance failed for '%s': %s", field_name, e)

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
            is_best_str = tags.get(_best_key(model_config.event_type, field_name))
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
                    architecture=model_config.architecture,
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
            if tags.get(_best_key(model_config.event_type, field_name)) == "true":
                model_detail = self.mlflow_service.get_model(model_config.model_id)
                score_str = tags.get(_score_key(field_name))
                eval_at_str = tags.get(_eval_at_key(field_name))
                metric_str = tags.get(_metric_key(field_name))
                baseline_str = tags.get(_baseline_key(field_name))

                return ModelPerformance(
                    model_id=model_config.model_id,
                    model_name=model_config.name,
                    architecture=model_config.architecture,
                    latest_version=model_detail.latest_version,
                    training_loss=model_detail.training_loss,
                    score=float(score_str) if score_str else None,
                    metric=metric_str or "rmse",
                    is_best=True,
                    baseline_score=float(baseline_str) if baseline_str else None,
                    last_trained_at=model_detail.last_trained_at,
                    evaluated_at=datetime.fromisoformat(eval_at_str) if eval_at_str else None,
                )

        return None

    def set_best_model(self, field_name: str, model_id: str) -> ModelPerformance:
        """
        Override best model designation for field_name without scoring.

        Removes best_for:{field} from the current best (if any) and sets it
        on the given model_id. Score/metric/baseline tags are left untouched.

        Raises ValueError if model_id does not exist or does not predict field_name.
        """
        # Validate model exists and predicts the field
        config = self.ml_config_service.get_config(model_id)
        if config is None:
            raise ValueError(f"Model ID '{model_id}' does not exist.")
        if field_name not in config.output_fields:
            raise ValueError(f"Model ID '{model_id}' does not predict field '{field_name}'.")

        event_type = config.event_type

        # Remove best tag from current best
        current_best = self.get_best_model(field_name)
        if current_best and current_best.model_id != model_id:
            try:
                self.client.delete_registered_model_tag(current_best.model_id, _best_key(event_type, field_name))
            except MlflowException:
                pass

        self.client.set_registered_model_tag(model_id, _best_key(event_type, field_name), "true")

        return self.get_best_model(field_name)

    async def monitor_best_model(
        self, field_name: str, trigger: str = "monitor"
    ) -> ModelPerformance:
        """
        Re-evaluate only the current best model for field_name.

        Reads eval_metric:{field} tag so monitoring always uses the same metric
        as the original election. Updates score_for and eval_at tags but does NOT
        change the best_for tag - designation only changes via a full /evaluate call.

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
        model_event_type = getattr(model_detail, "event_type", "")

        fallback_start_ts = await self.data_storage_client.probe_data_timestamp(
            config.window_duration_seconds, event_type=model_event_type or None
        )

        _model_uri = f"models:/{best.model_id}/{model_detail.latest_version}"
        score = await self._compute_score_for_model(_model_uri, config, field_name, model_event_type, metric, fallback_start_ts)

        evaluated_at = datetime.now(tz=timezone.utc)
        evaluated_at_str = evaluated_at.isoformat()

        # Update score + timestamp tags only - best_for and baseline_score stay unchanged
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
        """Return all output fields that have a best_for:{event}:{field} tag set."""
        fields: set[str] = set()
        for config in self.ml_config_service.list_all():
            tags = self._get_registered_model_tags(config.model_id)
            for key in tags:
                if key.startswith("best_for:"):
                    suffix = key.removeprefix("best_for:")  # e.g. "PERF_DATA:thrputUl_mbps_mean"
                    fields.add(suffix)
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
        model_uri = f"models:/{model_id}/{version}"
        logger.info(f"Loading model for evaluation: {model_uri}")
        return load_trained_model(config.architecture, model_uri, config)

    async def _compute_score_for_model(
        self,
        model_uri: str,
        config: ModelConfig,
        field_name: str,
        event_type: str,
        metric: str,
        fallback_start_ts: int | None = None,
    ) -> float | None:
        """Compute score by evaluating each slice (snssai+dnn) independently."""
        min_windows = config.lookback_steps + config.forecast_steps
        min_secs = min_windows * config.window_duration_seconds
        fetch_secs = max(min_secs, 86400)
        start_ts, end_ts = calculate_timestamps(fetch_secs)

        field_idx = config.output_fields.index(field_name)
        num_out = len(config.output_fields)
        all_pred_vals: list[float] = []
        all_truth_vals: list[float] = []

        # Fetch all data for this event_type
        data = await self.data_storage_client.fetch_data(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=config.window_duration_seconds,
            event=event_type or None,
        )

        # Fallback to probed timestamp if no current data
        if (not data or len(data) < min_windows) and fallback_start_ts is not None:
            alt_end = fallback_start_ts + fetch_secs
            data = await self.data_storage_client.fetch_data(
                start_timestamp=fallback_start_ts,
                end_timestamp=alt_end,
                window_duration_seconds=config.window_duration_seconds,
                event=event_type or None,
            )

        if not data:
            logger.warning("No data for score computation on field '%s' (metric=%s)", field_name, metric)
            return None

        # Group by slice context and score each independently
        slice_groups: dict[tuple, list[dict]] = {}
        for w in data:
            key = (w.get("snssai_sst", ""), w.get("snssai_sd", ""), w.get("dnn", ""))
            slice_groups.setdefault(key, []).append(w)

        for slice_key, slice_data in slice_groups.items():
            slice_data.sort(key=lambda x: x.get("window_start_time", 0))
            if len(slice_data) < min_windows:
                continue

            eval_data = slice_data[-min_windows:]
            x_windows = eval_data[: config.lookback_steps]
            truth_windows = eval_data[config.lookback_steps :]

            X = prepare_last_sequence(x_windows, config.input_fields, config.lookback_steps)

            try:
                pred = sync_safe_predict(config.architecture, model_uri, config, X)[0]
            except Exception as e:
                logger.warning("Prediction failed for slice %s: %s", slice_key, e)
                continue

            all_pred_vals.extend(pred[field_idx::num_out].tolist())
            all_truth_vals.extend([float(w.get(field_name) or 0.0) for w in truth_windows])

        if not all_pred_vals:
            logger.warning("No valid slices for score computation on field '%s' (metric=%s)", field_name, metric)
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

    async def _compute_permutation_importance(
        self,
        model_uri: str,
        config: ModelConfig,
        field_name: str,
        event_type: str,
        metric: str,
        fallback_start_ts: int | None,
        n_repeats: int = 5,
    ) -> dict[str, float] | None:
        """Compute permutation importance grouped by slice context."""
        from alibi.explainers import PermutationImportance

        min_windows = config.lookback_steps + config.forecast_steps
        fetch_secs = max(min_windows * config.window_duration_seconds, 86400)
        start_ts, end_ts = calculate_timestamps(fetch_secs)
        field_idx = config.output_fields.index(field_name)
        num_out = len(config.output_fields)

        X_list: list[np.ndarray] = []
        y_list: list[list[float]] = []

        data = await self.data_storage_client.fetch_data(
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            window_duration_seconds=config.window_duration_seconds,
            event=event_type or None,
        )
        if (not data or len(data) < min_windows) and fallback_start_ts is not None:
            data = await self.data_storage_client.fetch_data(
                start_timestamp=fallback_start_ts,
                end_timestamp=fallback_start_ts + fetch_secs,
                window_duration_seconds=config.window_duration_seconds,
                event=event_type or None,
            )

        if data:
            slice_groups: dict[tuple, list[dict]] = {}
            for w in data:
                key = (w.get("snssai_sst", ""), w.get("snssai_sd", ""), w.get("dnn", ""))
                slice_groups.setdefault(key, []).append(w)

            for slice_data in slice_groups.values():
                if len(slice_data) < min_windows:
                    continue
                slice_data.sort(key=lambda x: x.get("window_start_time", 0))
                eval_data = slice_data[-min_windows:]
                X = prepare_last_sequence(eval_data[: config.lookback_steps], config.input_fields, config.lookback_steps)
                X_list.append(X[0])
                y_list.append([float(w.get(field_name) or 0.0) for w in eval_data[config.lookback_steps:]])

        if not X_list:
            logger.warning("Permutation importance: no usable slices for '%s'", field_name)
            return None

        bridge = _SequenceIndexBridge(np.stack(X_list, axis=0))
        X_2d = bridge.encode(len(config.input_fields))
        y_1d = np.array([float(np.mean(y)) for y in y_list])

        def predict_fn(X: np.ndarray) -> np.ndarray:
            if len(X) == 0:
                return np.zeros((0,), dtype=np.float32)
            X_3d = bridge.reconstruct(X)
            preds = sync_safe_predict(config.architecture, model_uri, config, X_3d)
            return preds[:, field_idx::num_out].mean(axis=1)

        scorer = _make_alibi_scorer(metric, self._compute_score)
        explainer = PermutationImportance(predictor=predict_fn, score_fns=scorer, feature_names=config.input_fields)
        explanation = explainer.explain(X_2d, y_1d, n_repeats=n_repeats, kind="difference")

        f_names = explanation.data["feature_names"]
        f_importance = explanation.data["feature_importance"][0]
        importances = {
            f_names[i]: {
                "mean": round(float(f_importance[i]["mean"]), 6),
                "std":  round(float(f_importance[i]["std"]),  6),
            }
            for i in range(len(f_names))
        }

        ranked = sorted(importances.items(), key=lambda x: x[1]["mean"], reverse=True)
        logger.info(
            "Permutation importance for field '%s' (metric=%s, n_cells=%d, n_repeats=%d):",
            field_name, metric, len(X_list), n_repeats,
        )
        for name, vals in ranked:
            logger.info("  %-30s mean=%+.6f  std=%.6f", name, vals["mean"], vals["std"])

        return importances

    def get_feature_importance(
        self, field_name: str, model_id: str | None = None
    ) -> FeatureImportanceResponse | None:
        """Read cached permutation importance from MLflow tags. model_id=None -> best model."""
        if model_id is None:
            best = self.get_best_model(field_name)
            if best is None:
                return None
            model_id = best.model_id
            metric = best.metric or "rmse"
        else:
            tags = self._get_registered_model_tags(model_id)
            metric = tags.get(_metric_key(field_name), "rmse")
        tags = self._get_registered_model_tags(model_id)
        raw = tags.get(_importance_key(field_name))
        at_str = tags.get(_importance_at_key(field_name))
        if not raw:
            return None
        return FeatureImportanceResponse(
            field_name=field_name,
            model_id=model_id,
            importances=json.loads(raw),
            metric=metric,
            computed_at=datetime.fromisoformat(at_str) if at_str else None,
        )

    async def trigger_feature_importance(
        self, field_name: str, model_id: str
    ) -> FeatureImportanceResponse:
        """Load model, compute permutation importance on demand, store as MLflow tags."""
        model_detail = self.mlflow_service.get_model(model_id)
        if model_detail.latest_version is None:
            raise ValueError(f"Model '{model_id}' has no trained version")
        config = model_detail.config
        _model_uri = f"models:/{model_id}/{model_detail.latest_version}"
        event_type = getattr(model_detail, "event_type", "")
        fallback_start_ts = await self.data_storage_client.probe_data_timestamp(
            config.window_duration_seconds, event_type=event_type or None
        )
        tags = self._get_registered_model_tags(model_id)

        metric = tags.get(_metric_key(field_name), "rmse")
        importances = await self._compute_permutation_importance(
            model_uri=_model_uri,
            config=config,
            field_name=field_name,
            event_type=event_type,
            metric=metric,
            fallback_start_ts=fallback_start_ts,
        )
        if importances is None:
            raise ValueError(
                f"No usable data found for model '{model_id}' on field '{field_name}'"
            )
        computed_at = datetime.now(timezone.utc)
        self.client.set_registered_model_tag(model_id, _importance_key(field_name), json.dumps(importances))
        self.client.set_registered_model_tag(model_id, _importance_at_key(field_name), computed_at.isoformat())
        return FeatureImportanceResponse(
            field_name=field_name,
            model_id=model_id,
            importances=importances,
            metric=metric,
            computed_at=computed_at,
        )

    def _update_tags(
        self,
        event_type: str,
        field_name: str,
        scored_models: list[tuple[str, float]],
        best_model_id: str | None,
        evaluated_at_str: str,
        metric: str,
    ) -> None:
        """
        Persist scores and best designation as MLflow registered model tags.

        For every scored model: sets score_for:{field} and eval_at:{field}.
        For the winner only: sets best_for:{event}:{field}, eval_metric:{field},
        and baseline_score:{field} (baseline is never overwritten by monitoring).
        For non-winners: removes best_for:{event}:{field} if previously set.
        """
        for model_id, score in scored_models:
            self.client.set_registered_model_tag(model_id, _score_key(field_name), str(score))
            self.client.set_registered_model_tag(
                model_id, _eval_at_key(field_name), evaluated_at_str
            )

            if model_id == best_model_id:
                self.client.set_registered_model_tag(model_id, _best_key(event_type, field_name), "true")
                self.client.set_registered_model_tag(model_id, _metric_key(field_name), metric)
                self.client.set_registered_model_tag(
                    model_id, _baseline_key(field_name), str(score)
                )
            else:
                try:
                    self.client.delete_registered_model_tag(model_id, _best_key(event_type, field_name))
                except MlflowException:
                    pass

    def _get_registered_model_tags(self, model_id: str) -> dict[str, str]:
        """Fetch the tags dict for a registered model."""
        try:
            rm = self.client.get_registered_model(model_id)
            return dict(rm.tags) if rm.tags else {}
        except MlflowException:
            return {}
