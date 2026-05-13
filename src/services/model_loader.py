import mlflow

from src.services.architecture_service import ArchitectureService


def load_trained_model(architecture_id: str, model_uri: str, config) -> object:
    """Load a trained PyTorch model from MLflow using the registered architecture."""
    arch_svc = ArchitectureService()
    with arch_svc.load_architecture(architecture_id) as (cls, _):
        try:
            loaded = mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_uri}: {e}")

        model = cls(
            input_fields=config.input_fields,
            output_fields=config.output_fields,
            window_duration_seconds=config.window_duration_seconds,
            lookback_steps=config.lookback_steps,
            forecast_steps=config.forecast_steps,
            hidden_size=config.hidden_size,
        )
        model.model = loaded

    return model
