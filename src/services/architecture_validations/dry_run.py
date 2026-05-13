import sys
import types

import numpy as np
import torch.nn as nn

from src.models.model_interface import ModelInterface


def run(architecture_id: str, content: bytes, namespace_factory) -> None:
    module_name = f"_dryrun_{architecture_id}"
    namespace = namespace_factory(module_name)
    exec(content, namespace)

    mod = types.ModuleType(module_name)
    mod.__dict__.update(namespace)
    sys.modules[module_name] = mod

    try:
        cls = next(
            (
                v
                for v in namespace.values()
                if isinstance(v, type)
                and issubclass(v, ModelInterface)
                and v is not ModelInterface
            ),
            None,
        )
        if cls is None:
            raise ValueError("No ModelInterface subclass found")

        model = cls(
            input_fields=["x"],
            output_fields=["y"],
            window_duration_seconds=60,
            lookback_steps=2,
            forecast_steps=1,
            hidden_size=4,
        )
        X = np.zeros((2, 2, 1), dtype=np.float32)
        y = np.zeros((2, 1), dtype=np.float32)
        model.train(X, y, max_epochs=1)
        if not hasattr(model, 'model') or not isinstance(model.model, nn.Module):
            raise ValueError("After train(), self.model must be set to a torch.nn.Module instance")
        model.predict(X)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Dry run failed: {e}")
    finally:
        sys.modules.pop(module_name, None)
