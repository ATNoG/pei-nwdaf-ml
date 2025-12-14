from src.models.lstm import LSTM
from src.models.model_interface import ModelInterface

models = [LSTM]

# Dictionary for O(1) model lookup by name
models_dict = {
    cls.__name__.lower(): cls for cls in models
}
