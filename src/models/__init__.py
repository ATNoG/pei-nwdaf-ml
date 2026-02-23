from src.models.lstm import LSTM
from src.models.ann import ANN
from src.models.model_interface import ModelInterface
from src.schemas.model import ArchitectureType

models = [ANN, LSTM]

# Model registry for training service
MODEL_REGISTRY = {
    ArchitectureType.LSTM: LSTM,
    ArchitectureType.ANN: ANN,
}
