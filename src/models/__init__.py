from src.models.randomforest import RandomForest
from src.models.xgboost import XGBoost
from src.models.lstm import LSTM

models = [RandomForest, XGBoost, LSTM]

# Dictionary for O(1) model lookup by name
models_dict = {
    cls.__name__.lower(): cls for cls in models
}
