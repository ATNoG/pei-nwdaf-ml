# src/models/xgboost_model.py
from src.models.modelI import ModelI
import pickle
from typing import Type
import xgboost as xgb
import numpy as np

class XGBoost(ModelI):
    HEADER = b"XGBOOST"
    FRAMEWORK = "sklearn"

    def __init__(self):
        super().__init__()
        self.model = None

    def predict(self, data):
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(data)

    def train(self, max_epochs: int, X, y):
        model = xgb.XGBRegressor(n_estimators=max_epochs)
        model.fit(X, y)
        self.model = model
        # calculate loss as mean squared error
        y_pred = self.model.predict(X)
        loss = np.mean((y - y_pred) ** 2)
        return float(loss)

    def serialize(self) -> bytes:
        payload = {"model": self.model}
        return self.HEADER + pickle.dumps(payload)

    @classmethod
    def deserialize(cls: Type['XGBoost'], b: bytes) -> 'XGBoost':
        if not b.startswith(cls.HEADER):
            raise TypeError("Invalid model header for XGBoost.")
        data = pickle.loads(b[len(cls.HEADER):])
        obj = cls()
        obj.model = data["model"]
        return obj
