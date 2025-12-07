# src/models/randomforest_model.py
from src.models.modelI import ModelI
import pickle
from typing import Type
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class RandomForest(ModelI):
    HEADER = b"RANDOMFOREST_MODEL"

    def __init__(self):
        super().__init__()
        self.model = None

    def infer(self, **args):
        data = args.get("data")
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict(data)

    def train(self, min_loss: float, max_epochs: int, **args):
        X = args["X"]
        y = args["y"]
        model = RandomForestRegressor(n_estimators=max_epochs)
        model.fit(X, y)
        self.model = model
        # simulate loss as mean squared error
        y_pred = self.model.predict(X)
        loss = np.mean((y - y_pred) ** 2)
        return float(loss)

    def serialize(self) -> bytes:
        payload = {"model": self.model}
        return self.HEADER + pickle.dumps(payload)

    @classmethod
    def deserialize(cls: Type['RandomForest'], b: bytes) -> 'RandomForest':
        if not b.startswith(cls.HEADER):
            raise TypeError("Invalid model header for RandomForest.")
        data = pickle.loads(b[len(cls.HEADER):])
        obj = cls()
        obj.model = data["model"]
        return obj
