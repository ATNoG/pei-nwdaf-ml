

class BadModel(ModelInterface):
    def train(self, X, y, max_epochs=100, status_callback=None) -> float:
        _ = self.__class__
        return 0.0

    def predict(self, X):
        return []
