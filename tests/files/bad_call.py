

class BadModel(ModelInterface):
    def train(self, X, y, max_epochs=100, status_callback=None) -> float:
        eval("1+1")
        return 0.0

    def predict(self, X):
        return []
