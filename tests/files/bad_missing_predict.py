

class BadModel(ModelInterface):
    def train(self, X, y, max_epochs=100, status_callback=None) -> float:
        return 0.0
