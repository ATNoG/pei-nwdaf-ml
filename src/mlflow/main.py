# MLFLOW PYTHON BRIDGE
# Class for model registry, metrics, and such (WIP)
#
# Author: Miguel Neto

import mlflow
import logging

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    datefmt="%m-%d %H:%M:%S",
                    handlers=[
                        # logging.FileHandler(f"./logs/student_{datetime.datetime.now().strftime("%d_%m_%y_at_%H_%M_%S")}.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

class MLFlowBridge():
    def __init__(self):
        self.tracking_uri = None
        self.experiment_name = None

    def start(self, tracking_uri="http://localhost:5000", experiment_name="test-ml"):
        """Initialize a tracking URI and experiment."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow initialized with URI: {tracking_uri}, Experiment: {experiment_name}")

    def verify_connection(self):
        """Verify connection to MLflow"""
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Active Experiment: {mlflow.get_experiment_by_name('my-first-experiment')}")

        # Test logging
        with mlflow.start_run():
            mlflow.log_param("test_param", "test_value")
            print("Successfully connected to MLflow!")

    def log_metrics(self, metrics, step=None):
        """Log multiple metrics at once.

        Arguments:
            metrics (dict): Dictionary of metric name-value pairs
            step (int, optional): Training step number
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def register_model(self, model_uri, model_name):
        """Register a model to MLflow Model Registry.

        Arguments:
            model_uri (str): URI of the model (e.g., 'runs:/<run_id>/model')
            model_name (str): Name for the registered model

        Returns:
            ModelVersion: The registered model version
        """
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(f"Model registered: {model_name}, version {model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def load_model(self, model_uri):
        """Load a model from MLflow.

        arguments:
            model_uri (str): URI of the model to load

        Returns:
            Loaded model object
        """
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model loaded from: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None


if __name__ == '__main__':
    bd = MLFlowBridge()
    bd.start()
    bd.verify_connection()
