# Communication Interface for the ML Service component
#
# An abstraction layer which mediates messaging from
# different system components along with internal components
#
# Author: Miguel Neto

import fastapi
from kmw import PyKafBridge

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any

from src.mlflow.main import MLFlowBridge

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
                    datefmt="%m-%d %H:%M:%S",
                    handlers=[
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


class MLInterface():
    def __init__(self,
                 kafka_host: str = 'localhost',
                 kafka_port: str = '9092',
                 mlflow_tracking_uri: str = None,
                 mlflow_experiment_name: str = 'default'):

        self.bridge = PyKafBridge(
            hostname=kafka_host,
            port=kafka_port,
            debug_label='ML Interface Bridge'
        )
        self.topics = [
            'ml.inference.request',
            'ml.inference.response',
            'ml.training.request',
            'ml.training.complete',
            'network.data.processed',
            'network.data.request'
        ]
        self.bridge.add_n_topics(self.topics)

        self.mlflow_bridge = MLFlowBridge()
        self._mlflow_connected = False

        if mlflow_tracking_uri:
            self._initialize_mlflow(mlflow_tracking_uri, mlflow_experiment_name)
        else:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            self._initialize_mlflow(mlflow_uri, mlflow_experiment_name)

        self._component_status = {
            'inference': {'status': 'idle', 'current_model': None},
            'training': {'status': 'idle', 'active_runs': 0},
            'model_registry': {'status': 'unknown', 'model_count': 0}
        }

    def _initialize_mlflow(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize MLFlow connection"""
        try:
            self.mlflow_bridge.start(
                tracking_uri=tracking_uri,
                experiment_name=experiment_name
            )
            self._mlflow_connected = True
            self._component_status['model_registry']['status'] = 'connected'
            logger.info(f"MLFlow initialized at {tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLFlow: {e}")
            self._mlflow_connected = False
            self._component_status['model_registry']['status'] = 'disconnected'

    def is_connected(self) -> bool:
        """Check if Kafka connection is active"""
        return self.bridge is not None and self.bridge.consumer is not None

    def is_consumer_running(self) -> bool:
        """Check if Kafka consumer is running"""
        return self.bridge._running

    async def start_consumer(self) -> bool:
        """Start the Kafka consumer"""
        try:
            await self.bridge.start_consumer()
            logger.info("Started PyKaf Consumer")
            return True
        except Exception as e:
            logger.error(f"Failed to start PyKaf consumer: {e}")
            return False

    async def shutdown_bridge(self) -> None:
        """Shutdown Kafka bridge"""
        if self.bridge:
            await self.bridge.close()
        logger.info("PyKaf bridge has shutdown")

    def produce_to_kafka(self, topic: str, message: str) -> bool:
        """
        Produce a message to a specific Kafka topic

        Args:
            topic: Target topic name
            message: Message contents

        Returns True when successfully produced, otherwise False
        """
        ret = self.bridge.produce(topic, message)
        if ret:
            logger.debug(f"Produced message with success to {topic}")
        else:
            logger.error(f"Failed producing a message to {topic}")
        return ret

    def get_messages(self, topic: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve consumed messages from a topic

        Args:
            topic: Topic name
            limit: Maximum number of messages to return (None = all)

        Returns a list of messages (as dictionaries)
        """
        messages = self.bridge.get_topic(topic)
        if limit and messages:
            return messages[-limit:]
        return messages if messages else []

    def get_subscribed_topics(self) -> List[str]:
        """Get list of subscribed Kafka topics"""
        return self.topics

    def is_mlflow_connected(self) -> bool:
        """Check if MLFlow connection is active"""
        return self._mlflow_connected

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLFlow registry

        Args:
            model_uri: URI of the model (e.g., 'models:/ModelName/Production'
                       or 'models:/ModelName/1' or 'runs:/<run_id>/model')

        Returns:
            Loaded model object or None if failed
        """
        if not self._mlflow_connected:
            logger.error("MLFlow not connected - cannot load model")
            return None

        try:
            model = self.mlflow_bridge.load_model(model_uri)
            logger.info(f"Successfully loaded model from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_uri}: {e}")
            return None

    def get_model_by_name(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Any:
        """

        Get a model from MLFlow registry by name

        Args:
            model_name: Name of the registered model
            version: Specific version number (e.g., '1', '2')
            stage: Model stage ('Production', 'Staging', 'Archived', 'None')

        Returns:
            Loaded model object or None

        """
        if not self._mlflow_connected:
            logger.error("MLFlow not connected")
            return None

        # Construct model URI
        if stage:
            model_uri = f"models:/{model_name}/{stage}"
        elif version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            # Default to latest version
            model_uri = f"models:/{model_name}/latest"

        return self.load_model(model_uri)

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models in MLFlow

        Returns:
            List of model information dictionaries
        """
        if not self._mlflow_connected:
            logger.warning("MLFlow not connected - returning empty list")
            return []

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            registered_models = client.search_registered_models()

            models_info = []
            for rm in registered_models:
                latest_versions = rm.latest_versions
                models_info.append({
                    'name': rm.name,
                    'creation_timestamp': rm.creation_timestamp,
                    'last_updated_timestamp': rm.last_updated_timestamp,
                    'description': rm.description,
                    'latest_versions': [
                        {
                            'version': v.version,
                            'stage': v.current_stage,
                            'run_id': v.run_id
                        } for v in latest_versions
                    ]
                })

            self._component_status['model_registry']['model_count'] = len(models_info)
            return models_info

        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []

    def get_model_metrics(self, run_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific model run

        Args:
            run_id: MLFlow run ID

        Returns:
            Dictionary of metrics
        """
        if not self._mlflow_connected:
            logger.warning("MLFlow not connected")
            return {}

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            run = client.get_run(run_id)
            return run.data.metrics

        except Exception as e:
            logger.error(f"Failed to get metrics for run {run_id}: {e}")
            return {}

    def log_inference_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> bool:
        """
        Log inference metrics to MLFlow

        Args:
            metrics: Dictionary of metric name-value pairs
            step: Optional step number

        Returns:
            True if successful, False otherwise
        """
        if not self._mlflow_connected:
            logger.warning("MLFlow not connected - metrics not logged")
            return False

        try:
            self.mlflow_bridge.log_metrics(metrics, step=step)
            return True
        except Exception as e:
            logger.error(f"Failed to log inference metrics: {e}")
            return False

    def register_model(self, model_uri: str, model_name: str) -> Optional[Any]:
        """
        Register a model to MLFlow Model Registry

        Args:
            model_uri: URI of the model (e.g., 'runs:/<run_id>/model')
            model_name: Name for the registered model

        Returns:
            ModelVersion object or None
        """
        if not self._mlflow_connected:
            logger.error("MLFlow not connected - cannot register model")
            return None

        return self.mlflow_bridge.register_model(model_uri, model_name)

    def get_best_model(self, model_name_prefix: str = None, metric: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """
        Find the best performing model based on a metric

        Args:
            model_name_prefix: Filter models by name prefix (optional)
            metric: Metric to use for comparison (default: 'accuracy')

        Returns:
            Dictionary with model info or None
        """
        if not self._mlflow_connected:
            logger.warning("MLFlow not connected")
            return None

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            registered_models = client.search_registered_models()

            best_model = None
            best_metric_value = -float('inf')

            for rm in registered_models:
                # Filter by prefix if provided
                if model_name_prefix and not rm.name.startswith(model_name_prefix):
                    continue

                # Check latest production version
                for version in rm.latest_versions:
                    if version.current_stage == 'Production':
                        metrics = self.get_model_metrics(version.run_id)
                        metric_value = metrics.get(metric, -float('inf'))

                        if metric_value > best_metric_value:
                            best_metric_value = metric_value
                            best_model = {
                                'name': rm.name,
                                'version': version.version,
                                'stage': version.current_stage,
                                'run_id': version.run_id,
                                'metric': metric,
                                'metric_value': metric_value,
                                'model_uri': f"models:/{rm.name}/{version.version}"
                            }

            return best_model

        except Exception as e:
            logger.error(f"Failed to find best model: {e}")
            return None

    def update_component_status(self, component: str, status: str, **kwargs) -> None:
        """
        Update the status of an internal component

        Args:
            component: Component name ('inference', 'training', etc.)
            status: Status string
            **kwargs: Additional status fields
        """
        if component in self._component_status:
            self._component_status[component]['status'] = status
            self._component_status[component].update(kwargs)
            logger.debug(f"Updated {component} status: {status}")

    def get_component_status(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of a specific component or all components

        Args:
            component: Component name or None for all

        Returns:
            Status dictionary
        """
        if component:
            return self._component_status.get(component, {'status': 'unknown'})
        return self._component_status

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status

        Returns:
            Comprehensive health status dictionary
        """
        kafka_healthy = self.is_connected() and self.is_consumer_running()
        mlflow_healthy = self.is_mlflow_connected()

        # Determine overall status
        if kafka_healthy and mlflow_healthy:
            overall_status = 'healthy'
        elif kafka_healthy or mlflow_healthy:
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'

        return {
            'overall_status': overall_status,
            'kafka': {
                'connected': self.is_connected(),
                'consumer_running': self.is_consumer_running(),
                'subscribed_topics': self.get_subscribed_topics()
            },
            'mlflow': {
                'connected': self.is_mlflow_connected(),
                'tracking_uri': self.mlflow_bridge.tracking_uri if self._mlflow_connected else None
            },
            'components': self._component_status
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the interface"""
        logger.info("Shutting down ML Interface...")
        await self.shutdown_bridge()
        logger.info("ML Interface shutdown complete")
