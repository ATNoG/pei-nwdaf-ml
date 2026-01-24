#>TOCHECK< refactor in order to remove unused code and improve code readability
# Communication Interface for the ML Service component
#
# An abstraction layer which mediates messaging from
# different system components along with internal components
#
# Author: Miguel Neto

from utils.kmw import PyKafBridge

import asyncio
import logging
import os
import json
import urllib.request
import urllib.error
import urllib.parse
from threading import Thread
from typing import Optional, List, Dict, Any

from src.services.inference_service import InferenceService
from src.services.performance_monitor_service import PerformanceMonitor
from src.config.inference_type import InferenceConfig, get_all_inference_types
from src.routers.v1.websocket import performance_monitor_callback

from src.mlflow.mlf import MLFlowBridge

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
                    datefmt="%m-%d %H:%M:%S",
                    handlers=[
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


class MLInterface():

    class _topics():
        ML_INFERENCE_REQUEST='ml.inference.request'
        ML_INFERENCE_COMPLETE='ml.inference.complete'
        NETWORK_DATA_PROCESSED='network.data.processed'
        #'network.data.request'
    def __init__(self,
                 kafka_host: str = 'localhost',
                 kafka_port: str = '9092',
                 mlflow_tracking_uri: Optional[str] = None,
                 mlflow_experiment_name: str = 'default',
                 data_storage_api_url: Optional[str] = None):

        # Component status tracking
        self._component_status = {
            'inference': {'status': 'idle', 'current_model': None},
            'training': {'status': 'idle', 'active_runs': 0},
            'model_registry': {'status': 'unknown', 'model_count': 0},
            'data_storage': {'status': 'unknown', 'last_request': None}
        }

        self.bridge = PyKafBridge(
            hostname=kafka_host,
            port=kafka_port,
            debug_label='ML Interface Bridge'
        )
        self.topics = [
            'ml.inference.request',
            #'ml.inference.complete',
            'network.data.processed',
            #'network.data.request'
        ]
        self.mlflow_bridge = MLFlowBridge()
        self._mlflow_connected = False

        if mlflow_tracking_uri:
            self._initialize_mlflow(mlflow_tracking_uri, mlflow_experiment_name)
        else:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            self._initialize_mlflow(mlflow_uri, mlflow_experiment_name)

        # Data Storage API configuration
        self.data_storage_api_url = data_storage_api_url or os.getenv(
            "DATA_STORAGE_API_URL",
            "http://localhost:8001"
        )

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
        self.bridge.add_n_topics(self.topics)

        try:
            await self.bridge.start_consumer()
            logger.info("Started PyKaf Consumer")
            if self.bridge._consumer_task:
                await self.bridge._consumer_task
            return True
        except Exception as e:
            logger.error(f"Failed to start PyKaf consumer: {e}")
            return False

    def start_consumer_background(self) -> Thread:
        """
        Start the Kafka consumer in a background thread with its own event loop.
        This prevents rdkafka from blocking the main application.

        Returns:
            The background thread object
        """
        def kafka_worker():
            """Worker function that runs in background thread"""
            try:
                # Bind the handler BEFORE starting consumer
                self.bridge.add_n_topics(self.topics)
                self.bridge.bind_topic(self._topics.NETWORK_DATA_PROCESSED, self._predict_and_evaluate)
                self.bridge.bind_topic(self._topics.ML_INFERENCE_REQUEST, self._handle_inference_request)

                # Start consumer (without adding topics again)
                async def start():
                    await self.bridge.start_consumer()
                    logger.info("Started PyKaf Consumer")
                    if self.bridge._consumer_task:
                        await self.bridge._consumer_task

                asyncio.run(start())
                logger.info("Kafka consumer thread started successfully")

            except Exception as e:
                logger.error(f"Kafka consumer thread crashed: {e}")

        kafka_thread = Thread(
            target=kafka_worker,
            daemon=True,
            name="ml-kafka-consumer-thread"
        )
        kafka_thread.start()
        logger.info("Kafka consumer starting in background thread")
        return kafka_thread


    def schedule_async(self,coro):
        """
        Schedule an async coroutine from a sync function.
        Uses existing running loop if possible, otherwise creates one.
        """
        try:
            loop = asyncio.get_running_loop()
            # Running inside an event loop (e.g., FastAPI/WebSocket)
            return asyncio.create_task(coro)
        except RuntimeError:
            # No loop running, run in new loop (blocking)
            return asyncio.run(coro)


    async def _predict_and_store(self, service: InferenceService, config_name: str, cell_index: int, window_duration: int):
        """
        Async helper to run prediction and store result
        """
        try:
            response = await service.predict_cell_analytics(config_name, cell_index, window_duration)

            PerformanceMonitor.store(
                analytics_type=config_name,
                predicted_value=response.predicted_value,
                cell_index=cell_index,
                window_size=window_duration,
            )

        except Exception as e:
            logger.exception(f"Async prediction failed for {config_name}: {e}")


    def _predict_and_evaluate(self,  message_data: Dict[str, Any]) ->  Dict[str, Any]:
        content = message_data.get('content', '')
        if not content:
            logger.debug("Empty message content, skipping")
            return message_data

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse inference request JSON: {e}")
            return message_data

        """    transformed = {
            "window_start_time": window_start_dt,
            "window_end_time": window_end_dt,
            "window_duration_seconds": float(window_duration),
            "cell_index": data.get("cell_index"),
            "network": data.get("network") or "Unknown",
            "sample_count": sample_count,
            "primary_bandwidth": data.get("primary_bandwidth"),
            "ul_bandwidth": data.get("ul_bandwidth"),
        }

        # Flatten nested metric structures
        # Processor uses "mean_latency", storage uses "latency"
        metric_mapping = {
            "rsrp": "rsrp",
            "sinr": "sinr",
            "rsrq": "rsrq",
            "mean_latency": "latency",  # Map mean_latency -> latency
            "cqi": "cqi"
        }"""
        cell_index = data.get("cell_index",1)
        window_duration = int(data.get("window_duration_seconds",0))
        service:InferenceService = InferenceService(self)
        """def eval(cls,analytics_type , true_value:float, cell_index:int,window_size:int, window_start_time:int,   callback_function:Callable):
        """
        config:InferenceConfig
        for _,config in get_all_inference_types().items():
            try:
                true_value = data.get(f"mean_{config.name}",{}).get("mean",0)
                PerformanceMonitor.eval(config.name,
                    true_value,cell_index,
                    window_duration,
                    callback_function=performance_monitor_callback)

                self.schedule_async(
                    self._predict_and_store(service, config.name, cell_index, window_duration)
                )

            except Exception as e:
                logger.warning(f"Performance evaluation failled- {e}")
                continue

        return message_data


    def _handle_inference_request(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming inference requests from Kafka.
        Called automatically when messages arrive on ml.inference.request topic.

        Expected message format: (FOR NOW)
        {
            "cell_index": "12898855",
            "model_type": "ann",  # optional, defaults to ann
            "data": {
                "rsrp_mean": -86.0, "rsrp_max": -81.0, ...
            },
            "request_id": "unique-request-id"  # optional, for tracking
        }

        Args:
            message_data: Message data (from Kafka consumer)

        Returns the processed message data
        """
        import json
        import datetime
        from src.inference.inference import InferenceMaker

        logger.info(f"=== INFERENCE REQUEST HANDLER === Message keys: {list(message_data.keys()) if message_data else 'None'}")

        try:
            content = message_data.get('content', '')
            if not content:
                logger.debug("Empty message content, skipping")
                return message_data

            # Parse JSON content
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse inference request JSON: {e}")
                self._publish_inference_error(None, f"Invalid JSON: {e}")
                return message_data

            # Extract request parameters
            cell_index = data.get('cell_index')
            cell_indices = data.get('cell_indices')
            model_type = data.get('model_type', 'ann')
            inference_data = data.get('data')
            request_id = data.get('request_id', 'unknown')

            if not inference_data:
                logger.error("No 'data' field in inference request")
                self._publish_inference_error(request_id, "Missing 'data' field")
                return message_data

            if not cell_index and not cell_indices:
                logger.error("No cell_index or cell_indices in inference request")
                self._publish_inference_error(request_id, "Missing cell_index or cell_indices")
                return message_data

            logger.info(f"Processing inference request {request_id} for cell(s): {cell_index or cell_indices}")

            # Create inference maker and perform inference
            inference_maker = InferenceMaker(self)

            if cell_indices:
                # Batch inference
                result = inference_maker.infer(
                    cell_indices=cell_indices,
                    data=inference_data,
                    model_type=model_type
                )
                model_used = {
                    str(cell): inference_maker._get_cell_model_name(cell, model_type)
                    for cell in cell_indices
                }
            else:
                # Single cell inference
                result = inference_maker.infer(
                    cell_index=cell_index,
                    data=inference_data,
                    model_type=model_type
                )
                model_used = inference_maker._get_cell_model_name(cell_index, model_type)

            # Publish result to ml.inference.complete
            result_message = json.dumps({
                "request_id": request_id,
                "cell_index": cell_index,
                "cell_indices": cell_indices,
                "model_used": model_used,
                "result": result,
                "status": "success" if result is not None else "failed",
                "timestamp": datetime.datetime.now().timestamp()
            })

            self.produce_to_kafka(self._topics.ML_INFERENCE_COMPLETE, result_message)
            logger.info(f"Inference result published for request {request_id}")

            return message_data

        except Exception as e:
            logger.error(f"Error processing inference request: {e}")
            self._publish_inference_error(data.get('request_id', 'unknown') if 'data' in dir() else 'unknown', str(e))
            return message_data

    def _publish_inference_error(self, request_id: Optional[str], error: str) -> None:
        """Publish an inference error to the completion topic"""
        import json
        import datetime

        error_message = json.dumps({
            "request_id": request_id,
            "status": "error",
            "error": error,
            "timestamp": datetime.datetime.now().timestamp()
        })
        self.produce_to_kafka(self._topics.ML_INFERENCE_COMPLETE, error_message)

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

    def subscribe_topic(self, topic: str) -> bool:
        """Subscribe to a Kafka topic"""
        if not topic or not isinstance(topic, str):
            logger.error("Invalid topic provided to subscribe_topic")
            return False

        topic = topic.strip()
        if topic == "":
            logger.error("Empty topic name is not allowed")
            return False

        if topic not in self.topics:
            self.topics.append(topic)

        try:
            self.bridge.add_topic(topic)
            logger.info(f"Subscribed to topic: {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to topic {topic}: {e}")
            return False

    def get_subscribed_topics(self) -> List[str]:
        """Get list of subscribed Kafka topics"""
        return self.topics


    def request_data_from_storage(self,
                                   endpoint: str,
                                   params: Optional[Dict[str, Any]] = None,
                                   method: str = 'GET',
                                   data: Optional[Dict[str, Any]] = None,
                                   timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Request data from the Data Storage Query API

        Args:
            endpoint: API endpoint path (e.g., '/query/timeseries', '/query/analytics')
            params: Query parameters for GET requests
            method: HTTP method ('GET' or 'POST')
            data: JSON data for POST requests
            timeout: Request timeout in seconds

        Returns a dictionary with response data, or None if request failed
        """
        try:
            url = f"{self.data_storage_api_url.rstrip('/')}/{endpoint.lstrip('/')}"

            if params and method == 'GET':
                query_string = urllib.parse.urlencode(params)
                url = f"{url}?{query_string}"

            headers = {'Content-Type': 'application/json'}

            if method == 'POST' and data:
                json_data = json.dumps(data).encode('utf-8')
                req = urllib.request.Request(url, data=json_data, headers=headers, method='POST')
            else:
                req = urllib.request.Request(url, headers=headers, method=method)

            logger.info(f"Requesting data from storage: {method} {url}")
            self.update_component_status('data_storage', 'requesting', endpoint=endpoint)

            with urllib.request.urlopen(req, timeout=timeout) as response:
                response_data = response.read().decode('utf-8')
                result = json.loads(response_data)

                logger.info(f"Successfully received data from storage API: {endpoint}")
                self.update_component_status(
                    'data_storage',
                    'connected',
                    last_request=endpoint,
                    last_status='success'
                )

                return result

        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8') if e.fp else str(e)
            logger.error(f"HTTP error requesting data from storage: {e.code} - {error_msg}")
            self.update_component_status(
                'data_storage',
                'error',
                last_request=endpoint,
                last_status='http_error',
                error_code=e.code
            )
            return None

        except urllib.error.URLError as e:
            logger.error(f"URL error requesting data from storage: {e.reason}")
            self.update_component_status(
                'data_storage',
                'disconnected',
                last_request=endpoint,
                last_status='connection_error',
                error=str(e.reason)
            )
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from storage API: {e}")
            self.update_component_status(
                'data_storage',
                'error',
                last_request=endpoint,
                last_status='json_error'
            )
            return None

        except Exception as e:
            logger.error(f"Unexpected error requesting data from storage: {e}")
            self.update_component_status(
                'data_storage',
                'error',
                last_request=endpoint,
                last_status='unknown_error',
                error=str(e)
            )
            return None

    async def request_data_from_storage_async(self,
                                              endpoint: str,
                                              params: Optional[Dict[str, Any]] = None,
                                              method: str = 'GET',
                                              data: Optional[Dict[str, Any]] = None,
                                              timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        Wrapper for requesting data from Data Storage API

        Args:
            endpoint: API endpoint path
            params: Query parameters for GET requests
            method: HTTP method ('GET' or 'POST')
            data: JSON data for POST requests
            timeout: Request timeout in seconds

        Returns:
            Dictionary with response data or None if request failed
        """
        # Run sync method in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.request_data_from_storage,
            endpoint,
            params,
            method,
            data,
            timeout
        )

    def get_training_data(self,
                         start_time: Optional[str] = None,
                         end_time: Optional[str] = None,
                         data_type: str = 'latency',
                         filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Request training data from Data Storage component

        Args:
            start_time: Start timestamp for data query (ISO format)
            end_time: End timestamp for data query (ISO format)
            data_type: Type of data to fetch ('latency')
            filters: Additional filters for the query (e.g., cell_index, offset, limit)

        Returns:
            Training data dictionary or None if request failed
        """
        params = {}

        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
        if filters:
            params.update(filters)

        # Use the actual Data Storage API endpoint
        endpoint = '/api/v1/processed/latency/'

        logger.info(f"Requesting training data: {data_type} from {start_time} to {end_time}")
        return self.request_data_from_storage(endpoint, params=params)

    async def get_training_data_async(self,
                                     start_time: Optional[str] = None,
                                     end_time: Optional[str] = None,
                                     data_type: str = 'latency',
                                     filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Async version of get_training_data

        Args:
            start_time: Start timestamp for data query (ISO format)
            end_time: End timestamp for data query (ISO format)
            data_type: Type of data to fetch ('latency')
            filters: Additional filters for the query (e.g., cell_index, offset, limit)

        Returns:
            Training data dictionary or None if request failed
        """
        params = {}

        if start_time:
            params['start_time'] = start_time
        if end_time:
            params['end_time'] = end_time
        if filters:
            params.update(filters)

        # Use the actual Data Storage API endpoint
        endpoint = '/api/v1/processed/latency/'

        logger.info(f"Requesting training data (async): {data_type} from {start_time} to {end_time}")
        return await self.request_data_from_storage_async(endpoint, params=params)

    def get_latency_data(self,
                        start_time: str,
                        end_time: str,
                        cell_index: int,
                        offset: int = 0,
                        limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Request latency data from Data Storage component

        Args:
            start_time: Start timestamp for data query (ISO format)
            end_time: End timestamp for data query (ISO format)
            cell_index: Cell index (required)
            offset: Number of records to skip (default: 0)
            limit: Maximum number of records to return (default: 100, max: 1000)

        Returns:
            Latency data dictionary or None if request failed
        """
        params = {
            'start_time': start_time,
            'end_time': end_time,
            'cell_index': cell_index,
            'offset': offset,
            'limit': limit
        }

        endpoint = '/api/v1/processed/latency/'

        logger.info(f"Requesting latency data for cell {cell_index} from {start_time} to {end_time}")
        return self.request_data_from_storage(endpoint, params=params)

    async def get_latency_data_async(self,
                                    start_time: str,
                                    end_time: str,
                                    cell_index: int,
                                    offset: int = 0,
                                    limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Async version of get_latency_data

        Args:
            start_time: Start timestamp for data query (ISO format)
            end_time: End timestamp for data query (ISO format)
            cell_index: Cell index (required)
            offset: Number of records to skip (default: 0)
            limit: Maximum number of records to return (default: 100, max: 1000)

        Returns:
            Latency data dictionary or None if request failed
        """
        params = {
            'start_time': start_time,
            'end_time': end_time,
            'cell_index': cell_index,
            'offset': offset,
            'limit': limit
        }

        endpoint = '/api/v1/processed/latency/'

        logger.info(f"Requesting latency data (async) for cell {cell_index} from {start_time} to {end_time}")
        return await self.request_data_from_storage_async(endpoint, params=params)

    def fetch_known_cells(self) -> Optional[List[int]]:
        """
        Fetch list of known cell indexes from storage.

        Returns:
            List of cell indexes or None if request failed
        """
        try:
            #TODO: maybe don't hardcode this
            cells = self.request_data_from_storage(
                endpoint="/api/v1/cell/",
                method="GET"
            )

            if cells and isinstance(cells, list):
                logger.info(f"Fetched {len(cells)} known cells from storage")
                return cells

            logger.warning("No cells returned from storage")
            return []

        except Exception as e:
            logger.error(f"Error fetching known cells: {e}", exc_info=True)
            return None

    async def fetch_known_cells_async(self) -> Optional[List[int]]:
        """
        Async version of fetch_known_cells.

        Returns:
            List of cell indexes or None if request failed
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch_known_cells)

    async def fetch_latest_cell_data(
        self,
        endpoint: str,
        cell_index: int,
        window_duration_seconds: int,
        num_windows:int
    ) -> Optional[list[Dict[str, Any]]]:
        """
        Fetch latest data window for a cell.

        Args:
            endpoint: Storage endpoint to query
            cell_index: Cell identifier
            window_duration_seconds: Duration of window in seconds

        Returns:
            Latest data window or None if not found
        """
        try:
            params = {
                "cell_index": cell_index,
                "start_time": 0,
                "end_time": 9999999999,
                "offset": 0,
                "limit": num_windows,
                "window_duration_seconds": window_duration_seconds,
            }

            logger.info(f"Fetching latest data for cell {cell_index} with window {window_duration_seconds}s")

            data = await self.request_data_from_storage_async(
                endpoint=endpoint,
                params=params,
                method="GET"
            )

            if not data or len(data) == 0:
                logger.warning(f"No data found for cell {cell_index}")
                return None

            logger.info(f"Found data for cell {cell_index}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for cell {cell_index}: {e}", exc_info=True)
            return None

    def fetch_training_data_for_cells(
        self,
        endpoint: str,
        cell_indexes: List[int],
        window_duration_seconds: int,
        data_limit_per_cell: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch training data from storage for multiple cells.

        Args:
            endpoint: Storage endpoint to query
            cell_indexes: List of cell indexes to fetch data from
            window_duration_seconds: Duration of window in seconds
            data_limit_per_cell: Maximum samples to fetch per cell

        Returns:
            List of all data samples collected from all cells
        """
        all_data = []

        for cell_index in cell_indexes:
            try:
                params = {
                    "cell_index": cell_index,
                    "start_time": 0,
                    "end_time": 9999999999,
                    "offset": 0,
                    "limit": data_limit_per_cell,
                    "window_duration_seconds": window_duration_seconds
                }

                data = self.request_data_from_storage(
                    endpoint=endpoint,
                    params=params,
                    method="GET"
                )

                if data and len(data) > 0:
                    all_data.extend(data)
                    logger.info(f"Fetched {len(data)} samples from cell {cell_index}")

            except Exception as e:
                logger.warning(f"Error fetching data for cell {cell_index}: {e}")
                continue

        logger.info(f"Total collected: {len(all_data)} samples from {len(cell_indexes)} cells")
        return all_data

    def check_data_storage_connection(self) -> bool:
        """
        Check if Data Storage API is reachable

        Returns:
            True if connected, False otherwise
        """
        try:
            result = self.request_data_from_storage('/health', timeout=5)
            return result is not None
        except Exception as e:
            logger.warning(f"Data Storage API not reachable: {e}")
            return False

    def is_mlflow_connected(self) -> bool:
        """Check if MLFlow connection is active"""
        return self._mlflow_connected

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLFlow registry

        Args:
            model_uri: URI of the model (e.g., 'models:/ModelName/Production'
                       or 'models:/ModelName/1' or 'runs:/<run_id>/model')
            )

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
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            registered_models = client.search_registered_models()

            models_info = []
            for rm in registered_models or []:
                latest_versions = rm.latest_versions or []
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

    def get_best_model(self, model_name_prefix: Optional[str] = None, metric: str = 'accuracy') -> Optional[Dict[str, Any]]:
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

            for rm in registered_models or []:
                # Filter by prefix if provided
                if model_name_prefix and not rm.name.startswith(model_name_prefix):
                    continue

                # Check latest production version
                for version in rm.latest_versions or []:
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

        data_storage_connected = self.check_data_storage_connection()

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
            'data_storage': {
                'connected': data_storage_connected,
                'api_url': self.data_storage_api_url
            },
            'components': self._component_status
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the interface"""
        logger.info("Shutting down ML Interface...")
        await self.shutdown_bridge()
        logger.info("ML Interface shutdown complete")


    def download_model_artifact(self, model_name: str, artifact_path: str, dst_path: str) -> str:
        """
        Download an artifact file from MLflow for the given model.
        Returns local path to downloaded file.
        """
        import mlflow

        # Find latest production model version
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            raise ValueError(f"No Production version found for {model_name}")

        run_id = versions[0].run_id
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=dst_path)
        return local_path
