import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import json

from src.routers import data, inference, kafka
from src.schemas.data import DataQueryRequest, DataStorageRequest
from src.schemas.inference import InferenceRequest, ModelSelectionRequest, AutoModeRequest
from src.schemas.kafka import KafkaMessage


@pytest.fixture
def mock_ml_interface():
    """Create a mock ML interface"""
    mock_interface = Mock()
    mock_interface.get_training_data_async = AsyncMock()
    mock_interface.request_data_from_storage_async = AsyncMock()
    mock_interface.check_data_storage_connection = Mock(return_value=True)
    mock_interface.get_component_status = Mock(return_value={"status": "active"})
    mock_interface.data_storage_api_url = "http://mock-storage:8080"
    mock_interface.produce_to_kafka = Mock(return_value=True)
    mock_interface.get_messages = Mock(return_value=[])
    mock_interface.get_subscribed_topics = Mock(return_value=["test.topic"])
    mock_interface.is_consumer_running = Mock(return_value=True)
    mock_interface.is_mlflow_connected = Mock(return_value=True)
    mock_interface.list_registered_models = Mock(return_value=[])
    mock_interface.get_best_model = Mock(return_value=None)
    mock_interface.get_model_metrics = Mock(return_value={})
    return mock_interface


@pytest.fixture
def app_with_data_router(mock_ml_interface):
    """Create FastAPI app with data router"""
    app = FastAPI()
    app.include_router(data.router, prefix="/data", tags=["data"])
    app.state.ml_interface = mock_ml_interface
    return app


@pytest.fixture
def app_with_inference_router(mock_ml_interface):
    """Create FastAPI app with inference router"""
    app = FastAPI()
    app.include_router(inference.router, prefix="/ml", tags=["ml"])
    app.state.ml_interface = mock_ml_interface
    return app


@pytest.fixture
def app_with_kafka_router(mock_ml_interface):
    """Create FastAPI app with kafka router"""
    app = FastAPI()
    app.include_router(kafka.router, prefix="/kafka", tags=["kafka"])
    app.state.ml_interface = mock_ml_interface
    return app


class TestDataRouter:
    """Tests for data router endpoints"""

    def test_query_training_data_success(self, app_with_data_router, mock_ml_interface):
        """Test successful training data query"""
        client = TestClient(app_with_data_router)
        
        mock_ml_interface.get_training_data_async.return_value = {
            "records": [{"id": 1, "value": 100}],
            "count": 1
        }
        
        response = client.post(
            "/data/query",
            json={
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-01-02T00:00:00",
                "data_type": "timeseries",
                "filters": {"cell_id": "123"}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "query" in data

    def test_query_training_data_no_data(self, app_with_data_router, mock_ml_interface):
        """Test training data query with no data returned"""
        client = TestClient(app_with_data_router)
        
        mock_ml_interface.get_training_data_async.return_value = None
        
        response = client.post(
            "/data/query",
            json={
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-01-02T00:00:00",
                "data_type": "timeseries"
            }
        )
        
        assert response.status_code == 503
        assert "Failed to retrieve data" in response.json()["detail"]

    def test_query_training_data_exception(self, app_with_data_router, mock_ml_interface):
        """Test training data query with exception"""
        client = TestClient(app_with_data_router)
        
        mock_ml_interface.get_training_data_async.side_effect = Exception("Connection error")
        
        response = client.post(
            "/data/query",
            json={
                "start_time": "2024-01-01T00:00:00",
                "end_time": "2024-01-02T00:00:00",
                "data_type": "timeseries"
            }
        )
        
        assert response.status_code == 500
        assert "Query error" in response.json()["detail"]

    def test_request_from_storage_success(self, app_with_data_router, mock_ml_interface):
        """Test successful generic storage request"""
        client = TestClient(app_with_data_router)
        
        mock_ml_interface.request_data_from_storage_async.return_value = {
            "result": "success"
        }
        
        response = client.post(
            "/data/request",
            json={
                "endpoint": "/api/metrics",
                "method": "GET",
                "params": {"limit": 100},
                "timeout": 30
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "result" in data

    def test_get_storage_status_success(self, app_with_data_router, mock_ml_interface):
        """Test getting storage status"""
        client = TestClient(app_with_data_router)
        
        response = client.get("/data/storage/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["connected"] is True
        assert "api_url" in data

    def test_test_storage_connection_success(self, app_with_data_router, mock_ml_interface):
        """Test storage connection test endpoint"""
        client = TestClient(app_with_data_router)
        
        mock_ml_interface.request_data_from_storage_async.return_value = {
            "status": "healthy"
        }
        
        response = client.get("/data/storage/test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Successfully connected" in data["message"]


class TestInferenceRouter:
    """Tests for inference router endpoints"""

    def test_ml_inference_success(self, app_with_inference_router, mock_ml_interface):
        """Test successful ML inference"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.infer.return_value = [1.2, 3.4, 5.6]
            mock_inference_maker.get_current_model_info.return_value = {
                'model_id': 'models:/TestModel/1'
            }
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post(
                "/ml/inference",
                json={
                    "data": {"features": [1, 2, 3, 4, 5]},
                    "publish_result": False
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["predictions"] == [1.2, 3.4, 5.6]

    def test_ml_inference_with_specific_model(self, app_with_inference_router, mock_ml_interface):
        """Test ML inference with specific model request"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.set_model_by_name.return_value = True
            mock_inference_maker.infer.return_value = [7.8, 9.0]
            mock_inference_maker.get_current_model_info.return_value = {
                'model_id': 'models:/SpecificModel/2'
            }
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post(
                "/ml/inference",
                json={
                    "data": {"features": [1, 2, 3]},
                    "model_name": "SpecificModel",
                    "model_version": "2"
                }
            )
            
            assert response.status_code == 200
            mock_inference_maker.set_model_by_name.assert_called_once()

    def test_ml_inference_model_not_found(self, app_with_inference_router, mock_ml_interface):
        """Test ML inference with non-existent model"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.set_model_by_name.return_value = False
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post(
                "/ml/inference",
                json={
                    "data": {"features": [1, 2, 3]},
                    "model_name": "InvalidModel"
                }
            )
            
            assert response.status_code == 404
            assert "Model not found" in response.json()["detail"]

    def test_ml_inference_failure(self, app_with_inference_router, mock_ml_interface):
        """Test ML inference failure"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.infer.return_value = None
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post(
                "/ml/inference",
                json={"data": {"features": [1, 2, 3]}}
            )
            
            assert response.status_code == 500
            assert "Inference failed" in response.json()["detail"]

    def test_set_model_success(self, app_with_inference_router, mock_ml_interface):
        """Test successfully setting a model"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.set_model_by_name.return_value = True
            mock_inference_maker.get_current_model_info.return_value = {
                'model_id': 'models:/MyModel/Production',
                'auto_mode': False
            }
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post(
                "/ml/set-model",
                json={
                    "model_name": "MyModel",
                    "stage": "Production"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "model_info" in data

    def test_set_model_not_found(self, app_with_inference_router, mock_ml_interface):
        """Test setting a model that doesn't exist"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.set_model_by_name.return_value = False
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post(
                "/ml/set-model",
                json={"model_name": "InvalidModel"}
            )
            
            assert response.status_code == 404

    def test_toggle_auto_mode_enable(self, app_with_inference_router, mock_ml_interface):
        """Test enabling auto mode"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.toggle_auto_select = Mock()
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post(
                "/ml/auto-mode",
                json={"auto_mode": True}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["auto_mode"] is True
            mock_inference_maker.toggle_auto_select.assert_called_with(True)

    def test_get_current_model(self, app_with_inference_router, mock_ml_interface):
        """Test getting current model info"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.get_current_model_info.return_value = {
                'model_id': 'models:/TestModel/1',
                'auto_mode': False,
                'model_loaded': True,
                'failed_retrieves': 0
            }
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.get("/ml/current-model")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "model_info" in data

    def test_clear_cache(self, app_with_inference_router, mock_ml_interface):
        """Test clearing model cache"""
        client = TestClient(app_with_inference_router)
        
        with patch('src.routers.inference.get_inference_maker') as mock_get_inference:
            mock_inference_maker = Mock()
            mock_inference_maker.clear_cache = Mock()
            mock_get_inference.return_value = mock_inference_maker
            
            response = client.post("/ml/clear-cache")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            mock_inference_maker.clear_cache.assert_called_once()

    def test_list_models(self, app_with_inference_router, mock_ml_interface):
        """Test listing all models"""
        client = TestClient(app_with_inference_router)
        
        mock_ml_interface.list_registered_models.return_value = [
            {"name": "Model1", "creation_timestamp": 123456},
            {"name": "Model2", "creation_timestamp": 123457}
        ]
        
        response = client.get("/ml/models")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Model1"

    def test_get_best_model(self, app_with_inference_router, mock_ml_interface):
        """Test getting best model"""
        client = TestClient(app_with_inference_router)
        
        mock_ml_interface.get_best_model.return_value = {
            "name": "BestModel",
            "version": "1",
            "model_uri": "models:/BestModel/Production",
            "metric_value": 0.95
        }
        
        response = client.get("/ml/best-model?metric=accuracy")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "best_model" in data


class TestKafkaRouter:
    """Tests for Kafka router endpoints"""

    def test_produce_message_success(self, app_with_kafka_router, mock_ml_interface):
        """Test successfully producing a message to Kafka"""
        client = TestClient(app_with_kafka_router)
        
        mock_ml_interface.produce_to_kafka.return_value = True
        
        response = client.post(
            "/kafka/produce",
            json={
                "topic": "test.topic",
                "message": '{"data": "test"}'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["topic"] == "test.topic"
        mock_ml_interface.produce_to_kafka.assert_called_with(
            "test.topic", '{"data": "test"}'
        )

    def test_produce_message_failure(self, app_with_kafka_router, mock_ml_interface):
        """Test failed message production to Kafka"""
        client = TestClient(app_with_kafka_router)
        
        mock_ml_interface.produce_to_kafka.return_value = False
        
        response = client.post(
            "/kafka/produce",
            json={
                "topic": "test.topic",
                "message": "test message"
            }
        )
        
        assert response.status_code == 500
        assert "Failed to produce message" in response.json()["detail"]

    def test_get_messages_success(self, app_with_kafka_router, mock_ml_interface):
        """Test retrieving messages from a topic"""
        client = TestClient(app_with_kafka_router)
        
        mock_ml_interface.get_messages.return_value = [
            {"offset": 0, "message": "msg1"},
            {"offset": 1, "message": "msg2"}
        ]
        
        response = client.get("/kafka/messages/test.topic?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["topic"] == "test.topic"
        assert data["count"] == 2
        assert len(data["messages"]) == 2
        mock_ml_interface.get_messages.assert_called_with("test.topic", limit=10)

    def test_get_messages_empty(self, app_with_kafka_router, mock_ml_interface):
        """Test retrieving messages when none exist"""
        client = TestClient(app_with_kafka_router)
        
        mock_ml_interface.get_messages.return_value = []
        
        response = client.get("/kafka/messages/empty.topic")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["messages"] == []

    def test_list_topics_success(self, app_with_kafka_router, mock_ml_interface):
        """Test listing subscribed topics"""
        client = TestClient(app_with_kafka_router)
        
        mock_ml_interface.get_subscribed_topics.return_value = [
            "topic1", "topic2", "topic3"
        ]
        mock_ml_interface.is_consumer_running.return_value = True
        
        response = client.get("/kafka/topics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 3
        assert len(data["topics"]) == 3
        assert data["consumer_running"] is True