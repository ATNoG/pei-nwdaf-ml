import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.interface.mlint import MLInterface


class TestMLInterface:
    """Tests for MLInterface class"""

    @pytest.fixture
    def ml_interface_config(self):
        """Configuration for MLInterface"""
        return {
            'kafka_host': 'localhost',
            'kafka_port': '9092',
            'mlflow_tracking_uri': 'http://localhost:5000',
            'mlflow_experiment_name': 'test_experiment',
            'data_storage_api_url': 'http://localhost:8080'
        }

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_ml_interface_initialization(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test MLInterface initialization"""
        ml_interface = MLInterface(**ml_interface_config)

        assert ml_interface.data_storage_api_url == 'http://localhost:8080'
        assert ml_interface._component_status is not None
        assert 'inference' in ml_interface._component_status
        assert 'training' in ml_interface._component_status
        assert 'model_registry' in ml_interface._component_status
        assert 'data_storage' in ml_interface._component_status

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_produce_to_kafka_success(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test successfully producing message to Kafka"""
        mock_bridge_instance = Mock()
        mock_bridge_instance.produce.return_value = True
        mock_kafka_bridge.return_value = mock_bridge_instance

        ml_interface = MLInterface(**ml_interface_config)
        result = ml_interface.produce_to_kafka('test.topic', 'test message')

        assert result is True
        mock_bridge_instance.produce.assert_called_with('test.topic', 'test message')

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_produce_to_kafka_failure(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test producing to Kafka with failure"""
        mock_bridge_instance = Mock()
        mock_bridge_instance.produce.return_value = False
        mock_kafka_bridge.return_value = mock_bridge_instance

        ml_interface = MLInterface(**ml_interface_config)
        result = ml_interface.produce_to_kafka('test.topic', 'test message')

        assert result is False

    @patch('requests.request')
    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_request_data_from_storage_timeout(self, mock_mlflow_bridge, mock_kafka_bridge, mock_request, ml_interface_config):
        """Test data storage request timeout"""
        mock_request.side_effect = Exception("Timeout")

        ml_interface = MLInterface(**ml_interface_config)
        result = ml_interface.request_data_from_storage('/api/data', timeout=1)

        assert result is None

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_update_component_status(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test updating component status"""
        ml_interface = MLInterface(**ml_interface_config)

        ml_interface.update_component_status('inference', 'ready', model='test_model')
        status = ml_interface.get_component_status('inference')

        assert status['status'] == 'ready'
        assert status['model'] == 'test_model'


class TestMLInterfaceDataStorage:
    """Tests for MLInterface data storage methods"""

    @pytest.fixture
    def ml_interface_config(self):
        """Configuration for MLInterface"""
        return {
            'kafka_host': 'localhost',
            'kafka_port': '9092',
            'mlflow_tracking_uri': 'http://localhost:5000',
            'data_storage_api_url': 'http://localhost:8080'
        }

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_get_training_data_with_valid_response(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test that get_training_data returns data on valid response"""
        ml_interface = MLInterface(**ml_interface_config)

        # Mock the internal request_data_from_storage method
        with patch.object(ml_interface, 'request_data_from_storage', return_value={'data': 'test'}):
            result = ml_interface.get_training_data(
                start_time='2024-01-01',
                end_time='2024-01-02',
                data_type='timeseries'
            )

            # Verify result matches response
            assert result == {'data': 'test'}

    @pytest.mark.asyncio
    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    async def test_get_training_data_async_returns_none_on_error(self, mock_mlflow_bridge,
                                                                mock_kafka_bridge, ml_interface_config):
        """Test async training data returns None on connection error"""
        ml_interface = MLInterface(**ml_interface_config)

        with patch.object(ml_interface, 'request_data_from_storage_async',
                         new_callable=AsyncMock, return_value=None):
            result = await ml_interface.get_training_data_async(
                start_time='2024-01-01',
                end_time='2024-01-02'
            )

            # Should return None when storage request fails
            assert result is None

    @patch('requests.get')
    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_check_data_storage_connection_failure(self, mock_mlflow_bridge, mock_kafka_bridge,
                                                   mock_get, ml_interface_config):
        """Test checking data storage connection failure"""
        mock_get.side_effect = Exception("Connection error")

        ml_interface = MLInterface(**ml_interface_config)
        connected = ml_interface.check_data_storage_connection()

        assert connected is False


class TestMLInterfaceComponentStatus:
    """Tests for MLInterface component status tracking"""

    @pytest.fixture
    def ml_interface_config(self):
        """Configuration for MLInterface"""
        return {
            'kafka_host': 'localhost',
            'kafka_port': '9092',
            'data_storage_api_url': 'http://localhost:8080'
        }

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_get_component_status_existing(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test getting status of existing component"""
        ml_interface = MLInterface(**ml_interface_config)

        status = ml_interface.get_component_status('inference')

        assert status is not None
        assert 'status' in status

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_update_component_status_updates_existing(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test updating component status for existing component"""
        ml_interface = MLInterface(**ml_interface_config)

        # Update an existing component (inference is initialized in __init__)
        ml_interface.update_component_status('inference', 'active', test_key='test_value')
        status = ml_interface.get_component_status('inference')

        assert status['status'] == 'active'
        assert status['test_key'] == 'test_value'

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_get_system_health_returns_dict(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test getting system health returns proper structure"""
        ml_interface = MLInterface(**ml_interface_config)

        health = ml_interface.get_system_health()

        assert isinstance(health, dict)
        assert 'kafka' in health or 'overall_status' in health
        assert 'components' in health

    @patch('src.interface.mlint.PyKafBridge')
    @patch('src.interface.mlint.MLFlowBridge')
    def test_multiple_status_updates(self, mock_mlflow_bridge, mock_kafka_bridge, ml_interface_config):
        """Test multiple status updates for same component"""
        ml_interface = MLInterface(**ml_interface_config)

        # Use existing component 'training' from initialization
        ml_interface.update_component_status('training', 'initializing')
        ml_interface.update_component_status('training', 'ready', version='1.0')
        ml_interface.update_component_status('training', 'running', active=True)

        status = ml_interface.get_component_status('training')

        assert status['status'] == 'running'
        assert status['active'] is True
        assert status['version'] == '1.0'
