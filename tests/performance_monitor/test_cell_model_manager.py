"""
Tests for CellModelManager

Tests the core functionality of automatic model management per network cell.
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.performance_monitor.cell_model_manager import CellModelManager


class TestCellModelManager:
    """Tests for CellModelManager class"""

    @pytest.fixture
    def mock_ml_interface(self):
        """Create a mock ML interface"""
        mock_interface = Mock()
        mock_interface.is_mlflow_connected = Mock(return_value=True)
        return mock_interface

    @pytest.fixture
    def cell_manager(self, mock_ml_interface):
        """Create a CellModelManager instance with mocked dependencies"""
        return CellModelManager(mock_ml_interface)

    def test_init_creates_empty_cache(self, mock_ml_interface):
        """Test that CellModelManager initializes with empty cache"""
        manager = CellModelManager(mock_ml_interface)

        assert manager.cell_cache == {}
        assert manager.ml_interface == mock_ml_interface

    def test_extract_cell_index_valid(self, cell_manager):
        """Test extracting cell_index from valid data"""
        data = {"cell_index": "12898855", "other_field": "value"}

        result = cell_manager._extract_cell_index(data)

        assert result == "12898855"

    def test_extract_cell_index_missing(self, cell_manager):
        """Test extracting cell_index when field is missing"""
        data = {"other_field": "value"}

        result = cell_manager._extract_cell_index(data)

        assert result is None

    def test_extract_cell_index_empty_data(self, cell_manager):
        """Test extracting cell_index from empty dict"""
        result = cell_manager._extract_cell_index({})

        assert result is None

    def test_process_network_data_empty_content(self, cell_manager):
        """Test processing message with empty content"""
        message = {"content": ""}

        result = cell_manager.process_network_data(message)

        assert result == message
        assert cell_manager.cell_cache == {}

    def test_process_network_data_invalid_json(self, cell_manager):
        """Test processing message with invalid JSON"""
        message = {"content": "not valid json"}

        result = cell_manager.process_network_data(message)

        assert result == message
        assert cell_manager.cell_cache == {}

    def test_process_network_data_no_cell_index(self, cell_manager):
        """Test processing message without cell_index"""
        message = {"content": json.dumps({"some_field": "value"})}

        result = cell_manager.process_network_data(message)

        assert result == message

    def test_process_network_data_model_exists_in_cache(self, cell_manager):
        """Test processing when model already exists in cache"""
        cell_index = "12898855"
        cell_manager.cell_cache[cell_index] = {
            'model_name': f'cell_{cell_index}_xgboost',
            'status': 'created'
        }

        message = {"content": json.dumps({"cell_index": cell_index})}

        result = cell_manager.process_network_data(message)

        # Should return message without trying to create new model
        assert result == message
        assert cell_manager.cell_cache[cell_index]['status'] == 'created'

    @patch.object(CellModelManager, '_check_model_exists')
    @patch.object(CellModelManager, '_create_model_for_cell')
    def test_process_network_data_creates_model_when_missing(
        self, mock_create, mock_check, cell_manager
    ):
        """Test that model is created when it doesn't exist"""
        mock_check.return_value = False
        cell_index = "12898855"
        message = {"content": json.dumps({"cell_index": cell_index})}

        cell_manager.process_network_data(message)

        mock_check.assert_called_once_with(cell_index)
        mock_create.assert_called_once()

    @patch.object(CellModelManager, '_check_model_exists')
    @patch.object(CellModelManager, '_create_model_for_cell')
    def test_process_network_data_skips_creation_when_exists(
        self, mock_create, mock_check, cell_manager
    ):
        """Test that model creation is skipped when model exists"""
        mock_check.return_value = True
        message = {"content": json.dumps({"cell_index": "12898855"})}

        cell_manager.process_network_data(message)

        mock_create.assert_not_called()

    def test_process_network_data_handles_exception(self, cell_manager):
        """Test that exceptions are handled gracefully"""
        # Pass None to trigger exception
        result = cell_manager.process_network_data(None)

        # Should return None without raising
        assert result is None

    def test_check_model_exists_in_cache(self, cell_manager):
        """Test that cache hit returns True immediately"""
        cell_index = "12898855"
        cell_manager.cell_cache[cell_index] = {'model_name': 'test'}

        result = cell_manager._check_model_exists(cell_index)

        assert result is True

    def test_check_model_exists_mlflow_disconnected(self, cell_manager):
        """Test behavior when MLflow is disconnected"""
        cell_manager.ml_interface.is_mlflow_connected.return_value = False

        result = cell_manager._check_model_exists("12898855")

        assert result is False

    def test_get_cell_model_info_existing(self, cell_manager):
        """Test getting info for existing cell"""
        cell_index = "12898855"
        expected_info = {'model_name': 'test_model', 'status': 'created'}
        cell_manager.cell_cache[cell_index] = expected_info

        result = cell_manager.get_cell_model_info(cell_index)

        assert result == expected_info

    def test_get_cell_model_info_missing(self, cell_manager):
        """Test getting info for non-existing cell"""
        result = cell_manager.get_cell_model_info("nonexistent")

        assert result is None

    def test_list_cell_models_empty(self, cell_manager):
        """Test listing models when cache is empty"""
        result = cell_manager.list_cell_models()

        assert result == []

    def test_list_cell_models_with_entries(self, cell_manager):
        """Test listing models with cache entries"""
        cell_manager.cell_cache = {
            "12898855": {"model_name": "model1", "status": "created"},
            "12898856": {"model_name": "model2", "status": "created"}
        }

        result = cell_manager.list_cell_models()

        assert len(result) == 2
        assert any(item['cell_index'] == "12898855" for item in result)
        assert any(item['cell_index'] == "12898856" for item in result)

    def test_clear_cache(self, cell_manager):
        """Test clearing the cache"""
        cell_manager.cell_cache = {
            "12898855": {"model_name": "model1"},
            "12898856": {"model_name": "model2"}
        }

        cell_manager.clear_cache()

        assert cell_manager.cell_cache == {}

    def test_create_model_mlflow_disconnected(self, cell_manager):
        """Test model creation fails gracefully when MLflow disconnected"""
        cell_manager.ml_interface.is_mlflow_connected.return_value = False

        cell_manager._create_model_for_cell("12898855", {})

        # Should not add to cache when MLflow is disconnected
        assert "12898855" not in cell_manager.cell_cache

    def test_create_model_marks_instantiating_status(self, cell_manager):
        """Test that cache is marked as instantiating during creation"""
        cell_manager.ml_interface.is_mlflow_connected.return_value = True

        # Mock the model creation to capture intermediate state
        with patch.object(cell_manager, '_register_model_with_mlflow', return_value=None):
            with patch('src.performance_monitor.cell_model_manager.models', []):
                cell_manager._create_model_for_cell("12898855", {})

        # After failed creation (no models registered), status should be creation_failed
        assert cell_manager.cell_cache.get("12898855", {}).get('status') == 'creation_failed'
