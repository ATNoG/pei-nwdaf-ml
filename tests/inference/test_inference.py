import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from src.inference.inference import InferenceMaker
from src.models.xgboost import XGBoost


class TestInferenceMaker:
    """Tests for InferenceMaker class"""

    @pytest.fixture
    def mock_ml_interface(self):
        """Create a mock ML interface"""
        mock_interface = Mock()
        mock_interface.update_component_status = Mock()
        mock_interface.get_best_model = Mock()
        mock_interface.load_model = Mock()
        mock_interface.get_model_by_name = Mock()
        mock_interface.log_inference_metrics = Mock()
        return mock_interface

    @pytest.fixture
    def mock_model(self):
        """Create a mock trained model"""
        model = XGBoost()
        # Create simple mock data for training
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        model.train(min_loss=0.01, max_epochs=5, X=X, y=y)
        return model

    def test_inference_maker_init_auto_mode(self, mock_ml_interface):
        """Test InferenceMaker initialization in auto mode"""
        inference_maker = InferenceMaker(mock_ml_interface)

        assert inference_maker._auto_mode is True
        assert inference_maker._current_model is None
        assert inference_maker._current_model_id is None
        assert inference_maker._failed_retrieves == 0
        mock_ml_interface.update_component_status.assert_called()

    def test_inference_maker_init_manual_mode(self, mock_ml_interface, mock_model):
        """Test InferenceMaker initialization with specific model"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/TestModel/Production"
        )

        assert inference_maker._auto_mode is False
        assert inference_maker._current_model_id == "models:/TestModel/Production"
        mock_ml_interface.load_model.assert_called_with("models:/TestModel/Production")

    def test_inference_maker_init_invalid_model_fallback(self, mock_ml_interface):
        """Test InferenceMaker falls back to auto mode with invalid model"""
        mock_ml_interface.load_model.return_value = None

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/InvalidModel/Production"
        )

        assert inference_maker._auto_mode is True

    def test_toggle_auto_select_enable(self, mock_ml_interface):
        """Test toggling auto-select mode to enabled"""
        inference_maker = InferenceMaker(mock_ml_interface, selected_model_id=None)
        inference_maker._auto_mode = False

        inference_maker.toggle_auto_select(True)

        assert inference_maker._auto_mode is True
        mock_ml_interface.update_component_status.assert_called()

    def test_toggle_auto_select_disable(self, mock_ml_interface):
        """Test toggling auto-select mode to disabled"""
        inference_maker = InferenceMaker(mock_ml_interface)

        inference_maker.toggle_auto_select(False)

        assert inference_maker._auto_mode is False

    def test_toggle_auto_select_flip(self, mock_ml_interface):
        """Test toggling auto-select mode without explicit value"""
        inference_maker = InferenceMaker(mock_ml_interface)
        initial_mode = inference_maker._auto_mode

        inference_maker.toggle_auto_select()

        assert inference_maker._auto_mode != initial_mode

    def test_fetch_best_model_success(self, mock_ml_interface):
        """Test successfully fetching best model"""
        mock_ml_interface.get_best_model.return_value = {
            'name': 'BestModel',
            'version': '1',
            'model_uri': 'models:/BestModel/Production',
            'metric_value': 0.95
        }

        inference_maker = InferenceMaker(mock_ml_interface)
        model_uri = inference_maker._fetch_best_model()

        assert model_uri == 'models:/BestModel/Production'
        mock_ml_interface.get_best_model.assert_called_with(metric='accuracy')

    def test_fetch_best_model_not_found(self, mock_ml_interface):
        """Test fetching best model when none exists"""
        mock_ml_interface.get_best_model.return_value = None

        inference_maker = InferenceMaker(mock_ml_interface)
        model_uri = inference_maker._fetch_best_model()

        # When no model found (None returned), _failed_retrieves is NOT incremented
        # It only increments on exception
        assert model_uri is None
        assert inference_maker._failed_retrieves == 0

    def test_fetch_best_model_exception(self, mock_ml_interface):
        """Test fetching best model with exception"""
        mock_ml_interface.get_best_model.side_effect = Exception("Connection error")

        inference_maker = InferenceMaker(mock_ml_interface)
        model_uri = inference_maker._fetch_best_model()

        assert model_uri is None
        assert inference_maker._failed_retrieves == 1

    def test_load_model_auto_mode_success(self, mock_ml_interface, mock_model):
        """Test loading model in auto mode successfully"""
        mock_ml_interface.get_best_model.return_value = {
            'model_uri': 'models:/BestModel/Production',
            'name': 'BestModel',
            'version': '1',
            'metric_value': 0.95
        }
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(mock_ml_interface)
        loaded_model = inference_maker._load_model()

        assert loaded_model is not None
        assert inference_maker._current_model == mock_model
        assert inference_maker._failed_retrieves == 0
        mock_ml_interface.load_model.assert_called_with('models:/BestModel/Production')

    def test_load_model_manual_mode_success(self, mock_ml_interface, mock_model):
        """Test loading model in manual mode successfully"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/MyModel/1"
        )
        inference_maker._auto_mode = False
        inference_maker._current_model = None  # Clear cache

        loaded_model = inference_maker._load_model()

        assert loaded_model is not None
        assert inference_maker._current_model == mock_model

    def test_load_model_cached(self, mock_ml_interface, mock_model):
        """Test loading model uses cache when available"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/MyModel/1"
        )

        # First load
        first_load = inference_maker._load_model()
        # Second load (should use cache)
        second_load = inference_maker._load_model()

        assert first_load == second_load
        # Should only call load_model once during init
        assert mock_ml_interface.load_model.call_count <= 2

    def test_load_model_auto_mode_no_best_model(self, mock_ml_interface):
        """Test loading model in auto mode when no best model exists"""
        mock_ml_interface.get_best_model.return_value = None

        inference_maker = InferenceMaker(mock_ml_interface)
        loaded_model = inference_maker._load_model()

        assert loaded_model is None
        assert inference_maker._failed_retrieves > 0

    def test_set_model_success(self, mock_ml_interface, mock_model):
        """Test successfully setting a model by ID"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(mock_ml_interface)
        success = inference_maker._set_model("models:/TestModel/Production")

        assert success is True
        assert inference_maker._current_model_id == "models:/TestModel/Production"
        assert inference_maker._auto_mode is False

    def test_set_model_failure(self, mock_ml_interface):
        """Test setting a model that doesn't exist"""
        mock_ml_interface.load_model.return_value = None

        inference_maker = InferenceMaker(mock_ml_interface)
        success = inference_maker._set_model("models:/InvalidModel/Production")

        assert success is False

    def test_set_model_exception(self, mock_ml_interface):
        """Test setting a model with exception"""
        mock_ml_interface.load_model.side_effect = Exception("Load error")

        inference_maker = InferenceMaker(mock_ml_interface)
        success = inference_maker._set_model("models:/TestModel/Production")

        assert success is False

    def test_set_model_by_name_with_version(self, mock_ml_interface, mock_model):
        """Test setting model by name and version"""
        mock_ml_interface.get_model_by_name.return_value = mock_model

        inference_maker = InferenceMaker(mock_ml_interface)
        success = inference_maker.set_model_by_name("MyModel", version="2")

        assert success is True
        assert inference_maker._current_model == mock_model
        assert "MyModel" in inference_maker._current_model_id
        mock_ml_interface.get_model_by_name.assert_called_with(
            "MyModel", version="2", stage="Production"
        )

    def test_set_model_by_name_with_stage(self, mock_ml_interface, mock_model):
        """Test setting model by name and stage"""
        mock_ml_interface.get_model_by_name.return_value = mock_model

        inference_maker = InferenceMaker(mock_ml_interface)
        success = inference_maker.set_model_by_name("MyModel", stage="Staging")

        assert success is True
        assert "Staging" in inference_maker._current_model_id

    def test_set_model_by_name_failure(self, mock_ml_interface):
        """Test setting model by name when model doesn't exist"""
        mock_ml_interface.get_model_by_name.return_value = None

        inference_maker = InferenceMaker(mock_ml_interface)
        success = inference_maker.set_model_by_name("InvalidModel")

        assert success is False

    def test_infer_success(self, mock_ml_interface, mock_model):
        """Test successful inference"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/MyModel/1"
        )

        test_data = np.random.rand(5, 5)
        result = inference_maker.infer(data=test_data)

        assert result is not None
        assert len(result) == 5
        mock_ml_interface.log_inference_metrics.assert_called()

    def test_infer_model_load_failure(self, mock_ml_interface):
        """Test inference when model fails to load"""
        mock_ml_interface.get_best_model.return_value = None

        inference_maker = InferenceMaker(mock_ml_interface)
        result = inference_maker.infer(data=np.random.rand(5, 5))

        assert result is None

    def test_infer_exception(self, mock_ml_interface, mock_model):
        """Test inference with exception during prediction"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/MyModel/1"
        )

        # Pass invalid data to trigger exception
        result = inference_maker.infer(data="invalid_data")

        assert result is None

    def test_get_current_model_info(self, mock_ml_interface, mock_model):
        """Test getting current model information"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/MyModel/1"
        )

        info = inference_maker.get_current_model_info()

        assert 'model_id' in info
        assert 'auto_mode' in info
        assert 'model_loaded' in info
        assert 'failed_retrieves' in info
        assert info['model_id'] == "models:/MyModel/1"
        assert info['auto_mode'] is False
        assert info['model_loaded'] is True

    def test_clear_cache(self, mock_ml_interface, mock_model):
        """Test clearing model cache"""
        mock_ml_interface.load_model.return_value = mock_model

        inference_maker = InferenceMaker(
            mock_ml_interface,
            selected_model_id="models:/MyModel/1"
        )

        assert inference_maker._current_model is not None

        inference_maker.clear_cache()

        assert inference_maker._current_model is None
        assert inference_maker._current_model_id == "models:/MyModel/1"
        mock_ml_interface.update_component_status.assert_called()
