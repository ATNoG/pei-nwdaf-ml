"""Unit tests for the Autoencoder model."""

import numpy as np
import pytest

from src.models.autoencoder import Autoencoder, AutoencoderNetwork


class TestAutoencoderNetwork:
    """Tests for AutoencoderNetwork (nn.Module)."""

    def test_forward_shape(self):
        """Test that output shape matches input shape."""
        import torch

        net = AutoencoderNetwork(input_size=10, hidden_size=16)
        x = torch.randn(5, 10)
        out = net(x)
        assert out.shape == (5, 10)

    def test_bottleneck_size(self):
        """Test that bottleneck is hidden_size // 2 (min 4)."""
        net = AutoencoderNetwork(input_size=10, hidden_size=16)
        # bottleneck = 16 // 2 = 8
        assert net.encoder[2].in_features == 16
        assert net.encoder[2].out_features == 8

    def test_bottleneck_minimum(self):
        """Test that bottleneck doesn't go below 4."""
        net = AutoencoderNetwork(input_size=10, hidden_size=4)
        # bottleneck = max(4, 4 // 2) = max(4, 2) = 4
        assert net.encoder[2].out_features == 4


class TestAutoencoder:
    """Tests for Autoencoder wrapper."""

    @pytest.fixture
    def sample_data(self):
        """Synthetic training data: 100 samples, 5 features."""
        np.random.seed(42)
        return np.random.randn(100, 5).astype(np.float32)

    def test_train_returns_loss(self, sample_data):
        """Test that train returns a finite loss value."""
        ae = Autoencoder(input_size=5, hidden_size=16)
        loss = ae.train(sample_data, max_epochs=10)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_score_shape(self, sample_data):
        """Test that score returns per-sample errors with correct shape."""
        ae = Autoencoder(input_size=5, hidden_size=16)
        ae.train(sample_data, max_epochs=5)
        errors = ae.score(sample_data)
        assert errors.shape == (100,)

    def test_score_non_negative(self, sample_data):
        """Test that reconstruction errors are non-negative."""
        ae = Autoencoder(input_size=5, hidden_size=16)
        ae.train(sample_data, max_epochs=5)
        errors = ae.score(sample_data)
        assert (errors >= 0).all()

    def test_training_reduces_loss(self, sample_data):
        """Test that training reduces reconstruction error."""
        ae = Autoencoder(input_size=5, hidden_size=16)

        # Score before training
        errors_before = ae.score(sample_data).mean()

        # Train
        ae.train(sample_data, max_epochs=50)

        # Score after training
        errors_after = ae.score(sample_data).mean()

        assert errors_after < errors_before

    def test_outliers_score_higher(self):
        """Test that outliers have higher reconstruction error than normal data."""
        np.random.seed(42)
        # Normal data centered around 0
        normal = np.random.randn(200, 3).astype(np.float32)
        # Outliers far from center
        outliers = (np.random.randn(10, 3) + 10).astype(np.float32)

        ae = Autoencoder(input_size=3, hidden_size=16)
        ae.train(normal, max_epochs=50)

        normal_errors = ae.score(normal).mean()
        outlier_errors = ae.score(outliers).mean()

        assert outlier_errors > normal_errors

    def test_nan_handling(self):
        """Test that NaN values in input are handled (replaced with 0)."""
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
        ae = Autoencoder(input_size=3, hidden_size=8)
        loss = ae.train(data, max_epochs=5)
        assert np.isfinite(loss)

        errors = ae.score(data)
        assert errors.shape == (2,)
        assert np.all(np.isfinite(errors))

    def test_status_callback_called(self, sample_data):
        """Test that status_callback is invoked during training."""
        ae = Autoencoder(input_size=5, hidden_size=8)
        callback_epochs = []

        def callback(epoch, max_epochs, loss):
            callback_epochs.append(epoch)

        ae.train(sample_data, max_epochs=10, status_callback=callback)
        assert len(callback_epochs) == 10

    def test_status_callback_cancellation(self, sample_data):
        """Test that InterruptedError from callback stops training."""
        ae = Autoencoder(input_size=5, hidden_size=8)

        def callback(epoch, max_epochs, loss):
            if epoch >= 3:
                raise InterruptedError("cancelled")

        with pytest.raises(InterruptedError):
            ae.train(sample_data, max_epochs=100, status_callback=callback)

    def test_single_sample(self):
        """Test training and scoring with a single sample."""
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        ae = Autoencoder(input_size=3, hidden_size=8)
        loss = ae.train(data, max_epochs=5)
        assert np.isfinite(loss)

        errors = ae.score(data)
        assert errors.shape == (1,)
