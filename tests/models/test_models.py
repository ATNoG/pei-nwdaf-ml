import pytest
import numpy as np
from src.models.xgboost import XGBoost
from src.models.randomforest import RandomForest


def generate_mock_data(n_samples=100):
    """
    Generates mock temporal/network/cell info data
    Returns:
        X (np.ndarray), y (np.ndarray)
    """
    np.random.seed(42)

    window_duration_seconds = np.random.randint(60, 3600, size=n_samples)

    cell_index = np.random.randint(0, 10, size=n_samples)
    direction = np.random.choice([0, 1], size=n_samples)

    rsrp = np.random.uniform(-120, -60, size=n_samples)
    sinr = np.random.uniform(0, 30, size=n_samples)
    rsrq = np.random.uniform(-20, -3, size=n_samples)
    datarate = np.random.uniform(1, 100, size=n_samples)
    latency = np.random.uniform(10, 200, size=n_samples)
    cqi = np.random.uniform(0, 15, size=n_samples)

    primary_bandwidth = np.random.choice([5, 10, 20], size=n_samples)
    ul_bandwidth = np.random.choice([5, 10, 20], size=n_samples)

    amount_samples = np.random.randint(1, 100, size=n_samples)

    X = np.column_stack([
        window_duration_seconds,
        cell_index,
        direction,
        rsrp, sinr, rsrq, datarate, cqi,
        primary_bandwidth, ul_bandwidth,
        amount_samples
    ])

    y = latency

    return X, y

# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def mock_data():
    X, y = generate_mock_data(50)
    return X, y

# ----------------------------
# Tests
# ----------------------------
def test_xgboost_train_infer_serialize(mock_data):
    X, y = mock_data
    model = XGBoost()
    loss = model.train(min_loss=0.01, max_epochs=10, X=X, y=y)
    assert isinstance(loss, float)
    preds = model.infer(data=X[:5])
    assert preds.shape[0] == 5

    serialized = model.serialize()
    loaded_model = XGBoost.deserialize(serialized)

    # test if the same model was loaded
    preds_loaded = loaded_model.infer(data=X[:5])
    np.testing.assert_allclose(preds, preds_loaded, rtol=1e-5)

def test_randomforest_train_infer_serialize(mock_data):
    X, y = mock_data
    model = RandomForest()
    loss = model.train(min_loss=0.01, max_epochs=10, X=X, y=y)
    assert isinstance(loss, float)
    preds = model.infer(data=X[:5])
    assert preds.shape[0] == 5

    serialized = model.serialize()
    loaded_model = RandomForest.deserialize(serialized)

    # test if the same model was loaded
    preds_loaded = loaded_model.infer(data=X[:5])
    np.testing.assert_allclose(preds, preds_loaded, rtol=1e-5)
