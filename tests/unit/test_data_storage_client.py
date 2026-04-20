"""Unit tests for Data Storage client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.services.data_storage_client import DataStorageClient


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("src.services.data_storage_client.settings") as mock:
        mock.DATA_STORAGE_API_URL = "http://data-storage:8000"
        mock.DATA_STORAGE_EXAMPLE_ENDPOINT = "/api/v1/processed/latency/example"
        mock.DATA_STORAGE_EXCLUDED_FIELDS = "window_start_time,window_end_time,window_duration_seconds,cell_index,network,sample_count"
        mock.DATA_STORAGE_DATA_ENDPOINT = "/api/v1/processed/latency"
        mock.DATA_STORAGE_CELL_ENDPOINT = "/api/v1/cell"
        mock.ENCRYPTION_ENABLED = False
        yield mock


@pytest.fixture
def sample_example_response():
    """Sample response from data-storage example endpoint."""
    return [
        {
            "window_start_time": 1706745600,
            "window_end_time": 1706745660,
            "window_duration_seconds": 60.0,
            "cell_index": 12898855,
            "network": "5G",
            "rsrp_mean": -85.5,
            "rsrp_max": -80.0,
            "rsrp_min": -90.0,
            "rsrp_std": 2.5,
            "sinr_mean": 15.3,
            "sinr_max": 20.0,
            "sinr_min": 10.0,
            "sinr_std": 2.1,
            "rsrq_mean": -12.5,
            "rsrq_max": -10.0,
            "rsrq_min": -15.0,
            "rsrq_std": 1.2,
            "latency_mean": 25.5,
            "latency_max": 35.0,
            "latency_min": 20.0,
            "latency_std": 3.2,
            "cqi_mean": 12.5,
            "cqi_max": 15.0,
            "cqi_min": 10.0,
            "cqi_std": 1.5,
            "primary_bandwidth": 100.0,
            "ul_bandwidth": 50.0,
            "sample_count": 120,
        }
    ]


class TestDataStorageClient:
    """Tests for DataStorageClient."""

    @pytest.mark.asyncio
    async def test_get_available_fields_success(self, mock_settings, sample_example_response):
        """Test successfully fetching available fields."""
        import json as _json
        client = DataStorageClient()

        # Mock httpx response
        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.content = _json.dumps(sample_example_response).encode()
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Call method
            fields = await client.get_available_fields()

            # Verify request
            mock_client_instance.get.assert_called_once_with(
                "http://data-storage:8000/api/v1/processed/latency/example",
                timeout=10.0,
                headers={},
            )

            # Verify excluded fields are removed
            assert "window_start_time" not in fields
            assert "window_end_time" not in fields
            assert "window_duration_seconds" not in fields
            assert "cell_index" not in fields
            assert "network" not in fields
            assert "sample_count" not in fields

            # Verify metric fields are included
            assert "rsrp_mean" in fields
            assert "rsrp_max" in fields
            assert "sinr_mean" in fields
            assert "latency_mean" in fields
            assert "cqi_mean" in fields

            # Verify fields are sorted
            assert fields == sorted(fields)

    @pytest.mark.asyncio
    async def test_get_available_fields_empty_response(self, mock_settings):
        """Test handling empty response."""
        import json as _json
        client = DataStorageClient()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.content = _json.dumps([]).encode()
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            fields = await client.get_available_fields()

            assert fields == []

    @pytest.mark.asyncio
    async def test_get_available_fields_http_error(self, mock_settings):
        """Test handling HTTP errors."""
        client = DataStorageClient()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=MagicMock(),
                response=MagicMock(),
            )

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            with pytest.raises(httpx.HTTPStatusError):
                await client.get_available_fields()

    @pytest.mark.asyncio
    async def test_excluded_fields_parsing(self, mock_settings):
        """Test that excluded fields are correctly parsed from config."""
        client = DataStorageClient()

        expected_excluded = {
            "window_start_time",
            "window_end_time",
            "window_duration_seconds",
            "cell_index",
            "network",
            "sample_count",
        }

        assert client.excluded_fields == expected_excluded

    @pytest.mark.asyncio
    async def test_excluded_fields_with_whitespace(self):
        """Test that excluded fields handle extra whitespace."""
        with patch("src.services.data_storage_client.settings") as mock:
            mock.DATA_STORAGE_API_URL = "http://data-storage:8000"
            mock.DATA_STORAGE_EXAMPLE_ENDPOINT = "/api/v1/processed/latency/example"
            mock.DATA_STORAGE_EXCLUDED_FIELDS = " field1 , field2 ,  field3  "
            mock.ENCRYPTION_ENABLED = False

            client = DataStorageClient()

            assert client.excluded_fields == {"field1", "field2", "field3"}

    @pytest.mark.asyncio
    async def test_excluded_fields_empty_string(self):
        """Test handling empty excluded fields config."""
        with patch("src.services.data_storage_client.settings") as mock:
            mock.DATA_STORAGE_API_URL = "http://data-storage:8000"
            mock.DATA_STORAGE_EXAMPLE_ENDPOINT = "/api/v1/processed/latency/example"
            mock.DATA_STORAGE_EXCLUDED_FIELDS = ""
            mock.ENCRYPTION_ENABLED = False

            client = DataStorageClient()

            assert client.excluded_fields == set()

    @pytest.mark.asyncio
    async def test_validate_fields_all_valid(self, mock_settings, sample_example_response):
        """Test validating fields when all are valid."""
        import json as _json
        client = DataStorageClient()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.content = _json.dumps(sample_example_response).encode()
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Validate valid fields
            is_valid, invalid = await client.validate_fields(["rsrp_mean", "latency_mean"])

            assert is_valid is True
            assert invalid == []

    @pytest.mark.asyncio
    async def test_validate_fields_some_invalid(self, mock_settings, sample_example_response):
        """Test validating fields when some are invalid."""
        import json as _json
        client = DataStorageClient()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.content = _json.dumps(sample_example_response).encode()
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Validate with invalid fields
            is_valid, invalid = await client.validate_fields(
                ["rsrp_mean", "invalid_field", "another_invalid"]
            )

            assert is_valid is False
            assert set(invalid) == {"invalid_field", "another_invalid"}

    @pytest.mark.asyncio
    async def test_validate_fields_all_invalid(self, mock_settings, sample_example_response):
        """Test validating fields when all are invalid."""
        import json as _json
        client = DataStorageClient()

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.content = _json.dumps(sample_example_response).encode()
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            # Validate all invalid fields
            is_valid, invalid = await client.validate_fields(["foo", "bar", "baz"])

            assert is_valid is False
            assert set(invalid) == {"foo", "bar", "baz"}

    @pytest.mark.asyncio
    async def test_get_known_cells_success(self, mock_settings):
        """Test successfully fetching known cells."""
        import json as _json
        client = DataStorageClient()
        mock_cells = [0, 1, 2, 5, 7]

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.content = _json.dumps(mock_cells).encode()
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            cells = await client.get_known_cells()

            assert cells == [0, 1, 2, 5, 7]
            mock_client_instance.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_cell_data_success(self, mock_settings):
        """Test successfully fetching cell data."""
        import json as _json
        client = DataStorageClient()
        mock_data = [
            {
                "window_start_time": 1000,
                "window_end_time": 1060,
                "cell_index": 0,
                "rsrp_mean": -85.0,
                "latency_mean": 20.0,
            }
        ]

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_response = MagicMock()
            mock_response.content = _json.dumps(mock_data).encode()
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_async_client.return_value.__aenter__.return_value = mock_client_instance

            data = await client.fetch_cell_data(
                cell_index=0,
                start_timestamp=1000,
                end_timestamp=2000,
                window_duration_seconds=60,
            )

            assert len(data) == 1
            assert data[0]["rsrp_mean"] == -85.0
