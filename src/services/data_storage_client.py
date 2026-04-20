"""Client for interacting with the Data Storage API."""

import json
import logging

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)


class DataStorageClient:
    """Client for fetching data from the Data Storage service."""

    def __init__(self):
        self.base_url = settings.DATA_STORAGE_API_URL
        self.example_endpoint = settings.DATA_STORAGE_EXAMPLE_ENDPOINT
        self.data_endpoint = settings.DATA_STORAGE_DATA_ENDPOINT
        self.cell_endpoint = settings.DATA_STORAGE_CELL_ENDPOINT
        self.excluded_fields = set(
            f.strip() for f in settings.DATA_STORAGE_EXCLUDED_FIELDS.split(",") if f.strip()
        )
        self._encryptor_client = None
        self._handshake_done = False
        self._session_token: str | None = None
        if settings.ENCRYPTION_ENABLED:
            from encryptor.core.secure_channel_client import EncryptorClient
            self._encryptor_client = EncryptorClient()

    def _ensure_handshake(self) -> None:
        """Perform the DH handshake with data-storage if not yet done."""
        if self._encryptor_client is None or self._handshake_done:
            return
        try:
            self._encryptor_client.handshake(self.base_url)
            self._handshake_done = True
            self._session_token = self._encryptor_client.session_token
            logger.info(
                "Encryption handshake with data-storage completed (session=%s).",
                self._session_token,
            )
        except Exception as exc:
            logger.warning("Encryption handshake failed, continuing without encryption: %s", exc)
            self._encryptor_client = None

    @property
    def _auth_headers(self) -> dict[str, str]:
        """Headers to attach to every data-storage request."""
        if self._session_token:
            return {"X-Session-Token": self._session_token}
        return {}

    def _decrypt_response(self, response: httpx.Response) -> bytes:
        """Decrypt the response body if the server marked it as encrypted."""
        if (
            self._encryptor_client is not None
            and self._handshake_done
            and response.headers.get("X-Encrypted") == "true"
        ):
            return self._encryptor_client.decrypt(response.content)
        return response.content

    async def get_available_fields(self, component_id: str | None = None) -> list[str]:
        """
        Get available data field names from the data-storage example endpoint.

        Returns a list of field names, excluding metadata fields defined in
        DATA_STORAGE_EXCLUDED_FIELDS.

        Args:
            component_id: Optional component ID to pass as X-Component-ID header
                for policy enforcement.
        """
        self._ensure_handshake()
        url = f"{self.base_url}{self.example_endpoint}"
        headers = {}
        if component_id:
            headers["X-Component-ID"] = component_id

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0, headers=self._auth_headers)
            response.raise_for_status()

            raw = self._decrypt_response(response)
            data = json.loads(raw)

            # data is a list with at least one example record
            if not data or not isinstance(data, list):
                return []

            sample = data[0]

            # Extract field names, excluding metadata fields
            fields = [
                field_name
                for field_name in sample.keys()
                if field_name not in self.excluded_fields
            ]

            return sorted(fields)

    async def validate_fields(self, fields: list[str]) -> tuple[bool, list[str]]:
        """
        Validate that all provided fields exist in available fields.

        Args:
            fields: List of field names to validate

        Returns:
            Tuple of (all_valid: bool, invalid_fields: list[str])
        """
        available_fields = await self.get_available_fields()
        available_set = set(available_fields)

        invalid_fields = [field for field in fields if field not in available_set]

        return (len(invalid_fields) == 0, invalid_fields)

    async def get_known_cells(self, component_id: str | None = None) -> list[int]:
        """
        Get list of all known cell indexes from data-storage API.

        Args:
            component_id: Optional component ID to pass as X-Component-ID header
                for policy enforcement.

        Returns:
            List of cell indexes (integers)
        """
        self._ensure_handshake()
        url = f"{self.base_url}{self.cell_endpoint}"
        headers = {}
        if component_id:
            headers["X-Component-ID"] = component_id

        async with httpx.AsyncClient() as client:
<<<<<<< HEAD
            response = await client.get(url, timeout=30.0, headers=headers)
=======
            response = await client.get(url, timeout=30.0, headers=self._auth_headers)
>>>>>>> dev
            response.raise_for_status()
            return json.loads(self._decrypt_response(response))

    async def probe_data_timestamp(
        self, cells: list[int], window_duration_seconds: int, component_id: str | None = None
    ) -> int | None:
        """
        Discover any available data by issuing a single-record request over the
        full time range (epoch 0 to far future) for each of the first few cells.

        Args:
            cells: List of cell indexes to probe.
            window_duration_seconds: Window duration for aggregation.
            component_id: Optional component ID to pass as X-Component-ID header
                for policy enforcement.

        Returns the window_start_time of the first record found, or None if the
        data storage has no data at all.  Used to anchor evaluation windows when
        neither the current-time window nor the training-era window has data.
        """
        import time
        from datetime import datetime, timezone

        self._ensure_handshake()
        far_future = int(time.time()) + 10 * 365 * 24 * 3600
        url = f"{self.base_url}{self.data_endpoint}"

        async with httpx.AsyncClient() as client:
            for cell in cells[:5]:
                try:
                    params = {
                        "cell_index": cell,
                        "start_time": 0,
                        "end_time": far_future,
                        "window_duration_seconds": window_duration_seconds,
                        "offset": 0,
                        "limit": 1,
                    }
<<<<<<< HEAD
                    headers = {}
                    if component_id:
                        headers["X-Component-ID"] = component_id
                    response = await client.get(url, params=params, timeout=30.0, headers=headers)
=======
                    response = await client.get(
                        url, params=params, timeout=30.0, headers=self._auth_headers
                    )
>>>>>>> dev
                    if response.status_code == 200:
                        data = json.loads(self._decrypt_response(response))
                        if data and isinstance(data, list):
                            raw = data[0].get("window_start_time", 0)
                            if isinstance(raw, str):
                                ts = int(datetime.fromisoformat(raw).replace(tzinfo=timezone.utc).timestamp())
                            else:
                                ts = int(raw)
                            if ts > 0:
                                return ts
                except Exception:
                    continue
        return None

    async def fetch_cell_data(
        self,
        cell_index: int,
        start_timestamp: int,
        end_timestamp: int,
        window_duration_seconds: int,
        ip_src: str | None = None,
        component_id: str | None = None,
    ) -> list[dict]:
        """
        Fetch processed latency data for a specific cell with pagination.

        Automatically handles pagination when dataset is larger than 1000 records.

        Args:
            cell_index: Cell identifier
            start_timestamp: Start time (Unix epoch seconds)
            end_timestamp: End time (Unix epoch seconds)
            window_duration_seconds: Window duration for aggregation
            ip_src: Optional IP source filter. Use "*" to include ip_src in
                    the response grouped per IP. Default None = existing behavior.
            component_id: Optional component ID to pass as X-Component-ID header
                    for policy enforcement.

        Returns:
            List of data windows with metrics (all pages combined)
        """
        self._ensure_handshake()
        url = f"{self.base_url}{self.data_endpoint}"
        all_data = []
        offset = 0
        limit = 1000

        async with httpx.AsyncClient() as client:
            while True:
                params: dict[str, int | str] = {
                    "cell_index": cell_index,
                    "start_time": start_timestamp,
                    "end_time": end_timestamp,
                    "window_duration_seconds": window_duration_seconds,
                    "offset": offset,
                    "limit": limit,
                }
                if ip_src is not None:
                    params["ip_src"] = ip_src

<<<<<<< HEAD
                headers = {}
                if component_id:
                    headers["X-Component-ID"] = component_id

                response = await client.get(url, params=params, timeout=60.0, headers=headers)
=======
                response = await client.get(
                    url, params=params, timeout=60.0, headers=self._auth_headers
                )
>>>>>>> dev
                response.raise_for_status()
                batch = json.loads(self._decrypt_response(response))

                if not batch:
                    # No more data
                    break

                all_data.extend(batch)

                # If we got fewer than limit records, we've reached the end
                if len(batch) < limit:
                    break

                # Move to next page
                offset += limit

        return all_data
