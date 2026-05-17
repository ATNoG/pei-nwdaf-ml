"""Client for interacting with the Data Storage API."""

import json
import logging
import os
import time

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)

_KC_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080/auth")
_KC_REALM = os.getenv("KEYCLOAK_REALM", "aion")
_KC_CLIENT = os.getenv("KEYCLOAK_CLIENT_ID", "aion-services")
_KC_USER = os.getenv("ML_SERVICE_KC_USER", "ml-service")
_KC_PASS = os.getenv("ML_SERVICE_KC_PASSWORD", "ml-service")
_TOKEN_URL = f"{_KC_URL}/realms/{_KC_REALM}/protocol/openid-connect/token"

_cached_token: str | None = None
_token_expires_at: float = 0.0

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return _http_client


async def _get_service_token() -> str | None:
    global _cached_token, _token_expires_at
    if os.getenv("DEV_MODE", "").lower() == "true":
        return None
    if _cached_token and time.time() < _token_expires_at - 30:
        return _cached_token
    try:
        client = _get_http_client()
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type": "password",
                "client_id": _KC_CLIENT,
                "username": _KC_USER,
                "password": _KC_PASS,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        _cached_token = data["access_token"]
        _token_expires_at = time.time() + data.get("expires_in", 300)
        return _cached_token
    except Exception:
        logger.warning("Failed to fetch service token from Keycloak", exc_info=True)
        return None


class DataStorageClient:
    """Client for fetching data from the Data Storage service."""

    def __init__(self):
        self.base_url = settings.DATA_STORAGE_API_URL
        self.example_endpoint = settings.DATA_STORAGE_EXAMPLE_ENDPOINT
        self.data_endpoint = settings.DATA_STORAGE_DATA_ENDPOINT
        self.cell_endpoint = settings.DATA_STORAGE_CELL_ENDPOINT
        self.excluded_fields = set(
            f.strip()
            for f in settings.DATA_STORAGE_EXCLUDED_FIELDS.split(",")
            if f.strip()
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
            logger.warning(
                "Encryption handshake failed, continuing without encryption: %s", exc
            )
            self._encryptor_client = None

    async def _get_auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._session_token:
            headers["X-Session-Token"] = self._session_token
        token = await _get_service_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

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
        Get available data field names from the data-storage fields endpoint.

        Returns a list of field names, excluding metadata fields defined in
        DATA_STORAGE_EXCLUDED_FIELDS.
        """
        self._ensure_handshake()
        url = f"{self.base_url}{settings.DATA_STORAGE_FIELDS_ENDPOINT}"
        headers = {}
        if component_id:
            headers["X-Component-ID"] = component_id

        client = _get_http_client()
        auth_headers = await self._get_auth_headers()
        response = await client.get(
            url, timeout=10.0, headers={**auth_headers, **headers}
        )
        response.raise_for_status()

        raw = self._decrypt_response(response)
        data = json.loads(raw)

        # data is {"field_name": ["EVENT_TYPE"], ...}
        if not data or not isinstance(data, dict):
            return []

        return sorted(
            field_name
            for field_name in data.keys()
            if field_name not in self.excluded_fields
        )

    async def validate_fields(self, fields: list[str]) -> tuple[bool, list[str]]:
        """
        Validate that all provided fields exist in available fields.

        Args:
            fields: List of field names to validate

        Returns:
            Tuple of (all_valid: bool, invalid_fields: list[str])
        """
        field_event_map = await self.get_fields_with_events()
        available_set = set(field_event_map.keys())

        invalid_fields = [field for field in fields if field not in available_set]

        return (len(invalid_fields) == 0, invalid_fields)

    async def get_fields_with_events(self) -> dict[str, list[str]]:
        """Get field names with their associated event types from the data-storage fields endpoint."""
        self._ensure_handshake()
        url = f"{self.base_url}{settings.DATA_STORAGE_FIELDS_ENDPOINT}"
        client = _get_http_client()
        response = await client.get(
            url, timeout=10.0, headers=await self._get_auth_headers()
        )
        response.raise_for_status()
        raw = self._decrypt_response(response)
        return json.loads(raw)

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

        client = _get_http_client()
        response = await client.get(
            url, timeout=30.0, headers={**await self._get_auth_headers(), **headers}
        )
        response.raise_for_status()
        return json.loads(self._decrypt_response(response))

    async def probe_data_timestamp(
        self, window_duration_seconds: int, event_type: str | None = None
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

        try:
            data = await self.fetch_data(
                start_timestamp=0,
                end_timestamp=far_future,
                window_duration_seconds=window_duration_seconds,
                event=event_type,
            )
            if data:
                raw = data[0].get("window_start_time", 0)
                if isinstance(raw, str):
                    ts = int(
                        datetime.fromisoformat(raw)
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                    )
                else:
                    ts = int(raw)
                if ts > 0:
                    return ts
        except Exception:
            pass
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

        client = _get_http_client()
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

            headers = {**await self._get_auth_headers()}
            if component_id:
                headers["X-Component-ID"] = component_id

            response = await client.get(
                url, params=params, timeout=60.0, headers=headers
            )
            response.raise_for_status()
            batch = json.loads(self._decrypt_response(response))

            if not batch:
                break

            all_data.extend(batch)

            if len(batch) < limit:
                break

            offset += limit

        return all_data

    async def fetch_data(
        self,
        start_timestamp: int,
        end_timestamp: int,
        window_duration_seconds: int,
        snssai_sst: str | None = None,
        dnn: str | None = None,
        snssai_sd: str | None = None,
        event: str | None = None,
    ) -> list[dict]:
        """Fetch processed data with optional tag filters (no cell_index)."""
        self._ensure_handshake()
        url = f"{self.base_url}{self.data_endpoint}"
        all_data = []
        offset = 0
        limit = 1000

        client = _get_http_client()
        while True:
            params: dict[str, int | str] = {
                "start_time": start_timestamp,
                "end_time": end_timestamp,
                "window_duration_seconds": window_duration_seconds,
                "offset": offset,
                "limit": limit,
            }
            if snssai_sst is not None:
                params["snssai_sst"] = snssai_sst
            if dnn is not None:
                params["dnn"] = dnn
            if snssai_sd is not None:
                params["snssai_sd"] = snssai_sd
            if event is not None:
                params["event"] = event

            response = await client.get(
                url,
                params=params,
                timeout=60.0,
                headers=await self._get_auth_headers(),
            )
            response.raise_for_status()
            batch = json.loads(self._decrypt_response(response))

            if not batch:
                break
            all_data.extend(batch)
            if len(batch) < limit:
                break
            offset += limit

        return all_data


async def derive_event_type(
    fields: list[str],
    requested_event: str | None,
    data_storage_client: DataStorageClient,
) -> str:
    """Derive and validate the event type for a set of model fields.

    Fetches field→event mappings from data-storage, computes the intersection
    of events across all fields, and returns the resolved event type.
    """
    from fastapi import HTTPException

    field_event_map = await data_storage_client.get_fields_with_events()
    event_sets = [set(field_event_map.get(f, [])) for f in fields]
    intersection = event_sets[0].intersection(*event_sets[1:]) if event_sets else set()

    if not intersection:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Fields do not share a common event",
                "field_events": {f: field_event_map.get(f, []) for f in fields},
            },
        )

    if requested_event is not None and requested_event not in intersection:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Event '{requested_event}' is not valid for the provided fields",
                "valid_events": sorted(intersection),
            },
        )

    return requested_event if requested_event is not None else sorted(intersection)[0]
