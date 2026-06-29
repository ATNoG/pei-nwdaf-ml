"""Client for interacting with the Data Storage API."""

import asyncio
import json
import logging
import os
import time
from typing import Callable

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)

_KC_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080/auth")
_KC_REALM = os.getenv("KEYCLOAK_REALM", "aion")
_KC_CLIENT = os.getenv("KEYCLOAK_CLIENT_ID", "aion-services")
_KC_USER = os.getenv("ML_SERVICE_KC_USER", "ml-service")
_KC_PASS = os.getenv("ML_SERVICE_KC_PASSWORD", "ml-service")
_TOKEN_URL = f"{_KC_URL}/realms/{_KC_REALM}/protocol/openid-connect/token"
_ENCRYPTION_ENABLED = os.getenv("ENCRYPTION_ENABLED", "").lower() == "true"

_cached_token: str | None = None
_token_expires_at: float = 0.0

# Per-event-loop AsyncClient cache. Sharing one httpx client across loops
# triggers "Event loop is closed" / "bound to a different event loop" because
# httpx internals (asyncio.Lock, transport pools) are pinned to the loop where
# the client was instantiated. Training jobs run via asyncio.run() inside
# thread pool workers, creating short-lived loops separate from the FastAPI
# main loop, so a process-wide singleton breaks the moment a worker loop
# closes and the next call lands back on a different loop.
_http_clients: dict[int, httpx.AsyncClient] = {}


def _get_http_client() -> httpx.AsyncClient:
    loop = asyncio.get_event_loop()
    key = id(loop)
    client = _http_clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        _http_clients[key] = client
    return client


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

    async def _get_auth_headers(self) -> dict[str, str]:
        token = await _get_service_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    async def get_available_fields(self, component_id: str | None = None) -> list[str]:
        """
        Get available data field names from the data-storage fields endpoint.

        Returns a list of field names, excluding metadata fields defined in
        DATA_STORAGE_EXCLUDED_FIELDS.
        """
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

        data = response.json()

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
        url = f"{self.base_url}{settings.DATA_STORAGE_FIELDS_ENDPOINT}"
        client = _get_http_client()
        response = await client.get(
            url, timeout=10.0, headers=await self._get_auth_headers()
        )
        response.raise_for_status()
        return response.json()

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

    async def fetch_data(
        self,
        start_timestamp: int,
        end_timestamp: int,
        window_duration_seconds: int,
        snssai_sst: str | None = None,
        dnn: str | None = None,
        snssai_sd: str | None = None,
        event: str | None = None,
        public_key: bytes | None = None,
    ) -> list[dict]:
        """Fetch processed data with optional tag filters (no cell_index)."""
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

            headers = await self._get_auth_headers()
            if _ENCRYPTION_ENABLED and public_key:
                headers["X-Public-Key"] = public_key.hex()

            response = await client.get(
                url,
                params=params,
                timeout=60.0,
                headers=headers,
            )
            response.raise_for_status()

            if response.headers.get("content-type", "").startswith(
                "application/octet-stream"
            ):
                all_data.append(response.content)
                record_count = int(response.headers.get("x-record-count", limit))
                if record_count < limit:
                    break
            else:
                batch = response.json()
                if not batch:
                    break
                all_data.extend(batch)
                if len(batch) < limit:
                    break

            offset += limit

        return all_data


def decrypt_fetched(
    data: list,
    decrypt_fn: Callable[[bytes], bytes],
) -> list[dict]:
    """Decrypt encrypted pages returned by fetch_data/fetch_cell_data.

    When ENCRYPTION_ENABLED, fetch methods return list[bytes] (one blob per page).
    Pass that list here with the model's decrypt function to get list[dict].
    If data is already list[dict] (no encryption), returns as-is.
    """
    if not data or isinstance(data[0], dict):
        return data
    result: list[dict] = []
    for blob in data:
        result.extend(json.loads(decrypt_fn(blob)))
    return result


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
