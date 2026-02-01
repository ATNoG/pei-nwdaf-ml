"""Client for interacting with the Data Storage API."""

import httpx

from src.core.config import settings


class DataStorageClient:
    """Client for fetching data from the Data Storage service."""

    def __init__(self):
        self.base_url = settings.DATA_STORAGE_API_URL
        self.example_endpoint = settings.DATA_STORAGE_EXAMPLE_ENDPOINT
        self.excluded_fields = set(
            f.strip() for f in settings.DATA_STORAGE_EXCLUDED_FIELDS.split(",") if f.strip()
        )

    async def get_available_fields(self) -> list[str]:
        """
        Get available data field names from the data-storage example endpoint.

        Returns a list of field names, excluding metadata fields defined in
        DATA_STORAGE_EXCLUDED_FIELDS.
        """
        url = f"{self.base_url}{self.example_endpoint}"

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()

            data = response.json()

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
