from fastapi import APIRouter, Depends

from src.schemas.fields import FieldsResponse
from src.services.data_storage_client import DataStorageClient
router = APIRouter()

@router.get("/", response_model=FieldsResponse)
async def get_fields(client: DataStorageClient = Depends(DataStorageClient)) -> FieldsResponse:
    """Get all fields"""
    fields = await client.get_available_fields()
    return FieldsResponse(fields=fields)
