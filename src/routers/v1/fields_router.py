from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.core.dependencies import get_db
from src.db.model_config import ModelConfigDB
from src.schemas.fields import FieldsDetailResponse, FieldsResponse, FieldStatus
from src.services.data_storage_client import DataStorageClient

router = APIRouter()


@router.get("", response_model=FieldsResponse | FieldsDetailResponse)
async def get_fields(
    include_model_status: bool = False,
    client: DataStorageClient = Depends(DataStorageClient),
    db: Session = Depends(get_db),
) -> FieldsResponse | FieldsDetailResponse:
    """Get all available fields from data storage."""
    fields = await client.get_available_fields()
    if not include_model_status:
        return FieldsResponse(fields=fields)

    configs = db.query(ModelConfigDB).all()
    fields_with_models = {f for c in configs for f in c.output_fields}
    return FieldsDetailResponse(fields=[
        FieldStatus(name=f, has_models=f in fields_with_models)
        for f in fields
    ])
