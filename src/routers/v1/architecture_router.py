"""Router for custom model architecture management."""

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.schemas.architecture import ArchitectureHelpResponse, ArchitectureResponse, ArchitectureUploadResponse
from src.services.architecture_service import ArchitectureService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_architecture_service() -> ArchitectureService:
    return ArchitectureService()


@router.post("", status_code=201, response_model=ArchitectureUploadResponse)
async def upload_architecture(
    name: str = Query(..., pattern=r"^[a-zA-Z0-9_-]+$"),
    file: UploadFile = File(...),
    uploaded_by: str = Query(default="unknown"),
    db: Session = Depends(get_db),
    svc: ArchitectureService = Depends(get_architecture_service),
):
    if not file.filename.endswith(".py"):
        raise HTTPException(status_code=422, detail="File must be a .py file")

    content = await file.read()

    try:
        svc.save_architecture(name, content, uploaded_by, db)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ArchitectureUploadResponse(name=name, uploaded_by=uploaded_by)


@router.get("", response_model=list[ArchitectureResponse])
def list_architectures(
    db: Session = Depends(get_db),
    svc: ArchitectureService = Depends(get_architecture_service),
):
    return svc.list_architectures(db)


@router.get("/help", response_model=ArchitectureHelpResponse)
def architecture_help():
    return ArchitectureHelpResponse(constraints=ArchitectureService.help())


@router.get("/{architecture_id}/download")
def download_architecture(
    architecture_id: str,
    svc: ArchitectureService = Depends(get_architecture_service),
):
    try:
        _, content = svc._load(architecture_id)
        svc._unload(architecture_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return Response(
        content=content,
        media_type="text/x-python",
        headers={"Content-Disposition": f"attachment; filename={architecture_id}.py"},
    )


@router.delete("/{architecture_id}", status_code=204)
def delete_architecture(
    architecture_id: str,
    db: Session = Depends(get_db),
    svc: ArchitectureService = Depends(get_architecture_service),
):
    try:
        svc.delete_architecture(architecture_id, db)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
