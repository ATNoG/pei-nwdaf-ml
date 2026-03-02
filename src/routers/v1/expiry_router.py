"""REST endpoints for model expiry policy management and audit log."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from src.db.database import get_db
from src.db.expiry_audit import ExpiryAuditDB
from src.schemas.expiry import AuditEntry, ExpiryPolicy, ExpiryPolicyUpdate

router = APIRouter()


def _expiry_service(request: Request):
    svc = getattr(request.app.state, "expiry_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Expiry service not available")
    return svc


@router.get("/policy", response_model=ExpiryPolicy)
def get_policy(request: Request):
    """Return the current model expiry policy."""
    return _expiry_service(request).get_policy()


@router.patch("/policy", response_model=ExpiryPolicy)
def patch_policy(update: ExpiryPolicyUpdate, request: Request):
    """Partially update the expiry policy. Omitted fields keep their current value."""
    return _expiry_service(request).update_policy(update)


@router.put("/policy", response_model=ExpiryPolicy)
def replace_policy(policy: ExpiryPolicy, request: Request):
    """Fully replace the expiry policy."""
    return _expiry_service(request).replace_policy(policy)


@router.post("/sweep")
def trigger_sweep(request: Request):
    """Manually trigger an expiry sweep and return how many models were deleted."""
    try:
        deleted = _expiry_service(request).run_sweep()
        return {"deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit", response_model=list[AuditEntry])
def get_audit(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Return the audit log of expired models, newest first."""
    rows = (
        db.query(ExpiryAuditDB)
        .order_by(ExpiryAuditDB.expired_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return rows
