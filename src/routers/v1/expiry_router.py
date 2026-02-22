"""
Endpoints for ML model expiry policy management
"""

from fastapi import APIRouter, HTTPException, Request

from src.schemas.expiry import ExpiryPolicy, ExpiryPolicyUpdate

router = APIRouter()


def _get_expiry_service(request: Request):
    expiry_service = getattr(request.app.state, "expiry_service", None)
    if expiry_service is None:
        raise HTTPException(status_code=503, detail="Expiry service not available")
    return expiry_service


@router.get("/policy", response_model=ExpiryPolicy)
def get_policy(request: Request):
    """Return the current model expiry policy."""
    return _get_expiry_service(request).get_policy()


@router.patch("/policy", response_model=ExpiryPolicy)
def patch_policy(update: ExpiryPolicyUpdate, request: Request):
    """Partially update the expiry policy. Omitted fields keep their current value."""
    return _get_expiry_service(request).update_policy(update)


@router.put("/policy", response_model=ExpiryPolicy)
def replace_policy(policy: ExpiryPolicy, request: Request):
    """Fully replace the expiry policy."""
    return _get_expiry_service(request).replace_policy(policy)


@router.post("/sweep")
def trigger_sweep(request: Request):
    """Manually trigger an expiry sweep and return how many models were deleted."""
    try:
        deleted = _get_expiry_service(request).run_sweep()
        return {"deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
