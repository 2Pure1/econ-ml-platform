"""
health.py
---------
Liveness and readiness probes.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from app.services.model_registry import ModelRegistryService

router = APIRouter()

def get_registry(request: Request) -> ModelRegistryService:
    return request.app.state.registry

@router.get("/health", summary="Liveness probe")
async def health_check():
    """Returns 200 if the process is alive."""
    return {"status": "ok"}

@router.get("/ready", summary="Readiness probe")
async def readiness_check(registry: ModelRegistryService = Depends(get_registry)):
    """Returns 200 if all models are loaded, 503 otherwise."""
    if not registry or not registry.is_ready:
        raise HTTPException(status_code=503, detail="Models not yet loaded")
    return {"status": "ready"}
