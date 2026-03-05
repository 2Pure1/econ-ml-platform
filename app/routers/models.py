"""
models.py
---------
Endpoints for model metadata and hot-reloading.
"""

from fastapi import APIRouter, Depends, Request
from app.models.schemas import ModelsInfoResponse, ModelInfo
from app.services.model_registry import ModelRegistryService

router = APIRouter()

def get_registry(request: Request) -> ModelRegistryService:
    return request.app.state.registry

@router.get("/info", response_model=ModelsInfoResponse, summary="Get loaded models info")
async def get_models_info(registry: ModelRegistryService = Depends(get_registry)):
    """Returns metadata for all currently loaded models."""
    models_info = registry.get_all_info()
    return ModelsInfoResponse(
        models=[ModelInfo(**m) for m in models_info],
        api_version="1.0.0",
        uptime_s=registry.uptime_s
    )

@router.post("/reload", summary="Hot-reload models")
async def reload_models(target: str = None, registry: ModelRegistryService = Depends(get_registry)):
    """
    Reloads models from the MLflow registry.
    If target is provided, only that model is reloaded.
    """
    results = await registry.hot_reload(target=target)
    return {"status": "success", "results": results}
