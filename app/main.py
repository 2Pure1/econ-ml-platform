"""
main.py
-------
FastAPI application entrypoint for the econ-ml-platform.

Serves trained economic forecast models loaded from the MLflow Model Registry.
Three forecast targets:
  - GDP growth (QoQ %)
  - Unemployment rate (monthly level)
  - Fed Funds Rate direction (UP / FLAT / DOWN)

Endpoints:
  GET  /health                    liveness probe
  GET  /ready                     readiness probe (models loaded?)
  POST /predict/gdp_growth        GDP QoQ % forecast
  POST /predict/unemployment      unemployment rate forecast
  POST /predict/fed_funds         Fed Funds direction forecast
  GET  /models/info               currently loaded model versions
  POST /models/reload             hot-reload models from MLflow registry
  GET  /metrics                   Prometheus metrics scrape endpoint
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import make_asgi_app

from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.metrics import MetricsMiddleware
from app.routers import health, predict, models
from app.services.model_registry import ModelRegistryService

# ── App metadata ───────────────────────────────────────────────────────────────
APP_VERSION = "1.0.0"
APP_TITLE   = "Econ ML Platform"
APP_DESC    = """
Production API serving economic forecast models trained on US macro data.

**Forecasts:**
- **GDP growth** — Quarterly % change, 1 quarter ahead
- **Unemployment rate** — Monthly level, 1 month ahead
- **Fed Funds direction** — UP / FLAT / DOWN classification, 2 months ahead

Models are loaded from the **MLflow Model Registry** and hot-reloaded
without downtime. Each prediction is cached in Redis for 1 hour.
"""


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")

    # Load models from MLflow registry
    registry = ModelRegistryService()
    await registry.load_all_models()
    app.state.registry = registry

    logger.info("All models loaded — API ready")
    yield

    # Shutdown
    logger.info("Shutting down — releasing model resources")
    app.state.registry = None


# ── Build app ──────────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_TITLE,
        description=APP_DESC,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Custom middleware ──────────────────────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router,   tags=["Health"])
    app.include_router(predict.router,  prefix="/predict",  tags=["Predictions"])
    app.include_router(models.router,   prefix="/models",   tags=["Models"])

    # ── Prometheus metrics endpoint ────────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error":   "internal_server_error",
                "message": "An unexpected error occurred. Check /health for system status.",
                "path":    str(request.url.path),
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("ENV", "production") == "development",
        workers=int(os.environ.get("WORKERS", 1)),
        log_level="info",
    )
