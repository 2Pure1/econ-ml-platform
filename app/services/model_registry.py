"""
model_registry.py
-----------------
Loads the best model per target from the MLflow Model Registry on startup.
Exposes a predict() method used by all three routers.

Design:
  - On startup, queries MLflow for the latest Production-stage model per target.
  - Falls back to the highest-version model if no Production stage is set.
  - Models are held in memory — no per-request MLflow round-trip.
  - hot_reload() swaps in new model versions without downtime (used by POST /models/reload).
  - predict() returns raw numpy output + model metadata for the router to format.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from loguru import logger

# ── Configuration ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_TARGETS = {
    "gdp_growth":          "best_gdp_growth",
    "unemployment_rate":   "best_unemployment_rate",
    "fed_funds_direction": "best_fed_funds_direction",
}

# Feature columns expected by models (must match feature_engineering.py output)
# Models trained with ~75 features; we accept the subset the caller provides
# and fill the rest with training-time medians stored alongside the model.
CORE_FEATURES = [
    "unemployment_rate", "fed_funds_rate", "cpi_yoy_pct", "core_pce_yoy_pct",
    "gdp_billions_usd", "nonfarm_payrolls_mom_change", "m2_money_supply_billions",
    "real_fed_funds_rate", "month_num", "quarter_num",
]


@dataclass
class LoadedModel:
    target:        str
    model_name:    str
    model_version: str
    model_type:    str
    mlflow_run_id: Optional[str]
    metrics:       dict[str, float]
    loaded_at:     datetime
    _model:        Any = field(repr=False)
    _feature_medians: dict[str, float] = field(default_factory=dict, repr=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class ModelRegistryService:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self._client  = mlflow.tracking.MlflowClient()
        self._models:  dict[str, LoadedModel] = {}
        self._lock     = asyncio.Lock()
        self._start_ts = time.time()

    # ── Load all models ────────────────────────────────────────────────────────
    async def load_all_models(self) -> None:
        logger.info(f"Loading models from MLflow registry at {MLFLOW_TRACKING_URI}")
        tasks = [self._load_model(target, name) for target, name in MODEL_TARGETS.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for target, result in zip(MODEL_TARGETS, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to load model for {target}: {result}. "
                               f"Using fallback dummy model.")
                self._models[target] = self._make_fallback_model(target)
        loaded = sum(1 for m in self._models.values() if m.model_type != "fallback")
        logger.info(f"Loaded {loaded}/{len(MODEL_TARGETS)} models from MLflow registry")

    async def _load_model(self, target: str, registry_name: str) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model_sync, target, registry_name)

    def _load_model_sync(self, target: str, registry_name: str) -> None:
        try:
            # Try to get the Production-staged version first
            versions = self._client.get_latest_versions(
                registry_name, stages=["Production"]
            )
            if not versions:
                # Fall back to latest version regardless of stage
                versions = self._client.get_latest_versions(
                    registry_name, stages=["None", "Staging", "Archived"]
                )
            if not versions:
                raise ValueError(f"No versions found for registered model '{registry_name}'")

            version = versions[0]
            run_id  = version.run_id

            # Load the model artifact
            model_uri = f"models:/{registry_name}/{version.version}"
            model = mlflow.pyfunc.load_model(model_uri)

            # Pull training metrics from the run
            run = self._client.get_run(run_id)
            metrics    = {k: round(v, 4) for k, v in run.data.metrics.items()
                          if k in ("rmse", "mae", "r2", "accuracy", "f1_macro")}
            model_type = run.data.params.get("model_type", "unknown")

            # Load feature medians saved as artifact (for missing-value imputation)
            feature_medians = {}
            try:
                artifacts = self._client.list_artifacts(run_id, "feature_medians")
                if artifacts:
                    local = self._client.download_artifacts(run_id, "feature_medians/medians.json")
                    import json
                    with open(local) as f:
                        feature_medians = json.load(f)
            except Exception:
                pass  # medians optional — we fall back to zero

            self._models[target] = LoadedModel(
                target=target,
                model_name=registry_name,
                model_version=str(version.version),
                model_type=model_type,
                mlflow_run_id=run_id,
                metrics=metrics,
                loaded_at=datetime.now(timezone.utc),
                _model=model,
                _feature_medians=feature_medians,
            )
            logger.info(f"Loaded {target}: {registry_name} v{version.version} "
                        f"({model_type}) | metrics={metrics}")

        except Exception as e:
            logger.error(f"Could not load model for {target} from MLflow: {e}")
            raise

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict(self, target: str, features: dict) -> dict:
        """
        Run inference for a single target.
        Returns dict with raw output + model metadata for the router to format.
        """
        if target not in self._models:
            raise KeyError(f"No model loaded for target '{target}'")

        loaded = self._models[target]
        t0 = time.perf_counter()

        # Build feature DataFrame — fill missing with training medians or 0
        row = {}
        for col in CORE_FEATURES:
            val = features.get(col)
            if val is None:
                val = loaded._feature_medians.get(col, 0.0)
            row[col] = val

        # Add derived features the model expects
        if row.get("fed_funds_rate") and row.get("cpi_yoy_pct"):
            row["real_fed_funds_rate"] = row["fed_funds_rate"] - row["cpi_yoy_pct"]

        # Calendar features from observation_month
        obs_date = features.get("observation_month")
        if obs_date:
            try:
                dt = pd.Timestamp(obs_date)
                row["month_num"]   = dt.month
                row["quarter_num"] = dt.quarter
                row["is_q1"]       = int(dt.month == 1)
                row["is_q4"]       = int(dt.month == 10)
            except Exception:
                pass

        X = pd.DataFrame([row])

        # Run model — pyfunc handles sklearn Pipeline, PyTorch, Prophet etc.
        raw_output = loaded.predict(X)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "raw_output":   raw_output,
            "model_name":   loaded.model_name,
            "model_version":loaded.model_version,
            "model_type":   loaded.model_type,
            "run_id":       loaded.mlflow_run_id,
            "latency_ms":   round(latency_ms, 2),
            "prediction_id":str(uuid.uuid4()),
            "predicted_at": datetime.now(timezone.utc),
        }

    # ── Hot reload ─────────────────────────────────────────────────────────────
    async def hot_reload(self, target: str | None = None) -> dict:
        """
        Reload one or all models from MLflow registry without restarting the API.
        Returns dict of {target: new_version}.
        """
        async with self._lock:
            targets = [target] if target else list(MODEL_TARGETS.keys())
            results = {}
            for t in targets:
                old_version = self._models.get(t, None)
                old_v = old_version.model_version if old_version else "none"
                try:
                    await self._load_model(t, MODEL_TARGETS[t])
                    new_v = self._models[t].model_version
                    results[t] = {"old_version": old_v, "new_version": new_v,
                                  "status": "reloaded"}
                    logger.info(f"Hot-reloaded {t}: v{old_v} → v{new_v}")
                except Exception as e:
                    results[t] = {"status": "failed", "error": str(e)}
            return results

    # ── Readiness / info ───────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        return len(self._models) == len(MODEL_TARGETS)

    @property
    def uptime_s(self) -> float:
        return round(time.time() - self._start_ts, 1)

    def get_all_info(self) -> list[dict]:
        return [
            {
                "target":        m.target,
                "model_name":    m.model_name,
                "model_version": m.model_version,
                "model_type":    m.model_type,
                "loaded_at":     m.loaded_at,
                "mlflow_run_id": m.mlflow_run_id,
                "metrics":       m.metrics,
            }
            for m in self._models.values()
        ]

    # ── Fallback model (when MLflow unavailable) ───────────────────────────────
    def _make_fallback_model(self, target: str) -> LoadedModel:
        """
        Returns a deterministic fallback that uses historical averages.
        Ensures the API stays up even if MLflow is temporarily unavailable.
        Used for development and CI testing without a running MLflow server.
        """
        FALLBACK_VALUES = {
            "gdp_growth":          2.3,
            "unemployment_rate":   4.0,
            "fed_funds_direction": 0,   # FLAT
        }

        class _FallbackModel:
            def predict(self, X):
                return np.array([FALLBACK_VALUES.get(target, 0.0)] * len(X))

        return LoadedModel(
            target=target,
            model_name=f"fallback_{target}",
            model_version="0",
            model_type="fallback",
            mlflow_run_id=None,
            metrics={},
            loaded_at=datetime.now(timezone.utc),
            _model=_FallbackModel(),
        )
