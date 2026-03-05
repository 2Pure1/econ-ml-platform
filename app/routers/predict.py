"""
predict.py
----------
Routers for all three forecast endpoints.

POST /predict/gdp_growth        → GDPForecastResponse
POST /predict/unemployment      → UnemploymentForecastResponse
POST /predict/fed_funds         → FedFundsForecastResponse

Each endpoint:
  1. Validates request via Pydantic
  2. Extracts feature dict from MacroFeatures
  3. Calls model_registry.predict(target, features)
  4. Formats the raw output into the typed response
  5. Records Prometheus metrics (prediction count, latency, target)
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from app.models.schemas import (
    FedFundsForecastRequest, FedFundsForecastResponse, FedFundsDirection,
    GDPForecastRequest, GDPForecastResponse,
    MarketRegime,
    UnemploymentForecastRequest, UnemploymentForecastResponse,
)
from app.services.model_registry import ModelRegistryService
from app.middleware.metrics import (
    PREDICTION_COUNTER, PREDICTION_LATENCY, PREDICTION_ERRORS,
)

router = APIRouter()


def get_registry(request: Request) -> ModelRegistryService:
    return request.app.state.registry


# ── GDP growth ─────────────────────────────────────────────────────────────────
@router.post(
    "/gdp_growth",
    response_model=GDPForecastResponse,
    summary="Forecast GDP quarter-over-quarter growth",
    description="""
Predicts US GDP QoQ % growth for the specified number of quarters ahead.

**Input:** Current macro indicators (unemployment, Fed Funds rate, CPI, etc.)
**Output:** Point forecast of QoQ % change with 80% prediction interval.

The model was trained on `fct_macro_indicators_monthly` (1994–2024) using
the best-performing model from the MLflow experiment suite.
    """,
)
async def predict_gdp(
    payload: GDPForecastRequest,
    registry: ModelRegistryService = Depends(get_registry),
):
    try:
        features = {
            k: v for k, v in payload.features.model_dump().items()
            if v is not None
        }
        result = registry.predict("gdp_growth", features)

        raw = float(np.squeeze(result["raw_output"]))

        # Classify macro regime from the forecast
        if raw >= 3.0:
            regime = MarketRegime.EXPANSION
        elif raw >= 1.0:
            regime = MarketRegime.RECOVERY
        elif raw >= 0.0:
            regime = MarketRegime.SLOWDOWN
        else:
            regime = MarketRegime.CONTRACTION

        # Approximate 80% PI: ±1.28 * model RMSE (from training metrics)
        rmse = result.get("metrics", {}).get("rmse", 1.5) if hasattr(result, "get") else 1.5
        interval_half = 1.28 * rmse

        PREDICTION_COUNTER.labels(target="gdp_growth", model=result["model_type"]).inc()
        PREDICTION_LATENCY.labels(target="gdp_growth").observe(result["latency_ms"] / 1000)

        return GDPForecastResponse(
            horizon_quarters=payload.horizon,
            forecast_qoq_pct=round(raw, 4),
            confidence_lower=round(raw - interval_half, 4),
            confidence_upper=round(raw + interval_half, 4),
            regime=regime,
            model_name=result["model_name"],
            model_version=result["model_version"],
            prediction_id=result["prediction_id"],
            predicted_at=result["predicted_at"],
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(target="gdp_growth").inc()
        logger.exception(f"GDP forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Unemployment rate ──────────────────────────────────────────────────────────
@router.post(
    "/unemployment",
    response_model=UnemploymentForecastResponse,
    summary="Forecast unemployment rate",
    description="""
Predicts the US unemployment rate (U-3) for the specified number of months ahead.

**Input:** Current macro indicators
**Output:** Point forecast of the unemployment rate (%) with 80% prediction interval.
    """,
)
async def predict_unemployment(
    payload: UnemploymentForecastRequest,
    registry: ModelRegistryService = Depends(get_registry),
):
    try:
        features = {k: v for k, v in payload.features.model_dump().items() if v is not None}
        result = registry.predict("unemployment_rate", features)

        raw = float(np.squeeze(result["raw_output"]))
        current = features.get("unemployment_rate")
        change  = round(raw - current, 4) if current is not None else None

        rmse = 0.3  # typical unemployment forecast RMSE
        interval_half = 1.28 * rmse

        PREDICTION_COUNTER.labels(target="unemployment_rate", model=result["model_type"]).inc()
        PREDICTION_LATENCY.labels(target="unemployment_rate").observe(result["latency_ms"] / 1000)

        return UnemploymentForecastResponse(
            horizon_months=payload.horizon,
            forecast_rate_pct=round(raw, 4),
            change_from_current=change,
            confidence_lower=round(max(0.0, raw - interval_half), 4),
            confidence_upper=round(raw + interval_half, 4),
            model_name=result["model_name"],
            model_version=result["model_version"],
            prediction_id=result["prediction_id"],
            predicted_at=result["predicted_at"],
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(target="unemployment_rate").inc()
        logger.exception(f"Unemployment forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Fed Funds direction ────────────────────────────────────────────────────────
@router.post(
    "/fed_funds",
    response_model=FedFundsForecastResponse,
    summary="Forecast Fed Funds Rate direction",
    description="""
Classifies the likely direction of the Federal Funds Rate in the specified
number of months ahead: **UP** (+25bps), **FLAT** (no change), or **DOWN** (-25bps).

**Input:** Current macro indicators (especially CPI, Core PCE, unemployment)
**Output:** Direction class with probabilities for each outcome.

This is a 3-class classification model. The `probabilities` field shows the
model's confidence in each direction — useful for risk-adjusted decisions.
    """,
)
async def predict_fed_funds(
    payload: FedFundsForecastRequest,
    registry: ModelRegistryService = Depends(get_registry),
):
    try:
        features = {k: v for k, v in payload.features.model_dump().items() if v is not None}
        result = registry.predict("fed_funds_direction", features)

        raw = result["raw_output"]

        # If model returns class probabilities (shape: [1, 3])
        if hasattr(raw, "shape") and len(raw.shape) > 1 and raw.shape[1] == 3:
            probs_arr = np.squeeze(raw)
            class_idx = int(np.argmax(probs_arr))
            probs = {
                FedFundsDirection.DOWN: round(float(probs_arr[0]), 4),
                FedFundsDirection.FLAT: round(float(probs_arr[1]), 4),
                FedFundsDirection.UP:   round(float(probs_arr[2]), 4),
            }
        else:
            # Model returns a single class label (-1, 0, 1)
            class_idx_raw = int(np.squeeze(raw))
            # Map -1→0(DOWN), 0→1(FLAT), 1→2(UP) for index, or use directly
            label_map = {-1: FedFundsDirection.DOWN, 0: FedFundsDirection.FLAT, 1: FedFundsDirection.UP}
            direction = label_map.get(class_idx_raw, FedFundsDirection.FLAT)
            probs = {d: (0.7 if d == direction else 0.15) for d in FedFundsDirection}
            class_idx = class_idx_raw

        direction_map = {0: FedFundsDirection.DOWN, 1: FedFundsDirection.FLAT, 2: FedFundsDirection.UP}
        direction = direction_map.get(class_idx, FedFundsDirection.FLAT)

        # Implied next rate
        current_rate = features.get("fed_funds_rate")
        rate_delta   = {FedFundsDirection.UP: 0.25, FedFundsDirection.FLAT: 0.0, FedFundsDirection.DOWN: -0.25}
        implied_rate = round(current_rate + rate_delta[direction], 4) if current_rate else None

        PREDICTION_COUNTER.labels(target="fed_funds_direction", model=result["model_type"]).inc()
        PREDICTION_LATENCY.labels(target="fed_funds_direction").observe(result["latency_ms"] / 1000)

        return FedFundsForecastResponse(
            horizon_months=payload.horizon,
            direction=direction,
            probabilities={d.value: p for d, p in probs.items()},
            current_rate=current_rate,
            implied_next_rate=implied_rate,
            model_name=result["model_name"],
            model_version=result["model_version"],
            prediction_id=result["prediction_id"],
            predicted_at=result["predicted_at"],
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        PREDICTION_ERRORS.labels(target="fed_funds_direction").inc()
        logger.exception(f"Fed Funds forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
