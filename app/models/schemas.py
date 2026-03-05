"""
schemas.py
----------
Pydantic request and response models for all three forecast endpoints.
Strict validation — bad inputs are rejected with a 422 before touching models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────
class FedFundsDirection(str, Enum):
    UP   = "UP"
    FLAT = "FLAT"
    DOWN = "DOWN"


class MarketRegime(str, Enum):
    EXPANSION    = "EXPANSION"
    SLOWDOWN     = "SLOWDOWN"
    CONTRACTION  = "CONTRACTION"
    RECOVERY     = "RECOVERY"
    UNKNOWN      = "UNKNOWN"


# ── Shared macro feature input ─────────────────────────────────────────────────
class MacroFeatures(BaseModel):
    """
    Current macroeconomic indicators used as input features.
    All values should be the most recently published figures.
    If a value is unavailable, pass null — the model will use its training median.
    """
    # Core indicators
    unemployment_rate:          Optional[float] = Field(None, ge=0, le=30,   description="U-3 unemployment rate (%)")
    fed_funds_rate:             Optional[float] = Field(None, ge=0, le=25,   description="Effective Federal Funds Rate (%)")
    cpi_yoy_pct:                Optional[float] = Field(None, ge=-5, le=30,  description="CPI year-over-year change (%)")
    core_pce_yoy_pct:           Optional[float] = Field(None, ge=-5, le=20,  description="Core PCE year-over-year change (%)")
    gdp_billions_usd:           Optional[float] = Field(None, gt=0,          description="Nominal GDP (billions USD)")
    nonfarm_payrolls_mom_change:Optional[float] = Field(None, ge=-20000, le=5000, description="Nonfarm payrolls monthly change (thousands)")
    m2_money_supply_billions:   Optional[float] = Field(None, gt=0,          description="M2 money supply (billions USD)")

    # Derived / optional
    real_fed_funds_rate:        Optional[float] = Field(None, description="Fed Funds Rate minus trailing CPI (%)")
    observation_month:          Optional[str]   = Field(None, description="ISO date of the most recent observation e.g. '2024-12-01'")

    @field_validator("observation_month")
    @classmethod
    def validate_date(cls, v):
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError("observation_month must be YYYY-MM-DD format")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "unemployment_rate": 4.1,
            "fed_funds_rate": 4.5,
            "cpi_yoy_pct": 2.9,
            "core_pce_yoy_pct": 2.6,
            "gdp_billions_usd": 28200.0,
            "nonfarm_payrolls_mom_change": 180.0,
            "m2_money_supply_billions": 21000.0,
            "observation_month": "2024-12-01",
        }
    }}


# ── GDP forecast ───────────────────────────────────────────────────────────────
class GDPForecastRequest(BaseModel):
    features: MacroFeatures
    horizon:  int = Field(1, ge=1, le=4, description="Quarters ahead to forecast (1–4)")

    model_config = {"json_schema_extra": {"example": {
        "features": {
            "unemployment_rate": 4.1,
            "fed_funds_rate": 4.5,
            "cpi_yoy_pct": 2.9,
            "gdp_billions_usd": 28200.0,
        },
        "horizon": 1
    }}}


class GDPForecastResponse(BaseModel):
    target:           str   = "gdp_growth"
    horizon_quarters: int
    forecast_qoq_pct: float = Field(description="Predicted GDP quarter-over-quarter growth (%)")
    confidence_lower: Optional[float] = Field(None, description="80% prediction interval lower bound")
    confidence_upper: Optional[float] = Field(None, description="80% prediction interval upper bound")
    regime:           MarketRegime
    model_name:       str
    model_version:    str
    prediction_id:    str
    predicted_at:     datetime
    latency_ms:       float


# ── Unemployment forecast ──────────────────────────────────────────────────────
class UnemploymentForecastRequest(BaseModel):
    features: MacroFeatures
    horizon:  int = Field(1, ge=1, le=6, description="Months ahead to forecast (1–6)")

    model_config = {"json_schema_extra": {"example": {
        "features": {
            "unemployment_rate": 4.1,
            "fed_funds_rate": 4.5,
            "nonfarm_payrolls_mom_change": 150.0,
            "cpi_yoy_pct": 2.9,
        },
        "horizon": 1
    }}}


class UnemploymentForecastResponse(BaseModel):
    target:              str   = "unemployment_rate"
    horizon_months:      int
    forecast_rate_pct:   float = Field(description="Predicted unemployment rate (%)")
    change_from_current: Optional[float] = Field(None, description="Predicted change from current rate (pp)")
    confidence_lower:    Optional[float]
    confidence_upper:    Optional[float]
    model_name:          str
    model_version:       str
    prediction_id:       str
    predicted_at:        datetime
    latency_ms:          float


# ── Fed Funds direction forecast ───────────────────────────────────────────────
class FedFundsForecastRequest(BaseModel):
    features: MacroFeatures
    horizon:  int = Field(2, ge=1, le=6, description="Months ahead to forecast (1–6)")

    model_config = {"json_schema_extra": {"example": {
        "features": {
            "fed_funds_rate": 4.5,
            "cpi_yoy_pct": 2.9,
            "core_pce_yoy_pct": 2.6,
            "unemployment_rate": 4.1,
        },
        "horizon": 2
    }}}


class FedFundsForecastResponse(BaseModel):
    target:           str = "fed_funds_direction"
    horizon_months:   int
    direction:        FedFundsDirection
    probabilities:    dict[str, float] = Field(description="Probability for each direction class")
    current_rate:     Optional[float]
    implied_next_rate:Optional[float]  = Field(None, description="Implied rate if UP=+0.25, DOWN=-0.25, FLAT=0")
    model_name:       str
    model_version:    str
    prediction_id:    str
    predicted_at:     datetime
    latency_ms:       float


# ── Model info ─────────────────────────────────────────────────────────────────
class ModelInfo(BaseModel):
    target:        str
    model_name:    str
    model_version: str
    model_type:    str   # XGBoost / ARIMA / LSTM / Prophet
    loaded_at:     datetime
    mlflow_run_id: Optional[str]
    metrics:       dict[str, float]  # rmse / accuracy from training run


class ModelsInfoResponse(BaseModel):
    models:      list[ModelInfo]
    api_version: str
    uptime_s:    float
