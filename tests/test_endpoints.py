"""
test_endpoints.py
-----------------
Integration tests for all three forecast endpoints.
Uses TestClient (no real MLflow needed — registry uses fallback models).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from datetime import datetime, timezone

# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_registry():
    """Registry that returns deterministic fallback predictions."""
    registry = MagicMock()
    registry.is_ready = True
    registry.uptime_s = 42.0
    registry._models  = {
        "gdp_growth":          MagicMock(model_name="best_gdp_growth", model_version="3",
                                         model_type="XGBoost", loaded_at=datetime.now(timezone.utc),
                                         mlflow_run_id="abc123", metrics={"rmse": 1.2}),
        "unemployment_rate":   MagicMock(model_name="best_unemployment_rate", model_version="2",
                                         model_type="LightGBM", loaded_at=datetime.now(timezone.utc),
                                         mlflow_run_id="def456", metrics={"rmse": 0.28}),
        "fed_funds_direction": MagicMock(model_name="best_fed_funds_direction", model_version="4",
                                         model_type="XGBoost", loaded_at=datetime.now(timezone.utc),
                                         mlflow_run_id="ghi789", metrics={"accuracy": 0.71}),
    }

    def _predict(target, features):
        import uuid
        values = {"gdp_growth": np.array([2.3]), "unemployment_rate": np.array([4.2]),
                  "fed_funds_direction": np.array([0])}
        return {
            "raw_output":   values.get(target, np.array([0.0])),
            "model_name":   f"best_{target}",
            "model_version":"1",
            "model_type":   "XGBoost",
            "run_id":       "test-run",
            "latency_ms":   5.2,
            "prediction_id":str(uuid.uuid4()),
            "predicted_at": datetime.now(timezone.utc),
        }

    registry.predict = _predict
    registry.get_all_info.return_value = [
        {"target": t, "model_name": f"best_{t}", "model_version": "1",
         "model_type": "XGBoost", "loaded_at": datetime.now(timezone.utc),
         "mlflow_run_id": "abc", "metrics": {"rmse": 0.5}}
        for t in ["gdp_growth", "unemployment_rate", "fed_funds_direction"]
    ]
    return registry


@pytest.fixture
def client(mock_registry):
    from app.main import create_app
    app = create_app()

    # Bypass lifespan — inject mock registry directly
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.registry = mock_registry
        yield c


# ── Health ─────────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ready_ok(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


# ── GDP forecast ───────────────────────────────────────────────────────────────
class TestGDPForecast:
    ENDPOINT = "/predict/gdp_growth"

    def test_basic_prediction(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {
                "unemployment_rate": 4.1,
                "fed_funds_rate": 4.5,
                "cpi_yoy_pct": 2.9,
                "gdp_billions_usd": 28000.0,
            },
            "horizon": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["target"] == "gdp_growth"
        assert "forecast_qoq_pct" in data
        assert isinstance(data["forecast_qoq_pct"], float)
        assert data["horizon_quarters"] == 1
        assert "prediction_id" in data
        assert "latency_ms" in data

    def test_confidence_interval_present(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {"unemployment_rate": 4.1},
            "horizon": 1,
        })
        data = resp.json()
        assert data["confidence_lower"] < data["forecast_qoq_pct"]
        assert data["confidence_upper"] > data["forecast_qoq_pct"]

    def test_regime_classification(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {"unemployment_rate": 4.1, "fed_funds_rate": 4.5},
            "horizon": 1,
        })
        assert resp.json()["regime"] in ["EXPANSION", "RECOVERY", "SLOWDOWN", "CONTRACTION"]

    def test_invalid_unemployment_rejected(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {"unemployment_rate": 999},  # > 30 — invalid
            "horizon": 1,
        })
        assert resp.status_code == 422

    def test_horizon_out_of_range_rejected(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {"unemployment_rate": 4.1},
            "horizon": 10,  # max is 4
        })
        assert resp.status_code == 422

    def test_null_features_accepted(self, client):
        """All null features should use fallback medians — not error."""
        resp = client.post(self.ENDPOINT, json={"features": {}, "horizon": 1})
        assert resp.status_code == 200


# ── Unemployment forecast ──────────────────────────────────────────────────────
class TestUnemploymentForecast:
    ENDPOINT = "/predict/unemployment"

    def test_basic_prediction(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {
                "unemployment_rate": 4.1,
                "nonfarm_payrolls_mom_change": 150.0,
                "fed_funds_rate": 4.5,
            },
            "horizon": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["target"] == "unemployment_rate"
        assert 0 <= data["forecast_rate_pct"] <= 30
        assert "change_from_current" in data

    def test_change_computed_correctly(self, client):
        """change_from_current = forecast - current."""
        resp = client.post(self.ENDPOINT, json={
            "features": {"unemployment_rate": 4.0},
            "horizon": 1,
        })
        data = resp.json()
        expected_change = round(data["forecast_rate_pct"] - 4.0, 4)
        assert abs(data["change_from_current"] - expected_change) < 0.001

    def test_lower_bound_non_negative(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {"unemployment_rate": 0.5},  # very low — CI lower could go negative
            "horizon": 1,
        })
        assert resp.json()["confidence_lower"] >= 0.0


# ── Fed Funds direction ────────────────────────────────────────────────────────
class TestFedFundsForecast:
    ENDPOINT = "/predict/fed_funds"

    def test_basic_prediction(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {
                "fed_funds_rate": 4.5,
                "cpi_yoy_pct": 2.9,
                "core_pce_yoy_pct": 2.6,
                "unemployment_rate": 4.1,
            },
            "horizon": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["direction"] in ["UP", "FLAT", "DOWN"]
        assert set(data["probabilities"].keys()) == {"UP", "FLAT", "DOWN"}
        assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01

    def test_implied_rate_computed(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {"fed_funds_rate": 4.5},
            "horizon": 2,
        })
        data = resp.json()
        assert data["current_rate"] == 4.5
        assert data["implied_next_rate"] in [4.25, 4.5, 4.75]

    def test_current_rate_null_still_works(self, client):
        resp = client.post(self.ENDPOINT, json={
            "features": {"cpi_yoy_pct": 2.9},
            "horizon": 2,
        })
        assert resp.status_code == 200
        assert resp.json()["current_rate"] is None


# ── Models info ────────────────────────────────────────────────────────────────
class TestModelsInfo:
    def test_models_info_returns_three_models(self, client):
        resp = client.get("/models/info")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 3
        targets = {m["target"] for m in data["models"]}
        assert "gdp_growth" in targets
        assert "unemployment_rate" in targets
        assert "fed_funds_direction" in targets
