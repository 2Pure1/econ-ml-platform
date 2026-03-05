# 🚀 Econ ML Platform

> **Production FastAPI service** serving economic forecast models from the MLflow Model Registry — GDP growth, unemployment rate, and Fed Funds Rate direction — with Prometheus metrics and Grafana dashboards.

![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2?logo=mlflow)
![Prometheus](https://img.shields.io/badge/Prometheus-2.48-E6522C?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-10.2-F46800?logo=grafana)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)

---

## 🏗️ Architecture

```
MLflow Model Registry
  best_gdp_growth       v3  (XGBoost)
  best_unemployment_rate v2  (LightGBM)         ← loaded on startup
  best_fed_funds_direction v4 (XGBoost)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│                    FastAPI App                         │
│                                                        │
│  POST /predict/gdp_growth                              │
│  POST /predict/unemployment          Pydantic          │
│  POST /predict/fed_funds             validation        │
│                                                        │
│  GET  /models/info                   Model metadata    │
│  POST /models/reload                 Hot-reload        │
│  GET  /health   GET /ready           Probes            │
│  GET  /metrics                       Prometheus scrape │
│                                                        │
│  Middleware:                                           │
│    RequestLoggingMiddleware  (structured logs)         │
│    MetricsMiddleware         (Prometheus counters)     │
└──────────────────────────┬─────────────────────────────┘
                           │ scrape /metrics
                           ▼
                    ┌────────────┐
                    │ Prometheus │ ← 15s scrape interval
                    └─────┬──────┘
                          │ datasource
                          ▼
                    ┌────────────┐
                    │  Grafana   │ → econ-ml-platform dashboard
                    └────────────┘
```

---

## 📡 API Endpoints

### `POST /predict/gdp_growth`

Forecasts US GDP quarter-over-quarter growth (%).

**Request:**
```json
{
  "features": {
    "unemployment_rate": 4.1,
    "fed_funds_rate": 4.5,
    "cpi_yoy_pct": 2.9,
    "gdp_billions_usd": 28200.0
  },
  "horizon": 1
}
```

**Response:**
```json
{
  "target": "gdp_growth",
  "horizon_quarters": 1,
  "forecast_qoq_pct": 2.31,
  "confidence_lower": 0.77,
  "confidence_upper": 3.85,
  "regime": "RECOVERY",
  "model_name": "best_gdp_growth",
  "model_version": "3",
  "prediction_id": "550e8400-e29b-41d4-a716",
  "predicted_at": "2025-01-15T14:32:01Z",
  "latency_ms": 8.4
}
```

---

### `POST /predict/unemployment`

Forecasts US unemployment rate (%).

**Response includes:**
- `forecast_rate_pct` — predicted unemployment rate
- `change_from_current` — pp change from the input `unemployment_rate`
- `confidence_lower` / `confidence_upper` — 80% prediction interval

---

### `POST /predict/fed_funds`

Classifies the likely direction of the Fed Funds Rate.

**Response:**
```json
{
  "target": "fed_funds_direction",
  "horizon_months": 2,
  "direction": "DOWN",
  "probabilities": { "UP": 0.08, "FLAT": 0.21, "DOWN": 0.71 },
  "current_rate": 4.5,
  "implied_next_rate": 4.25,
  "model_name": "best_fed_funds_direction",
  "model_version": "4",
  "latency_ms": 6.1
}
```

---

### Other endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe — always returns 200 if process is alive |
| `/ready` | GET | Readiness probe — 503 until all models are loaded |
| `/models/info` | GET | Currently loaded model names, versions, and training metrics |
| `/models/reload` | POST | Hot-reload models from MLflow without restart |
| `/metrics` | GET | Prometheus metrics scrape |
| `/docs` | GET | Interactive Swagger UI |
| `/redoc` | GET | ReDoc API documentation |

---

## 📊 Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, path, status_code | All HTTP requests |
| `http_request_duration_seconds` | Histogram | method, path | End-to-end latency |
| `http_requests_in_progress` | Gauge | method, path | Concurrent requests |
| `prediction_total` | Counter | target, model | Predictions served |
| `prediction_latency_seconds` | Histogram | target | Model inference time |
| `prediction_errors_total` | Counter | target | Failed predictions |

---

## 🖥️ Grafana Dashboard

Auto-provisioned dashboard at `http://localhost:3000` (admin/admin) shows:

- **Total predictions / errors / p99 latency / request rate** — top stat row
- **Predictions by target over time** — line chart
- **Inference latency p50/p95/p99** — percentile histogram chart
- **HTTP status code breakdown** — 2xx / 4xx / 5xx
- **Requests in progress** — concurrency gauge

---

## 🚀 Quick Start

### Prerequisites
- econ-forecast-engine models trained and registered in MLflow (Project 2)
- MLflow server running (`docker compose -f ../econ-forecast-engine/docker/docker-compose.mlflow.yml up -d`)

### 1. Start the platform

```bash
cd econ-ml-platform
set -a && source ../.env && set +a
docker compose -f docker/docker-compose.yml up -d
```

### 2. Verify all services

```bash
# API health
curl http://localhost:8000/health

# Models loaded?
curl http://localhost:8000/ready

# Which models are running?
curl http://localhost:8000/models/info | python3 -m json.tool
```

### 3. Make a prediction

```bash
# GDP growth forecast
curl -X POST http://localhost:8000/predict/gdp_growth \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "unemployment_rate": 4.1,
      "fed_funds_rate": 4.5,
      "cpi_yoy_pct": 2.9,
      "gdp_billions_usd": 28200.0
    },
    "horizon": 1
  }'

# Fed Funds direction (Dec 2024 data)
curl -X POST http://localhost:8000/predict/fed_funds \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "fed_funds_rate": 4.5,
      "cpi_yoy_pct": 2.7,
      "core_pce_yoy_pct": 2.4,
      "unemployment_rate": 4.2
    },
    "horizon": 2
  }'
```

### 4. Access UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs (Swagger) | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

### 5. Hot-reload a model after retraining

```bash
# Reload all models
curl -X POST "http://localhost:8000/models/reload"

# Reload a single target
curl -X POST "http://localhost:8000/models/reload?target=gdp_growth"
```

### 6. Run tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## 🔬 Design Decisions

**Why load models into memory on startup rather than loading per request?**
MLflow model loading involves downloading artifacts, deserialising the sklearn Pipeline or PyTorch checkpoint, and warming JIT caches. This takes 2–10 seconds per model. Loading once on startup means every request hits an already-warm model — p99 latency stays under 50ms instead of spiking to 10+ seconds on cold requests.

**Why `foreachBatch`-style hot reload instead of model server frameworks like Triton or BentoML?**
For 3 models serving tabular data, a full model server adds operational complexity without benefit. The `hot_reload()` method reloads a model under an asyncio lock, so in-flight requests complete against the old model before the swap. This is sufficient for monthly retrain cadence.

**Why Pydantic for input validation?**
Malformed inputs that reach model inference produce silent wrong answers — validation errors are much easier to debug than cryptic numpy shape errors. Pydantic gives us field-level validation (unemployment ≤ 30%, horizon ≤ 6), clear 422 error responses with the exact field that failed, and auto-generated OpenAPI docs from the schema.

**Why Prometheus + Grafana over a SaaS solution?**
For a portfolio project it demonstrates end-to-end observability infrastructure. In production you'd likely use Datadog or AWS CloudWatch, but the Prometheus data model (counters, histograms, labels) is identical — the skills transfer directly.

---

## 📈 Upstream / Downstream

| Direction | Project | Relationship |
|-----------|---------|-------------|
| Upstream | `econ-forecast-engine` | Reads trained models from MLflow Model Registry |
| Upstream | `econ-data-pipeline` | Shares PostgreSQL + Docker network |
| Downstream | `econ-forecaster-dashboard` | Calls `/predict/*` endpoints for live forecast display |

---

## 📄 License

MIT
