"""
metrics.py
----------
Prometheus metrics for the FastAPI app.
Collected metrics:
  - http_requests_total          (method, path, status_code)
  - http_request_duration_seconds (method, path)
  - prediction_total             (target, model)
  - prediction_latency_seconds   (target)
  - prediction_errors_total      (target)

Scraped by Prometheus at GET /metrics (mounted in main.py).
Visualised in Grafana via the econ-ml-platform dashboard.
"""

import time

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ── HTTP request metrics ───────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    ["method", "path"],
)

# ── Prediction metrics ─────────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "prediction_total",
    "Total predictions served",
    ["target", "model"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Model inference latency in seconds",
    ["target"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total prediction errors",
    ["target"],
)

MODEL_LOAD_TIME = Gauge(
    "model_load_timestamp_seconds",
    "Unix timestamp when the model was last loaded",
    ["target", "version"],
)


# ── Middleware ─────────────────────────────────────────────────────────────────
class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Intercepts every request/response and records Prometheus metrics.
    Skips /metrics endpoint itself to avoid recursive counting.
    """
    async def dispatch(self, request: Request, call_next):
        path   = request.url.path
        method = request.method

        # Don't track the metrics scrape endpoint
        if path == "/metrics":
            return await call_next(request)

        REQUESTS_IN_PROGRESS.labels(method=method, path=path).inc()
        t0 = time.perf_counter()

        try:
            response = await call_next(request)
            status   = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.perf_counter() - t0
            REQUEST_COUNT.labels(method=method, path=path, status_code=status).inc()
            REQUEST_LATENCY.labels(method=method, path=path).observe(elapsed)
            REQUESTS_IN_PROGRESS.labels(method=method, path=path).dec()

        return response
