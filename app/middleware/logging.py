"""
logging.py
----------
Structured request/response logging middleware using loguru.
Logs method, path, status code, and latency for every request.
"""

import time
from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        
        # Extract request info
        method = request.method
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"
        
        try:
            response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000
            
            # Log successful requests
            logger.info(
                f"{method} {path} | Status: {response.status_code} | "
                f"Latency: {process_time:.2f}ms | IP: {client_host}"
            )
            return response
            
        except Exception as e:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"{method} {path} | FAILED | Latency: {process_time:.2f}ms | "
                f"IP: {client_host} | Error: {e}"
            )
            raise
