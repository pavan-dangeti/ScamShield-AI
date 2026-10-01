"""Structured logs, request IDs and Prometheus metrics for the API.

One JSON object per log line, so a log platform can filter on fields instead of
parsing text.  Every line written while a request is in flight carries that
request's ID, which is also returned to the client as ``X-Request-ID``: a user
reporting an error can quote it and the matching log lines are one search away.
That is the tracing this single-service app needs; there are no downstream calls
to propagate a trace to.

Message text is never logged.  It can contain personal data, and the analysis
log in SQLite (scrubbed) is the place to look at messages.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from prometheus_client import Counter, Gauge, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

REQUESTS = Counter("scamshield_http_requests_total", "HTTP requests", ["route", "method", "status"])
LATENCY = Histogram(
    "scamshield_http_request_duration_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter("scamshield_predictions_total", "Predictions served", ["label", "language", "script"])
# The live score distribution is the drift signal: it moves before accuracy does,
# and it is what scripts/monitor_drift.py compares against the reference.
SCORES = Histogram(
    "scamshield_scam_probability",
    "Distribution of served scam probabilities",
    ["model"],
    buckets=tuple(round(0.1 * step, 1) for step in range(1, 11)),
)
DB_ERRORS = Counter("scamshield_db_errors_total", "Failed analysis-log writes", ["operation"])
MODEL_INFO = Gauge("scamshield_model_info", "Model currently serving", ["model"])

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn's access log duplicates the request log below, in plain text.
    logging.getLogger("uvicorn.access").disabled = True


class RequestContextMiddleware:
    """Assigns a request ID, times the request, logs it and records metrics.

    Pure ASGI rather than BaseHTTPMiddleware, which would add a task hop to every
    request.  The route label is the matched path template, not the raw URL, so
    the metric's cardinality stays bounded.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.log = logging.getLogger("scamshield.request")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        # Accept a caller's ID only if it is short and plain, so it cannot inject into logs.
        request_id = incoming if 0 < len(incoming) <= 64 and incoming.replace("-", "").isalnum() else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        status = 500
        started = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            elapsed = time.perf_counter() - started
            route = getattr(scope.get("route"), "path", None) or ("static" if status < 400 else "unmatched")
            REQUESTS.labels(route, scope["method"], str(status)).inc()
            LATENCY.labels(route).observe(elapsed)
            if route != "/metrics":
                self.log.info(
                    "request",
                    extra={"method": scope["method"], "route": route, "status": status, "ms": round(elapsed * 1000, 2)},
                )
            request_id_var.reset(token)
