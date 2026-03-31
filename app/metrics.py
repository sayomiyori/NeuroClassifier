import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
)

DATASET_UPLOADS_TOTAL = Counter(
    "dataset_uploads_total",
    "Total dataset upload attempts",
    ["status"],
)

INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["model_id", "status"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "inference_duration_seconds",
    "Inference duration in seconds",
    ["model_id"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2],
)

ACTIVE_TRAINING_JOBS = Gauge(
    "active_training_jobs",
    "Number of currently running training jobs",
)


def metrics_endpoint() -> Response:
    """FastAPI endpoint handler that returns Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
