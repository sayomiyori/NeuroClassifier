import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP
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

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASET_UPLOADS_TOTAL = Counter(
    "dataset_uploads_total",
    "Total dataset upload attempts",
    ["status"],
)

DATASETS_TOTAL = Gauge(
    "datasets_total",
    "Total number of datasets in the system",
)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAIN_DURATION_SECONDS = Histogram(
    "train_duration_seconds",
    "End-to-end training job duration in seconds",
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
)

MODELS_TRAINED_TOTAL = Counter(
    "models_trained_total",
    "Total number of successfully trained models",
)

ACTIVE_TRAINING_JOBS = Gauge(
    "active_training_jobs",
    "Number of currently running training jobs",
)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["model_id", "status"],
)

INFERENCE_DURATION_SECONDS = Histogram(
    "inference_duration_seconds",
    "Inference duration in seconds",
    ["model_id"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2],
)

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions made, by model and top predicted class",
    ["model_id", "class_name"],
)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

def metrics_endpoint() -> Response:
    """FastAPI handler that returns Prometheus text format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
