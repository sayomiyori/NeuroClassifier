import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.v1 import datasets, models, predict, training
from app.api.v1 import train as train_api
from app.config import get_settings
from app.db.session import engine, Base
from app.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    DATASETS_TOTAL,
    MODELS_TRAINED_TOTAL,
    metrics_endpoint,
)
from app.models.dataset import Dataset, DatasetStatus
from app.models.ml_model import MLModel, ModelStatus
from sqlalchemy import select, func

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup and seed Prometheus gauges from DB."""
    logger.info("Starting NeuroClassifier …")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured.")

    # Seed gauges from DB so they reflect current state after restart
    from app.db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        ds_count = (await db.execute(select(func.count()).select_from(Dataset))).scalar_one()
        m_count = (await db.execute(
            select(func.count()).select_from(MLModel).where(MLModel.status == ModelStatus.READY)
        )).scalar_one()
        DATASETS_TOTAL.set(ds_count)
        if m_count:
            MODELS_TRAINED_TOTAL.inc(m_count)
    logger.info("Metrics seeded: datasets=%d, models=%d", ds_count, m_count)

    yield
    logger.info("Shutting down NeuroClassifier.")
    await engine.dispose()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "**NeuroClassifier** — ML inference platform for image classification.\n\n"
        "Upload datasets, train models via LoRA fine-tuning, and run inference with ONNX Runtime."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration = time.perf_counter() - start

    endpoint = request.url.path
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code,
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(datasets.router, prefix=API_PREFIX)
app.include_router(training.router, prefix=API_PREFIX)  # legacy
app.include_router(train_api.router, prefix=API_PREFIX)
app.include_router(models.router, prefix=API_PREFIX)
app.include_router(predict.router, prefix=API_PREFIX)


@app.get("/health", tags=["Health"], summary="Health check")
async def health():
    return {"status": "ok", "version": settings.app_version}


@app.get("/metrics", tags=["Observability"], summary="Prometheus metrics")
async def metrics():
    return metrics_endpoint()
