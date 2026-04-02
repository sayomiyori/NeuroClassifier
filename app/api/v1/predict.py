import logging
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.metrics import (
    INFERENCE_DURATION_SECONDS,
    INFERENCE_REQUESTS_TOTAL,
    PREDICTIONS_TOTAL,
)
from app.models.ml_model import MLModel, ModelStatus
from app.services import s3 as s3_service
from app.services.inference import predict as run_predict
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Inference"])
settings = get_settings()

MAX_IMAGE_BYTES = 10 * 1024 * 1024   # 10 MB
MAX_BATCH_BYTES = 500 * 1024 * 1024  # 500 MB


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    class_name: str
    confidence: float


class PredictResponse(BaseModel):
    model_id: uuid.UUID
    predictions: List[Prediction]
    latency_ms: float


class BatchJobResponse(BaseModel):
    job_id: str
    status: str


class BatchResultResponse(BaseModel):
    job_id: str
    status: str
    results_url: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_ready_model(db: AsyncSession, model_id: uuid.UUID) -> MLModel:
    result = await db.execute(select(MLModel).where(MLModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {model_id} is not ready (status: {model.status}).",
        )
    if not model.s3_bucket or not model.onnx_s3_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model artifact not available in storage.",
        )
    return model


def _class_names(model: MLModel) -> List[str]:
    if not model.class_names:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model has no class names registered.",
        )
    # class_names stored as {name: index} dict → sort by index to get ordered list
    if isinstance(model.class_names, dict):
        return [k for k, _ in sorted(model.class_names.items(), key=lambda x: x[1])]
    return list(model.class_names)


# ---------------------------------------------------------------------------
# POST /predict  — single image
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=PredictResponse,
    summary="Run inference on a single image",
    description=(
        "Upload an image (JPEG/PNG/WebP) and a model_id form field. "
        "Returns top-K class predictions with confidence scores."
    ),
)
async def predict_single(
    db: Annotated[AsyncSession, Depends(get_db)],
    model_id: uuid.UUID = Form(..., description="Registered model UUID"),
    file: UploadFile = File(..., description="Image file (JPEG / PNG / WebP)"),
    top_k: int = Form(default=5),
):
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type: {file.content_type}. Allowed: {sorted(allowed)}",
        )

    model = await _get_ready_model(db, model_id)
    names = _class_names(model)

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    mid = str(model_id)
    try:
        predictions, latency_ms = run_predict(
            model_id=mid,
            image_bytes=image_bytes,
            s3_bucket=model.s3_bucket,
            s3_key=model.onnx_s3_key,
            class_names=names,
            top_k=top_k,
        )
        INFERENCE_REQUESTS_TOTAL.labels(model_id=mid, status="success").inc()
        if predictions:
            PREDICTIONS_TOTAL.labels(model_id=mid, class_name=predictions[0]["class_name"]).inc()
    except Exception as exc:
        INFERENCE_REQUESTS_TOTAL.labels(model_id=mid, status="error").inc()
        logger.exception("Inference failed for model %s", model_id)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    INFERENCE_DURATION_SECONDS.labels(model_id=mid).observe(latency_ms / 1000)

    return PredictResponse(
        model_id=model_id,
        predictions=[Prediction(**p) for p in predictions],
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /predict/batch  — zip of images → async Celery job
# ---------------------------------------------------------------------------

@router.post(
    "/batch",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a batch inference job (ZIP of images)",
    description=(
        "Upload a ZIP archive of images and a model_id. "
        "Returns a job_id to poll for results."
    ),
)
async def predict_batch(
    db: Annotated[AsyncSession, Depends(get_db)],
    model_id: uuid.UUID = Form(..., description="Registered model UUID"),
    file: UploadFile = File(..., description="ZIP archive containing images"),
):
    if file.content_type not in {"application/zip", "application/x-zip-compressed"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Expected a ZIP archive.",
        )

    await _get_ready_model(db, model_id)

    zip_bytes = await file.read(MAX_BATCH_BYTES + 1)
    if len(zip_bytes) > MAX_BATCH_BYTES:
        raise HTTPException(status_code=413, detail="ZIP exceeds 500 MB limit.")

    job_id = str(uuid.uuid4())
    zip_key = f"batch_jobs/{job_id}/input.zip"
    s3_service.ensure_bucket(settings.models_bucket)
    s3_service.upload_bytes(zip_bytes, settings.models_bucket, zip_key, content_type="application/zip")

    from app.workers.train_worker import batch_predict_task
    batch_predict_task.apply_async(
        args=[job_id, str(model_id), settings.models_bucket, zip_key],
        task_id=job_id,
    )
    logger.info("Dispatched batch predict job %s for model %s", job_id, model_id)

    return BatchJobResponse(job_id=job_id, status="processing")


# ---------------------------------------------------------------------------
# GET /predict/batch/{job_id}  — poll batch job status
# ---------------------------------------------------------------------------

@router.get(
    "/batch/{job_id}",
    response_model=BatchResultResponse,
    summary="Poll batch inference job status / get results URL",
)
async def get_batch_result(job_id: str):
    bucket = settings.models_bucket
    results_key = f"batch_jobs/{job_id}/results.json"
    error_key   = f"batch_jobs/{job_id}/error.txt"

    client = s3_service.get_s3_client()

    try:
        obj = client.get_object(Bucket=bucket, Key=error_key)
        error_msg = obj["Body"].read().decode()
        return BatchResultResponse(job_id=job_id, status="failed", error=error_msg)
    except Exception:
        pass

    try:
        client.head_object(Bucket=bucket, Key=results_key)
        url = s3_service.generate_presigned_url(bucket, results_key, expires_in=3600)
        return BatchResultResponse(job_id=job_id, status="completed", results_url=url)
    except Exception:
        pass

    return BatchResultResponse(job_id=job_id, status="processing")
