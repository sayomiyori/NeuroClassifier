import logging
import time
import uuid
from typing import Annotated, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.metrics import INFERENCE_DURATION_SECONDS, INFERENCE_REQUESTS_TOTAL
from app.models.ml_model import MLModel, ModelStatus
from app.services.inference import run_inference

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Inference"])

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    class_name: str
    probability: float


class PredictResponse(BaseModel):
    model_id: uuid.UUID
    model_name: str
    predictions: List[Prediction]
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{model_id}",
    response_model=PredictResponse,
    summary="Run inference on an uploaded image",
    description=(
        "Upload an image (JPEG/PNG) and get class predictions from a registered ONNX model. "
        "Returns top-5 classes with confidence scores."
    ),
)
async def predict(
    model_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(..., description="Image file (JPEG / PNG)"),
    top_k: int = 5,
):
    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type: {file.content_type}. Allowed: {allowed}",
        )

    # Fetch model from registry
    result = await db.execute(select(MLModel).where(MLModel.id == model_id))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model {model_id} is not ready (status: {model.status}).",
        )
    if not model.s3_bucket or not model.s3_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model artifact not available in storage.",
        )

    # Read image bytes
    image_bytes = await file.read(MAX_IMAGE_SIZE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds 10 MB limit.",
        )

    class_names: List[str] = (
        list(model.class_names.keys()) if model.class_names else []
    )
    if not class_names:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model has no class names registered.",
        )

    # Run inference
    start = time.perf_counter()
    try:
        raw_predictions = run_inference(
            image_bytes=image_bytes,
            model_id=str(model_id),
            s3_bucket=model.s3_bucket,
            s3_key=model.s3_key,
            class_names=class_names,
            top_k=top_k,
        )
        INFERENCE_REQUESTS_TOTAL.labels(model_id=str(model_id), status="success").inc()
    except Exception as exc:
        INFERENCE_REQUESTS_TOTAL.labels(model_id=str(model_id), status="error").inc()
        logger.exception("Inference failed for model %s", model_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    INFERENCE_DURATION_SECONDS.labels(model_id=str(model_id)).observe(elapsed_ms / 1000)

    predictions = [
        Prediction(class_name=p["class"], probability=round(p["probability"], 6))
        for p in raw_predictions
    ]

    return PredictResponse(
        model_id=model_id,
        model_name=model.name,
        predictions=predictions,
        inference_time_ms=round(elapsed_ms, 2),
    )
