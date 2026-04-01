import logging
import uuid
from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.ml_model import MLModel, ModelStatus
from app.models.training_job import TrainingJob
from app.services import s3 as s3_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Model Registry"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class MLModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    version: str
    status: ModelStatus
    base_model: str
    class_names: Optional[dict]
    accuracy: Optional[float]
    s3_bucket: Optional[str]
    onnx_s3_key: Optional[str]
    adapter_s3_key: Optional[str]
    download_url: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class MLModelList(BaseModel):
    items: List[MLModelResponse]
    total: int


class EpochMetric(BaseModel):
    epoch: int
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None


class ModelMetricsResponse(BaseModel):
    model_id: str
    metrics: Optional[List[EpochMetric]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_response(m: MLModel, include_url: bool = False) -> MLModelResponse:
    download_url = None
    if include_url and m.s3_bucket and m.onnx_s3_key and m.status == ModelStatus.READY:
        try:
            download_url = s3_service.generate_presigned_url(m.s3_bucket, m.onnx_s3_key)
        except Exception:
            pass

    return MLModelResponse(
        id=m.id,
        name=m.name,
        description=m.description,
        version=m.version,
        status=m.status,
        base_model=m.base_model,
        class_names=m.class_names,
        accuracy=m.accuracy,
        s3_bucket=m.s3_bucket,
        onnx_s3_key=m.onnx_s3_key,
        adapter_s3_key=m.adapter_s3_key,
        download_url=download_url,
        created_at=m.created_at.isoformat(),
    )


async def _get_model(db: AsyncSession, model_id: uuid.UUID) -> MLModel:
    result = await db.execute(select(MLModel).where(MLModel.id == model_id))
    m = result.scalar_one_or_none()
    if not m:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    return m


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=MLModelList,
    summary="List registered models",
)
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[ModelStatus] = None,
):
    query = select(MLModel).order_by(MLModel.created_at.desc()).offset(skip).limit(limit)
    if status_filter:
        query = query.where(MLModel.status == status_filter)
    result = await db.execute(query)
    models = list(result.scalars().all())
    return MLModelList(items=[_model_to_response(m) for m in models], total=len(models))


@router.get(
    "/{model_id}",
    response_model=MLModelResponse,
    summary="Get model details + presigned download URL",
)
async def get_model(
    model_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    m = await _get_model(db, model_id)
    return _model_to_response(m, include_url=True)


@router.get(
    "/{model_id}/metrics",
    response_model=ModelMetricsResponse,
    summary="Get model learning-curve metrics (if available)",
)
async def get_model_metrics(
    model_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    m = await _get_model(db, model_id)

    # Per-epoch metrics live on the associated TrainingJob.
    job_result = await db.execute(
        select(TrainingJob)
        .where(TrainingJob.ml_model_id == model_id)
        .order_by(TrainingJob.created_at.desc())
        .limit(1)
    )
    job: Optional[TrainingJob] = job_result.scalar_one_or_none()

    metrics: Optional[List[EpochMetric]] = None
    if job and job.metrics:
        metrics = [EpochMetric(**entry) for entry in job.metrics]

    return ModelMetricsResponse(model_id=str(m.id), metrics=metrics)


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a model from registry and MinIO",
)
async def delete_model(
    model_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    m = await _get_model(db, model_id)
    if m.s3_bucket:
        try:
            # Delete ONNX file
            if m.onnx_s3_key:
                client = s3_service.get_s3_client()
                client.delete_object(Bucket=m.s3_bucket, Key=m.onnx_s3_key)
            # Delete adapter prefix (folder-like)
            if m.adapter_s3_key:
                s3_service.delete_prefix(m.s3_bucket, m.adapter_s3_key)
        except Exception as exc:
            logger.warning("Could not delete S3 artifacts for model %s: %s", model_id, exc)
    await db.delete(m)
