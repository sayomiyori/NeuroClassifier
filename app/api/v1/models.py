import logging
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.ml_model import MLModel, ModelStatus
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
    num_classes: Optional[str]
    class_names: Optional[dict]
    accuracy: Optional[float]
    val_loss: Optional[float]
    metrics: Optional[dict]
    s3_bucket: Optional[str]
    s3_key: Optional[str]
    download_url: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class MLModelList(BaseModel):
    items: List[MLModelResponse]
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_response(m: MLModel, include_url: bool = False) -> MLModelResponse:
    download_url = None
    if include_url and m.s3_bucket and m.s3_key and m.status == ModelStatus.READY:
        try:
            download_url = s3_service.generate_presigned_url(m.s3_bucket, m.s3_key)
        except Exception:
            pass

    return MLModelResponse(
        id=m.id,
        name=m.name,
        description=m.description,
        version=m.version,
        status=m.status,
        base_model=m.base_model,
        num_classes=m.num_classes,
        class_names=m.class_names,
        accuracy=m.accuracy,
        val_loss=m.val_loss,
        metrics=m.metrics,
        s3_bucket=m.s3_bucket,
        s3_key=m.s3_key,
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
    if m.s3_bucket and m.s3_key:
        try:
            client = s3_service.get_s3_client()
            client.delete_object(Bucket=m.s3_bucket, Key=m.s3_key)
        except Exception as exc:
            logger.warning("Could not delete S3 artifact for model %s: %s", model_id, exc)
    await db.delete(m)
