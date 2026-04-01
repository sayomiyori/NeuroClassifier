import logging
import uuid
from typing import Annotated, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.dataset import Dataset, DatasetStatus
from app.models.training_job import TrainingJob, JobStatus
from app.workers.train_worker import train_model_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/train", tags=["Training"])


class TrainConfig(BaseModel):
    base_model: str = Field(default="google/vit-base-patch16-224")
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4


class TrainRequest(BaseModel):
    dataset_id: uuid.UUID
    config: Optional[TrainConfig] = None


class TrainCreateResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus


class TrainJobResponse(BaseModel):
    id: uuid.UUID
    dataset_id: Optional[uuid.UUID]
    ml_model_id: Optional[uuid.UUID]
    status: JobStatus
    base_model: str
    config: Optional[dict]
    progress_pct: Optional[float]
    epochs_completed: Optional[int]
    metrics: Optional[list[dict]]
    error_message: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    class Config:
        from_attributes = True


class TrainJobList(BaseModel):
    items: List[TrainJobResponse]
    total: int


def _job_to_response(j: TrainingJob) -> TrainJobResponse:
    return TrainJobResponse(
        id=j.id,
        dataset_id=j.dataset_id,
        ml_model_id=j.ml_model_id,
        status=j.status,
        base_model=j.base_model,
        config=j.config,
        progress_pct=j.progress_pct,
        epochs_completed=j.epochs_completed,
        metrics=j.metrics,
        error_message=j.error_message,
        created_at=j.created_at.isoformat(),
        started_at=j.started_at.isoformat() if j.started_at else None,
        completed_at=j.completed_at.isoformat() if j.completed_at else None,
    )


async def _get_job(db: AsyncSession, job_id: uuid.UUID) -> TrainingJob:
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"TrainingJob {job_id} not found.")
    return job


@router.post(
    "",
    response_model=TrainCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start training pipeline (LoRA)",
)
async def start_training(
    payload: TrainRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    ds = await db.get(Dataset, payload.dataset_id)
    if not ds or ds.status != DatasetStatus.READY:
        raise HTTPException(status_code=400, detail="Dataset must exist and be READY before training.")

    cfg = (payload.config or TrainConfig()).model_dump()
    job = TrainingJob(
        dataset_id=payload.dataset_id,
        status=JobStatus.QUEUED,
        base_model=cfg["base_model"],
        config=cfg,
        progress_pct=0.0,
        epochs_completed=0,
        metrics=[],
    )
    db.add(job)
    await db.flush()

    train_model_task.apply_async(
        args=[str(job.id)],
        task_id=str(uuid.uuid4()),
    )

    logger.info("Dispatched training job %s for dataset %s", job.id, payload.dataset_id)
    return TrainCreateResponse(job_id=job.id, status=job.status)


@router.get("", response_model=TrainJobList, summary="List training jobs")
async def list_jobs(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
):
    result = await db.execute(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).offset(skip).limit(limit)
    )
    jobs = list(result.scalars().all())
    return TrainJobList(items=[_job_to_response(j) for j in jobs], total=len(jobs))


@router.get("/{job_id}", response_model=TrainJobResponse, summary="Get training job")
async def get_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    job = await _get_job(db, job_id)
    return _job_to_response(job)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel a training job",
)
async def cancel_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    job = await _get_job(db, job_id)
    if job.celery_task_id:
        from app.workers.train_worker import celery_app

        celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
    job.status = JobStatus.CANCELLED
    await db.flush()

