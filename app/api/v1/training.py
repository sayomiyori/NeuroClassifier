import logging
import uuid
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.metrics import ACTIVE_TRAINING_JOBS
from app.models.training_job import TrainingJob, JobStatus
from app.workers.train_worker import train_model_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["Training"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TrainingJobCreate(BaseModel):
    dataset_id: uuid.UUID
    base_model: str = Field(default="mobilenet_v2", examples=["mobilenet_v2", "efficientnet_b0"])
    hyperparams: Optional[dict] = Field(
        default=None,
        examples=[{"lr": 1e-4, "epochs": 10, "batch_size": 32}],
    )


class TrainingJobResponse(BaseModel):
    id: uuid.UUID
    dataset_id: Optional[uuid.UUID]
    status: JobStatus
    base_model: str
    hyperparams: Optional[dict]
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    train_loss: Optional[float]
    val_accuracy: Optional[float]
    error_message: Optional[str]
    celery_task_id: Optional[str]
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]

    class Config:
        from_attributes = True


class TrainingJobList(BaseModel):
    items: List[TrainingJobResponse]
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_to_response(j: TrainingJob) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=j.id,
        dataset_id=j.dataset_id,
        status=j.status,
        base_model=j.base_model,
        hyperparams=j.hyperparams,
        current_epoch=j.current_epoch,
        total_epochs=j.total_epochs,
        train_loss=j.train_loss,
        val_accuracy=j.val_accuracy,
        error_message=j.error_message,
        celery_task_id=j.celery_task_id,
        created_at=j.created_at.isoformat(),
        started_at=j.started_at.isoformat() if j.started_at else None,
        finished_at=j.finished_at.isoformat() if j.finished_at else None,
    )


async def _get_job(db: AsyncSession, job_id: uuid.UUID) -> TrainingJob:
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"TrainingJob {job_id} not found.")
    return job


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a training job",
)
async def start_training(
    payload: TrainingJobCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    job = TrainingJob(
        dataset_id=payload.dataset_id,
        status=JobStatus.PENDING,
        base_model=payload.base_model,
        hyperparams=payload.hyperparams,
    )
    db.add(job)
    await db.flush()

    # Dispatch Celery task
    train_model_task.apply_async(args=[str(job.id)], task_id=str(uuid.uuid4()))
    ACTIVE_TRAINING_JOBS.inc()

    logger.info("Dispatched training job %s", job.id)
    return _job_to_response(job)


@router.get(
    "",
    response_model=TrainingJobList,
    summary="List training jobs",
)
async def list_jobs(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
):
    result = await db.execute(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).offset(skip).limit(limit)
    )
    jobs = list(result.scalars().all())
    return TrainingJobList(items=[_job_to_response(j) for j in jobs], total=len(jobs))


@router.get(
    "/{job_id}",
    response_model=TrainingJobResponse,
    summary="Get training job status",
)
async def get_job(
    job_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    job = await _get_job(db, job_id)
    return _job_to_response(job)


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel / delete a training job",
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
    ACTIVE_TRAINING_JOBS.dec()
