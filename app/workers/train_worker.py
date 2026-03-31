import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings
from app.models.dataset import Dataset, DatasetStatus
from app.models.training_job import TrainingJob, JobStatus
from app.services.dataset_service import process_dataset_zip

settings = get_settings()
logger = logging.getLogger(__name__)

# Celery app
celery_app = Celery(
    "neuroclassifier",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Sync engine for Celery (not async)
sync_engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
SyncSession = sessionmaker(bind=sync_engine)


def get_sync_db() -> Session:
    return SyncSession()


# ---------------------------------------------------------------------------
# Dataset processing task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="process_dataset", max_retries=3, default_retry_delay=10)
def process_dataset_task(self, dataset_id: str, zip_path: str, dataset_name: str):
    """
    Background task: extract zip, validate, split, upload to MinIO, update DB.
    """
    db = get_sync_db()
    try:
        # Mark as processing
        dataset = db.get(Dataset, uuid.UUID(dataset_id))
        if not dataset:
            logger.error("Dataset %s not found", dataset_id)
            return

        dataset.status = DatasetStatus.PROCESSING
        db.commit()

        # Do the actual work
        metadata = process_dataset_zip(zip_path, dataset_id, dataset_name)

        # Update dataset with results
        dataset = db.get(Dataset, uuid.UUID(dataset_id))
        dataset.status = DatasetStatus.READY
        dataset.total_images = metadata["total_images"]
        dataset.num_classes = metadata["num_classes"]
        dataset.class_names = metadata["class_names"]
        dataset.train_count = metadata["train_count"]
        dataset.val_count = metadata["val_count"]
        dataset.s3_bucket = metadata["s3_bucket"]
        dataset.s3_prefix = metadata["s3_prefix"]
        db.commit()
        logger.info("Dataset %s is READY", dataset_id)

        # Cleanup ZIP only after success
        try:
            if zip_path and os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass

    except ValueError as exc:
        # Validation errors should not be retried.
        logger.exception("Dataset %s validation failed", dataset_id)
        try:
            dataset = db.get(Dataset, uuid.UUID(dataset_id))
            if dataset:
                dataset.status = DatasetStatus.FAILED
                dataset.error_message = str(exc)
                db.commit()
        except Exception:
            pass
        try:
            if zip_path and os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception:
            pass
        return

    except Exception as exc:
        logger.exception("Failed to process dataset %s", dataset_id)
        try:
            dataset = db.get(Dataset, uuid.UUID(dataset_id))
            if dataset:
                dataset.status = DatasetStatus.FAILED
                dataset.error_message = str(exc)
                db.commit()
        except Exception:
            pass
        # Do not delete ZIP before retry; it must remain for the next attempt.
        raise self.retry(exc=exc)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Training task (LoRA fine-tune stub – extend with real trainer)
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="train_model", max_retries=1)
def train_model_task(self, job_id: str):
    """
    Background training task.
    Stub implementation – replace body with real LoRA fine-tuning logic.
    """
    db = get_sync_db()
    try:
        job = db.get(TrainingJob, uuid.UUID(job_id))
        if not job:
            logger.error("TrainingJob %s not found", job_id)
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.celery_task_id = self.request.id
        db.commit()

        logger.info("Training job %s started", job_id)

        # TODO: implement LoRA fine-tuning here
        # from app.workers.lora_trainer import run_lora
        # run_lora(job, db)

        job = db.get(TrainingJob, uuid.UUID(job_id))
        job.status = JobStatus.COMPLETED
        job.finished_at = datetime.utcnow()
        db.commit()
        logger.info("Training job %s completed", job_id)

    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        try:
            job = db.get(TrainingJob, uuid.UUID(job_id))
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                job.finished_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
        raise
    finally:
        db.close()
