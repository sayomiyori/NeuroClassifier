import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, Enum as SAEnum, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class JobStatus(str, PyEnum):
    QUEUED = "queued"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    ml_model_id = Column(UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="SET NULL"), nullable=True, index=True)

    status = Column(
        SAEnum(JobStatus, name="job_status"),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )

    # Celery task ID for tracking
    celery_task_id = Column(String(255), nullable=True, unique=True)

    # Config (full training config)
    base_model = Column(String(255), nullable=False, default="google/vit-base-patch16-224")
    config = Column(JSONB, nullable=True)  # {"lora_rank": 8, "epochs": 5, ...}

    # Progress
    progress_pct = Column(Float, nullable=True)
    epochs_completed = Column(Integer, nullable=True)
    metrics = Column(JSONB, nullable=True)  # [{"epoch": 1, "train_loss": ..., "val_loss": ..., "val_accuracy": ...}]

    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", foreign_keys=[dataset_id], lazy="select")
    ml_model = relationship("MLModel", foreign_keys=[ml_model_id], lazy="select")
