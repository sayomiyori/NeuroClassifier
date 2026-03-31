import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, Enum as SAEnum, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.db.session import Base


class JobStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
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
        default=JobStatus.PENDING,
        index=True,
    )

    # Celery task ID for tracking
    celery_task_id = Column(String(255), nullable=True, unique=True)

    # Config
    base_model = Column(String(255), nullable=False, default="mobilenet_v2")
    hyperparams = Column(JSONB, nullable=True)   # {"lr": 1e-4, "epochs": 10, ...}

    # Progress
    current_epoch = Column(Integer, nullable=True)
    total_epochs = Column(Integer, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    logs = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    dataset = relationship("Dataset", foreign_keys=[dataset_id], lazy="select")
    ml_model = relationship("MLModel", foreign_keys=[ml_model_id], lazy="select")
