import uuid
from enum import Enum as PyEnum

from sqlalchemy import Column, String, Float, DateTime, Text, Enum as SAEnum, func, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.db.session import Base


class ModelStatus(str, PyEnum):
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String(64), nullable=False, default="1.0.0")

    status = Column(
        SAEnum(ModelStatus, name="model_status"),
        nullable=False,
        default=ModelStatus.TRAINING,
        index=True,
    )

    base_model = Column(String(255), nullable=False)
    class_names = Column(JSON, nullable=True)   # ["cats","dogs"] or {"cats":0,...} (API uses dict today)

    # Metrics
    accuracy = Column(Float, nullable=True)
    training_time_seconds = Column(Float, nullable=True)

    # Lineage
    dataset_id = Column(UUID(as_uuid=True), nullable=True)
    training_job_id = Column(UUID(as_uuid=True), nullable=True)

    # Storage – S3 paths
    s3_bucket = Column(String(255), nullable=True)
    onnx_s3_key = Column(String(512), nullable=True)
    adapter_s3_key = Column(String(512), nullable=True)  # prefix or a marker object

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<MLModel id={self.id} name={self.name!r} status={self.status}>"
