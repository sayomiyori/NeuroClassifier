import uuid
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, DateTime, Text, Enum as SAEnum, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.db.session import Base


class DatasetStatus(str, PyEnum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(
        SAEnum(DatasetStatus, name="dataset_status"),
        nullable=False,
        default=DatasetStatus.UPLOADING,
        index=True,
    )

    # Image statistics
    total_images = Column(Integer, nullable=True)
    num_classes = Column(Integer, nullable=True)
    class_names = Column(JSONB, nullable=True)          # {"cats": 150, "dogs": 140}
    train_count = Column(Integer, nullable=True)
    val_count = Column(Integer, nullable=True)

    # Storage
    s3_bucket = Column(String(255), nullable=True)
    s3_prefix = Column(String(512), nullable=True)      # e.g. "<dataset_id>/"

    # Error info
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name={self.name!r} status={self.status}>"
