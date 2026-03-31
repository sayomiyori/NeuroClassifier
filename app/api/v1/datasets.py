import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.session import get_db
from app.metrics import DATASET_UPLOADS_TOTAL
from app.models.dataset import Dataset, DatasetStatus
from app.services.dataset_service import delete_dataset, get_dataset, list_datasets
from app.workers.train_worker import process_dataset_task

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DatasetCreateResponse(BaseModel):
    dataset_id: uuid.UUID
    name: str
    status: DatasetStatus
    message: str


class ClassDistribution(BaseModel):
    class_name: str
    total: int
    train: int
    val: int


class DatasetDetail(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    status: DatasetStatus
    total_images: Optional[int]
    num_classes: Optional[int]
    class_names: Optional[dict]
    train_count: Optional[int]
    val_count: Optional[int]
    s3_bucket: Optional[str]
    s3_prefix: Optional[str]
    error_message: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class DatasetListItem(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    status: DatasetStatus
    total_images: Optional[int]
    num_classes: Optional[int]
    created_at: str

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    items: List[DatasetListItem]
    total: int


class DeleteResponse(BaseModel):
    message: str
    dataset_id: uuid.UUID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_to_detail(d: Dataset) -> DatasetDetail:
    return DatasetDetail(
        id=d.id,
        name=d.name,
        description=d.description,
        status=d.status,
        total_images=d.total_images,
        num_classes=d.num_classes,
        class_names=d.class_names,
        train_count=d.train_count,
        val_count=d.val_count,
        s3_bucket=d.s3_bucket,
        s3_prefix=d.s3_prefix,
        error_message=d.error_message,
        created_at=d.created_at.isoformat(),
    )


def _dataset_to_list_item(d: Dataset) -> DatasetListItem:
    return DatasetListItem(
        id=d.id,
        name=d.name,
        description=d.description,
        status=d.status,
        total_images=d.total_images,
        num_classes=d.num_classes,
        created_at=d.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=DatasetCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a dataset ZIP file",
    description=(
        "Upload a ZIP archive with structure `<class_name>/<image_files>`. "
        "Minimum 2 classes, minimum 10 images per class. "
        "The archive is extracted, validated, split 80/20 and uploaded to MinIO asynchronously."
    ),
)
async def upload_dataset(
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(..., description="ZIP archive containing class folders"),
    name: str = Form(..., description="Human-readable dataset name"),
    description: str = Form("", description="Optional description"),
):
    # Basic mime check
    if file.content_type not in ("application/zip", "application/x-zip-compressed", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".zip"):
            DATASET_UPLOADS_TOTAL.labels(status="rejected_format").inc()
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only ZIP archives are accepted.",
            )

    # Size guard (stream to temp file)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    suffix = ".zip"
    shared_tmp_dir = Path(settings.shared_tmp_dir)
    shared_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=str(shared_tmp_dir))
    total_bytes = 0
    try:
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    DATASET_UPLOADS_TOTAL.labels(status="rejected_size").inc()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {settings.max_upload_size_mb} MB.",
                    )
                tmp_file.write(chunk)
    except HTTPException:
        os.unlink(tmp_path)
        raise

    # Create DB record
    dataset_id = uuid.uuid4()
    dataset = Dataset(
        id=dataset_id,
        name=name.strip(),
        description=description.strip() or None,
        status=DatasetStatus.UPLOADING,
    )
    db.add(dataset)
    await db.flush()  # get id without commit so we hold the session open

    logger.info("Created dataset record %s, dispatching Celery task", dataset_id)

    # Dispatch Celery task (non-blocking)
    process_dataset_task.apply_async(
        args=[str(dataset_id), tmp_path, name],
        task_id=str(uuid.uuid4()),
    )

    DATASET_UPLOADS_TOTAL.labels(status="accepted").inc()

    return DatasetCreateResponse(
        dataset_id=dataset_id,
        name=name,
        status=DatasetStatus.UPLOADING,
        message="Dataset upload accepted. Processing in background — poll GET /datasets/{id} for status.",
    )


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List all datasets",
)
async def list_all_datasets(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = 0,
    limit: int = 50,
):
    datasets = await list_datasets(db, skip=skip, limit=limit)
    return DatasetListResponse(
        items=[_dataset_to_list_item(d) for d in datasets],
        total=len(datasets),
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetDetail,
    summary="Get dataset metadata and class distribution",
)
async def get_dataset_detail(
    dataset_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    dataset = await get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")
    return _dataset_to_detail(dataset)


@router.delete(
    "/{dataset_id}",
    response_model=DeleteResponse,
    summary="Delete a dataset (MinIO + PostgreSQL)",
)
async def delete_dataset_endpoint(
    dataset_id: uuid.UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    dataset = await get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found.")

    await delete_dataset(db, dataset)
    return DeleteResponse(message="Dataset deleted successfully.", dataset_id=dataset_id)
