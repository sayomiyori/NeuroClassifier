import io
import logging
import os
import random
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.models.dataset import Dataset, DatasetStatus
from app.services import s3 as s3_service

settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, Exception):
        return False


def _scan_extracted_dir(extract_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan extracted directory and return {class_name: [image_paths]}.
    Expected structure:
        <extract_dir>/
            cats/img1.jpg
            dogs/img2.jpg
    """
    classes: Dict[str, List[Path]] = {}

    for item in sorted(extract_dir.iterdir()):
        if not item.is_dir():
            continue
        class_name = item.name
        images = [
            p for p in sorted(item.rglob("*"))
            if p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS and p.is_file()
        ]
        valid_images = [p for p in images if _is_valid_image(p)]
        if valid_images:
            classes[class_name] = valid_images

    return classes


def _stratified_split(
    class_images: Dict[str, List[Path]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Split each class into train/val retaining class proportions."""
    rng = random.Random(seed)
    train: Dict[str, List[Path]] = {}
    val: Dict[str, List[Path]] = {}

    for cls, images in class_images.items():
        shuffled = images[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        # Ensure at least 1 sample ends up in both splits (when possible).
        split_idx = int(n * train_ratio)
        split_idx = max(1, split_idx)
        if n > 1:
            split_idx = min(split_idx, n - 1)
        train[cls] = shuffled[:split_idx]
        val[cls] = shuffled[split_idx:]

    return train, val


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """
    Extract ZIP into dest_dir while preventing zip-slip path traversal.
    """
    dest_dir = dest_dir.resolve()
    for member in zf.infolist():
        # Normalize to forward slashes, then let Path handle it.
        member_path = Path(member.filename)
        if member_path.is_absolute():
            raise ValueError("ZIP contains absolute paths, which is not allowed.")

        target_path = (dest_dir / member_path).resolve()
        if dest_dir not in target_path.parents and target_path != dest_dir:
            raise ValueError("ZIP contains illegal path traversal entries.")

        zf.extract(member, dest_dir)


# ---------------------------------------------------------------------------
# Upload orchestration
# ---------------------------------------------------------------------------

def _upload_split_to_s3(
    split: Dict[str, List[Path]],
    dataset_id: str,
    subset: str,  # "train" or "val"
) -> None:
    """Upload all images in a split to MinIO."""
    bucket = settings.datasets_bucket
    s3_service.ensure_bucket(bucket)

    for class_name, images in split.items():
        for img_path in images:
            key = f"{dataset_id}/{subset}/{class_name}/{img_path.name}"
            with open(img_path, "rb") as f:
                suffix = img_path.suffix.lower()
                content_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
                s3_service.upload_file(f, bucket, key, content_type=content_type)


def process_dataset_zip(
    zip_path: str,
    dataset_id: str,
    dataset_name: str,
) -> dict:
    """
    Synchronous processing function called from Celery worker.
    Returns metadata dict on success, raises on failure.
    """
    extract_dir = Path(tempfile.mkdtemp(prefix=f"nc_{dataset_id}_"))
    try:
        # 1. Extract zip
        logger.info("[%s] Extracting zip: %s", dataset_id, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, extract_dir)

        # Some zips have a top-level folder; unwrap it transparently
        items = list(extract_dir.iterdir())
        if len(items) == 1 and items[0].is_dir():
            extract_dir = items[0]

        # 2. Scan class structure
        class_images = _scan_extracted_dir(extract_dir)
        logger.info("[%s] Found classes: %s", dataset_id, list(class_images.keys()))

        # 3. Validate
        if len(class_images) < settings.min_classes:
            raise ValueError(
                f"Need at least {settings.min_classes} classes, found {len(class_images)}: "
                f"{list(class_images.keys())}"
            )

        small_classes = {
            cls: len(imgs)
            for cls, imgs in class_images.items()
            if len(imgs) < settings.min_images_per_class
        }
        if small_classes:
            raise ValueError(
                f"Classes with fewer than {settings.min_images_per_class} valid images: {small_classes}"
            )

        # 4. Stratified split
        train_split, val_split = _stratified_split(class_images, train_ratio=settings.train_split_ratio)

        # 5. Upload to MinIO
        logger.info("[%s] Uploading train split …", dataset_id)
        _upload_split_to_s3(train_split, dataset_id, "train")
        logger.info("[%s] Uploading val split …", dataset_id)
        _upload_split_to_s3(val_split, dataset_id, "val")

        # 6. Build metadata
        class_distribution = {cls: len(imgs) for cls, imgs in class_images.items()}
        train_count = sum(len(imgs) for imgs in train_split.values())
        val_count = sum(len(imgs) for imgs in val_split.values())

        return {
            "total_images": train_count + val_count,
            "num_classes": len(class_images),
            "class_names": class_distribution,
            "train_count": train_count,
            "val_count": val_count,
            "s3_bucket": settings.datasets_bucket,
            "s3_prefix": f"{dataset_id}/",
        }

    finally:
        shutil.rmtree(str(extract_dir.parent), ignore_errors=True)
        # Do not delete zip_path here: the Celery task may retry, and the ZIP
        # must remain available until the task is окончательно finished.


# ---------------------------------------------------------------------------
# Database helpers (async)
# ---------------------------------------------------------------------------

async def get_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> Dataset | None:
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    return result.scalar_one_or_none()


async def list_datasets(db: AsyncSession, skip: int = 0, limit: int = 100) -> List[Dataset]:
    result = await db.execute(
        select(Dataset).order_by(Dataset.created_at.desc()).offset(skip).limit(limit)
    )
    return list(result.scalars().all())


async def delete_dataset(db: AsyncSession, dataset: Dataset) -> None:
    """Remove dataset from MinIO then from PostgreSQL."""
    if dataset.s3_bucket and dataset.s3_prefix:
        deleted = s3_service.delete_prefix(dataset.s3_bucket, dataset.s3_prefix)
        logger.info("Deleted %d S3 objects for dataset %s", deleted, dataset.id)
    await db.delete(dataset)
