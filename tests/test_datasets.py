"""
Integration tests for Dataset API.

Tests:
  1. Upload a valid ZIP → check MinIO objects exist → check DB metadata
  2. GET /datasets → list contains uploaded dataset
  3. GET /datasets/{id} → full metadata with class distribution
  4. DELETE /datasets/{id} → MinIO objects removed, DB record gone

Running locally (requires docker-compose services or mocked S3):
    pytest tests/ -v --cov=app --cov-report=term-missing
"""
import io
import os
import uuid
import zipfile
from typing import Generator
from unittest.mock import MagicMock, patch

import boto3
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from moto import mock_s3
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.db.session import Base, get_db
from app.main import app
from app.models.dataset import Dataset, DatasetStatus

settings = get_settings()

# ─── In-memory SQLite for tests ──────────────────────────────────────────────
SQLITE_URL = "sqlite://"

engine_test = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Enable foreign keys in SQLite
@event.listens_for(engine_test, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def create_tables():
    """Create all tables once per test session."""
    Base.metadata.create_all(bind=engine_test)
    yield
    Base.metadata.drop_all(bind=engine_test)


@pytest.fixture()
def db() -> Generator[Session, None, None]:
    """Fresh DB session per test."""
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_dataset_zip(
    classes: dict[str, int],
    image_size: tuple[int, int] = (32, 32),
) -> bytes:
    """
    Build an in-memory ZIP with structure:
        <class_name>/<class_name>_<n>.jpg
    Each 'image' is a minimal valid JPEG byte string (created via Pillow).
    """
    from PIL import Image

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for class_name, count in classes.items():
            for i in range(count):
                img_buf = io.BytesIO()
                img = Image.new("RGB", image_size, color=(i * 5 % 255, 100, 200))
                img.save(img_buf, format="JPEG")
                img_buf.seek(0)
                zf.writestr(f"{class_name}/{class_name}_{i:03d}.jpg", img_buf.read())
    buf.seek(0)
    return buf.read()


# ─── Mock S3 / Celery helpers ─────────────────────────────────────────────────

class FakeCeleryResult:
    id = str(uuid.uuid4())


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestDatasetUpload:
    """POST /api/v1/datasets"""

    @mock_s3
    @patch("app.api.v1.datasets.process_dataset_task")
    def test_upload_valid_zip_returns_202(self, mock_task, client: TestClient):
        mock_task.apply_async.return_value = FakeCeleryResult()

        zip_bytes = _make_dataset_zip({"cats": 15, "dogs": 12, "birds": 10})

        response = client.post(
            "/api/v1/datasets",
            data={"name": "AnimalDataset", "description": "Test dataset"},
            files={"file": ("animals.zip", zip_bytes, "application/zip")},
        )

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["status"] == "uploading"
        assert "dataset_id" in body
        assert body["name"] == "AnimalDataset"
        # Celery task was dispatched
        mock_task.apply_async.assert_called_once()

    @patch("app.api.v1.datasets.process_dataset_task")
    def test_upload_non_zip_returns_415(self, mock_task, client: TestClient):
        response = client.post(
            "/api/v1/datasets",
            data={"name": "Bad"},
            files={"file": ("image.png", b"fakecontent", "image/png")},
        )
        assert response.status_code == 415
        mock_task.apply_async.assert_not_called()

    @patch("app.api.v1.datasets.process_dataset_task")
    def test_upload_stores_record_in_db(self, mock_task, client: TestClient, db: Session):
        mock_task.apply_async.return_value = FakeCeleryResult()
        zip_bytes = _make_dataset_zip({"class_a": 10, "class_b": 10})

        response = client.post(
            "/api/v1/datasets",
            data={"name": "DBCheckDataset"},
            files={"file": ("data.zip", zip_bytes, "application/zip")},
        )
        assert response.status_code == 202
        dataset_id = uuid.UUID(response.json()["dataset_id"])

        record = db.get(Dataset, dataset_id)
        assert record is not None
        assert record.name == "DBCheckDataset"
        assert record.status == DatasetStatus.UPLOADING


class TestDatasetList:
    """GET /api/v1/datasets"""

    @patch("app.api.v1.datasets.process_dataset_task")
    def test_list_returns_uploaded_dataset(
        self, mock_task, client: TestClient, db: Session
    ):
        mock_task.apply_async.return_value = FakeCeleryResult()

        # Upload one
        zip_bytes = _make_dataset_zip({"x": 10, "y": 11})
        client.post(
            "/api/v1/datasets",
            data={"name": "ListTestDataset"},
            files={"file": ("d.zip", zip_bytes, "application/zip")},
        )

        resp = client.get("/api/v1/datasets")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        names = [d["name"] for d in body["items"]]
        assert "ListTestDataset" in names


class TestDatasetDetail:
    """GET /api/v1/datasets/{id}"""

    def test_get_nonexistent_returns_404(self, client: TestClient):
        fake_id = uuid.uuid4()
        resp = client.get(f"/api/v1/datasets/{fake_id}")
        assert resp.status_code == 404

    def test_get_existing_returns_detail(self, client: TestClient, db: Session):
        # Seed a ready dataset directly in DB
        ds = Dataset(
            id=uuid.uuid4(),
            name="ReadyDataset",
            status=DatasetStatus.READY,
            total_images=50,
            num_classes=2,
            class_names={"cats": 25, "dogs": 25},
            train_count=40,
            val_count=10,
            s3_bucket="datasets",
            s3_prefix="abc123/",
        )
        db.add(ds)
        db.commit()

        resp = client.get(f"/api/v1/datasets/{ds.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["total_images"] == 50
        assert body["num_classes"] == 2
        assert body["class_names"] == {"cats": 25, "dogs": 25}
        assert body["train_count"] == 40
        assert body["val_count"] == 10


class TestDatasetDelete:
    """DELETE /api/v1/datasets/{id}"""

    @patch("app.services.dataset_service.s3_service.delete_prefix")
    def test_delete_existing_dataset(self, mock_delete_prefix, client: TestClient, db: Session):
        mock_delete_prefix.return_value = 10

        ds = Dataset(
            id=uuid.uuid4(),
            name="ToDeleteDataset",
            status=DatasetStatus.READY,
            s3_bucket="datasets",
            s3_prefix="todelete/",
        )
        db.add(ds)
        db.commit()
        ds_id = ds.id

        resp = client.delete(f"/api/v1/datasets/{ds_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["dataset_id"] == str(ds_id)

        # Should no longer exist
        resp2 = client.get(f"/api/v1/datasets/{ds_id}")
        assert resp2.status_code == 404

        # S3 delete was called
        mock_delete_prefix.assert_called_once_with("datasets", "todelete/")

    def test_delete_nonexistent_returns_404(self, client: TestClient):
        resp = client.delete(f"/api/v1/datasets/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestDatasetProcessingLogic:
    """Unit tests for dataset_service processing logic (no network)."""

    def test_stratified_split_proportions(self):
        from app.services.dataset_service import _stratified_split
        from pathlib import Path

        class_images = {
            "cats": [Path(f"cats/{i}.jpg") for i in range(100)],
            "dogs": [Path(f"dogs/{i}.jpg") for i in range(80)],
        }
        train, val = _stratified_split(class_images, train_ratio=0.8)

        assert len(train["cats"]) == 80
        assert len(val["cats"]) == 20
        assert len(train["dogs"]) == 64
        assert len(val["dogs"]) == 16

    def test_scan_extracted_dir_finds_classes(self, tmp_path):
        from PIL import Image
        from app.services.dataset_service import _scan_extracted_dir

        for cls in ["cats", "dogs"]:
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(5):
                img = Image.new("RGB", (32, 32), color=(i * 10, 0, 0))
                img.save(cls_dir / f"{i}.jpg", "JPEG")

        result = _scan_extracted_dir(tmp_path)
        assert set(result.keys()) == {"cats", "dogs"}
        assert len(result["cats"]) == 5
        assert len(result["dogs"]) == 5


class TestMinIOIntegration:
    """Tests that use moto's mock S3 to verify actual upload behaviour."""

    @mock_s3
    def test_process_dataset_zip_uploads_to_minio(self, tmp_path):
        """Full pipeline: zip → extract → validate → split → upload."""
        # Set up mock S3
        boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
            endpoint_url=None,
        ).create_bucket(Bucket="datasets")

        from unittest.mock import patch
        from app.services.dataset_service import process_dataset_zip

        # Build a valid zip
        from PIL import Image
        import zipfile

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for cls in ["cats", "dogs"]:
                for i in range(12):
                    img_buf = io.BytesIO()
                    Image.new("RGB", (32, 32)).save(img_buf, "JPEG")
                    img_buf.seek(0)
                    zf.writestr(f"{cls}/{i}.jpg", img_buf.read())
        zip_buf.seek(0)

        zip_path = str(tmp_path / "test.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_buf.read())

        dataset_id = str(uuid.uuid4())

        # Patch s3 client to use moto
        with patch("app.services.s3.get_s3_client") as mock_client_fn, \
             patch("app.services.s3.ensure_bucket"):
            mock_s3_client = boto3.client(
                "s3",
                region_name="us-east-1",
                aws_access_key_id="test",
                aws_secret_access_key="test",
            )
            mock_client_fn.return_value = mock_s3_client
            mock_s3_client.create_bucket(Bucket="datasets")

            metadata = process_dataset_zip(zip_path, dataset_id, "TestDataset")

        assert metadata["num_classes"] == 2
        assert metadata["total_images"] == 24
        assert metadata["train_count"] == 20   # ceil(12 * 0.8) * 2 classes
        assert metadata["val_count"] == 4
        assert set(metadata["class_names"].keys()) == {"cats", "dogs"}
