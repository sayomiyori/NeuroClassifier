import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import get_settings
from app.models.dataset import Dataset, DatasetStatus
from app.models.training_job import TrainingJob, JobStatus
from app.models.ml_model import MLModel, ModelStatus
from app.services.dataset_service import process_dataset_zip
from app.metrics import TRAIN_DURATION_SECONDS, MODELS_TRAINED_TOTAL, ACTIVE_TRAINING_JOBS, DATASETS_TOTAL

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
        DATASETS_TOTAL.inc()
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
    LoRA fine-tuning pipeline (lazy-imports heavy ML deps).
    """
    db = get_sync_db()
    try:
        job = db.get(TrainingJob, uuid.UUID(job_id))
        if not job:
            logger.error("TrainingJob %s not found", job_id)
            return

        job.status = JobStatus.TRAINING
        job.started_at = datetime.utcnow()
        job.celery_task_id = self.request.id
        db.commit()

        cfg = (job.config or {}) if isinstance(job.config, dict) else (job.config or {})
        base_model = cfg.get("base_model", job.base_model or "google/vit-base-patch16-224")
        epochs = int(cfg.get("epochs", 5))
        batch_size = int(cfg.get("batch_size", 8))
        learning_rate = float(cfg.get("learning_rate", 1e-4))
        lora_rank = int(cfg.get("lora_rank", 8))
        lora_alpha = int(cfg.get("lora_alpha", 16))
        lora_dropout = float(cfg.get("lora_dropout", 0.1))

        dataset = db.get(Dataset, job.dataset_id) if job.dataset_id else None
        if not dataset or dataset.status != DatasetStatus.READY or not dataset.s3_bucket or not dataset.s3_prefix:
            raise ValueError("Dataset is not READY or has no S3 location.")

        # Lazy imports (keep API lightweight until training is invoked)
        try:
            import torch
            from torch.utils.data import DataLoader
            from torch import nn
            from torchvision import transforms
            from torchvision.datasets import ImageFolder
            from transformers import AutoModelForImageClassification
            from peft import LoraConfig, get_peft_model
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Training dependencies are missing. Install torch/torchvision/transformers/peft."
            ) from e

        from app.services import s3 as s3_service

        t0 = time.time()
        work_dir = tempfile.mkdtemp(prefix=f"nc_train_{job_id}_")
        train_dir = os.path.join(work_dir, "train")
        val_dir = os.path.join(work_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        def _download_prefix(prefix: str, out_dir: str) -> None:
            client = s3_service.get_s3_client()
            objs = s3_service.list_objects(dataset.s3_bucket, prefix)
            for obj in objs:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel = key[len(prefix):].lstrip("/")
                local_path = os.path.join(out_dir, rel)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    client.download_fileobj(dataset.s3_bucket, key, f)

        # 1) Download dataset from MinIO
        _download_prefix(f"{dataset.id}/train/", train_dir)
        _download_prefix(f"{dataset.id}/val/", val_dir)

        # 2) ImageFolder datasets
        tfm = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        train_ds = ImageFolder(train_dir, transform=tfm)
        val_ds = ImageFolder(val_dir, transform=tfm)
        class_names = train_ds.classes

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 3) Load pretrained model
        model = AutoModelForImageClassification.from_pretrained(
            base_model,
            num_labels=len(class_names),
            ignore_mismatched_sizes=True,
        )

        # 4) Apply LoRA via PEFT
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=lora_dropout,
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 5) Train loop
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        job.metrics = []
        job.progress_pct = 0.0
        job.epochs_completed = 0
        db.commit()

        def _eval() -> tuple[float, float]:
            model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    out = model(pixel_values=x)
                    logits = out.logits
                    loss = loss_fn(logits, y)
                    total_loss += float(loss.item()) * x.size(0)
                    preds = logits.argmax(dim=1)
                    correct += int((preds == y).sum().item())
                    total += int(y.numel())
            return (total_loss / max(1, total), correct / max(1, total))

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            seen = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                out = model(pixel_values=x)
                logits = out.logits
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                running += float(loss.item()) * x.size(0)
                seen += int(y.numel())

            train_loss = running / max(1, seen)
            val_loss, val_acc = _eval()

            metrics_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            job.metrics = list(job.metrics or []) + [metrics_row]
            job.epochs_completed = epoch
            job.progress_pct = float(epoch) / float(epochs) * 100.0
            db.commit()

        # 6) Persist artifacts (adapter + ONNX) to MinIO and model metadata to DB
        model_id = uuid.uuid4()
        adapter_dir = os.path.join(work_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)

        # Upload adapter directory
        bucket = settings.models_bucket
        s3_service.ensure_bucket(bucket)
        adapter_prefix = f"{model_id}/adapter/"
        client = s3_service.get_s3_client()
        for root, _, files in os.walk(adapter_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, adapter_dir).replace("\\", "/")
                key = adapter_prefix + rel
                with open(full, "rb") as f:
                    client.upload_fileobj(f, bucket, key)

        # Export to ONNX
        merged = model.merge_and_unload()
        merged.eval()
        dummy = torch.randn(1, 3, 224, 224, device=device)
        onnx_path = os.path.join(work_dir, "model.onnx")
        torch.onnx.export(
            merged,
            dummy,
            onnx_path,
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        )
        onnx_key = f"{model_id}/model.onnx"
        with open(onnx_path, "rb") as f:
            client.upload_fileobj(f, bucket, onnx_key, ExtraArgs={"ContentType": "application/onnx"})
        # Upload external data file if torch created one (large models)
        onnx_data_path = onnx_path + ".data"
        if os.path.exists(onnx_data_path):
            with open(onnx_data_path, "rb") as f:
                client.upload_fileobj(f, bucket, onnx_key + ".data")

        # Save MLModel
        m = MLModel(
            id=model_id,
            name=f"model_{model_id}",
            description=None,
            version="1.0.0",
            status=ModelStatus.READY,
            base_model=base_model,
            dataset_id=job.dataset_id,
            training_job_id=job.id,
            accuracy=float((job.metrics or [])[-1]["val_accuracy"]) if job.metrics else None,
            training_time_seconds=float(time.time() - t0),
            class_names={name: i for i, name in enumerate(class_names)},
            s3_bucket=bucket,
            onnx_s3_key=onnx_key,
            adapter_s3_key=adapter_prefix,
        )
        db.add(m)
        db.commit()

        job.ml_model_id = m.id
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        db.commit()

        MODELS_TRAINED_TOTAL.inc()
        TRAIN_DURATION_SECONDS.observe(float(time.time() - t0))
        ACTIVE_TRAINING_JOBS.dec()
        logger.info("Training job %s completed; model=%s", job_id, m.id)

    except Exception as exc:
        logger.exception("Training job %s failed", job_id)
        try:
            job = db.get(TrainingJob, uuid.UUID(job_id))
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
        ACTIVE_TRAINING_JOBS.dec()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Batch inference task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="batch_predict", max_retries=1)
def batch_predict_task(self, job_id: str, model_id: str, bucket: str, zip_key: str):
    """
    Download a ZIP of images from MinIO, run ONNX inference on each file,
    write results.json back to MinIO under batch_jobs/{job_id}/.
    """
    import io
    import zipfile

    from app.services import s3 as s3_service
    from app.services.inference import predict as run_predict
    from app.models.ml_model import MLModel, ModelStatus

    db = get_sync_db()
    try:
        model = db.get(MLModel, uuid.UUID(model_id))
        if not model or model.status != ModelStatus.READY:
            raise ValueError(f"Model {model_id} not found or not READY.")

        class_names: list
        if isinstance(model.class_names, dict):
            class_names = [k for k, _ in sorted(model.class_names.items(), key=lambda x: x[1])]
        else:
            class_names = list(model.class_names or [])

        # Download ZIP
        client = s3_service.get_s3_client()
        zip_buf = io.BytesIO()
        client.download_fileobj(bucket, zip_key, zip_buf)
        zip_buf.seek(0)

        results = []
        image_types = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        with zipfile.ZipFile(zip_buf) as zf:
            for name in zf.namelist():
                ext = os.path.splitext(name.lower())[1]
                if ext not in image_types:
                    continue
                with zf.open(name) as img_file:
                    image_bytes = img_file.read()
                try:
                    predictions, latency_ms = run_predict(
                        model_id=model_id,
                        image_bytes=image_bytes,
                        s3_bucket=model.s3_bucket,
                        s3_key=model.onnx_s3_key,
                        class_names=class_names,
                    )
                    results.append({
                        "filename": name,
                        "predictions": predictions,
                        "latency_ms": round(latency_ms, 2),
                    })
                except Exception as exc:
                    results.append({"filename": name, "error": str(exc)})

        # Upload results JSON
        results_key = f"batch_jobs/{job_id}/results.json"
        payload = json.dumps({"job_id": job_id, "model_id": model_id, "results": results}).encode()
        s3_service.upload_bytes(payload, bucket, results_key, content_type="application/json")
        logger.info("Batch predict job %s done: %d images", job_id, len(results))

    except Exception as exc:
        logger.exception("Batch predict job %s failed", job_id)
        error_key = f"batch_jobs/{job_id}/error.txt"
        try:
            s3_service.upload_bytes(str(exc).encode(), bucket, error_key, content_type="text/plain")
        except Exception:
            pass
        raise self.retry(exc=exc)
    finally:
        db.close()
