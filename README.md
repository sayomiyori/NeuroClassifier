# NeuroClassifier

ML inference platform for image classification — upload datasets, fine-tune models, run ONNX inference.

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + uvicorn |
| Database | PostgreSQL 16 + SQLAlchemy 2 (async) |
| Migrations | Alembic |
| Object Storage | MinIO (S3-compatible) |
| Task Queue | Celery + Redis |
| Inference | ONNX Runtime |
| Monitoring | Prometheus metrics |

---

## Quick start

```bash
# 1. Copy env file
cp .env.example .env

# 2. Start all services
docker compose up -d --build

# 3. Run Alembic migrations (first time)
docker compose exec app alembic upgrade head

# 4. Open API docs
open http://localhost:8000/docs
```

Services after startup:

| Service | URL |
|---|---|
| API docs (Swagger) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Prometheus metrics | http://localhost:8000/metrics |
| MinIO console | http://localhost:9001 (minioadmin / minioadmin) |
| Celery Flower | http://localhost:5555 |

---

## Dataset API

### Upload dataset

```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -F "name=MyDataset" \
  -F "description=Cat vs Dog classifier" \
  -F "file=@/path/to/dataset.zip"
```

Expected ZIP structure:
```
dataset.zip
├── cats/
│   ├── img001.jpg
│   └── img002.jpg
└── dogs/
    ├── img001.jpg
    └── img002.jpg
```

Rules:
- Minimum **2 classes** (top-level folders)
- Minimum **10 valid images** per class
- Auto-split: **80% train / 20% val** (stratified)
- Images uploaded to MinIO: `datasets/{dataset_id}/train/<class>/` and `.../val/<class>/`

Response `202 Accepted`:
```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "MyDataset",
  "status": "uploading",
  "message": "Dataset upload accepted. Processing in background..."
}
```

Poll status:
```bash
curl http://localhost:8000/api/v1/datasets/550e8400-e29b-41d4-a716-446655440000
```

### Other endpoints

```bash
# List all datasets
GET  /api/v1/datasets

# Dataset detail + class distribution
GET  /api/v1/datasets/{id}

# Delete dataset (MinIO + PostgreSQL)
DELETE /api/v1/datasets/{id}
```

---

## Training API

```bash
# Start training job
POST /api/v1/training
{
  "dataset_id": "<uuid>",
  "base_model": "mobilenet_v2",
  "hyperparams": {"lr": 1e-4, "epochs": 10, "batch_size": 32}
}

# Monitor job
GET /api/v1/training/{job_id}

# Cancel job
DELETE /api/v1/training/{job_id}
```

---

## Inference API

```bash
# Run inference
curl -X POST http://localhost:8000/api/v1/predict/{model_id} \
  -F "file=@/path/to/image.jpg" \
  -F "top_k=5"
```

Response:
```json
{
  "model_id": "...",
  "model_name": "CatDogClassifier",
  "predictions": [
    {"class_name": "cat", "probability": 0.97},
    {"class_name": "dog", "probability": 0.03}
  ],
  "inference_time_ms": 12.5
}
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Project Structure

```
neuroclassifier/
├── app/
│   ├── main.py              # FastAPI app factory
│   ├── config.py            # Pydantic settings
│   ├── metrics.py           # Prometheus counters / histograms
│   ├── api/v1/
│   │   ├── datasets.py      # Upload + manage datasets
│   │   ├── training.py      # Start / monitor training jobs
│   │   ├── models.py        # Model registry
│   │   └── predict.py       # ONNX inference endpoint
│   ├── models/              # SQLAlchemy ORM models
│   ├── services/
│   │   ├── dataset_service.py  # Extract, validate, split, upload
│   │   ├── s3.py               # MinIO client wrapper
│   │   └── inference.py        # ONNX Runtime runner
│   ├── workers/
│   │   └── train_worker.py  # Celery tasks
│   └── db/
│       ├── session.py
│       └── migrations/      # Alembic
├── tests/
│   └── test_datasets.py
├── docker-compose.yml
├── Dockerfile
├── alembic.ini
└── requirements.txt
```

## Alembic commands

```bash
# Create new migration
alembic revision --autogenerate -m "add column X"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```
