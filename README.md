# NeuroClassifier

![CI](https://github.com/your-org/neuroclassifier/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![License](https://img.shields.io/badge/license-MIT-blue)

Production-ready image classification platform. Upload a labelled dataset, fine-tune a vision transformer with LoRA in the background, and serve predictions via ONNX Runtime — all through a single REST API.

---

## Architecture

```
┌─────────────┐     upload / predict     ┌──────────────────────┐
│   Client    │ ───────────────────────► │  FastAPI (port 8000) │
└─────────────┘                          └──────┬───────────────┘
                                                │  enqueue task
                                         ┌──────▼───────────────┐
                                         │   Redis (Celery)     │
                                         └──────┬───────────────┘
                                                │  train / infer
                                         ┌──────▼───────────────┐
                                         │   Celery Worker      │
                                         │  (LoRA fine-tune /   │
                                         │   batch inference)   │
                                         └──────┬───────────────┘
                                                │  store artefacts
                              ┌─────────────────▼────────────────┐
                              │         MinIO (S3-compatible)    │
                              │   datasets/  ·  models/          │
                              └──────────────────────────────────┘
                                                │
                              ┌─────────────────▼────────────────┐
                              │   PostgreSQL — metadata & jobs   │
                              └──────────────────────────────────┘

Observability: FastAPI → /metrics → Prometheus → Grafana
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Task queue | Celery + Redis |
| Object storage | MinIO (S3-compatible) |
| Database | PostgreSQL + SQLAlchemy (async) |
| Training | PyTorch + HuggingFace Transformers + PEFT (LoRA) |
| Inference | ONNX Runtime |
| Monitoring | Prometheus + Grafana |
| CI | GitHub Actions |

---

## Architecture Decisions

**LoRA over full fine-tuning** — trains only ~1 % of model parameters (`query` + `value` projections). Converges in minutes on CPU, no GPU required for small datasets.

**ONNX Runtime for inference** — 3–5× faster than PyTorch `model.forward()`, no dependency on PyTorch at serve-time. The worker exports a merged (LoRA-unloaded) ONNX graph after training.

**MinIO for model storage** — S3-compatible object store that runs in Docker. Versioned adapter weights and ONNX artefacts are stored side-by-side, making rollback trivial.

**Celery for training and batch inference** — keeps the API non-blocking. Training jobs can run for hours; the API just returns a `job_id` immediately and the client polls for status.

---

## Quick Start

```bash
# 1. Clone & start everything
git clone https://github.com/your-org/neuroclassifier.git
cd neuroclassifier
docker compose up -d

# 2. Check health
curl http://localhost:18015/health
# {"status":"ok","version":"0.1.0"}
```

Services after `docker compose up`:

| Service | URL |
|---|---|
| API + Swagger | http://localhost:18015/docs |
| MinIO console | http://localhost:29001 (admin / minioadmin) |
| Flower (tasks) | http://localhost:15555 |
| Prometheus | http://localhost:19090 |
| Grafana | http://localhost:13000 (admin / admin) |

---

## API

### Datasets

```bash
# Upload labelled ZIP (class_name/image.jpg structure)
curl -X POST http://localhost:18015/api/v1/datasets \
  -F "name=cats_vs_dogs" -F "file=@dataset.zip"

# List datasets
curl http://localhost:18015/api/v1/datasets
```

### Training

```bash
# Start LoRA fine-tuning
curl -X POST http://localhost:18015/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "<dataset_uuid>",
    "base_model": "google/vit-base-patch16-224",
    "epochs": 5,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "lora_rank": 8
  }'

# Poll job status
curl http://localhost:18015/api/v1/train/<job_id>

# Learning curve
curl http://localhost:18015/api/v1/models/<model_id>/metrics
```

### Inference

```bash
# Single image
curl -X POST http://localhost:18015/api/v1/predict \
  -F "model_id=<model_uuid>" \
  -F "file=@cat.jpg"
# {"model_id":"...","predictions":[{"class_name":"cat","confidence":0.97}],"latency_ms":12.4}

# Batch (ZIP of images) — async
curl -X POST http://localhost:18015/api/v1/predict/batch \
  -F "model_id=<model_uuid>" \
  -F "file=@images.zip"
# {"job_id":"...","status":"processing"}

# Poll batch results
curl http://localhost:18015/api/v1/predict/batch/<job_id>
# {"job_id":"...","status":"completed","results_url":"http://...presigned..."}
```

---

## Monitoring

Grafana dashboard at **http://localhost:13000** (admin / admin) includes:

- **Training** — jobs over time, avg duration, active jobs
- **Inference** — latency p50/p95/p99, RPS by model, predictions by class
- **HTTP** — request rate and latency by endpoint

Prometheus scrapes `/metrics` every 15 s.

---

## Development

```bash
# Install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=app

# Lint
ruff check app/
```
