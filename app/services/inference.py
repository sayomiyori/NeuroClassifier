import io
import logging
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore
    _ORT_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "onnxruntime is not installed. Inference endpoints will be unavailable."
    )

from app.config import get_settings
from app.services import s3 as s3_service

settings = get_settings()
logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_LRU_MAX = 5

# LRU cache entry: {"session", "class_names", "tmp_dir" (optional)}
_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evict_if_needed() -> None:
    while len(_cache) >= _LRU_MAX:
        evicted_id, entry = _cache.popitem(last=False)
        tmp_dir = entry.get("tmp_dir")
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Evicted model %s from inference cache (LRU)", evicted_id)


def _preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...]


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _download_onnx(s3_bucket: str, s3_key: str) -> Tuple[str, Optional[str]]:
    """
    Download model.onnx to a temp dir.  Also downloads model.onnx.data if present.
    Returns (onnx_path, tmp_dir).  Caller owns tmp_dir lifetime.
    """
    client = s3_service.get_s3_client()
    tmp_dir = tempfile.mkdtemp(prefix="nc_ort_")
    onnx_path = os.path.join(tmp_dir, "model.onnx")

    with open(onnx_path, "wb") as f:
        client.download_fileobj(s3_bucket, s3_key, f)

    data_key = s3_key + ".data"
    try:
        client.head_object(Bucket=s3_bucket, Key=data_key)
        with open(onnx_path + ".data", "wb") as f:
            client.download_fileobj(s3_bucket, data_key, f)
        logger.info("Downloaded external ONNX data for %s", s3_key)
    except Exception:
        pass

    return onnx_path, tmp_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(model_id: str, s3_bucket: str, s3_key: str, class_names: List[str]) -> None:
    """
    Download ONNX from MinIO, create an ORT InferenceSession and store in LRU cache.
    Temp dir is kept alive as long as the session is cached (needed for external data).
    """
    if not _ORT_AVAILABLE:
        raise RuntimeError(
            "onnxruntime is not installed; inference is unavailable in this environment."
        )
    if model_id in _cache:
        _cache.move_to_end(model_id)
        return

    _evict_if_needed()

    onnx_path, tmp_dir = _download_onnx(s3_bucket, s3_key)
    try:
        session = ort.InferenceSession(  # type: ignore[union-attr]
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    _cache[model_id] = {"session": session, "class_names": class_names, "tmp_dir": tmp_dir}
    _cache.move_to_end(model_id)
    logger.info("Loaded ONNX session for model %s (cache size=%d)", model_id, len(_cache))


def predict(
    model_id: str,
    image_bytes: bytes,
    s3_bucket: str,
    s3_key: str,
    class_names: List[str],
    top_k: int = 5,
) -> Tuple[List[Dict], float]:
    load_model(model_id, s3_bucket, s3_key, class_names)
    entry = _cache[model_id]
    _cache.move_to_end(model_id)

    session: Any = entry["session"]
    names: List[str] = entry["class_names"]

    tensor = _preprocess(image_bytes)
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: tensor})
    latency_ms = (time.perf_counter() - t0) * 1000

    logits = outputs[0][0]
    probs = _softmax(logits)

    k = min(top_k, len(names))
    top_idx = np.argsort(probs)[::-1][:k]
    predictions = [
        {"class_name": names[i], "confidence": float(probs[i])}
        for i in top_idx
    ]
    return predictions, latency_ms


def clear_model_cache(model_id: Optional[str] = None) -> None:
    if model_id is None:
        for entry in _cache.values():
            tmp_dir = entry.get("tmp_dir")
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        _cache.clear()
    else:
        entry = _cache.pop(model_id, None)
        if entry:
            tmp_dir = entry.get("tmp_dir")
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Legacy shim
# ---------------------------------------------------------------------------

def run_inference(
    image_bytes: bytes,
    model_id: str,
    s3_bucket: str,
    s3_key: str,
    class_names: List[str],
    top_k: int = 5,
) -> List[Dict]:
    preds, _ = predict(model_id, image_bytes, s3_bucket, s3_key, class_names, top_k)
    return [{"class": p["class_name"], "probability": p["confidence"]} for p in preds]
