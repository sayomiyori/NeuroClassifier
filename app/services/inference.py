import io
import logging
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore
    _ORT_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "onnxruntime is not installed. Inference endpoints will be unavailable. "
        "Install a compatible wheel: pip install onnxruntime"
    )

if TYPE_CHECKING:  # pragma: no cover
    import onnxruntime as ort_typing

from app.config import get_settings
from app.services import s3 as s3_service

settings = get_settings()
logger = logging.getLogger(__name__)

# In-memory session cache:  model_id -> (session, class_names)
_session_cache: Dict[str, Tuple[Any, List[str]]] = {}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _preprocess(image_bytes: bytes, input_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Decode image bytes → normalised float32 NCHW tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(input_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis, ...]   # add batch dim → NCHW


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _load_session(model_id: str, s3_bucket: str, s3_key: str) -> Any:
    """Download ONNX model from MinIO and create an ORT session (cached)."""
    if not _ORT_AVAILABLE:
        raise RuntimeError(
            "onnxruntime is not installed; inference is unavailable in this environment."
        )
    if model_id in _session_cache:
        return _session_cache[model_id][0]

    client = s3_service.get_s3_client()
    buf = io.BytesIO()
    client.download_fileobj(s3_bucket, s3_key, buf)
    buf.seek(0)

    # ort is guaranteed to be available here (checked above)
    session = ort.InferenceSession(buf.read(), providers=["CPUExecutionProvider"])  # type: ignore[union-attr]
    logger.info("Loaded ONNX session for model %s", model_id)
    return session


def run_inference(
    image_bytes: bytes,
    model_id: str,
    s3_bucket: str,
    s3_key: str,
    class_names: List[str],
    top_k: int = 5,
) -> List[Dict]:
    """
    Run inference on raw image bytes.
    Returns list of {class, probability} sorted by probability DESC.
    """
    session = _load_session(model_id, s3_bucket, s3_key)
    input_name = session.get_inputs()[0].name

    tensor = _preprocess(image_bytes)
    outputs = session.run(None, {input_name: tensor})
    logits = outputs[0][0]  # shape (num_classes,)
    probs = _softmax(logits)

    top_k = min(top_k, len(class_names))
    top_indices = np.argsort(probs)[::-1][:top_k]

    return [
        {"class": class_names[i], "probability": float(probs[i])}
        for i in top_indices
    ]


def clear_model_cache(model_id: str) -> None:
    _session_cache.pop(model_id, None)
