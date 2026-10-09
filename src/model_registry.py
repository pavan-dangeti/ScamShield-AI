"""Model registry: pick which trained model the API serves.

The production default is the int8 ONNX encoder, because it is the best measured
model that is fast enough to serve (4.4 ms p50 on CPU, F1 0.998 on the validation
split). If its files are not on disk, they are fetched from Hugging Face Hub and
checked against the SHA-256 sums in ``configs/model_hub.json``. The TF-IDF baseline is selectable for environments where 278 MB is not
acceptable, and the mapping keeps the choice a single environment variable.

Set ``SCAMSHIELD_MODEL`` to one of:
  ``xlmr-aug-int8``  - int8 ONNX XLM-R fine-tuned with augmentation (default; fetched if absent)
  ``xlmr-aug-fp32``  - fp32 ONNX XLM-R
  ``tfidf-lr``       - retrained TF-IDF baseline (no torch/onnx needed)
"""

from __future__ import annotations

import importlib.util
import logging
import os

from src.models.base import Predictor

log = logging.getLogger("scamshield.registry")


def resolve_model_name(requested: str | None = None) -> str:
    if requested:
        return requested
    return os.getenv("SCAMSHIELD_MODEL", "xlmr-aug-int8")


def onnx_path_for(name: str) -> str | None:
    if name == "xlmr-aug-int8":
        return "models/onnx/xlm-roberta-base-aug/model.int8.onnx"
    if name == "xlmr-aug-fp32":
        return "models/onnx/xlm-roberta-base-aug/model.onnx"
    return None


def load_predictor(name: str) -> Predictor:
    """Return an object exposing .predict(texts) -> list[Prediction].

    Falls back to the TF-IDF baseline if the requested neural artefact is absent, so
    a fresh clone still serves a working API.
    """
    onnx_path = onnx_path_for(name)
    if name == "xlmr-aug-int8" and onnx_path and not os.path.exists(onnx_path):
        onnx_path = _fetch_published_model() or onnx_path
    if onnx_path and not os.path.exists(onnx_path):
        log.warning("model artefact missing, falling back to tfidf-lr", extra={"model": name, "path": onnx_path})
    if onnx_path and os.path.exists(onnx_path):
        try:
            from src.models.onnx_infer import OnnxScamPredictor

            return OnnxScamPredictor(name, onnx_path)
        except Exception as error:  # noqa: BLE001 - degrade to baseline rather than crash
            log.warning("could not load model, falling back to tfidf-lr", extra={"model": name, "error": str(error)})
    from src.models.tfidf import TfidfScamModel

    return TfidfScamModel.load("models/tfidf-lr.pkl")


def _fetch_published_model() -> str | None:
    """Download the int8 model from the Hub, or None if it cannot be served here."""
    # Without the runtime the files are useless, so do not spend 295 MB finding out.
    missing = [m for m in ("onnxruntime", "transformers") if importlib.util.find_spec(m) is None]
    if missing:
        log.warning("int8 model needs requirements-onnx.txt", extra={"missing": missing})
        return None
    try:
        from src.model_hub import ensure_model

        return ensure_model()
    except Exception as error:  # noqa: BLE001 - no network or unpublished: serve the baseline
        log.warning("could not fetch the published model", extra={"error": str(error)})
        return None
