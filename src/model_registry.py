"""Model registry: pick which trained model the API serves.

The production default is the int8 ONNX encoder, because it is the best measured
model that is fast enough to serve (6 ms p50 on CPU, F1 0.998 on the validation
split). The TF-IDF baseline is selectable for environments where 278 MB is not
acceptable, and the mapping keeps the choice a single environment variable.

Set ``SCAMSHIELD_MODEL`` to one of:
  ``xlmr-aug-int8``  - int8 ONNX XLM-R fine-tuned with augmentation (default if present)
  ``xlmr-aug-fp32``  - fp32 ONNX XLM-R
  ``tfidf-lr``       - retrained TF-IDF baseline (no torch/onnx needed)
"""

from __future__ import annotations

import logging
import os

from src.models.base import Predictor


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
    if onnx_path and not os.path.exists(onnx_path):
        logging.getLogger("scamshield.registry").warning(
            "model artefact missing, falling back to tfidf-lr", extra={"model": name, "path": onnx_path}
        )
    if onnx_path and os.path.exists(onnx_path):
        try:
            from src.models.onnx_infer import OnnxScamPredictor

            return OnnxScamPredictor(name, onnx_path)
        except Exception as error:  # noqa: BLE001 - degrade to baseline rather than crash
            logging.getLogger("scamshield.registry").warning(
                "could not load model, falling back to tfidf-lr", extra={"model": name, "error": str(error)}
            )
    from src.models.tfidf import TfidfScamModel

    return TfidfScamModel.load("models/tfidf-lr.pkl")
