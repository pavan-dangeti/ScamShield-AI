"""ONNX Runtime predictor for the quantised encoder, exposed through the same
interface as every other ScamShield model (see ``src/models/base.py``)."""

from __future__ import annotations

import os

import numpy as np

from src.models.base import Prediction, TacticPrediction
from src.taxonomy import TACTICS


class OnnxScamPredictor:
    name = "xlmr-aug-int8"

    def __init__(self, name: str, onnx_path: str, max_length: int | None = None) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.name = name
        directory = os.path.dirname(onnx_path)
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.max_length = max_length or 160
        self.binary_output = self.session.get_outputs()[0].name

    def predict_binary(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            list(texts), truncation=True, max_length=self.max_length, padding=True, return_tensors="np"
        )
        feed = {name: encoded[name] for name in self.input_names if name in encoded}
        outputs = self.session.run(None, feed)
        return 1.0 / (1.0 + np.exp(-outputs[0]))

    def predict(self, texts: list[str]) -> list[Prediction]:
        texts = list(texts)
        encoded = self.tokenizer(
            texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="np"
        )
        feed = {name: encoded[name] for name in self.input_names if name in encoded}
        binary_logits, tactic_logits = self.session.run(None, feed)
        probabilities = 1.0 / (1.0 + np.exp(-binary_logits))
        tactic_probabilities = 1.0 / (1.0 + np.exp(-tactic_logits))
        predictions = []
        for index in range(len(texts)):
            tactics = []
            if probabilities[index] >= 0.5:
                for position, tactic in enumerate(TACTICS):
                    if tactic_probabilities[index][position] >= 0.5:
                        tactics.append(TacticPrediction(tactic, float(tactic_probabilities[index][position])))
            predictions.append(Prediction(scam_probability=float(probabilities[index]), tactics=tactics))
        return predictions
