"""Shared interface for every ScamShield model.

A predictor turns a list of messages into a scam probability and, for a subset of
them, a multi-label set of manipulation tactics with confidences.  Keeping one
interface means the TF-IDF baseline, the fine-tuned encoders, the prompted LLM
and the LoRA model are compared by exactly the same harness, and the API can
serve any of them without changes (see ``src/model_registry.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class TacticPrediction:
    tactic: str
    probability: float


@dataclass
class Prediction:
    """One message's worth of model output."""

    scam_probability: float
    tactics: list[TacticPrediction] = field(default_factory=list)
    explanation: str = ""

    @property
    def label(self) -> str:
        return "scam" if self.scam_probability >= 0.5 else "legit"


class Predictor(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Score a batch of messages."""
        ...

    def predict_binary(self, texts: list[str]) -> np.ndarray:
        """Return the scam probability for each message."""
        ...
