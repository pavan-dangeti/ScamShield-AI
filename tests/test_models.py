"""Tests for the model interface and the shared evaluation harness."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from scripts.evaluate_models import evaluate
from src.models.base import Prediction, TacticPrediction
from src.models.tfidf import TfidfScamModel, load_split
from src.taxonomy import TACTICS


def test_predict_returns_one_prediction_per_message(tiny_model):
    texts = ["Urgent: your account will be blocked, click bit.ly/x", "See you tomorrow"]
    predictions = tiny_model.predict(texts)
    assert len(predictions) == len(texts)
    assert all(isinstance(p, Prediction) for p in predictions)
    assert all(0.0 <= p.scam_probability <= 1.0 for p in predictions)


def test_label_follows_the_threshold(tiny_model):
    predictions = tiny_model.predict(
        ["Urgent: your account will be blocked in 10 minutes, click bit.ly/abc now"]
    )
    prediction = predictions[0]
    assert prediction.label == ("scam" if prediction.scam_probability >= 0.5 else "legit")


def test_tactic_predictions_use_the_taxonomy(tiny_model):
    predictions = tiny_model.predict(
        ["Urgent! Your account will be blocked in 10 minutes, click bit.ly/abc now"]
    )
    for tactic in predictions[0].tactics:
        assert isinstance(tactic, TacticPrediction)
        assert tactic.tactic in TACTICS
        assert 0.0 <= tactic.probability <= 1.0


def test_evidence_is_a_span_of_the_message(tiny_model):
    text = "Urgent! Your SBI account will be blocked in 10 minutes, click bit.ly/abc now"
    head = next(iter(tiny_model.tactic_models))  # whatever head this tiny model learned
    span = tiny_model.evidence(text, head)
    assert span
    assert span.lower() in text.lower()


def test_tactic_heads_are_skipped_when_there_are_too_few_positives():
    """A tactic with almost no positive examples must not produce an unstable head."""
    rows = [
        {"text": f"message number {i} with ordinary words", "label": "scam", "tactics": ["urgency"],
         "language": "english", "script": "latin"}
        for i in range(20)
    ] + [
        {"text": f"legit note number {i} about delivery", "label": "legit", "tactics": [],
         "language": "english", "script": "latin"}
        for i in range(20)
    ]
    model = TfidfScamModel(c_grid=[1.0]).fit(rows)
    assert "urgency" in model.tactic_models
    assert "false_reward" not in model.tactic_models  # zero positives in this data


def test_evaluate_reports_cells_and_tactics(tiny_model, monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.evaluate_models.binary_scores",
                        lambda name, rows: (np.array([0.9, 0.1, 0.8, 0.2]), [[], [], ["urgency"], []]))
    rows = [
        {"id": "1", "text": "a", "language": "hindi", "script": "native", "label": "scam", "tactics": ["urgency"]},
        {"id": "2", "text": "b", "language": "hindi", "script": "native", "label": "legit", "tactics": []},
        {"id": "3", "text": "c", "language": "tamil", "script": "romanized", "label": "scam", "tactics": ["urgency"]},
        {"id": "4", "text": "d", "language": "tamil", "script": "romanized", "label": "legit", "tactics": []},
    ]
    result = evaluate("stub", rows, threshold=0.5)
    assert result["f1"] == 1.0
    assert set(result["by_cell"]) == {"hindi|native", "tamil|romanized"}
    assert result["tactics"]["labelled_rows"] == 2
    assert "urgency" in result["tactics"]["per_tactic_f1"]


def test_evaluate_handles_rows_without_tactic_labels():
    rows = [
        {"id": "1", "text": "a", "language": "english", "script": "latin", "label": "scam", "tactics": []},
        {"id": "2", "text": "b", "language": "english", "script": "latin", "label": "legit", "tactics": []},
    ]
    result = evaluate("tfidf-lr", rows, threshold=0.5) if os.path.exists("models/tfidf-lr.pkl") else None
    if result is not None:
        assert result["tactics"]["labelled_rows"] == 0
