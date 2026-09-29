"""Inference wrapper for a fine-tuned encoder (IndicBERTv2, XLM-R)."""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.taxonomy import TACTICS


def _model_dir(model_name: str) -> str:
    return os.path.join("models", model_name.replace("/", "_"))


def load_encoder(model_name: str):
    from transformers import AutoTokenizer

    from src.models.transformer import ScamEncoderModel

    directory = _model_dir(model_name)
    with open(os.path.join(directory, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = ScamEncoderModel(config["model_name"], dropout=config.get("dropout", 0.1))
    model.load_state_dict(torch.load(os.path.join(directory, "model.pt"), map_location="cpu"))
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer, config, device


@torch.no_grad()
def predict_encoder(model_name: str, texts: list[str], rows: list[dict], batch_size: int = 64):
    model, tokenizer, config, device = load_encoder(model_name)
    max_length = config.get("max_length", 128)
    probabilities: list[float] = []
    tactics: list[list[str]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        ).to(device)
        binary_logits, tactic_logits = model(
            encoded["input_ids"], encoded["attention_mask"], encoded.get("token_type_ids")
        )
        binary_probabilities = torch.sigmoid(binary_logits).cpu().numpy()
        tactic_probabilities = (torch.sigmoid(tactic_logits) >= 0.5).cpu().numpy()
        probabilities.extend(binary_probabilities.tolist())
        for index in range(len(batch)):
            if binary_probabilities[index] >= 0.5:
                tactics.append(
                    [TACTICS[i] for i in range(len(TACTICS)) if tactic_probabilities[index][i]]
                )
            else:
                tactics.append([])
    return np.array(probabilities), tactics
