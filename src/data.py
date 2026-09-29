"""Data access shared by every model script."""

from __future__ import annotations

import json
import os
import random

PROCESSED_DIR = "data/processed"


def load_split(name: str) -> list[dict]:
    """Read a split written by ``scripts.build_dataset``."""
    path = os.path.join(PROCESSED_DIR, f"{name}.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def subsample_rows(rows: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """Seeded stratified subsample over (language, script, label) cells.

    Generative models are far slower per message than the trained ones, so they
    are scored on a subsample. Using the same function and seed here and in
    ``scripts/evaluate_models.py`` guarantees every model in the comparison sees
    exactly the same rows.
    """
    if not limit or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["language"], row["script"], row["label"]), []).append(row)
    per_cell = max(1, limit // max(1, len(cells)))
    sampled: list[dict] = []
    for cell_rows in cells.values():
        sampled.extend(rng.sample(cell_rows, min(per_cell, len(cell_rows))))
    rng.shuffle(sampled)
    return sampled[:limit]
