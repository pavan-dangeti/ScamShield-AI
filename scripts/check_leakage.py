"""Check that ScamShield splits do not leak messages into each other.

Reports exact text overlap and, for every pair of splits, the largest
character-n-gram cosine similarity between a row in one split and any row in
the other.  The check re-reads the written split files and recomputes the
metric used by ``build_dataset.py`` (``NEAR_DUP_VECTORIZER``); near-duplicate
detection is metric dependent, so this is a regression check on the written
artefacts rather than an independent measure of similarity.

Usage (from the repository root, after ``python -m scripts.build_dataset``):

    python -m scripts.check_leakage --threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from scripts.build_dataset import NEAR_DUP_VECTORIZER, PROCESSED_DIR, normalize_text

CHECK_VECTORIZER = TfidfVectorizer(**NEAR_DUP_VECTORIZER)


def load_split(name: str) -> list[dict]:
    path = os.path.join(PROCESSED_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def max_cross_similarity(left: list[str], right: list[str]) -> np.ndarray:
    """For each row in ``right``, the best cosine similarity against ``left``."""
    check = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1, n_jobs=-1)
    check.fit(CHECK_VECTORIZER.fit_transform(left))
    distances, _ = check.kneighbors(CHECK_VECTORIZER.transform(right))
    return 1.0 - distances[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()

    splits = {name: load_split(name) for name in ("train", "val", "test")}
    splits = {name: rows for name, rows in splits.items() if rows}
    if not splits:
        raise SystemExit("No splits found; run scripts.build_dataset first.")

    for name, rows in splits.items():
        print(f"{name}: {len(rows)} rows")

    print("\nExact normalized text overlap between splits:")
    keys = {name: {normalize_text(row["text"]) for row in rows} for name, rows in splits.items()}
    for left, right in combinations(keys, 2):
        print(f"  {left} ∩ {right}: {len(keys[left] & keys[right])}")

    print(f"\nCross-split near-duplicate check (threshold {args.threshold}):")
    failures = 0
    for left, right in combinations(splits, 2):
        similarities = max_cross_similarity(
            [row["text"] for row in splits[left]], [row["text"] for row in splits[right]]
        )
        over = int((similarities >= args.threshold).sum())
        failures += over
        print(
            f"  {left}->{right}: max={similarities.max():.3f} "
            f"mean={similarities.mean():.3f} p99={np.percentile(similarities, 99):.3f} "
            f"pairs>={args.threshold}: {over}"
        )

    if failures:
        print(f"\nFAIL: {failures} cross-split near-duplicate pairs at or above {args.threshold}")
        raise SystemExit(1)
    print(f"\nPASS: no cross-split near-duplicate pair at or above {args.threshold}")


if __name__ == "__main__":
    main()
