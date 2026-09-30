"""Build the candidate pool for the hand-verified test set.

Only messages with ``provenance == "real"`` are eligible.  Today that means the
UCI SMS Spam Collection (English) plus anything the owner imports from their own
phone with ``scripts/import_user_messages.py``.  Nothing synthetic, and nothing
from a source whose provenance is unstated, can reach the test set.

Usage::

    python -m scripts.make_test_candidates --per-class 100
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

import pandas as pd

from scripts.build_dataset import PROCESSED_DIR, RESERVE_HINT, load_uci, normalize_text

CANDIDATE_DIR = "data/test_candidates"
USER_IMPORT = os.path.join(CANDIDATE_DIR, "user_messages.jsonl")
CANDIDATES = os.path.join(CANDIDATE_DIR, "candidates.jsonl")


def load_user_messages() -> list[dict]:
    if not os.path.exists(USER_IMPORT):
        return []
    return [json.loads(line) for line in open(USER_IMPORT, encoding="utf-8") if line.strip()]


def used_texts() -> set[str]:
    """Text already present in the training pool, so it can never be a test row."""
    used: set[str] = set()
    for name in ("train", "val"):
        path = os.path.join(PROCESSED_DIR, f"{name}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                used.update(normalize_text(json.loads(line)["text"]) for line in handle if line.strip())
    return used


def candidates_from_sources(per_class: int, seed: int) -> list[dict]:
    """Scam candidates come from the held-out UCI reserve; every class is unlabelled."""
    rows: list[dict] = []
    uci = load_uci()
    reserve = uci[uci["split_hint"] == RESERVE_HINT]
    for label in ("scam", "legit"):
        pool = reserve[reserve["label"] == label].sample(frac=1.0, random_state=seed)
        for _, row in pool.head(per_class).iterrows():
            rows.append(
                {
                    "id": f"test-{row['id']}",
                    "text": row["text"],
                    "language": "english",
                    "script": "latin",
                    "label": None,
                    "tactics": [],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "licence": "CC-BY-4.0",
                    "provenance": "real",
                    "synthetic": False,
                    "cell": "english|latin",
                }
            )
    for message in load_user_messages():
        rows.append(
            {
                "id": f"test-{message['id']}",
                "text": message["text"],
                "language": message["language"],
                "script": message["script"],
                "label": None,
                "tactics": [],
                "source": "user_contributed",
                "source_id": message["id"],
                "licence": "project",
                "provenance": "real",
                "synthetic": False,
                "cell": f"{message['language']}|{message['script']}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=100, help="UCI candidates per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = candidates_from_sources(args.per_class, args.seed)
    used = used_texts()
    before_overlap = len(rows)
    rows = [row for row in rows if normalize_text(row["text"]) not in used]
    overlaps = before_overlap - len(rows)

    seen: set[str] = set()
    unique = []
    for row in rows:
        key = normalize_text(row["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    with open(CANDIDATES, "w", encoding="utf-8") as handle:
        for row in unique:
            row["pool_built"] = date.today().isoformat()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = pd.Series([row["cell"] for row in unique]).value_counts().to_dict()
    print(f"Wrote {len(unique)} candidates to {CANDIDATES}")
    print(f"Dropped {overlaps} rows already in the training pool and {len(rows) - len(unique)} duplicates")
    for cell, count in sorted(counts.items()):
        print(f"  {cell}: {count}")


if __name__ == "__main__":
    main()
