"""Train and save the TF-IDF baseline on the built corpus.

    python -m scripts.train_tfidf

Writes ``models/tfidf-lr.pkl`` and prints the per-head cross-validated F1 and the
selected regularisation strength, so the tuning is visible rather than implied.
"""

from __future__ import annotations

import json

from src.data import load_split
from src.models.tfidf import TfidfScamModel


def main() -> None:
    rows = load_split("train")
    print(f"training on {len(rows)} rows")
    model = TfidfScamModel().fit(rows)
    model.save("models/tfidf-lr.pkl")
    print(json.dumps(model.search_results, indent=2))
    print("saved models/tfidf-lr.pkl")


if __name__ == "__main__":
    main()
