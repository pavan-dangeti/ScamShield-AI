"""Measure how much each model degrades under adversarial perturbation.

For every model and every perturbation family we report precision, recall and F1 on
the clean split and on the perturbed split, the change in F1, and the *flip rate*:
the fraction of messages whose scam/legit decision changed. Flip rate is the metric
that matters for a filter - a scam that stops being flagged is a miss, regardless of
aggregate F1.

Usage::

    python -m scripts.evaluate_robustness --models tfidf-lr indicbertv2-mlm xlmr-base
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from scripts.adversarial import FAMILIES, build_adversarial_split
from src.data import load_split

RESULTS_DIR = "results"


def score(model_name: str, rows: list[dict], threshold: float) -> tuple[dict, np.ndarray]:
    from scripts.evaluate_models import binary_scores

    probabilities, _ = binary_scores(model_name, rows)
    labels = np.array([1 if row["label"] == "scam" else 0 for row in rows])
    predicted = (probabilities >= threshold).astype(int)
    return (
        {
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "scam_calls": int(predicted.sum()),
        },
        predicted,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tfidf-lr"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "robustness.json"))
    args = parser.parse_args()

    rows = load_split(args.split)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {"split": args.split, "rows": len(rows), "threshold": args.threshold, "models": {}}

    for model_name in args.models:
        print(f"scoring {model_name}...", flush=True)
        clean, clean_pred = score(model_name, rows, args.threshold)
        entry = {"clean": clean, "attacks": {}}
        for family in FAMILIES:
            perturbed, unchanged = build_adversarial_split(rows, family, seed=42)
            attacked, attacked_pred = score(model_name, perturbed, args.threshold)
            flips = int((clean_pred != attacked_pred).sum())
            entry["attacks"][family] = {
                **attacked,
                "f1_delta": round(attacked["f1"] - clean["f1"], 4),
                "flip_rate": round(flips / len(rows), 4),
                "rows_unchanged_by_attack": unchanged,
                "scam_recall_delta": round(attacked["recall"] - clean["recall"], 4),
            }
        report["models"][model_name] = entry
        worst = min(entry["attacks"].items(), key=lambda item: item[1]["f1"])
        print(
            f"  clean F1 {clean['f1']:.4f} -> worst attack {worst[0]} F1 {worst[1]['f1']:.4f} "
            f"(delta {worst[1]['f1_delta']:+.4f}, flip rate {worst[1]['flip_rate']:.3f})",
            flush=True,
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    families = list(FAMILIES)
    lines = [
        "# Robustness under adversarial perturbation",
        "",
        f"Split: `{args.split}` ({len(rows)} rows) · threshold {args.threshold} · "
        f"generated {__import__('datetime').datetime.now().isoformat()[:10]}",
        "",
        "F1 (delta vs clean) and flip rate = share of messages whose decision changed.",
        "",
        "| Model | Clean F1 | " + " | ".join(f"{family} F1" for family in families) + " | Mean flip rate |",
        "|---" * (len(families) + 3) + "|",
    ]
    for model_name, entry in report["models"].items():
        attacks = entry["attacks"]
        mean_flip = sum(attacks[f]["flip_rate"] for f in families) / len(families)
        cells = " | ".join(f"{attacks[f]['f1']:.3f} ({attacks[f]['f1_delta']:+.3f})" for f in families)
        lines.append(
            f"| {model_name} | {entry['clean']['f1']:.4f} | {cells} | {mean_flip:.3f} |"
        )
    lines += ["", "| Model | " + " | ".join(f"{family} flip" for family in families) + " |", "|---" * (len(families) + 1) + "|"]
    for model_name, entry in report["models"].items():
        attacks = entry["attacks"]
        lines.append(
            f"| {model_name} | " + " | ".join(f"{attacks[f]['flip_rate']:.3f}" for f in families) + " |"
        )
    markdown_path = args.out.replace(".json", ".md")
    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} and {markdown_path}")


if __name__ == "__main__":
    main()
