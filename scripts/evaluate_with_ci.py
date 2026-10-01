"""Phase 3 evaluation: confidence intervals, calibration, thresholds, latency.

Everything here is measurable from the artefacts in the repository, so every
number in the README can be reproduced with one command.

Usage::

    python -m scripts.evaluate_with_ci --models tfidf-lr indicbertv2-mlm xlmr-base
    python -m scripts.benchmark_latency --models tfidf-lr indicbertv2-mlm xlmr-base
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from src.data import load_split, subsample_rows

RESULTS_DIR = "results"

# A missed scam costs the reader money or credentials; a false alarm costs a
# look at a real message. The ratio is a product decision, stated here and swept
# below so the reader can see how sensitive the choice is.
DEFAULT_COST_RATIO = 10.0


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI for a statistic over rows."""
    rng = np.random.default_rng(seed)
    n = len(values)
    samples = []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        samples.append(statistic(values[index]))
    return float(np.percentile(samples, 100 * alpha / 2)), float(np.percentile(samples, 100 * (1 - alpha / 2)))


def f1_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    return float(f1_score(labels, (scores >= threshold).astype(int), zero_division=0))


def expected_cost(scores: np.ndarray, labels: np.ndarray, threshold: float, ratio: float) -> float:
    predicted = (scores >= threshold).astype(int)
    false_negative = int(((labels == 1) & (predicted == 0)).sum())
    false_positive = int(((labels == 0) & (predicted == 1)).sum())
    return (ratio * false_negative + false_positive) / len(labels)


def calibration_bins(scores: np.ndarray, labels: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    output = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (scores >= low) & (scores < high if high < 1 else scores <= 1)
        if mask.sum() == 0:
            continue
        output.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "count": int(mask.sum()),
                "mean_score": float(scores[mask].mean()),
                "observed_rate": float(labels[mask].mean()),
            }
        )
    return output


def expected_calibration_error(bins: list[dict]) -> float:
    total = sum(b["count"] for b in bins) or 1
    return float(sum(b["count"] * abs(b["mean_score"] - b["observed_rate"]) for b in bins) / total)


def analyze(scores: np.ndarray, labels: np.ndarray, rows: list[dict], n_boot: int) -> dict:
    predicted = (scores >= 0.5).astype(int)
    precision = float(precision_score(labels, predicted, zero_division=0))
    recall = float(recall_score(labels, predicted, zero_division=0))
    f1 = float(f1_score(labels, predicted, zero_division=0))

    # Per-cell metrics with CIs, only for cells that contain both classes.
    cells: dict[str, dict] = {}
    for key in sorted({f"{row['language']}|{row['script']}" for row in rows}):
        index = np.array([i for i, row in enumerate(rows) if f"{row['language']}|{row['script']}" == key])
        cell_labels = labels[index]
        if len(set(cell_labels.tolist())) < 2:
            cells[key] = {"rows": int(len(index)), "note": "single class in this cell"}
            continue
        combined = np.stack([scores[index], cell_labels], axis=1)
        stat = lambda v: f1_at(v[:, 0], v[:, 1], 0.5)  # noqa: E731
        low, high = bootstrap_ci(combined, stat, n_boot=n_boot)
        cells[key] = {
            "rows": int(len(index)),
            "scam_rows": int(cell_labels.sum()),
            "precision": float(precision_score(cell_labels, predicted[index], zero_division=0)),
            "recall": float(recall_score(cell_labels, predicted[index], zero_division=0)),
            "f1": stat(combined),
            "f1_ci95": [low, high],
        }
    macro = [c["f1"] for c in cells.values() if "f1" in c]

    # Threshold sweep under a stated cost ratio.
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    sweep = [
        {
            "threshold": t,
            "precision": float(precision_score(labels, (scores >= t).astype(int), zero_division=0)),
            "recall": float(recall_score(labels, (scores >= t).astype(int), zero_division=0)),
            "f1": f1_at(scores, labels, t),
            "expected_cost": expected_cost(scores, labels, t, DEFAULT_COST_RATIO),
        }
        for t in thresholds
    ]
    best_cost = min(sweep, key=lambda r: r["expected_cost"])
    best_f1 = max(sweep, key=lambda r: r["f1"])

    bins = calibration_bins(scores, labels)
    return {
        "rows": int(len(labels)),
        "scam_rows": int(labels.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_ci95": list(bootstrap_ci(np.stack([scores, labels], axis=1), lambda v: f1_at(v[:, 0], v[:, 1], 0.5), n_boot=n_boot)),
        "macro_f1_over_cells": float(np.mean(macro)) if macro else None,
        "brier": float(brier_score_loss(labels, scores)),
        "expected_calibration_error": expected_calibration_error(bins),
        "calibration_bins": bins,
        "threshold_sweep": sweep,
        "cost_ratio": DEFAULT_COST_RATIO,
        "best_threshold_by_cost": best_cost,
        "best_threshold_by_f1": best_f1,
        "by_cell": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tfidf-lr"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "evaluation_with_ci.json"))
    args = parser.parse_args()

    from scripts.evaluate_models import binary_scores  # reuse the one scoring path

    rows = load_split(args.split)
    if args.subsample:
        rows = subsample_rows(rows, args.subsample, 42)
    labels = np.array([1 if row["label"] == "scam" else 0 for row in rows])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {"split": args.split, "rows": len(rows), "models": {}}
    for model_name in args.models:
        print(f"analysing {model_name}...", flush=True)
        scores, _ = binary_scores(model_name, rows)
        report["models"][model_name] = analyze(scores, labels, rows, args.bootstrap)
        summary = report["models"][model_name]
        print(
            f"  F1={summary['f1']:.4f} {summary['f1_ci95'][0]:.3f}-{summary['f1_ci95'][1]:.3f} | "
            f"macro={summary['macro_f1_over_cells']:.4f} | ECE={summary['expected_calibration_error']:.4f} | "
            f"best-cost threshold={summary['best_threshold_by_cost']['threshold']}",
            flush=True,
        )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
