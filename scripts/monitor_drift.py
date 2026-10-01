"""Score-distribution drift monitor.

The model that is serving today was validated on a corpus built in 2026. When the
traffic changes - a new scam campaign, a new bank, a new language mix - the score
distribution moves before the accuracy does. This script compares the serving
score distribution against a stored reference and reports population stability
index (PSI), the standard cheap drift signal, per language/script as well as
overall.

    python -m scripts.monitor_drift --reference results/reference_scores.json
    python -m scripts.monitor_drift --record            # store today's scores as the reference
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

RESULTS_DIR = "results"
REFERENCE = os.path.join(RESULTS_DIR, "reference_scores.json")
BINS = 10


def scores_from_db() -> list[dict]:
    from src.database import get_db_connection

    conn = get_db_connection()
    rows = conn.execute("SELECT text, predicted_label, predicted_prob, language, script FROM analysis_logs").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def scores_from_split(name: str = "val") -> list[dict]:
    from src.data import load_split
    from src.model_registry import load_predictor, resolve_model_name

    predictor = load_predictor(resolve_model_name())
    rows = load_split(name)
    probabilities = predictor.predict_binary([row["text"] for row in rows])
    return [
        {
            "text": row["text"],
            "predicted_label": "scam" if p >= 0.5 else "legit",
            "predicted_prob": float(p),
            "language": row["language"],
            "script": row["script"],
        }
        for row, p in zip(rows, probabilities)
    ]


def histogram(probabilities: list[float], bins: int = BINS) -> list[float]:
    edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(probabilities, bins=edges)
    total = counts.sum() or 1
    return (counts / total).tolist()


def psi(reference: list[float], current: list[float], epsilon: float = 1e-6) -> float:
    """Population stability index; < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant."""
    a = np.clip(np.array(reference), epsilon, None)
    b = np.clip(np.array(current), epsilon, None)
    return float(np.sum((b - a) * np.log(b / a)))


def group_key(row: dict) -> str:
    return f"{row.get('language', 'unknown')}/{row.get('script', 'unknown')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="db", choices=["db", "val"])
    parser.add_argument("--record", action="store_true", help="store the current distribution as the reference")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "drift.json"))
    args = parser.parse_args()

    rows = scores_from_db() if args.source == "db" else scores_from_split()
    if len(rows) < 50:
        raise SystemExit(f"only {len(rows)} scored messages; need at least 50 to judge drift")
    probabilities = [row["predicted_prob"] for row in rows]

    if args.record:
        payload = {
            "recorded_from": args.source,
            "rows": len(rows),
            "overall": histogram(probabilities),
            "by_cell": {
                key: histogram([r["predicted_prob"] for r in rows if group_key(r) == key])
                for key in sorted({group_key(r) for r in rows})
            },
        }
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(args.reference, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"recorded reference for {len(rows)} messages -> {args.reference}")
        return

    if not os.path.exists(args.reference):
        raise SystemExit(f"no reference at {args.reference}; run with --record first")
    with open(args.reference, encoding="utf-8") as handle:
        reference = json.load(handle)

    current = histogram(probabilities)
    report = {
        "source": args.source,
        "current_rows": len(rows),
        "reference_rows": reference["rows"],
        "overall_psi": round(psi(reference["overall"], current), 4),
        "scam_rate_now": round(float(np.mean([p >= 0.5 for p in probabilities])), 4),
        "by_cell": {},
    }
    for key, reference_bins in reference["by_cell"].items():
        cell_probabilities = [r["predicted_prob"] for r in rows if group_key(r) == key]
        if len(cell_probabilities) < 20:
            continue
        report["by_cell"][key] = {
            "rows": len(cell_probabilities),
            "psi": round(psi(reference_bins, histogram(cell_probabilities)), 4),
        }

    overall = report["overall_psi"]
    verdict = "stable" if overall < 0.1 else ("moderate drift - investigate" if overall < 0.25 else "SIGNIFICANT DRIFT - retrain or investigate")
    report["verdict"] = verdict
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
