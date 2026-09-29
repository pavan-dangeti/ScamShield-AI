"""Score every ScamShield model with one harness, so the comparison is fair.

Each model is evaluated on the same split, at the same threshold, with the same
per-language and per-script breakdown, and the same tactic scoring.  Results are
written to ``results/`` as JSON and as the markdown table used in the README.

Usage (from the repository root, after training)::

    python -m scripts.evaluate_models --models tfidf-lr indicbertv2-mlm xlmr-base
    python -m scripts.evaluate_models --models llm-0shot results/llm_0shot_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.data import load_split, subsample_rows
from src.models.tfidf import TfidfScamModel
from src.taxonomy import TACTICS

RESULTS_DIR = "results"

# Short names used in the comparison table, mapped to the artefacts on disk.
MODEL_REGISTRY = {
    "tfidf-lr": None,
    "indicbertv2-mlm": "ai4bharat/IndicBERTv2-MLM-only",
    "xlmr-base": "xlm-roberta-base",
    "muril-base": "google/muril-base",
}


def binary_scores(model_name: str, rows: list[dict]) -> tuple[np.ndarray, list[list[str]]]:
    """Return (scam probability, predicted tactics) for the given model."""
    texts = [row["text"] for row in rows]
    if model_name == "tfidf-lr":
        model = TfidfScamModel.load("models/tfidf-lr.pkl")
        probabilities = model.predict_binary(texts)
        tactics = [
            [t.tactic for t in prediction.tactics] for prediction in model.predict(texts)
        ]
        return probabilities, tactics
    if model_name.startswith("llm-"):
        shots = re.search(r"(\d+)", model_name).group(1)
        path = f"results/llm_{shots}shot_val.jsonl"
        with open(path, encoding="utf-8") as handle:
            verdicts = {json.loads(line)["id"]: json.loads(line) for line in handle if line.strip()}
        probabilities, tactics = [], []
        for row in rows:
            verdict = verdicts.get(row["id"])
            probabilities.append(1.0 if verdict and verdict["label"] == "scam" else 0.0)
            tactics.append(verdict["tactics"] if verdict else [])
        return np.array(probabilities), tactics
    if model_name.startswith("lora-"):
        from src.models.lora_infer import predict_lora

        return predict_lora(model_name.split("-", 1)[1], texts, rows)
    from src.models.transformer_infer import predict_encoder

    hf_name = MODEL_REGISTRY.get(model_name, model_name)
    return predict_encoder(hf_name, texts, rows)


def evaluate(model_name: str, rows: list[dict], threshold: float) -> dict:
    probabilities, predicted_tactics = binary_scores(model_name, rows)
    labels = np.array([1 if row["label"] == "scam" else 0 for row in rows])
    predicted = (probabilities >= threshold).astype(int)

    result = {
        "model": model_name,
        "threshold": threshold,
        "rows": len(rows),
        "scam_support": int(labels.sum()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }
    if len(set(labels.tolist())) > 1:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))

    # Per-cell breakdown: a cell is language + script + true label.
    cells: dict[str, dict] = {}
    for index, row in enumerate(rows):
        key = f"{row['language']}|{row['script']}"
        cell = cells.setdefault(key, {"true": [], "pred": [], "prob": []})
        cell["true"].append(int(labels[index]))
        cell["pred"].append(int(predicted[index]))
        cell["prob"].append(float(probabilities[index]))
    result["by_cell"] = {}
    for key, cell in sorted(cells.items()):
        cell_true = np.array(cell["true"])
        cell_pred = np.array(cell["pred"])
        entry = {
            "rows": int(len(cell_true)),
            "scam_rows": int(cell_true.sum()),
            "precision": float(precision_score(cell_true, cell_pred, zero_division=0)),
            "recall": float(recall_score(cell_true, cell_pred, zero_division=0)),
            "f1": float(f1_score(cell_true, cell_pred, zero_division=0)),
        }
        if 0 < cell_true.sum() < len(cell_true):
            entry["roc_auc"] = float(roc_auc_score(cell_true, np.array(cell["prob"])))
        result["by_cell"][key] = entry

    # Tactic scoring, restricted to rows that carry gold tactic labels.
    gold_rows = [index for index, row in enumerate(rows) if row.get("tactics")]
    if gold_rows:
        gold = np.zeros((len(gold_rows), len(TACTICS)))
        pred = np.zeros((len(gold_rows), len(TACTICS)))
        for position, index in enumerate(gold_rows):
            for tactic in rows[index]["tactics"]:
                gold[position, TACTICS.index(tactic)] = 1
            for tactic in predicted_tactics[index]:
                if tactic in TACTICS:
                    pred[position, TACTICS.index(tactic)] = 1
        result["tactics"] = {
            "labelled_rows": len(gold_rows),
            "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(gold, pred, average="micro", zero_division=0)),
            "per_tactic_f1": {
                tactic: float(f1_score(gold[:, i], pred[:, i], zero_division=0))
                for i, tactic in enumerate(TACTICS)
            },
        }
    else:
        result["tactics"] = {"labelled_rows": 0, "note": "no gold tactic labels in this split"}

    # Explanation faithfulness: does the quoted evidence actually occur in the message?
    if model_name == "tfidf-lr":
        model = TfidfScamModel.load("models/tfidf-lr.pkl")
        checked, faithful, attributed = 0, 0, 0
        for index, row in enumerate(rows):
            for tactic in predicted_tactics[index]:
                span = model.evidence(row["text"], tactic)
                checked += 1
                if span and span.lower() in row["text"].lower():
                    faithful += 1
                if span and span != "suspicious phrasing":
                    attributed += 1
        result["evidence_in_message"] = {
            "checked": checked,
            "faithful": faithful,
            "attributed": attributed,
            "faithful_rate": round(faithful / checked, 4) if checked else None,
            "attribution_rate": round(attributed / checked, 4) if checked else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tfidf-lr"])
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="score every model on the same seeded stratified subsample (for slow generative models)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = load_split(args.split)
    if args.subsample:
        rows = subsample_rows(rows, args.subsample, args.seed)
    out = args.out or os.path.join(RESULTS_DIR, "model_comparison.json")
    if args.subsample:
        out = os.path.join(RESULTS_DIR, f"model_comparison_subsample{args.subsample}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    for model_name in args.models:
        print(f"evaluating {model_name} on {len(rows)} {args.split} rows...", flush=True)
        result = evaluate(model_name, rows, args.threshold)
        results.append(result)
        print(
            f"  P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f} "
            f"tactic_macro_f1={result['tactics'].get('macro_f1')}",
            flush=True,
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "subsample": args.subsample or None,
        "seed": args.seed,
        "rows": len(rows),
        "results": results,
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    lines = [
        "# Model comparison",
        "",
        f"Split: `{args.split}` ({len(rows)} rows"
        + (f", seeded stratified subsample of {args.subsample}" if args.subsample else "")
        + f") · threshold {args.threshold} · generated {payload['generated_at'][:10]}",
        "",
        "| Model | Precision | Recall | F1 | ROC-AUC | Tactic macro-F1 | Tactic rows |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        tactics = result["tactics"]
        tactic_cell = f"{tactics['macro_f1']:.4f}" if tactics.get("labelled_rows") else "n/a"
        lines.append(
            f"| {result['model']} | {result['precision']:.4f} | {result['recall']:.4f} | "
            f"{result['f1']:.4f} | {result.get('roc_auc', float('nan')):.4f} | {tactic_cell} | "
            f"{tactics.get('labelled_rows', 0)} |"
        )
    with open(out.replace(".json", ".md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {out} and {out.replace('.json', '.md')}")


if __name__ == "__main__":
    main()
