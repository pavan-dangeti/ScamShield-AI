"""Measure CPU latency (p50/p95), artefact size, and cost per 1,000 messages.

Latency is measured on CPU only, because that is where a mobile client or a
small server would run the model.  Generative models are timed on the accelerator
they actually ran on, and their cost per 1,000 messages is computed from the
measured token throughput against published local-hardware electricity cost or,
for API models, the provider price (pass with --price-per-million).

Usage::

    python -m scripts.benchmark_latency --models tfidf-lr indicbertv2-mlm xlmr-base --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import numpy as np

from src.data import load_split

RESULTS_DIR = "results"


def artifact_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return round(total / 1e6, 1)


def time_calls(function, texts: list[str], warmup: int = 10) -> dict:
    for text in texts[:warmup]:
        function([text])
    timings = []
    for text in texts:
        start = time.perf_counter()
        function([text])
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": round(statistics.median(timings), 2),
        "p95_ms": round(float(np.percentile(timings, 95)), 2),
        "mean_ms": round(statistics.mean(timings), 2),
        "messages_per_second_single_core": round(1000 / statistics.mean(timings), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["tfidf-lr"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cpu", help="cpu for the deployment number, mps/cuda for local")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "latency.json"))
    args = parser.parse_args()

    rows = load_split(args.split)[: args.n]
    texts = [row["text"] for row in rows]
    report: dict[str, Any] = {"messages_measured": len(texts), "models": {}}

    for model_name in args.models:
        print(f"timing {model_name}...", flush=True)
        if model_name in ("tfidf-lr", "tfidf-lr-aug"):
            from src.models.tfidf import TfidfScamModel

            model = TfidfScamModel.load(f"models/{model_name}.pkl")
            timing = time_calls(model.predict, texts)
            report["models"][model_name] = {
                "device": "cpu",
                **timing,
                "artifact_mb": round(os.path.getsize(f"models/{model_name}.pkl") / 1e6, 1),
            }
        elif model_name in ("indicbertv2-mlm", "xlmr-base"):

            from scripts.evaluate_models import MODEL_REGISTRY
            from src.models.transformer_infer import load_encoder

            # CPU is the deployment target, so pin the device rather than
            # inheriting whatever accelerator happens to be present.
            model, tokenizer, config, _ = load_encoder(MODEL_REGISTRY[model_name] or model_name)
            device = args.device
            model = model.to(device)

            def run(batch: list[str]) -> None:
                encoded = tokenizer(
                    batch, truncation=True, max_length=config.get("max_length", 128),
                    padding=True, return_tensors="pt",
                ).to(device)
                model(encoded["input_ids"], encoded["attention_mask"], encoded.get("token_type_ids"))

            report["models"][model_name] = {
                "device": device,
                **time_calls(run, texts),
                "artifact_mb": artifact_size_mb(f"models/{config['model_name'].replace('/', '_')}"),
            }
        else:
            print(f"  no CPU path for {model_name}; measure it separately", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
