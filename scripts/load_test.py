"""Scale check: push a large number of messages through the production path and
report throughput, so the project has a measured "how much can it handle" number
rather than a guess.

Two modes:
* ``--n 10000`` batches messages through the predictor directly (measures model
  throughput without HTTP overhead), reusing real validation messages.
* ``--serve`` hits a running ``/api/analyze`` over HTTP to include request overhead
  and the rate limiter (points at RATE_LIMIT_PER_MINUTE; raise it for the test).

Usage::

    python -m scripts.load_test --model xlmr-aug-int8 --n 10000 --batch 32
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

RESULTS_DIR = "results"


def direct_throughput(model_name: str, n: int, batch: int) -> dict:
    from src.data import load_split
    from src.model_registry import load_predictor

    predictor = load_predictor(model_name)
    pool = [row["text"] for row in load_split("val")]
    if not pool:
        raise SystemExit("no validation rows; run scripts.build_dataset first")
    texts = [pool[i % len(pool)] for i in range(n)]

    start = time.perf_counter()
    processed = 0
    latencies = []
    for start_index in range(0, n, batch):
        chunk = texts[start_index : start_index + batch]
        batch_start = time.perf_counter()
        predictor.predict(chunk)
        latencies.append((time.perf_counter() - batch_start) * 1000)
        processed += len(chunk)
    elapsed = time.perf_counter() - start
    return {
        "mode": "direct",
        "model": model_name,
        "messages": processed,
        "batch_size": batch,
        "seconds": round(elapsed, 2),
        "messages_per_second": round(processed / elapsed, 1),
        "mean_batch_ms": round(statistics.mean(latencies), 2),
        "p95_batch_ms": round(float(np.percentile(latencies, 95)), 2),
    }


def http_throughput(url: str, n: int) -> dict:
    import requests

    from src.data import load_split

    pool = [row["text"] for row in load_split("val")]
    latencies = []
    start = time.perf_counter()
    for index in range(n):
        payload = {"text": pool[index % len(pool)]}
        one = time.perf_counter()
        requests.post(url, json=payload, timeout=30)
        latencies.append((time.perf_counter() - one) * 1000)
    elapsed = time.perf_counter() - start
    return {
        "mode": "http",
        "url": url,
        "messages": n,
        "seconds": round(elapsed, 2),
        "requests_per_second": round(n / elapsed, 1),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xlmr-aug-int8")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/analyze")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "load_test.json"))
    args = parser.parse_args()

    report = http_throughput(args.url, args.n) if args.serve else direct_throughput(args.model, args.n, args.batch)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
