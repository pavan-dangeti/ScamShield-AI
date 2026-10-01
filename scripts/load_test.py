"""Scale check: push a large number of messages through the production path and
report throughput, so the project has a measured "how much can it handle" number
rather than a guess.

Two modes:
* ``--n 10000`` batches messages through the predictor directly (measures model
  throughput without HTTP overhead), reusing real validation messages.
* ``--serve`` sends real HTTP requests to a running ``/api/analyze`` from
  ``--concurrency`` parallel clients, so it includes serialisation, the web
  framework, SQLite logging and contention between requests.  Start the server
  with the rate limit raised, or it will (correctly) answer 429:

      RATE_LIMIT_PER_MINUTE=1000000 SCAMSHIELD_MODEL=tfidf-lr \
          uvicorn src.main:app --port 8000 --workers 2 --log-level warning

Usage::

    python -m scripts.load_test --model xlmr-aug-int8 --n 10000 --batch 32
    python -m scripts.load_test --serve --n 3000 --concurrency 1 8 32
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import statistics
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

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


def http_throughput(url: str, n: int, concurrency: int) -> dict:
    import requests

    from src.data import load_split

    pool = [row["text"] for row in load_split("val")]
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=concurrency, pool_maxsize=concurrency)
    session.mount("http://", adapter)

    def one(index: int) -> tuple[float, str]:
        started = time.perf_counter()
        try:
            status = str(session.post(url, json={"text": pool[index % len(pool)]}, timeout=30).status_code)
        except requests.RequestException as error:
            status = type(error).__name__
        return (time.perf_counter() - started) * 1000, status

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(one, range(n)))
    elapsed = time.perf_counter() - start
    latencies = [ms for ms, _ in results]
    statuses = Counter(status for _, status in results)
    return {
        "mode": "http",
        "url": url,
        "concurrency": concurrency,
        "messages": n,
        "seconds": round(elapsed, 2),
        "requests_per_second": round(n / elapsed, 1),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "error_rate": round(1 - statuses.get("200", 0) / n, 4),
        "status_counts": dict(statuses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xlmr-aug-int8")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--server", default="", help="how the server was started, recorded in the report")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/analyze")
    parser.add_argument("--out", help="default: results/load_test.json, or load_test_http.json with --serve")
    args = parser.parse_args()

    report: dict | list[dict]
    if args.serve:
        report = [http_throughput(args.url, args.n, level) for level in args.concurrency]
    else:
        report = direct_throughput(args.model, args.n, args.batch)
    out = args.out or os.path.join(RESULTS_DIR, "load_test_http.json" if args.serve else "load_test.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if isinstance(report, list):
        argv = sys.argv[1:]
        if "--server" in argv:
            del argv[argv.index("--server") : argv.index("--server") + 2]
        write_markdown(report, out.replace(".json", ".md"), shlex.join(argv), args.server)
    print(json.dumps(report, indent=2))


def write_markdown(levels: list[dict], path: str, argv: str, server: str) -> None:
    with urllib.request.urlopen(levels[0]["url"].replace("/api/analyze", "/api/health"), timeout=10) as reply:
        model = json.load(reply)["model"]
    lines = [
        f"# HTTP load test: {model}",
        "",
        f"Server: `{server}`" if server else "Server: not recorded",
        "",
        f"Load: `python -m scripts.load_test {argv}`",
        "",
        f"Machine: {platform.machine()}, {os.cpu_count()} CPU cores, {platform.system()}; the load generator"
        " shares the machine, so these are lower bounds for the server alone.",
        "",
        "| Concurrent clients | Requests | Req/s | p50 ms | p95 ms | p99 ms | Error rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for level in levels:
        lines.append(
            f"| {level['concurrency']} | {level['messages']} | {level['requests_per_second']} | {level['p50_ms']} "
            f"| {level['p95_ms']} | {level['p99_ms']} | {level['error_rate']:.2%} |"
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
