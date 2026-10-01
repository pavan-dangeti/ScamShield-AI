# HTTP load test: tfidf-lr

Server: `RATE_LIMIT_PER_MINUTE=1000000 SCAMSHIELD_MODEL=tfidf-lr uvicorn src.main:app --port 8011 --workers 4 --log-level warning`

Load: `python -m scripts.load_test --serve --url http://127.0.0.1:8011/api/analyze --n 3000 --concurrency 1 8 32 --out results/load_test_http_tfidf_w4.json`

Machine: arm64, 10 CPU cores, Darwin; the load generator shares the machine, so these are lower bounds for the server alone.

| Concurrent clients | Requests | Req/s | p50 ms | p95 ms | p99 ms | Error rate |
|---|---|---|---|---|---|---|
| 1 | 3000 | 610.8 | 1.52 | 2.38 | 3.13 | 0.00% |
| 8 | 3000 | 1250.0 | 3.23 | 21.26 | 62.47 | 0.00% |
| 32 | 3000 | 1209.5 | 5.02 | 96.66 | 460.28 | 0.00% |
