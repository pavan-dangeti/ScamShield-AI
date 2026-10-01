# HTTP load test: tfidf-lr

Server: `RATE_LIMIT_PER_MINUTE=1000000 SCAMSHIELD_MODEL=tfidf-lr uvicorn src.main:app --port 8011 --workers 1 --log-level warning`

Load: `python -m scripts.load_test --serve --url http://127.0.0.1:8011/api/analyze --n 3000 --concurrency 1 8 32 --out results/load_test_http_tfidf_w1.json`

Machine: arm64, 10 CPU cores, Darwin; the load generator shares the machine, so these are lower bounds for the server alone.

| Concurrent clients | Requests | Req/s | p50 ms | p95 ms | p99 ms | Error rate |
|---|---|---|---|---|---|---|
| 1 | 3000 | 612.5 | 1.52 | 2.37 | 3.14 | 0.00% |
| 8 | 3000 | 823.0 | 4.64 | 23.88 | 96.43 | 0.00% |
| 32 | 3000 | 799.0 | 7.57 | 144.97 | 760.77 | 0.00% |
