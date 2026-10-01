# HTTP load test: xlmr-aug-int8

Server: `RATE_LIMIT_PER_MINUTE=1000000 SCAMSHIELD_MODEL=xlmr-aug-int8 uvicorn src.main:app --port 8011 --workers 1 --log-level warning`

Load: `python -m scripts.load_test --serve --url http://127.0.0.1:8011/api/analyze --n 1000 --concurrency 1 8 --out results/load_test_http_onnx_w1.json`

Machine: arm64, 10 CPU cores, Darwin; the load generator shares the machine, so these are lower bounds for the server alone.

| Concurrent clients | Requests | Req/s | p50 ms | p95 ms | p99 ms | Error rate |
|---|---|---|---|---|---|---|
| 1 | 1000 | 113.9 | 8.31 | 13.59 | 18.99 | 0.00% |
| 8 | 1000 | 388.1 | 19.32 | 33.62 | 47.64 | 0.00% |
