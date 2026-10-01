# ScamShield design

## Context

India processes billions of UPI transactions a month, and a parallel stream of
payment-fraud SMS and WhatsApp messages arrives in the languages people actually
use: English mixed with Hindi, Tamil, Telugu and Bengali, in native script or
romanised on the fly. Commercial spam and fraud filters are trained almost
exclusively on English and on Western message formats, so they miss the phrasing,
the script mixing, and the urgency-and-authority patterns of an Indian payment
scam.

The repository this project started from detected these messages with a TF-IDF
plus logistic-regression model over 190 hand-written templates. It reported a
perfect F1, because the test set was drawn from the same templates. That is the
problem this design addresses: not "can we train a model" but "how do we know
what the model actually learned".

## Goals

- A corpus whose provenance is known per row: real, synthetic or template, with a
  licence for each source.
- A test set of **real** messages labelled by a human, never used for training,
  tuning or threshold selection.
- Measured, reproducible comparisons between model families, per language and per
  script, with confidence intervals.
- A model that is fast enough and small enough to deploy, chosen on measurement.
- An explanation the reader can audit: which manipulation tactic, and a span of
  the message that supports it.

## Non-goals

- A hosted multi-tenant product. This is a single-process API and a demo.
- Blocking messages automatically. The output is a warning with a reason; a
  human decides.
- Real-time SMS interception on Android. The integration point is an HTTP
  webhook, not a background service.
- Perfect per-language coverage. Kannada, Malayalam, Gujarati, Punjabi and Odia
  are out of scope for the first version; adding them is a language pack plus
  data, not a code change.

## Design

### 1. Data pipeline (`scripts/download_data.py` → `build_dataset` → `check_leakage`)

Sources are declared in code with a licence, a provenance and a SHA-256, and the
hash is re-verified on every download. The build then scrubs PII, drops exact and
near duplicates, and splits **groups** rather than rows, so no template family
can straddle a split.

```
raw source ──▶ PII scrub ──▶ exact dedup ──▶ near-dup grouping (cos ≥ 0.82)
                                                        │
                        frozen test set ◀── excluded ────┤
                                                        ▼
                                       group-aware stratified train / val
```

The near-duplicate threshold is calibrated, not chosen: 0.82 is the value at
which an independent check finds no cross-split pair above 0.90.

**Why group-aware splitting.** The single most important measurement decision in
the project. A random row split over a corpus built from templates puts the same
template in train and test and reports 1.0 F1. Grouping near-duplicates and
splitting whole groups is what turns the reported number into an estimate.

### 2. A single model interface (`src/models/base.py`)

Every model - TF-IDF, fine-tuned encoder, prompted LLM, LoRA-tuned decoder - is a
`Predictor` that maps messages to a scam probability and per-tactic confidences.
One interface means the evaluation harness cannot accidentally favour one family,
and the API can serve any of them from one environment variable.

### 3. Two heads, not two models

Both the baseline and the encoders predict the scam/legit decision and the tactic
set from one backbone. The tactics are the product's differentiator ("which
manipulation tactic is this?"), and predicting them from the same representation
is both cheaper and more consistent than a separate pipeline.

### 4. Fair comparison (`scripts/evaluate_models.py`)

The same split, the same threshold, the same metric code, the same per-cell
breakdown. The generative arms are scored on a seeded stratified subsample
because generation costs seconds per message; the trained arms use the full
split. The subsample is produced by a function shared with the LLM runner, so
every generative arm sees exactly the same rows.

### 5. Robustness as training, not as a layer (`scripts/adversarial.py`)

Six deterministic perturbation families, applied to the text only. The augmented
copies go into training; the validation split stays clean. This is a deliberate
choice over normalising input: normalisation is a fixed function an attacker can
invert, while augmentation changes what the model learned.

### 6. Deployment shape

ONNX int8 encoder behind a model registry, with the TF-IDF model as a fallback
for environments where 278 MB is unacceptable. The API logs to SQLite for
feedback and drift monitoring, and the WhatsApp webhook replies with a two-line
verdict.

The live demo serves the TF-IDF model. That is a measured constraint, not a
preference: a server process holding the int8 encoder uses about 1.1 GB of
memory, and the free hosting tier allows 512 MB.

### 7. Operating it (`src/observability.py`)

- **Logs** are one JSON object per line on stdout, each carrying the request ID
  that is also returned to the caller as `X-Request-ID`. With one service and no
  downstream calls, that ID is the trace; a distributed-tracing SDK would add a
  dependency and nothing it can show. Message text is never logged.
- **Metrics** are Prometheus, at `/metrics`: request count and latency by route
  template, predictions by label, language and script, the served score
  histogram (the live drift signal `scripts/monitor_drift.py` compares against
  its reference), failed analysis-log writes, and which model is loaded.
- **Stored messages are scrubbed** of phone numbers, emails, UPI handles and long
  identifiers before they reach SQLite, because the live service receives real
  people's messages.

Each failure mode is induced on purpose in `tests/test_api.py`:

| Failure | Behaviour |
|---|---|
| Model raises | 500 with a generic message and the request ID; the cause is logged, never returned |
| Model artefact missing or corrupt | Registry logs a warning and serves the TF-IDF baseline; `/api/health` names the model actually serving |
| Database unavailable | Analysis still answers; the failed write is logged and counted in `scamshield_db_errors_total` |
| Oversized, empty or malformed input | 422 before inference runs |
| Client over its rate | 429 |
| Webhook without a valid Twilio signature | 403 |
| Model fails inside the webhook | 200 with a TwiML apology, because a 5xx makes Twilio retry the same message |

## Alternatives considered

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Split unit | near-duplicate group | random row | random row split on templates reports 1.0 F1 and measures nothing |
| Model | XLM-R base + augmentation, int8 ONNX | prompted 7B LLM | prompting over-flags (334/354 scam); a 135M LoRA model beat the 7B by 14 F1 points |
| Baseline | TF-IDF + LR | none - it stays as the comparison point | it is the incumbent, and it is 10 MB and 1.95 ms, which is a real deployment point |
| Tactic head | multi-label head on the same backbone | separate tactic model | cheaper, and the shared representation is consistent between decision and explanation |
| Robustness | augmentation | input normalisation | normalisation is invertible; augmentation changes the model |
| Inference | ONNX Runtime int8 | PyTorch fp32 | 4× smaller and 5× faster on CPU for no measured F1 loss |
| Explanation span | quote only a verbatim substring | quote the top n-gram | a rebuilt word n-gram can be text the sender never wrote |

## Trade-offs

- **Size vs accuracy.** The int8 encoder is 278 MB and 6.1 ms p50; TF-IDF is
  10 MB and 1.95 ms. The TF-IDF model is 0.09 F1 worse on the validation split.
  Both are selectable at runtime.
- **Threshold.** Treating a missed scam as 10× a false alarm moves the optimal
  threshold from 0.45 to 0.30 for the baseline. This is a product judgement; the
  sweep is published so the sensitivity is visible.
- **Latency vs batch.** The load test reports 85 msg/s single-process on CPU for
  the encoder. A deployment that needs more should batch; the ONNX path accepts
  batches of 32.
- **Threads vs processes.** Inference is CPU-bound Python, so request threads
  share one core under the GIL; throughput scales with worker processes, and
  each process holds its own copy of the model. That is cheap at 10 MB and
  expensive at 1.1 GB, which is another reason the baseline is the one on the
  free tier.
- **Evidence spans.** The baseline can attribute a span from its coefficients; a
  contextual encoder has no comparable per-token weight, so it reports generic
  phrasing. That is a real loss of explainability, taken deliberately rather than
  papered over with an invented quote.

## Risks

- **The corpus is mostly synthetic.** This is the dominant risk in the project.
  Every number in the repository is a validation number until the hand-labelled
  test set is frozen, and the README says so in the same place as the results.
- **Bengali dominates the corpus** from a source whose collection method is
  unstated, so the per-language table over-represents Bengali relative to real
  traffic.
- **The held-out test set does not yet exist** for the Indic languages. There is
  no public real corpus for them, so the test set depends on messages the owner
  supplies.
- **Tactic labels are template-derived** for every training row, so tactic
  metrics measure template-family identification and are optimistic.
- **Adversarial coverage is the six families we implemented.** An adaptive
  attack that mutates against the model is stronger and is not implemented.
- **One contributor.** The project has a single author; there is no second pair
  of eyes on the labels, so the test set is single-rater and the inter-rater
  agreement check in `docs/labeling_guidelines.md` cannot be run.

## What would change the design

If the frozen test set showed materially different behaviour from validation -
especially a large drop on the Indic cells - the right response is not to
re-tune, but to state the gap, acquire labelled data for the languages that
matter, and report the per-language table as the headline instead of the
aggregate.
