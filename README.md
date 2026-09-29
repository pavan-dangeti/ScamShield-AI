# 🛡️ ScamShield — Multilingual UPI/Payment Scam Detector with Explainable Tactics

> A real-time, explainable scam-detection system for India's most common financial fraud messages — built to understand **code-mixed regional languages** (Hinglish, Tanglish, Tenglish, Banglish, and more), not just English.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 The Problem

India processes over 15 billion UPI transactions a month — and alongside that volume comes a flood of payment scams via SMS and WhatsApp: fake KYC alerts, "you've won a prize" messages, fraudulent refund requests, and OTP-phishing attempts.

Most commercial spam/fraud filters are trained primarily on **English-language data from Western contexts**. They miss the specific phrasing patterns used in **code-mixed Indian languages** — messages that switch between English and Hindi, Tamil, Telugu, or Bengali mid-sentence, often in romanized script ("Unga account 30 nimishathula block aagidum...").

**ScamShield** closes that gap. It doesn't just say "this is a scam" — it explains **which manipulation tactic** is being used (urgency, fake authority, false rewards, credential phishing, etc.), in the language the message was written in, and works live over WhatsApp.

---

## ✨ Key Features

- **Binary scam classification** — TF-IDF + Logistic Regression pipeline, trained and evaluated with per-language/script breakdowns
- **Multi-label tactic detection** — identifies *why* a message is dangerous: urgency, authority impersonation, false reward, loss aversion, credential phishing, suspicious links
- **Evidence extraction** — highlights the exact phrase in the message that triggered each tactic flag
- **Multilingual by design** — extensible "language pack" architecture (JSON-based) currently covering English, Hindi/Hinglish, Tamil/Tanglish, Telugu/Tenglish, and Bengali/Banglish, with native-script and romanized variants
- **Rule-based language/script detection** — Unicode-range detection for native scripts + dictionary-based fuzzy matching for romanized text, with graceful fallback for unsupported languages
- **Live WhatsApp integration** — powered by Twilio's WhatsApp Sandbox; forward any suspicious message and get an instant analysis reply
- **Self-aware evaluation** — `metrics_report.md` documents exactly where the model performs well vs. poorly, broken down by language and script, including low-support warnings rather than misleading perfect scores
- **Feedback loop** — every analysis is logged to SQLite with a correction mechanism, designed to support future retraining

---

## 🌐 Live Demo

**Web App**
https://scamshield-ai-glhm.onrender.com/

**API Docs**
https://scamshield-ai-glhm.onrender.com/docs

---

## 🏗️ Architecture

```
┌──────────────────┐      ┌───────────────────┐      ┌────────────────────────┐
│  WhatsApp / SMS   │─────▶│   FastAPI Backend  │─────▶│   Detection Pipeline    │
│  (Twilio webhook) │◀─────│   (main.py)        │◀─────│  - Script/lang detector │
└──────────────────┘      │                    │      │  - Binary classifier    │
                           │   Web Dashboard    │      │  - Tactic classifier    │
┌──────────────────┐      │   (static/)        │      │  - Evidence extractor   │
│  Browser UI       │◀────▶│                   │      │  - Explanation builder  │
└──────────────────┘      └─────────┬──────────┘      └────────────────────────┘
                                     │
                                     ▼
                           ┌───────────────────┐
                           │  SQLite (feedback) │
                           │  source: web/sms   │
                           └───────────────────┘
```

### Detection Pipeline

1. **Script & Language Detection** (`detector.py`) — Unicode block ranges identify native scripts (Devanagari, Tamil, Telugu, Bengali, etc.); a fuzzy-matched dictionary of common transliterated words identifies romanized regional languages (Hinglish, Tanglish, Tenglish, Banglish). Falls back to `unsupported`/`ambiguous` gracefully.
2. **Binary Classification** (Model A) — TF-IDF (word + character n-grams) + Logistic Regression, calibrated for probability output.
3. **Multi-Label Tactic Classification** (Model B) — One-vs-rest TF-IDF + Logistic Regression per tactic category, with evidence spans extracted via top-weighted n-gram matching.
4. **Explanation Synthesis** — Template-based natural-language explanation combining the detected tactics and evidence.

---

## 🌐 Language Coverage

| Language | Native Script | Romanized | Status |
|---|---|---|---|
| English | — | ✅ | Production-ready |
| Hindi | ✅ Devanagari | ✅ Hinglish | Production-ready |
| Tamil | ✅ Tamil script | ✅ Tanglish | Production-ready |
| Telugu | ✅ Telugu script | ✅ Tenglish | Growing dataset |
| Bengali | ✅ Bengali script | ✅ Banglish | Growing dataset |
| Kannada, Malayalam, Gujarati, Punjabi, Odia | Script detection only | — | Roadmap (see `docs/adding_a_language.md`) |

Adding a new language requires **zero code changes** — just a new JSON language pack. See [`docs/adding_a_language.md`](docs/adding_a_language.md).

---

## 🗂️ Data and the hand-verified test set

ScamShield separates three things that are easy to confuse:

- **Training data** — licence-checked corpora, template packs, and (training-only) LLM-written
  synthetic messages. Sources, licences and provenance: [`docs/data_sources.md`](docs/data_sources.md).
- **The hand-verified test set** — **real** messages only, labelled by a human against
  [`docs/labeling_guidelines.md`](docs/labeling_guidelines.md), never used for training or
  threshold tuning. Composition and gaps: `docs/test_set.md`.
- **Synthetic data** — LLM-generated, always marked, and **never** allowed in the test set.

Build the corpus and prove it is clean:

```bash
python -m scripts.download_data        # fetch sources, SHA-256 pinned
python -m scripts.build_dataset           # PII scrub, dedup, near-dup-aware train/val split
python -m scripts.check_leakage           # no message appears (even near-duplicated) across splits
```

Prepare the test candidate pool, and help fill the languages that have no public corpus:

```bash
python -m scripts.make_test_candidates    # unlabelled, real-only candidates
python -m scripts.import_user_messages --input my_messages.csv   # add your own real messages
```

Label with `docs/labeling/labeler.html` (opens in a browser, works offline), export
`labels.jsonl`, then freeze the set:

```bash
python -m scripts.freeze_test             # -> data/processed/test.jsonl + docs/test_set.md
```

> **Honest limitation:** the only licence-clean, real, labelled scam corpus available for this
> task is the UCI SMS Spam Collection, which is **English**. For Hindi, Tamil, Telugu and
> Bengali there is no public real corpus, so those test cells are filled from messages the
> owner supplies. This gap is documented rather than papered over with synthetic data.

---

## 📊 Evaluation

The current production model (TF-IDF + logistic regression) is trained and evaluated on the
template-generated corpus described in [`docs/data_sources.md`](docs/data_sources.md). Its
metrics are written to `models/metrics_report.md` on every `python -m src.train` run, with a
per-language and per-script breakdown and low-support warnings.

**Know what the numbers mean.** The legacy corpus expands a small set of hand-written
templates with random placeholders, and the model is evaluated on held-out samples drawn from
those same templates. That measures template memorisation, not real-world accuracy — which is
exactly why the project now separates a licence-checked corpus and a hand-verified test set
from the training data.

The reproducible data pipeline is the source of truth for evaluation going forward:

```bash
python -m scripts.download_data      # fetch licence-checked sources, pinned by SHA-256
python -m scripts.build_dataset         # scrub, dedup, near-dup-aware split -> data/processed/
python -m scripts.check_leakage         # prove no message leaks across splits
```

Every figure quoted in this README comes from a script in this repository, with the exact
command shown next to it. See [`docs/data_card.md`](docs/data_card.md) for what is in the
corpus, what is real vs. synthetic vs. unverified, and the known biases.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/scamshield.git
cd scamshield
pip install -r requirements.txt
cp .env.example .env
```

### Generate the dataset and train the models

```bash
python scripts/dataset_generator.py
python -m src.train
```

This produces `data/scam_dataset.csv`, the trained model artifacts in `models/`, and `models/metrics_report.md`.

> This is the original template-based pipeline, kept so the current production model still
> trains. The licence-checked corpus and its hand-verified test set are built separately —
> see [`docs/data_sources.md`](docs/data_sources.md) and
> [`docs/data_card.md`](docs/data_card.md) for `python -m scripts.download_data`,
> `python -m scripts.build_dataset` and `python -m scripts.check_leakage`.

### Run the server

```bash
python -m src.main
```

Visit `http://127.0.0.1:8000` for the web dashboard, or `http://127.0.0.1:8000/docs` for the interactive API documentation.

### (Optional) Connect a live WhatsApp number

See [`docs/twilio_setup.md`](docs/twilio_setup.md) for the full Twilio Sandbox + ngrok setup — takes about 10 minutes and is completely free.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | POST | Analyze a message; returns scam probability, language/script, detected tactics with evidence, and a plain-language explanation |
| `/api/sms-webhook` | POST | Twilio webhook for WhatsApp/SMS — auto-replies with an analysis summary |
| `/api/feedback` | POST | Submit a correction for a previously analyzed message |
| `/api/stats` | GET | Aggregate statistics: tactic frequency, language distribution, web vs. messaging traffic |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Unga account 30 nimishathula block aagidum, ippo click pannunga: bit.ly/xyz"}'
```

---

## 📁 Project Structure

```
scamshield/
├── data/
│   ├── language_packs/        # Per-language scam/legit templates (extensible)
│   ├── raw/                   # downloaded sources (not committed, re-fetched by script)
│   ├── processed/             # built train/val (+ frozen test) splits (not committed)
│   ├── synthetic/             # locally generated, training-only messages
│   ├── test_candidates/       # unlabelled pool for the hand-verified test set
│   └── scam_dataset.csv       # legacy template dataset (kept for the current model)
├── docs/
│   ├── data_sources.md        # every source: licence, provenance, eligibility
│   ├── data_card.md           # corpus composition, biases, limitations
│   ├── labeling_guidelines.md # scam vs legit rules + per-tactic definitions
│   ├── labeling/labeler.html  # keyboard-driven test-set labelling tool
│   ├── adding_a_language.md
│   └── twilio_setup.md
├── models/
│   ├── binary_classifier.pkl
│   ├── tactic_classifier.pkl
│   ├── vectorizers.pkl
│   └── metrics_report.md
├── scripts/
│   ├── download_data.py       # fetch licence-checked sources (SHA-256 pinned)
│   ├── build_dataset.py       # scrub, dedup, near-dup-aware train/val split
│   ├── check_leakage.py       # prove no leakage across splits
│   ├── generate_synthetic.py  # LLM-written training data (training only)
│   ├── make_test_candidates.py# assemble the unlabelled test candidate pool
│   ├── import_user_messages.py# import owner-contributed real messages
│   ├── freeze_test.py         # freeze hand-verified test set + write docs/test_set.md
│   ├── scrub_pii.py           # PII scrubbing
│   └── dataset_generator.py   # legacy template expansion
├── src/
│   ├── detector.py             # Script/language detection
│   ├── train.py                # Training + evaluation
│   ├── inference.py            # Prediction + explanation logic
│   ├── taxonomy.py             # manipulation-tactic taxonomy (single source of truth)
│   ├── database.py             # SQLite logging
│   └── main.py                 # FastAPI app
├── tests/                      # data-pipeline + integration tests
└── static/                     # Web dashboard (HTML/CSS/JS)
```

---
## 📸 Screenshots

### 🏠 Homepage

![Homepage](https://drive.google.com/file/d/1qK6HmU9a2i-933J3QzXRBhWCaiEdphDm/view?usp=sharing)

---

### 🔍 Scam Detection

![Scam Detection](https://drive.google.com/file/d/1FmY6CUJPL-o2p6yuM0oF9-QCR0SoYBok/view?usp=sharing)

---

### 📊 Analytics Dashboard

![Dashboard](https://drive.google.com/file/d/1jReU3FRVR_3VNFxguShv0gK6lNKf7K0H/view?usp=sharing)

---

### 📖 API Documentation

![Swagger](https://drive.google.com/file/d/121t7nmyuaSdT5cWZV1-eydyr9qrYylef/view?usp=sharing)

---

### 📱 WhatsApp Integration

![WhatsApp Demo](https://drive.google.com/file/d/16Kja6olxFLfeUYJ9JQOmx7rah621BVD-/view?usp=sharing)

---

## 🔭 Limitations & Future Work

- **Dataset size and balance** vary significantly by language — Telugu and Bengali coverage is currently smaller than Hindi/Tamil/English. The metrics report tracks this transparently.
- **Synthetic-to-real generalization**: the dataset is template-generated and supplemented with a small set of real-world examples; a larger real-world corpus would improve robustness against novel scam phrasing.
- **Romanized text normalization** currently uses fuzzy dictionary matching; a learned transliteration model could improve coverage of spelling variants.
- **Planned**: expand to Kannada, Malayalam, Gujarati, Punjabi, and Odia language packs; multilingual explanation output (currently English-only); a lightweight Android client for on-device SMS scanning.

---

## 🧠 Why This Project

This started as an exploration of a real, underexplored gap: most fraud-detection systems are evaluated almost entirely in English, despite the fact that hundreds of millions of users in India communicate — and get scammed — in code-mixed regional languages. ScamShield is an attempt to build a system that is honest about *where it works and where it doesn't*, rather than optimizing for a single headline accuracy number.

---

## 📄 License

MIT
