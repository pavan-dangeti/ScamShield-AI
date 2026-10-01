# Data sources

Every source ScamShield downloads is listed here with its licence, its
provenance, and whether it is allowed in the hand-verified test set.  The rules
are enforced in code, not just on paper:

* a source with no clear licence is **not** used;
* synthetic (LLM-written) data is **training only**;
* data whose provenance is unstated in the source's own documentation is
  **training only**, because it cannot be shown to be real.

Download and verify everything with:

```bash
python -m scripts.download_data
```

Hashes below are recorded in `data/raw_manifest.json` and re-checked on every
run.  Raw files land in `data/raw/` and are never committed.

---

## Used for training, and (only where marked) for the test set

### 1. UCI SMS Spam Collection

| Field | Value |
|---|---|
| Home | <https://archive.ics.uci.edu/dataset/228/sms+spam+collection> |
| File | `SMSSpamCollection` |
| SHA-256 | `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |
| Rows | 5,572 English SMS (747 spam / 4,825 ham) |
| Licence | **CC BY 4.0** |
| Provenance | **Real** messages collected from UK mobile users (2004-2005) |
| Use | Training **and** test candidates (English cell) |
| Citation | Almeida, Gómez Hidalgo, Yamakami (2011), *Contributions to the Study of SMS Spam Filtering*, ACM DocEng |

Note on labels: the corpus labels everything as `spam` or `ham`.  ScamShield maps
`spam -> scam` and `ham -> legit` for training.  Test rows are re-labelled by the
owner against `docs/labeling_guidelines.md`, because "spam" and "fraud" are not
identical (an aggressive advert is spam but not a scam).  Any test row taken from
this corpus is scrubbed of phone numbers, emails, UPI handles and link query
strings by `src/pii.py`.

### 2. Bengali SMS Smishing Dataset (training only)

| Field | Value |
|---|---|
| Home | <https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset> |
| Files | `train/validation/test-00000-of-00001.parquet` |
| SHA-256 | see `data/raw_manifest.json` |
| Rows | 7,005 (smish 2,809 / normal 2,488 / promo 1,708) in Bengali, Banglish, English and code-mixed |
| Licence | **MIT** (dataset card) |
| Provenance | **Unstated.** The card and the associated thesis repository give statistics and a class list, but neither says whether the messages were collected from real traffic or machine-written |
| Use | Training only |
| Citation | Islam, S. (2025). *SmishDetect-LLM: Multilingual SMS Phishing Detection Using Large Language Models.* Murdoch University |

Label mapping used by the build: `smish -> scam`, `normal -> legit`,
`promo -> legit`.  This corpus is the only source of Bengali and Banglish text
in the project; if the author confirms the collection method it can be promoted
to the test set, which is the single cheapest way to close the largest gap in
the evaluation.

### 3. Indian Scam SMS (synthetic, audited) — training only

| Field | Value |
|---|---|
| Home | <https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited> |
| File | `synthetic_audited.csv` |
| SHA-256 | `45bde5f156a1a42698bab209e32bcd41b3639013d772f644cc3208ed39fb61e2` |
| Rows | 1,580 (scam 797 / safe 783) across en, hi, hinglish, bn_en, ta_en, te_en, mr_en |
| Licence | **CC BY 4.0** |
| Provenance | **Synthetic** - every row was written by an LLM. The card itself says: "Do not use it to estimate real-world accuracy." |
| Use | Training only |

Its value is the *hard negatives*: real-looking bank, courier, bill and job
messages that are not scams.  `mr_en` (Marathi) is dropped because Marathi is
outside the project's language scope.

### 4. Project language packs — training only

`data/language_packs/*.json` holds the 190 hand-written templates that predate
this work.  `scripts/dataset_generator.py` expands them with random bank names,
amounts, links and phone numbers.  All 2,700 rows are **synthetic by
construction** and carry no information beyond those 190 strings, which is
precisely why the pre-upgrade metrics read 1.0.  They are kept because they are
the only rows with a clean, taxonomy-level tactic label, and because
`src/detector.py` needs `romanized_words` from the same files.

### 5. Locally generated synthetic messages — training only

`python -m scripts.generate_synthetic` writes `data/synthetic/*.jsonl` with a
local model (default `mlx-community/Qwen2.5-3B-Instruct-4bit` on Apple
silicon, or any OpenAI-compatible endpoint).  Every row records the generator,
the model and the prompt version.  Generation is gated on script compliance
(the right script for the language, no contamination from other scripts), no
placeholder artefacts, and - for messages generated as legitimate - no
credential request, because a small model will otherwise "helpfully" write a
scam and label it safe.

---

## Considered and rejected

| Source | Why not |
|---|---|
| `haxrits/adaption-sms-fraud-classification` (HF) | No licence declared |
| `BAJIRAO/spam_data` (HF) | No licence declared, no dataset card |
| `dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset` (HF) | GPL-labelled but of unclear derivation; the hi/bn rows duplicate corpora with no stated provenance |
| `FredZhang7/all-scam-spam` (HF) | Apache-2.0, but the corpus is e-mail spam (Enron / SpamAssassin), not SMS and not Indic |
| `mshenoda/spam-messages` (HF) | MIT, but a cleaned derivative of the UCI corpus; no extra provenance |
| `Ridham115` test split usage | The dataset is a single file with no official split, and it is synthetic: it cannot be used to measure anything |
| Government advisories and press articles quoting scam SMS | No dataset licence; short quotations are used in `docs/labeling_guidelines.md` as illustrations only, never as data |
| Own phone's spam folder | Not redistributable and personal. The owner imports a scrubbed selection with `scripts/import_user_messages.py`; only PII-free rows are eligible, and the owner decides what to share |

## What this means for the test set

Real, licence-clean, labelled scam messages exist **only in English** (UCI).  For
Hindi, Tamil, Telugu and Bengali the project has no public corpus, so the
hand-verified test set for those languages is built from messages the owner
supplies.  This is a limitation of the ecosystem, not a shortcut, and it is
stated in `docs/data_card.md` and the README rather than papered over with
synthetic test data.
