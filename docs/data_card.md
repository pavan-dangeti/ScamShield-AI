# Data card - ScamShield message corpus

**Version:** 1 · **Frozen training pool:** `data/processed/` (rebuilt by
`python -m scripts.build_dataset`) · **Hand-verified test set:**
`data/processed/test.jsonl`, described in `docs/test_set.md`

---

## 1. Purpose

Train and evaluate a scam detector for short SMS and WhatsApp messages in
English and in code-mixed Indian languages (Hindi/Hinglish, Tamil/Tanglish,
Telugu/Tenglish, Bengali/Banglish), and evaluate whether the explanation of
*which* manipulation tactic was used is correct.

## 2. Composition of the training pool

Rebuilt with `python -m scripts.build_dataset`; every count below is read from
`data/processed/stats.json`, which that command writes.

| Split | Rows |
|---|---|
| train | 13,923 |
| val | 1,554 |
| **total** | **15,477** |

| By provenance | Rows | By script | Rows | By label | Rows |
|---|---|---|---|---|---|
| unverified provenance (Bengali corpus) | 6,855 | latin | 7,130 | legit | 10,278 |
| real (UCI) | 4,904 | romanized | 5,464 | scam | 5,199 |
| template (project packs) | 2,180 | native | 2,883 | | |
| synthetic (audited corpus + generated) | 1,538 | | | | |

By language: English 7,130 · Bengali 5,783 · Hindi 1,317 · Tamil 635 · Telugu 612.

300 UCI messages are held out of the pool entirely as the test reserve (see
`docs/test_set.md`), so the real rows above are 4,904 rather than 5,144.

Tactic labels exist only for rows whose label was produced together with a
template: the 2,180 project template rows and the generated synthetic rows.
Roughly half of the scam rows in the pool therefore have no tactic label, and
tactic-level evaluation is limited to what the hand-labelled test set
provides.

## 3. How the pool was built

1. Download every source with `python -m scripts.download_data`, verified by
   SHA-256 against `data/raw_manifest.json`.
2. Hold out the UCI test reserve (300 messages) before any deduplication.
3. Scrub personal data from every row (`src/pii.py`): emails, UPI
   handles, phone numbers, 12-16 digit identifiers, and link query strings
   become `[EMAIL]`, `[UPI_ID]`, `[PHONE]`, `[ID]`. The link host and path are
   kept because the link is the signal.
4. Drop exact duplicates after Unicode NFKC normalisation (1,038 rows) and drop
   any message that carries two different labels in different sources (0 rows
   in this build).
5. Group near-duplicates (character 4-5 gram cosine >= 0.82) and keep each group
   inside one split. 13,548 groups for 15,477 rows.
6. Assign whole groups to train/val, stratified by language, script and label.
7. Exclude every training row that is a near-duplicate of a frozen test
   message.

`python -m scripts.check_leakage` re-checks step 4 on the written files and
fails if any train/val/test pair is at or above 0.90 cosine.

## 4. The hand-verified test set

Rules, enforced in `scripts/freeze_test.py`: real messages only, no synthetic or
unverified-provenance rows, no duplicates, every scam labelled with at least one
tactic, one label per language and script, and a reported gap for any cell below
the target count.  The test set is never used for training, model selection, or
threshold tuning; the decision threshold is chosen on the validation split.

The owner labels with `docs/labeling/labeler.html` against
`docs/labeling_guidelines.md` v1, blind to the source and its original label.
Label quality is measured two ways and reported in `docs/test_set.md`:
self-agreement against a blind re-check of ~10% of items, and agreement with the
original UCI annotation as an independent second rater.

**Current state: not yet frozen.** Only the English cell has a licence-clean
real source (UCI). The other eight cells wait on owner-supplied messages.

## 5. Personal data

- All real messages pass through the scrubber before they are stored.
- The scrubber cannot recognise personal **names**, employers, or places.  The
  owner is instructed to remove them by hand before importing.
- Test rows record `provenance: real`, `synthetic: false`, source and licence,
  and are never silently mixed with synthetic rows.
- The running application stores analysed message text in SQLite. That is
  addressed in the privacy work of the production phase, not here.

## 6. Known biases and limitations

1. **Only English has real data.** UCI is UK, 2004-2011, mobile-operator SMS.
   Indian payment scams (UPI, KYC, free-data bait) are barely represented.
2. **Bengali is dominant and unverified.** 44% of the pool comes from one
   corpus whose collection method is not stated; it dominates because it is
   available, not because it is representative.
3. **Hindi, Tamil and Telugu are thin** (about 1,300, 630 and 610 rows), and
   nearly all of it is synthetic. Expect weak real-world behaviour and treat
   per-language numbers for these as indicative only.
4. **Label noise by construction.** Synthetic rows are labelled by the model
   that wrote them, checked only by scripted gates (script compliance,
   artefacts, credential demands in "legitimate" rows).
5. **Class imbalance**: roughly 2:1 legit:scam in the pool.
6. **Class imbalance in the sources**: the UCI "spam" class bundles fraud with
   aggressive advertising; the test set separates them, the training labels do
   not.
7. **No adversarial variants yet** - homoglyphs, spacing tricks, transliteration
   drift and link disguises are added in the robustness phase.
8. **Tactic labels are template-derived** for all training rows, so tactic
   detection is really "which template family is this", until the test set
   provides independently labelled tactics.
9. **A 3B local model cannot write convincing native-script Indic text.** The
   generation gates reject gibberish, wrong-script output and cross-script
   contamination, which is the correct behaviour, but it means the committed
   synthetic batch is English-heavy (85 generated rows survive the gates, mostly
   English and Hindi native). Usable native Tamil/Telugu/Bengali synthetic data
   needs a stronger model - a 7B-class local model, or an API model with the
   owner's approval - and is deliberately not faked with templates.

## 7. Intended use and limits

Intended: research and evaluation of scam-message classification and tactic
explanation for Indian-language, code-mixed traffic.

Not suitable for: automated blocking decisions without human review; any claim
of performance on languages or message types absent from the table above;
training a model to send messages.

## 8. Ethics and licensing

All sources carry an explicit licence (CC BY 4.0, MIT, or the project's own
MIT-licensed template packs); see `docs/data_sources.md` for the per-source
record and citation. Reuse of this corpus should keep the attribution in
`docs/data_sources.md`.  Scam messages are studied to help people avoid them;
no script in this repository produces or sends scam messages.
