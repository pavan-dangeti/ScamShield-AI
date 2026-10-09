# Model card - ScamShield scam detector

**Model:** `xlmr-base-aug`, XLM-R base fine-tuned with a binary head and a
six-way tactic head, trained with adversarial augmentation, exported to ONNX and
dynamically quantised to int8. **Version:** 1.0 · **Date:** 2026-09-30 ·
**Published:** [pavan-dangeti/scamshield-xlmr-int8](https://huggingface.co/pavan-dangeti/scamshield-xlmr-int8) (CC BY-NC 4.0)

## Intended use

Flagging short SMS and WhatsApp messages that are likely to be payment fraud, in
English and in code-mixed Indian languages, and naming the manipulation tactic so
a reader can decide. Intended users are people checking a suspicious message and
operators triaging a message queue.

**Out of scope:** automatic blocking, any claim about languages the model was not
evaluated on, and use as the sole control in a financial workflow.

## Measured performance

Validation split, 1,554 rows, threshold 0.5, from
`python -m scripts.evaluate_with_ci --models xlmr-base-aug`:

| Metric | Value |
|---|---|
| Precision / Recall / F1 | 0.9960 / 1.0000 / 0.9980 |
| F1 95% CI (2,000 bootstrap) | 0.995 - 1.000 |
| Macro F1 over language/script cells | 0.9992 |
| Brier score | 0.0010 |
| Expected calibration error | 0.0016 |
| Tactic macro-F1 | 0.987 (230 rows with tactic labels) |
| Worst adversarial attack (spacing) | F1 0.985, mean flip rate 0.003 |
| CPU latency (ONNX int8, one message) | 4.44 ms p50, 9.72 ms p95 (`results/latency.json`) |
| Throughput (1 process, one message at a time) | 178 msg/s; 91 msg/s in batches of 32 (`results/load_test_direct_int8_b*.json`) |
| Artefact | 278 MB int8 ONNX |
| Server memory with this model loaded | ~1.1 GB, so it does not fit a 512 MB free tier |

Under a 10:1 cost of a missed scam to a false alarm, the cost-optimal threshold is
0.75, which is also the best-F1 threshold for this model. The full threshold sweep
is in `results/evaluation_with_ci_aug.json`.

## What these numbers are not

**They are validation numbers on a corpus that is mostly synthetic.** The
hand-verified real-message test set is not frozen, so no figure here measures the
model on real traffic. The corpus's own data card (`docs/data_card.md`) documents
the composition: 4,904 real English messages from UCI, 6,855 Bengali messages
from a source whose collection method is unstated, 2,180 template rows and 1,538
synthetic rows.

Three consequences the reader should apply:

1. Bengali native and romanized cells are over-represented relative to real
   traffic, because the Bengali corpus is the largest available one.
2. Tactic macro-F1 is measured on template-derived labels, so it measures
   "which template family is this" and is optimistic by construction.
3. The real-data rows are UK mobile SMS from 2004-2011. Indian payment scams
   (UPI, KYC, free-data bait) are barely represented.

## Robustness

Six deterministic perturbation families - leetspeak, look-alike characters,
romanised spelling drift, spacing tricks, emoji insertion, disguised links. The
model was trained with augmentation over four of them. Worst case after
augmentation: F1 0.993 on leetspeak, mean flip rate 0.003. See
`docs/robustness.md`, including what these attacks do not cover.

## Fairness and bias

- The detector is more reliable on scripts with more data. Telugu and Tamil cells
  have 24-38 validation rows, so their per-language figures carry intervals wide
  enough to be uninformative; that is stated in the results table rather than
  hidden behind an aggregate.
- Messages that are not scams in English (UCI) are the only real negatives, and
  they are British banking and carrier messages. A model tuned to this corpus
  should be expected to over-flag legitimate marketing it has not seen.
- No demographic attributes are used or inferred, and none were used in the
  corpus. The PII scrubber removes phone numbers, emails, UPI handles and long
  identifiers from real messages; it cannot remove personal names.

## Limitations that matter in deployment

1. **Tactic explanations are weaker than the decision.** The tactic head is
   calibrated against template labels, and a contextual encoder cannot quote a
   supporting span, so it reports generic phrasing rather than inventing a quote.
2. **A 10:1 cost ratio is an assumption.** The threshold moves with it; the
   sweep is published so the trade-off is explicit.
3. **The evidence path is the weak link.** Explaining *why* a message is a scam
   is only as good as the tactic labels, and those are template-derived today.
4. **Not adversarially hardened.** Six static families, no adaptive attack.

## Ethical considerations

The model classifies messages that real people receive. A false positive costs a
glance at a real bank notification; a false negative can cost a person their
money or their account. The default deployment is advisory, the threshold is
published, and the raw training data is not redistributed - the sources are
fetched by script with their licences recorded in `docs/data_sources.md`.

## Reproducing

```bash
python -m scripts.download_data
python -m scripts.build_dataset
python -m scripts.build_augmented_dataset --fraction 0.5
python -m src.models.transformer --model xlm-roberta-base --config configs/xlmr.json --train-split train_augmented --output-suffix=-aug
python -m scripts.export_model --model xlm-roberta-base-aug --quantize
python -m scripts.evaluate_with_ci --models xlmr-base-aug
python -m scripts.evaluate_robustness --models xlmr-base-aug
python -m scripts.benchmark_latency --models tfidf-lr --n 100 --device cpu
```
