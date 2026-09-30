# Phases 2 and 3 - model comparison and evaluation

Everything below is produced by scripts in this repository. Each table states
the command that regenerates it.

## 1. What was compared

| Arm | Model | How it was run | Command |
|---|---|---|---|
| Baseline | TF-IDF (word 1-2 + char 3-5) + logistic regression, 7 heads | retrained on the new corpus, `C` chosen per head by 3-fold CV | `python -m scripts.train_tfidf` |
| Encoder | `ai4bharat/IndicBERTv2-MLM-only` | fine-tuned, binary head + 6 tactic heads | `python -m src.models.transformer --model ai4bharat/IndicBERTv2-MLM-only --config configs/indicbertv2.json` |
| Encoder | `xlm-roberta-base` | fine-tuned, same heads | `python -m src.models.transformer --model xlm-roberta-base --config configs/xlmr.json` |
| Prompted LLM | Qwen2.5-7B-Instruct-4bit, 0-shot | rules + JSON output, greedy | `python -m src.models.llm_prompted --shots 0 --limit 400` |
| Prompted LLM | Qwen2.5-7B-Instruct-4bit, 3-shot | 3 class-balanced examples from train | `python -m src.models.llm_prompted --shots 3 --limit 400` |
| LoRA | SmolLM2-135M-Instruct, LoRA r=16 | 1 epoch, answer-only loss | `python -m src.models.lora --model HuggingFaceTB/SmolLM2-135M-Instruct --config configs/lora_smollm135m.json --device cpu` |

MuRIL is the encoder named in the project brief, but every MuRIL repository on
Hugging Face is gated and returns 401 without an access token, so the Indic arm
uses AI4Bharat's IndicBERTv2 instead. The code path is identical; point `--model`
at `google/muril-base` with your token and it will run.

## 2. Headline result

`python -m scripts.evaluate_models --models tfidf-lr indicbertv2-mlm xlmr-base`

Validation split, 1,554 rows, threshold 0.5:

| Model | Precision | Recall | F1 | ROC-AUC | Tactic macro-F1 |
|---|---|---|---|---|---|
| tfidf-lr | 0.8522 | 0.9722 | 0.9082 | 0.9892 | 0.8254 |
| indicbertv2-mlm | 0.9980 | 0.9881 | 0.9930 | 1.0000 | 0.9621 |
| xlmr-base | 0.9941 | 1.0000 | 0.9970 | 0.9999 | 0.9267 |

**Read these numbers carefully.** The corpus is still mostly synthetic, and the
train/val split is a near-duplicate-aware split of it, not the hand-verified test
set. The encoders' near-perfect scores say the data is easy for a contextual
encoder, not that the system works on real messages. The gap between 0.908 and
0.997 is a gap on this corpus, and Phase 1 already established that this corpus
overstates real-world performance. The honest conclusion is: *the encoders learn
this corpus better than the baseline*, and *the corpus must be replaced by the
frozen test set before any of these numbers describe the product.*

## 3. Prompted vs trained LLM: prompting over-flags, fine-tuning does not

`python -m scripts.evaluate_models --models llm-0shot llm-3shot lora-SmolLM2-135M-Instruct --subsample 400`

Seeded stratified subsample, 354 rows, so the generative arms are compared on
exactly the same messages as each other:

| Model | Precision | Recall | F1 | Tactic macro-F1 | Scam calls |
|---|---|---|---|---|---|
| llm-0shot | 0.5329 | 0.9889 | 0.6926 | 0.0919 | 334 / 354 |
| llm-3shot | 0.5696 | 1.0000 | 0.7258 | 0.4030 | 316 / 354 |
| lora-SmolLM2-135M | 0.9597 | 0.7944 | 0.8693 | 0.1822 | - |

The prompted 7B model labels almost every message a scam - zero-shot 334 of 354,
and still 316 of 354 after three examples. It has learned the *topic* of fraud
detection rather than the *boundary*, and it stamps the training templates' tactics
onto nearly everything (tactic macro-F1 0.09 zero-shot). Few-shot examples improve
the tactic list and barely move the decision bias.

**Fine-tuning fixes exactly that bias, for free.** A 135M model - 50x smaller than
the 7B it is compared against - reaches F1 0.869 with precision 0.96, because the
answer token it is trained to emit carries the prior from the corpus rather than the
prior from the prompt. The residual weakness moved rather than disappeared: recall
drops to 0.79 (it misses scams) while its tactic labels stay weak (0.18), so the
decision is learned but the explanation is not.

This is the clearest result in the project: for a detection task on an imbalanced
corpus, **fine-tuning a small model beats prompting a large one**, and the failure
mode of the prompted model is exactly the kind a careless evaluation hides behind
an F1.

## 4. Per-language and per-script results with confidence intervals

`python -m scripts.evaluate_with_ci --models tfidf-lr indicbertv2-mlm xlmr-base`

Percentile bootstrap, 2,000 resamples, 95% CI, macro average over cells that
contain both classes:

| Model | F1 | 95% CI | Macro F1 over cells | Brier | ECE |
|---|---|---|---|---|---|
| tfidf-lr | 0.9082 | 0.889 - 0.926 | 0.9277 | 0.0495 | 0.0631 |
| indicbertv2-mlm | 0.9930 | 0.987 - 0.998 | 0.9972 | 0.0040 | 0.0059 |
| xlmr-base | 0.9970 | 0.993 - 1.000 | 0.9965 | 0.0023 | 0.0037 |

Weakest cells, where the sample is small enough that the CI is wide:

| Model | Cell | Rows | F1 | 95% CI |
|---|---|---|---|---|
| tfidf-lr | bengali \| native | 192 | 0.652 | 0.576 - 0.721 |
| tfidf-lr | telugu \| native | 24 | 0.870 | 0.667 - 1.000 |
| indicbertv2-mlm | english \| latin | 699 | 0.975 | 0.954 - 0.992 |
| xlmr-base | tamil \| romanized | 38 | 0.976 | 0.914 - 1.000 |

The baseline's Bengali-native failure is its worst result by a wide margin:
precision 0.48, recall 1.00 - it flags most legitimate Bengali native messages as
scams. That is the kind of error a per-language breakdown exists to catch, and it
would be invisible in an aggregate number.

## 5. Choosing the decision threshold

A missed scam costs the reader money or credentials. A false alarm costs a glance
at a real bank notification. **Cost ratio: a false negative is treated as 10x a
false positive.** That is a product judgement, not a measurement, and the sweep
shows how much the choice matters:

| Model | Best-F1 threshold | Cost-optimal threshold (10:1) | P / R there | Expected cost |
|---|---|---|---|---|
| tfidf-lr | 0.45 | 0.30 | 0.827 / 0.998 | 0.074 |
| indicbertv2-mlm | 0.05 | 0.05 | 0.998 / 0.992 | 0.026 |
| xlmr-base | 0.45 | 0.45 | 0.994 / 1.000 | 0.0019 |

For the baseline, moving the threshold from 0.45 to 0.30 trades precision for
recall and lowers expected cost by a third, because the baseline's errors are
mostly false negatives. For XLM-R the cost-optimal and F1-optimal thresholds are
the same, which means the choice does not matter once the model is good. Full
sweep for every threshold from 0.05 to 0.95 is in `results/evaluation_with_ci.json`.

A production system should expose the threshold as a setting and pick it against
measured cost, not adopt whatever the validation F1 happens to like.

## 6. Latency, size and cost

`python -m scripts.benchmark_latency --models tfidf-lr indicbertv2-mlm xlmr-base --n 100 --device cpu`

CPU, one message at a time, 100 messages:

| Model | p50 | p95 | Throughput | Artefact |
|---|---|---|---|---|
| tfidf-lr | 2.31 ms | 4.00 ms | 389 msg/s | 10.2 MB |
| indicbertv2-mlm | 38.67 ms | 134.74 ms | 20.6 msg/s | 1120 MB |
| xlmr-base | 31.89 ms | 82.70 ms | 26.3 msg/s | 1129 MB |

The baseline is 13-16x faster and 110x smaller. The encoders are still fast enough
for a server or an on-device check on a modern phone, but they are 1.1 GB fp32
downloads unless quantised, which is Phase 5 work.

Prompted LLM cost, `python -m scripts.llm_cost --shots 3`, from measured token
counts (440 input + 13 output tokens per message):

| | Per 1,000 messages |
|---|---|
| Local (own hardware, 25 W draw, $0.15/kWh) | $0.0038 and about 60 minutes of machine time |
| Hosted API at a blended $0.30 / 1M tokens | $0.14 |

The wall time is the real cost of the prompted LLM: 60 minutes per 1,000
messages on one laptop, against 2.6 seconds for XLM-R on the same CPU.

## 7. Explanation quality

| Model | Tactic macro-F1 | Tactic micro-F1 | Evidence quoted verbatim |
|---|---|---|---|
| tfidf-lr | 0.8254 | 0.8326 | 374 / 374 (100%) |
| indicbertv2-mlm | 0.9621 | 0.9764 | not extracted |
| xlmr-base | 0.9267 | 0.9400 | not extracted |
| llm-0shot | 0.0919 | 0.2199 | n/a |
| llm-3shot | 0.4030 | 0.5016 | n/a |
| lora-SmolLM2-135M | 0.1822 | - | n/a |

Tactic scores are measured on the 230 validation rows that carry gold tactic
labels. Those labels come from the template packs, so they measure "which
template family is this", not human-judged tactics. The encoder arms score well
because they can memorise template families; the hand-labelled test set is where
this has to be re-measured.

The baseline quotes an evidence span for 100% of its tactic predictions, and
every span it quotes occurs verbatim in the message. Getting to 100% required a
fix: word n-grams are rebuilt from tokens, so a bigram like "http electricity"
can be produced from "http://electricity-bill" without ever appearing in the
message. The evidence code now quotes a span only if it is a literal substring,
and reports "suspicious phrasing" otherwise. A model that explains itself must
not invent quotes.

## 8. What this phase does not settle

1. **No real-message evaluation yet.** These are validation numbers on a corpus
   that is mostly synthetic. The hand-verified test set from Phase 1 is not
   frozen, so nothing here measures the product.
2. **Bengali dominates the corpus** (37%) from a source of unstated provenance, so
   the per-language table over-represents Bengali relative to real traffic.
3. **Tactic labels are template-derived** for every training row, so tactic
   metrics are optimistic by construction.
4. **MuRIL was not run** (gated repository), the prompted LLM was run with one
   model and one prompt, and the LoRA arm is a 135M model trained for one epoch
   because Hugging Face throttled the larger checkpoint in this environment.
5. **No model was quantised or exported**, so the encoders carry 1.1 GB fp32
   weights; that is Phase 5 work and it will change the latency table.
5. **Cost per 1,000 messages** is computed for the local case; the API figure is
   arithmetic on measured token counts at a stated price, not a billed amount.
