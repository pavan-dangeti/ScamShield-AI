---
license: cc-by-nc-4.0
language:
  - en
  - hi
  - ta
  - te
  - bn
library_name: onnx
pipeline_tag: text-classification
base_model: FacebookAI/xlm-roberta-base
datasets:
  - shariul-islam/bengali-sms-smishing-dataset
  - Ridham115/indian-scam-sms-synthetic-audited
tags:
  - scam-detection
  - fraud-detection
  - code-mixed
  - indic
  - xlm-roberta
  - onnx
  - int8
---

# ScamShield XLM-R (int8 ONNX)

Classifies SMS and WhatsApp messages as payment scams or legitimate and scores six
manipulation tactics. Built for Indian traffic, including code-mixed and romanised
text such as Hinglish and Tanglish. This is the model served by
[ScamShield](https://github.com/pavan-dangeti/ScamShield-AI).

**Built with Qwen.** Part of the training data (85 of 13,923 training rows) was
generated with Qwen2.5-3B-Instruct, which is licensed under the Qwen Research
License Agreement.

> **Read the evaluation as validation-only.** All figures below come from a held-out
> split of a corpus that is mostly synthetic or of unverified provenance. No
> real-traffic test set has been frozen yet. Do not use these numbers to estimate
> accuracy on real messages.

## Files

FILES_TABLE

The same sums are in `SHA256SUMS` (`shasum -a 256 -c SHA256SUMS`).

## Usage

**With ScamShield** (downloads this repository at a pinned revision and verifies every file):

```bash
git clone https://github.com/pavan-dangeti/ScamShield-AI.git && cd ScamShield-AI
pip install -r requirements-onnx.txt
python -m scripts.download_model
SCAMSHIELD_MODEL=xlmr-aug-int8 python -m src.main
```

**Standalone**, with `onnxruntime`, `transformers` and `huggingface_hub`:

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

path = snapshot_download("REPO_ID")
tokenizer = AutoTokenizer.from_pretrained(path)
session = ort.InferenceSession(f"{path}/model.int8.onnx", providers=["CPUExecutionProvider"])

TACTICS = ["urgency", "authority_impersonation", "false_reward",
           "loss_aversion", "credential_phishing", "suspicious_link"]
messages = ["Congratulations! You won Rs 15000 cashback. Claim now at bit.ly/payupi"]
encoded = tokenizer(messages, truncation=True, max_length=160, padding=True, return_tensors="np")
binary_logits, tactic_logits = session.run(
    None, {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
)
scam_probability = 1 / (1 + np.exp(-binary_logits))   # shape [batch]
tactic_probability = 1 / (1 + np.exp(-tactic_logits))  # shape [batch, 6], order of TACTICS
# scam_probability ≈ 1.0; false_reward ≈ 0.70 is the only tactic at or above 0.5
```

Inputs: `input_ids`, `attention_mask` (int64, `[batch, sequence]`, at most 160 tokens).
Outputs: `binary_logits` `[batch]` and `tactic_logits` `[batch, 6]`, both logits.
A message is a scam at probability ≥ 0.5; tactics are reported for scams at ≥ 0.5.

## Model

XLM-R base (`FacebookAI/xlm-roberta-base`, MIT) with two heads on the `[CLS]`
representation: a binary scam head and a six-way multi-label tactic head. Fine-tuned
for 3 epochs (learning rate 2e-5, batch 16, weight decay 0.01, dropout 0.1, maximum
160 tokens, seed 42) on the training split plus perturbed copies of half its rows
(character swaps, look-alike characters, spacing tricks, disguised links). Exported
to ONNX and dynamically quantised to int8; validation F1 is the same before and after
quantisation (0.9980).

## Training data

| Source | Rows used | Licence | Provenance |
|---|---|---|---|
| [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | 4,904 | CC BY 4.0 | Real English SMS, UK, 2004–2011 |
| [Bengali SMS Smishing Dataset](https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset) | 6,855 | MIT | Collection method not stated |
| [Indian Scam SMS (synthetic, audited)](https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited) | 1,453 | CC BY 4.0 | LLM-generated |
| ScamShield language-pack templates | 2,180 | MIT | Hand-written templates, expanded |
| Generated with Qwen2.5-3B-Instruct | 85 | Qwen Research License | LLM-generated, quality-gated |

15,477 rows after PII scrubbing and de-duplication, split 13,923 train / 1,554
validation by near-duplicate group, so no near-duplicate crosses the split. Spam is
mapped to scam; the Bengali corpus's `promo` class is mapped to legitimate.

Attribution: Almeida, Gómez Hidalgo and Yamakami (2011), *Contributions to the Study of
SMS Spam Filtering*, ACM DocEng; Islam, S. (2025), *SmishDetect-LLM*, Murdoch University;
Ridham115, *indian-scam-sms-synthetic-audited*.

## Evaluation (validation split, 1,554 messages, 504 scams)

| Metric | Value |
|---|---|
| Precision / recall / F1 | 0.9960 / 1.0000 / 0.9980 |
| F1, 95% bootstrap CI | 0.995–1.000 |
| Tactic macro-F1 | 0.987 (230 messages with tactic labels) |
| Brier score / expected calibration error | 0.0010 / 0.0016 |
| F1 under character swaps / look-alikes / spacing | 0.993 / 0.986 / 0.985 |
| Cost-optimal threshold (missed scam = 10× false alarm) | 0.75 |
| CPU latency, one message | 4.44 ms p50, 9.72 ms p95 |
| Memory of a server holding the model | about 1.1 GB |

Per language and script, F1 is 1.000 on every Indic cell and 0.993 on English, but
the Indic cells are small (24 to 58 messages per cell for Tamil, Telugu and native
Hindi) and mostly template or synthetic text. The comparison baseline, TF-IDF with
logistic regression, scores F1 0.9082 on the same split.

## Limitations

1. **No real-traffic measurement.** The validation set is mostly synthetic, template
   or of unverified origin; near-perfect scores mean the corpus was learned, not that
   the task is solved.
2. **The only real data with a known origin is English SMS from the UK, 2004–2011.**
   Indian payment scams are represented mainly by synthetic and template text.
   Expect over-flagging of legitimate Indian marketing.
3. **Bengali is 37% of the corpus**, from a source that does not state its collection
   method.
4. **Tactic labels are template-derived**, so tactic scores reflect template families and
   can miss obvious tactics in messages unlike the templates: "your SBI account will be
   blocked in 30 minutes" is scored as a scam with probability 1.0, but urgency only 0.02.
5. **Robustness was tested against six static attack families only**; no adaptive attack.
6. **Five languages only.** Behaviour on other languages is unknown.

## Intended use

Advisory flagging of suspicious messages for a person to review, and research on
code-mixed scam detection. Not for automatic blocking, and not as the only control
in a financial workflow.

## Licence

CC BY-NC 4.0: free for non-commercial use with attribution. The licence is
non-commercial because part of the training data was generated with a model whose
licence permits non-commercial use. The base model is MIT and the external datasets
are CC BY 4.0 or MIT; their attributions are listed above.
