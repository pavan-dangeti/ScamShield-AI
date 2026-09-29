# Hand-verified test set

**Status: not yet frozen. This file is replaced by `python -m scripts.freeze_test`
once the labelling pass is complete.**

## Why it is not frozen yet

The test set must contain **real** messages, labelled by a human against
[`labeling_guidelines.md`](labeling_guidelines.md). The labelling pass has not
happened yet, so there is nothing to freeze. This file exists so that the README
and the data card can point at the real state of the evaluation data instead of
at a promise.

## What is ready

| Item | Value |
|---|---|
| Candidate pool | `data/test_candidates/candidates.jsonl` |
| Candidates available | 189 real messages (UCI SMS Spam reserve, English) |
| Held out of training | 300 UCI messages (`UCI_RESERVE` in `scripts/build_dataset.py`), labelled or not |
| Labelling tool | `docs/labeling/labeler.html` (offline, keyboard-driven) |
| Export format | JSONL, one row per labelled message |
| Freeze command | `python -m scripts.freeze_test` |

No synthetic message and no message of unstated provenance can enter the pool:
`scripts/freeze_test.py` rejects both, and `tests/test_data_pipeline.py` covers it.

## What is still missing

The eight non-English cells have no public real corpus, so they need messages
supplied by the owner:

| Cell | Scam | Legit | Status |
|---|---|---|---|
| english / latin | candidates ready | candidates ready | awaiting labelling |
| hindi / native | 0 | 0 | needs owner messages |
| hindi / romanized | 0 | 0 | needs owner messages |
| tamil / native | 0 | 0 | needs owner messages |
| tamil / romanized | 0 | 0 | needs owner messages |
| telugu / native | 0 | 0 | needs owner messages |
| telugu / romanized | 0 | 0 | needs owner messages |
| bengali / native | 0 | 0 | needs owner messages (or provenance confirmation for the smishing corpus) |
| bengali / romanized | 0 | 0 | needs owner messages (or provenance confirmation for the smishing corpus) |

To contribute messages: put them in a CSV with `text,language,script` columns,
remove anything identifying, and run

```bash
python -m scripts.import_user_messages --input my_messages.csv
```

The importer scrubs phone numbers, emails, UPI handles, long identifiers and
link query strings, and refuses languages outside the project's five. It cannot
remove personal names, which is why the CSV is the owner's responsibility.

## What freezing will produce

`python -m scripts.freeze_test` writes this file with:

* rows per language, script and label;
* an explicit list of gaps (cells below the target of 25 per class, or missing
  entirely) rather than a silently smaller set;
* a breakdown of message sources.

After freezing, run `python -m scripts.build_dataset` once more: the build
removes every training row that is a near-duplicate of a frozen test message, so
the reported metrics stay honest.
