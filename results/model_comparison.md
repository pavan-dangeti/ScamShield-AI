# Model comparison

Split: `val` (1554 rows) · threshold 0.5 · generated 2026-09-29

| Model | Precision | Recall | F1 | ROC-AUC | Tactic macro-F1 | Tactic rows |
|---|---|---|---|---|---|---|
| tfidf-lr | 0.8522 | 0.9722 | 0.9082 | 0.9892 | 0.8254 | 230 |
| indicbertv2-mlm | 0.9980 | 0.9881 | 0.9930 | 1.0000 | 0.9621 | 230 |
| xlmr-base | 0.9941 | 1.0000 | 0.9970 | 0.9999 | 0.9267 | 230 |
