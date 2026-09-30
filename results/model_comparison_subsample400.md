# Model comparison

Split: `val` (354 rows, seeded stratified subsample of 400) · threshold 0.5 · generated 2026-09-29

| Model | Precision | Recall | F1 | ROC-AUC | Tactic macro-F1 | Tactic rows |
|---|---|---|---|---|---|---|
| llm-0shot | 0.5329 | 0.9889 | 0.6926 | 0.5462 | 0.0919 | 128 |
| llm-3shot | 0.5696 | 1.0000 | 0.7258 | 0.6092 | 0.4030 | 128 |
| lora-SmolLM2-135M-Instruct | 0.9597 | 0.7944 | 0.8693 | 0.8800 | 0.1822 | 128 |
