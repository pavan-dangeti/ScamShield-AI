# Robustness under adversarial perturbation

Split: `val` (1554 rows) · threshold 0.5 · generated 2026-09-29

F1 (delta vs clean) and flip rate = share of messages whose decision changed.

| Model | Clean F1 | char_swap F1 | lookalike F1 | transliteration F1 | spacing F1 | emoji F1 | disguised_link F1 | Mean flip rate |
|---|---|---|---|---|---|---|---|---|
| tfidf-lr | 0.9082 | 0.863 (-0.045) | 0.884 (-0.024) | 0.901 (-0.007) | 0.886 (-0.022) | 0.907 (-0.001) | 0.903 (-0.005) | 0.017 |
| indicbertv2-mlm | 0.9930 | 0.910 (-0.083) | 0.961 (-0.032) | 0.993 (+0.000) | 0.938 (-0.055) | 0.993 (+0.000) | 0.994 (+0.001) | 0.020 |
| xlmr-base | 0.9970 | 0.964 (-0.033) | 0.977 (-0.020) | 0.990 (-0.007) | 0.971 (-0.026) | 0.994 (-0.003) | 0.996 (-0.001) | 0.011 |

| Model | char_swap flip | lookalike flip | transliteration flip | spacing flip | emoji flip | disguised_link flip |
|---|---|---|---|---|---|---|
| tfidf-lr | 0.041 | 0.031 | 0.008 | 0.019 | 0.002 | 0.003 |
| indicbertv2-mlm | 0.061 | 0.024 | 0.000 | 0.036 | 0.000 | 0.002 |
| xlmr-base | 0.024 | 0.017 | 0.004 | 0.019 | 0.003 | 0.001 |
