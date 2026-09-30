# Robustness under adversarial perturbation

Split: `val` (1554 rows) · threshold 0.5 · generated 2026-09-30

F1 (delta vs clean) and flip rate = share of messages whose decision changed.

| Model | Clean F1 | char_swap F1 | lookalike F1 | transliteration F1 | spacing F1 | emoji F1 | disguised_link F1 | Mean flip rate |
|---|---|---|---|---|---|---|---|---|
| tfidf-lr | 0.9082 | 0.863 (-0.045) | 0.884 (-0.024) | 0.901 (-0.007) | 0.886 (-0.022) | 0.907 (-0.001) | 0.903 (-0.005) | 0.017 |
| tfidf-lr-aug | 0.9198 | 0.902 (-0.018) | 0.909 (-0.011) | 0.911 (-0.009) | 0.911 (-0.009) | 0.919 (-0.001) | 0.917 (-0.003) | 0.008 |
| xlmr-base | 0.9970 | 0.964 (-0.033) | 0.977 (-0.020) | 0.990 (-0.007) | 0.971 (-0.026) | 0.994 (-0.003) | 0.996 (-0.001) | 0.011 |
| xlmr-base-aug | 0.9980 | 0.993 (-0.005) | 0.986 (-0.012) | 0.998 (+0.000) | 0.985 (-0.013) | 0.998 (+0.000) | 0.998 (+0.000) | 0.003 |

| Model | char_swap flip | lookalike flip | transliteration flip | spacing flip | emoji flip | disguised_link flip |
|---|---|---|---|---|---|---|
| tfidf-lr | 0.041 | 0.031 | 0.008 | 0.019 | 0.002 | 0.003 |
| tfidf-lr-aug | 0.015 | 0.015 | 0.006 | 0.011 | 0.001 | 0.002 |
| xlmr-base | 0.024 | 0.017 | 0.004 | 0.019 | 0.003 | 0.001 |
| xlmr-base-aug | 0.003 | 0.009 | 0.000 | 0.008 | 0.000 | 0.000 |
