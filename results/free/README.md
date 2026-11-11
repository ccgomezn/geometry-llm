# Free follow-up analyses (CPU only, saved results)

## 1. Do two independent trainings find the same 2-D code? (gemma3-12b, layer 14)

- Principal angles between the exploratory (seeds 5/6) and confirmation (seeds 105/106) subspaces: **47.0° and 56.1°** (cosines 0.682, 0.558).
- Random 2-D subspaces of the same 3,840-d space: largest cosine 0.056 at the 99th percentile, 0.070 at most (2000 draws).

## gemma3-12b

### 2. Full edit (α = 1): where do the answers go?
Questions whose answer differs between the two stated layouts (model-relative).

| | share |
|---|---|
| follows edit | 0.70 |
| stays at original | 0.18 |
| something else | 0.11 |

By question type:

| qtype | follows edit | something else | stays at original | n |
|---|---|---|---|---|
| closest | 0.28 | 0.00 | 0.72 | 18 |
| direction | 0.58 | 0.21 | 0.20 | 302 |
| farthest | 0.44 | 0.00 | 0.56 | 18 |
| more_east | 0.93 | 0.00 | 0.07 | 90 |
| more_north | 0.92 | 0.00 | 0.08 | 132 |

By whether the model answers the moved scene correctly when it reads it as text:

| model_right_on_moved | follows edit | something else | stays at original | n |
|---|---|---|---|---|
| False | 0.48 | 0.24 | 0.28 | 93 |
| True | 0.75 | 0.09 | 0.16 | 467 |

### 3. Snap direction by α

| alpha | n | keeps_original | takes_moved | other |
|---|---|---|---|---|
| 0.00 | 560 | 0.99 | 0.00 | 0.00 |
| 0.25 | 560 | 0.94 | 0.05 | 0.02 |
| 0.50 | 560 | 0.63 | 0.31 | 0.06 |
| 0.75 | 560 | 0.28 | 0.62 | 0.10 |
| 1.00 | 560 | 0.18 | 0.70 | 0.11 |
| 1.50 | 560 | 0.14 | 0.72 | 0.14 |

## gemma3-27b

### 2. Full edit (α = 1): where do the answers go?
Questions whose answer differs between the two stated layouts (model-relative).

| | share |
|---|---|
| follows edit | 0.82 |
| stays at original | 0.14 |
| something else | 0.04 |

By question type:

| qtype | follows edit | something else | stays at original | n |
|---|---|---|---|---|
| closest | 0.37 | 0.00 | 0.63 | 19 |
| direction | 0.83 | 0.08 | 0.10 | 302 |
| farthest | 0.14 | 0.00 | 0.86 | 14 |
| more_east | 0.86 | 0.00 | 0.14 | 95 |
| more_north | 0.92 | 0.00 | 0.08 | 140 |

By whether the model answers the moved scene correctly when it reads it as text:

| model_right_on_moved | follows edit | something else | stays at original | n |
|---|---|---|---|---|
| False | 0.59 | 0.19 | 0.22 | 64 |
| True | 0.85 | 0.02 | 0.12 | 506 |

### 3. Snap direction by α

| alpha | n | keeps_original | takes_moved | other |
|---|---|---|---|---|
| 0.00 | 570 | 0.99 | 0.01 | 0.00 |
| 0.25 | 570 | 0.96 | 0.04 | 0.01 |
| 0.50 | 570 | 0.67 | 0.29 | 0.04 |
| 0.75 | 570 | 0.24 | 0.73 | 0.03 |
| 1.00 | 570 | 0.14 | 0.82 | 0.04 |
| 1.50 | 570 | 0.09 | 0.86 | 0.04 |
