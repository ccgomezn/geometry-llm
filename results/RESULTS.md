# E1 results log

Dated entries are appended after each phase. Every number carries a 95% bootstrap CI. Thresholds are preregistered in `docs/preregistration.md`.

## 2026-11-09 16:06 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### qwen3-1.7b
- 4-object accuracy: 0.40 [0.38, 0.43]; accuracy above chance on 2+ hop questions: 0.03 [0.01, 0.04] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.49 [0.44, 0.53] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.54 [0.48, 0.61] | 0.37 [0.31, 0.43] | 0.30 [0.25, 0.35] |
| direction | 0.12 | 0.24 [0.19, 0.28] | 0.15 [0.11, 0.19] | 0.18 [0.14, 0.23] |
| farthest | 0.36 | 0.50 [0.44, 0.57] | 0.33 [0.27, 0.40] | 0.23 [0.17, 0.28] |
| more_east | 0.50 | 0.54 [0.47, 0.61] | 0.53 [0.47, 0.59] | 0.54 [0.48, 0.60] |
| more_north | 0.50 | 0.57 [0.51, 0.63] | 0.52 [0.46, 0.57] | 0.52 [0.47, 0.57] |

- By deepest object involved: hop 1: 0.42 [0.40, 0.44]; hop 2: 0.43 [0.41, 0.44]; hop 3: 0.41 [0.38, 0.44]; hop 4: 0.43 [0.37, 0.50]
- Error signature (standardised logistic coefficients): hop -0.04 [-0.11, 0.02], log margin 0.07 [0.02, 0.11]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.32 [0.29, 0.35]
- Object-list line: with 0.38, without 0.38, difference -0.01 [-0.03, 0.02]
- Chain of thought on: 0.52 [0.40, 0.63] vs off 0.42 [0.30, 0.53] (same 60 questions; unparsed 38%, median 1024 tokens)

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 16:42 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### llama3.2-3b
- 4-object accuracy: 0.55 [0.52, 0.58]; accuracy above chance on 2+ hop questions: 0.13 [0.11, 0.15] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.51 [0.46, 0.55] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.64 [0.58, 0.70] | 0.54 [0.47, 0.59] | 0.48 [0.42, 0.54] |
| direction | 0.12 | 0.53 [0.47, 0.58] | 0.47 [0.41, 0.52] | 0.39 [0.34, 0.45] |
| farthest | 0.36 | 0.66 [0.60, 0.73] | 0.54 [0.48, 0.60] | 0.40 [0.33, 0.47] |
| more_east | 0.50 | 0.56 [0.48, 0.62] | 0.63 [0.57, 0.68] | 0.61 [0.56, 0.67] |
| more_north | 0.50 | 0.66 [0.59, 0.72] | 0.62 [0.56, 0.68] | 0.64 [0.58, 0.70] |

- By deepest object involved: hop 1: 0.61 [0.58, 0.63]; hop 2: 0.54 [0.52, 0.56]; hop 3: 0.50 [0.46, 0.53]; hop 4: 0.48 [0.41, 0.56]
- Error signature (standardised logistic coefficients): hop -0.16 [-0.22, -0.10], log margin 0.23 [0.17, 0.29]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.36 [0.32, 0.39]
- Object-list line: with 0.53, without 0.51, difference 0.02 [-0.01, 0.04]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 17:34 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### qwen3-4b
- 4-object accuracy: 0.44 [0.42, 0.46]; accuracy above chance on 2+ hop questions: 0.03 [0.01, 0.05] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.53 [0.47, 0.58] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.53 [0.47, 0.60] | 0.41 [0.34, 0.47] | 0.27 [0.22, 0.32] |
| direction | 0.12 | 0.44 [0.38, 0.50] | 0.26 [0.21, 0.30] | 0.22 [0.18, 0.27] |
| farthest | 0.36 | 0.53 [0.46, 0.59] | 0.31 [0.25, 0.38] | 0.19 [0.14, 0.25] |
| more_east | 0.50 | 0.71 [0.64, 0.77] | 0.54 [0.48, 0.60] | 0.48 [0.41, 0.54] |
| more_north | 0.50 | 0.72 [0.65, 0.78] | 0.59 [0.53, 0.65] | 0.53 [0.48, 0.59] |

- By deepest object involved: hop 1: 0.53 [0.50, 0.56]; hop 2: 0.45 [0.43, 0.47]; hop 3: 0.37 [0.34, 0.40]; hop 4: 0.39 [0.33, 0.45]
- Error signature (standardised logistic coefficients): hop -0.27 [-0.33, -0.21], log margin 0.10 [0.04, 0.15]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.31 [0.28, 0.35]
- Object-list line: with 0.42, without 0.44, difference -0.02 [-0.05, 0.01]
- Chain of thought on: 0.43 [0.32, 0.57] vs off 0.45 [0.32, 0.58] (same 60 questions; unparsed 43%, median 1024 tokens)

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 18:05 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### gemma3-4b
- 4-object accuracy: 0.53 [0.51, 0.55]; accuracy above chance on 2+ hop questions: 0.13 [0.11, 0.14] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.49 [0.44, 0.53] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.61 [0.54, 0.66] | 0.43 [0.37, 0.48] | 0.37 [0.32, 0.42] |
| direction | 0.12 | 0.48 [0.43, 0.54] | 0.48 [0.42, 0.53] | 0.38 [0.33, 0.44] |
| farthest | 0.36 | 0.62 [0.56, 0.68] | 0.45 [0.38, 0.52] | 0.44 [0.37, 0.50] |
| more_east | 0.50 | 0.71 [0.65, 0.77] | 0.62 [0.56, 0.68] | 0.60 [0.54, 0.66] |
| more_north | 0.50 | 0.64 [0.58, 0.70] | 0.66 [0.60, 0.72] | 0.67 [0.61, 0.73] |

- By deepest object involved: hop 1: 0.57 [0.55, 0.60]; hop 2: 0.52 [0.50, 0.54]; hop 3: 0.50 [0.47, 0.54]; hop 4: 0.55 [0.49, 0.62]
- Error signature (standardised logistic coefficients): hop -0.08 [-0.14, -0.03], log margin 0.21 [0.16, 0.26]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.36 [0.32, 0.39]
- Object-list line: with 0.53, without 0.48, difference 0.05 [0.01, 0.08]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 19:27 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### qwen3-8b
- 4-object accuracy: 0.42 [0.41, 0.44]; accuracy above chance on 2+ hop questions: 0.01 [0.00, 0.02] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.49 [0.46, 0.51] | 0.49 [0.46, 0.51] | 0.48 [0.45, 0.51] |
| closest | 0.36 | 0.59 [0.55, 0.62] | 0.34 [0.31, 0.38] | 0.26 [0.24, 0.29] |
| direction | 0.12 | 0.36 [0.33, 0.39] | 0.18 [0.15, 0.20] | 0.13 [0.11, 0.15] |
| farthest | 0.36 | 0.44 [0.40, 0.47] | 0.32 [0.29, 0.35] | 0.22 [0.19, 0.26] |
| more_east | 0.50 | 0.69 [0.66, 0.73] | 0.57 [0.54, 0.60] | 0.50 [0.47, 0.53] |
| more_north | 0.50 | 0.73 [0.70, 0.76] | 0.63 [0.60, 0.66] | 0.55 [0.52, 0.58] |

- By deepest object involved: hop 1: 0.52 [0.50, 0.53]; hop 2: 0.42 [0.41, 0.43]; hop 3: 0.36 [0.34, 0.38]; hop 4: 0.35 [0.31, 0.38]
- Error signature (standardised logistic coefficients): hop -0.29 [-0.32, -0.26], log margin 0.08 [0.05, 0.11]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.36 [0.34, 0.39]
- Object-list line: with 0.43, without 0.42, difference 0.00 [-0.01, 0.02]
- Chain of thought on: 0.87 [0.77, 0.95] vs off 0.27 [0.17, 0.38] (same 60 questions; unparsed 10%, median 1625 tokens)

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 20:44 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### qwen3-14b
- 4-object accuracy: 0.58 [0.57, 0.60]; accuracy above chance on 2+ hop questions: 0.17 [0.16, 0.18] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.57 [0.54, 0.60] | 0.59 [0.56, 0.62] | 0.60 [0.57, 0.64] |
| closest | 0.36 | 0.63 [0.59, 0.66] | 0.39 [0.35, 0.42] | 0.28 [0.25, 0.30] |
| direction | 0.12 | 0.61 [0.57, 0.64] | 0.45 [0.41, 0.48] | 0.33 [0.30, 0.36] |
| farthest | 0.36 | 0.73 [0.70, 0.76] | 0.53 [0.49, 0.56] | 0.29 [0.26, 0.33] |
| more_east | 0.50 | 0.81 [0.79, 0.84] | 0.74 [0.71, 0.77] | 0.71 [0.68, 0.74] |
| more_north | 0.50 | 0.87 [0.85, 0.90] | 0.80 [0.77, 0.82] | 0.77 [0.75, 0.80] |

- By deepest object involved: hop 1: 0.66 [0.65, 0.67]; hop 2: 0.58 [0.57, 0.60]; hop 3: 0.52 [0.50, 0.54]; hop 4: 0.48 [0.44, 0.52]
- Error signature (standardised logistic coefficients): hop -0.27 [-0.30, -0.24], log margin 0.33 [0.29, 0.37]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.40 [0.37, 0.43]
- Object-list line: with 0.57, without 0.55, difference 0.02 [-0.00, 0.04]
- Chain of thought on: 0.98 [0.95, 1.00] vs off 0.57 [0.43, 0.70] (same 60 questions; unparsed 2%, median 844 tokens)

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 21:08 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### gemma3-12b
- 4-object accuracy: 0.66 [0.65, 0.68]; accuracy above chance on 2+ hop questions: 0.22 [0.21, 0.23] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.50 [0.47, 0.53] | 0.49 [0.46, 0.52] | 0.48 [0.45, 0.51] |
| closest | 0.36 | 0.79 [0.76, 0.82] | 0.65 [0.62, 0.68] | 0.55 [0.51, 0.58] |
| direction | 0.12 | 0.72 [0.69, 0.75] | 0.56 [0.53, 0.60] | 0.48 [0.45, 0.51] |
| farthest | 0.36 | 0.75 [0.72, 0.78] | 0.60 [0.56, 0.64] | 0.45 [0.41, 0.49] |
| more_east | 0.50 | 0.86 [0.83, 0.88] | 0.85 [0.83, 0.87] | 0.78 [0.76, 0.81] |
| more_north | 0.50 | 0.87 [0.85, 0.89] | 0.84 [0.81, 0.86] | 0.81 [0.78, 0.83] |

- By deepest object involved: hop 1: 0.77 [0.75, 0.78]; hop 2: 0.63 [0.62, 0.64]; hop 3: 0.59 [0.57, 0.61]; hop 4: 0.56 [0.53, 0.60]
- Error signature (standardised logistic coefficients): hop -0.29 [-0.33, -0.25], log margin 0.35 [0.31, 0.39]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.44 [0.41, 0.46]
- Object-list line: with 0.65, without 0.65, difference 0.01 [-0.01, 0.02]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 21:56 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### gemma3-27b
- 4-object accuracy: 0.68 [0.67, 0.69]; accuracy above chance on 2+ hop questions: 0.27 [0.26, 0.27] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.53 [0.50, 0.56] | 0.53 [0.50, 0.56] | 0.54 [0.51, 0.57] |
| closest | 0.36 | 0.76 [0.73, 0.79] | 0.62 [0.59, 0.66] | 0.55 [0.52, 0.58] |
| direction | 0.12 | 0.77 [0.75, 0.80] | 0.60 [0.57, 0.64] | 0.53 [0.50, 0.56] |
| farthest | 0.36 | 0.77 [0.74, 0.80] | 0.68 [0.64, 0.71] | 0.59 [0.55, 0.62] |
| more_east | 0.50 | 0.87 [0.84, 0.89] | 0.84 [0.81, 0.86] | 0.79 [0.76, 0.81] |
| more_north | 0.50 | 0.86 [0.83, 0.89] | 0.82 [0.80, 0.85] | 0.79 [0.76, 0.82] |

- By deepest object involved: hop 1: 0.75 [0.74, 0.76]; hop 2: 0.67 [0.66, 0.68]; hop 3: 0.63 [0.61, 0.65]; hop 4: 0.62 [0.58, 0.66]
- Error signature (standardised logistic coefficients): hop -0.18 [-0.21, -0.13], log margin 0.33 [0.30, 0.36]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.48 [0.45, 0.51]
- Object-list line: with 0.67, without 0.64, difference 0.03 [0.02, 0.05]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 21:57 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### qwen3-1.7b
- 4-object accuracy: 0.40 [0.38, 0.43]; accuracy above chance on 2+ hop questions: 0.03 [0.01, 0.04] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.49 [0.44, 0.53] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.54 [0.48, 0.61] | 0.37 [0.31, 0.43] | 0.30 [0.25, 0.35] |
| direction | 0.12 | 0.24 [0.19, 0.28] | 0.15 [0.11, 0.19] | 0.18 [0.14, 0.23] |
| farthest | 0.36 | 0.50 [0.44, 0.57] | 0.33 [0.27, 0.40] | 0.23 [0.17, 0.28] |
| more_east | 0.50 | 0.54 [0.47, 0.61] | 0.53 [0.47, 0.59] | 0.54 [0.48, 0.60] |
| more_north | 0.50 | 0.57 [0.51, 0.63] | 0.52 [0.46, 0.57] | 0.52 [0.47, 0.57] |

- By deepest object involved: hop 1: 0.42 [0.40, 0.44]; hop 2: 0.43 [0.41, 0.44]; hop 3: 0.41 [0.38, 0.44]; hop 4: 0.43 [0.37, 0.50]
- Error signature (standardised logistic coefficients): hop -0.04 [-0.11, 0.02], log margin 0.07 [0.02, 0.11]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.32 [0.29, 0.35]
- Object-list line: with 0.38, without 0.38, difference -0.01 [-0.03, 0.02]
- Chain of thought on: 0.52 [0.40, 0.63] vs off 0.42 [0.30, 0.53] (same 60 questions; unparsed 38%, median 1024 tokens)

### llama3.2-3b
- 4-object accuracy: 0.55 [0.52, 0.58]; accuracy above chance on 2+ hop questions: 0.13 [0.11, 0.15] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.51 [0.46, 0.55] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.64 [0.58, 0.70] | 0.54 [0.47, 0.59] | 0.48 [0.42, 0.54] |
| direction | 0.12 | 0.53 [0.47, 0.58] | 0.47 [0.41, 0.52] | 0.39 [0.34, 0.45] |
| farthest | 0.36 | 0.66 [0.60, 0.73] | 0.54 [0.48, 0.60] | 0.40 [0.33, 0.47] |
| more_east | 0.50 | 0.56 [0.48, 0.62] | 0.63 [0.57, 0.68] | 0.61 [0.56, 0.67] |
| more_north | 0.50 | 0.66 [0.59, 0.72] | 0.62 [0.56, 0.68] | 0.64 [0.58, 0.70] |

- By deepest object involved: hop 1: 0.61 [0.58, 0.63]; hop 2: 0.54 [0.52, 0.56]; hop 3: 0.50 [0.46, 0.53]; hop 4: 0.48 [0.41, 0.56]
- Error signature (standardised logistic coefficients): hop -0.16 [-0.22, -0.10], log margin 0.23 [0.17, 0.29]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.36 [0.32, 0.39]
- Object-list line: with 0.53, without 0.51, difference 0.02 [-0.01, 0.04]

### qwen3-4b
- 4-object accuracy: 0.44 [0.42, 0.46]; accuracy above chance on 2+ hop questions: 0.03 [0.01, 0.05] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.53 [0.47, 0.58] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.53 [0.47, 0.60] | 0.41 [0.34, 0.47] | 0.27 [0.22, 0.32] |
| direction | 0.12 | 0.44 [0.38, 0.50] | 0.26 [0.21, 0.30] | 0.22 [0.18, 0.27] |
| farthest | 0.36 | 0.53 [0.46, 0.59] | 0.31 [0.25, 0.38] | 0.19 [0.14, 0.25] |
| more_east | 0.50 | 0.71 [0.64, 0.77] | 0.54 [0.48, 0.60] | 0.48 [0.41, 0.54] |
| more_north | 0.50 | 0.72 [0.65, 0.78] | 0.59 [0.53, 0.65] | 0.53 [0.48, 0.59] |

- By deepest object involved: hop 1: 0.53 [0.50, 0.56]; hop 2: 0.45 [0.43, 0.47]; hop 3: 0.37 [0.34, 0.40]; hop 4: 0.39 [0.33, 0.45]
- Error signature (standardised logistic coefficients): hop -0.27 [-0.33, -0.21], log margin 0.10 [0.04, 0.15]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.31 [0.28, 0.35]
- Object-list line: with 0.42, without 0.44, difference -0.02 [-0.05, 0.01]
- Chain of thought on: 0.43 [0.32, 0.57] vs off 0.45 [0.32, 0.58] (same 60 questions; unparsed 43%, median 1024 tokens)

### gemma3-4b
- 4-object accuracy: 0.53 [0.51, 0.55]; accuracy above chance on 2+ hop questions: 0.13 [0.11, 0.14] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.49 [0.44, 0.53] | 0.52 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.61 [0.54, 0.66] | 0.43 [0.37, 0.48] | 0.37 [0.32, 0.42] |
| direction | 0.12 | 0.48 [0.43, 0.54] | 0.48 [0.42, 0.53] | 0.38 [0.33, 0.44] |
| farthest | 0.36 | 0.62 [0.56, 0.68] | 0.45 [0.38, 0.52] | 0.44 [0.37, 0.50] |
| more_east | 0.50 | 0.71 [0.65, 0.77] | 0.62 [0.56, 0.68] | 0.60 [0.54, 0.66] |
| more_north | 0.50 | 0.64 [0.58, 0.70] | 0.66 [0.60, 0.72] | 0.67 [0.61, 0.73] |

- By deepest object involved: hop 1: 0.57 [0.55, 0.60]; hop 2: 0.52 [0.50, 0.54]; hop 3: 0.50 [0.47, 0.54]; hop 4: 0.55 [0.49, 0.62]
- Error signature (standardised logistic coefficients): hop -0.08 [-0.14, -0.03], log margin 0.21 [0.16, 0.26]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.36 [0.32, 0.39]
- Object-list line: with 0.53, without 0.48, difference 0.05 [0.01, 0.08]

### qwen3-8b
- 4-object accuracy: 0.42 [0.41, 0.44]; accuracy above chance on 2+ hop questions: 0.01 [0.00, 0.02] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.49 [0.46, 0.51] | 0.49 [0.46, 0.51] | 0.48 [0.45, 0.51] |
| closest | 0.36 | 0.59 [0.55, 0.62] | 0.34 [0.31, 0.38] | 0.26 [0.24, 0.29] |
| direction | 0.12 | 0.36 [0.33, 0.39] | 0.18 [0.15, 0.20] | 0.13 [0.11, 0.15] |
| farthest | 0.36 | 0.44 [0.40, 0.47] | 0.32 [0.29, 0.35] | 0.22 [0.19, 0.26] |
| more_east | 0.50 | 0.69 [0.66, 0.73] | 0.57 [0.54, 0.60] | 0.50 [0.47, 0.53] |
| more_north | 0.50 | 0.73 [0.70, 0.76] | 0.63 [0.60, 0.66] | 0.55 [0.52, 0.58] |

- By deepest object involved: hop 1: 0.52 [0.50, 0.53]; hop 2: 0.42 [0.41, 0.43]; hop 3: 0.36 [0.34, 0.38]; hop 4: 0.35 [0.31, 0.38]
- Error signature (standardised logistic coefficients): hop -0.29 [-0.32, -0.26], log margin 0.08 [0.05, 0.11]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.36 [0.34, 0.39]
- Object-list line: with 0.43, without 0.42, difference 0.00 [-0.01, 0.02]
- Chain of thought on: 0.87 [0.77, 0.95] vs off 0.27 [0.17, 0.38] (same 60 questions; unparsed 10%, median 1625 tokens)

### qwen3-14b
- 4-object accuracy: 0.58 [0.57, 0.60]; accuracy above chance on 2+ hop questions: 0.17 [0.16, 0.18] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.57 [0.54, 0.60] | 0.59 [0.56, 0.62] | 0.60 [0.57, 0.64] |
| closest | 0.36 | 0.63 [0.59, 0.66] | 0.39 [0.35, 0.42] | 0.28 [0.25, 0.30] |
| direction | 0.12 | 0.61 [0.57, 0.64] | 0.45 [0.41, 0.48] | 0.33 [0.30, 0.36] |
| farthest | 0.36 | 0.73 [0.70, 0.76] | 0.53 [0.49, 0.56] | 0.29 [0.26, 0.33] |
| more_east | 0.50 | 0.81 [0.79, 0.84] | 0.74 [0.71, 0.77] | 0.71 [0.68, 0.74] |
| more_north | 0.50 | 0.87 [0.85, 0.90] | 0.80 [0.77, 0.82] | 0.77 [0.75, 0.80] |

- By deepest object involved: hop 1: 0.66 [0.65, 0.67]; hop 2: 0.58 [0.57, 0.60]; hop 3: 0.52 [0.50, 0.54]; hop 4: 0.48 [0.44, 0.52]
- Error signature (standardised logistic coefficients): hop -0.27 [-0.30, -0.24], log margin 0.33 [0.29, 0.37]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.40 [0.37, 0.43]
- Object-list line: with 0.57, without 0.55, difference 0.02 [-0.00, 0.04]
- Chain of thought on: 0.98 [0.95, 1.00] vs off 0.57 [0.43, 0.70] (same 60 questions; unparsed 2%, median 844 tokens)

### gemma3-12b
- 4-object accuracy: 0.66 [0.65, 0.68]; accuracy above chance on 2+ hop questions: 0.22 [0.21, 0.23] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.50 [0.47, 0.53] | 0.49 [0.46, 0.52] | 0.48 [0.45, 0.51] |
| closest | 0.36 | 0.79 [0.76, 0.82] | 0.65 [0.62, 0.68] | 0.55 [0.51, 0.58] |
| direction | 0.12 | 0.72 [0.69, 0.75] | 0.56 [0.53, 0.60] | 0.48 [0.45, 0.51] |
| farthest | 0.36 | 0.75 [0.72, 0.78] | 0.60 [0.56, 0.64] | 0.45 [0.41, 0.49] |
| more_east | 0.50 | 0.86 [0.83, 0.88] | 0.85 [0.83, 0.87] | 0.78 [0.76, 0.81] |
| more_north | 0.50 | 0.87 [0.85, 0.89] | 0.84 [0.81, 0.86] | 0.81 [0.78, 0.83] |

- By deepest object involved: hop 1: 0.77 [0.75, 0.78]; hop 2: 0.63 [0.62, 0.64]; hop 3: 0.59 [0.57, 0.61]; hop 4: 0.56 [0.53, 0.60]
- Error signature (standardised logistic coefficients): hop -0.29 [-0.33, -0.25], log margin 0.35 [0.31, 0.39]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.44 [0.41, 0.46]
- Object-list line: with 0.65, without 0.65, difference 0.01 [-0.01, 0.02]

### gemma3-27b
- 4-object accuracy: 0.68 [0.67, 0.69]; accuracy above chance on 2+ hop questions: 0.27 [0.26, 0.27] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.53 [0.50, 0.56] | 0.53 [0.50, 0.56] | 0.54 [0.51, 0.57] |
| closest | 0.36 | 0.76 [0.73, 0.79] | 0.62 [0.59, 0.66] | 0.55 [0.52, 0.58] |
| direction | 0.12 | 0.77 [0.75, 0.80] | 0.60 [0.57, 0.64] | 0.53 [0.50, 0.56] |
| farthest | 0.36 | 0.77 [0.74, 0.80] | 0.68 [0.64, 0.71] | 0.59 [0.55, 0.62] |
| more_east | 0.50 | 0.87 [0.84, 0.89] | 0.84 [0.81, 0.86] | 0.79 [0.76, 0.81] |
| more_north | 0.50 | 0.86 [0.83, 0.89] | 0.82 [0.80, 0.85] | 0.79 [0.76, 0.82] |

- By deepest object involved: hop 1: 0.75 [0.74, 0.76]; hop 2: 0.67 [0.66, 0.68]; hop 3: 0.63 [0.61, 0.65]; hop 4: 0.62 [0.58, 0.66]
- Error signature (standardised logistic coefficients): hop -0.18 [-0.21, -0.13], log margin 0.33 [0.30, 0.36]
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.48 [0.45, 0.51]
- Object-list line: with 0.67, without 0.64, difference 0.03 [0.02, 0.05]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 23:13 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### gemma3-12b
- 4-object accuracy: 0.68 [0.66, 0.69]; accuracy above chance on 2+ hop questions: 0.23 [0.22, 0.24] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.51 [0.48, 0.54] | 0.49 [0.46, 0.52] | 0.48 [0.45, 0.51] |
| closest | 0.36 | 0.79 [0.76, 0.82] | 0.65 [0.62, 0.69] | 0.55 [0.51, 0.58] |
| direction | 0.12 | 0.77 [0.74, 0.79] | 0.62 [0.59, 0.66] | 0.50 [0.47, 0.53] |
| farthest | 0.36 | 0.75 [0.72, 0.78] | 0.60 [0.57, 0.64] | 0.45 [0.41, 0.49] |
| more_east | 0.50 | 0.86 [0.83, 0.88] | 0.85 [0.83, 0.87] | 0.78 [0.76, 0.81] |
| more_north | 0.50 | 0.87 [0.85, 0.89] | 0.83 [0.81, 0.86] | 0.81 [0.78, 0.83] |

- By deepest object involved: hop 1: 0.78 [0.76, 0.79]; hop 2: 0.64 [0.63, 0.65]; hop 3: 0.59 [0.57, 0.61]; hop 4: 0.57 [0.53, 0.61]
- Error signature (standardised logistic coefficients): hop -0.30 [-0.33, -0.26], log margin 0.31 [0.27, 0.33]
- Validity: median probability on the allowed answers 1.00 (5th percentile 0.95)
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.45 [0.42, 0.48]
- Object-list line: with 0.66, without 0.66, difference 0.01 [-0.01, 0.03]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 23:27 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### qwen3-14b
- 4-object accuracy: 0.56 [0.55, 0.57]; accuracy above chance on 2+ hop questions: 0.15 [0.14, 0.16] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.55 [0.52, 0.58] | 0.53 [0.50, 0.56] | 0.51 [0.48, 0.55] |
| closest | 0.36 | 0.63 [0.59, 0.66] | 0.39 [0.35, 0.42] | 0.28 [0.25, 0.30] |
| direction | 0.12 | 0.55 [0.52, 0.58] | 0.38 [0.34, 0.41] | 0.25 [0.22, 0.28] |
| farthest | 0.36 | 0.73 [0.70, 0.76] | 0.53 [0.49, 0.56] | 0.29 [0.25, 0.33] |
| more_east | 0.50 | 0.82 [0.79, 0.85] | 0.74 [0.71, 0.77] | 0.71 [0.68, 0.74] |
| more_north | 0.50 | 0.87 [0.85, 0.90] | 0.80 [0.77, 0.82] | 0.77 [0.75, 0.80] |

- By deepest object involved: hop 1: 0.63 [0.62, 0.65]; hop 2: 0.57 [0.56, 0.58]; hop 3: 0.49 [0.47, 0.50]; hop 4: 0.45 [0.41, 0.48]
- Error signature (standardised logistic coefficients): hop -0.27 [-0.31, -0.24], log margin 0.25 [0.21, 0.28]
- Validity: median probability on the allowed answers 0.02 (5th percentile 0.00)
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.41 [0.38, 0.44]
- Object-list line: with 0.55, without 0.53, difference 0.02 [-0.01, 0.04]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-09 23:34 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### qwen3-8b
- 4-object accuracy: 0.45 [0.43, 0.47]; accuracy above chance on 2+ hop questions: 0.04 [0.02, 0.05] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.49 [0.45, 0.54] | 0.53 [0.47, 0.58] | 0.52 [0.47, 0.57] |
| closest | 0.36 | 0.54 [0.48, 0.60] | 0.40 [0.34, 0.47] | 0.27 [0.22, 0.33] |
| direction | 0.12 | 0.31 [0.26, 0.36] | 0.25 [0.21, 0.30] | 0.21 [0.17, 0.26] |
| farthest | 0.36 | 0.45 [0.39, 0.52] | 0.34 [0.28, 0.41] | 0.23 [0.18, 0.29] |
| more_east | 0.50 | 0.69 [0.63, 0.75] | 0.55 [0.49, 0.60] | 0.51 [0.44, 0.57] |
| more_north | 0.50 | 0.76 [0.69, 0.81] | 0.61 [0.55, 0.67] | 0.56 [0.50, 0.62] |

- By deepest object involved: hop 1: 0.50 [0.48, 0.53]; hop 2: 0.44 [0.42, 0.46]; hop 3: 0.41 [0.38, 0.44]; hop 4: 0.43 [0.37, 0.50]
- Error signature (standardised logistic coefficients): hop -0.17 [-0.22, -0.11], log margin 0.12 [0.06, 0.17]
- Validity: median probability on the allowed answers 0.00 (5th percentile 0.00)

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-10 00:32 · P1 behaviour screen
**No model passed the gate**, so the fallback applies: qwen3-8b locally, then 14B on a rented GPU, then 3-object scenes.

### gemma3-27b
- 4-object accuracy: 0.70 [0.68, 0.71]; accuracy above chance on 2+ hop questions: 0.28 [0.27, 0.29] → gate **fail**

| question | chance | 3 objects | 4 objects | 5 objects |
|---|---|---|---|---|
| closer_than | 0.50 | 0.53 [0.50, 0.56] | 0.53 [0.50, 0.56] | 0.54 [0.51, 0.57] |
| closest | 0.36 | 0.76 [0.73, 0.79] | 0.62 [0.59, 0.66] | 0.55 [0.52, 0.58] |
| direction | 0.12 | 0.86 [0.83, 0.88] | 0.70 [0.67, 0.73] | 0.59 [0.56, 0.62] |
| farthest | 0.36 | 0.77 [0.74, 0.80] | 0.68 [0.64, 0.71] | 0.59 [0.55, 0.62] |
| more_east | 0.50 | 0.87 [0.84, 0.89] | 0.84 [0.81, 0.86] | 0.79 [0.76, 0.81] |
| more_north | 0.50 | 0.86 [0.83, 0.89] | 0.82 [0.80, 0.85] | 0.79 [0.76, 0.82] |

- By deepest object involved: hop 1: 0.76 [0.75, 0.78]; hop 2: 0.69 [0.68, 0.70]; hop 3: 0.64 [0.62, 0.66]; hop 4: 0.62 [0.58, 0.67]
- Error signature (standardised logistic coefficients): hop -0.18 [-0.22, -0.15], log margin 0.28 [0.24, 0.31]
- Validity: median probability on the allowed answers 1.00 (5th percentile 0.05)
- Near-tie set (margin 0.25–1 unit / 2.5–10°): 0.51 [0.48, 0.54]
- Object-list line: with 0.68, without 0.64, difference 0.05 [0.03, 0.07]

![p1_accuracy](figures/p1_accuracy.png)

## 2026-11-10 00:32 · P1 rescored: model choice
**Chosen: gemma3-12b** on 3-object scenes; excluded question types (at chance): closer_than.

| model | 4-object accuracy | 3-object accuracy | 4-object gate |
|---|---|---|---|
| qwen3-1.7b | 0.42 | 0.47 | fail |
| llama3.2-3b | 0.56 | 0.61 | fail |
| qwen3-4b | 0.43 | 0.55 | fail |
| gemma3-4b | 0.53 | 0.60 | fail |
| qwen3-8b | 0.45 | 0.53 | fail |
| gemma3-12b | 0.68 | 0.76 | fail |
| qwen3-14b | 0.56 | 0.69 | fail |
| gemma3-27b | 0.70 | 0.77 | fail |

## 2026-11-10 00:53 · P2 probes (gemma3-12b)
Model **gemma3-12b**, 3000 scenes (80/20 split by scene), per-object readout in the object list; ridge penalty chosen by scene-grouped cross-validation.

- **Best layer 21: R² = 0.74 [0.72, 0.75]** (1000 bootstrap resamples over test scenes)
- Controls at that layer: shuffled -0.00, layer 0 -0.00, random-init (same layer) -0.00
- Stated-relation baseline (each object's own sentence only): overall 0.70; 1-step 0.92, 2+-step 0.22
- By chain length at layer 21: probe 1-step 0.76, 2+-step 0.74 (gap +0.01); 2+-step advantage over the stated-relation baseline +0.53
- RSA (activation vs true distances, Spearman) at layer 21: 0.19 [0.16, 0.21]; peak 0.20 at layer 19
- Joint end-of-list probe (all objects from one position): peak R² 0.03 at layer 29

Against the preregistered P2 criteria (as clarified):
- R² ≥ 0.6: yes
- beats shuffled, layer-0 and random-init by ≥ 0.3, and the stated-relation baseline by ≥ 0.3 on 2-step objects: yes
- 2-step objects within 0.1 of 1-step: yes
- H-rel signature (R² < 0.3, or 2-step more than 0.2 below 1-step): no

![p2_r2_by_layer](figures/p2_r2_by_layer.png)

![p2_r2_by_hop](figures/p2_r2_by_hop.png)

## 2026-11-10 03:45 · P4 slider (gemma3-12b)
Model **gemma3-12b**, layer 21, using probe directions; 50 scenes, x and y sweeps of ±5.0 units in 0.5-unit steps (including positions no sentence states).

- Agreement with geometry along the sweeps: 0.64 [0.60, 0.68]
- Sweeps where geometry predicts a flip: 268; the model also flips in 2% of them
- Predicted vs observed flip position: r = 0.03; median offset -2.50 units
- Steered activations within the natural norm range (1st–99th percentile): 100%

![p4_slider_demo](figures/p4_slider_demo.png)

![p4_slider_offsets](figures/p4_slider_offsets.png)
