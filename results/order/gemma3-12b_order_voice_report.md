## Order and voice: what does the one-sentence rule follow? (gemma3-12b) — preregistered

Model **gemma3-12b**. 1800 new C–R axis questions (seed 2001), each in four renderings of the same scene: sentence order MC (C's sentence last, as before) or CM (C's sentence first), and C's relation in subject voice ("C is d east of M") or object voice ("M is d west of C"). Answer = sign of logit(C) − logit(R). Scene-bootstrap 95% CIs.

**Rule coefficients** (pooled regression of the logit difference on own_yes, word_yes, last_yes, d1, d2, name order, order and voice):

| feature | coefficient [95% CI] |
|---|---|
| own_yes | +8.03 [+6.85, +9.20] |
| word_yes | +11.19 [+10.66, +11.72] |
| last_yes | -1.32 [-1.84, -0.78] |

| difference | estimate [95% CI] |
|---|---|
| own − word | -3.15 [-4.52, -1.88] |
| own − last | +9.36 [+8.09, +10.60] |
| word − last | +12.51 [+11.60, +13.33] |

Other coefficients: d1 +0.43/unit, d2 +0.72/unit, C named first -0.40, CM order +6.10, object voice +5.04.

**Accuracy** by kind of question and rendering:

| kind | MC-subj | CM-subj | MC-obj | CM-obj |
|---|---|---|---|---|
| own step only | 0.97 | 0.97 | 0.41 | 0.47 |
| agree | 0.95 | 0.91 | 0.59 | 0.54 |
| first step only | 0.49 | 0.52 | 0.45 | 0.44 |
| conflict, C larger | 0.86 | 0.88 | 0.35 | 0.62 |
| conflict, M larger | 0.31 | 0.18 | 0.54 | 0.38 |

**Share answering C**, by the three features (own, word, last):

| rendering | own | word | last | n | answers C |
|---|---|---|---|---|---|
| CM-obj | 0 | 0 | 0 | 140 | 0.49 |
| CM-obj | 0 | 0 | 1 | 160 | 0.38 |
| CM-obj | 0 | 1 | 0 | 288 | 0.81 |
| CM-obj | 0 | 1 | 1 | 456 | 0.61 |
| CM-obj | 1 | 0 | 0 | 590 | 0.87 |
| CM-obj | 1 | 0 | 1 | 166 | 0.69 |
| CM-subj | 0 | 0 | 0 | 428 | 0.02 |
| CM-subj | 0 | 0 | 1 | 616 | 0.04 |
| CM-subj | 1 | 1 | 0 | 590 | 0.75 |
| CM-subj | 1 | 1 | 1 | 166 | 0.85 |
| MC-obj | 0 | 0 | 0 | 300 | 0.03 |
| MC-obj | 0 | 1 | 1 | 744 | 0.33 |
| MC-obj | 1 | 0 | 0 | 756 | 0.25 |
| MC-subj | 0 | 0 | 0 | 1044 | 0.01 |
| MC-subj | 1 | 1 | 1 | 756 | 0.68 |

**Decisive questions** (M's step larger and opposite) by rendering:

| rendering | accuracy | right when C correct | right when R correct | AUC |
|---|---|---|---|---|
| MC-subj | 0.31 | 0.01 | 0.65 | 0.14 |
| CM-subj | 0.18 | 0.00 | 0.39 | 0.09 |
| MC-obj | 0.54 | 0.21 | 0.92 | 0.57 |
| CM-obj | 0.38 | 0.54 | 0.20 | 0.37 |

Weight of M's step (d1 coefficient, logits/unit) by rendering: MC-subj +1.20, CM-subj +0.51, MC-obj +0.18, CM-obj -0.42.

Preregistered verdict: **the rule follows the direction word**. Rules: docs/preregistration_order_voice.md.
