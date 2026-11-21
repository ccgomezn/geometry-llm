## Step by step: does written reasoning fix the one-sentence rule? (llama3.1-8b) — preregistered

Model **llama3.1-8b**. 600 C–R axis questions from the flow set's generator (seed 1601; the first 300 decisive and 75 of each other kind). Same weights, two conditions: *one word* (sign of logit(C) − logit(R)) and *step by step* (greedy, up to 1024 new tokens, last 'Answer: <name>' line; none = wrong). Balanced accuracy (mean of the accuracies when the answer is C and when it is R), 95% scene-bootstrap CIs, differences paired.

| questions | step by step | one word | step by step − one word |
|---|---|---|---|
| decisive (*conflict, M larger*) | 0.57 [0.52, 0.64] | 0.20 [0.15, 0.25] | 0.37 [0.31, 0.44] |
| the other four kinds | 0.82 [0.78, 0.86] | 0.72 [0.67, 0.78] | |

**Accuracy by kind:**

| kind | n | step by step | one word |
|---|---|---|---|
| own step only | 75 | 0.93 | 0.73 |
| agree | 75 | 0.93 | 0.83 |
| first step only | 75 | 0.48 | 0.33 |
| conflict, C larger | 75 | 0.89 | 0.92 |
| conflict, M larger | 300 | 0.59 | 0.21 |

No final answer: 60; hit the token limit: 31; median reply length 161 tokens.
Exploratory, decisive questions: 8 of 300 replies write (x, y) coordinates (accuracy 0.88), 292 do not (accuracy 0.58). Replies giving positions to both C and R: 4; the written positions put C on the correct side in 1.00, and the final answer agrees with them in 1.00.
Check: one-word accuracy on the decisive questions here 0.21; in the bfloat16 flow run of llama3.1-8b (all 600) 0.18.

Preregistered verdict: **reasoning helps**. Rules: docs/preregistration_cot_axis.md.
