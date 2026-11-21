## Step by step: does written reasoning fix the one-sentence rule? (qwen3-8b) — preregistered, rescored with the corrected answer parser

Model **qwen3-8b**. 600 C–R axis questions from the flow set's generator (seed 1601; the first 300 decisive and 75 of each other kind). Same weights, two conditions: *one word* (sign of logit(C) − logit(R)) and *step by step* (greedy, up to 1024 new tokens, last 'Answer: <name>' line; none = wrong). Balanced accuracy (mean of the accuracies when the answer is C and when it is R), 95% scene-bootstrap CIs, differences paired.

| questions | step by step | one word | step by step − one word |
|---|---|---|---|
| decisive (*conflict, M larger*) | 0.99 [0.98, 1.00] | 0.36 [0.29, 0.43] | 0.63 [0.56, 0.71] |
| the other four kinds | 0.98 [0.96, 1.00] | 0.65 [0.58, 0.71] | |

**Accuracy by kind:**

| kind | n | step by step | one word |
|---|---|---|---|
| own step only | 75 | 0.99 | 0.65 |
| agree | 75 | 1.00 | 0.69 |
| first step only | 75 | 0.96 | 0.52 |
| conflict, C larger | 75 | 0.97 | 0.73 |
| conflict, M larger | 300 | 0.99 | 0.36 |

No final answer: 4; hit the token limit: 1; median reply length 287 tokens.
Exploratory, decisive questions: 206 of 300 replies write (x, y) coordinates (accuracy 1.00), 94 do not (accuracy 0.98). Replies giving positions to both C and R: 202; the written positions put C on the correct side in 1.00, and the final answer agrees with them in 1.00.
Check: one-word accuracy on the decisive questions here 0.36; in the bfloat16 flow run of qwen3-8b (all 600) 0.36.

Preregistered verdict: **reasoning fixes the rule**. Rules: docs/preregistration_cot_axis.md.
