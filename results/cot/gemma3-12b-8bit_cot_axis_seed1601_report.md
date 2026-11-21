## Step by step: does written reasoning fix the one-sentence rule? (gemma3-12b-8bit) — preregistered, rescored with the corrected answer parser

Model **gemma3-12b-8bit**. 600 C–R axis questions from the flow set's generator (seed 1601; the first 300 decisive and 75 of each other kind). Same weights, two conditions: *one word* (sign of logit(C) − logit(R)) and *step by step* (greedy, up to 1024 new tokens, last 'Answer: <name>' line; none = wrong). Balanced accuracy (mean of the accuracies when the answer is C and when it is R), 95% scene-bootstrap CIs, differences paired.

| questions | step by step | one word | step by step − one word |
|---|---|---|---|
| decisive (*conflict, M larger*) | 0.81 [0.76, 0.85] | 0.36 [0.32, 0.40] | 0.44 [0.38, 0.51] |
| the other four kinds | 0.93 [0.90, 0.96] | 0.81 [0.77, 0.85] | |

**Accuracy by kind:**

| kind | n | step by step | one word |
|---|---|---|---|
| own step only | 75 | 0.99 | 0.99 |
| agree | 75 | 1.00 | 0.91 |
| first step only | 75 | 0.72 | 0.36 |
| conflict, C larger | 75 | 1.00 | 0.89 |
| conflict, M larger | 300 | 0.83 | 0.42 |

No final answer: 5; hit the token limit: 0; median reply length 133 tokens.
Exploratory, decisive questions: 71 of 300 replies write (x, y) coordinates (accuracy 0.93), 229 do not (accuracy 0.80). Replies giving positions to both C and R: 26; the written positions put C on the correct side in 1.00, and the final answer agrees with them in 0.88.
Check: one-word accuracy on the decisive questions here 0.42; in the bfloat16 flow run of gemma3-12b (all 600) 0.40.

Preregistered verdict: **reasoning fixes the rule**. Rules: docs/preregistration_cot_axis.md.
