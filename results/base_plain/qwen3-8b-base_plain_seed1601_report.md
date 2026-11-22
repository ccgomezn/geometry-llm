## Base models in plain text: Qwen3-8B-Base, plain prompt — preregistered

Model **Qwen3-8B-Base** (`qwen3-8b-base`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.80 [0.75, 0.84] | 0.80 | 0.80 |
| agree | 0.78 [0.75, 0.82] | 0.74 | 0.74 |
| first step only | 0.50 [0.48, 0.52] | 0.48 | 0.94 |
| conflict, C larger | 0.74 [0.69, 0.78] | 0.72 | 0.72 |
| conflict, M larger | 0.44 [0.41, 0.47] | 0.47 | 0.53 |

Final position, layers 20–36 of 36: best held-out R² of M's step 0.79, of C's own step 0.84. Truth probe on the 104 held-out decisive questions (best layer 24): 0.87 [0.79, 0.92]; the model on the same questions 0.42 [0.33, 0.52]. Exploratory: probe at the window's first layer 0.84, mean over the window 0.80; on held-out first-step-only 1.00.

Exploratory: the answer names the first-named object in 0.57 of all questions. Same questions as the chat-format run of `qwen3-8b`: the answers agree on 0.65 of all questions and 0.52 of the decisive ones (chat-format balanced accuracy on the decisive questions 0.37 [0.32, 0.42]).

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_plain.md.
