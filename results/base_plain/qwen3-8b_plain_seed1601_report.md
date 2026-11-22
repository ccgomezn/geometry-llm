## Base models in plain text: Qwen3-8B, plain prompt — preregistered

Model **Qwen3-8B** (`qwen3-8b`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.70 [0.66, 0.75] | 0.70 | 0.70 |
| agree | 0.73 [0.68, 0.77] | 0.73 | 0.73 |
| first step only | 0.61 [0.57, 0.64] | 0.60 | 0.69 |
| conflict, C larger | 0.65 [0.61, 0.70] | 0.66 | 0.66 |
| conflict, M larger | 0.45 [0.41, 0.48] | 0.45 | 0.55 |

Final position, layers 20–36 of 36: best held-out R² of M's step 0.77, of C's own step 0.81. Truth probe on the 104 held-out decisive questions (best layer 20): 0.82 [0.73, 0.88]; the model on the same questions 0.39 [0.31, 0.49]. Exploratory: probe at the window's first layer 0.82, mean over the window 0.77; on held-out first-step-only 1.00.

Exploratory: the answer names the first-named object in 0.71 of all questions. Same questions as the chat-format run of `qwen3-8b`: the answers agree on 0.62 of all questions and 0.56 of the decisive ones (chat-format balanced accuracy on the decisive questions 0.37 [0.32, 0.42]).

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_plain.md.
