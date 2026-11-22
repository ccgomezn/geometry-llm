## Base models in plain text: Llama 3.1 8B Instruct, plain prompt — preregistered

Model **Llama 3.1 8B Instruct** (`llama3.1-8b`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.81 [0.77, 0.85] | 0.81 | 0.81 |
| agree | 0.83 [0.79, 0.87] | 0.82 | 0.82 |
| first step only | 0.52 [0.49, 0.55] | 0.51 | 0.91 |
| conflict, C larger | 0.80 [0.76, 0.85] | 0.81 | 0.81 |
| conflict, M larger | 0.33 [0.28, 0.38] | 0.33 | 0.67 |

Final position, layers 18–32 of 32: best held-out R² of M's step 0.78, of C's own step 0.78. Truth probe on the 104 held-out decisive questions (best layer 18): 0.86 [0.78, 0.91]; the model on the same questions 0.35 [0.26, 0.44]. Exploratory: probe at the window's first layer 0.86, mean over the window 0.73; on held-out first-step-only 1.00.

Exploratory: the answer names the first-named object in 0.52 of all questions. Same questions as the chat-format run of `llama3.1-8b`: the answers agree on 0.81 of all questions and 0.72 of the decisive ones (chat-format balanced accuracy on the decisive questions 0.19 [0.15, 0.23]).

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_plain.md.
