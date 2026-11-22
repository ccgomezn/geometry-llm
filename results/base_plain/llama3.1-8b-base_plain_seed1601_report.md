## Base models in plain text: Llama 3.1 8B (base), plain prompt — preregistered

Model **Llama 3.1 8B (base)** (`llama3.1-8b-base`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.57 [0.53, 0.62] | 0.57 | 0.57 |
| agree | 0.57 [0.53, 0.62] | 0.56 | 0.56 |
| first step only | 0.50 [0.47, 0.53] | 0.49 | 0.67 |
| conflict, C larger | 0.62 [0.58, 0.66] | 0.62 | 0.62 |
| conflict, M larger | 0.45 [0.42, 0.48] | 0.46 | 0.54 |

Final position, layers 18–32 of 32: best held-out R² of M's step 0.66, of C's own step 0.62. Truth probe on the 104 held-out decisive questions (best layer 18): 0.81 [0.72, 0.87]; the model on the same questions 0.45 [0.36, 0.55]. Exploratory: probe at the window's first layer 0.81, mean over the window 0.69; on held-out first-step-only 1.00.

Exploratory: the answer names the first-named object in 0.79 of all questions. Same questions as the chat-format run of `llama3.1-8b`: the answers agree on 0.57 of all questions and 0.50 of the decisive ones (chat-format balanced accuracy on the decisive questions 0.19 [0.15, 0.23]).

Preregistered verdict: **uninformative**. Rules: docs/preregistration_base_plain.md.
