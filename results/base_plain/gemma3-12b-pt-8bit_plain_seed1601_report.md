## Base models in plain text: Gemma 3 12B pt (8-bit), plain prompt — preregistered

Model **Gemma 3 12B pt (8-bit)** (`gemma3-12b-pt-8bit`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.84 [0.80, 0.89] | 0.84 | 0.84 |
| agree | 0.84 [0.80, 0.88] | 0.81 | 0.81 |
| first step only | 0.55 [0.52, 0.58] | 0.54 | 0.87 |
| conflict, C larger | 0.90 [0.86, 0.94] | 0.90 | 0.90 |
| conflict, M larger | 0.24 [0.21, 0.28] | 0.26 | 0.74 |

Final position, layers 28–48 of 48: best held-out R² of M's step 0.73, of C's own step 0.79. Truth probe on the 104 held-out decisive questions (best layer 30): 0.85 [0.76, 0.90]; the model on the same questions 0.25 [0.18, 0.34]. Exploratory: probe at the window's first layer 0.81, mean over the window 0.72; on held-out first-step-only 0.97.

Exploratory: the answer names the first-named object in 0.48 of all questions. Same questions as the chat-format run of `gemma3-12b`: the answers agree on 0.82 of all questions and 0.75 of the decisive ones (chat-format balanced accuracy on the decisive questions 0.37 [0.34, 0.39]).

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_plain.md.
