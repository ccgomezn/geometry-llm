## Base models in plain text: Gemma 3 12B it (8-bit), plain prompt — preregistered

Model **Gemma 3 12B it (8-bit)** (`gemma3-12b-8bit`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.99 [0.98, 1.00] | 0.99 | 0.99 |
| agree | 0.97 [0.96, 0.99] | 0.97 | 0.97 |
| first step only | 0.55 [0.53, 0.57] | 0.53 | 0.95 |
| conflict, C larger | 0.87 [0.83, 0.91] | 0.86 | 0.86 |
| conflict, M larger | 0.40 [0.38, 0.43] | 0.45 | 0.55 |

Final position, layers 28–48 of 48: best held-out R² of M's step 0.83, of C's own step 0.87. Truth probe on the 104 held-out decisive questions (best layer 28): 0.89 [0.82, 0.94]; the model on the same questions 0.37 [0.28, 0.46]. Exploratory: probe at the window's first layer 0.89, mean over the window 0.83; on held-out first-step-only 1.00.

Exploratory: the answer names the first-named object in 0.50 of all questions. Same questions as the chat-format run of `gemma3-12b`: the answers agree on 0.95 of all questions and 0.90 of the decisive ones (chat-format balanced accuracy on the decisive questions 0.37 [0.34, 0.39]).

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_plain.md.
