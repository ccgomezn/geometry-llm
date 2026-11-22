## Base models with worked examples (neutral): Llama 3.1 8B (base) — preregistered

Model **Llama 3.1 8B (base)** (`llama3.1-8b-base`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space), preceded by six worked **neutral** examples (seed-7 bank, no names shared with the test scene). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.66 [0.61, 0.71] | 0.66 | 0.66 |
| agree | 0.69 [0.65, 0.73] | 0.68 | 0.68 |
| first step only | 0.55 [0.51, 0.58] | 0.55 | 0.64 |
| conflict, C larger | 0.67 [0.63, 0.71] | 0.68 | 0.68 |
| conflict, M larger | 0.45 [0.42, 0.48] | 0.45 | 0.55 |

Final position, layers 18–32 of 32: best held-out R² of M's step 0.78, of C's own step 0.79. Truth probe on the 104 held-out decisive questions (best layer 26): 0.88 [0.80, 0.93]; the model on the same questions 0.45 [0.36, 0.55]. Exploratory: probe at the window's first layer 0.87, mean over the window 0.80; on held-out first-step-only 0.97.

Exploratory: the answer names the first-named object in 0.74 of all questions. Weight of M's step relative to C's own step in the margin 0.16 (1 = composes). Answers following M's sentence (C iff M's step points the asked way): own step only 0.53, agree 0.68, first step only 0.55, conflict, C larger 0.33, conflict, M larger 0.45. Agreement with the zero-shot plain run: 0.75 of all questions, 0.79 of the decisive ones.

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_fewshot.md.
