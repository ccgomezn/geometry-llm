## Base models with worked examples (neutral): Gemma 3 12B pt (8-bit) — preregistered

Model **Gemma 3 12B pt (8-bit)** (`gemma3-12b-pt-8bit`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space), preceded by six worked **neutral** examples (seed-7 bank, no names shared with the test scene). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.94 [0.91, 0.97] | 0.94 | 0.94 |
| agree | 0.95 [0.93, 0.97] | 0.95 | 0.95 |
| first step only | 0.53 [0.50, 0.55] | 0.51 | 0.94 |
| conflict, C larger | 0.99 [0.98, 1.00] | 0.99 | 0.99 |
| conflict, M larger | 0.12 [0.09, 0.15] | 0.13 | 0.87 |

Final position, layers 28–48 of 48: best held-out R² of M's step 0.88, of C's own step 0.90. Truth probe on the 104 held-out decisive questions (best layer 28): 0.96 [0.91, 0.98]; the model on the same questions 0.13 [0.08, 0.21]. Exploratory: probe at the window's first layer 0.96, mean over the window 0.90; on held-out first-step-only 1.00.

Exploratory: the answer names the first-named object in 0.52 of all questions. Weight of M's step relative to C's own step in the margin 0.08 (1 = composes). Answers following M's sentence (C iff M's step points the asked way): own step only 0.48, agree 0.95, first step only 0.51, conflict, C larger 0.01, conflict, M larger 0.13. Agreement with the zero-shot plain run: 0.83 of all questions, 0.76 of the decisive ones.

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_fewshot.md.
