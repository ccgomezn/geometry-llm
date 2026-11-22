## Base models with worked examples (neutral): Qwen3-8B-Base — preregistered

Model **Qwen3-8B-Base** (`qwen3-8b-base`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space), preceded by six worked **neutral** examples (seed-7 bank, no names shared with the test scene). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.97 [0.96, 0.99] | 0.97 | 0.97 |
| agree | 0.94 [0.91, 0.96] | 0.93 | 0.93 |
| first step only | 0.53 [0.52, 0.55] | 0.52 | 0.96 |
| conflict, C larger | 0.95 [0.93, 0.97] | 0.94 | 0.94 |
| conflict, M larger | 0.31 [0.27, 0.34] | 0.34 | 0.66 |

Final position, layers 20–36 of 36: best held-out R² of M's step 0.90, of C's own step 0.92. Truth probe on the 104 held-out decisive questions (best layer 22): 0.99 [0.95, 1.00]; the model on the same questions 0.32 [0.24, 0.41]. Exploratory: probe at the window's first layer 0.97, mean over the window 0.92; on held-out first-step-only 1.00.

Exploratory: the answer names the first-named object in 0.47 of all questions. Weight of M's step relative to C's own step in the margin 0.20 (1 = composes). Answers following M's sentence (C iff M's step points the asked way): own step only 0.54, agree 0.93, first step only 0.52, conflict, C larger 0.06, conflict, M larger 0.34. Agreement with the zero-shot plain run: 0.80 of all questions, 0.73 of the decisive ones.

Preregistered verdict: **the map is outvoted**. Rules: docs/preregistration_base_fewshot.md.
