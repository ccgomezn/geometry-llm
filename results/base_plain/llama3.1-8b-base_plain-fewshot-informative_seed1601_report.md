## Base models with worked examples (informative): Llama 3.1 8B (base) — preregistered

Model **Llama 3.1 8B (base)** (`llama3.1-8b-base`), **plain** prompt (begin-of-text token, scene, `Question: …`, `Answer:`; candidates read with a leading space), preceded by six worked **informative** examples (seed-7 bank, no names shared with the test scene). The flow set's questions (seed 1601, 2400 questions in 859 scenes); answer = sign of log p(C) − log p(R). Balanced accuracy with 95% scene-bootstrap CIs.

| question kind | balanced accuracy | accuracy | answer follows the one-sentence rule |
|---|---|---|---|
| own step only | 0.60 [0.54, 0.66] | 0.60 | 0.60 |
| agree | 0.69 [0.64, 0.74] | 0.69 | 0.69 |
| first step only | 0.53 [0.49, 0.58] | 0.53 | 0.65 |
| conflict, C larger | 0.67 [0.62, 0.72] | 0.67 | 0.67 |
| conflict, M larger | 0.47 [0.43, 0.51] | 0.47 | 0.53 |

Final position, layers 18–32 of 32: best held-out R² of M's step 0.81, of C's own step 0.81. Truth probe on the 104 held-out decisive questions (best layer 18): 0.95 [0.89, 0.98]; the model on the same questions 0.52 [0.42, 0.61]. Exploratory: probe at the window's first layer 0.95, mean over the window 0.78; on held-out first-step-only 0.99.

Exploratory: the answer names the first-named object in 0.62 of all questions. Weight of M's step relative to C's own step in the margin 0.26 (1 = composes). Answers following M's sentence (C iff M's step points the asked way): own step only 0.54, agree 0.69, first step only 0.53, conflict, C larger 0.33, conflict, M larger 0.47. Agreement with the zero-shot plain run: 0.69 of all questions, 0.71 of the decisive ones.

Preregistered verdict: **uninformative**. Rules: docs/preregistration_base_fewshot.md.
