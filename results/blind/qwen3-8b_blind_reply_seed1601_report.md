## Blind reply: can written reasoning use the processed state when M's sentence is hidden? (qwen3-8b) — preregistered

Model **qwen3-8b**. 200 decisive questions (flow set's generator, seed 1601), greedy step-by-step replies in four conditions: *normal*; *blind reply* (prompt read normally, the reply cannot attend to M's sentence); *floor* (neither the prompt positions after M's sentence nor the reply can attend to it); *list control* (the reply cannot attend to the object list). Positions numbered as in the normal prompt. Balanced accuracy, 95% scene-bootstrap CIs; no answer = wrong.

| condition | balanced accuracy | accuracy | no answer |
|---|---|---|---|
| normal | 0.99 [0.97, 1.00] | 0.99 | 1 |
| blind reply | 0.12 [0.07, 0.17] | 0.12 | 56 |
| floor | 0.06 [0.03, 0.10] | 0.04 | 168 |
| list control | 0.98 [0.96, 1.00] | 0.98 | 0 |

**Recovery** (blind reply − floor) / (normal − floor): 0.06 [-0.00, 0.12].

Exploratory, blind-reply replies: 71 give M a position relative to R, on the correct side in 0.37; 29 of 200 say something is not given or unknown.

Preregistered verdict: **the processed state is not usable by the reply**. Rules: docs/preregistration_blind_reply.md.
