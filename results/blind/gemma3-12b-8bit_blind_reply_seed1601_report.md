## Blind reply: can written reasoning use the processed state when M's sentence is hidden? (gemma3-12b-8bit) — preregistered

Model **gemma3-12b-8bit**. 200 decisive questions (flow set's generator, seed 1601), greedy step-by-step replies in four conditions: *normal*; *blind reply* (prompt read normally, the reply cannot attend to M's sentence); *floor* (neither the prompt positions after M's sentence nor the reply can attend to it); *list control* (the reply cannot attend to the object list). Positions numbered as in the normal prompt. Balanced accuracy, 95% scene-bootstrap CIs; no answer = wrong.

| condition | balanced accuracy | accuracy | no answer |
|---|---|---|---|
| normal | 0.81 [0.74, 0.87] | 0.84 | 2 |
| blind reply | 0.23 [0.17, 0.29] | 0.26 | 4 |
| floor | 0.16 [0.10, 0.21] | 0.12 | 23 |
| list control | 0.76 [0.69, 0.83] | 0.81 | 1 |

**Recovery** (blind reply − floor) / (normal − floor): 0.11 [-0.01, 0.23].

Exploratory, blind-reply replies: 8 give M a position relative to R, on the correct side in 0.88; 14 of 200 say something is not given or unknown.

Preregistered verdict: **partial**. Rules: docs/preregistration_blind_reply.md.
