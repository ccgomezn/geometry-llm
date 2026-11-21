## Read-back: does written reasoning take M's step from its sentence or the processed state? (gemma3-12b-8bit) — preregistered

Model **gemma3-12b-8bit**. 200 decisive questions (flow set's generator, seed 1601) with a valid flipped version (M's step negated on the asked axis, which flips the answer). Greedy step-by-step replies in four conditions: original, flipped, and two mixes in which M's sentence comes from one version and every layer's state at every later position from the other (mixed A: sentence original, state flipped; mixed B: the reverse). Shares over replies with an answer.

| condition | replies with an answer | follows M's sentence | follows the downstream state |
|---|---|---|---|
| original | 198 of 200 | 0.87 | (same version) |
| flipped | 200 of 200 | 0.99 | (same version) |
| mixed A | 200 of 200 | 0.79 | 0.21 |
| mixed B | 200 of 200 | 1.00 | 0.00 |

Accuracy original 0.87, flipped 0.99; mixed replies following M's sentence 0.89. **Index** (1 = follows M's sentence, 0 = follows the downstream state): 0.95 [0.91, 0.99].

Exploratory, mixed replies: 50 give M a position relative to R, which follows M's sentence in 0.90; 25 give C one, which follows M's sentence in 0.96; of 19 giving both, 0 mix the versions (M from one, C from the other). No answer: original 2, flipped 0, mixed A 0, mixed B 0.

Preregistered verdict: **the reasoning re-reads the sentence**. Rules: docs/preregistration_readback.md.
