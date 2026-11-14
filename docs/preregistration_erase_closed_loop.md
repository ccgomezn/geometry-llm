# Preregistration: erase the map so that it stays erased (closed-loop erasure), then test the answers

*Committed 2026-11-14, before any data of this test exists. Gemma 3 12B instruct, bfloat16 (Colab A100).*

## Why

P5c erased the probe-readable map (INLP, every token, layers 12–30) and the answers lost no more accuracy than with a
random subspace, which we read as "the map is not used". Check B (`docs/preregistration_map_use_checks.md`) showed the
erasure did not hold in the running model: a probe refitted on the erased run still reads positions with R² 0.57 at
layer 21 (0.68 clean). The basis was fitted once on clean activations; in the erased forward pass the map reappears in
other directions. Check A found that the model's mistakes follow the decoded map's on the composed (C–R) questions. So
whether the answers use the map is open. This test erases the map in the running model and checks that it is gone
before scoring the answers.

## Design

### Where and how to erase

- **Layers:** every layer from 12 (where the map starts to form) to the last (48), each with its own basis. P5c stopped
  at 30, and the map is still readable after it (Check B, layer 36).
- **Positions:** every token **except the first** (`<bos>`). Change from P5c, decided before any data: the first token
  cannot carry scene information (causal attention), and in Gemma it holds very large activations that act as an
  attention anchor, so projecting it only damages the model. That damage hits random bases at least as much as probe
  bases and is a plausible reason why P5c's random erasure cost so much (8 random directions at layer 21: −0.09 on
  stated-only questions).
- **Operation:** as P5c, h ← h − (h·B − c)Bᵀ, with c = μ·B (μ = mean object-list activation).

### Fitting the basis in closed loop

- **Data:** 2,000 new 3-object scenes (seed 1001), prompts without a question, object-list positions, scene-centred
  (x, y), as in P5c.
- **Start:** P5c's bases at layers 12–30, empty at 31–48.
- **Each round:** run the model with the current bases erased, capture the object-list activations at every erased
  layer *after* the erasure, and at each layer fit the Study 1 ridge probe (scene-grouped 3-fold CV). Where its CV R²
  ≥ 0.05, run INLP on those activations (up to 5 steps, stopping at CV R² < 0.05) and add the new directions to the
  layer's basis (orthonormalised); c is recomputed from the erased run's mean activation.
- **Stop** when every layer's CV R² < 0.05 in the same round, or after 10 rounds.
- **Random control:** at each layer a random orthonormal basis of the final rank (seeded by layer), with c from the
  clean mean activation, applied at the same layers and positions.

### Is the erasure effective? (fresh scenes)

- 1,000 new 3-object scenes (seed 1002), activations captured under each erasure.
- Linear ridge probe CV R² at **every** erased layer; MLP probe (as in check B) at layers 14, 21, 28, 36 and 44. The
  same five layers are probed under the random erasure.
- **Effective** if the linear R² is < 0.10 at every erased layer under the probe-basis erasure.

### Do the answers suffer?

- **Test:** P5c's test set: 300 chain scenes R → M → C (seed 202), all questions passing the 1-unit / 10° margins
  except *closer than*. *Composed* questions involve the two-step object C; *stated-only* ones involve only R and M.
- **Conditions:** clean, probe-basis erased, random-basis erased.
- **Metric (as P5c):** drop = clean accuracy − erased accuracy; *excess drop* = drop (probe) − drop (random); 95%
  scene-bootstrap CIs.
- **Chance:** per question 1 / number of answer options, averaged over the composed questions.

## Decision rule

1. Not effective (linear R² ≥ 0.10 at some erased layer on the fresh scenes): **the map could not be erased**. No
   verdict on use.
2. Otherwise, if the random erasure leaves composed accuracy below chance + 0.15: **uninformative** (the erasure
   damages the model in general).
3. Otherwise, as P5c:
   - **map used:** composed excess drop ≥ 0.10 and at least 0.05 above the stated-only excess drop;
   - **partial use:** composed excess drop CI lower bound > 0 and composed excess > stated-only excess;
   - **map not used:** composed excess drop ≤ 0.03;
   - **unclear** otherwise.
   If the MLP R² at layer 21 under the probe erasure is ≥ 0.30, "map not used" becomes **"linear map not used"**
   (a nonlinear copy remained).

## Reported, not part of the rule

Rounds taken and rank per layer; R² per layer and round; every probe on the fresh scenes; accuracy per question type
and class (CM, CR, MR); the same test scored with P5c's original bases but sparing the first token (a check of the
first-token explanation).

## What each outcome leads to

- **Map used:** the headline. Next: replicate on 4–5-object scenes (P6) and on 27B; check which question classes
  carry the drop (CR expected, from check A).
- **Partial use or unclear:** report as is; decide with the author.
- **Map not used:** "decodable but not used" is then supported by an erasure that holds. Next: the long-chain / walk
  experiment, where re-reading is too costly.
- **Could not be erased / uninformative:** report; decide with the author.
