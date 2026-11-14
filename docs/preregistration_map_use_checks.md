# Preregistration: two checks on whether Gemma 3 12B uses its map

*Committed 2026-11-14, before either check is run. Check A uses existing data (P9); its outcome statistic has not been
computed. Check B needs new activations (Colab, A100).*

## Why

Study 1 found a linearly decodable map of object positions (probe CV R² up to 0.71 at layer 21, P9), but the answers
do not depend on it: erasing it with INLP at every token across layers 12–30 cost no more accuracy than erasing a
random subspace of the same rank (P5c). Two gaps remain before "decodable but not used" is solid:

- **A. Do the model's mistakes follow the map's mistakes?** If the model answered from its map, then where the decoded
  map is wrong, the model should tend to make *the same* wrong answer, more than difficulty alone explains. This
  needs no intervention.
- **B. Does a nonlinear map survive the erasure?** INLP removes only what a linear probe can read. If a nonlinear
  probe still recovers positions after the erasure, P5c's null covered only the linear copy of the map.

## Check A: error overlap (existing P9 data)

- **Data:** `results/p9/gemma3-12b_p9.parquet`, the 300 test chains (seed 602, 5,350 questions). For each question:
  the model's own answer and the answer computed from the probe-decoded map. **Primary layer 21** (P9's primary
  layer); layers 18, 24 and 28 are reported. The scenes and questions are regenerated from the seed, and the
  regeneration is checked against the stored truth, question type and class.
- **Statistic, S (follows the map's mistakes):** among questions where the decoded map's answer is wrong, the share
  where the model gives that same answer.
- **Null (blurred true map):** a map made by adding independent Gaussian noise (isotropic, same sd for every object)
  to the true positions, answered with the same geometry. The noise sd is calibrated **separately in each stratum
  (question type × class)** so that the blurred map's accuracy in that stratum equals the decoded map's (mean over
  draws, matched to within 0.01 by bisection). 200 draws. S_blur is computed in the same way with expected counts over
  draws (per question: the probability that the blurred map is wrong, and the probability that it is wrong with the
  model's answer). This holds fixed how often the map errs and on which kinds of question, so it tests whether the
  model shares the decoded map's *specific* errors beyond what near-ties and question difficulty produce.
- **Effect:** S − S_blur, with a 95% scene-bootstrap CI (1,000 resamples; ratios of sums recomputed per resample).
- **Decision rule (layer 21, all questions):**
  1. **The model's mistakes follow the map:** CI lower bound > 0 and effect ≥ 0.05.
  2. Otherwise, **small overlap:** CI lower bound > 0.
  3. Otherwise: **no sign that the mistakes follow the map**.
- **Reported, not part of the rule:** the same by class (CM, CR, MR) and by layer; S and S_blur separately; the
  calibrated noise per stratum; the correlation of correctness (phi) for the decoded and the blurred map.
- **Caveat stated in advance:** an overlap shows a shared source of error, not necessarily that the answer reads the
  map (both could inherit the same misreading of a sentence). No overlap, with a map this accurate, is evidence
  against the model answering from it.

## Check B: nonlinear probe after erasure (new run)

- **Model and erasure:** gemma3-12b, bfloat16. P5c's INLP bases and its random bases of the same rank
  (`results/p5/gemma3-12b_p5c_bases.pt`), applied exactly as in P5c's "window 12–30" condition: at every token
  position, at the output of layers 12–30.
- **Data:** 2,000 new 3-object scenes (seed 901, the generator of P5c's fit), prompts without a question. Activations
  at the object-list positions, captured after the erasure, at layers **14, 21, 28** (inside the window) and **36**
  (after it).
- **Conditions:** clean, probe-erased, random-erased.
- **Probes**, both on scene-centred activations and scene-centred (x, y), 3-fold scene-grouped CV, R² pooled over x
  and y (as in Study 1):
  - *Linear:* the Study 1 ridge probe (GroupRidge).
  - *MLP:* inputs standardised with the training fold's statistics; two hidden layers of 512 units, GELU, dropout
    0.1; AdamW (learning rate 1e-3, weight decay 1e-2), batches of 256, at most 300 epochs, early stopping (patience
    20) on 10% of the training fold's scenes; predictions from the best epoch.
  - R² CIs: 95% scene bootstrap of the out-of-fold predictions (1,000 resamples).
- **Validity check:** the MLP must be a fair probe. On clean activations at layer 21 its R² must be at least the
  linear probe's minus 0.05. Otherwise the check is **inconclusive**.
- **Decision rule (layer 21, MLP R² after probe erasure):**
  1. **A nonlinear map survives the erasure:** R² ≥ 0.30.
  2. **The erasure removes the map, linear and nonlinear:** R² < 0.10, while the MLP's R² after random erasure is
     ≥ 0.50.
  3. Otherwise: **partly**.
- **Reported, not part of the rule:** every layer × condition × probe, with CIs; R² by object hop (root, one step,
  two steps).

## What each outcome leads to

- **A follows the map, or B survives:** the "not used" conclusion is weaker than stated. Next: test use of the
  nonlinear map (or the shared error source) directly before any long-chain study.
- **A no sign and B removes the map:** "decodable but not used" stands with both gaps closed. Next: the long-chain /
  walk experiment, built so that the answer cannot be found by re-reading one or two sentences, to find questions
  where the model needs its map.
- Anything else: stop and decide with the author.
