# Study 2: the shortcut dial (triangles) — design

Status: draft for approval · 2026-11-13 · Builds on Study 1 (paper: doi 10.5281/zenodo.24178043)

## Why

Study 1 found that Gemma 3 builds an accurate spatial map of a scene described in text, but its answers never read it:
erasing the map costs nothing, and the answers come from a shortcut over each stated sentence. Every Study 1 question
could be answered by chaining stated sentences, so the shortcut was always available.

Study 2 asks the obvious next question: **does the model use its map when the shortcut stops working?** We build one
family of scenes (a triangle plus a point) and ask questions on a ladder that runs from "a stated sentence answers it"
to "only the geometry answers it". At every rung we measure how much the answers depend on the map.

The author's hoped-for headline is "the model keeps its map as a backup and uses it when stated facts are not enough".
The design must be able to show that, and must be equally able to show the opposite ("it never reads its map, even
when the map is the only route to the answer"). Both are publishable; preregistration fixes the rules before data.

## Goals

1. A scene generator for a triangle A, B, C plus a point P, described only through relations (Study 1 style).
2. A question ladder with four rungs on the same scenes (below), with exact ground truth.
3. A behavioural pilot (answers only) that decides whether the main study is worth running.
4. A main study measuring, per rung: accuracy, damage from erasing the map vs erasing random directions, agreement
   between the model and answers read off its map, and whether moving a corner inside the map flips rung-4 answers as
   geometry predicts.
5. Preregistration of the pilot and of each main-study test, committed before its data exists.

## Non-goals (later studies, not this one)

- Other model families and sizes (Llama, Qwen, Gemma 1B/4B). Keep the code model-agnostic so this is cheap later.
- Scenes shown as images (Gemma 3 multimodal).
- Chain-of-thought / thinking mode.
- Fine-tuning the model to read its map (P11).
- Packaging the generator as a public benchmark (keep it clean enough that this is easy later).

## Anti-goals (would make it a failure even if it runs)

- Changing scenes, rungs, layers or thresholds after seeing test data to reach the hoped-for headline.
- A rung-4 result the box cheat (below) could explain.
- A headline that holds only at one hand-picked layer. Layers are chosen from training-split probe fit, before test data.
- Moving or renaming Study 1 files (published links point at them).

## Constraints

- Compute: Google Colab Pro, as for P5–P10. Resumable notebooks; partial results written to Drive
  (`MyDrive/geomap/results/s2/...`) with `df_step` / `.done` markers; Slack `notify()` optional.
- Main model: Gemma 3 12B instruct, bfloat16 (fits a Colab A100 40 GB). 27B is optional and only if Colab offers an
  80 GB GPU; it is not required for any preregistered decision.
- Reuse the Study 1 library (`src/geomap/`: scoring, probes/GroupRidge, INLP erasure in `followups.py`, steering,
  patching, checkpoint) rather than re-implementing. New code goes in new modules.
- Colab TensorFlow fix and pinned versions from P5–P10 (`transformers==4.57.3`, `USE_TF=0` etc.).
- Every script supports `--smoke --out` and runs end to end on the tiny test model; unit tests need no GPU.

## The scenes

- Four objects: corners A, B, C and point P, with single-token names (as Study 1, `single_token_names`).
- Positions on an integer grid, north = +y, east = +x, described as a tree of relations, e.g.
  "B is 4 units east of A. C is 3 units north of B. P is 2 units east and 1 unit north of A."
  The corner tree is A → B → C (C hangs off B, so some corner facts already need a chain); P hangs off one corner,
  chosen at random.
- Triangles must be non-degenerate (area at least 4 square units — *my call*) and P must be at least 0.5 units from
  every edge line segment and not on a corner, so "inside" is never ambiguous.
- P's placement is stratified so the rung-4 test cannot be passed by a cheap rule:
  - **inside** the triangle — 50%;
  - **box-hard**: outside the triangle but inside its bounding box — at least 35%;
  - **outside the box** — the rest.
  Results on rung 4 are reported on the inside vs box-hard comparison, where the box rule is at chance by construction.
- Grid bound and offset ranges: *builder's call*, extending Study 1's generator; triangles should be large enough that
  box-hard points exist with margin (Study 1 used bound 6; a bound around 8 is likely needed).

## The ladder (same scenes, four rungs)

| Rung | Question (yes/no or two-choice) | What answers it |
|---|---|---|
| 1 Direct | "Which is farther north, P or Q?" where Q is P's parent (P's own sentence states it) | One stated sentence |
| 2 Chain | Same form, but Q is a corner that is not P's parent | Two or more sentences combined (Study 1's "CR" type) |
| 3 Extremes | "Is P farther north than all of A, B and C? Answer yes or no." (also east/south/west) | Many 1-D comparisons |
| 4 Inside | "Is P inside the triangle formed by A, B and C? Answer yes or no." | The 2-D shape |

- Rungs 1–2 reuse the existing `MORE_NORTH` / `MORE_EAST` question types. Rungs 3–4 are new question types with
  ground truth and a margin (distance from flipping), added alongside the existing ones in `questions.py`.
- Yes/no rungs are balanced (about 50% yes) and scored with balanced accuracy, because models often lean "yes".
- Answers are scored by candidate log-probability, as in Study 1.

## Baselines (cheap rules the model must beat)

- **Box rule**: "inside" iff P is inside the bounding box of A, B, C. At chance on box-hard items by construction.
- **Feature rule**: logistic regression on stated-sentence features only (P's offset from its parent, the stated
  offsets of the corners), fitted on training scenes — a stand-in for "a smart shortcut without a map".
- **Centroid rule** (reference only, not a gate): "inside" iff P's distance to the centroid, divided by the triangle's
  mean centroid-to-corner distance, is below a threshold fitted on training scenes. This rule needs true 2-D
  positions, so it is a crude *map* method: it shows how well a rough map would do, and a model that matches it may
  well be using its map. It must not stop the study.
The pilot's go rule compares the model with the best **map-free** baseline (box rule, feature rule) on held-out scenes.

## Step 1: pilot (answers only, one Colab session)

- Gemma 3 12B, about 400 new scenes, all four rungs; no probes, no interventions.
- Report per rung: accuracy / balanced accuracy with scene-bootstrap 95% CIs, and each baseline's score.
- **Go rule** (to preregister): on rung 4, inside vs box-hard, the model's balanced accuracy is at least **0.65**
  (point estimate), its CI lower bound is above 0.5, and its point estimate is above the best map-free baseline.
  The 0.65 floor is the headroom guard (as `min_headroom` in Study 1's P6): below it, erasure has too little room to
  show anything, and Headline B would be meaningless. Rungs 1–2 should look like Study 1 (sanity check, not a gate).
- **Weak pass** (above chance and above the map-free baselines, but under 0.65): stop and decide with the author.
- **If the pilot fails**, stop and decide with the author before any main-study work. Options then: bigger triangles
  and larger margins, or an easier rung-4 wording. Any change is a new preregistration, not a silent tweak.

## Step 2: main study (only if the pilot passes)

Each test below gets its own preregistration, committed before its data.

**Shared rules for all main-study tests**

- **One metric on every rung: balanced accuracy, chance 0.5.** Rungs 1–2 are two-choice, so balanced accuracy is
  taken over which of the two names is correct; rungs 3–4 over yes/no.
- **Comparing rungs uses the normalised drop**: excess drop ÷ (clean balanced accuracy − 0.5), i.e. the share of
  above-chance skill that erasing the map removes. This makes a near-ceiling rung 1 and a harder rung 4 comparable.
- **Layers.** Fit probes at every layer on the training split. The **primary window** is the contiguous run of layers
  around the best layer whose training-split CV R² is at least 90% of the best. The **primary layer** (for tests 3
  and 4) is the best layer. Both are fixed and committed before any test data. Single-layer erasures (best layer and
  the window's two ends) are secondary and reported, never used for the verdict, so no headline rests on a hand-picked
  layer.
- **Thresholds** carry over from Study 1 (P5/P6), with the rung comparison added:
  - Headline A ("uses its map when needed"): on rung 4, excess drop ≥ 0.10 with CI lower bound > 0, **and** the
    rung-4 minus rung-1 normalised drop has CI lower bound > 0.
  - Headline B ("never reads it, even here"): on rung 4, the CI upper bound of the excess drop is below 0.03.
  - Anything else: inconclusive, reported as such.
  - Erasure stops when the probe's CV R² falls below 0.1, as in Study 1.

**The tests**

1. **Map probes.** Fit scene-centred (x, y) probes for A, B, C, P at the object-list positions on a training split
   (GroupRidge, scene-grouped CV, as in P9). This fixes the primary window and layer.
2. **Erasure by rung (primary test; decides the headline).** INLP-erase the map at every token across the primary
   window, with the equal-rank random-basis control, as in P5/P6. Report excess and normalised drops per rung and apply
   the thresholds above.
3. **Map vs model by rung (secondary).** Answer every question geometrically from the decoded positions at the primary
   layer (as P9) and compare with the model's answers: accuracy of each, and agreement. Standalone verdict: agreement
   on rung 4 is higher than on rung 2 (CI on the difference).
4. **Move a corner (causal, secondary).** Move only leaf objects, C or P, so no descendants move with them. Overwrite
   the moved object's decoded (x, y) at its object-list position across the primary window (multi-layer write, as in
   P8), by a shift geometry says flips P from inside to outside, or the reverse; control shifts of equal size that
   geometry says do not flip it. Run the same moves on rung-2 questions as a comparison.
   - Standalone verdict: on rung 4, the flip-rate contrast (flip shifts minus control shifts) has CI lower bound > 0
     and is larger than the same contrast on rung 2.
   - Why the object list: Study 1 found the list is not read for chain questions (P4 slider 2% flips, P7). So "rung 4
     reads the list while rung 2 does not" is exactly the claim to test, and the rung-2 run guards against reading a
     generic null as meaningful.

**How the headline is worded (fixed now, before any data)**

- Test 2 alone decides Headline A, Headline B or inconclusive.
- The strong form, "the model reads its map to answer", is used only if test 2 gives Headline A **and** test 4's
  verdict holds. If only test 2 holds, the claim is "the map is needed for these answers" without saying how it is
  read.
- Tests 3 and 4 are always reported with their own verdicts, whatever they show.

## Outputs and where they go

- Library: new module(s) for triangle scenes and the new question types; existing modules extended, not moved.
- Scripts: `scripts/s2_pilot.py`, then one script per main-study test (`s2_*`), each with `--smoke --out`.
- Notebooks: `colab/s2_pilot.ipynb`, then one per main-study run, same pattern as P5–P10.
- Preregistrations: `docs/preregistration_s2_pilot.md`, `docs/preregistration_s2_<test>.md`.
- Results: `results/s2/...`, logged in `results/RESULTS.md` like every Study 1 run.
- Later: the front-door page grows a Study 2 entry (not part of this design).

## Left to the builder

Grid bound and offset ranges, exact name pools and templates, number of training scenes for probes and baselines,
bootstrap sizes, how box-hard points are sampled, notebook layout, and test structure — following Study 1's patterns.

## Calls I made (not the author's decisions)

- Corner tree A → B → C with P off a random corner; minimum triangle area 4; P at least 0.5 units from edges.
- The 50 / ≥35 / rest split for P's placement and judging rung 4 on inside vs box-hard.
- The baselines, with the centroid rule as a reference rather than a gate (it needs a map itself).
- The pilot go rule, including the 0.65 headroom floor.
- Balanced accuracy on every rung and the normalised drop for comparing rungs.
- The layer rule (window at ≥ 90% of the best training-split R²; best layer for tests 3–4).
- Study 1's thresholds (0.10 / 0.03 / R² 0.1) carried over.
- Test 4 writes at the object list across the window, moves only leaves, and includes a rung-2 comparison.
- Erasure decides the headline; the strong wording also needs the corner-move test.
- Gemma 3 12B only; 27B optional.
