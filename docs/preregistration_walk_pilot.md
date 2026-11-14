# Preregistration: walk pilot, can Gemma 3 tell where a long walk has brought it back to?

*Committed 2026-11-15, before any walk data exists. Gemma 3 12B instruct (Colab, one A100). Answers only: no probes,
no interventions. Background: `reports/Where LLMs use internal maps.md`, recommendation 1.*

## Why

Study 1 found that Gemma 3 builds a decodable map of a layout described in text but answers from the stated
sentences: on C–R axis questions it follows C's own sentence, even below chance (0.34) where that sentence misleads.
The literature review found that every clear case of a model *using* an internal map has one thing in common: no
input span states the answer, so the model has to integrate many steps. A walk that comes back to an earlier place
by a loop has that property. No sentence says "you are back at the bell", and no single move tells you where you are.

The main study (probes for the running position, erasure against a matched random control, patching) only means
something if the model can answer these questions well above every map-free shortcut. This pilot checks that and
picks the walk and loop lengths for the main study.

## Walks

- **Grid and moves:** an unbounded grid of city blocks; each move is one block north, south, east or west (unit
  moves only, so "7 blocks north" arithmetic cannot drive failures). A move never undoes the move before it (no
  "north" right after "south"), except in the 2-move loop below, where undoing is the point.
- **Landmarks:** 4 objects per walk, single-token common nouns (clock, bell, …; filtered for the tokenizer), at 4
  different steps between the start (step 0) and the last step before the end. Each landmark's block is visited only
  at its own step, so the text never passes a landmark without naming it. Every landmark sits an **even** number of
  moves before the end: on a grid you can only be back where you were after an even number of moves, so this way
  counting moves modulo 2 rules no object out.
- **Return:** after its last move the walk is back at one landmark's block, the *target*. The **loop length** ℓ is
  the number of moves since the target was seen. Between the target and the end the walk does not touch the
  target's block.
- **Target rank:** where the target sits in the order of the 4 landmarks is drawn uniformly, before the walk, from
  the ranks the cell allows; walks are resampled until one fits. For ℓ ≥ 4 at least one other landmark is seen
  inside the loop, so "the last object seen" is always wrong. For ℓ = 2 the loop is "one block out, one block back"
  with no landmark inside: the shortcut-sufficient control (reverse the last move, or name the last object seen).
- **Design:** walk length N ∈ {14, 20, 26} moves × loop length ℓ ∈ {2, 6, 8, 12}: 12 cells × 200 walks = 2,400
  walks (seed 1401). ℓ = 4 is left out: with landmarks at even distances its only inside slot is 2 moves after the
  target, so the target would always be the second-to-last object seen.
- **Expected baseline levels** (computed from the walks, no model): the best fixed-rank rule is about 0.33 for
  ℓ ∈ {8, 12} at N ≥ 20, and about 0.5 for ℓ = 6 and for N = 14, ℓ = 12, where only two ranks are possible.

## Prompts (same walks, two formats)

- **user** (the main format): the walk in the user message, one sentence per move ("You walk one block north."),
  landmarks as "There is a bell here.", then "You are now back at one of the objects you saw. Which object is it?
  Answer with one word."
- **prefill** (secondary): the user asks the model to describe such a walk; the model's turn is prefilled with the
  same walk in the first person, ending "I am now back at the", and the next word is scored. Lepori et al. (ACL 2026)
  found in-context structure usable in Gemma 3 only when it sat in the model's own turn.
- **Answer:** the candidate (one of the walk's 4 objects) with the highest log-probability, scored as in Study 1.
  Chance is 0.25.

## Map-free baselines (computed from the walk text, no model)

- **Recency:** the last object seen. Always right for ℓ = 2, always wrong for ℓ ≥ 4 by construction.
- **Primacy:** the first object seen.
- **Fixed rank:** always the k-th object seen, k = 1…4.
- **Parity guess:** a uniform guess among the objects an even number of moves ago. It is 0.25 by construction and
  is kept as a check of the generator.

In each cell the **best map-free** baseline is the one with the highest accuracy on the test walks themselves, which
favours the baselines (conservative).

## Metric

- Per cell and format: **accuracy** with a 95% walk-bootstrap CI (1,000 resamples), and the **gap** = model − best
  map-free, as a paired bootstrap over walks (same resampled walks for both).

## Decision rule

A cell **passes** when accuracy ≥ 0.60 and the gap's CI lower bound > 0.

- **Go:** at least one cell with ℓ ≥ 4 passes, in either format. The main study uses the **user** format if any
  ℓ ≥ 4 cell passes there, otherwise **prefill**; within that format, the passing ℓ ≥ 4 cell with the largest ℓ
  (then the largest N), together with ℓ = 2 at the same N as the shortcut contrast.
- **Weak pass:** no cell passes, but at least one ℓ ≥ 4 cell has a gap CI lower bound > 0.
- **Fail:** anything else.

The 0.60 floor (chance 0.25) leaves room for an erasure effect in the main study.

## What each outcome leads to

- **Go:** write the main-study preregistration for the chosen cells: probes for the running (x, y) at each move and
  at the question, closed-loop erasure against the variance-matched random control, the contrast between ℓ ≥ 4 and
  ℓ = 2, and patching the position from a walk that ends at another object.
- **Weak pass or fail:** stop and decide with the author (options: Gemma 3 27B, shorter walks, or recommendation 2,
  a small model trained from scratch). Any change gets a new preregistration. Nothing is tuned silently.

## Reported, not part of the rule

- Accuracy against ℓ and N in both formats (the ℓ = 2 cells should be near ceiling, a sanity check).
- Every baseline's accuracy per cell.
- The probability the model puts on the 4 allowed answers.

## Revision (2026-11-15, before any Gemma data)

A smoke run of the script with the tiny random test model (no Gemma, no real data) showed two cheap cues in the first
version: landmarks an odd number of moves before the end can never be the answer, so a parity guess reached about
0.45; and drawing the target rank from what each walk allowed made "the second-to-last object" right 0.6–0.7 of the
time. Changes: landmarks only at even distances from the end, the rank drawn first and walks resampled to fit it,
and ℓ = 4 replaced by ℓ = 6 (see Design). No Gemma data existed when this was changed.
