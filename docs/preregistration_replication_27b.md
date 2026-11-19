# Preregistration: replication on Gemma 3 27B of the flow map, Stage 2 and Stage 2b

*Committed 2026-11-19, before any 27B data of these tests exists. Gemma 3 27B instruct, bfloat16, eager attention, the
whole model on one GPU of at least 70 GB (Colab H100, G4 or A100 80 GB).*

## Why

On Gemma 3 12B the three tests give one account of "which is farther east / north, C or R?" (R → M → C):

- **Flow map** (`docs/preregistration_flow.md`): M's step reaches the final position as readably as C's own step,
  and a truth probe there beats the model; the answer is written late.
- **Stage 2** (`docs/preregistration_components.md`): exact attribution finds a few late components that write the
  shortcut ("C iff C's own sentence points the asked way").
- **Stage 2b** (`docs/preregistration_components2.md`): removing the top 20 while keeping the names takes the
  shortcut off but leaves no correct read-out (*no correct path underneath the shortcut*); given the twin's values,
  those 20 components set the answer.

This run asks whether the same account holds in the 27B model of the same family.

## What is run, unchanged

The three scripts as committed, with the same seeds (flow 1601, Stage 2 discovery 1701 / test 1801, Stage 2b 1901),
question counts, cells, measurements and decision rules, in that order. The two models share a tokenizer, so the
questions are the same as in the 12B runs.

## What is adapted to the deeper model (62 layers instead of 48)

- **Flow map layers.** Probes at every second layer (0, 2, …, 62); copies at every fourth layer up to the last but
  four, plus the last but two (0, 4, …, 56, 60). The rule's layer windows keep the same share of the depth, rounded
  to the probed layers: C's sentence end layers 16–42 (12B: 12–32), question tokens 20–52 (16–40), final position
  36–62 (28–48). For 48 layers the formula gives the 12B windows exactly.
- **Stage 2b components.** Chosen by Stage 2's rule on the 27B discovery set (top 20 by own_yes coefficient,
  first-step writers top 10 by d1 coefficient, read from `results/components/gemma3-27b_components_components.csv`),
  before the Stage 2b questions are built. Matched controls drawn as in Stage 2b.
- Batch size 8 (memory only).

## Replication criteria (stated in advance)

The 12B account **replicates** on 27B if all three hold:

1. **Behaviour.** The model follows the shortcut: accuracy on *conflict, M larger* below 0.5 in the flow run's clean
   answers. (If 27B does not follow it, the account has nothing to explain there; reported as *the shortcut does not
   replicate*, and 2 and 3 are reported but not judged.)
2. **Flow map** verdict *reaches the answer position (the read-out ignores it)*.
3. **Stage 2b** verdict *no correct path underneath the shortcut* (which requires an effective removal), and, given
   the twin's values, the top 20 set the answer: the answer follows the twin's shortcut in at least 0.80 of
   *own step only* questions (12B: 0.98).

Otherwise it is a **partial replication**, reported item by item. Stage 2's own verdict is reported but not part of
the criteria: its test (mean ablation) proved unsuitable for name-copying heads, as 12B showed.

## Reported, not part of the criteria

Everything the three scripts report, next to the 12B numbers: where the answer is written (logit lens), the layers of
the top writers as a share of depth, how much of the removed shortcut comes back and from which components (writers
vs brakes), and the truth probe at the final position against the model.

## Limits

One more model of the same family and training recipe; a replication here shows the account is not specific to one
model size, not that it holds across families.
