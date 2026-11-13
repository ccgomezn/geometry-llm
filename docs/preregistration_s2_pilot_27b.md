# Preregistration addendum: Study 2 pilot on Gemma 3 27B

*Committed 2026-11-13, before any 27B data of this pilot exists.*

## Why

On Gemma 3 12B the pilot failed (balanced accuracy 0.51 on the hard "inside" questions, below the size rule's 0.63),
and the map decoded from its activations was at chance too (`docs/preregistration_s2_map.md`). This run asks whether a
larger model of the same family can tell inside from outside at all, before any map work on it.

## What is the same

Everything in `docs/preregistration_s2_pilot.md`, including its revision: the 400 test scenes (seed 702), the 2,000
baseline-training scenes (seed 701), the ladder, the metric, the baselines and the decision rule (go / weak pass /
fail). The script is unchanged: `scripts/s2_pilot.py --model gemma3-27b`.

## What differs

- Model: `google/gemma-3-27b-it`, bfloat16.
- Hardware: Colab. On a GPU with less than the model's 54 GB, part of the weights stay in CPU memory
  (`device_map="auto"`). The computation is the same; only speed changes.
- Batch size 4 instead of 16 (memory). Answers do not depend on it beyond floating-point noise.

## What each outcome leads to

- **Go:** rerun the map test (`scripts/s2_map.py`) on 27B under its own preregistration, then the main study.
- **Weak pass or fail:** stop and decide with the author.
