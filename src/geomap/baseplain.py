"""Base models in plain text (docs/preregistration_base_plain.md) and with worked examples
(docs/preregistration_base_fewshot.md): the preregistered verdict per run and overall, and the examples."""

from __future__ import annotations

import math

import numpy as np

READS, OUTVOTED, UNINF = "reads the composed position", "the map is outvoted", "uninformative"


def rule_answer(d2) -> np.ndarray:
    """The one-sentence rule in the C frame: +1 (C) if C's own step points the asked way, else -1 (R)."""
    return np.where(np.asarray(d2) > 0, 1, -1)


def verdict_base_plain(dec: tuple, first: tuple, own: tuple, r2_d1: float, r2_d2: float, probe: tuple,
                       model: tuple, conflict_c: tuple | None = None) -> str:
    """dec, first, own: balanced accuracy (estimate, lo, hi) on the decisive, first-step-only and own-step-only
    questions; r2_d1, r2_d2: best held-out R² at the final position in the final window; probe, model: Wilson (share,
    lo, hi) on the held-out decisive questions. conflict_c (few-shot amendment): balanced accuracy on conflict C
    larger, which "reads the composed position" then also needs above chance."""
    vals = [*dec, *first, *own, r2_d1, r2_d2, *probe, *model, *(conflict_c or ())]
    if any(isinstance(x, float) and math.isnan(x) for x in vals) or own[1] <= 0.60:
        return UNINF
    cc_ok = conflict_c is None or conflict_c[1] > 0.5
    if dec[1] > 0.5 and first[1] > 0.5 and cc_ok:
        return READS
    if dec[1] > 0.5:
        miss = [k for k, ok in (("first step only", first[1] > 0.5), ("conflict C larger", cc_ok)) if not ok]
        return f"partial (decisive questions right, {' and '.join(miss)} not above chance)"
    if dec[2] >= 0.5:
        return "partial (decisive questions at chance)"
    failed = []
    if r2_d1 < 0.5:
        failed.append("M's step not readable at the answer")
    if r2_d2 < 0.5:
        failed.append("C's own step not readable at the answer")
    if not probe[1] > model[2]:
        failed.append("probe does not beat the model")
    return OUTVOTED if not failed else f"partial ({'; '.join(failed)})"


def overall_base_plain(verdicts: dict[str, str]) -> str:
    """verdicts: base model -> its verdict."""
    inf = {k: v for k, v in verdicts.items() if not v.startswith(UNINF)}
    if not inf:
        return "uninformative (no informative base model)"
    if all(v == OUTVOTED for v in inf.values()):
        return "the failure comes from pretraining"
    if all(v == READS for v in inf.values()):
        return "instruction tuning or the format brings the failure"
    return "mixed: " + "; ".join(f"{k}: {v}" for k, v in inf.items())


# ---- worked examples (docs/preregistration_base_fewshot.md) ----

SHOT_SEED = 7
DEC, FIRST, OWN, AGREE, CC = "conflict, M larger", "first step only", "own step only", "agree", "conflict, C larger"
SLOTS = {"neutral": [(OWN, "C"), (OWN, "R"), (AGREE, "C"), (AGREE, "R"), (CC, "C"), (CC, "R")],
         "informative": [(OWN, "C"), (AGREE, "R"), (FIRST, "C"), (CC, "R"), (DEC, "C"), (DEC, "R")]}


def example_text(s, q, answer: str) -> str:
    """One worked example: the plain item exactly as the zero-shot prompt writes it, then " <answer>"."""
    from geomap.scenes import relation_text
    return (" ".join(relation_text(r) for r in s.relations) + "\n\nThe objects are: " + ", ".join(s.names) + "."
            + f"\n\nQuestion: {q.text}\nAnswer: {answer}")


def shot_bank(m, per_cell: int = 200, seed: int = SHOT_SEED) -> list[dict]:
    """The flow set's generator with its own seed: every C–R axis question with its kind, correct answer and axis."""
    from geomap.axis import CELLS, axis_bulk_items
    from geomap.flow import axis_of, steps
    out = []
    for s, mid, child, qq in axis_bulk_items(m, {c: per_cell for c in CELLS}, seed=seed):
        for q, cell in qq:
            truth = steps(s, mid, child, q)[2]
            out.append(dict(scene=s, mid=mid, child=child, q=q, cell=cell, axis=axis_of(q),
                            ans="C" if truth == 1 else "R", answer=child if truth == 1 else s.root))
    return out


def pick_shots(bank: list[dict], test_names: set, arm: str, rng) -> list[dict]:
    """Slot i takes the first bank question of its kind, answer and axis (i % 2) whose scene shares no name with the
    test scene or the examples already chosen; the six are then shuffled with rng."""
    used, chosen = set(test_names), []
    for i, (cell, ans) in enumerate(SLOTS[arm]):
        e = next((e for e in bank if e["cell"] == cell and e["ans"] == ans and e["axis"] == i % 2
                  and not used & set(e["scene"].names)), None)
        if e is None:
            raise ValueError(f"no bank question for slot {i} ({cell}, {ans})")
        chosen.append({**e, "slot": i})
        used |= set(e["scene"].names)
    return [chosen[k] for k in rng.permutation(len(chosen))]


def shots_text(chosen: list[dict]) -> str:
    return "".join(example_text(e["scene"], e["q"], e["answer"]) + "\n\n" for e in chosen)


def overall_fewshot(verdicts: dict[str, str], arm: str) -> str:
    """verdicts: base model -> its verdict in this arm."""
    inf = {k: v for k, v in verdicts.items() if not v.startswith(UNINF)}
    if not inf:
        return "uninformative (no informative base model)"
    reads, out = (("the zero-shot format hid map use", "the zero-shot format did not hide map use") if arm == "neutral"
                  else ("examples that need M's step make the answer read the composed position",
                        "the map stays outvoted even with such examples"))
    if all(v == OUTVOTED for v in inf.values()):
        return out
    if all(v == READS for v in inf.values()):
        return reads
    return "mixed: " + "; ".join(f"{k}: {v}" for k, v in inf.items())
