"""Base models in plain text (docs/preregistration_base_plain.md): the preregistered verdict per run and overall."""

from __future__ import annotations

import math

import numpy as np

READS, OUTVOTED, UNINF = "reads the composed position", "the map is outvoted", "uninformative"


def rule_answer(d2) -> np.ndarray:
    """The one-sentence rule in the C frame: +1 (C) if C's own step points the asked way, else -1 (R)."""
    return np.where(np.asarray(d2) > 0, 1, -1)


def verdict_base_plain(dec: tuple, first: tuple, own: tuple, r2_d1: float, r2_d2: float, probe: tuple,
                       model: tuple) -> str:
    """dec, first, own: balanced accuracy (estimate, lo, hi) on the decisive, first-step-only and own-step-only
    questions; r2_d1, r2_d2: best held-out R² at the final position in the final window; probe, model: Wilson (share,
    lo, hi) on the held-out decisive questions."""
    vals = [*dec, *first, *own, r2_d1, r2_d2, *probe, *model]
    if any(isinstance(x, float) and math.isnan(x) for x in vals) or own[1] <= 0.60:
        return UNINF
    if dec[1] > 0.5 and first[1] > 0.5:
        return READS
    if dec[1] > 0.5:
        return "partial (decisive questions right, first step only not above chance)"
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
