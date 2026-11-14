"""C–R axis questions split by how the two steps combine on the asked axis (docs/preregistration_axis_bulk.md)."""

from __future__ import annotations

import numpy as np

from geomap.composition import _chain
from geomap.prompts import NAME_POOLS, single_token_names
from geomap.questions import QType, all_questions
from geomap.scenes import generate_scene

CELLS = ("own step only", "agree", "first step only", "conflict, C larger", "conflict, M larger")


def axis_cell(s, mid: str, child: str, q) -> str | None:
    """Cell of a C–R axis question (d1 = R→M step, d2 = M→C step on the asked axis); None for other questions."""
    if q.qtype not in (QType.MORE_EAST, QType.MORE_NORTH) or set(q.args) != {s.root, child}:
        return None
    ax = 0 if q.qtype == QType.MORE_EAST else 1
    d1 = s.coords[mid][ax] - s.coords[s.root][ax]
    d2 = s.coords[child][ax] - s.coords[mid][ax]
    if d1 + d2 == 0:
        return None
    if d1 == 0:
        return "own step only"
    if d2 == 0:
        return "first step only"
    if d1 * d2 > 0:
        return "agree"
    return "conflict, C larger" if abs(d2) > abs(d1) else "conflict, M larger"


def axis_bulk_items(m, sizes: dict, seed: int, min_margin: float = 1.0, min_angle: float = 10.0):
    """3-object chains R → M → C (as followups.chain_items), keeping only C–R axis questions, until each cell holds
    sizes[cell] questions. Returns [(scene, mid, child, [(question, cell)])]."""
    rng = np.random.default_rng(seed)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    left = dict(sizes)
    out, k = [], 0
    while any(v > 0 for v in left.values()):
        s = generate_scene(rng, pools[k % 2], 3, p_shuffle=0.0)
        k += 1
        mc = _chain(s)
        if mc is None:
            continue
        mid, child = mc
        keep = []
        for q in all_questions(s, min_margin, min_angle):
            c = axis_cell(s, mid, child, q)
            if c is not None and left.get(c, 0) > 0:
                keep.append((q, c))
                left[c] -= 1
        if keep:
            out.append((s, mid, child, keep))
    return out


def verdict_behaviour(lo: float, hi: float) -> str:
    """Rule 1: clean accuracy CI in 'conflict, M larger'."""
    if hi < 0.5:
        return "the model follows C's own step"
    if lo > 0.5:
        return "the model composes the steps"
    return "undecided"


def verdict_erasure(excess: float, lo: float, hi: float, min_effect: float = 0.03) -> str:
    """Rule 2: excess drop (fair − map-erased accuracy) in 'conflict, M larger'."""
    if lo > 0 and excess >= min_effect:
        return "the answers read C's composed position from the map"
    if hi < 0:
        return "the erased directions support the own-step shortcut, not the composed position"
    return "undecided"
