"""Flow map (docs/preregistration_flow.md): on "which is farther east / north, C or R?" (chain R → M → C), where on
the way from the text to the answer is the first step (R → M) still present, and where does each step move the
answer?

d1 = M's step from R on the asked axis, d2 = C's own step from M, truth = sign(d1 + d2) (+1: C is farther in the asked
direction). Everything is in this "C frame", independent of which object the question names first."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from geomap.compose import _r2, fit_predict
from geomap.questions import QType

SITES = ("M_end", "C_end", "Q_C", "Q_R", "Q_after", "final")
PATCH_SITES = ("M_sentence", "C_sentence", "question", "final")
LOST, PRESENT, ANCHOR = 0.25, 0.5, 0.5


def axis_of(q) -> int:
    return 0 if q.qtype == QType.MORE_EAST else 1


def steps(s, mid: str, child: str, q) -> tuple[int, int, int]:
    ax = axis_of(q)
    d1 = s.coords[mid][ax] - s.coords[s.root][ax]
    d2 = s.coords[child][ax] - s.coords[mid][ax]
    return int(d1), int(d2), int(np.sign(d1 + d2))


def flip_step(s, who: str, ax: int):
    """The scene with `who`'s own stated step negated on axis `ax` (same words otherwise); objects placed relative
    to it move with it. Raises ValueError if that step is 0 or two objects would share a position."""
    rel = next(r for r in s.relations if r.subj == who)
    if (rel.dx, rel.dy)[ax] == 0:
        raise ValueError("zero step on that axis")
    new = replace(rel, dx=-rel.dx if ax == 0 else rel.dx, dy=-rel.dy if ax == 1 else rel.dy)
    rels = tuple(new if r.subj == who else r for r in s.relations)
    by = {r.subj: r for r in rels}
    coords = {s.root: s.coords[s.root]}
    while len(coords) < len(s.names):
        for n in s.names:
            if n not in coords and s.parent[n] in coords:
                px, py = coords[s.parent[n]]
                coords[n] = (px + by[n].dx, py + by[n].dy)
    if len(set(coords.values())) < len(coords):
        raise ValueError("collision")
    return replace(s, coords=coords, relations=rels)


def site_positions(spec, mid: str, child: str, q, root: str) -> dict[str, int]:
    """Token positions of SITES: the last token of M's and of C's sentence, C's and R's names in the question, the
    token right after the second-named object, and the final position (where the answer is read out)."""
    second = max(spec.question_pos[child], spec.question_pos[root])
    return {"M_end": spec.relation_spans[mid][1] - 1, "C_end": spec.relation_spans[child][1] - 1,
            "Q_C": spec.question_pos[child], "Q_R": spec.question_pos[root], "Q_after": second + 1,
            "final": len(spec.input_ids) - 1}


def patch_positions(spec, mid: str, child: str) -> dict[str, list[int]]:
    """Positions of PATCH_SITES: M's sentence, C's sentence, the question and chat header up to (not including)
    the final position, and the final position."""
    T = len(spec.input_ids)
    return {"M_sentence": list(range(*spec.relation_spans[mid])), "C_sentence": list(range(*spec.relation_spans[child])),
            "question": list(range(spec.prefix_len, T - 1)), "final": [T - 1]}


def axis_r2(X, y, ax, groups, tr) -> float:
    """Held-out R² of a ridge probe fitted separately on east and on north questions, averaged over the two."""
    X, y, ax, groups, tr = map(np.asarray, (X, y, ax, groups, tr))
    out = []
    for a in (0, 1):
        r = ax == a
        a_tr, a_te = r & tr, r & ~tr
        p = fit_predict(X[a_tr].astype(float), y[a_tr].astype(float), groups[a_tr], X[a_te].astype(float))
        out.append(_r2(y[a_te].astype(float), p))
    return float(np.mean(out))


def axis_predict(X, y, ax, groups, tr) -> np.ndarray:
    """Held-out predictions (NaN on training rows) of per-axis ridge probes."""
    X, y, ax, groups, tr = map(np.asarray, (X, y, ax, groups, tr))
    p = np.full(len(y), np.nan)
    for a in (0, 1):
        r = ax == a
        a_tr, a_te = r & tr, r & ~tr
        p[a_te] = fit_predict(X[a_tr].astype(float), y[a_tr].astype(float), groups[a_tr], X[a_te].astype(float))
    return p


def layer_grid(n_layers: int) -> tuple[list[int], list[int], dict]:
    """Probed layers (every second), copied layers (every fourth up to n − 4, plus n − 2) and the rule's layer windows
    as the same share of the depth as on the 48-layer 12B model, rounded to the probed (even) layers."""
    ev = lambda x: 2 * round(x / 2)
    n = n_layers
    win = {"C_end": (ev(n / 4), ev(2 * n / 3)), "question": (ev(n / 3), ev(5 * n / 6)), "final": (ev(7 * n / 12), n)}
    return list(range(0, n + 1, 2)), list(range(0, n - 3, 4)) + [n - 2], win


def verdict_flow(p1: float, p2: float) -> str:
    """p1, p2: best held-out R² of the first step (d1) and of C's own step (d2) at the final position, over the final
    window of layer_grid (layers 28-48 on 12B)."""
    if p2 < ANCHOR:
        return "uninformative (C's own step is not readable at the answer position either)"
    if p1 < LOST:
        return "lost before the answer position"
    if p1 >= PRESENT:
        return "reaches the answer position (the read-out ignores it)"
    return "partial"


def where_lost(best: dict) -> str | None:
    """First hop (in the order given) where the first step's best R² is below LOST."""
    return next((k for k, v in best.items() if v < LOST), None)
