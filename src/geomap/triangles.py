"""Study 2: a triangle A, B, C plus a point P, described through relations, and the four-rung question ladder.

Convention as in Study 1: north = +y, east = +x; A sits at the origin. Rungs 3-4 use their own question types
(TriQType) so Study 1's QType, all_questions and battery stay unchanged."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from geomap.questions import QType, Question, answer_for, make_question
from geomap.scenes import TEMPLATES, Relation, Scene

Point = tuple[int, int]
STRATA = ("inside", "box_hard", "outside_box")


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def in_triangle(p, a, b, c) -> bool:
    """Strict interior: a point on an edge is not inside."""
    d = (_cross(a, b, p), _cross(b, c, p), _cross(c, a, p))
    return all(x > 0 for x in d) or all(x < 0 for x in d)


def _seg_dist(p, a, b) -> float:
    ax, ay, bx, by, px, py = *a, *b, *p
    vx, vy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def edge_distance(p, a, b, c) -> float:
    return min(_seg_dist(p, a, b), _seg_dist(p, b, c), _seg_dist(p, c, a))


def in_box(p, a, b, c) -> bool:
    xs, ys = (a[0], b[0], c[0]), (a[1], b[1], c[1])
    return min(xs) <= p[0] <= max(xs) and min(ys) <= p[1] <= max(ys)


def stratum(p, a, b, c) -> str:
    if in_triangle(p, a, b, c):
        return "inside"
    return "box_hard" if in_box(p, a, b, c) else "outside_box"


@dataclass(frozen=True)
class TriScene:
    scene: Scene
    corners: tuple[str, str, str]  # A, B, C
    point: str  # P
    stratum: str


def generate_tri_scene(rng: np.random.Generator, names_pool: Sequence[str], bound: int = 8, min_area: float = 4.0,
                       min_edge_dist: float = 0.5, p_strata=(0.5, 0.35, 0.15)) -> TriScene:
    seed = int(rng.integers(2**31))
    a, b, c, p = (str(x) for x in rng.choice(list(names_pool), size=4, replace=False))
    want = STRATA[int(rng.choice(3, p=np.asarray(p_strata) / np.sum(p_strata)))]
    grid = [(x, y) for x in range(-bound, bound + 1) for y in range(-bound, bound + 1)]
    A = (0, 0)
    # Redraw the triangle until it is large enough and admits a P position in EVERY stratum, so the triangle's
    # shape carries no information about the stratum drawn above.
    while True:
        B = tuple(int(v) for v in rng.integers(-bound, bound + 1, 2))
        C = tuple(int(v) for v in rng.integers(-bound, bound + 1, 2))
        if len({A, B, C}) < 3 or abs(_cross(A, B, C)) / 2 < min_area:
            continue
        by: dict[str, list] = {k: [] for k in STRATA}
        for q in grid:
            if q not in (A, B, C) and edge_distance(q, A, B, C) >= min_edge_dist:
                by[stratum(q, A, B, C)].append(q)
        if all(by.values()):
            P = by[want][int(rng.integers(len(by[want])))]
            break
    par = (a, b, c)[int(rng.integers(3))]
    coords = {a: A, b: B, c: C, p: P}
    parent = {a: None, b: a, c: b, p: par}

    def rel(s, o):
        return Relation(s, o, coords[s][0] - coords[o][0], coords[s][1] - coords[o][1],
                        int(rng.integers(len(TEMPLATES))))

    rels = (rel(b, a), rel(c, b), rel(p, par))
    hop = {a: 0, b: 1, c: 2}
    hop[p] = hop[par] + 1
    names = tuple((a, b, c, p)[i] for i in rng.permutation(4))
    scene = Scene(names=names, coords=coords, relations=rels, parent=parent, hop=hop, seed=seed)
    return TriScene(scene=scene, corners=(a, b, c), point=p, stratum=want)


class TriQType(str, Enum):
    EXTREME = "extreme"
    INSIDE = "inside"


@dataclass(frozen=True)
class LadderQ:
    question: Question
    rung: int  # 1 direct, 2 chain, 3 extremes, 4 inside
    truth: str
    positive: str  # the answer counted as the positive class for balanced accuracy


_DIRS = {"north": (1, 1), "east": (0, 1), "south": (1, -1), "west": (0, -1)}  # axis, sign


def ladder(ts: TriScene, rng: np.random.Generator) -> list[LadderQ]:
    s, p, (a, b, c) = ts.scene, ts.point, ts.corners
    co, par = s.coords, s.parent[ts.point]
    out = []
    for corner in (a, b, c):  # rungs 1-2: P against each corner on each axis
        rung = 1 if corner == par else 2
        for qt, ax in ((QType.MORE_NORTH, 1), (QType.MORE_EAST, 0)):
            if abs(co[p][ax] - co[corner][ax]) < 1:
                continue
            args = (p, corner) if rng.random() < 0.5 else (corner, p)
            q = make_question(qt, args, s.names)
            out.append(LadderQ(q, rung, answer_for(q, co), args[0]))
    for d, (ax, sign) in _DIRS.items():  # rung 3: beyond all three corners?
        extreme = max(sign * co[k][ax] for k in (a, b, c))
        if abs(sign * co[p][ax] - extreme) < 1:
            continue
        q = Question(TriQType.EXTREME, (p, a, b, c),
                     f"Is {p} farther {d} than all of {a}, {b} and {c}? Answer yes or no.", ("yes", "no"))
        out.append(LadderQ(q, 3, "yes" if sign * co[p][ax] > extreme else "no", "yes"))
    q = Question(TriQType.INSIDE, (p, a, b, c),
                 f"Is {p} inside the triangle formed by {a}, {b} and {c}? Answer yes or no.", ("yes", "no"))
    out.append(LadderQ(q, 4, "yes" if in_triangle(co[p], co[a], co[b], co[c]) else "no", "yes"))
    return out


def ladder_answer(q: Question, coords) -> str:
    """The correct answer to a ladder question for any coordinates (true, or decoded from the model's map)."""
    if q.qtype == TriQType.INSIDE:
        p, a, b, c = q.args
        return "yes" if in_triangle(coords[p], coords[a], coords[b], coords[c]) else "no"
    if q.qtype == TriQType.EXTREME:
        ax, sign = _DIRS[q.text.split()[3]]
        p, *cs = q.args
        return "yes" if all(sign * (coords[p][ax] - coords[k][ax]) > 0 for k in cs) else "no"
    return answer_for(q, coords)
