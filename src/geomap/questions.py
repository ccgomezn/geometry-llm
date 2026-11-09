"""Question battery over a scene, ground-truth answers for any coordinates, and counterfactual pairs."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from geomap.scenes import Scene, leaves, move_leaf

COMPASS = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
# compass name -> centre angle in degrees (math convention: east = 0, north = 90)
_CENTRE = {"east": 0, "northeast": 45, "north": 90, "northwest": 135, "west": 180,
           "southwest": 225, "south": 270, "southeast": 315}


class QType(str, Enum):
    FARTHEST = "farthest"
    CLOSEST = "closest"
    DIRECTION = "direction"
    MORE_NORTH = "more_north"
    MORE_EAST = "more_east"
    CLOSER_THAN = "closer_than"


_TEXT = {
    QType.FARTHEST: "Which object is farthest from {0}?",
    QType.CLOSEST: "Which object is closest to {0}?",
    QType.DIRECTION: "In which compass direction is {1} from {0}?",
    QType.MORE_NORTH: "Which is farther north, {0} or {1}?",
    QType.MORE_EAST: "Which is farther east, {0} or {1}?",
    QType.CLOSER_THAN: "Is {1} closer to {0} than {2} is? Answer yes or no.",
}


@dataclass(frozen=True)
class Question:
    qtype: QType
    args: tuple[str, ...]
    text: str
    candidates: tuple[str, ...]


Coords = Mapping[str, tuple[float, float]]


def _dist(c: Coords, a: str, b: str) -> float:
    return math.dist(c[a], c[b])


def _angle(c: Coords, frm: str, to: str) -> float:
    (x0, y0), (x1, y1) = c[frm], c[to]
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360


def _others(c: Coords, x: str) -> list[str]:
    return [n for n in c if n != x]


def answer_for(q: Question, coords: Coords) -> str:
    t, a = q.qtype, q.args
    if t in (QType.FARTHEST, QType.CLOSEST):
        others = _others(coords, a[0])
        key = lambda n: _dist(coords, a[0], n)
        return max(others, key=key) if t == QType.FARTHEST else min(others, key=key)
    if t == QType.DIRECTION:
        ang = _angle(coords, a[0], a[1])
        return min(COMPASS, key=lambda d: abs((ang - _CENTRE[d] + 180) % 360 - 180))
    if t == QType.MORE_NORTH:
        return a[0] if coords[a[0]][1] > coords[a[1]][1] else a[1]
    if t == QType.MORE_EAST:
        return a[0] if coords[a[0]][0] > coords[a[1]][0] else a[1]
    if t == QType.CLOSER_THAN:
        return "yes" if _dist(coords, a[0], a[1]) < _dist(coords, a[0], a[2]) else "no"
    raise ValueError(t)


def margin(q: Question, coords: Coords) -> float:
    """How far the answer is from flipping: units for distance-type questions, degrees for DIRECTION."""
    t, a = q.qtype, q.args
    if t in (QType.FARTHEST, QType.CLOSEST):
        d = sorted((_dist(coords, a[0], n) for n in _others(coords, a[0])), reverse=(t == QType.FARTHEST))
        return abs(d[0] - d[1]) if len(d) > 1 else math.inf
    if t == QType.DIRECTION:
        ang = _angle(coords, a[0], a[1])
        off = abs((ang - _CENTRE[answer_for(q, coords)] + 180) % 360 - 180)
        return 22.5 - off
    if t == QType.MORE_NORTH:
        return abs(coords[a[0]][1] - coords[a[1]][1])
    if t == QType.MORE_EAST:
        return abs(coords[a[0]][0] - coords[a[1]][0])
    if t == QType.CLOSER_THAN:
        return abs(_dist(coords, a[0], a[1]) - _dist(coords, a[0], a[2]))
    raise ValueError(t)


def make_question(qtype: QType, args: tuple[str, ...], names: tuple[str, ...]) -> Question:
    if qtype in (QType.FARTHEST, QType.CLOSEST):
        cands = tuple(n for n in names if n != args[0])
    elif qtype == QType.DIRECTION:
        cands = COMPASS
    elif qtype in (QType.MORE_NORTH, QType.MORE_EAST):
        cands = (args[0], args[1])
    else:
        cands = ("yes", "no")
    return Question(qtype, args, _TEXT[qtype].format(*args), cands)


def _all_args(qtype: QType, names: tuple[str, ...]):
    if qtype in (QType.FARTHEST, QType.CLOSEST):
        return [(x,) for x in names] if len(names) >= 3 else []
    if qtype == QType.DIRECTION:
        return list(itertools.permutations(names, 2))
    if qtype in (QType.MORE_NORTH, QType.MORE_EAST):
        return list(itertools.permutations(names, 2))
    return list(itertools.permutations(names, 3))


def passes_margin(q: Question, coords: Coords, min_margin: float = 1.0, min_angle: float = 10.0) -> bool:
    return margin(q, coords) >= (min_angle if q.qtype == QType.DIRECTION else min_margin)


def all_questions(scene: Scene, min_margin: float = 1.0, min_angle: float = 10.0) -> list[Question]:
    out = []
    for t in QType:
        for args in _all_args(t, scene.names):
            q = make_question(t, args, scene.names)
            if passes_margin(q, scene.coords, min_margin, min_angle):
                out.append(q)
    return out


def battery(scene: Scene, rng: np.random.Generator, per_type: int = 2,
            min_margin: float = 1.0, min_angle: float = 10.0) -> list[Question]:
    pool = all_questions(scene, min_margin, min_angle)
    out = []
    for t in QType:
        qs = [q for q in pool if q.qtype == t]
        if qs:
            idx = rng.permutation(len(qs))[:per_type]
            out.extend(qs[i] for i in sorted(idx))
    return out


def _resample_component(rng: np.random.Generator, k: int) -> int:
    mag = 1 if abs(k) == 1 else int(rng.integers(2, 6))
    return mag * (1 if rng.random() < 0.5 else -1)


def make_pair(scene: Scene, rng: np.random.Generator, max_tries: int = 50) -> tuple[Scene, Scene, str]:
    """Return (base, source, leaf): source moves one leaf, keeping its template and unit/units pattern."""
    cands = leaves(scene)
    for _ in range(max_tries):
        leaf = cands[int(rng.integers(len(cands)))]
        rel = next(r for r in scene.relations if r.subj == leaf)
        if rel.dx and rel.dy:
            dx, dy = _resample_component(rng, rel.dx), _resample_component(rng, rel.dy)
        else:
            k = _resample_component(rng, rel.dx or rel.dy)
            dx, dy = (k, 0) if rng.random() < 0.5 else (0, k)
        if (dx, dy) == (rel.dx, rel.dy):
            continue
        try:
            return scene, move_leaf(scene, leaf, dx, dy), leaf
        except ValueError:
            continue
    raise ValueError("could not build a pair")


def split_questions(qs: list[Question], base_coords: Coords, src_coords: Coords,
                    min_margin: float = 1.0, min_angle: float = 10.0) -> tuple[list[Question], list[Question]]:
    diag, ctrl = [], []
    for q in qs:
        if not (passes_margin(q, base_coords, min_margin, min_angle) and passes_margin(q, src_coords, min_margin, min_angle)):
            continue
        (diag if answer_for(q, base_coords) != answer_for(q, src_coords) else ctrl).append(q)
    return diag, ctrl
