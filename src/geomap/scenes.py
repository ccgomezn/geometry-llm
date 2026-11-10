"""Spatial scenes described in text: objects on an integer grid, fixed by a tree of relations.

Convention: north = +y, east = +x. A relation's (dx, dy) is the subject's offset from the object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

TEMPLATES = (
    "{s} is {d} of {o}.",
    "{s} lies {d} of {o}.",
    "{s} sits {d} of {o}.",
    "{s} is located {d} of {o}.",
)


@dataclass(frozen=True)
class Relation:
    subj: str
    obj: str
    dx: int
    dy: int
    template_id: int


@dataclass(frozen=True)
class Scene:
    names: tuple[str, ...]  # order of the object list in the prompt
    coords: dict[str, tuple[int, int]]
    relations: tuple[Relation, ...]  # text order
    parent: dict[str, str | None]
    hop: dict[str, int]
    seed: int

    @property
    def root(self) -> str:
        return next(n for n, p in self.parent.items() if p is None)


def _units(k: int) -> str:
    return f"{k} unit" if k == 1 else f"{k} units"


def offset_phrase(dx: int, dy: int) -> str:
    parts = []
    if dy:
        parts.append(f"{_units(abs(dy))} {'north' if dy > 0 else 'south'}")
    if dx:
        parts.append(f"{_units(abs(dx))} {'east' if dx > 0 else 'west'}")
    if not parts:
        raise ValueError("zero offset")
    return " and ".join(parts)


def relation_text(rel: Relation) -> str:
    return TEMPLATES[rel.template_id].format(s=rel.subj, d=offset_phrase(rel.dx, rel.dy), o=rel.obj)


def describe(scene: Scene) -> str:
    return " ".join(relation_text(r) for r in scene.relations)


def leaves(scene: Scene) -> list[str]:
    parents = {p for p in scene.parent.values() if p is not None}
    return [n for n in scene.names if n not in parents and scene.parent[n] is not None]


def _random_offset(rng: np.random.Generator, p_compound: float) -> tuple[int, int]:
    def comp() -> int:
        return int(rng.integers(1, 6)) * (1 if rng.random() < 0.5 else -1)

    if rng.random() < p_compound:
        return comp(), comp()
    return (comp(), 0) if rng.random() < 0.5 else (0, comp())


def generate_scene(
    rng: np.random.Generator,
    names_pool: Sequence[str],
    n_objects: int,
    p_compound: float = 0.3,
    p_shuffle: float = 0.2,
    bound: int = 6,
) -> Scene:
    seed = int(rng.integers(2**31))
    while True:
        chosen = [str(x) for x in rng.choice(list(names_pool), size=n_objects, replace=False)]
        root = chosen[0]
        coords = {root: (0, 0)}
        parent: dict[str, str | None] = {root: None}
        placed = [root]
        rels: list[Relation] = []
        ok = True
        for name in chosen[1:]:
            for _ in range(100):
                # bias toward chains so hop depth varies (ledger ruling)
                par = placed[-1] if (len(placed) > 1 and rng.random() < 0.5) else placed[int(rng.integers(len(placed)))]
                dx, dy = _random_offset(rng, p_compound)
                px, py = coords[par]
                pos = (px + dx, py + dy)
                if abs(pos[0]) <= bound and abs(pos[1]) <= bound and pos not in coords.values():
                    break
            else:
                ok = False
                break
            coords[name] = pos
            parent[name] = par
            placed.append(name)
            rels.append(Relation(name, par, dx, dy, int(rng.integers(len(TEMPLATES)))))
        if ok:
            break

    if rng.random() < p_shuffle:
        rels = [rels[i] for i in rng.permutation(len(rels))]
        first = next(i for i, r in enumerate(rels) if r.obj == root)
        rels.insert(0, rels.pop(first))

    hop = {root: 0}
    for name in chosen[1:]:  # creation order: parents come first
        hop[name] = hop[parent[name]] + 1
    names = tuple(chosen[i] for i in rng.permutation(n_objects))
    return Scene(names=names, coords=coords, relations=tuple(rels), parent=parent, hop=hop, seed=seed)


def move_leaf(scene: Scene, leaf: str, dx: int, dy: int, bound: int = 6) -> Scene:
    if leaf not in leaves(scene):
        raise ValueError(f"{leaf} is not a leaf")
    if (dx, dy) == (0, 0):
        raise ValueError("zero offset")
    px, py = scene.coords[scene.parent[leaf]]
    pos = (px + dx, py + dy)
    if abs(pos[0]) > bound or abs(pos[1]) > bound:
        raise ValueError("out of bounds")
    if any(scene.coords[n] == pos for n in scene.names if n != leaf):
        raise ValueError("collision")
    rels = tuple(replace(r, dx=dx, dy=dy) if r.subj == leaf else r for r in scene.relations)
    coords = dict(scene.coords)
    coords[leaf] = pos
    return replace(scene, coords=coords, relations=rels)


def descendants(scene: Scene, name: str) -> set[str]:
    out, frontier = set(), {name}
    while frontier:
        frontier = {n for n, p in scene.parent.items() if p in frontier} - out
        out |= frontier
    return out


def move_object(scene: Scene, name: str, dx: int, dy: int, bound: int = 6) -> Scene:
    """Set `name`'s offset from its parent to (dx, dy); everything placed relative to it moves with it."""
    if scene.parent[name] is None:
        raise ValueError("cannot move the root")
    if (dx, dy) == (0, 0):
        raise ValueError("zero offset")
    px, py = scene.coords[scene.parent[name]]
    ox, oy = scene.coords[name]
    sx, sy = px + dx - ox, py + dy - oy
    moving = {name} | descendants(scene, name)
    coords = {n: ((x + sx, y + sy) if n in moving else (x, y)) for n, (x, y) in scene.coords.items()}
    if any(abs(x) > bound or abs(y) > bound for x, y in coords.values()):
        raise ValueError("out of bounds")
    if len(set(coords.values())) < len(coords):
        raise ValueError("collision")
    rels = tuple(replace(r, dx=dx, dy=dy) if r.subj == name else r for r in scene.relations)
    return replace(scene, coords=coords, relations=rels)


def hybrid(base: Scene, src: Scene, child: str) -> Scene:
    """`src`, but with the leaf `child` re-expressed relative to its (moved) parent so it keeps its base position."""
    if child not in leaves(src):
        raise ValueError(f"{child} is not a leaf")
    pos = base.coords[child]
    px, py = src.coords[src.parent[child]]
    dx, dy = pos[0] - px, pos[1] - py
    if (dx, dy) == (0, 0):
        raise ValueError("zero offset")
    if any(src.coords[n] == pos for n in src.names if n != child):
        raise ValueError("collision")
    rels = tuple(replace(r, dx=dx, dy=dy) if r.subj == child else r for r in src.relations)
    coords = dict(src.coords)
    coords[child] = pos
    return replace(src, coords=coords, relations=rels)
