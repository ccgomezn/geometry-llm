import numpy as np
import pytest

from geomap.scenes import Relation, describe, generate_scene, leaves, move_leaf, relation_text

POOL = [f"N{i}" for i in range(40)]


def _scenes(n_seeds=500):
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        yield generate_scene(rng, POOL, n_objects=3 + seed % 4)


def test_coords_consistent_with_relations():
    for s in _scenes(200):
        for r in s.relations:
            sx, sy = s.coords[r.subj]
            ox, oy = s.coords[r.obj]
            assert (sx - ox, sy - oy) == (r.dx, r.dy)


def test_root_at_origin_and_hops():
    for s in _scenes(200):
        assert s.coords[s.root] == (0, 0)
        assert s.hop[s.root] == 0
        assert s.root in (s.relations[0].subj, s.relations[0].obj)
        for name in s.names:
            if name != s.root:
                assert s.hop[name] == s.hop[s.parent[name]] + 1


def test_distinct_positions_in_bounds():
    for s in _scenes(500):
        pts = list(s.coords.values())
        assert len(set(pts)) == len(pts)
        assert all(abs(x) <= 6 and abs(y) <= 6 for x, y in pts)
        assert len(s.names) == len(s.relations) + 1


def test_relation_text_cardinal_and_compound():
    assert "3 units north of Bo" in relation_text(Relation("Ana", "Bo", 0, 3, 0))
    t = relation_text(Relation("Ana", "Bo", 2, -1, 0))
    assert "1 unit south" in t and "2 units east" in t


def test_describe_mentions_every_object():
    s = next(_scenes(1))
    text = describe(s)
    assert all(n in text for n in s.names)


def test_move_leaf_only_moves_leaf():
    for s in _scenes(100):
        leaf = leaves(s)[0]
        p = s.parent[leaf]
        for dx, dy in [(1, 0), (0, -2), (2, 3), (-3, 1)]:
            try:
                s2 = move_leaf(s, leaf, dx, dy)
            except ValueError:
                continue
            px, py = s.coords[p]
            assert s2.coords[leaf] == (px + dx, py + dy)
            assert all(s2.coords[n] == s.coords[n] for n in s.names if n != leaf)
            idx = [r.subj for r in s.relations].index(leaf)
            assert s2.relations[idx].template_id == s.relations[idx].template_id
            break


def test_move_leaf_rejects_collision():
    s = next(_scenes(1))
    leaf = leaves(s)[0]
    other = next(n for n in s.names if n != leaf)
    px, py = s.coords[s.parent[leaf]]
    ox, oy = s.coords[other]
    if (ox - px, oy - py) != (0, 0):
        with pytest.raises(ValueError):
            move_leaf(s, leaf, ox - px, oy - py)


def test_deterministic_from_seed():
    a = generate_scene(np.random.default_rng(7), POOL, 5)
    b = generate_scene(np.random.default_rng(7), POOL, 5)
    assert a == b
