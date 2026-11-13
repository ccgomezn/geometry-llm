import numpy as np
import pytest

from geomap.models import load_tiny_for_tests
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for
from geomap.triangles import STRATA, edge_distance, generate_tri_scene, in_triangle, ladder, stratum

POOL = ["Ann", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal"]
T = ((0, 0), (4, 0), (0, 4))


def coords_from_relations(s):
    co = {s.root: (0, 0)}
    pending = list(s.relations)
    while pending:
        r = next(r for r in pending if r.obj in co)
        ox, oy = co[r.obj]
        co[r.subj] = (ox + r.dx, oy + r.dy)
        pending.remove(r)
    return co


def test_in_triangle():
    assert in_triangle((1, 1), *T) and not in_triangle((3, 3), *T)
    assert not in_triangle((2, 0), *T)  # on an edge
    assert not in_triangle((5, 5), *T)


def test_edge_distance():
    assert edge_distance((1, 1), *T) == pytest.approx(1.0)
    assert edge_distance((2, 2), *T) == pytest.approx(0.0)  # on the hypotenuse


def test_stratum():
    assert stratum((1, 1), *T) == "inside"
    assert stratum((3, 3), *T) == "box_hard"
    assert stratum((5, 0), *T) == "outside_box"


def test_generator_invariants():
    rng = np.random.default_rng(0)
    for _ in range(200):
        ts = generate_tri_scene(rng, POOL)
        s, (a, b, c), p = ts.scene, ts.corners, ts.point
        assert coords_from_relations(s) == s.coords
        A, B, C, P = (s.coords[n] for n in (a, b, c, p))
        assert abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / 2 >= 4
        assert edge_distance(P, A, B, C) >= 0.5
        assert stratum(P, A, B, C) == ts.stratum
        assert s.parent[p] in ts.corners and s.parent[c] == b and s.parent[b] == a
        assert all(abs(x) <= 8 and abs(y) <= 8 for x, y in s.coords.values())
        assert s.relations[0].subj == b and s.relations[0].obj == a


def test_strata_proportions():
    rng = np.random.default_rng(1)
    st = [generate_tri_scene(rng, POOL).stratum for _ in range(1000)]
    assert 0.45 <= st.count("inside") / 1000 <= 0.55
    assert st.count("box_hard") / 1000 >= 0.30


def test_generator_terminates_small_bound():  # Review Focus 1
    rng = np.random.default_rng(2)
    for _ in range(50):
        generate_tri_scene(rng, POOL, bound=3, min_area=2.0)


def test_ladder_truths_and_rungs():
    rng = np.random.default_rng(3)
    for _ in range(100):
        ts = generate_tri_scene(rng, POOL)
        items = ladder(ts, rng)
        co, p, par = ts.scene.coords, ts.point, ts.scene.parent[ts.point]
        assert sum(it.rung == 4 for it in items) == 1
        for it in items:
            q = it.question
            if it.rung in (1, 2):
                other = next(a for a in q.args if a != p)
                assert (other == par) == (it.rung == 1) and other in ts.corners
                assert it.truth == answer_for(q, co) and it.positive == q.args[0]
            else:
                assert q.candidates == ("yes", "no") and it.positive == "yes"
            if it.rung == 4:
                assert (it.truth == "yes") == in_triangle(co[p], *(co[c] for c in ts.corners))
            if it.rung == 3:
                ax, sign = {"north": (1, 1), "east": (0, 1), "south": (1, -1), "west": (0, -1)}[q.text.split()[3]]
                beyond = all(sign * (co[p][ax] - co[c][ax]) > 0 for c in ts.corners)
                assert (it.truth == "yes") == beyond


def test_ladder_prompts_build():
    m = load_tiny_for_tests()
    pool = single_token_names(m.tok, NAME_POOLS["people"])
    rng = np.random.default_rng(4)
    for _ in range(5):
        ts = generate_tri_scene(rng, pool)
        for it in ladder(ts, rng):
            spec = build_prompt(ts.scene, it.question, m.tok, m.family)
            assert set(spec.question_pos) == set(it.question.args)


def test_every_triangle_admits_all_strata():  # review #1: the stratum must not depend on the triangle's shape
    rng = np.random.default_rng(9)
    grid = [(x, y) for x in range(-8, 9) for y in range(-8, 9)]
    for _ in range(300):
        ts = generate_tri_scene(rng, POOL)
        A, B, C = (ts.scene.coords[k] for k in ts.corners)
        got = {stratum(q, A, B, C) for q in grid
               if q not in (A, B, C) and edge_distance(q, A, B, C) >= 0.5}
        assert got == set(STRATA)


def test_ladder_answer_matches_truth_and_moves_with_coords():
    from geomap.triangles import ladder_answer
    rng = np.random.default_rng(12)
    for _ in range(100):
        ts = generate_tri_scene(rng, POOL)
        for it in ladder(ts, rng):
            assert ladder_answer(it.question, ts.scene.coords) == it.truth
    # float coordinates (a decoded map) work, and moving P changes the inside answer
    ts = generate_tri_scene(np.random.default_rng(13), POOL)
    q = next(it for it in ladder(ts, np.random.default_rng(0)) if it.rung == 4).question
    a, b, c = (ts.scene.coords[k] for k in ts.corners)
    g = ((a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3)
    co = {**{k: tuple(map(float, v)) for k, v in ts.scene.coords.items()}, ts.point: g}
    assert ladder_answer(q, co) == "yes"
    co[ts.point] = (100.0, 100.0)
    assert ladder_answer(q, co) == "no"
