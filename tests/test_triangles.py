import numpy as np
import pytest

from geomap.triangles import edge_distance, generate_tri_scene, in_triangle, stratum

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
