import numpy as np
import pytest

from geomap.questions import QType, make_question
from geomap.scenes import Relation, Scene


def _chain():
    # R at (0,0); M is 3 east of R; C is 2 west and 1 north of M -> C at (1, 1)
    return Scene(names=("C", "R", "M"), coords={"R": (0, 0), "M": (3, 0), "C": (1, 1)},
                 relations=(Relation("M", "R", 3, 0, 0), Relation("C", "M", -2, 1, 1)),
                 parent={"R": None, "M": "R", "C": "M"}, hop={"R": 0, "M": 1, "C": 2}, seed=0)


def test_steps_on_the_asked_axis_in_the_c_frame():
    from geomap.flow import steps
    s = _chain()
    east = make_question(QType.MORE_EAST, ("R", "C"), s.names)
    north = make_question(QType.MORE_NORTH, ("C", "R"), s.names)
    assert steps(s, "M", "C", east) == (3, -2, 1)      # C is farther east than R: x_C - x_R = 1 > 0
    assert steps(s, "M", "C", north) == (0, 1, 1)


def test_flip_step_negates_one_sentence_on_one_axis_and_moves_the_dependents():
    from geomap.flow import flip_step
    s = _chain()
    f = flip_step(s, "M", 0)                            # M becomes 3 west of R; C moves with it
    assert f.coords == {"R": (0, 0), "M": (-3, 0), "C": (-5, 1)}
    assert [(r.subj, r.dx, r.dy, r.template_id) for r in f.relations] == [("M", -3, 0, 0), ("C", -2, 1, 1)]
    g = flip_step(s, "C", 1)                            # only C's north step flips
    assert g.coords == {"R": (0, 0), "M": (3, 0), "C": (1, -1)}
    assert f.names == s.names and f.parent == s.parent and f.hop == s.hop


def test_flip_step_refuses_a_zero_step_and_a_collision():
    from geomap.flow import flip_step
    s = _chain()
    with pytest.raises(ValueError):
        flip_step(s, "M", 1)                            # M's north step is 0: nothing to flip
    t = Scene(names=("R", "M", "C"), coords={"R": (0, 0), "M": (1, 0), "C": (2, 0)},
              relations=(Relation("M", "R", 1, 0, 0), Relation("C", "M", 1, 0, 0)),
              parent={"R": None, "M": "R", "C": "M"}, hop={"R": 0, "M": 1, "C": 2}, seed=0)
    with pytest.raises(ValueError):
        flip_step(t, "C", 0)                            # C would land on R


def test_axis_r2_fits_each_axis_separately_and_averages():
    from geomap.flow import axis_r2
    rng = np.random.default_rng(0)
    n = 600
    ax = np.repeat([0, 1], n // 2)
    groups = np.arange(n) // 2
    y = rng.integers(-5, 6, n).astype(float)
    X = rng.normal(size=(n, 40))
    X[:, 0] = np.where(ax == 0, y, 0)                   # the value lives in a different feature for each axis
    X[:, 1] = np.where(ax == 1, y, 0)
    tr = groups % 5 != 0
    assert axis_r2(X, y, ax, groups, tr) > 0.9
    assert axis_r2(X, rng.permutation(y), ax, groups, tr) < 0.1


def test_verdict_flow():
    from geomap.flow import verdict_flow
    assert verdict_flow(p1=0.1, p2=0.7).startswith("lost before")
    assert verdict_flow(p1=0.6, p2=0.7).startswith("reaches")
    assert verdict_flow(p1=0.35, p2=0.7) == "partial"
    assert verdict_flow(p1=0.6, p2=0.3).startswith("uninformative")


def test_where_lost_is_the_first_hop_below_threshold():
    from geomap.flow import where_lost
    assert where_lost({"C_end": 0.6, "question": 0.1, "final": 0.05}) == "question"
    assert where_lost({"C_end": 0.1, "question": 0.1, "final": 0.05}) == "C_end"
    assert where_lost({"C_end": 0.6, "question": 0.5, "final": 0.4}) is None


def test_layer_grid_reproduces_12b_and_scales_by_depth():
    from geomap.flow import layer_grid
    layers, players, win = layer_grid(48)
    assert layers == list(range(0, 49, 2)) and players == list(range(0, 45, 4)) + [46]
    assert win == {"C_end": (12, 32), "question": (16, 40), "final": (28, 48)}
    layers, players, win = layer_grid(62)
    assert layers == list(range(0, 63, 2)) and players == list(range(0, 57, 4)) + [60]
    assert win == {"C_end": (16, 42), "question": (20, 52), "final": (36, 62)}
