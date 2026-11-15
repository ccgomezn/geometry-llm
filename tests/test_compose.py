import numpy as np
import pandas as pd
import pytest

NAMES = [f"n{i}" for i in range(40)]


def _scenes(n=600, seed=3):
    from geomap.scenes import generate_scene
    rng = np.random.default_rng(seed)
    return [generate_scene(rng, NAMES, 3 + i % 4) for i in range(n)]


def test_part_rows_targets_and_per_sentence_codes():
    from geomap.compose import part_rows
    from geomap.scenes import Relation, Scene
    s = Scene(names=("R", "M", "C"), coords={"R": (0, 0), "M": (3, 0), "C": (1, 2)},
              relations=(Relation("M", "R", 3, 0, 0), Relation("C", "M", -2, 2, 0)),
              parent={"R": None, "M": "R", "C": "M"}, hop={"R": 0, "M": 1, "C": 2}, seed=0)
    meta, parts, pos = part_rows([s], [0], hops=(1, 2))
    assert list(meta.name) == ["M", "C"] and list(meta.slot) == [1, 2] and list(meta.hop) == [1, 2]
    c = meta.iloc[1]
    assert c.dist == pytest.approx(np.hypot(1, 2)) and (c.abs_x, c.abs_y) == (1, 2)
    # one one-hot per sentence over the joint (dx, dy) of that sentence: own block, then parent block
    assert parts.shape == (2, 242) and parts.sum(1).tolist() == [1, 2]
    assert parts[1, (-2 + 5) * 11 + (2 + 5)] == 1 and parts[1, 121 + (3 + 5) * 11 + (0 + 5)] == 1
    assert parts[0, 121:].sum() == 0          # a one-step object's parent is the root: no parent sentence
    assert pos.shape == (2, 5) and pos[1].tolist() == [1, 2, 1, 4, 2]


def test_part_rows_keeps_only_the_asked_hops():
    from geomap.compose import part_rows
    sc = _scenes(50)
    meta, parts, _ = part_rows(sc, range(50), hops=(2,))
    assert len(meta) and set(meta.hop) == {2} and len(parts) == len(meta)


def _synthetic(sc, kind, rng):
    """Fake activations: each sentence's code side by side ("parts"), or a code of the composed position."""
    from geomap.compose import part_rows
    meta, parts, _ = part_rows(sc, range(len(sc)), hops=(2,))
    if kind == "parts":
        X = parts @ rng.normal(size=(242, 300))
    else:
        rx = np.array([sc[i].coords[sc[i].root] for i in meta.scene])
        xy = np.array([sc[i].coords[n] for i, n in zip(meta.scene, meta.name)]) - rx
        onehot = np.zeros((len(meta), 13 * 13))
        onehot[np.arange(len(meta)), (xy[:, 0] + 6) * 13 + xy[:, 1] + 6] = 1
        X = onehot @ rng.normal(size=(169, 300))
    return meta, parts, X + 0.01 * rng.normal(size=X.shape)


@pytest.mark.parametrize("kind,composed", [("parts", False), ("composed", True)])
def test_gap_separates_side_by_side_parts_from_a_composed_position(kind, composed):
    from geomap.compose import site_scores
    sc = _scenes(900)
    rng = np.random.default_rng(0)
    meta, parts, X = _synthetic(sc, kind, rng)
    tr = meta.scene.values < 700
    r = site_scores(X, parts, meta, tr, ~tr, target="dist", n_boot=200)
    if composed:
        assert r["r2_act"] > 0.8 and r["gap_lo"] > 0.3
    else:
        assert abs(r["gap"]) < 0.05 and r["gap_lo"] < 0


def test_verdict_compose():
    from geomap.compose import verdict_compose
    row = lambda site, gap, lo, hi: dict(site=site, target="dist", primary=True, gap=gap, gap_lo=lo, gap_hi=hi)
    yes = pd.DataFrame([row("list", 0.3, 0.2, 0.4), row("sentence", 0.02, -0.03, 0.04)])
    assert verdict_compose(yes) == ("composed by the model", ["list"])
    no = pd.DataFrame([row("list", 0.01, -0.03, 0.04), row("sentence", 0.0, -0.04, 0.03)])
    assert verdict_compose(no) == ("parts only", [])
    small = pd.DataFrame([row("list", 0.08, 0.02, 0.14), row("sentence", 0.0, -0.04, 0.03)])
    assert verdict_compose(small) == ("undecided", [])


def test_unseen_pairs_marks_test_rows_whose_two_offsets_never_occur_together_in_training():
    from geomap.compose import unseen_pairs
    p = np.zeros((4, 242), dtype=np.float32)
    for row, (own, par) in enumerate([(3, 7), (3, 8), (3, 7), (4, 8)]):
        p[row, own] = p[row, 121 + par] = 1
    tr = np.array([True, True, False, False])
    assert unseen_pairs(p, tr).tolist() == [False, False, False, True]
