import numpy as np
import pandas as pd
import pytest

from geomap.study2 import (balanced_accuracy, box_rule, centroid_rule, check_answers, feature_matrix,
                           fit_centroid_rule, fit_magnitude_rule, magnitude_matrix, magnitude_rule, pilot_summary,
                           verdict_pilot)
from geomap.triangles import generate_tri_scene, ladder

POOL = ["Ann", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal"]


def test_balanced_accuracy():
    assert balanced_accuracy(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"], "yes") == 0.75
    assert np.isnan(balanced_accuracy(["no", "no"], ["no", "yes"], "yes"))  # Review Focus 3


def test_two_choice_positive_class():
    # positive = args[0] per item, so `positive` may be an array
    assert balanced_accuracy(["Ann", "Bo"], ["Ann", "Ann"], np.array(["Ann", "Ann"])) == 0.5


def test_box_rule_at_chance_on_hard():
    rng = np.random.default_rng(5)
    tss = [generate_tri_scene(rng, POOL) for _ in range(500)]
    hard = [t for t in tss if t.stratum in ("inside", "box_hard")]
    truth = ["yes" if t.stratum == "inside" else "no" for t in hard]
    assert balanced_accuracy(truth, box_rule(hard), "yes") == 0.5


def test_feature_matrix():
    rng = np.random.default_rng(6)
    t = generate_tri_scene(rng, POOL)
    X = feature_matrix([t])
    co = t.scene.coords
    a, b, c = t.corners
    assert X.shape == (1, 9)
    assert tuple(X[0, :2]) == (co[b][0] - co[a][0], co[b][1] - co[a][1])
    assert X[0, 6:].sum() == 1


def test_centroid_rule_beats_chance_on_train():
    rng = np.random.default_rng(7)
    tss = [generate_tri_scene(rng, POOL) for _ in range(400)]
    thr = fit_centroid_rule(tss)
    truth = ["yes" if t.stratum == "inside" else "no" for t in tss]
    assert balanced_accuracy(truth, centroid_rule(tss, thr), "yes") > 0.6


@pytest.mark.parametrize("model_row,diff_lo,expected", [
    ((0.72, 0.62), 0.02, "go"),
    ((0.60, 0.53), 0.01, "weak pass"),
    ((0.72, 0.62), -0.01, "fail"),  # not clearly above the best map-free baseline
    ((0.58, 0.48), 0.02, "fail"),  # not clearly above chance
    ((float("nan"), float("nan")), 0.02, "fail"),
])
def test_verdict_pilot(model_row, diff_lo, expected):
    rows = [dict(who="model", rung=4, subset="hard", bal_acc=model_row[0], lo=model_row[1]),
            dict(who="model_minus_best_map_free", rung=4, subset="hard", bal_acc=0.05, lo=diff_lo),
            dict(who="centroid", rung=4, subset="hard", bal_acc=0.99, lo=0.95)]  # reference only, ignored
    assert verdict_pilot(pd.DataFrame(rows)) == expected


def test_magnitude_matrix():
    rng = np.random.default_rng(10)
    t = generate_tri_scene(rng, POOL)
    X = magnitude_matrix([t])
    assert X.shape == (1, 14) and (X[0, :11] >= 0).all() and X[0, 11:].sum() == 1


def test_magnitude_rule_finds_the_size_shortcut():  # review #1: a real shortcut baseline, well above chance
    rng = np.random.default_rng(11)
    train = [generate_tri_scene(rng, POOL) for _ in range(2000)]
    test = [t for t in (generate_tri_scene(rng, POOL) for _ in range(800)) if t.stratum in ("inside", "box_hard")]
    truth = ["yes" if t.stratum == "inside" else "no" for t in test]
    assert balanced_accuracy(truth, magnitude_rule(fit_magnitude_rule(train), test), "yes") > 0.55


def test_check_answers_rejects_stale():  # review #2
    df = pd.DataFrame(dict(text=["q1", "q2"], truth=["yes", "no"]))
    check_answers(df, ["q1", "q2"], ["yes", "no"])
    for texts, truths in ((["q1"], ["yes"]), (["q1", "qX"], ["yes", "no"]), (["q1", "q2"], ["yes", "yes"])):
        with pytest.raises(ValueError):
            check_answers(df, texts, truths)


def test_pilot_summary_shape():
    rng = np.random.default_rng(8)
    tss = [generate_tri_scene(rng, POOL) for _ in range(30)]
    rows = []
    for si, t in enumerate(tss):
        for it in ladder(t, rng):
            rows.append(dict(scene=si, rung=it.rung, stratum=t.stratum, truth=it.truth, pred=it.truth,
                             positive=it.positive, mass=0.0))
    s = pilot_summary(pd.DataFrame(rows), {"box": box_rule(tss)}, n_boot=50, map_free=("box",))
    hard = s[(s.who == "model") & (s.rung == 4) & (s.subset == "hard")].iloc[0]
    assert hard.bal_acc == 1.0
    d = s[s.who == "model_minus_best_map_free"].iloc[0]  # paired scene bootstrap vs the best map-free rule
    assert d.bal_acc == pytest.approx(0.5) and d.lo > 0 and d.vs == "box"
    assert set(s[s.who == "model"].rung) == {1, 2, 3, 4}
    assert {"inside", "box_hard", "outside_box"} <= set(s[(s.who == "model") & (s.rung == 4)].subset)


def test_auc_ci():
    from geomap.study2 import auc_ci
    lab = np.array([1, 1, 0, 0] * 10)
    g = np.repeat(np.arange(20), 2)
    assert auc_ci(np.where(lab == 1, 2.0, -1.0), lab, g, n_boot=50)[0] == 1.0
    assert auc_ci(np.where(lab == 1, -2.0, 1.0), lab, g, n_boot=50)[0] == 0.0
    est, lo, hi = auc_ci(np.zeros(40), lab, g, n_boot=50)
    assert est == 0.5 and lo <= 0.5 <= hi
