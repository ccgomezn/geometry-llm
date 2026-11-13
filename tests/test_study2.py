import numpy as np
import pandas as pd
import pytest

from geomap.study2 import (balanced_accuracy, box_rule, centroid_rule, feature_matrix, fit_centroid_rule,
                           pilot_summary, verdict_pilot)
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


@pytest.mark.parametrize("model_row,base,expected", [
    ((0.72, 0.62), {"box": 0.5, "feature": 0.6, "centroid": 0.9}, "go"),  # centroid ignored
    ((0.60, 0.53), {"box": 0.5, "feature": 0.55}, "weak pass"),
    ((0.70, 0.62), {"box": 0.5, "feature": 0.75}, "fail"),  # below a map-free baseline
    ((0.58, 0.48), {"box": 0.5, "feature": 0.5}, "fail"),
    ((float("nan"), float("nan")), {"box": 0.5, "feature": 0.5}, "fail"),
])
def test_verdict_pilot(model_row, base, expected):
    rows = [dict(who="model", rung=4, subset="hard", bal_acc=model_row[0], lo=model_row[1])]
    rows += [dict(who=k, rung=4, subset="hard", bal_acc=v, lo=v - 0.05) for k, v in base.items()]
    assert verdict_pilot(pd.DataFrame(rows)) == expected


def test_pilot_summary_shape():
    rng = np.random.default_rng(8)
    tss = [generate_tri_scene(rng, POOL) for _ in range(30)]
    rows = []
    for si, t in enumerate(tss):
        for it in ladder(t, rng):
            rows.append(dict(scene=si, rung=it.rung, stratum=t.stratum, truth=it.truth, pred=it.truth,
                             positive=it.positive, mass=0.0))
    s = pilot_summary(pd.DataFrame(rows), {"box": box_rule(tss)}, n_boot=50)
    hard = s[(s.who == "model") & (s.rung == 4) & (s.subset == "hard")].iloc[0]
    assert hard.bal_acc == 1.0
    assert set(s[s.who == "model"].rung) == {1, 2, 3, 4}
    assert {"inside", "box_hard", "outside_box"} <= set(s[(s.who == "model") & (s.rung == 4)].subset)
