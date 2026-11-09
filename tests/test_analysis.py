import numpy as np
import pandas as pd

from geomap.analysis import bootstrap_ci, chance, error_signature
from geomap.questions import COMPASS, QType, Question


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1, 400)
    est, lo, hi = bootstrap_ci(x)
    assert lo < 5 < hi and lo < est < hi
    assert hi - lo < 0.4


def test_bootstrap_ci_grouped():
    # two groups of identical values: resampling groups gives a wide interval
    x = np.array([0.0] * 50 + [1.0] * 50)
    g = np.array([0] * 50 + [1] * 50)
    est, lo, hi = bootstrap_ci(x, groups=g)
    assert est == 0.5 and lo == 0.0 and hi == 1.0


def test_chance_values():
    assert chance(Question(QType.FARTHEST, ("A",), "", ("B", "C", "D"))) == 1 / 3
    assert chance(Question(QType.DIRECTION, ("A", "B"), "", COMPASS)) == 1 / 8
    assert chance(Question(QType.CLOSER_THAN, ("A", "B", "C"), "", ("yes", "no"))) == 1 / 2


def test_error_signature_recovers_planted_effect():
    rng = np.random.default_rng(1)
    n = 3000
    qtype = rng.choice(["farthest", "direction"], n)
    margin = rng.lognormal(0, 0.7, n) * np.where(qtype == "direction", 20, 1)
    hop = rng.integers(1, 5, n)
    z = np.log(margin)
    z = (z - pd.Series(z).groupby(qtype).transform("mean").values) / pd.Series(z).groupby(qtype).transform("std").values
    p = 1 / (1 + np.exp(-(0.5 + 1.5 * z)))
    df = pd.DataFrame({"qtype": qtype, "margin": margin, "max_hop": hop, "correct": rng.random(n) < p,
                       "scene_id": np.arange(n) // 6})
    sig = error_signature(df, n_boot=100)
    m_est, m_lo, m_hi = sig["log_margin"]
    h_est, h_lo, h_hi = sig["hop"]
    assert m_lo > 0
    assert h_lo < 0 < h_hi


def test_gate_pass_and_fail():
    from geomap.analysis import gate
    rng = np.random.default_rng(0)
    n = 2000
    base = pd.DataFrame({"set": "main", "n_objects": 4, "scene_id": np.arange(n) // 10,
                         "max_hop": rng.integers(1, 4, n), "chance": 0.25})
    good = base.assign(correct=rng.random(n) < 0.9)
    bad = base.assign(correct=rng.random(n) < 0.3)
    assert gate(good)["pass"] and not gate(bad)["pass"]
    g = gate(good)
    assert 0.85 < g["acc4"][0] < 0.95 and g["hop2_excess"][1] > 0.15


def _p1(acc4, acc3, closer3=0.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n, acc in [(4, acc4), (3, acc3)]:
        for i in range(600):
            qt = ["farthest", "direction", "closer_than"][i % 3]
            p = closer3 if (qt == "closer_than") else acc
            rows.append(dict(set="main", n_objects=n, scene_id=n * 1000 + i // 3, max_hop=2, qtype=qt,
                             chance=0.5 if qt == "closer_than" else 0.3, correct=rng.random() < p))
    return pd.DataFrame(rows)


def test_choose_model_prefers_smallest_on_4_objects_then_falls_back():
    from geomap.analysis import choose_model
    order = ["small", "big"]
    c = choose_model({"small": _p1(0.5, 0.6), "big": _p1(0.9, 0.95)}, order)
    assert c["model"] == "big" and c["n_objects"] == [4]
    c = choose_model({"small": _p1(0.5, 0.8, closer3=0.8), "big": _p1(0.6, 0.9, closer3=0.8)}, order)
    assert c["model"] == "small" and c["n_objects"] == [3] and c["exclude"] == []
    c = choose_model({"small": _p1(0.5, 0.6), "big": _p1(0.6, 0.92)}, order)
    assert c["model"] == "big" and c["exclude"] == ["closer_than"]  # at chance on the chosen configuration
    assert choose_model({"small": _p1(0.3, 0.3)}, ["small"])["model"] is None
