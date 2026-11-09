import numpy as np
import pandas as pd
import pytest

from geomap.models import load_tiny_for_tests
from geomap.prompts import NAME_POOLS, build_prompt
from geomap.questions import QType, all_questions, make_question
from geomap.scenes import Scene, generate_scene, leaves
from geomap.scoring import predict, score
from geomap.steering import natural_range, predicted_answers, slider, slider_agreement


def test_predicted_flip_points():
    s = Scene(names=("A", "B", "C", "T"), coords={"A": (0, 0), "B": (-3, 0), "C": (0, 4), "T": (1, 0)},
              relations=(), parent={"A": None, "B": "A", "C": "A", "T": "A"}, hop={"A": 0, "B": 1, "C": 1, "T": 1}, seed=0)
    q = make_question(QType.FARTHEST, ("B",), s.names)
    deltas = np.array([[dx, 0.0] for dx in np.arange(0, 4.01, 0.5)])
    pred = predicted_answers(s, "T", deltas, q)
    # distance B->C = 5; B->T = 4 + dx, so T becomes farthest once dx > 1
    assert pred[:3] == ["C", "C", "C"] and all(p == "T" for p in pred[3:])


def test_agreement_math():
    rows = []
    for step, (pred, mod) in enumerate([("C", "C"), ("C", "C"), ("T", "C"), ("T", "T"), ("T", "T")]):
        rows.append(dict(scene=0, axis="x", qid=0, step=step, dx=step * 0.5, dy=0.0, predicted_ans=pred, model_ans=mod))
    a = slider_agreement(pd.DataFrame(rows), n_boot=20)
    assert a["agreement"] == 0.8
    assert a["n_sweeps_with_flip"] == 1 and a["frac_observed_flip"] == 1.0
    assert a["flip_offsets"] == [0.5]  # model flips 0.5 units after the geometric threshold


def test_natural_range():
    ref = np.random.default_rng(0).normal(10, 1, 1000)
    assert natural_range(np.array([10.0, 10.5, 30.0]), ref) == pytest.approx(2 / 3)


def test_zero_delta_is_identity():
    m = load_tiny_for_tests()
    s = generate_scene(np.random.default_rng(2), NAME_POOLS["people"], 4)
    leaf = leaves(s)[0]
    qs = all_questions(s)[:4]
    dirs = np.random.default_rng(0).normal(size=(m.d_model, 2))
    df = slider(m, s, leaf, layer=1, dirs=dirs, deltas=np.zeros((1, 2)), qs=qs)
    ref = [predict(sc) for sc in score(m, [(build_prompt(s, q, m.tok, m.family), q) for q in qs])]
    assert df.sort_values("qid").model_ans.tolist() == ref
