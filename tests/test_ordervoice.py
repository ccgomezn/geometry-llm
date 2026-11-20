import numpy as np
import pandas as pd
import pytest

from geomap.scenes import Relation, Scene, describe


def _example():
    # the paper's example: Beth is 4 east of Amy; Leo is 1 west of Beth -> Leo at 3
    return Scene(names=("Leo", "Beth", "Amy"), coords={"Amy": (0, 0), "Beth": (4, 0), "Leo": (3, 0)},
                 relations=(Relation("Beth", "Amy", 4, 0, 0), Relation("Leo", "Beth", -1, 0, 0)),
                 parent={"Amy": None, "Beth": "Amy", "Leo": "Beth"}, hop={"Amy": 0, "Beth": 1, "Leo": 2}, seed=0)


def test_render_changes_only_order_and_voice_of_the_description():
    from geomap.ordervoice import render
    s = _example()
    assert describe(render(s, "Beth", "Leo", "MC-subj")) == "Beth is 4 units east of Amy. Leo is 1 unit west of Beth."
    assert describe(render(s, "Beth", "Leo", "CM-subj")) == "Leo is 1 unit west of Beth. Beth is 4 units east of Amy."
    assert describe(render(s, "Beth", "Leo", "MC-obj")) == "Beth is 4 units east of Amy. Beth is 1 unit east of Leo."
    assert describe(render(s, "Beth", "Leo", "CM-obj")) == "Beth is 1 unit east of Leo. Beth is 4 units east of Amy."
    for r in ("MC-subj", "CM-subj", "MC-obj", "CM-obj"):
        t = render(s, "Beth", "Leo", r)
        assert t.coords == s.coords and t.names == s.names and t.root == s.root


def test_object_voice_states_the_same_fact():
    # every relation, read as subj = obj + (dx, dy), must agree with the coordinates
    from geomap.ordervoice import render
    s = _example()
    for r in ("MC-obj", "CM-obj"):
        t = render(s, "Beth", "Leo", r)
        for rel in t.relations:
            sx, sy = t.coords[rel.subj]
            ox, oy = t.coords[rel.obj]
            assert (sx - ox, sy - oy) == (rel.dx, rel.dy)


def test_features_separate_own_step_word_and_last_sentence():
    from geomap.ordervoice import features
    # C's step points the asked way (d2 > 0), M's step against it (d1 < 0)
    assert features(-4, 1, "MC-subj") == (1, 1, 1)
    assert features(-4, 1, "CM-subj") == (1, 1, 0)      # last sentence is M's, which says the opposite word
    assert features(-4, 1, "MC-obj") == (1, 0, 0)       # "M is 1 unit west of C": the word is reversed
    assert features(-4, 1, "CM-obj") == (1, 0, 0)
    # C's step points away, M's step the asked way
    assert features(4, -1, "MC-subj") == (0, 0, 0)
    assert features(4, -1, "CM-subj") == (0, 0, 1)
    assert features(4, -1, "MC-obj") == (0, 1, 1)
    assert features(4, -1, "CM-obj") == (0, 1, 1)
    # no step of C on the asked axis: no direction word in C's sentence on this axis
    assert features(3, 0, "MC-obj") == (0, 0, 0)
    assert features(3, 0, "CM-obj") == (0, 0, 1)


def test_design_columns_and_identifiability():
    from geomap.ordervoice import RENDERINGS, design, features
    rows = []
    for d1, d2 in [(-4, 1), (4, -1), (2, 1), (0, 2), (0, -2), (3, 0), (-3, 0), (1, -3)]:
        for r in RENDERINGS:
            own, word, last = features(d1, d2, r)
            rows.append(dict(d1=d1, d2=d2, rendering=r, c_first=bool(d1 % 2), own_yes=own, word_yes=word,
                             last_yes=last))
    F = design(pd.DataFrame(rows))
    assert F.shape[1] == 9
    assert np.linalg.matrix_rank(F) == 9          # own, word and last are separable across renderings


def test_verdict_rule():
    from geomap.ordervoice import verdict_order_voice
    coef = {"own": (10.0, 8.0, 12.0), "word": (1.0, -1.0, 3.0), "last": (0.5, -1.0, 2.0)}
    diff = {("own", "word"): (9.0, 6.0, 12.0), ("own", "last"): (9.5, 7.0, 12.0), ("word", "last"): (0.5, -2.0, 3.0)}
    assert verdict_order_voice(coef, diff) == "the rule follows C's own step"
    coef2 = {"own": (1.0, -1.0, 3.0), "word": (9.0, 7.0, 11.0), "last": (2.0, 0.5, 3.5)}
    diff2 = {("own", "word"): (-8.0, -10.0, -6.0), ("own", "last"): (-1.0, -3.0, 1.0), ("word", "last"): (7.0, 5.0, 9.0)}
    assert verdict_order_voice(coef2, diff2) == "the rule follows the direction word"
    coef3 = {"own": (5.0, 3.0, 7.0), "word": (4.0, 2.0, 6.0), "last": (0.0, -1.0, 1.0)}
    diff3 = {("own", "word"): (1.0, -1.0, 3.0), ("own", "last"): (5.0, 3.0, 7.0), ("word", "last"): (4.0, 2.0, 6.0)}
    assert verdict_order_voice(coef3, diff3) == "mixed: C's own step, the direction word"
