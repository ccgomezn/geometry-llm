import numpy as np
import pandas as pd
import pytest

from geomap.composition import make_chain_pairs, run_chain_pair, summarize, verdict
from geomap.models import load_tiny_for_tests
from geomap.prompts import aligned, build_prompt
from geomap.scenes import Relation, Scene, hybrid, move_leaf, move_object


def chain():
    # R at (0,0); M 2 north of R; C 3 east of M
    rels = (Relation("M", "R", 0, 2, 0), Relation("C", "M", 3, 0, 1))
    return Scene(names=("C", "R", "M"), coords={"R": (0, 0), "M": (0, 2), "C": (3, 2)}, relations=rels,
                 parent={"R": None, "M": "R", "C": "M"}, hop={"R": 0, "M": 1, "C": 2}, seed=0)


def test_move_object_drags_descendants():
    s = move_object(chain(), "M", -1, 0)
    assert s.coords == {"R": (0, 0), "M": (-1, 0), "C": (2, 0)}
    assert s.relations[0] == Relation("M", "R", -1, 0, 0) and s.relations[1] == chain().relations[1]


def test_move_object_on_leaf_equals_move_leaf():
    assert move_object(chain(), "C", 0, -1) == move_leaf(chain(), "C", 0, -1)


def test_move_object_rejects_out_of_bounds_and_collisions():
    with pytest.raises(ValueError):
        move_object(chain(), "M", 5, 0)  # C would be at (8, 0)
    with pytest.raises(ValueError):
        move_object(chain(), "M", -3, 0)  # M at (-3, 0), so C lands on R at (0, 0)


def test_hybrid_keeps_child_at_base_position():
    base = chain()
    src = move_object(base, "M", -1, 0)
    h = hybrid(base, src, "C")
    assert h.coords == {"R": (0, 0), "M": (-1, 0), "C": (3, 2)}
    rc = next(r for r in h.relations if r.subj == "C")
    assert (rc.dx, rc.dy) == (4, 2) and rc.template_id == 1


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def pairs(m):
    return make_chain_pairs(m, 4, seed=0)


def test_make_chain_pairs_structure(m, pairs):
    assert len(pairs) == 4
    for p in pairs:
        b, s, h = p.base, p.src, p.hyb
        assert b.parent[p.mid] == b.root and b.parent[p.child] == p.mid
        order = [r.subj for r in b.relations]
        assert order.index(p.mid) < order.index(p.child)  # M's sentence comes first
        assert aligned(build_prompt(b, None, m.tok, m.family), build_prompt(s, None, m.tok, m.family), p.mid)
        d = lambda sc: tuple(np.subtract(sc.coords[p.child], sc.coords[p.mid]))
        assert d(b) == d(s) and s.coords[p.mid] != b.coords[p.mid]
        assert h.coords[p.child] == b.coords[p.child] and h.coords[p.mid] == s.coords[p.mid]
        assert p.qs


def test_layer0_patch_equals_source_run(m, pairs):
    df = run_chain_pair(m, pairs[0], layers=[0, 2])
    src = df[df.site == "source"].set_index("qid").model_ans
    p0 = df[(df.site == "patched") & (df.layer == 0)].set_index("qid").model_ans
    assert (p0 == src.reindex(p0.index)).all()
    assert {"none", "source", "hybrid", "patched"} == set(df.site)


def _synthetic():
    rows = []
    for pair in range(10):
        # q0: an M-question (base != src, src == hyb); q1: a C-question (src != hyb)
        clean = {"none": ("a", "x"), "source": ("b", "y"), "hybrid": ("b", "z")}
        for site, (a0, a1) in clean.items():
            rows += [dict(pair=pair, site=site, layer=-1, qid=0, model_ans=a0),
                     dict(pair=pair, site=site, layer=-1, qid=1, model_ans=a1)]
        rows += [dict(pair=pair, site="patched", layer=0, qid=0, model_ans="b"),
                 dict(pair=pair, site="patched", layer=0, qid=1, model_ans="y"),
                 dict(pair=pair, site="patched", layer=20, qid=0, model_ans="b"),
                 dict(pair=pair, site="patched", layer=20, qid=1, model_ans="z")]
    return pd.DataFrame(rows)


def test_summarize_rates():
    s = summarize(_synthetic(), n_boot=50).set_index("layer")
    assert s.loc[0, "follow_M"] == 1 and s.loc[0, "follow_C"] == 1 and s.loc[0, "stay_C"] == 0
    assert s.loc[20, "follow_M"] == 1 and s.loc[20, "follow_C"] == 0 and s.loc[20, "stay_C"] == 1
    assert s.loc[20, "n_C"] == 10 and s.loc[20, "n_M"] == 10


def test_verdict_rules():
    eager = pd.DataFrame(dict(layer=[0, 10, 20], follow_M=[1, 0.9, 0.5], follow_C=[1, 0.2, 0.1], stay_C=[0, 0.7, 0.8]))
    lazy = pd.DataFrame(dict(layer=[0, 10, 20], follow_M=[1, 0.9, 0.5], follow_C=[1, 0.85, 0.1], stay_C=[0, 0.1, 0.8]))
    mixed = pd.DataFrame(dict(layer=[0, 10], follow_M=[1, 0.9], follow_C=[1, 0.5], stay_C=[0, 0.3]))
    assert verdict(eager)[0] == "eager" and verdict(eager)[1] == 10
    assert verdict(lazy)[0] == "lazy"
    assert verdict(mixed)[0] == "mixed"


def test_summarize_strict_drops_c_questions_unchanged_by_the_move():
    df = _synthetic()
    # add a C-question whose base answer equals the source answer (C moved with M, relation to M unchanged)
    extra = []
    for pair in range(10):
        for site, ans in (("none", "q"), ("source", "q"), ("hybrid", "w")):
            extra.append(dict(pair=pair, site=site, layer=-1, qid=2, model_ans=ans))
        extra += [dict(pair=pair, site="patched", layer=0, qid=2, model_ans="q"),
                  dict(pair=pair, site="patched", layer=20, qid=2, model_ans="q")]
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    loose = summarize(df, n_boot=20).set_index("layer")
    strict = summarize(df, n_boot=20, strict=True).set_index("layer")
    assert loose.loc[20, "n_C"] == 20 and loose.loc[20, "follow_C"] == 0.5  # the unchanged question "follows"
    assert strict.loc[20, "n_C"] == 10 and strict.loc[20, "follow_C"] == 0 and strict.loc[20, "stay_C"] == 1
