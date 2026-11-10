import numpy as np
import pandas as pd
import pytest

from geomap.models import load_tiny_for_tests
from geomap.prompts import build_prompt
from geomap.questions import answer_for
from geomap.scenes import Relation, Scene, move_object
from geomap.transfer import make_transfer_pairs, run_transfer_pair, summarize_transfer, target_scene, verdict_transfer


def chain():
    rels = (Relation("M", "R", 0, 2, 0), Relation("C", "M", 3, 0, 1))
    return Scene(names=("C", "R", "M"), coords={"R": (0, 0), "M": (0, 2), "C": (3, 2)}, relations=rels,
                 parent={"R": None, "M": "R", "C": "M"}, hop={"R": 0, "M": 1, "C": 2}, seed=0)


def test_target_is_base_with_child_at_its_source_position():
    base = chain()
    src = move_object(base, "M", -1, 0)  # M's offset from R becomes (-1, 0): M -> (-1, 0), C -> (2, 0)
    t = target_scene(base, src, "C")
    assert t.coords == {"R": (0, 0), "M": (0, 2), "C": (2, 0)}
    rc = next(r for r in t.relations if r.subj == "C")
    assert (rc.dx, rc.dy) == (2, -2)


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def tps(m):
    return make_transfer_pairs(m, 4, seed=0)


def test_make_transfer_pairs_structure(m, tps):
    assert len(tps) == 4
    for p in tps:
        bp, sp = build_prompt(p.base, None, m.tok, m.family), build_prompt(p.src, None, m.tok, m.family)
        lo, hi = bp.relation_spans[p.child]
        assert bp.input_ids[lo:hi] == sp.input_ids[lo:hi]  # C's own sentence is word-for-word the same
        assert p.src.coords[p.child] != p.base.coords[p.child]  # ...but C sits elsewhere (it moved with M)
        assert p.tgt.coords[p.child] == p.src.coords[p.child] and p.tgt.coords[p.mid] == p.base.coords[p.mid]
        assert any(answer_for(q, p.tgt.coords) != answer_for(q, p.base.coords) for q in p.qs)


def test_layer0_patch_of_identical_sentence_changes_nothing(m, tps):
    df = run_transfer_pair(m, tps[0], layers=[0, 2])
    none = df[df.site == "none"].set_index("qid")
    p0 = df[(df.site == "patched") & (df.layer == 0)].set_index("qid")
    assert np.allclose(p0.lp_base_ans, none.lp_base_ans.reindex(p0.index), atol=1e-4)
    assert {"none", "source", "target", "patched"} == set(df.site)


def test_summarize_transfer_rates():
    rows = []
    for pair in range(10):
        # q0: target question (target answer differs from base); q1: control (same answer in base and target)
        clean = {"none": ("a", "c"), "source": ("s", "c"), "target": ("t", "c")}
        for site, (a0, a1) in clean.items():
            rows += [dict(pair=pair, site=site, layer=-1, qid=0, model_ans=a0),
                     dict(pair=pair, site=site, layer=-1, qid=1, model_ans=a1)]
        rows += [dict(pair=pair, site="patched", layer=0, qid=0, model_ans="a"),
                 dict(pair=pair, site="patched", layer=0, qid=1, model_ans="c"),
                 dict(pair=pair, site="patched", layer=20, qid=0, model_ans="t"),
                 dict(pair=pair, site="patched", layer=20, qid=1, model_ans="x")]
    s = summarize_transfer(pd.DataFrame(rows), n_boot=50).set_index("layer")
    assert s.loc[0, "follow_T"] == 0 and s.loc[0, "stable"] == 1
    assert s.loc[20, "follow_T"] == 1 and s.loc[20, "stable"] == 0 and s.loc[20, "follow_S"] == 0
    assert s.loc[20, "n_T"] == 10 and s.loc[20, "n_ctrl"] == 10


def test_verdict_transfer_rules():
    mk = lambda fT, st: pd.DataFrame(dict(layer=[0, 10, 20], follow_T=fT, stable=st))
    assert verdict_transfer(mk([0.0, 0.6, 0.3], [1.0, 0.95, 0.97])) == ("carried", 10)
    assert verdict_transfer(mk([0.0, 0.6, 0.3], [1.0, 0.80, 0.97]))[0] == "partial"  # moves other answers too
    assert verdict_transfer(mk([0.0, 0.05, 0.1], [1.0, 1.0, 1.0]))[0] == "not carried"
    assert verdict_transfer(mk([0.9, 0.05, 0.08], [1.0, 1.0, 1.0]))[0] == "not carried"  # layer 0 is not counted
    assert verdict_transfer(mk([0.0, 0.3, 0.2], [1.0, 1.0, 1.0]))[0] == "partial"
