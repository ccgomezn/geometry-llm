import numpy as np
import pandas as pd
import pytest

from geomap.models import load_tiny_for_tests
from geomap.patching import consistency, run_pair, site_positions
from geomap.prompts import NAME_POOLS, build_prompt, make_pair_aligned
from geomap.questions import all_questions, split_questions
from geomap.scenes import generate_scene


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def pair(m):
    s = generate_scene(np.random.default_rng(4), NAME_POOLS["people"], 4)
    base, src, leaf = make_pair_aligned(s, np.random.default_rng(0), m.tok, m.family)
    diag, ctrl = split_questions(all_questions(base), base.coords, src.coords)
    return base, src, leaf, (diag[:3] + ctrl[:3])


def test_site_positions(m, pair):
    base, _, leaf, _ = pair
    spec = build_prompt(base, None, m.tok, m.family)
    (p,) = site_positions(spec, "list", leaf)
    assert m.tok.decode([spec.input_ids[p]]).strip() == leaf
    assert site_positions(spec, "relation", leaf) == list(range(*spec.relation_spans[leaf]))
    assert site_positions(spec, "eol", leaf) == [spec.eol_pos]
    assert site_positions(spec, "all", leaf) == list(range(spec.prefix_len))


def test_patch_with_self_is_identity(m, pair):
    base, _, leaf, qs = pair
    df = run_pair(m, base, base, leaf, qs, sites=["all", "list"], layers=[1, 2])
    none = df[df.site == "none"].set_index("qid")
    for _, r in df[df.site.isin(["all", "list"])].iterrows():
        assert abs(r.lp_src_ans - none.loc[r.qid, "lp_src_ans"]) < 1e-3


def test_patch_all_at_embeddings_equals_source_run(m, pair):
    base, src, leaf, qs = pair
    df = run_pair(m, base, src, leaf, qs, sites=["all"], layers=[0])
    clean_src = df[df.site == "source"].set_index("qid")
    for _, r in df[df.site == "all"].iterrows():
        assert abs(r.lp_src_ans - clean_src.loc[r.qid, "lp_src_ans"]) < 1e-3
        assert r.model_ans == clean_src.loc[r.qid, "model_ans"]


def test_joint_consistency_math():
    rows = []
    # pair 0: all diagnostics follow the source and the control is stable -> consistent
    rows += [dict(pair=0, site="list", layer=3, qtype="farthest", qid=0, diagnostic=True, base_ans="A", src_ans="B", model_ans="B"),
             dict(pair=0, site="list", layer=3, qtype="direction", qid=1, diagnostic=True, base_ans="north", src_ans="east", model_ans="east"),
             dict(pair=0, site="list", layer=3, qtype="more_north", qid=2, diagnostic=False, base_ans="C", src_ans="C", model_ans="C")]
    # pair 1: one diagnostic fails -> inconsistent
    rows += [dict(pair=1, site="list", layer=3, qtype="farthest", qid=0, diagnostic=True, base_ans="A", src_ans="B", model_ans="B"),
             dict(pair=1, site="list", layer=3, qtype="direction", qid=1, diagnostic=True, base_ans="north", src_ans="east", model_ans="north")]
    # pair 2: no diagnostic question -> dropped
    rows += [dict(pair=2, site="list", layer=3, qtype="farthest", qid=0, diagnostic=False, base_ans="A", src_ans="A", model_ans="A")]
    c = consistency(pd.DataFrame(rows), n_boot=50).iloc[0]
    assert c.joint_consistency == 0.5 and c.n_pairs == 2 and c.n_dropped == 1
    assert c.flip_rate == 0.75 and c.control_stability == 1.0
    assert c.flip_farthest == 1.0 and c.flip_direction == 0.5
