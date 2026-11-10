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


def test_make_pairs(m):
    from geomap.patching import make_pairs
    pairs = make_pairs(m, 5, seed=0)
    assert len(pairs) == 5
    for base, src, leaf, qs in pairs:
        diag, ctrl = split_questions(qs, base.coords, src.coords)
        assert len(diag) >= 1 and len(diag) + len(ctrl) == len(qs)


def test_make_pairs_object_count_and_exclusion(m):
    from geomap.patching import make_pairs
    pairs = make_pairs(m, 6, seed=1, n_objects=(3,), exclude_qtypes=("closer_than",))
    for base, src, leaf, qs in pairs:
        assert len(base.names) == 3
        assert all(q.qtype.value != "closer_than" for q in qs)


def test_model_relative_joint_iia():
    def r(pair, site, layer, qid, ans, b="A", s="B"):
        return dict(pair=pair, site=site, layer=layer, qtype="farthest", qid=qid, diagnostic=b != s,
                    base_ans=b, src_ans=s, model_ans=ans)
    rows = []
    # pair 0: model is wrong on the clean base for q1 (geometry says A, model says C); patch reproduces the
    # model's own source answers on both questions -> joint IIA 1 even though geometry-joint would fail
    rows += [r(0, "none", -1, 0, "A"), r(0, "source", -1, 0, "B"), r(0, "list", 5, 0, "B"),
             r(0, "none", -1, 1, "C", "A", "A"), r(0, "source", -1, 1, "C", "A", "A"), r(0, "list", 5, 1, "C", "A", "A")]
    # pair 1: patch fails to move q0
    rows += [r(1, "none", -1, 0, "A"), r(1, "source", -1, 0, "B"), r(1, "list", 5, 0, "A")]
    # pair 2: model gives the same answer on base and source for every question -> dropped
    rows += [r(2, "none", -1, 0, "A"), r(2, "source", -1, 0, "A"), r(2, "list", 5, 0, "A")]
    c = consistency(pd.DataFrame(rows), n_boot=20).iloc[0]
    assert c.joint_iia == 0.5 and c.n_pairs_iia == 2 and c.iia_rate == 0.5


def test_site_all_but_readout(m, pair):
    # necessity test: copy the whole description except every object's list position and the end of the list
    base, _, leaf, _ = pair
    spec = build_prompt(base, None, m.tok, m.family)
    pos = site_positions(spec, "all_but_readout", leaf)
    readout = set(spec.list_pos.values()) | {spec.eol_pos}
    assert set(pos) == set(range(spec.prefix_len)) - readout
    assert pos == sorted(pos)


def test_sentence_end_sites(m, pair):
    base, _, leaf, _ = pair
    spec = build_prompt(base, None, m.tok, m.family)
    lo, hi = spec.relation_spans[leaf]
    assert site_positions(spec, "sent_end", leaf) == [hi - 1]
    assert site_positions(spec, "sent_rest", leaf) == list(range(lo, hi - 1))


def test_question_patch_at_embeddings_is_identity(m, pair):
    # the question text is identical in base and source, so at layer 0 copying it changes nothing
    from geomap.patching import run_pair_question
    base, src, leaf, qs = pair
    d = run_pair_question(m, base, src, leaf, qs, layers=[0, 1])
    clean = d[d.site == "none"].set_index("qid")
    for site in ("question", "question_last"):
        p0 = d[(d.site == site) & (d.layer == 0)].set_index("qid")
        assert np.allclose(p0.lp_base_ans, clean.lp_base_ans.reindex(p0.index), atol=1e-4)
    assert {"none", "source", "question", "question_last"} == set(d.site)


def test_question_positions(m, pair):
    from geomap.patching import question_positions
    base, _, _, qs = pair
    spec = build_prompt(base, qs[0], m.tok, m.family)
    assert question_positions(spec, "question") == list(range(spec.prefix_len, len(spec.input_ids)))
    assert question_positions(spec, "question_last") == [len(spec.input_ids) - 1]


def test_mention_site_is_the_leafs_token_in_the_question(m, pair):
    from geomap.patching import question_positions
    base, _, leaf, qs = pair
    for q in qs:
        spec = build_prompt(base, q, m.tok, m.family)
        pos = question_positions(spec, "mention", leaf)
        assert pos == ([spec.question_pos[leaf]] if leaf in q.args else [])


def test_mention_rows_only_for_questions_naming_the_leaf(m, pair):
    from geomap.patching import run_pair_question
    base, src, leaf, qs = pair
    d = run_pair_question(m, base, src, leaf, qs, layers=[1], sites=("mention",))
    named = {i for i, q in enumerate(qs) if leaf in q.args}
    assert set(d[d.site == "mention"].qid) == named
