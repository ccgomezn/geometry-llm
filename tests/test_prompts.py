import numpy as np
import pytest
from transformers import AutoTokenizer

from geomap.prompts import NAME_POOLS, aligned, build_prompt, make_pair_aligned, single_token_names
from geomap.questions import QType, battery
from geomap.scenes import generate_scene, relation_text


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


@pytest.fixture(scope="module")
def names(tok):
    return single_token_names(tok, NAME_POOLS["people"])


def _scene(seed, names, n=4):
    return generate_scene(np.random.default_rng(seed), names, n)


def test_pools_are_single_token(tok):
    for pool in NAME_POOLS.values():
        assert len(pool) >= 30
        assert single_token_names(tok, pool) == pool


def test_prefix_matches_full_prompt(tok, names):
    for seed in range(50):
        s = _scene(seed, names, 3 + seed % 4)
        pre = build_prompt(s, None, tok, "qwen3")
        assert len(pre.input_ids) == pre.prefix_len
        for q in battery(s, np.random.default_rng(seed)):
            full = build_prompt(s, q, tok, "qwen3")
            assert full.input_ids[: pre.prefix_len] == pre.input_ids
            assert full.list_pos == pre.list_pos and full.eol_pos == pre.eol_pos


def test_list_pos_decodes_to_names(tok, names):
    s = _scene(1, names)
    spec = build_prompt(s, None, tok, "qwen3")
    assert set(spec.list_pos) == set(s.names)
    for n, i in spec.list_pos.items():
        assert tok.decode([spec.input_ids[i]]).strip() == n
    # list positions are in list order and after every relation sentence
    pos = [spec.list_pos[n] for n in s.names]
    assert pos == sorted(pos)
    assert min(pos) > max(e for _, e in spec.relation_spans.values())


def test_eol_is_period(tok, names):
    s = _scene(2, names)
    spec = build_prompt(s, None, tok, "qwen3")
    assert "." in tok.decode([spec.input_ids[spec.eol_pos]])
    assert spec.eol_pos > max(spec.list_pos.values())


def test_relation_span_contains_subject_and_object(tok, names):
    s = _scene(3, names, 5)
    spec = build_prompt(s, None, tok, "qwen3")
    for r in s.relations:
        a, b = spec.relation_spans[r.subj]
        text = tok.decode(spec.input_ids[a:b])
        assert r.subj in text and r.obj in text
        assert relation_text(r).rstrip(".") in text


def test_pair_prompts_aligned(tok, names):
    rng = np.random.default_rng(0)
    for seed in range(200):
        s = _scene(seed, names, 3 + seed % 4)
        base, src, leaf = make_pair_aligned(s, rng, tok, "qwen3")
        b = build_prompt(base, None, tok, "qwen3")
        c = build_prompt(src, None, tok, "qwen3")
        assert aligned(b, c, leaf)
        assert len(b.input_ids) == len(c.input_ids)
        lo, hi = b.relation_spans[leaf]
        assert b.input_ids[:lo] == c.input_ids[:lo] and b.input_ids[hi:] == c.input_ids[hi:]
        assert b.input_ids[lo:hi] != c.input_ids[lo:hi]


def test_question_text_in_prompt(tok, names):
    s = _scene(4, names)
    q = next(q for q in battery(s, np.random.default_rng(0)) if q.qtype == QType.DIRECTION)
    text = tok.decode(build_prompt(s, q, tok, "qwen3").input_ids)
    assert q.text in text and "Answer with one word." in text
    assert "The objects are: " + ", ".join(s.names) + "." in text


@pytest.mark.parametrize("family,repo", [("llama3", "meta-llama/Llama-3.2-3B-Instruct"),
                                         ("gemma3", "google/gemma-3-4b-it")])
def test_other_families_prefix_and_alignment(family, repo):
    t = AutoTokenizer.from_pretrained(repo)
    nm = NAME_POOLS["objects"]
    assert single_token_names(t, nm) == nm
    rng = np.random.default_rng(1)
    for seed in range(40):
        s = _scene(seed, nm, 3 + seed % 4)
        pre = build_prompt(s, None, t, family)
        for q in battery(s, np.random.default_rng(seed)):
            assert build_prompt(s, q, t, family).input_ids[: pre.prefix_len] == pre.input_ids
        for n, i in pre.list_pos.items():
            assert t.decode([pre.input_ids[i]]).strip() == n
        base, src, leaf = make_pair_aligned(s, rng, t, family)
        assert aligned(build_prompt(base, None, t, family), build_prompt(src, None, t, family), leaf)
