import math

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from geomap.axis import CELLS, axis_bulk_items
from geomap.baseplain import overall_base_plain, rule_answer, verdict_base_plain
from geomap.models import FAMILY, MODELS, QUANT, load_tiny_for_tests
from geomap.prompts import build_prompt
from geomap.scenes import relation_text
from geomap.scoring import candidate_sequences, score


@pytest.fixture(scope="module")
def tiny():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def items(tiny):
    return axis_bulk_items(tiny, {c: 2 for c in CELLS}, seed=1601)[:4]


def test_plain_prompt_is_scene_question_answer_without_chat_template(tiny, items):
    for s, mid, child, qq in items:
        q = qq[0][0]
        sp = build_prompt(s, q, tiny.tok, tiny.family, fmt="plain")
        text = tiny.tok.decode(sp.input_ids)
        assert "<|im_start|>" not in text and "Answer with one word" not in text
        assert text.endswith(f"\n\nQuestion: {q.text}\nAnswer:")
        assert text.startswith(relation_text(s.relations[0]))           # Qwen has no begin-of-text token
        for r in s.relations:
            a, b = sp.relation_spans[r.subj]
            assert tiny.tok.decode(sp.input_ids[a:b]).strip() == relation_text(r)
        for n, p in sp.list_pos.items():
            assert tiny.tok.decode([sp.input_ids[p]]).strip() == n
        for n, p in sp.question_pos.items():
            assert tiny.tok.decode([sp.input_ids[p]]).strip() == n
        assert sp.lead == " "
        chat = build_prompt(s, q, tiny.tok, tiny.family)
        assert chat.lead == ""


@pytest.mark.parametrize("repo,family", [("unsloth/gemma-3-12b-pt", "gemma3"), ("unsloth/Meta-Llama-3.1-8B", "llama3")])
def test_plain_prompt_starts_with_the_begin_of_text_token(tiny, items, repo, family):
    tok = AutoTokenizer.from_pretrained(repo)
    s, mid, child, qq = items[0]
    sp = build_prompt(s, qq[0][0], tok, family, fmt="plain")
    assert sp.input_ids[0] == tok.bos_token_id and sp.input_ids.count(tok.bos_token_id) == 1
    assert tok.decode(sp.input_ids[sp.question_pos[child]]).strip() == child


def test_plain_candidates_carry_the_leading_space(tiny):
    seqs = candidate_sequences(tiny.tok, tiny.family, ["Beth", "Kevin"], lead=" ")
    assert seqs == {"Beth": [tiny.tok.encode(" Beth", add_special_tokens=False)],
                    "Kevin": [tiny.tok.encode(" Kevin", add_special_tokens=False)]}
    assert candidate_sequences(tiny.tok, tiny.family, ["Beth"])["Beth"] == [tiny.tok.encode("Beth",
                                                                                           add_special_tokens=False)]


def test_score_on_plain_prompt_is_the_next_token_log_probability_of_the_spaced_name(tiny, items):
    s, mid, child, qq = items[0]
    q = qq[0][0]
    sp = build_prompt(s, q, tiny.tok, tiny.family, fmt="plain")
    got = score(tiny, [(sp, q)])[0]
    with torch.no_grad():
        lp = torch.log_softmax(tiny.hf(torch.tensor([sp.input_ids]).to(tiny.hf.device)).logits[0, -1].float(), -1)
    for c in q.candidates:
        assert got[c] == pytest.approx(float(lp[tiny.tok.encode(" " + c, add_special_tokens=False)[0]]), abs=1e-4)


def test_base_models_are_registered():
    assert MODELS["qwen3-8b-base"] == "Qwen/Qwen3-8B-Base" and FAMILY["qwen3-8b-base"] == "qwen3"
    assert MODELS["llama3.1-8b-base"] == "meta-llama/Llama-3.1-8B" and FAMILY["llama3.1-8b-base"] == "llama3"
    for k in ("gemma3-12b-pt", "gemma3-12b-pt-8bit"):
        assert MODELS[k] == "google/gemma-3-12b-pt" and FAMILY[k] == "gemma3"
    assert QUANT["gemma3-12b-pt-8bit"] == {"load_in_8bit": True} and "gemma3-12b-pt" not in QUANT


def test_rule_answer_follows_c_own_step():
    assert list(rule_answer(np.array([3, -2, 0]))) == [1, -1, -1]


OK = dict(own=(0.95, 0.9, 0.98), r2_d1=0.8, r2_d2=0.9, probe=(0.9, 0.8, 0.95), model=(0.3, 0.2, 0.4))


def test_verdict_base_plain():
    v = lambda dec, first, **kw: verdict_base_plain(dec=dec, first=first, **{**OK, **kw})
    assert v((0.8, 0.7, 0.9), (0.8, 0.7, 0.9)) == "reads the composed position"
    assert v((0.3, 0.2, 0.4), (0.5, 0.49, 0.51)) == "the map is outvoted"
    assert v((0.3, 0.2, 0.4), (0.5, 0.49, 0.51), own=(0.7, 0.6, 0.8)) == "uninformative"
    assert v((0.8, 0.7, 0.9), (0.55, 0.5, 0.6)).startswith("partial")                  # first-step bound not above 0.5
    assert v((0.3, 0.2, 0.4), (0.5, 0.49, 0.51), r2_d1=0.4) == "partial (M's step not readable at the answer)"
    assert v((0.3, 0.2, 0.4), (0.5, 0.49, 0.51), probe=(0.5, 0.39, 0.6)) == "partial (probe does not beat the model)"
    assert v((0.45, 0.4, 0.55), (0.5, 0.49, 0.51)) == "partial (decisive questions at chance)"
    assert v((math.nan,) * 3, (0.5, 0.49, 0.51)) == "uninformative"


def test_overall_base_plain():
    assert overall_base_plain({"a": "the map is outvoted", "b": "the map is outvoted", "c": "uninformative"}) == \
        "the failure comes from pretraining"
    assert overall_base_plain({"a": "reads the composed position", "b": "reads the composed position"}) == \
        "instruction tuning or the format brings the failure"
    assert overall_base_plain({"a": "reads the composed position", "b": "the map is outvoted"}).startswith("mixed")
    assert overall_base_plain({"a": "uninformative"}) == "uninformative (no informative base model)"
