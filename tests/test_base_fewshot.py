import numpy as np
import pytest

from geomap.axis import CELLS, axis_bulk_items
from geomap.baseplain import (SLOTS, example_text, overall_fewshot, pick_shots, shot_bank, shots_text,
                              verdict_base_plain)
from geomap.flow import axis_of, steps
from geomap.models import load_tiny_for_tests
from geomap.prompts import build_prompt
from geomap.scenes import relation_text


@pytest.fixture(scope="module")
def tiny():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def items(tiny):
    return axis_bulk_items(tiny, {c: 2 for c in CELLS}, seed=1601)[:4]


@pytest.fixture(scope="module")
def bank(tiny):
    return shot_bank(tiny, per_cell=200)


def test_example_text_is_the_zero_shot_item_followed_by_its_answer(tiny, items):
    s, mid, child, qq = items[0]
    q = qq[0][0]
    plain = tiny.tok.decode(build_prompt(s, q, tiny.tok, tiny.family, fmt="plain").input_ids)
    assert example_text(s, q, child) == plain + " " + child


def test_fewshot_prompt_is_examples_then_the_unchanged_test_item(tiny, items, bank):
    s, mid, child, qq = items[0]
    q = qq[0][0]
    shots = shots_text(pick_shots(bank, set(s.names), "informative", np.random.default_rng(0)))
    assert shots.endswith("\n\n") and shots.count("\nAnswer: ") == 6
    sp = build_prompt(s, q, tiny.tok, tiny.family, fmt="plain", shots=shots)
    zero = build_prompt(s, q, tiny.tok, tiny.family, fmt="plain")
    assert tiny.tok.decode(sp.input_ids) == shots + tiny.tok.decode(zero.input_ids)
    for r in s.relations:                                   # spans and positions point into the test item
        a, b = sp.relation_spans[r.subj]
        assert tiny.tok.decode(sp.input_ids[a:b]).strip() == relation_text(r)
    for n, p in sp.question_pos.items():
        assert tiny.tok.decode([sp.input_ids[p]]).strip() == n
    assert tiny.tok.decode([sp.input_ids[-1]]) == ":" and sp.lead == " "
    with pytest.raises(ValueError):
        build_prompt(s, q, tiny.tok, tiny.family, fmt="chat", shots=shots)


@pytest.mark.parametrize("arm", ["neutral", "informative"])
def test_pick_shots_fills_the_preregistered_slots(items, bank, arm):
    s = items[0][0]
    rng = lambda: np.random.default_rng(3)
    got = pick_shots(bank, set(s.names), arm, rng())
    assert [(e["cell"], e["ans"], e["axis"]) for e in sorted(got, key=lambda e: e["slot"])] == \
        [(c, a, i % 2) for i, (c, a) in enumerate(SLOTS[arm])]
    names = [set(e["scene"].names) for e in got]
    assert all(not (n & set(s.names)) for n in names)                          # no name shared with the test scene
    assert all(not (names[i] & names[j]) for i in range(6) for j in range(i))  # nor between examples
    for e in got:                                                              # the shown answer is correct
        d1, d2, truth = steps(e["scene"], e["mid"], e["child"], e["q"])
        assert e["answer"] == (e["child"] if truth == 1 else e["scene"].root)
        assert axis_of(e["q"]) == e["axis"]
    assert [e["slot"] for e in got] == [e["slot"] for e in pick_shots(bank, set(s.names), arm, rng())]
    assert sorted(e["slot"] for e in got) == list(range(6))


def test_slot_rules_as_preregistered(bank):
    for arm, c_fails, m_fails in (("neutral", 0, None), ("informative", 3, 2)):
        got = pick_shots(bank, set(), arm, np.random.default_rng(0))
        tr = np.array([steps(e["scene"], e["mid"], e["child"], e["q"]) for e in got])
        c_rule, m_rule = np.where(tr[:, 1] > 0, 1, -1), np.where(tr[:, 0] > 0, 1, -1)
        assert int(np.sum(c_rule != tr[:, 2])) == c_fails
        if m_fails is not None:
            assert int(np.sum(m_rule != tr[:, 2])) == m_fails


def test_bank_shares_no_scene_with_the_test_set(tiny, bank):
    test = {tuple(relation_text(r) for r in s.relations)
            for s, *_ in axis_bulk_items(tiny, {c: 30 for c in CELLS}, seed=1601)}
    assert not test & {tuple(relation_text(r) for r in e["scene"].relations) for e in bank}


OK = dict(own=(0.95, 0.9, 0.98), r2_d1=0.8, r2_d2=0.9, probe=(0.9, 0.8, 0.95), model=(0.3, 0.2, 0.4))


def test_composed_position_also_needs_conflict_c_larger():
    v = lambda cc: verdict_base_plain(dec=(0.8, 0.7, 0.9), first=(0.8, 0.7, 0.9), conflict_c=cc, **OK)
    assert v((0.8, 0.7, 0.9)) == "reads the composed position"
    assert v((0.3, 0.2, 0.4)).startswith("partial")          # "follow M's sentence" profile
    assert verdict_base_plain(dec=(0.8, 0.7, 0.9), first=(0.8, 0.7, 0.9), **OK) == "reads the composed position"


def test_overall_fewshot():
    out, reads, uninf = "the map is outvoted", "reads the composed position", "uninformative"
    assert overall_fewshot({"a": out, "b": out, "c": uninf}, "neutral") == "the zero-shot format did not hide map use"
    assert overall_fewshot({"a": reads, "b": reads}, "neutral") == "the zero-shot format hid map use"
    assert overall_fewshot({"a": reads, "b": reads}, "informative") == \
        "examples that need M's step make the answer read the composed position"
    assert overall_fewshot({"a": out}, "informative") == "the map stays outvoted even with such examples"
    assert overall_fewshot({"a": reads, "b": out}, "informative").startswith("mixed")
    assert overall_fewshot({"a": uninf}, "neutral") == "uninformative (no informative base model)"
