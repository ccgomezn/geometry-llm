import pytest

from geomap.axis import CELLS, axis_bulk_items
from geomap.cot_axis import (COT_AXIS_SUFFIX, cot_axis_prompt, named_coords, parse_final_name, select_items,
                             verdict_cot_axis, written_side)
from geomap.models import load_tiny_for_tests
from geomap.prompts import ANSWER_SUFFIX, build_prompt


@pytest.fixture(scope="module")
def tiny():
    return load_tiny_for_tests()


def test_prompt_differs_from_one_word_only_in_the_instruction(tiny):
    items = axis_bulk_items(tiny, {c: 2 for c in CELLS}, seed=1601)
    s, mid, child, qq = items[0]
    q = qq[0][0]
    one = tiny.tok.decode(build_prompt(s, q, tiny.tok, tiny.family).input_ids)
    cot = cot_axis_prompt(s, q, tiny.tok, tiny.family)
    assert one.replace(ANSWER_SUFFIX, COT_AXIS_SUFFIX) == cot
    assert COT_AXIS_SUFFIX in cot and ANSWER_SUFFIX not in cot


def test_select_items_takes_the_first_of_each_kind_in_order(tiny):
    items = axis_bulk_items(tiny, {c: 3 for c in CELLS}, seed=1601)
    sel = select_items(items, {c: 2 for c in CELLS})
    assert sorted(c for *_, c in sel) == sorted([c for c in CELLS for _ in range(2)])
    flat = [(si, q.text) for si, (_, _, _, qq) in enumerate(items) for q, _ in qq]
    picked = [(si, q.text) for si, q, *_ in sel]
    assert picked == sorted(picked, key=flat.index)


def test_parse_final_name():
    names = ("Leo", "Amy")
    assert parse_final_name("So Leo is farther east.\nAnswer: Leo", names) == "Leo"
    assert parse_final_name("Answer: Amy ... wait.\n**Answer:** **leo**", names) == "Leo"
    assert parse_final_name("Answer: Beth", names) is None
    assert parse_final_name("Leo is farther east.", names) is None


def test_parse_final_name_after_a_final_answer_heading():
    # Qwen3 ends with a heading, then the answer line; the heading's "Answer:" must not swallow the next "Answer"
    names = ("Anna", "Leo")
    assert parse_final_name("So Anna is north.\n\n### Final Answer:\nAnswer: Anna", names) == "Anna"
    assert parse_final_name("---\n\n### Final Answer:\n\nAnswer: Leo", names) == "Leo"
    assert parse_final_name("**Final Answer:**\n**Answer:** Leo", names) == "Leo"


def test_named_coords_takes_the_last_position_given_to_each_name():
    text = ("Put Amy at (0, 0). Beth is 2 units east of Amy, so Beth = (2, 0). Leo is 3 units west of Beth: "
            "Leo is at (-1, 0). Correction: Leo: (-1, 1).")
    c = named_coords(text, ("Leo", "Amy", "Beth"))
    assert c == {"Amy": (0.0, 0.0), "Beth": (2.0, 0.0), "Leo": (-1.0, 1.0)}


def test_written_side():
    coords = {"Leo": (-1.0, 1.0), "Amy": (0.0, 0.0)}
    assert written_side(coords, "Leo", "Amy", axis=0) == -1      # Leo west of Amy
    assert written_side(coords, "Leo", "Amy", axis=1) == 1       # Leo north of Amy
    assert written_side({"Leo": (0.0, 0.0)}, "Leo", "Amy", axis=0) is None


def test_verdict_rules():
    ok = dict(dec=(0.85, 0.79, 0.90), diff=(0.5, 0.4, 0.6), others=0.9)
    assert verdict_cot_axis(**ok) == "reasoning fixes the rule"
    assert verdict_cot_axis(**{**ok, "others": 0.7}) == "reasoning helps"
    assert verdict_cot_axis(**{**ok, "dec": (0.70, 0.62, 0.78)}) == "reasoning helps"
    assert verdict_cot_axis(**{**ok, "dec": (0.55, 0.45, 0.65), "diff": (0.1, -0.02, 0.2)}) == "no help"


def test_prediction_label_treats_missing_answers_as_none():
    import math
    from geomap.cot_axis import prediction_label
    assert prediction_label("Leo", "Leo") == "C"
    assert prediction_label("Amy", "Leo") == "R"
    assert prediction_label(None, "Leo") == "none"
    assert prediction_label(math.nan, "Leo") == "none"       # pandas' string columns turn None into NaN
