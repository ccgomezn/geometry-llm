import numpy as np
import pandas as pd

from geomap.models import load_tiny_for_tests
from geomap.prompts import ANSWER_SUFFIX
from geomap.reasoning import (COT_SUFFIX, cot_prompt_text, mentions_coordinates, parse_final_answer,
                              verdict_s2_reasoning)
from geomap.triangles import generate_tri_scene, ladder

POOL = ["Ann", "Bo", "Cy", "Dee", "Eli", "Fay", "Gus", "Hal"]


def test_parse_final_answer():
    assert parse_final_answer("P is left of AB...\nAnswer: yes") == "yes"
    assert parse_final_answer("**Answer:** No") == "no"
    assert parse_final_answer("answer: YES.") == "yes"
    assert parse_final_answer("First guess Answer: yes. Wait, recheck.\nAnswer: no") == "no"  # the last one counts
    assert parse_final_answer("I think it is inside.") is None
    assert parse_final_answer("Answer: maybe") is None
    assert parse_final_answer("") is None


def test_cot_prompt_matches_pilot_except_suffix():
    m = load_tiny_for_tests()
    rng = np.random.default_rng(3)
    ts = generate_tri_scene(rng, POOL)
    q = next(it for it in ladder(ts, rng) if it.rung == 4).question
    text = cot_prompt_text(ts.scene, q, m.tok, m.family)
    assert COT_SUFFIX in text and ANSWER_SUFFIX not in text
    assert q.text in text and "The objects are: " + ", ".join(ts.scene.names) + "." in text
    # identical to the pilot prompt apart from the instruction
    from geomap.prompts import build_prompt
    pilot = m.tok.decode(build_prompt(ts.scene, q, m.tok, m.family).input_ids)
    assert pilot.replace(ANSWER_SUFFIX, COT_SUFFIX) == m.tok.decode(m.tok(text, add_special_tokens=False).input_ids)


def test_mentions_coordinates():
    assert mentions_coordinates("Put A at (0, 0) and B at (3,-2).")
    assert not mentions_coordinates("B is east of A, so P is inside.")


def _summary(ba, lo, mag_lo, nr_lo):
    return pd.DataFrame([dict(subset="hard", reasoning_bal_acc=ba, reasoning_bal_acc_lo=lo,
                              reasoning_minus_magnitude_lo=mag_lo, reasoning_minus_plain_lo=nr_lo)])


def test_verdict_s2_reasoning():
    assert verdict_s2_reasoning(_summary(0.80, 0.75, 0.10, 0.20)) == "step-by-step reasoning can tell inside"
    assert verdict_s2_reasoning(_summary(0.62, 0.55, -0.02, 0.03)) == "partial"  # helps, but not past the size rule
    assert verdict_s2_reasoning(_summary(0.70, 0.62, 0.01, -0.01)) == "step-by-step reasoning can tell inside"
    assert verdict_s2_reasoning(_summary(0.55, 0.49, -0.10, -0.05)) == "no"
    assert verdict_s2_reasoning(_summary(0.60, 0.52, -0.05, -0.01)) == "no"  # above chance but no better than plain
