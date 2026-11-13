"""Study 2, step-by-step reasoning (docs/preregistration_s2_reasoning.md): the pilot's rung-4 prompt with the one-word
instruction replaced by "think step by step, then give a final answer line", and the helpers to read the reply."""

from __future__ import annotations

import math
import re

import pandas as pd

from geomap.prompts import relation_text, render_chat
from geomap.scenes import Scene

COT_SUFFIX = (" Think it through step by step, then end your reply with a final line of the form "
              "'Answer: yes' or 'Answer: no'.")
_FINAL = re.compile(r"answer\s*\**\s*:\s*\**\s*(yes|no)\b", re.IGNORECASE)
_COORD = re.compile(r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)")


def cot_prompt_text(scene: Scene, question, tok, family: str) -> str:
    """The chat-rendered prompt: identical to build_prompt's, except for the instruction after the question."""
    content = " ".join(relation_text(r) for r in scene.relations)
    content += "\n\nThe objects are: " + ", ".join(scene.names) + "."
    content += "\n\n" + question.text + COT_SUFFIX
    return render_chat(tok, family, content)


def parse_final_answer(text: str) -> str | None:
    """The last 'Answer: yes/no' in the reply (markdown bold allowed), lower-cased; None if there is none."""
    hits = _FINAL.findall(text or "")
    return hits[-1].lower() if hits else None


def mentions_coordinates(text: str) -> bool:
    """Does the reply write at least one (x, y) pair? Exploratory: did the model build explicit coordinates?"""
    return bool(_COORD.search(text or ""))


def verdict_s2_reasoning(summary: pd.DataFrame, floor: float = 0.65) -> str:
    """docs/preregistration_s2_reasoning.md, hard subset (inside vs box_hard)."""
    h = summary[summary.subset == "hard"].iloc[0]
    lo = lambda c: float(h[c]) if not math.isnan(float(h[c])) else -math.inf
    if h.reasoning_bal_acc >= floor and lo("reasoning_bal_acc_lo") > 0.5 and lo("reasoning_minus_magnitude_lo") > 0:
        return "step-by-step reasoning can tell inside"
    if lo("reasoning_bal_acc_lo") > 0.5 and lo("reasoning_minus_plain_lo") > 0:
        return "partial"
    return "no"
