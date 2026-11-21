"""Step-by-step reasoning on the C–R axis questions (docs/preregistration_cot_axis.md): the flow prompt with the
one-word instruction replaced by a step-by-step one, the helpers that read the reply, and the preregistered rule."""

from __future__ import annotations

import re

from geomap.prompts import relation_text, render_chat

COT_AXIS_SUFFIX = (" Think it through step by step, then end your reply with a final line of the form 'Answer: X', "
                   "where X is one of the two names.")
# the name must be on the same line as its "Answer:" (so a "### Final Answer:" heading cannot take the next line's word)
_FINAL = re.compile(r"answer[^\S\n]*\**[^\S\n]*:[ \t*_]*([A-Za-z][A-Za-z-]*)", re.IGNORECASE)
_COORD = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
NAME_GAP = 25          # most characters between a name and the "(" of the pair it is given


def cot_axis_prompt(scene, question, tok, family: str) -> str:
    """The chat-rendered flow prompt (as build_prompt builds it) with the step-by-step instruction at the end."""
    content = " ".join(relation_text(r) for r in scene.relations)
    content += "\n\nThe objects are: " + ", ".join(scene.names) + "."
    content += "\n\n" + question.text + COT_AXIS_SUFFIX
    return render_chat(tok, family, content)


def select_items(items, sizes: dict) -> list[tuple]:
    """The first sizes[cell] questions of each kind, in generation order: (scene index, question, scene, mid, child,
    cell)."""
    left, out = dict(sizes), []
    for si, (s, mid, child, qq) in enumerate(items):
        for q, cell in qq:
            if left.get(cell, 0) > 0:
                out.append((si, q, s, mid, child, cell))
                left[cell] -= 1
    return out


def parse_final_name(text: str, names) -> str | None:
    """The name in the last 'Answer: …' line (markdown allowed), as spelled in `names`; None if the last answer names
    neither."""
    hits = _FINAL.findall(text or "")
    if not hits:
        return None
    low = {n.lower(): n for n in names}
    return low.get(hits[-1].lower())


def named_coords(text: str, names) -> dict[str, tuple[float, float]]:
    """Positions the reply gives to each name: every (x, y) pair is assigned to the nearest name written before it on
    the same line, at most NAME_GAP characters earlier; later pairs overwrite earlier ones."""
    text = text or ""
    mentions = sorted((m.end(), n) for n in names for m in re.finditer(rf"\b{re.escape(n)}\b", text))
    out = {}
    for c in _COORD.finditer(text):
        before = [(end, n) for end, n in mentions if end <= c.start()]
        if not before:
            continue
        end, n = before[-1]
        gap = text[end:c.start()]
        if len(gap) <= NAME_GAP and "\n" not in gap:
            out[n] = (float(c.group(1)), float(c.group(2)))
    return out


def written_side(coords: dict, child: str, root: str, axis: int) -> int | None:
    """Sign of C's written position minus R's on the asked axis (0 = east, 1 = north); None unless both are given."""
    if child not in coords or root not in coords:
        return None
    d = coords[child][axis] - coords[root][axis]
    return int(d > 0) - int(d < 0)


def verdict_cot_axis(dec: tuple, diff: tuple, others: float, floor: float = 0.80) -> str:
    """dec: step-by-step balanced accuracy on the decisive questions (est, lo, hi); diff: paired step by step − one
    word on them; others: step-by-step balanced accuracy on the other four kinds pooled."""
    if dec[0] >= floor and dec[1] > 0.5 and diff[1] > 0 and others >= floor:
        return "reasoning fixes the rule"
    if diff[1] > 0:
        return "reasoning helps"
    return "no help"


def prediction_label(final, child: str) -> str:
    """'C' or 'R' for a parsed final name, 'none' when there is none (None, or NaN once stored in a pandas column)."""
    if not isinstance(final, str):
        return "none"
    return "C" if final == child else "R"
