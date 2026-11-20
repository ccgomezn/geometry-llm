"""Order and voice (docs/preregistration_order_voice.md): the same R -> M -> C scene rendered with C's sentence last or
first, and with C as the subject ("C is d east of M") or the object ("M is d west of C"), to tell apart three rules
that fit the earlier data: C's own step, the direction word in C's relation sentence, and the last sentence."""

from dataclasses import replace

import numpy as np

from geomap.scenes import Relation, Scene

RENDERINGS = ("MC-subj", "CM-subj", "MC-obj", "CM-obj")
FEATURES = ("own", "word", "last")
_LABEL = {"own": "C's own step", "word": "the direction word", "last": "the last sentence"}


def render(s: Scene, mid: str, child: str, rendering: str) -> Scene:
    """The scene with its two sentences in the given order (MC or CM) and C's relation in the given voice."""
    order, voice = rendering.split("-")
    m_rel = next(r for r in s.relations if r.subj == mid)
    c_rel = next(r for r in s.relations if r.subj == child)
    if voice == "obj":
        c_rel = Relation(mid, child, -c_rel.dx, -c_rel.dy, c_rel.template_id)
    rels = (m_rel, c_rel) if order == "MC" else (c_rel, m_rel)
    return replace(s, relations=rels)


def features(d1: int, d2: int, rendering: str) -> tuple[int, int, int]:
    """(own_yes, word_yes, last_yes) for steps d1 (M from R) and d2 (C from M) on the asked axis, in the C frame
    (positive = the asked direction)."""
    order, voice = rendering.split("-")
    own = int(d2 > 0)
    word = int(d2 > 0) if voice == "subj" else int(d2 < 0)
    last = word if order == "MC" else int(d1 > 0)
    return own, word, last


def design(meta) -> np.ndarray:
    """[1, own_yes, word_yes, last_yes, d1, d2, name order (+1 C first), CM order, object voice] per row."""
    r = meta.rendering.astype(str)
    return np.c_[np.ones(len(meta)), meta.own_yes.to_numpy(float), meta.word_yes.to_numpy(float),
                 meta.last_yes.to_numpy(float), meta.d1.to_numpy(float), meta.d2.to_numpy(float),
                 np.where(meta.c_first.to_numpy(), 1.0, -1.0), r.str.startswith("CM").to_numpy(float),
                 r.str.endswith("obj").to_numpy(float)]


def verdict_order_voice(coef: dict, diff: dict) -> str:
    """coef[f] = (estimate, lo, hi) for f in own/word/last; diff[(f, g)] = CI of b_f - b_g (one order per pair).
    The rule follows X if b_X's CI is above 0 and every b_X - b_Y CI is above 0; otherwise 'mixed: ...' listing the
    features whose CI is above 0."""
    def above(f, g):
        if (f, g) in diff:
            return diff[(f, g)][1] > 0
        return diff[(g, f)][2] < 0
    for f in FEATURES:
        if coef[f][1] > 0 and all(above(f, g) for g in FEATURES if g != f):
            return f"the rule follows {_LABEL[f]}"
    pos = [_LABEL[f] for f in FEATURES if coef[f][1] > 0]
    return "mixed: " + (", ".join(pos) if pos else "no feature above 0")
