"""Exploratory, defined after seeing the blind-reply data (docs/preregistration_blind_reply.md): in the step-by-step
replies, what do the models write for M's step on the asked axis? Direction: "M is ... east/west of R" (or "R is ... of
M", negated). Size: "M is N units east/west of R". Reads results/blind; no model.

On decisive questions M's step is always opposite to C's, so "opposite of C's direction" is readable from C's
sentence, which the blind reply still sees; the floor (same cue, nothing of M) is the baseline for direction."""

import re
import sys

import numpy as np
import pandas as pd

from geomap.report import ROOT

W = {0: {"east": 1, "west": -1}, 1: {"north": 1, "south": -1}}
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def m_direction(reply, mid, root, axis):
    w = "|".join(W[axis])
    for a, b, sgn in ((mid, root, 1), (root, mid, -1)):
        m = re.search(rf"\b{a}\b[^.\n]*?\b({w})\b[^.\n]*?\bof {b}\b", reply)
        if m:
            return sgn * W[axis][m.group(1)]
    return None


def m_size(reply, mid, root, axis):
    w = "|".join(W[axis])
    for a, b in ((mid, root), (root, mid)):
        m = re.search(rf"\b{a}\b[^.\n]*?\b(\d+|{'|'.join(NUM)})\s+units?\s+({w})\b[^.\n]*?\bof {b}\b", reply, re.I)
        if m:
            x = m.group(1).lower()
            return int(x) if x.isdigit() else NUM[x]
    return None


def bal(said, true):
    return float(np.mean([np.mean(said[true == t] == t) for t in (1, -1)]))


def main(folder=ROOT / "results" / "blind"):
    for model in ("qwen3-8b", "gemma3-12b-8bit"):
        run = f"{model}_blind_reply_seed1601"
        items = pd.read_parquet(folder / f"{run}_items.parquet")
        R = pd.read_parquet(folder / f"{run}_replies.parquet")
        D = R.merge(items.drop(columns=["ids"]).reset_index().rename(columns={"index": "i"}), on="i")
        print(f"== {model}")
        for cond in ("normal", "blind reply", "floor"):
            x = D[D.cond == cond]
            s = pd.DataFrame([(m_direction(r.reply, r.mid, r.root, int(r.axis)), int(np.sign(r.d1)), r.scene)
                              for r in x.itertuples()], columns=["said", "true", "scene"]).dropna()
            rng, sc = np.random.default_rng(0), s.scene.unique()
            g = {k: s[s.scene == k] for k in sc}
            bs = []
            for _ in range(1000):
                b = pd.concat([g[k] for k in rng.choice(sc, len(sc))])
                bs.append(bal(b.said.to_numpy(), b.true.to_numpy()))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            z = [(m_size(r.reply, r.mid, r.root, int(r.axis)), abs(r.d1), abs(r.d2)) for r in x.itertuples()]
            z = [t for t in z if t[0] is not None]
            print(f"  {cond:12s} direction given {len(s):3d}/200, balanced correct "
                  f"{bal(s.said.to_numpy(), s.true.to_numpy()):.2f} [{lo:.2f}, {hi:.2f}] | size given {len(z):3d}: "
                  f"= M's step {np.mean([a == b for a, b, _ in z]):.2f}, = C's step {np.mean([a == c for a, _, c in z]):.2f}"
                  f", larger than C's {np.mean([a > c for a, _, c in z]):.2f}" if z else "")


if __name__ == "__main__":
    main(*(map(__import__("pathlib").Path, sys.argv[1:])))
