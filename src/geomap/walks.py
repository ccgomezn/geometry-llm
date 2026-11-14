"""Walks on a grid of city blocks that come back to an earlier landmark (docs/preregistration_walk_pilot.md).

No sentence states where the walk ends: the answer needs every move since the landmark was seen. A 2-move loop
("one block out, one block back") is the shortcut-sufficient control."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from geomap.analysis import bootstrap_ci
from geomap.prompts import ANSWER_SUFFIX, render_chat

MOVES = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
DIRS = list(MOVES)

# Concrete things you could pass in a street; filtered per tokenizer by landmark_pool.
LANDMARKS = ["clock", "bell", "tree", "chair", "table", "lamp", "rock", "box", "book", "ring", "coin", "shell",
             "stone", "car", "ball", "hat", "pen", "bag", "desk", "door", "bench", "statue", "fountain", "boat",
             "cart", "gate", "flag", "tower", "bridge", "truck", "bike", "cup", "key", "shoe", "sign", "van",
             "fence", "barrel", "bus", "piano", "sofa", "bed", "anchor", "basket", "bucket", "mirror", "kite"]

INTRO = "You are walking on a grid of city blocks. Each move takes you one block north, south, east or west."
QUESTION = "You are now back at one of the objects you saw. Which object is it?"
PREFILL_ASK = ("Describe a walk on a grid of city blocks, one block north, south, east or west per move. Mention the "
               "objects you pass, and end back at one of them.")
PREFILL_END = "I am now back at the"
BASELINES = ["recency", "primacy", "rank 1", "rank 2", "rank 3", "rank 4", "parity guess"]
FORMATS = ("user", "prefill")


@dataclass
class Walk:
    moves: list[str]           # one per move
    landmarks: dict[int, str]  # step (number of moves made so far) -> object seen there
    target: str                # the object the walk ends back at
    loop: int                  # moves since the target was seen

    @property
    def n(self) -> int:
        return len(self.moves)

    @property
    def target_step(self) -> int:
        return self.n - self.loop


@dataclass
class WalkPrompt:
    input_ids: list[int]
    candidates: list[str]
    answer: str


def positions(moves: list[str]) -> list[tuple[int, int]]:
    p = [(0, 0)]
    for mv in moves:
        dx, dy = MOVES[mv]
        p.append((p[-1][0] + dx, p[-1][1] + dy))
    return p


def _free_moves(rng, k: int, prev: str | None) -> list[str]:
    out = []
    for _ in range(k):
        options = [d for d in DIRS if prev is None or d != OPPOSITE[prev]]
        prev = options[rng.integers(len(options))]
        out.append(prev)
    return out


def _closed_loop(rng, loop: int, prev: str | None, max_tries: int = 5000) -> list[str] | None:
    """`loop` moves that end where they start without touching it earlier and never undo the move before (for
    loop = 2: one block out and straight back)."""
    if loop == 2:
        d = _free_moves(rng, 1, prev)[0]
        return [d, OPPOSITE[d]]
    half = loop // 2
    for _ in range(max_tries):
        a = int(rng.integers(1, half))  # north-south pairs; the rest are east-west pairs
        seq = ["north"] * a + ["south"] * a + ["east"] * (half - a) + ["west"] * (half - a)
        rng.shuffle(seq)
        if any(seq[i] == OPPOSITE[seq[i - 1]] for i in range(1, loop)) or (prev and seq[0] == OPPOSITE[prev]):
            continue
        if (0, 0) in positions(seq)[1:-1]:
            continue
        return seq
    return None


def allowed_ranks(n: int, loop: int, k: int = 4) -> list[int]:
    """Places the target can take among the k landmarks (1 = first seen). Landmarks sit an even number of moves
    before the end; for loop >= 4 at least one is seen inside the loop, for loop = 2 none is."""
    before, inside = (n - loop) // 2, loop // 2 - 1
    return [r for r in range(1, k + 1) if r - 1 <= before and k - r <= inside
            and (k - r == 0 if loop == 2 else k - r >= 1)]


def generate_walk(rng, names, n: int, loop: int, k: int = 4, max_tries: int = 20000) -> Walk:
    if loop < 2 or loop % 2 or loop > n:
        raise ValueError(f"loop must be even, >= 2 and <= n (got {loop}, n={n})")
    t = n - loop
    r = int(rng.choice(allowed_ranks(n, loop, k)))  # drawn before the walk, so no rank rule beats 1/len(allowed)
    for _ in range(max_tries):
        prefix = _free_moves(rng, t, None)
        tail = _closed_loop(rng, loop, prefix[-1] if prefix else None)
        if tail is None:
            continue
        moves = prefix + tail
        pos = positions(moves)
        if [j for j, p in enumerate(pos) if p == pos[t]] != [t, n]:
            continue
        seen = Counter(pos)
        # landmarks only an even number of moves before the end, so move-count parity rules nothing out
        free = [s for s in range(n) if s != t and seen[pos[s]] == 1 and (n - s) % 2 == 0]
        before, after = [s for s in free if s < t], [s for s in free if s > t]
        if r - 1 > len(before) or k - r > len(after):
            continue
        steps = [t]
        if r > 1:
            steps += [int(s) for s in rng.choice(before, r - 1, replace=False)]
        if k > r:
            steps += [int(s) for s in rng.choice(after, k - r, replace=False)]
        picked = [str(x) for x in rng.choice(list(names), k, replace=False)]
        marks = dict(zip(sorted(steps), picked))
        return Walk(moves=moves, landmarks=marks, target=marks[t], loop=loop)
    raise ValueError(f"no walk found for n={n}, loop={loop}")


def walk_items(names, ns=(14, 20, 26), loops=(2, 6, 8, 12), per_cell: int = 200, seed: int = 1401) -> list[Walk]:
    rng = np.random.default_rng(seed)
    return [generate_walk(rng, names, n, loop) for n in ns for loop in loops for _ in range(per_cell)]


def _a(x: str) -> str:
    return ("an " if x[0] in "aeiou" else "a ") + x


def walk_text(w: Walk, person: str = "you") -> str:
    P = {"you": "You", "I": "I"}[person]
    out = [f"{P} start next to {_a(w.landmarks[0])}." if 0 in w.landmarks else f"{P} start at a street corner."]
    for i, mv in enumerate(w.moves, 1):
        out.append(f"{P} walk one block {mv}.")
        if i in w.landmarks:
            out.append(f"There is {_a(w.landmarks[i])} here.")
    return " ".join(out)


def build_walk_prompt(w: Walk, tok, family: str, fmt: str = "user") -> WalkPrompt:
    seen = [w.landmarks[s] for s in sorted(w.landmarks)]
    if fmt == "user":
        text = render_chat(tok, family, f"{INTRO}\n\n{walk_text(w)}\n\n{QUESTION}{ANSWER_SUFFIX}")
        cands, ans = seen, w.target
    elif fmt == "prefill":
        text = render_chat(tok, family, PREFILL_ASK) + f"{walk_text(w, person='I')}\n\n{PREFILL_END}"
        cands, ans = [" " + x for x in seen], " " + w.target
    else:
        raise ValueError(fmt)
    return WalkPrompt(input_ids=list(tok(text, add_special_tokens=False)["input_ids"]), candidates=cands, answer=ans)


def landmark_pool(tok) -> list[str]:
    return [x for x in LANDMARKS if len(tok.encode(" " + x, add_special_tokens=False)) == 1
            and len(tok.encode(x, add_special_tokens=False)) == 1]


def baselines(w: Walk) -> dict[str, float]:
    """Expected correctness of each map-free rule on this walk."""
    order = sorted(w.landmarks)
    t = w.target_step
    out = {"recency": float(order[-1] == t), "primacy": float(order[0] == t)}
    out.update({f"rank {k}": float(len(order) >= k and order[k - 1] == t) for k in range(1, 5)})
    even = [s for s in order if (w.n - s) % 2 == 0]
    out["parity guess"] = 1.0 / len(even) if t in even else 0.0
    return out


def walk_summary(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, floor: float = 0.60) -> pd.DataFrame:
    """Per format and cell: accuracy, the best map-free baseline, and model − that baseline (paired over walks)."""
    rows = []
    for (fmt, n, loop), d in df.groupby(["fmt", "n", "loop"], sort=True):
        acc, lo, hi = bootstrap_ci(d.correct.values, n_boot=n_boot, seed=seed)
        base = {k: float(d[f"b_{k}"].mean()) for k in BASELINES}
        best = max(base, key=base.get)
        gap, glo, ghi = bootstrap_ci(d.correct.values - d[f"b_{best}"].values, n_boot=n_boot, seed=seed)
        rows.append(dict(fmt=fmt, n=int(n), loop=int(loop), n_walks=len(d), acc=acc, lo=lo, hi=hi,
                         best_baseline=best, base_acc=base[best], gap=gap, gap_lo=glo, gap_hi=ghi,
                         mass=float(d.mass.mean()), passes=bool(acc >= floor and glo > 0),
                         **{f"b_{k}": v for k, v in base.items()}))
    return pd.DataFrame(rows)


def verdict_walk_pilot(s: pd.DataFrame) -> tuple[str, dict | None]:
    long = s[s.loop >= 4]
    for fmt in FORMATS:
        ok = long[(long.fmt == fmt) & long.passes]
        if len(ok):
            best = ok.sort_values(["loop", "n"]).iloc[-1]
            return "go", dict(fmt=fmt, n=int(best.n), loop=int(best.loop))
    if (long.gap_lo > 0).any():
        return "weak pass", None
    return "fail", None
