from collections import Counter

import numpy as np
import pandas as pd
import pytest

from geomap.models import load_tiny_for_tests

STEP = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
NAMES = ["clock", "bell", "tree", "chair", "lamp", "anchor", "book", "ring"]


@pytest.fixture(scope="module")
def tok():
    return load_tiny_for_tests().tok


def _positions(moves):
    p = [(0, 0)]
    for mv in moves:
        p.append((p[-1][0] + STEP[mv][0], p[-1][1] + STEP[mv][1]))
    return p


@pytest.mark.parametrize("n,loop", [(14, 2), (14, 4), (14, 6), (20, 8), (26, 12), (14, 12)])
def test_generate_walk_returns_to_the_target_by_a_loop_of_the_asked_length(n, loop):
    from geomap.walks import generate_walk
    rng = np.random.default_rng(0)
    for _ in range(30):
        w = generate_walk(rng, NAMES, n, loop)
        pos = _positions(w.moves)
        t = n - loop
        assert len(w.moves) == n and w.loop == loop and len(w.landmarks) == 4
        assert w.landmarks[t] == w.target and pos[n] == pos[t]
        assert all(0 <= s < n and (n - s) % 2 == 0 for s in w.landmarks) and len(set(w.landmarks.values())) == 4
        for s in w.landmarks:   # a landmark's block is visited only at its own step (the target also at the end)
            visits = [j for j, p in enumerate(pos) if p == pos[s]]
            assert visits == ([s, n] if s == t else [s])
        rev = [i for i in range(1, n) if w.moves[i] == OPP[w.moves[i - 1]]]
        assert rev == ([n - 1] if loop == 2 else [])
        inside = [s for s in w.landmarks if t < s < n]
        assert (inside == []) if loop == 2 else (len(inside) >= 1)


def test_target_rank_is_spread_over_the_allowed_ranks():
    from geomap.walks import generate_walk
    rng = np.random.default_rng(1)
    ranks = Counter()
    for _ in range(300):
        w = generate_walk(rng, NAMES, 20, 8)
        ranks[sorted(w.landmarks).index(20 - 8) + 1] += 1
    assert set(ranks) == {1, 2, 3} and min(ranks.values()) > 60


def test_generate_walk_is_deterministic():
    from geomap.walks import generate_walk
    a = generate_walk(np.random.default_rng(7), NAMES, 20, 8)
    b = generate_walk(np.random.default_rng(7), NAMES, 20, 8)
    assert (a.moves, a.landmarks) == (b.moves, b.landmarks)


def test_walk_items_cover_every_cell():
    from geomap.walks import walk_items
    items = walk_items(NAMES, ns=(14, 20), loops=(2, 8), per_cell=3, seed=4)
    assert Counter((w.n, w.loop) for w in items) == {(14, 2): 3, (14, 8): 3, (20, 2): 3, (20, 8): 3}


def _walk():
    from geomap.walks import Walk
    # clock at (0,0), bell at (1,1) after 2 moves, tree at (1,2), lamp at (2,1); the last move returns to (1,1)
    moves = ["north", "east", "north", "east", "south", "south", "west", "north"]
    return Walk(moves=moves, landmarks={0: "clock", 2: "bell", 3: "tree", 5: "lamp"}, target="bell", loop=6)


def test_walk_text_one_sentence_per_move_and_landmarks_in_place():
    from geomap.walks import walk_text
    w = _walk()
    t = walk_text(w)
    assert t.startswith("You start next to a clock. You walk one block north. You walk one block east. "
                        "There is a bell here. You walk one block north. There is a tree here.")
    assert t.count("You walk one block") == 8 and t.endswith("You walk one block north.")
    assert walk_text(w, person="I").startswith("I start next to a clock. I walk one block north.")
    from geomap.walks import Walk
    w2 = Walk(moves=["east", "west"], landmarks={0: "anchor"}, target="anchor", loop=2)
    assert walk_text(w2).startswith("You start next to an anchor.")
    w3 = Walk(moves=["east", "north"], landmarks={1: "bell"}, target="bell", loop=1)
    assert walk_text(w3).startswith("You start at a street corner. You walk one block east. There is a bell here.")


def test_build_walk_prompt_formats(tok):
    from geomap.walks import build_walk_prompt, walk_text
    w = _walk()
    u = build_walk_prompt(w, tok, "qwen3", "user")
    text = tok.decode(u.input_ids)
    assert walk_text(w) in text
    assert "Which object is it? Answer with one word." in text
    assert sorted(u.candidates) == sorted(w.landmarks.values()) and u.answer == "bell"
    p = build_walk_prompt(w, tok, "qwen3", "prefill")
    ptext = tok.decode(p.input_ids)
    assert walk_text(w, person="I") in ptext and ptext.endswith("I am now back at the")
    assert sorted(p.candidates) == sorted(" " + x for x in w.landmarks.values()) and p.answer == " bell"
    with pytest.raises(ValueError):
        build_walk_prompt(w, tok, "qwen3", "other")


def test_landmark_pool_is_single_token_with_and_without_space(tok):
    from geomap.walks import LANDMARKS, landmark_pool
    pool = landmark_pool(tok)
    assert len(pool) >= 8 and set(pool) <= set(LANDMARKS)
    for x in pool:
        assert len(tok.encode(" " + x, add_special_tokens=False)) == 1
        assert len(tok.encode(x, add_special_tokens=False)) == 1


def test_baselines_on_a_hand_built_walk():
    from geomap.walks import baselines
    b = baselines(_walk())   # landmarks in order: clock(0) bell(2) tree(3) lamp(5); n = 8, target bell
    assert b["recency"] == 0.0 and b["primacy"] == 0.0
    assert b["rank 2"] == 1.0 and b["rank 1"] == b["rank 3"] == b["rank 4"] == 0.0
    # moves ago: clock 8, bell 6, tree 5, lamp 3 -> even: clock, bell -> a uniform guess is right half the time
    assert b["parity guess"] == 0.5


def test_walk_summary_and_verdict():
    from geomap.walks import BASELINES, verdict_walk_pilot, walk_summary
    rng = np.random.default_rng(0)
    rows = []
    for fmt in ("user", "prefill"):
        for n in (14, 20):
            for loop in (2, 8):
                good = fmt == "user" and loop == 8 and n == 20
                for i in range(200):
                    r = dict(walk=f"{n}-{loop}-{i}", n=n, loop=loop, fmt=fmt, mass=0.9,
                             correct=float(rng.random() < (0.85 if good or loop == 2 else 0.3)))
                    r.update({f"b_{k}": float(loop == 2) if k == "recency" else 0.25 for k in BASELINES})
                    rows.append(r)
    s = walk_summary(pd.DataFrame(rows), n_boot=200)
    assert set(s.columns) >= {"fmt", "n", "loop", "n_walks", "acc", "lo", "hi", "best_baseline", "base_acc",
                              "gap", "gap_lo", "gap_hi", "passes"}
    row = s[(s.fmt == "user") & (s.n == 20) & (s.loop == 8)].iloc[0]
    assert row.passes and row.acc > 0.75 and row.gap_lo > 0
    assert not s[(s.loop == 2)].passes.any()   # recency is perfect there, so no gap
    v, cfg = verdict_walk_pilot(s)
    assert v == "go" and cfg == dict(fmt="user", n=20, loop=8)
    weak = s.assign(passes=False, gap_lo=np.where(s.loop == 8, 0.01, -0.1))
    assert verdict_walk_pilot(weak) == ("weak pass", None)
    assert verdict_walk_pilot(s.assign(passes=False, gap_lo=-0.1)) == ("fail", None)
    pre = s.assign(passes=(s.fmt == "prefill") & (s.loop == 8))
    assert verdict_walk_pilot(pre) == ("go", dict(fmt="prefill", n=20, loop=8))


def test_allowed_ranks():
    from geomap.walks import allowed_ranks
    assert allowed_ranks(14, 2) == [4]           # nothing inside a 2-move loop: the target is the last object seen
    assert allowed_ranks(20, 6) == [2, 3]        # 2 even-distance slots inside the loop
    assert allowed_ranks(20, 8) == [1, 2, 3]
    assert allowed_ranks(14, 12) == [1, 2]       # only step 0 lies before the target
