import numpy as np

from geomap.questions import (
    COMPASS,
    QType,
    Question,
    answer_for,
    battery,
    make_pair,
    margin,
    split_questions,
)
from geomap.scenes import generate_scene, leaves

POOL = [f"N{i}" for i in range(40)]
LAYOUT = {"A": (0, 0), "B": (3, 0), "C": (0, -4), "D": (1, 1)}


def q(qtype, *args):
    return Question(qtype, tuple(args), "", ())


def test_answer_for_known_layout():
    assert answer_for(q(QType.FARTHEST, "A"), LAYOUT) == "C"
    assert answer_for(q(QType.CLOSEST, "A"), LAYOUT) == "D"
    assert answer_for(q(QType.DIRECTION, "A", "D"), LAYOUT) == "northeast"
    assert answer_for(q(QType.DIRECTION, "A", "C"), LAYOUT) == "south"
    assert answer_for(q(QType.MORE_NORTH, "B", "D"), LAYOUT) == "D"
    assert answer_for(q(QType.MORE_EAST, "B", "D"), LAYOUT) == "B"
    assert answer_for(q(QType.CLOSER_THAN, "A", "B", "C"), LAYOUT) == "yes"


def test_margin_values():
    assert margin(q(QType.FARTHEST, "A"), LAYOUT) == 1.0  # 4 - 3
    assert margin(q(QType.DIRECTION, "A", "D"), LAYOUT) == 22.5  # dead centre of northeast
    assert margin(q(QType.MORE_NORTH, "B", "D"), LAYOUT) == 1.0


def test_battery_respects_margins_and_texts():
    for seed in range(200):
        s = generate_scene(np.random.default_rng(seed), POOL, 3 + seed % 4)
        qs = battery(s, np.random.default_rng(seed))
        assert qs
        for qq in qs:
            m = margin(qq, s.coords)
            assert m >= (10.0 if qq.qtype == QType.DIRECTION else 1.0)
            assert answer_for(qq, s.coords) in qq.candidates
            assert qq.text
    qs = battery(generate_scene(np.random.default_rng(1), POOL, 4), np.random.default_rng(0))
    dir_q = next(x for x in qs if x.qtype == QType.DIRECTION)
    assert dir_q.candidates == COMPASS
    assert dir_q.text == f"In which compass direction is {dir_q.args[1]} from {dir_q.args[0]}?"


def test_make_pair_moves_one_leaf_same_template():
    for seed in range(100):
        s = generate_scene(np.random.default_rng(seed), POOL, 4)
        base, src, leaf = make_pair(s, np.random.default_rng(seed))
        assert leaf in leaves(base)
        assert base.coords[leaf] != src.coords[leaf]
        assert all(base.coords[n] == src.coords[n] for n in base.names if n != leaf)
        rb = next(r for r in base.relations if r.subj == leaf)
        rs = next(r for r in src.relations if r.subj == leaf)
        assert rb.template_id == rs.template_id
        # same singular/plural pattern and same number of components
        assert (rb.dx != 0) + (rb.dy != 0) == (rs.dx != 0) + (rs.dy != 0)
        assert sorted([abs(rb.dx) == 1, abs(rb.dy) == 1]) == sorted([abs(rs.dx) == 1, abs(rs.dy) == 1])


def test_split_questions_partition():
    s = generate_scene(np.random.default_rng(3), POOL, 5)
    base, src, _ = make_pair(s, np.random.default_rng(3))
    qs = battery(base, np.random.default_rng(0), per_type=10)
    diag, ctrl = split_questions(qs, base.coords, src.coords)
    for qq in diag:
        assert answer_for(qq, base.coords) != answer_for(qq, src.coords)
    for qq in ctrl:
        assert answer_for(qq, base.coords) == answer_for(qq, src.coords)
    assert len(diag) + len(ctrl) <= len(qs)
