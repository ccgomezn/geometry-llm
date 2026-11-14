from types import SimpleNamespace

import pytest

from geomap.models import load_tiny_for_tests
from geomap.questions import QType, answer_for


@pytest.fixture(scope="module")
def tok():
    return load_tiny_for_tests().tok


def _steps(s, mid, child, q):
    ax = 0 if q.qtype == QType.MORE_EAST else 1
    return s.coords[mid][ax] - s.coords[s.root][ax], s.coords[child][ax] - s.coords[mid][ax]


def test_axis_bulk_items_fill_each_cell_with_correct_labels(tok):
    from geomap.axis import CELLS, axis_bulk_items
    sizes = {c: 6 for c in CELLS}
    items = axis_bulk_items(SimpleNamespace(tok=tok), sizes, seed=5)
    got = {c: 0 for c in CELLS}
    for s, mid, child, qs in items:
        for q, cell in qs:
            assert q.qtype in (QType.MORE_EAST, QType.MORE_NORTH) and set(q.args) == {s.root, child}
            d1, d2 = _steps(s, mid, child, q)
            assert abs(d1 + d2) >= 1
            expect = ("own step only" if d1 == 0 else "first step only" if d2 == 0 else "agree" if d1 * d2 > 0
                      else "conflict, C larger" if abs(d2) > abs(d1) else "conflict, M larger")
            assert cell == expect
            assert answer_for(q, s.coords) == (child if d1 + d2 > 0 else s.root)
            if cell == "conflict, M larger":   # the sign of C's own step gives the wrong answer here
                assert (d2 > 0) != (d1 + d2 > 0)
            got[cell] += 1
    assert got == sizes


def test_axis_bulk_items_are_deterministic(tok):
    from geomap.axis import CELLS, axis_bulk_items
    a = axis_bulk_items(SimpleNamespace(tok=tok), {c: 3 for c in CELLS}, seed=9)
    b = axis_bulk_items(SimpleNamespace(tok=tok), {c: 3 for c in CELLS}, seed=9)
    assert [(s.coords, [q.text for q, _ in qs]) for s, _, _, qs in a] == [(s.coords, [q.text for q, _ in qs])
                                                                           for s, _, _, qs in b]


def test_verdicts():
    from geomap.axis import verdict_behaviour, verdict_erasure
    assert verdict_behaviour(0.30, 0.45) == "the model follows C's own step"
    assert verdict_behaviour(0.55, 0.60) == "the model composes the steps"
    assert verdict_behaviour(0.45, 0.55) == "undecided"
    assert verdict_erasure(0.05, 0.02, 0.08) == "the answers read C's composed position from the map"
    assert verdict_erasure(-0.06, -0.10, -0.02) == "the erased directions support the own-step shortcut, not the composed position"
    assert verdict_erasure(0.02, 0.005, 0.04) == "undecided"
    assert verdict_erasure(-0.01, -0.04, 0.02) == "undecided"
