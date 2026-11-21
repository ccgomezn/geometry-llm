import pytest
import torch

from geomap.axis import CELLS, axis_bulk_items
from geomap.flow import axis_of, flip_step, steps
from geomap.models import load_tiny_for_tests
from geomap.readback import last_logits, mix_spec, pad_left, readback_index, record_layers, patched, verdict_readback


@pytest.fixture(scope="module")
def tiny():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def pairs(tiny):
    out = []
    for s, mid, child, qq in axis_bulk_items(tiny, {c: 6 for c in CELLS}, seed=1601):
        for q, cell in qq:
            if cell != "conflict, M larger":
                continue
            try:
                sf = flip_step(s, mid, axis_of(q))
            except ValueError:
                continue
            spec = mix_spec(s, sf, q, mid, tiny.tok, tiny.family)
            if spec is not None:
                out.append((s, sf, q, mid, child, spec))
    assert len(out) >= 3
    return out


def test_flipped_prompt_differs_only_inside_m_sentence(pairs):
    for s, sf, q, mid, child, (ids_o, ids_f, a, b) in pairs:
        assert len(ids_o) == len(ids_f)
        diff = [i for i, (x, y) in enumerate(zip(ids_o, ids_f)) if x != y]
        assert diff and all(a <= i < b for i in diff)
        assert steps(s, mid, child, q)[2] == -steps(sf, mid, child, q)[2]      # flipping M flips the answer


def test_downstream_patch_gives_the_other_versions_last_prediction(tiny, pairs):
    ids_o, ids_f, starts = [], [], []
    for *_, (o, f, a, b) in pairs[:3]:
        ids_o.append(o), ids_f.append(f), starts.append(b)
    X, am, st = pad_left(ids_o, tiny.tok, starts)
    Y, _, _ = pad_left(ids_f, tiny.tok, starts)
    rec = record_layers(tiny, Y, am)
    with patched(tiny, rec, st, X.shape[1]):
        mixed = last_logits(tiny, X, am)
    assert torch.allclose(mixed, last_logits(tiny, Y, am), atol=1e-4)
    assert not torch.allclose(last_logits(tiny, X, am), last_logits(tiny, Y, am), atol=1e-4)


def test_patching_a_version_with_itself_changes_nothing(tiny, pairs):
    ids, starts = [p[-1][0] for p in pairs[:3]], [p[-1][3] for p in pairs[:3]]
    X, am, st = pad_left(ids, tiny.tok, starts)
    rec = record_layers(tiny, X, am)
    with patched(tiny, rec, st, X.shape[1]):
        a = last_logits(tiny, X, am)
    assert torch.allclose(a, last_logits(tiny, X, am), atol=1e-5)


def test_readback_index():
    # follows M's sentence exactly as often as the model is right: 1; follows the downstream state as often: 0
    assert readback_index(f_text=0.9, acc=0.9) == pytest.approx(1.0)
    assert readback_index(f_text=0.1, acc=0.9) == pytest.approx(0.0)
    assert readback_index(f_text=0.5, acc=0.9) == pytest.approx(0.5)


def test_verdict_rules():
    v = lambda lo, hi, acc=0.9, o=0.9, f=0.9: verdict_readback((0.8, lo, hi), acc, o, f)
    assert v(0.75, 0.95) == "the reasoning re-reads the sentence"
    assert v(0.05, 0.25) == "the reasoning reads the processed state"
    assert v(0.40, 0.80) == "both"
    assert v(0.75, 0.95, acc=0.70) == "uninformative"
    assert v(0.75, 0.95, f=0.65) == "uninformative"


def test_verdict_is_uninformative_without_answers():
    nan = float("nan")
    assert verdict_readback((nan, nan, nan), nan, nan, nan) == "uninformative"
