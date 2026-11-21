import pytest
import torch

from geomap.axis import CELLS, axis_bulk_items
from geomap.blind import blind_spans, greedy_masked, recovery, verdict_blind
from geomap.models import load_tiny_for_tests
from geomap.readback import _gen, pad_left


@pytest.fixture(scope="module")
def tiny():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def prompts(tiny):
    out = []
    for s, mid, child, qq in axis_bulk_items(tiny, {c: 3 for c in CELLS}, seed=1601):
        q = qq[0][0]
        out.append(blind_spans(s, q, mid, child, tiny.tok, tiny.family))
        if len(out) == 3:
            return out


def test_spans_are_m_sentence_and_object_list(tiny, prompts):
    for ids, m_span, l_span in prompts:
        a, b = m_span
        c, d = l_span
        assert 0 < a < b <= c < d < len(ids)
        assert "objects" in tiny.tok.decode(ids[c:d])


def test_normal_loop_reproduces_greedy_generate(tiny, prompts):
    seqs = [p[0] for p in prompts]
    ours = greedy_masked(tiny, seqs, [[]] * 3, [[]] * 3, max_new_tokens=8)
    X, am, _ = pad_left(seqs, tiny.tok, [0] * 3)
    ref = _gen(tiny, X, am, max_new_tokens=8)[:, X.shape[1]:]
    for j in range(3):
        assert ours[j] == [t for t in ref[j].tolist() if t != tiny.tok.pad_token_id][:len(ours[j])]


def _scramble(span_lists):
    def corrupt(cache, starts):
        torch.manual_seed(1)
        for layer in cache.layers:
            for b, (spans, st) in enumerate(zip(span_lists, starts)):
                for a, e in spans:
                    layer.keys[b, :, st + a:st + e] = torch.randn_like(layer.keys[b, :, st + a:st + e]) * 5
                    layer.values[b, :, st + a:st + e] = torch.randn_like(layer.values[b, :, st + a:st + e]) * 5
    return corrupt


def test_blind_reply_ignores_the_hidden_span(tiny, prompts):
    seqs = [p[0] for p in prompts]
    m = [[p[1]] for p in prompts]
    blind = greedy_masked(tiny, seqs, [[]] * 3, m, max_new_tokens=8)
    blind_scrambled = greedy_masked(tiny, seqs, [[]] * 3, m, max_new_tokens=8, after_prefill=_scramble(m))
    assert blind == blind_scrambled
    normal = greedy_masked(tiny, seqs, [[]] * 3, [[]] * 3, max_new_tokens=8)
    normal_scrambled = greedy_masked(tiny, seqs, [[]] * 3, [[]] * 3, max_new_tokens=8, after_prefill=_scramble(m))
    assert normal != normal_scrambled                      # the scramble matters when the span is visible
    assert [x[0] for x in blind] == [x[0] for x in normal]   # the prompt is read as in normal


def test_floor_does_not_depend_on_the_words_of_m_sentence(tiny, prompts):
    seqs = [p[0] for p in prompts]
    m = [[p[1]] for p in prompts]
    other = [list(s) for s in seqs]
    for s, (a, b) in zip(other, (p[1] for p in prompts)):
        s[a:b] = [s[a]] * (b - a)                        # different words, same length
    assert greedy_masked(tiny, seqs, m, m, 8) == greedy_masked(tiny, other, m, m, 8)


def test_recovery_and_verdict():
    assert recovery(blind=0.8, floor=0.3, normal=0.8) == pytest.approx(1.0)
    assert recovery(blind=0.3, floor=0.3, normal=0.8) == pytest.approx(0.0)
    ok = dict(normal=(0.9, 0.85, 0.95), floor=(0.3, 0.25, 0.35), blind=(0.8, 0.75, 0.85), control=0.88)
    assert verdict_blind(rec=(0.8, 0.6, 0.9), **ok) == "the processed state is usable"
    assert verdict_blind(rec=(0.1, 0.0, 0.15), **{**ok, "blind": (0.35, 0.3, 0.4)}) == \
        "the processed state is not usable by the reply"
    assert verdict_blind(rec=(0.4, 0.2, 0.6), **ok) == "partial"
    assert verdict_blind(rec=(0.8, 0.6, 0.9), **{**ok, "control": 0.7}) == "uninformative"
    assert verdict_blind(rec=(0.8, 0.6, 0.9), **{**ok, "normal": (0.7, 0.65, 0.75)}) == "uninformative"
    assert verdict_blind(rec=(0.8, 0.6, 0.9), **{**ok, "floor": (0.8, 0.7, 0.9)}) == "uninformative"


def test_recovery_is_undefined_when_normal_equals_floor():
    import math
    assert math.isnan(recovery(blind=0.5, floor=0.5, normal=0.5))
