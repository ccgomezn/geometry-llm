import re

import numpy as np
import pytest
import torch

from geomap.models import layer_out, load_tiny_for_tests
from geomap.prompts import NAME_POOLS, build_prompt
from geomap.questions import QType, battery
from geomap.report import append_entry
from geomap.scenes import generate_scene
from geomap.scoring import candidate_ids, predict, score


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


def _items(m, n_scenes=3):
    items = []
    for seed in range(n_scenes):
        s = generate_scene(np.random.default_rng(seed), NAME_POOLS["people"], 4)
        for q in battery(s, np.random.default_rng(seed), per_type=1):
            items.append((build_prompt(s, q, m.tok, m.family), q))
    return items


def test_candidate_ids_in_answer_context(m):
    spec, _ = _items(m, 1)[0]
    text = m.tok.decode(spec.input_ids)
    for cand in ["northeast", "Alice", "yes"]:
        full = m.tok(text + cand, add_special_tokens=False)["input_ids"]
        assert full[: len(spec.input_ids)] == spec.input_ids
        assert full[len(spec.input_ids):] == candidate_ids(m.tok, cand)


def test_candidate_sequences_terminate_prefixes_and_cover_case(m):
    from geomap.questions import COMPASS
    from geomap.scoring import candidate_sequences, terminator_ids
    seqs = candidate_sequences(m.tok, m.family, COMPASS)
    terms = set(terminator_ids(m.tok, m.family))
    north_variants = {tuple(m.tok.encode(v, add_special_tokens=False)) for v in ("north", "North")}
    # "north" is a token-prefix of "northwest": every north sequence must end with a terminator
    for sq in seqs["north"]:
        assert sq[-1] in terms and tuple(sq[:-1]) in north_variants
    assert {tuple(sq[:-1]) for sq in seqs["north"]} == north_variants
    # a candidate that prefixes nothing is scored without a terminator, in both cases
    ne = {tuple(m.tok.encode(v, add_special_tokens=False)) for v in ("northeast", "Northeast")}
    assert {tuple(sq) for sq in seqs["northeast"]} <= ne | {t + (x,) for t in ne for x in terms}
    assert len(seqs["northeast"]) >= 2


def test_score_matches_manual_logprob(m):
    from geomap.scoring import candidate_sequences
    spec, q = next((s, q) for s, q in _items(m, 1) if q.qtype == QType.DIRECTION)
    got = score(m, [(spec, q)])[0]
    seqs = candidate_sequences(m.tok, m.family, q.candidates)
    hf = m.hf
    dev = next(hf.parameters()).device
    n = len(spec.input_ids)
    for cand in ["north", "northwest", "south"]:
        parts = []
        for c in seqs[cand]:
            ids = torch.tensor([spec.input_ids + c], device=dev)
            with torch.no_grad():
                lp = hf(input_ids=ids).logits[0].float().log_softmax(-1)
            parts.append(sum(lp[n - 1 + j, t].item() for j, t in enumerate(c)))
        manual = float(torch.logsumexp(torch.tensor(parts), 0))
        assert abs(got[cand] - manual) < 1e-3
    assert set(got) == set(q.candidates)


def test_candidate_mass_at_most_one(m):
    from geomap.scoring import candidate_mass
    for spec, q in _items(m, 2):
        assert candidate_mass(score(m, [(spec, q)])[0]) <= 1e-4  # log of total probability <= 0


def test_batching_invariance(m):
    items = _items(m, 2)
    a = score(m, items, batch_size=1)
    b = score(m, items, batch_size=16)
    for x, y in zip(a, b):
        for k in x:
            assert abs(x[k] - y[k]) < 1e-3


def test_noop_edit_is_identity(m):
    items = _items(m, 1)[:4]

    def noop(mm, row_items):
        h = layer_out(mm, 1)
        h[:] = h.clone()

    a, b = score(m, items), score(m, items, edit=noop)
    for x, y in zip(a, b):
        for k in x:
            assert abs(x[k] - y[k]) < 1e-4


def test_edit_changes_scores(m):
    items = _items(m, 1)[:4]

    def zero(mm, row_items):
        h = layer_out(mm, 1)
        h[:, 5:20, :] = 0

    a, b = score(m, items), score(m, items, edit=zero)
    assert any(abs(x[k] - y[k]) > 1e-3 for x, y in zip(a, b) for k in x)


def test_predict():
    assert predict({"a": -3.0, "b": -1.0}) == "b"


def test_append_entry_writes_header(tmp_path):
    p = tmp_path / "R.md"
    p.write_text("# log\n")
    append_entry("P0 benchmark", "body text", [tmp_path / "figures" / "x.png"], path=p)
    out = p.read_text()
    assert re.search(r"## \d{4}-\d{2}-\d{2} \d{2}:\d{2} · P0 benchmark", out)
    assert "body text" in out and "![x](figures/x.png)" in out


def test_parse_cot_answer():
    from geomap.scoring import parse_cot_answer
    from geomap.questions import COMPASS
    assert parse_cot_answer("<think>north... east</think>\n\nNortheast", COMPASS) == "northeast"
    assert parse_cot_answer("<think>x</think> The answer is south.", COMPASS) == "south"
    assert parse_cot_answer("<think>never closed", COMPASS) is None
    assert parse_cot_answer("<think>x</think> Bob, not Alice... final: Alice", ("Alice", "Bob")) == "Alice"
