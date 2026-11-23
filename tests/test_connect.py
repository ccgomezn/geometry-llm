import copy

import numpy as np
import pytest
import torch

from geomap.axis import CELLS
from geomap.connect import (SEED_DEV, SEED_TRAIN, add_lora, first_token_accuracy, general_items, general_prompts,
                            last_token_logits, lora_state, make_set, merge_lora, name_pools, training_examples,
                            unwrap_lora)
from geomap.models import load_tiny_for_tests
from geomap.questions import answer_for
from geomap.scenes import relation_text


@pytest.fixture(scope="module")
def tiny():
    return load_tiny_for_tests()


SIZES = {c: 6 for c in CELLS}


def test_training_set_has_every_cell_and_one_step_questions_with_geometric_labels(tiny):
    rows = make_set(tiny.tok, SEED_TRAIN, SIZES, one_step_per_scene=2)
    cr = [r for r in rows if r["kind"] == "C-R"]
    assert sorted(set(r["cell"] for r in cr)) == sorted(CELLS)
    assert all(sum(r["cell"] == c for r in cr) == 6 for c in CELLS)
    assert {r["kind"] for r in rows} == {"C-R", "C-M", "M-R"}
    for r in rows:
        assert r["answer"] == answer_for(r["q"], r["s"].coords)
        if r["kind"] == "C-R":
            assert r["truth"] == np.sign(r["d1"] + r["d2"]) and (r["answer"] == r["child"]) == (r["truth"] > 0)
    # both one-sentence rules fail somewhere in training: "follow C's step" on conflict M larger, "follow M" on C larger
    assert any(r["cell"] == "conflict, M larger" for r in cr) and any(r["cell"] == "conflict, C larger" for r in cr)


def test_dev_scenes_and_held_out_names_are_disjoint_from_training(tiny):
    key = lambda r: tuple(relation_text(x) for x in r["s"].relations)
    tr = make_set(tiny.tok, SEED_TRAIN, SIZES)
    dv = make_set(tiny.tok, SEED_DEV, SIZES, transfer_per_scene=2)
    assert not {key(r) for r in tr} & {key(r) for r in dv}
    assert {r["kind"] for r in dv} >= {"farthest"} or {r["kind"] for r in dv} >= {"closest"}
    a, b = name_pools(tiny.tok, False), name_pools(tiny.tok, True)
    assert all(not set(x) & set(y) for x, y in zip(a, b))
    nn = make_set(tiny.tok, SEED_DEV, SIZES, held_out_names=True)
    seen = set(a[0]) | set(a[1])
    assert all(not set(r["s"].names) & seen for r in nn)


def test_zero_lora_is_the_identity_and_merging_reproduces_the_wrapped_model(tiny):
    hf = copy.deepcopy(tiny.hf)
    ids = [[1, 5, 9, 14, 3], [2, 7, 7]]
    pad = tiny.tok.pad_token_id or 0
    with torch.no_grad():
        clean = last_token_logits(hf, ids, pad)
        loras = add_lora(hf, "model.layers", from_layer=1, r=4, alpha=8.0)
        assert loras and all(k.startswith("1.") for k in loras)                     # only blocks >= from_layer
        assert torch.allclose(last_token_logits(hf, ids, pad), clean, atol=1e-6)
        for v in loras.values():
            v.B.normal_(0, 0.5)
        wrapped = last_token_logits(hf, ids, pad)
        assert not torch.allclose(wrapped, clean, atol=1e-3)
    st = lora_state(loras)
    fresh = copy.deepcopy(tiny.hf)
    merge_lora(fresh, "model.layers", st)
    with torch.no_grad():
        assert torch.allclose(last_token_logits(fresh, ids, pad), wrapped, atol=1e-4)
    unwrap_lora(hf, "model.layers")
    with torch.no_grad():
        assert torch.allclose(last_token_logits(hf, ids, pad), clean, atol=1e-6)


def test_training_the_lora_lowers_the_answer_loss(tiny):
    hf = copy.deepcopy(tiny.hf)
    rows = make_set(tiny.tok, SEED_TRAIN, {c: 2 for c in CELLS})
    ex = training_examples(tiny.tok, tiny.family, rows)[:8]
    pad = tiny.tok.pad_token_id or 0
    loras = add_lora(hf, "model.layers", from_layer=0, r=4, alpha=8.0)
    opt = torch.optim.AdamW([p for v in loras.values() for p in (v.A, v.B)], lr=1e-2)
    tgt = torch.tensor([t for _, t in ex])
    loss = lambda: (lambda lg: torch.nn.functional.cross_entropy(lg.float(), tgt.to(lg.device)))(
        last_token_logits(hf, [x for x, _ in ex], pad))
    first = float(loss())
    for _ in range(20):
        opt.zero_grad()
        l = loss()
        l.backward()
        opt.step()
    assert float(loss()) < first - 0.5
    assert all(not p.requires_grad for n, p in hf.named_parameters() if not n.endswith((".A", ".B")))


def test_general_items_are_unambiguous_one_token_answers(tiny):
    items = general_items()
    assert len(items) > 60 and all(a and q.endswith("?") for q, a in items)
    assert ("What is 2 plus 3?", "5") in items and ("What is the capital of France?", "Paris") in items
    ex = general_prompts(tiny.tok, tiny.family)
    assert len(ex) == len(items)
    acc = first_token_accuracy(tiny.hf, ex[:5], tiny.tok.pad_token_id or 0)
    assert 0.0 <= acc <= 1.0


def test_attaching_a_saved_adapter_reproduces_the_trained_model(tiny):
    from geomap.connect import attach_lora
    hf = copy.deepcopy(tiny.hf)
    ids = [[1, 5, 9, 14, 3], [2, 7, 7]]
    pad = tiny.tok.pad_token_id or 0
    loras = add_lora(hf, "model.layers", from_layer=1, r=4, alpha=8.0, targets=("q_proj", "down_proj"))
    with torch.no_grad():
        for v in loras.values():
            v.B.normal_(0, 0.5)
        want = last_token_logits(hf, ids, pad)
    fresh = copy.deepcopy(tiny.hf)
    attach_lora(fresh, "model.layers", lora_state(loras))
    with torch.no_grad():
        assert torch.allclose(last_token_logits(fresh, ids, pad), want, atol=1e-5)
    assert all(not p.requires_grad for p in fresh.parameters())
