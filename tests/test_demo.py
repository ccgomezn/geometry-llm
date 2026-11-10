import numpy as np
import pytest
import torch

from geomap.das import SubspaceIntervention
from geomap.demo import anchored_positions, demo_questions, leaf_targets, readout
from geomap.models import load_tiny_for_tests
from geomap.patching import prefix_acts
from geomap.prompts import NAME_POOLS, aligned, build_prompt, single_token_names
from geomap.questions import QType
from geomap.scenes import generate_scene, leaves


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


@pytest.fixture(scope="module")
def scene(m):
    names = single_token_names(m.tok, NAME_POOLS["people"])
    return generate_scene(np.random.default_rng(3), names, 3, p_shuffle=0.0)


def test_demo_questions_all_involve_the_leaf(scene):
    leaf = leaves(scene)[0]
    qs = demo_questions(scene, leaf)
    assert len(qs) == 6 and all(q.qtype != QType.CLOSER_THAN for q in qs)
    assert all(leaf in q.args or q.qtype == QType.FARTHEST for q in qs)


def test_leaf_targets_are_aligned_moves_of_the_leaf_only(m, scene):
    leaf = leaves(scene)[0]
    ts = leaf_targets(m, scene, leaf, max_n=8, rng=np.random.default_rng(0))
    assert ts
    bp = build_prompt(scene, None, m.tok, m.family)
    assert len({t.coords[leaf] for t in ts}) == len(ts)
    for t in ts:
        assert t.coords[leaf] != scene.coords[leaf]
        assert all(t.coords[n] == scene.coords[n] for n in scene.names if n != leaf)
        assert aligned(bp, build_prompt(t, None, m.tok, m.family), leaf)


def test_readout_without_edit_matches_prefix_acts(m, scene):
    spec = build_prompt(scene, None, m.tok, m.family)
    pos = list(spec.list_pos.values())
    ref = prefix_acts(m, spec, [2])[2][pos].float()
    assert torch.allclose(readout(m, spec, None, 1, [], None, 2, pos), ref, atol=1e-3)


def test_readout_full_rank_edit_writes_the_target_state(m, scene):
    spec = build_prompt(scene, None, m.tok, m.family)
    leaf = leaves(scene)[0]
    lo, hi = spec.relation_spans[leaf]
    pos = list(range(lo, hi))
    target = torch.randn(len(pos), m.d_model)
    out = readout(m, spec, SubspaceIntervention(m.d_model, m.d_model), 2, pos, target, 2, pos)
    assert torch.allclose(out, target, atol=1e-3)


def test_anchored_positions_pin_the_root():
    dec = {"R": np.array([1.0, 1.0]), "A": np.array([3.0, 0.0])}
    out = anchored_positions(dec, "R", (0, 2))
    assert out == {"R": (0.0, 2.0), "A": (2.0, 1.0)}
