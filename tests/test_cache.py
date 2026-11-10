import numpy as np
import pytest
import torch

from geomap.cache import cache_prefix, load_cache, save_cache
from geomap.models import layer_out, load_tiny_for_tests
from geomap.prompts import NAME_POOLS, build_prompt
from geomap.questions import battery
from geomap.scenes import generate_scene


@pytest.fixture(scope="module")
def m():
    return load_tiny_for_tests()


def _scenes(k=5):
    return [generate_scene(np.random.default_rng(i), NAME_POOLS["people"], 3 + i % 4) for i in range(k)]


def test_cache_shapes_and_padding(m):
    scenes = _scenes()
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    c = cache_prefix(m, specs, layers=[0, 1, 2], batch_size=2)
    assert c["list"].shape == (5, 3, 6, m.d_model) and c["eol"].shape == (5, 3, m.d_model)
    assert c["list"].dtype == torch.bfloat16  # float16 overflows on Gemma 3 residuals (>65504)
    for i, s in enumerate(scenes):
        n = len(s.names)
        assert not torch.isnan(c["list"][i, :, :n]).any()
        assert torch.isnan(c["list"][i, :, n:]).all()


def test_cache_matches_full_prompt_run(m):
    scenes = _scenes(3)
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    c = cache_prefix(m, specs, layers=[0, 2], batch_size=3)
    for i, s in enumerate(scenes):
        q = battery(s, np.random.default_rng(0))[0]
        full = build_prompt(s, q, m.tok, m.family)
        ids = torch.tensor([full.input_ids])
        with torch.no_grad(), m.lm.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}):
            h0 = layer_out(m, 0).cpu().save()
            h2 = layer_out(m, 2).cpu().save()
        for slot, name in enumerate(s.names):
            p = full.list_pos[name]
            assert torch.allclose(c["list"][i, 0, slot].float(), h0[0, p].float(), atol=1e-3)
            assert torch.allclose(c["list"][i, 1, slot].float(), h2[0, p].float(), atol=1e-3)
        assert torch.allclose(c["eol"][i, 1].float(), h2[0, full.eol_pos].float(), atol=1e-3)


def test_save_load_roundtrip(m, tmp_path):
    scenes = _scenes(2)
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    c = cache_prefix(m, specs, layers=[1])
    save_cache(tmp_path / "x.pt", c, scenes, layers=[1])
    c2, s2, layers = load_cache(tmp_path / "x.pt")
    assert s2 == scenes and layers == [1]
    assert torch.equal(torch.nan_to_num(c2["list"]), torch.nan_to_num(c["list"]))


def test_cache_rejects_non_finite(m, monkeypatch):
    import geomap.cache as gc
    scenes = _scenes(2)
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    real = gc.layer_out

    def huge(mm, layer):
        h = real(mm, layer)
        h[:] = h * 1e39
        return h

    monkeypatch.setattr(gc, "layer_out", huge)
    with pytest.raises(ValueError):
        cache_prefix(m, specs, layers=[1])


def test_sentence_end_positions(m):
    from geomap.cache import sentence_end_positions
    for s in _scenes(4):
        spec = build_prompt(s, None, m.tok, m.family)
        pos = sentence_end_positions(spec, s)
        assert list(pos) == list(s.names)  # same slot order as the object list
        first = s.relations[0].subj
        for n in s.names:
            src = first if n == s.root else n  # the root has no sentence: use the end of the first sentence
            assert pos[n] == spec.relation_spans[src][1] - 1
