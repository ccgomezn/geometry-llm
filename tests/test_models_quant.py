def test_only_the_70b_model_is_loaded_in_8_bit():
    from geomap.models import FAMILY, MODELS, quant_config
    assert MODELS["llama3.3-70b"] == "meta-llama/Llama-3.3-70B-Instruct" and FAMILY["llama3.3-70b"] == "llama3"
    assert quant_config("gemma3-27b") is None and quant_config("qwen3-32b") is None
    assert quant_config("llama3.3-70b") == {"load_in_8bit": True}


def test_8_bit_model_is_placed_whole_on_gpu_0(monkeypatch):
    # device_map="auto" keeps ~10% of the GPU spare and spilled a few 70B layers to the CPU, which 8-bit refuses
    import geomap.models as M
    seen = {}

    def fake(repo, **kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(M.AutoTokenizer, "from_pretrained", lambda repo: None)
    monkeypatch.setattr(M.AutoModelForCausalLM, "from_pretrained", fake)
    try:
        M.load("llama3.3-70b")
    except RuntimeError:
        pass
    assert seen["device_map"] == {"": 0} and "quantization_config" in seen
