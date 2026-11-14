def test_only_the_70b_model_is_loaded_in_8_bit():
    from geomap.models import FAMILY, MODELS, quant_config
    assert MODELS["llama3.3-70b"] == "meta-llama/Llama-3.3-70B-Instruct" and FAMILY["llama3.3-70b"] == "llama3"
    assert quant_config("gemma3-27b") is None and quant_config("qwen3-32b") is None
    assert quant_config("llama3.3-70b") == {"load_in_8bit": True}
