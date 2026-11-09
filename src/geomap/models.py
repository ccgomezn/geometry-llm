"""Model loading and the single place that knows where the residual stream lives."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from nnsight import LanguageModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-14b": "Qwen/Qwen3-14B",
    "gemma3-4b": "google/gemma-3-4b-it",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}
FAMILY = {"qwen3-1.7b": "qwen3", "qwen3-4b": "qwen3", "qwen3-8b": "qwen3", "qwen3-14b": "qwen3",
          "gemma3-4b": "gemma3", "llama3.2-3b": "llama3", "tiny": "qwen3"}


@dataclass
class LoadedModel:
    lm: LanguageModel
    hf: torch.nn.Module
    tok: object
    key: str
    family: str
    n_layers: int
    d_model: int
    layers_path: str
    embed_path: str


def _find_paths(hf) -> tuple[str, str]:
    names = dict(hf.named_modules())
    layer_paths = [n for n, mod in names.items() if n.endswith("layers") and isinstance(mod, torch.nn.ModuleList)
                   and "vision" not in n and "visual" not in n]
    layers_path = min(layer_paths, key=len)
    embed_path = layers_path.rsplit(".", 1)[0] + ".embed_tokens"
    if embed_path not in names:
        raise ValueError(f"no embed_tokens next to {layers_path}")
    return layers_path, embed_path


def _wrap(hf, tok, key: str) -> LoadedModel:
    hf.eval()
    layers_path, embed_path = _find_paths(hf)
    layers = functools.reduce(getattr, layers_path.split("."), hf)
    cfg = getattr(hf.config, "text_config", hf.config)
    return LoadedModel(lm=LanguageModel(hf, tokenizer=tok), hf=hf, tok=tok, key=key, family=FAMILY[key],
                       n_layers=len(layers), d_model=cfg.hidden_size,
                       layers_path=layers_path, embed_path=embed_path)


def load(key: str, device: str = "mps", dtype=torch.bfloat16, random_init: bool = False) -> LoadedModel:
    if key == "tiny":  # smoke tests of the scripts
        return load_tiny_for_tests()
    repo = MODELS[key]
    tok = AutoTokenizer.from_pretrained(repo)
    if random_init:
        torch.manual_seed(0)
        hf = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(repo), torch_dtype=dtype)
    else:
        hf = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype)
    return _wrap(hf.to(device), tok, key)


def load_tiny_for_tests() -> LoadedModel:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    torch.manual_seed(0)
    return _wrap(Qwen3ForCausalLM(cfg).to(torch.float32), tok, "tiny")


def _envoy(m: LoadedModel, path: str):
    return functools.reduce(getattr, path.split("."), m.lm)


def layer_out(m: LoadedModel, layer: int):
    """Residual stream at `layer` inside an nnsight trace. 0 = embeddings, l = output of decoder block l-1.

    Must be accessed in increasing layer order within one trace (nnsight runs the model in order).
    """
    if layer == 0:
        out = _envoy(m, m.embed_path).output
    else:
        out = _envoy(m, m.layers_path)[layer - 1].output
    return out[0] if isinstance(out, tuple) else out
