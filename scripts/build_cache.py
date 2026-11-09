"""Build activation caches (all layers) at the object-list and end-of-list positions."""

import argparse
import time

import numpy as np

from geomap.cache import save_cache, cache_prefix
from geomap.models import load
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.report import ROOT
from geomap.scenes import generate_scene


def make_scenes(tok, dataset: str, n: int, seed: int, n_objects: int | None = None):
    rng = np.random.default_rng(seed)
    pools = [single_token_names(tok, NAME_POOLS[k]) for k in ("people", "objects")]
    if dataset == "main":
        return [generate_scene(rng, pools[i % 2], n_objects or 3 + i % 4) for i in range(n)]
    if dataset == "eol":
        return [generate_scene(rng, pools[i % 2], n_objects or 4) for i in range(n)]
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="main", choices=["main", "eol"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--n-objects", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    m = load(a.model, random_init=a.random_init)
    scenes = make_scenes(m.tok, a.dataset, a.n, a.seed, a.n_objects)
    name = a.dataset if a.dataset == "main" else f"eol{a.n_objects or 4}"
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    layers = list(range(m.n_layers + 1))
    t0 = time.time()
    c = cache_prefix(m, specs, layers, a.batch_size)
    tag = a.model + ("-rand" if a.random_init else "")
    path = ROOT / "cache" / tag / f"{name}.pt"
    save_cache(path, c, scenes, layers)
    print(f"{path}: list {tuple(c['list'].shape)}, eol {tuple(c['eol'].shape)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
