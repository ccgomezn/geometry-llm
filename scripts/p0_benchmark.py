"""P0: measure local throughput and extrapolate the cost of later phases."""

import argparse
import time

import numpy as np
import torch

from geomap.models import layer_out, load
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import battery
from geomap.report import append_entry
from geomap.scenes import generate_scene
from geomap.scoring import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    t0 = time.time()
    m = load(a.model)
    load_s = time.time() - t0
    names = single_token_names(m.tok, NAME_POOLS["people"])
    rng = np.random.default_rng(0)
    scenes = [generate_scene(rng, names, 4) for _ in range(a.n)]
    items = [(build_prompt(s, q, m.tok, m.family), q) for s in scenes for q in battery(s, rng, per_type=1)]
    rows = sum(len(q.candidates) for _, q in items)

    score(m, items[:8], batch_size=a.batch_size)  # warm-up
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.time()
    score(m, items, batch_size=a.batch_size)
    score_s = time.time() - t0
    per_item = score_s / len(items)

    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    t0 = time.time()
    for b in range(0, len(specs), a.batch_size):
        batch = specs[b:b + a.batch_size]
        T = max(len(s.input_ids) for s in batch)
        ids = torch.zeros((len(batch), T), dtype=torch.long)
        am = torch.zeros_like(ids)
        for j, s in enumerate(batch):
            ids[j, :len(s.input_ids)] = torch.tensor(s.input_ids)
            am[j, :len(s.input_ids)] = 1
        with torch.no_grad(), m.lm.trace({"input_ids": ids, "attention_mask": am}):
            saved = [layer_out(m, l)[:, :, :].cpu().save() for l in range(m.n_layers + 1)]
    cache_s = (time.time() - t0) / len(specs)

    p1_items = 1700 * 12  # 1500 main + 200 near-tie scenes, ~12 questions each
    p3_items = 100 * 4 * (m.n_layers // 2) * 10  # pairs x sites x every other layer x questions
    est = {"P1 behaviour (per model)": p1_items * per_item / 3600,
           "P2 cache (3,000 scenes, all layers)": 3000 * cache_s / 3600,
           "P3 sweep stage A": p3_items * per_item / 3600}
    body = [f"Model **{a.model}** on {m.hf.device} (bf16), batch size {a.batch_size}, {m.n_layers} layers, d={m.d_model}.",
            "",
            f"- Load time: {load_s:.0f} s",
            f"- Scoring: {per_item * 1000:.0f} ms per question ({rows / len(items):.1f} candidates each); {len(items)} questions in {score_s:.0f} s",
            f"- Caching all layers for one prompt prefix: {cache_s * 1000:.0f} ms",
            "",
            "Extrapolated local run time:",
            ""] + [f"- {k}: {v:.1f} h" for k, v in est.items()]
    append_entry(f"P0 benchmark ({a.model})", "\n".join(body))
    print("\n".join(body))


if __name__ == "__main__":
    main()
