"""Sum-of-parts test (docs/preregistration_compose.md): on new scenes, is a two-step object's distance to the root
(not a sum of the stated offsets) readable beyond the best side-by-side baseline, at its list mention (layer 21) and
at the end of its own sentence (layer 24)?

Resumable: the activations are cached under results/compose and reused on a rerun."""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix, sentence_end_positions
from geomap.checkpoint import notify
from geomap.compose import TARGETS, fit_predict, part_rows, site_scores, unseen_pairs, verdict_compose, _r2
from geomap.models import load
from geomap.probes import scene_split
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene

SEED, N = 1501, 3000
PRIMARY = {"list": 21, "sentence": 24}
SWEEP = [0, 6, 12, 14, 16, 18, 21, 24, 27, 30, 36, 42, 48]


def make_scenes(tok, n, seed):
    """As build_cache.make_scenes('main'): 3-6 objects, alternating people and object names."""
    rng = np.random.default_rng(seed)
    pools = [single_token_names(tok, NAME_POOLS[k]) for k in ("people", "objects")]
    return [generate_scene(rng, pools[i % 2], 3 + i % 4) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="tiny model, 300 scenes, its 3 layers; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/compose)")
    a = ap.parse_args()
    out, n, primary, sweep = ROOT / "results" / "compose", N, dict(PRIMARY), list(SWEEP)
    if a.smoke:
        import tempfile
        a.model, n, out, a.n_boot = "tiny", 300, Path(tempfile.mkdtemp()), 100
        primary, sweep = {"list": 2, "sentence": 2}, [0, 1, 2]
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_compose_seed{SEED}"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[compose] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[compose] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    scenes = make_scenes(m.tok, n, SEED)
    layers = sorted(set(sweep) | set(primary.values()))
    tr_s, te_s = scene_split(len(scenes))
    meta, parts, pos = part_rows(scenes, range(len(scenes)), hops=(2,))
    tr = np.isin(meta.scene.to_numpy(), tr_s)
    log(f"{len(scenes)} scenes, {len(meta)} two-step objects ({tr.sum()} train, {(~tr).sum()} test)")
    notify(f"[Compose test {a.model}] started: {len(meta)} two-step objects")

    acts = {}
    for site in ("list", "sentence"):
        path = out / f"{run}_{site}_acts.pt"
        if path.exists():
            acts[site] = torch.load(path)
            continue
        specs = []
        for s in scenes:
            sp = build_prompt(s, None, m.tok, m.family)
            specs.append(sp if site == "list" else replace(sp, list_pos=sentence_end_positions(sp, s)))
        c = cache_prefix(m, specs, layers, a.batch_size)["list"]          # [N, L, 6, d], slots in list order
        rows = c[torch.tensor(meta.scene.to_numpy()), :, torch.tensor(meta.slot.to_numpy())]  # [rows, L, d]
        acts[site] = {"layers": layers, "x": rows.contiguous()}
        torch.save(acts[site], path)
        log(f"cached {site} activations {tuple(rows.shape)}")

    res = []
    for site in ("list", "sentence"):
        A = acts[site]
        for li, layer in enumerate(A["layers"]):
            if layer not in sweep and layer != primary[site]:
                continue
            X = A["x"][:, li].float().numpy()
            for t in TARGETS:
                is_primary = layer == primary[site]
                if not is_primary and t != "dist":
                    continue
                r = site_scores(X, parts, meta, tr, ~tr, target=t, n_boot=a.n_boot if is_primary else 200)
                res.append(dict(site=site, layer=layer, target=t, primary=is_primary, **r))
        log(f"{site}: probes done")
    s = pd.DataFrame(res)
    s.to_csv(out / f"{run}_summary.csv", index=False)
    v, sites = verdict_compose(s)

    # reported, not part of the rule: shuffled targets, the true-position reference, unseen offset pairs
    y_all = {t: meta[t].to_numpy(float) for t in TARGETS}
    g = meta.scene.to_numpy()
    ref = {t: _r2(y_all[t][~tr], fit_predict(pos[tr], y_all[t][tr], g[tr], pos[~tr])) for t in TARGETS}
    unseen = unseen_pairs(parts, tr)
    extra = []
    for site, layer in primary.items():
        A = acts[site]
        X = A["x"][:, A["layers"].index(layer)].float().numpy()
        y = y_all["dist"]
        shuf = np.random.default_rng(0).permutation(y[tr])
        p_shuf = fit_predict(X[tr], shuf, g[tr], X[~tr])
        p_act = fit_predict(X[tr], y[tr], g[tr], X)
        p_base = fit_predict(parts[tr].astype(float), y[tr], g[tr], parts.astype(float))
        extra.append(dict(site=site, layer=layer, shuffled=_r2(y[~tr], p_shuf), n_unseen=int(unseen.sum()),
                          unseen_act=_r2(y[unseen], p_act[unseen]) if unseen.sum() > 5 else float("nan"),
                          unseen_base=_r2(y[unseen], p_base[unseen]) if unseen.sum() > 5 else float("nan")))
    ex = pd.DataFrame(extra)
    ex.to_csv(out / f"{run}_extra.csv", index=False)

    name = {"list": "list mention", "sentence": "sentence end"}
    lines = [f"Model **{a.model}**. {len(scenes)} new scenes (seed {SEED}), {len(meta)} two-step objects "
             f"({tr.sum()} train / {(~tr).sum()} test rows, split by scene). Targets relative to the root, none a sum "
             "of the stated offsets. *Parts baseline*: one one-hot code per sentence over its joint (dx, dy), the "
             "object's own and its parent's (the best a side-by-side store can do). *Gap* = probe − baseline, "
             "held-out R², paired 95% scene-bootstrap CI.", "",
             "| site | layer | target | probe R² | parts baseline R² | gap [95% CI] |", "|---|---|---|---|---|---|"]
    for r in s[s.primary].itertuples():
        lines.append(f"| {name[r.site]} | {r.layer} | {r.target} | {r.r2_act:.2f} | {r.r2_base:.2f} | "
                     f"{r.gap:+.2f} [{r.gap_lo:+.2f}, {r.gap_hi:+.2f}] |")
    lines += ["", f"True-position reference (x, y, x², y², xy): distance {ref['dist']:.2f}, |x| {ref['abs_x']:.2f}, "
                  f"|y| {ref['abs_y']:.2f}.", ""]
    for r in ex.itertuples():
        lines.append(f"- {name[r.site]} (layer {r.layer}): shuffled-target control R² {r.shuffled:.2f}; test rows "
                     f"with an (own, parent) offset pair unseen in training: {r.n_unseen}, probe R² "
                     f"{r.unseen_act:.2f}, baseline {r.unseen_base:.2f}")
    lines += ["", "Distance, layer sweep (probe R² / gap):", "", "| layer | list mention | sentence end |",
              "|---|---|---|"]
    d = s[s.target == "dist"]
    for layer in sweep:
        cells = []
        for site in ("list", "sentence"):
            q = d[(d.site == site) & (d.layer == layer)]
            cells.append(f"{q.r2_act.iloc[0]:.2f} / {q.gap.iloc[0]:+.2f}" if len(q) else "–")
        lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    where = f" Sites: {', '.join(name[x] for x in sites)}." if sites else ""
    lines += ["", f"Preregistered verdict: **{v}**.{where} Rules: docs/preregistration_compose.md."]
    if not a.smoke:
        append_entry(f"Sum-of-parts test: does the model combine the two sentences? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v + where)
    notify(f"[Compose test {a.model}] COMPLETE. verdict: {v}.{where}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
