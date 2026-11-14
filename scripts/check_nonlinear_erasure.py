"""Check B (docs/preregistration_map_use_checks.md): after P5c's erasure (INLP basis, every token, layers 12–30), can a
nonlinear (MLP) probe still read object positions? Compared with a random basis of the same rank and with no erasure.

Resumable: each condition's probe results are saved under results/checks and skipped on a rerun. Progress goes to
Slack if $SLACK_WEBHOOK_URL is set."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.checks import linear_cv, mlp_cv, r2, verdict_nonlinear
from geomap.models import load
from geomap.probes import _r2_ci, centered_xy
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="tiny model, random bases, 60 scenes; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/checks)")
    a = ap.parse_args()
    out, layers, window, primary = ROOT / "results" / "checks", [14, 21, 28, 36], list(range(12, 31)), 21
    mlp_kw = {}
    if a.smoke:
        import tempfile
        a.model, out, layers, window, primary, a.n, a.n_boot = "tiny", Path(tempfile.mkdtemp()), [1, 2], [1, 2], 1, 60, 20
        mlp_kw = dict(hidden=32, max_epochs=5)
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_nonlinear_erasure_n{a.n}"
    tag = f"Check B {a.model}"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[check B] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)

    if a.smoke:
        g = torch.Generator().manual_seed(0)
        mk = lambda: {l: (torch.linalg.qr(torch.randn(m.d_model, 2, generator=g))[0], torch.zeros(2)) for l in window}
        bases, rand = mk(), mk()
    else:
        saved = torch.load(ROOT / "results" / "p5" / f"{a.model}_p5c_bases.pt")
        bases, rand = saved["bases"], saved["rand"]
        if sorted(bases) != window:
            raise ValueError(f"P5c bases cover layers {sorted(bases)}, expected {window[0]}-{window[-1]}")
    conds = {"clean": None, "probe": {l: bases[l] for l in window}, "random": {l: rand[l] for l in window}}
    ranks = {l: int(bases[l][0].shape[1]) for l in window}

    rng = np.random.default_rng(901)
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    scenes = [generate_scene(rng, pools[i % 2], 3) for i in range(a.n)]
    specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
    notify(f"[{tag}] started: {a.n} scenes, conditions {list(conds)}, layers {layers}")

    def probe_condition(cond):
        acts = cache_prefix(m, specs, layers, batch_size=a.batch_size, erase=conds[cond])["list"]
        print(f"[check B] {cond}: cached {tuple(acts.shape)} ({time.time() - t0:.0f}s)", flush=True)
        rows = []
        for li, layer in enumerate(layers):
            X, Y, meta = centered_xy(acts[:, li].float().numpy(), scenes, np.arange(len(scenes)))
            grp = meta.scene.values
            for probe, P in (("linear", linear_cv(X, Y, grp)), ("mlp", mlp_cv(X, Y, grp, **mlp_kw))):
                lo, hi = _r2_ci(Y, P, grp, a.n_boot)
                r = dict(cond=cond, layer=layer, probe=probe, r2=r2(Y, P), r2_lo=lo, r2_hi=hi)
                for h in sorted(meta.hop.unique()):
                    k = (meta.hop == h).values
                    r[f"r2_hop{h}"] = r2(Y[k], P[k])
                rows.append(r)
                print(f"[check B] {cond} L{layer} {probe}: R² {r['r2']:.3f} [{lo:.3f}, {hi:.3f}] "
                      f"({time.time() - t0:.0f}s)", flush=True)
        return pd.DataFrame(rows)

    dfs = []
    for cond in conds:
        path = out / f"{run}_{cond}.parquet"
        fresh = not path.exists()
        d = df_step(path, lambda c=cond: probe_condition(c))
        dfs.append(d)
        if fresh:
            x = d[(d.layer == primary)].set_index("probe").r2
            notify(f"[{tag}] {cond} done: layer {primary} linear R² {x['linear']:.2f}, MLP R² {x['mlp']:.2f}")
    s = pd.concat(dfs, ignore_index=True)
    s.to_csv(out / f"{run}_summary.csv", index=False)
    get = lambda c, p: float(s[(s.cond == c) & (s.layer == primary) & (s.probe == p)].r2.iloc[0])
    v = verdict_nonlinear(mlp_clean=get("clean", "mlp"), lin_clean=get("clean", "linear"),
                          mlp_probe=get("probe", "mlp"), mlp_random=get("random", "mlp"))

    hops = sorted(c for c in s.columns if c.startswith("r2_hop"))
    lines = [f"Model **{a.model}**. {a.n} new 3-object scenes (seed 901); object-list positions. Erasure as in P5c's "
             f"\"window 12–30\": the INLP basis (rank {min(ranks.values())}–{max(ranks.values())} per layer) or a random "
             "basis of the same rank, removed at every token at the output of layers "
             f"{window[0]}–{window[-1]}, and the activations captured after it. Probes on scene-centred activations, "
             "3-fold scene-grouped CV; R² pooled over x and y, 95% scene-bootstrap CIs. MLP: 2 × 512 GELU, early "
             "stopping.", "",
             "| layer | probe | no erasure | probe basis erased | random basis erased |", "|---|---|---|---|---|"]
    for layer in layers:
        for probe in ("linear", "mlp"):
            cell = lambda c: (lambda r: f"{r.r2:.2f} [{r.r2_lo:.2f}, {r.r2_hi:.2f}]")(
                s[(s.cond == c) & (s.layer == layer) & (s.probe == probe)].iloc[0])
            lines.append(f"| {layer} | {probe} | {cell('clean')} | {cell('probe')} | {cell('random')} |")
    lines += ["", f"MLP R² by hop at layer {primary} (" + ", ".join(h.replace("r2_hop", "hop ") for h in hops) + "): "
              + "; ".join(f"{c} " + " / ".join(f"{float(s[(s.cond == c) & (s.layer == primary) & (s.probe == 'mlp')][h].iloc[0]):.2f}"
                                               for h in hops) for c in conds) + ".",
              "", f"Preregistered verdict (layer {primary}): **{v}**. Rules: docs/preregistration_map_use_checks.md "
              "(check B)."]
    if not a.smoke:
        append_entry(f"Check B: does a nonlinear map survive the erasure? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v)
    notify(f"[{tag}] COMPLETE. {v}")
    print("\n".join(lines))
    print(f"verdict: {v}")


if __name__ == "__main__":
    main()
