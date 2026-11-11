"""P5c (docs/preregistration_p5.md): erase the probe-readable coordinates (INLP basis, every token position) and
check whether answers about the worked-out (two-step) object suffer more than with a random basis of the same rank.

Resumable: the per-layer bases and every scoring run are saved under results/p5 and skipped on a rerun. Progress
messages go to Slack when $SLACK_WEBHOOK_URL is set."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.followups import chain_items, erase_edit, erase_summary, inlp_basis, verdict_erase
from geomap.models import load
from geomap.probes import centered_xy
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene
from geomap.scoring import predict, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-fit", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 1-2; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/p5)")
    a = ap.parse_args()
    out, layers = ROOT / "results" / "p5", list(range(12, 31))
    configs = {"L14": [14], "L21": [21], "L28": [28], "window 12-30": layers}
    if a.smoke:
        import tempfile
        out, layers = Path(tempfile.mkdtemp()), [1, 2]
        configs = {"L1": [1], "window 1-2": layers}
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"P5c {a.model}"
    done = out / f"{a.model}_p5c.done"
    if done.exists():
        print(f"[p5c] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    m = load(a.model)

    # fit: INLP basis per layer at the object-list positions (checkpoint: one file, rewritten after each layer)
    bpath = out / f"{a.model}_p5c_bases.pt"
    saved = torch.load(bpath) if bpath.exists() else {"bases": {}, "rand": {}, "fit": []}
    todo = [l for l in layers if l not in saved["bases"]]
    if todo:
        rng = np.random.default_rng(201)
        pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
        scenes = [generate_scene(rng, pools[i % 2], 3) for i in range(a.n_fit)]
        specs = [build_prompt(s, None, m.tok, m.family) for s in scenes]
        acts = cache_prefix(m, specs, todo, batch_size=a.batch_size)["list"]  # [N, len(todo), 6, d]
        print(f"[p5c] cached {tuple(acts.shape)} for layers {todo}, {time.time() - t0:.0f}s", flush=True)
        for li, layer in enumerate(todo):
            A = acts[:, li].float().numpy()
            X, Y, meta = centered_xy(A, scenes, np.arange(len(scenes)))
            B, hist = inlp_basis(X, Y, meta.scene.values, max_rounds=20, r2_stop=0.1)
            mu = A[~np.isnan(A[:, :, 0])].mean(0)
            Bt = torch.as_tensor(B, dtype=torch.float32)
            g = torch.Generator().manual_seed(layer)
            R = torch.linalg.qr(torch.randn(m.d_model, B.shape[1], generator=g))[0]
            saved["bases"][layer] = (Bt, torch.as_tensor(mu @ B, dtype=torch.float32))
            saved["rand"][layer] = (R, torch.as_tensor(mu, dtype=torch.float32) @ R)
            saved["fit"].append(dict(layer=layer, rank=B.shape[1], r2_before=hist[0], r2_after=hist[-1],
                                     rounds=len(hist) - 1))
            tmp = bpath.with_name(bpath.name + ".tmp")
            torch.save(saved, tmp)
            tmp.replace(bpath)
            print(f"[p5c] layer {layer}: rank {B.shape[1]}, R² {hist[0]:.2f} -> {hist[-1]:.3f}", flush=True)
        del acts
        notify(f"[{tag}] erasure bases fitted for layers {layers[0]}-{layers[-1]} ({time.time() - t0:.0f}s)")
    bases, rand = saved["bases"], saved["rand"]
    fit = pd.DataFrame(saved["fit"]).sort_values("layer")
    fit.to_csv(out / f"{a.model}_p5c_inlp.csv", index=False)

    # test: clean vs erased answers on new chains (checkpoint: one parquet per run)
    items = chain_items(m, a.n_test, seed=202, exclude_qtypes=("closer_than",))
    flat = [(si, qi, s, q, comp) for si, (s, _, _, qs) in enumerate(items) for qi, (q, comp) in enumerate(qs)]
    specs = [(build_prompt(s, q, m.tok, m.family), q) for _, _, s, q, _ in flat]
    truth = [answer_for(q, s.coords) for _, _, s, q, _ in flat]
    print(f"[p5c] {len(flat)} questions in {len(items)} chains", flush=True)

    def run(cond, cfg, edit):
        preds = [predict(sc) for sc in score(m, specs, edit=edit, batch_size=a.batch_size)]
        return pd.DataFrame([dict(scene=si, qi=qi, qtype=q.qtype.value, composed=comp, config=cfg, cond=cond,
                                  pred=p, truth=t, correct=float(p == t))
                             for (si, qi, _, q, comp), p, t in zip(flat, preds, truth)])

    runs = [("clean", "-", None)] + [(c, cfg, erase_edit({l: (bases if c == "probe" else rand)[l] for l in ls}))
                                     for cfg, ls in configs.items() for c in ("probe", "random")]
    dfs = []
    for cond, cfg, edit in runs:
        path = out / f"{a.model}_p5c_run_{cond}_{cfg.replace(' ', '_')}.parquet"
        fresh = not path.exists()
        d = df_step(path, lambda c=cond, f=cfg, e=edit: run(c, f, e))
        dfs.append(d)
        msg = f"{cond} {cfg}: acc {d.correct.mean():.3f}"
        print(f"[p5c] {msg} ({time.time() - t0:.0f}s)", flush=True)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out / f"{a.model}_p5c.parquet")
    s = erase_summary(df)
    s.to_csv(out / f"{a.model}_p5c_summary.csv", index=False)
    r2_after = float(fit.r2_after.max())
    v = verdict_erase(s, r2_after=r2_after)

    lines = [f"Model **{a.model}**. INLP basis fitted per layer at the object-list positions ({a.n_fit} scenes); "
             f"erased at every token position. Rank per layer {fit['rank'].min()}–{fit['rank'].max()}; probe R² "
             f"before {fit.r2_before.min():.2f}–{fit.r2_before.max():.2f}, after ≤ {r2_after:.3f}. "
             f"Test: {len(items)} new chains, {len(flat)} questions.", "",
             "| erased at | questions | clean acc | drop (probe basis) | drop (random basis) | excess drop [95% CI] | n |",
             "|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.config} | {'composed (involve C)' if r.composed else 'stated only'} | {r.clean:.2f} | "
                     f"{r.drop_probe:.3f} | {r.drop_random:.3f} | {r.excess:.3f} [{r.excess_lo:.3f}, {r.excess_hi:.3f}] "
                     f"| {r.n} |")
    lines += ["", f"Preregistered verdict: **{v}**. Rules: docs/preregistration_p5.md."]
    if not a.smoke:
        append_entry(f"P5c erase the map ({a.model}) — preregistered", "\n".join(lines), [])
    done.write_text(v)
    comp = s[s.composed].set_index("config").excess.round(3).to_dict()
    notify(f"[{tag}] COMPLETE. Verdict: {v}. Excess drop on composed questions by config: {comp}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
