"""Fair random control for the verified erasure (docs/preregistration_erase_fair_control.md).

Reuses the confirmatory test's bases (results/erase/<model>_erase_axis_state.pt) and its clean and map-erased answers
(..._erase_axis_test_{clean,probe}.parquet). Builds a random basis per layer matched to the map basis in removed
variance in both directions (α in [−8, 8]), checks it on the validation scenes, scores the same test set once with it,
and applies the amendment's two rules.

Resumable: the control, its validation and its scoring run are saved under results/erase and skipped on a rerun."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.checks import mlp_cv, r2, variance_matched_random, verdict_axis, verdict_overall
from geomap.followups import chain_items, erase_edit, erase_summary, p7_class
from geomap.models import load
from geomap.probes import GroupRidge, centered_xy
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene
from geomap.scoring import predict, score

FIT_SEED, VAL_SEED, TEST_SEED, START, ALPHA_MIN = 1101, 1102, 1203, 1, -8.0
FAMILY = {"more_east": "axis", "more_north": "axis", "closest": "distance", "farthest": "distance",
          "direction": "direction"}


def save_atomic(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-sigma", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 1-2, random bases; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/erase)")
    a = ap.parse_args()
    out = ROOT / "results" / "erase"
    mlp_layers, mlp_kw = [14, 21, 28, 36, 44], {}
    if a.smoke:
        import tempfile
        a.model, out = "tiny", Path(tempfile.mkdtemp())
        a.n_sigma, a.n_val, a.n_test = 60, 30, 6
        mlp_layers, mlp_kw = [1, 2], dict(hidden=32, max_epochs=5)
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"Fair control {a.model}"
    base, run = f"{a.model}_erase_axis", f"{a.model}_erase_fair"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[fair] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[fair] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    window = [1, 2] if a.smoke else list(range(12, m.n_layers + 1))
    mlp_layers = [l for l in mlp_layers if l in window]
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    d = m.d_model

    def scenes_for(seed, n):
        rng = np.random.default_rng(seed)
        return [generate_scene(rng, pools[i % 2], 3) for i in range(n)]

    def specs_for(sc):
        return [build_prompt(s, None, m.tok, m.family) for s in sc]

    def as_t(bases):
        return {l: (torch.as_tensor(B, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32))
                for l, (B, c) in bases.items()}

    # ---- the map bases from the confirmatory test ----
    spath = out / f"{base}_state.pt"
    if a.smoke and not spath.exists():
        sc = scenes_for(FIT_SEED, a.n_sigma)
        acts = cache_prefix(m, specs_for(sc), window, batch_size=a.batch_size)["list"]
        mu = {l: np.nanmean(acts[:, li].float().numpy().reshape(-1, d), 0) for li, l in enumerate(window)}
        Bs = {l: np.linalg.qr(np.random.default_rng(l).standard_normal((d, 3)))[0] for l in window}
        save_atomic(dict(mu_clean=mu, bases={l: (Bs[l], mu[l] @ Bs[l]) for l in window}), spath)
    st = torch.load(spath, weights_only=False)
    bases = {l: st["bases"][l] for l in window}
    ranks = {l: B.shape[1] for l, (B, _) in bases.items()}

    # ---- the fair control ----
    fpath = out / f"{run}_control.pt"
    if not fpath.exists():
        sig_sc = scenes_for(FIT_SEED, a.n_sigma)
        val_sc = scenes_for(VAL_SEED, a.n_val)
        acts = cache_prefix(m, specs_for(sig_sc), window, batch_size=a.batch_size)["list"]
        vacts = cache_prefix(m, specs_for(val_sc), window, batch_size=a.batch_size)["list"]
        ctrl, info = {}, []
        for li, l in enumerate(window):
            X, _, _ = centered_xy(acts[:, li].float().numpy(), sig_sc, np.arange(len(sig_sc)))
            Xv, _, _ = centered_xy(vacts[:, li].float().numpy(), val_sc, np.arange(len(val_sc)))
            B = bases[l][0]
            R, alpha = variance_matched_random(X, B, seed=l, alpha_min=ALPHA_MIN)
            ctrl[l] = (R, st["mu_clean"][l] @ R)
            var = lambda M, A: float(((A @ M) ** 2).sum() / len(A))
            info.append(dict(layer=l, rank=ranks[l], alpha=alpha, var_probe=var(B, X), var_fair=var(R, X),
                             var_probe_val=var(B, Xv), var_fair_val=var(R, Xv)))
            log(f"layer {l}: rank {ranks[l]}, α {alpha:+.2f}, removed variance fair/map "
                f"{info[-1]['var_fair'] / max(info[-1]['var_probe'], 1e-12):.3f}")
        del acts, vacts
        save_atomic(dict(control=ctrl, info=info), fpath)
    fc = torch.load(fpath, weights_only=False)
    info = pd.DataFrame(fc["info"])
    info.to_csv(out / f"{run}_control.csv", index=False)
    fair = as_t(fc["control"])
    ok = info[info["rank"] > 0]
    ratio, ratio_v = ok.var_fair / ok.var_probe, ok.var_fair_val / ok.var_probe_val
    notify(f"[{tag}] control built: removed variance fair/map {ratio.min():.3f}–{ratio.max():.3f} (fit scenes), "
           f"{ratio_v.min():.2f}–{ratio_v.max():.2f} (validation scenes)")

    # ---- the control should leave the map readable ----
    val_sc = scenes_for(VAL_SEED, a.n_val)

    def validate():
        acts = cache_prefix(m, specs_for(val_sc), mlp_layers, batch_size=a.batch_size, erase=fair,
                            erase_start=START)["list"]
        rows = []
        for li, l in enumerate(mlp_layers):
            X, Y, meta = centered_xy(acts[:, li].float().numpy(), val_sc, np.arange(len(val_sc)))
            grp = meta.scene.values
            rows.append(dict(cond="fair", layer=l, probe="linear", r2=GroupRidge().fit(X, Y, grp).cv_r2_))
            rows.append(dict(cond="fair", layer=l, probe="mlp", r2=r2(Y, mlp_cv(X, Y, grp, **mlp_kw))))
        return pd.DataFrame(rows)

    val = df_step(out / f"{run}_validation.parquet", validate)
    log("validation done")

    # ---- the answers: same test set; clean and map-erased reused ----
    items = chain_items(m, a.n_test, seed=TEST_SEED, exclude_qtypes=("closer_than",))
    flat = [(si, qi, s, q, comp, p7_class(q, s.root, mid, child))
            for si, (s, mid, child, qs) in enumerate(items) for qi, (q, comp) in enumerate(qs)]
    specs = [(build_prompt(s, q, m.tok, m.family), q) for _, _, s, q, _, _ in flat]
    truth = [answer_for(q, s.coords) for _, _, s, q, _, _ in flat]
    chance = np.array([1 / len(q.candidates) for _, _, _, q, _, _ in flat])

    def scored(cond, edit):
        preds = [predict(x) for x in score(m, specs, edit=edit, batch_size=a.batch_size)]
        return pd.DataFrame([dict(scene=si, qi=qi, qtype=q.qtype.value, composed=comp, cls=cls, cond=cond, pred=p,
                                  truth=t, correct=float(p == t), chance=ch)
                             for (si, qi, _, q, comp, cls), p, t, ch in zip(flat, preds, truth, chance)])

    dfs = []
    for cond, path, edit in [("clean", out / f"{base}_test_clean.parquet", None),
                             ("probe", out / f"{base}_test_probe.parquet", erase_edit(as_t(bases), start=START)),
                             ("fair", out / f"{run}_test_fair.parquet", erase_edit(fair, start=START))]:
        fresh = not path.exists()
        dd = df_step(path, lambda c=cond, e=edit: scored(c, e))
        if len(dd) != len(flat) or list(dd.truth) != truth or list(dd.qtype) != [x[3].qtype.value for x in flat]:
            raise ValueError(f"{path.name} does not match the regenerated test set")
        dfs.append(dd)
        msg = f"{cond}: accuracy {dd.correct.mean():.3f} (composed {dd[dd.composed].correct.mean():.3f})"
        log(msg)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    df["family"] = df.qtype.map(FAMILY)
    df["subset"] = df.cls + " " + df.family
    df.to_parquet(out / f"{run}_test.parquet")

    d2 = df.replace({"cond": {"fair": "random"}}).assign(config="fair random")
    s = erase_summary(d2, group="subset")
    so = erase_summary(d2, group="composed")
    s.to_csv(out / f"{run}_summary.csv", index=False)
    so.to_csv(out / f"{run}_summary_overall.csv", index=False)
    rc = float(df[(df.cond == "fair") & df.composed].correct.mean())
    comp_chance = float(df[(df.cond == "clean") & df.composed].chance.mean())
    pa = s[s.subset == "CR axis"].iloc[0]
    pb = so[so.composed].iloc[0]
    va = verdict_axis(pa.excess, pa.excess_lo, max_val_r2=0.009 if not a.smoke else 0.0, random_comp_acc=rc,
                      comp_chance=comp_chance)
    vb = verdict_overall(pb.excess, pb.excess_lo, random_comp_acc=rc, comp_chance=comp_chance)

    # ---- report ----
    mp = val.pivot_table(index="layer", columns="probe", values="r2")
    lines = [f"Model **{a.model}**. Amends the confirmatory erasure: same map bases (rank "
             f"{min(ranks.values())}–{max(ranks.values())} per layer, layers {window[0]}–{window[-1]}, every token except "
             f"the first), same test set and the same clean and map-erased answers. New control: random bases matched "
             f"in removed variance in both directions (α {info.alpha.min():+.2f} to {info.alpha.max():+.2f}); removed "
             f"variance fair/map {ratio.min():.3f}–{ratio.max():.3f} on the fit scenes, {ratio_v.min():.2f}–"
             f"{ratio_v.max():.2f} on the validation scenes.", "",
             "Probes under the fair control (validation scenes; clean layer 21: linear 0.68, MLP 0.73): " +
             "; ".join(f"L{l} linear {mp.loc[l, 'linear']:.2f}, MLP {mp.loc[l, 'mlp']:.2f}" for l in mlp_layers) + ".",
             "", f"Test set: {len(items)} chains (seed {TEST_SEED}), {len(flat)} questions. Excess drop = fair-control "
             "accuracy − map-erased accuracy; 95% scene-bootstrap CIs.", "",
             "| questions | n | clean | map erased | fair random | excess drop [95% CI] |", "|---|---|---|---|---|---|"]
    for x in so.itertuples():
        lines.append(f"| {'**all composed** (primary B)' if x.composed else 'all stated-only'} | {x.n} | {x.clean:.2f} | "
                     f"{x.clean - x.drop_probe:.2f} | {x.clean - x.drop_random:.2f} | {x.excess:+.3f} "
                     f"[{x.excess_lo:+.3f}, {x.excess_hi:+.3f}] |")
    for x in s.sort_values("subset").itertuples():
        star = " **(primary A)**" if x.subset == "CR axis" else ""
        lines.append(f"| {x.subset}{star} | {x.n} | {x.clean:.2f} | {x.clean - x.drop_probe:.2f} | "
                     f"{x.clean - x.drop_random:.2f} | {x.excess:+.3f} [{x.excess_lo:+.3f}, {x.excess_hi:+.3f}] |")
    lines += ["", f"Fair-control composed accuracy {rc:.2f} (chance {comp_chance:.2f}).", "",
              f"Preregistered verdicts: primary A (C–R axis) **{va}**; primary B (all composed) **{vb}**. "
              "Rules: docs/preregistration_erase_fair_control.md."]
    if not a.smoke:
        append_entry(f"Fair random control for the verified erasure ({a.model}) — preregistered amendment",
                     "\n".join(lines), [])
    done.write_text(f"A: {va}; B: {vb}")
    notify(f"[{tag}] COMPLETE. A (C–R axis): {va}, {pa.excess:+.3f} [{pa.excess_lo:+.3f}, {pa.excess_hi:+.3f}]. "
           f"B (composed): {vb}, {pb.excess:+.3f} [{pb.excess_lo:+.3f}, {pb.excess_hi:+.3f}]")
    print("\n".join(lines))
    print(f"verdicts: A {va}; B {vb}")


if __name__ == "__main__":
    main()
