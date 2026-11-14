"""Closed-loop erasure (docs/preregistration_erase_closed_loop.md): fit the erasure basis on activations captured *under*
the erasure, round after round, until a freshly fitted probe can no longer read positions at any erased layer; check
that on fresh scenes; then score P5c's test questions clean, with the map erased, and with a random basis of the same
rank. Erasure at every token except the first (<bos>), at every layer from 12 to the last.

Resumable: the bases (after every round), the validation and every scoring run are saved under results/erase and
skipped on a rerun. Progress goes to Slack if $SLACK_WEBHOOK_URL is set."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.checks import extend_basis, mlp_cv, r2, verdict_closed_loop
from geomap.followups import chain_items, erase_edit, erase_summary, p7_class
from geomap.models import load
from geomap.probes import GroupRidge, centered_xy
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene
from geomap.scoring import predict, score

FIT_SEED, VAL_SEED, TEST_SEED = 1001, 1002, 202
R2_STOP, START = 0.05, 1  # START = 1: the first token (<bos>) is never erased


def save_atomic(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-fit", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 1-2, few scenes; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/erase)")
    a = ap.parse_args()
    out = ROOT / "results" / "erase"
    mlp_layers, primary, mlp_kw = [14, 21, 28, 36, 44], 21, {}
    if a.smoke:
        import tempfile
        a.model, out, a.n_fit, a.n_val, a.n_test, a.max_rounds = "tiny", Path(tempfile.mkdtemp()), 60, 30, 6, 2
        mlp_layers, primary, mlp_kw = [1, 2], 1, dict(hidden=32, max_epochs=5)
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"Closed-loop erasure {a.model}"
    run = f"{a.model}_erase_closed_loop"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[erase] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[erase] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    window = [1, 2] if a.smoke else list(range(12, m.n_layers + 1))
    mlp_layers = [l for l in mlp_layers if l in window]
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    d = m.d_model

    def scenes_for(seed, n):
        rng = np.random.default_rng(seed)
        return [generate_scene(rng, pools[i % 2], 3) for i in range(n)]

    def as_t(bases):
        return {l: (torch.as_tensor(B, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32))
                for l, (B, c) in bases.items()}

    # ---- fit, closed loop (checkpoint after every round) ----
    fit_sc = scenes_for(FIT_SEED, a.n_fit)
    fit_specs = [build_prompt(s, None, m.tok, m.family) for s in fit_sc]
    idx = np.arange(len(fit_sc))
    bpath = out / f"{run}_bases.pt"
    st = torch.load(bpath, weights_only=False) if bpath.exists() else {}
    if "mu_clean" not in st:
        acts = cache_prefix(m, fit_specs, window, batch_size=a.batch_size)["list"]
        st["mu_clean"] = {l: np.nanmean(acts[:, li].float().numpy().reshape(-1, d), 0) for li, l in enumerate(window)}
        if a.smoke:
            st["bases"] = {l: (np.zeros((d, 0)), np.zeros(0)) for l in window}
            st["p5c"] = None
        else:
            p5 = torch.load(ROOT / "results" / "p5" / f"{a.model}_p5c_bases.pt")
            st["p5c"] = {k: {l: (B.numpy(), c.numpy()) for l, (B, c) in p5[k].items()} for k in ("bases", "rand")}
            st["bases"] = {l: st["p5c"]["bases"].get(l, (np.zeros((d, 0)), np.zeros(0))) for l in window}
            st["bases"] = {l: (np.asarray(B, np.float64), np.asarray(c, np.float64)) for l, (B, c) in st["bases"].items()}
        st.update(history=[], rounds=0, converged=False)
        del acts
        save_atomic(st, bpath)
        log("clean means saved; starting from P5c's bases")
    while not st["converged"] and st["rounds"] < a.max_rounds:
        r = st["rounds"] + 1
        acts = cache_prefix(m, fit_specs, window, batch_size=a.batch_size, erase=as_t(st["bases"]),
                            erase_start=START)["list"]
        all_below = True
        for li, l in enumerate(window):
            A = acts[:, li].float().numpy()
            X, Y, meta = centered_xy(A, fit_sc, idx)
            B_old, _ = st["bases"][l]
            B, hist = extend_basis(B_old, X, Y, meta.scene.values, max_steps=5, r2_stop=R2_STOP)
            all_below &= hist[0] < R2_STOP
            mu = np.nanmean(A.reshape(-1, d), 0)
            st["bases"][l] = (B, mu @ B)
            st["history"].append(dict(round=r, layer=l, r2=hist[0], rank_before=B_old.shape[1], rank_after=B.shape[1]))
        del acts
        st["rounds"], st["converged"] = r, bool(all_below)
        save_atomic(st, bpath)
        h = pd.DataFrame([x for x in st["history"] if x["round"] == r])
        msg = (f"round {r}: max probe R² {h.r2.max():.3f} (layer {int(h.loc[h.r2.idxmax(), 'layer'])}), "
               f"{int((h.r2 >= R2_STOP).sum())} layers extended, rank {h.rank_after.min()}–{h.rank_after.max()}")
        log(msg)
        notify(f"[{tag}] {msg}")
    hist = pd.DataFrame(st["history"])
    hist.to_csv(out / f"{run}_fit_history.csv", index=False)
    bases = {l: st["bases"][l] for l in window}
    ranks = {l: B.shape[1] for l, (B, _) in bases.items()}
    rand = {}
    for l in window:
        g = torch.Generator().manual_seed(l)
        R = torch.linalg.qr(torch.randn(d, max(ranks[l], 1), generator=g, dtype=torch.float64))[0][:, :ranks[l]].numpy()
        rand[l] = (R, st["mu_clean"][l] @ R)
    conds = {"probe": as_t(bases), "random": as_t(rand)}
    if st["p5c"] is not None:
        conds["p5c_probe"] = as_t({l: st["p5c"]["bases"][l] for l in st["p5c"]["bases"]})
        conds["p5c_random"] = as_t({l: st["p5c"]["rand"][l] for l in st["p5c"]["rand"]})

    # ---- is the erasure effective? fresh scenes ----
    val_sc = scenes_for(VAL_SEED, a.n_val)
    val_specs = [build_prompt(s, None, m.tok, m.family) for s in val_sc]
    vidx = np.arange(len(val_sc))

    def validate():
        rows = []
        for cond in ("probe", "random"):
            layers = window if cond == "probe" else mlp_layers
            acts = cache_prefix(m, val_specs, layers, batch_size=a.batch_size, erase=conds[cond],
                                erase_start=START)["list"]
            for li, l in enumerate(layers):
                X, Y, meta = centered_xy(acts[:, li].float().numpy(), val_sc, vidx)
                grp = meta.scene.values
                rows.append(dict(cond=cond, layer=l, probe="linear", r2=GroupRidge().fit(X, Y, grp).cv_r2_))
                if l in mlp_layers:
                    rows.append(dict(cond=cond, layer=l, probe="mlp", r2=r2(Y, mlp_cv(X, Y, grp, **mlp_kw))))
            del acts
            log(f"validation {cond} done")
        return pd.DataFrame(rows)

    val = df_step(out / f"{run}_validation.parquet", validate)
    vp = val[(val.cond == "probe") & (val.probe == "linear")]
    max_val_r2 = float(vp.r2.max())
    mlp21 = val[(val.cond == "probe") & (val.probe == "mlp") & (val.layer == primary)].r2
    mlp21 = float(mlp21.iloc[0]) if len(mlp21) else float("nan")
    notify(f"[{tag}] validation: max linear R² after erasure {max_val_r2:.3f}; MLP L{primary} {mlp21:.3f}")

    # ---- the answers ----
    items = chain_items(m, a.n_test, seed=TEST_SEED, exclude_qtypes=("closer_than",))
    flat = [(si, qi, s, q, comp, p7_class(q, s.root, mid, child))
            for si, (s, mid, child, qs) in enumerate(items) for qi, (q, comp) in enumerate(qs)]
    specs = [(build_prompt(s, q, m.tok, m.family), q) for _, _, s, q, _, _ in flat]
    truth = [answer_for(q, s.coords) for _, _, s, q, _, _ in flat]
    chance = np.array([1 / len(q.candidates) for _, _, _, q, _, _ in flat])
    log(f"{len(flat)} test questions in {len(items)} chains")

    def scored(cond, config, edit):
        preds = [predict(x) for x in score(m, specs, edit=edit, batch_size=a.batch_size)]
        return pd.DataFrame([dict(scene=si, qi=qi, qtype=q.qtype.value, composed=comp, cls=cls, config=config,
                                  cond=c2, pred=p, truth=t, correct=float(p == t), chance=ch)
                             for (si, qi, _, q, comp, cls), p, t, ch, c2 in
                             zip(flat, preds, truth, chance, [cond.replace("p5c_", "")] * len(flat))])

    runs = [("clean", "-", None)] + [(c, "closed loop" if not c.startswith("p5c") else "P5c bases, first token spared",
                                      erase_edit(conds[c], start=START)) for c in conds]
    dfs = []
    for cond, config, edit in runs:
        path = out / f"{run}_test_{cond}.parquet"
        fresh = not path.exists()
        dd = df_step(path, lambda c=cond, f=config, e=edit: scored(c, f, e))
        dfs.append(dd)
        msg = f"{cond}: accuracy {dd.correct.mean():.3f} (composed {dd[dd.composed].correct.mean():.3f})"
        log(msg)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out / f"{run}_test.parquet")
    s = erase_summary(df)
    s.to_csv(out / f"{run}_summary.csv", index=False)
    main_s = s[s.config == "closed loop"]
    rc = df[(df.config == "closed loop") & (df.cond == "random") & df.composed]
    comp_chance = float(df[(df.cond == "clean") & df.composed].chance.mean())
    v = verdict_closed_loop(main_s, max_val_r2=max_val_r2, random_comp_acc=float(rc.correct.mean()),
                            comp_chance=comp_chance, mlp21=mlp21)

    # ---- report ----
    last = hist[hist["round"] == hist["round"].max()]
    acc = df.groupby(["config", "cond", "cls"]).correct.mean().unstack("cls")
    mlp = val[val.probe == "mlp"].pivot_table(index="layer", columns="cond", values="r2")
    vr = val[(val.cond == "random") & (val.probe == "linear")].set_index("layer").r2
    lines = [f"Model **{a.model}**. Erasure at every token except the first, at every layer {window[0]}–{window[-1]}. "
             f"Basis fitted in closed loop on {a.n_fit} scenes (seed {FIT_SEED}), starting from P5c's: {st['rounds']} "
             f"rounds, {'converged' if st['converged'] else 'NOT converged'} (every layer's refitted probe R² < "
             f"{R2_STOP}); rank per layer {min(ranks.values())}–{max(ranks.values())} (total {sum(ranks.values())}). "
             f"Random control: random orthonormal bases of the same rank.", "",
             f"**Is the map gone?** Fresh scenes (seed {VAL_SEED}, {a.n_val}): largest linear-probe R² over the erased "
             f"layers after the erasure **{max_val_r2:.3f}** (layer {int(vp.loc[vp.r2.idxmax(), 'layer'])}).", "",
             "| layer | linear R², map erased | linear R², random erased | MLP R², map erased | MLP R², random erased |",
             "|---|---|---|---|---|"]
    for l in mlp_layers:
        lines.append(f"| {l} | {float(vp.set_index('layer').r2[l]):.2f} | {float(vr.get(l, np.nan)):.2f} | "
                     f"{float(mlp.loc[l, 'probe']):.2f} | {float(mlp.loc[l, 'random']):.2f} |")
    lines += ["", f"**Do the answers suffer?** P5c's test set ({len(items)} chains, {len(flat)} questions). Excess drop = "
              "drop with the map erased − drop with the random basis; 95% scene-bootstrap CIs.", "",
              "| erasure | questions | clean acc | drop (map erased) | drop (random) | excess drop [95% CI] | n |",
              "|---|---|---|---|---|---|---|"]
    for x in s.itertuples():
        lines.append(f"| {x.config} | {'composed (involve C)' if x.composed else 'stated only'} | {x.clean:.2f} | "
                     f"{x.drop_probe:.3f} | {x.drop_random:.3f} | {x.excess:+.3f} [{x.excess_lo:+.3f}, "
                     f"{x.excess_hi:+.3f}] | {x.n} |")
    lines += ["", "Accuracy by question class (CM: C and M; CR: C and R; MR: no C):", "",
              "| erasure | condition | " + " | ".join(acc.columns) + " |", "|---|---|" + "---|" * len(acc.columns)]
    for (cfg, cond), row in acc.iterrows():
        lines.append(f"| {cfg} | {cond} | " + " | ".join(f"{v_:.2f}" for v_ in row.values) + " |")
    lines += ["", f"Random erasure, composed accuracy {rc.correct.mean():.2f} (chance {comp_chance:.2f}). MLP R² at layer "
              f"{primary} after the erasure: {mlp21:.2f}. Fit rounds: max refitted R² per round "
              + ", ".join(f"{g.r2.max():.2f}" for _, g in hist.groupby("round")) + ".",
              "", f"Preregistered verdict: **{v}**. Rules: docs/preregistration_erase_closed_loop.md."]
    if not a.smoke:
        append_entry(f"Closed-loop erasure: does erasing the map (for real) hurt the answers? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v)
    cm = main_s[main_s.composed].iloc[0]
    notify(f"[{tag}] COMPLETE. {v}. Composed excess drop {cm.excess:+.3f} [{cm.excess_lo:+.3f}, {cm.excess_hi:+.3f}]")
    print("\n".join(lines))
    print(f"verdict: {v}")


if __name__ == "__main__":
    main()
