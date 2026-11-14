"""Confirmatory erasure test (docs/preregistration_erase_axis_confirm.md): do composed "more east / more north" answers
use the map? Closed-loop erasure at layers 12-48 (every token except the first), fitted on 4,000 scenes from the
closed-loop test's bases down to probe R² < 0.02, with a dev-set generalisation check; validated on fresh scenes;
scored on a new test set (600 chains) clean, map erased, and with two random controls of the same rank: matched in
removed variance (primary) and isotropic.

Resumable: the fit state (after every round), the validation and every scoring run are saved under results/erase and
skipped on a rerun. Progress goes to Slack if $SLACK_WEBHOOK_URL is set."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.checks import extend_basis, mlp_cv, r2, variance_matched_random, verdict_axis
from geomap.followups import chain_items, erase_edit, erase_summary, p7_class
from geomap.models import load
from geomap.probes import GroupRidge, centered_xy
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene
from geomap.scoring import predict, score

FIT_SEED, VAL_SEED, TEST_SEED, DEV_SEEDS = 1101, 1102, 1203, (1103, 1104, 1105)
R2_STOP, DEV_MAX, MAX_DEV_ADD, START = 0.02, 0.07, 2, 1
FAMILY = {"more_east": "axis", "more_north": "axis", "closest": "distance", "farthest": "distance",
          "direction": "direction"}
CONTROLS = {"matched": "matched random (primary)", "iso": "isotropic random"}


def save_atomic(obj, path: Path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--n-fit", type=int, default=4000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-sigma", type=int, default=1000, help="fit scenes used for the covariance of the matched control")
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--max-rounds", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 1-2, few scenes; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/erase)")
    a = ap.parse_args()
    out = ROOT / "results" / "erase"
    mlp_layers, mlp_kw = [14, 21, 28, 36, 44], {}
    if a.smoke:
        import tempfile
        a.model, out = "tiny", Path(tempfile.mkdtemp())
        a.n_fit, a.n_dev, a.n_val, a.n_sigma, a.n_test, a.max_rounds = 60, 30, 30, 30, 6, 3
        mlp_layers, mlp_kw = [1, 2], dict(hidden=32, max_epochs=5)
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"Axis confirm {a.model}"
    run = f"{a.model}_erase_axis"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[axis] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[axis] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    window = [1, 2] if a.smoke else list(range(12, m.n_layers + 1))
    mlp_layers = [l for l in mlp_layers if l in window]
    pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
    d = m.d_model
    empty = (np.zeros((d, 0)), np.zeros(0))

    def scenes_for(seed, n):
        rng = np.random.default_rng(seed)
        return [generate_scene(rng, pools[i % 2], 3) for i in range(n)]

    def specs_for(sc):
        return [build_prompt(s, None, m.tok, m.family) for s in sc]

    def as_t(bases):
        return {l: (torch.as_tensor(B, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32))
                for l, (B, c) in bases.items()}

    def probe_r2s(sc, specs, bases, layers):
        """Linear probe CV R² per layer on activations captured under the erasure."""
        acts = cache_prefix(m, specs, layers, batch_size=a.batch_size, erase=as_t(bases), erase_start=START)["list"]
        res = {}
        for li, l in enumerate(layers):
            X, Y, meta = centered_xy(acts[:, li].float().numpy(), sc, np.arange(len(sc)))
            res[l] = GroupRidge().fit(X, Y, meta.scene.values).cv_r2_
        return res

    # ---- fit, closed loop with dev checks (checkpoint after every step) ----
    spath = out / f"{run}_state.pt"
    st = torch.load(spath, weights_only=False) if spath.exists() else {}
    if "mu_clean" not in st:
        base_sc = scenes_for(FIT_SEED, a.n_fit)
        acts = cache_prefix(m, specs_for(base_sc), window, batch_size=a.batch_size)["list"]
        st["mu_clean"] = {l: np.nanmean(acts[:, li].float().numpy().reshape(-1, d), 0) for li, l in enumerate(window)}
        del acts
        if a.smoke:
            st["bases"] = {l: empty for l in window}
        else:
            prev = torch.load(out / f"{a.model}_erase_closed_loop_bases.pt", weights_only=False)["bases"]
            st["bases"] = {l: tuple(np.asarray(v, np.float64) for v in prev.get(l, empty)) for l in window}
        st.update(history=[], dev=[], fit_seeds=[], rounds=0, fit_ok=False, fit_done=False)
        save_atomic(st, spath)
        log("clean means saved; starting from the closed-loop test's bases")

    def fit_set():
        sc = scenes_for(FIT_SEED, a.n_fit)
        for s_ in st["fit_seeds"]:
            sc += scenes_for(s_, a.n_dev)
        return sc

    fit_sc = fit_set()
    fit_specs = specs_for(fit_sc)
    while not st["fit_done"]:
        if st["fit_ok"]:
            k = len(st["dev"])
            seed = DEV_SEEDS[k]
            dsc = scenes_for(seed, a.n_dev)
            r2s = probe_r2s(dsc, specs_for(dsc), st["bases"], window)
            worst = max(r2s, key=r2s.get)
            add = r2s[worst] >= DEV_MAX and len(st["fit_seeds"]) < MAX_DEV_ADD
            st["dev"].append(dict(seed=seed, after_round=st["rounds"], max_r2=r2s[worst], layer=worst, added=add,
                                  r2=r2s))
            if add:
                st["fit_seeds"].append(seed)
                st["fit_ok"] = False
                fit_sc = fit_set()
                fit_specs = specs_for(fit_sc)
            else:
                st["fit_done"] = True
            save_atomic(st, spath)
            msg = (f"dev check (seed {seed}): max R² {r2s[worst]:.3f} at layer {worst}; "
                   + ("added to the fit set" if add else "fit finished"))
            log(msg)
            notify(f"[{tag}] {msg}")
            continue
        if st["rounds"] >= a.max_rounds:
            st["fit_done"] = True
            save_atomic(st, spath)
            log("round limit reached")
            break
        r = st["rounds"] + 1
        acts = cache_prefix(m, fit_specs, window, batch_size=a.batch_size, erase=as_t(st["bases"]),
                            erase_start=START)["list"]
        all_below = True
        for li, l in enumerate(window):
            A = acts[:, li].float().numpy()
            X, Y, meta = centered_xy(A, fit_sc, np.arange(len(fit_sc)))
            B_old, _ = st["bases"][l]
            B, hist = extend_basis(B_old, X, Y, meta.scene.values, max_steps=5, r2_stop=R2_STOP)
            all_below &= hist[0] < R2_STOP
            st["bases"][l] = (B, np.nanmean(A.reshape(-1, d), 0) @ B)
            st["history"].append(dict(round=r, layer=l, n_fit=len(fit_sc), r2=hist[0], rank_before=B_old.shape[1],
                                      rank_after=B.shape[1]))
        del acts
        st["rounds"], st["fit_ok"] = r, bool(all_below)
        save_atomic(st, spath)
        h = pd.DataFrame([x for x in st["history"] if x["round"] == r])
        msg = (f"round {r} ({len(fit_sc)} scenes): max probe R² {h.r2.max():.3f} (layer "
               f"{int(h.loc[h.r2.idxmax(), 'layer'])}), {int((h.r2 >= R2_STOP).sum())} layers extended, "
               f"rank {h.rank_after.min()}–{h.rank_after.max()}")
        log(msg)
        notify(f"[{tag}] {msg}")
    hist = pd.DataFrame(st["history"])
    hist.to_csv(out / f"{run}_fit_history.csv", index=False)
    pd.DataFrame([{k: v for k, v in x.items() if k != "r2"} for x in st["dev"]]).to_csv(out / f"{run}_dev.csv",
                                                                                       index=False)
    bases = {l: st["bases"][l] for l in window}
    ranks = {l: B.shape[1] for l, (B, _) in bases.items()}

    # ---- random controls ----
    if "rand_matched" not in st:
        sig_sc = scenes_for(FIT_SEED, a.n_sigma)
        acts = cache_prefix(m, specs_for(sig_sc), window, batch_size=a.batch_size)["list"]
        st["rand_matched"], st["rand_info"] = {}, []
        for li, l in enumerate(window):
            X, _, _ = centered_xy(acts[:, li].float().numpy(), sig_sc, np.arange(len(sig_sc)))
            B = bases[l][0]
            R, alpha = variance_matched_random(X, B, seed=l)
            var = lambda M: float(((X @ M) ** 2).sum() / len(X))
            st["rand_matched"][l] = (R, st["mu_clean"][l] @ R)
            g = torch.Generator().manual_seed(l)
            I = torch.linalg.qr(torch.randn(d, max(ranks[l], 1), generator=g, dtype=torch.float64))[0][:, :ranks[l]]
            I = I.numpy()
            st.setdefault("rand_iso", {})[l] = (I, st["mu_clean"][l] @ I)
            st["rand_info"].append(dict(layer=l, rank=ranks[l], alpha=alpha, var_probe=var(B), var_matched=var(R),
                                        var_iso=var(I)))
        del acts
        save_atomic(st, spath)
        log("random controls built")
    info = pd.DataFrame(st["rand_info"])
    info.to_csv(out / f"{run}_random_controls.csv", index=False)
    conds = {"probe": as_t(bases), "matched": as_t(st["rand_matched"]), "iso": as_t(st["rand_iso"])}

    # ---- is the erasure effective? fresh scenes ----
    val_sc = scenes_for(VAL_SEED, a.n_val)
    val_specs = specs_for(val_sc)

    def validate():
        rows = []
        for cond in ("probe", "matched", "iso"):
            layers = window if cond == "probe" else mlp_layers
            acts = cache_prefix(m, val_specs, layers, batch_size=a.batch_size, erase=conds[cond],
                                erase_start=START)["list"]
            for li, l in enumerate(layers):
                X, Y, meta = centered_xy(acts[:, li].float().numpy(), val_sc, np.arange(len(val_sc)))
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
    notify(f"[{tag}] validation: max linear R² after erasure {max_val_r2:.3f} "
           f"(layer {int(vp.loc[vp.r2.idxmax(), 'layer'])}); rule needs < 0.10")

    # ---- the answers (new test set) ----
    items = chain_items(m, a.n_test, seed=TEST_SEED, exclude_qtypes=("closer_than",))
    flat = [(si, qi, s, q, comp, p7_class(q, s.root, mid, child))
            for si, (s, mid, child, qs) in enumerate(items) for qi, (q, comp) in enumerate(qs)]
    specs = [(build_prompt(s, q, m.tok, m.family), q) for _, _, s, q, _, _ in flat]
    truth = [answer_for(q, s.coords) for _, _, s, q, _, _ in flat]
    chance = np.array([1 / len(q.candidates) for _, _, _, q, _, _ in flat])
    log(f"{len(flat)} test questions in {len(items)} chains")

    def scored(cond, edit):
        preds = [predict(x) for x in score(m, specs, edit=edit, batch_size=a.batch_size)]
        return pd.DataFrame([dict(scene=si, qi=qi, qtype=q.qtype.value, composed=comp, cls=cls, cond=cond, pred=p,
                                  truth=t, correct=float(p == t), chance=ch)
                             for (si, qi, _, q, comp, cls), p, t, ch in zip(flat, preds, truth, chance)])

    dfs = []
    for cond, edit in [("clean", None)] + [(c, erase_edit(conds[c], start=START)) for c in conds]:
        path = out / f"{run}_test_{cond}.parquet"
        fresh = not path.exists()
        dd = df_step(path, lambda c=cond, e=edit: scored(c, e))
        dfs.append(dd)
        msg = f"{cond}: accuracy {dd.correct.mean():.3f} (composed {dd[dd.composed].correct.mean():.3f})"
        log(msg)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    df["family"] = df.qtype.map(FAMILY)
    df["subset"] = df.cls + " " + df.family
    df.to_parquet(out / f"{run}_test.parquet")

    sums, overall = [], []
    for ctrl, name in CONTROLS.items():
        d2 = df[df.cond.isin(["clean", "probe", ctrl])].replace({"cond": {ctrl: "random"}}).assign(config=name)
        sums.append(erase_summary(d2, group="subset"))
        overall.append(erase_summary(d2, group="composed"))
    s = pd.concat(sums, ignore_index=True)
    s.to_csv(out / f"{run}_summary.csv", index=False)
    so = pd.concat(overall, ignore_index=True)
    so.to_csv(out / f"{run}_summary_overall.csv", index=False)
    prim = s[(s.config == CONTROLS["matched"]) & (s.subset == "CR axis")].iloc[0]
    rc = df[(df.cond == "matched") & df.composed]
    comp_chance = float(df[(df.cond == "clean") & df.composed].chance.mean())
    v = verdict_axis(prim.excess, prim.excess_lo, max_val_r2=max_val_r2, random_comp_acc=float(rc.correct.mean()),
                     comp_chance=comp_chance)

    # ---- report ----
    last = hist[hist["round"] == hist["round"].max()] if len(hist) else hist
    mlp = val[val.probe == "mlp"].pivot_table(index="layer", columns="cond", values="r2")
    lin = val[val.probe == "linear"].pivot_table(index="layer", columns="cond", values="r2")
    dev_txt = "; ".join(f"seed {x['seed']} after round {x['after_round']}: max R² {x['max_r2']:.3f} (layer "
                        f"{x['layer']}){', added' if x['added'] else ''}" for x in st["dev"]) or "none"
    lines = [f"Model **{a.model}**. Erasure at every token except the first, at every layer {window[0]}–{window[-1]}, "
             f"fitted in closed loop from the closed-loop test's bases on {len(fit_sc)} scenes (seed {FIT_SEED}"
             f"{' + dev seeds ' + ', '.join(map(str, st['fit_seeds'])) if st['fit_seeds'] else ''}) down to refitted "
             f"probe R² < {R2_STOP}: {st['rounds']} rounds; dev checks: {dev_txt}. Rank per layer "
             f"{min(ranks.values())}–{max(ranks.values())} (total {sum(ranks.values())}). Matched random control: "
             f"α {info.alpha.min():.2f}–{info.alpha.max():.2f}; removed variance matched/probe "
             f"{(info.var_matched / info.var_probe.where(info.var_probe > 0)).min():.2f}–"
             f"{(info.var_matched / info.var_probe.where(info.var_probe > 0)).max():.2f}; isotropic/probe "
             f"{(info.var_iso / info.var_probe.where(info.var_probe > 0)).median():.2f} (median).", "",
             f"**Is the map gone?** Fresh scenes (seed {VAL_SEED}, {a.n_val}): largest linear-probe R² over the erased "
             f"layers **{max_val_r2:.3f}** (layer {int(vp.loc[vp.r2.idxmax(), 'layer'])}); layers ≥ 0.10: "
             f"{', '.join(str(int(l)) for l in vp[vp.r2 >= 0.10].layer) or 'none'}.", "",
             "| layer | linear, map erased | linear, matched random | linear, isotropic | MLP, map erased | "
             "MLP, matched random | MLP, isotropic |", "|---|---|---|---|---|---|---|"]
    for l in mlp_layers:
        lines.append(f"| {l} | {lin.loc[l, 'probe']:.2f} | {lin.loc[l, 'matched']:.2f} | {lin.loc[l, 'iso']:.2f} | "
                     f"{mlp.loc[l, 'probe']:.2f} | {mlp.loc[l, 'matched']:.2f} | {mlp.loc[l, 'iso']:.2f} |")
    lines += ["", f"**Do the answers suffer?** New test set ({len(items)} chains, seed {TEST_SEED}, {len(flat)} "
              "questions). Excess drop = random-erased accuracy − map-erased accuracy; 95% scene-bootstrap CIs. "
              "Primary: **CR axis** against the matched control.", "",
              "| control | questions | n | clean | map erased | random | excess drop [95% CI] |",
              "|---|---|---|---|---|---|---|"]
    for x in s.sort_values(["config", "subset"]).itertuples():
        star = " **(primary)**" if x.config == CONTROLS["matched"] and x.subset == "CR axis" else ""
        lines.append(f"| {x.config} | {x.subset}{star} | {x.n} | {x.clean:.2f} | {x.clean - x.drop_probe:.2f} | "
                     f"{x.clean - x.drop_random:.2f} | {x.excess:+.3f} [{x.excess_lo:+.3f}, {x.excess_hi:+.3f}] |")
    lines += ["", "Overall:", "", "| control | questions | n | clean | excess drop [95% CI] |", "|---|---|---|---|---|"]
    for x in so.itertuples():
        lines.append(f"| {x.config} | {'composed' if x.composed else 'stated only'} | {x.n} | {x.clean:.2f} | "
                     f"{x.excess:+.3f} [{x.excess_lo:+.3f}, {x.excess_hi:+.3f}] |")
    lines += ["", f"Matched random erasure, composed accuracy {rc.correct.mean():.2f} (chance {comp_chance:.2f}).",
              "", f"Preregistered verdict: **{v}**. Rules: docs/preregistration_erase_axis_confirm.md."]
    if not a.smoke:
        append_entry(f"Confirmatory erasure: do composed axis answers use the map? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v)
    notify(f"[{tag}] COMPLETE. {v}. CR axis excess drop {prim.excess:+.3f} "
           f"[{prim.excess_lo:+.3f}, {prim.excess_hi:+.3f}]")
    print("\n".join(lines))
    print(f"verdict: {v}")


if __name__ == "__main__":
    main()
