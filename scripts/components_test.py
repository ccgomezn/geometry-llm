"""Stage 2 (docs/preregistration_components.md): which attention heads and MLPs write the answer of "which is farther
east / north, C or R?" at the final position, how does each weigh C's own direction and M's step, and is there a
correct path underneath the shortcut once its writers are ablated?

Discovery set (seed 1701): direct logit attribution per component and per source token; the components are ranked
there. Test set (seed 1801): mean ablations of the selected components, scored on new questions."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.analysis import bootstrap_ci
from geomap.axis import CELLS, axis_bulk_items
from geomap.checkpoint import notify
from geomap.components import (FEATURES, GROUPS, ablation_ctx, betas, component_dla, design, group_ids, head_shape,
                               pick_matched, verdict_components)
from geomap.flow import axis_of, steps
from geomap.models import load
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry
from geomap.scoring import candidate_ids, score

SEED_DISC, SEED_TEST = 1701, 1801
SIZES_DISC = {c: 300 for c in CELLS}
SIZES_TEST = {"own step only": 300, "agree": 300, "first step only": 500, "conflict, C larger": 300,
              "conflict, M larger": 500}
K_MAIN, K_ALL, K_FIRST, ALPHA, N_RAND = 10, (5, 10, 20), 10, 3.0, 3
DECISIVE = ("conflict, M larger", "first step only")
SHORTCUT_RIGHT = ("own step only", "agree", "conflict, C larger")


def build(m, sizes, seed):
    items = axis_bulk_items(m, sizes, seed=seed)
    rows, specs, qs, groups = [], [], [], []
    for si, (s, mid, child, qq) in enumerate(items):
        for q, cell in qq:
            d1, d2, truth = steps(s, mid, child, q)
            sp = build_prompt(s, q, m.tok, m.family)
            rows.append(dict(scene=si, cell=cell, axis=axis_of(q), c_first=q.args[0] == child, d1=d1, d2=d2,
                             truth=truth, child=child, root=s.root, mid=mid))
            specs.append(sp)
            qs.append(q)
            groups.append(group_ids(sp, mid, child, s.root))
    meta = pd.DataFrame(rows)
    for name in set(meta.child) | set(meta.root):
        assert len(candidate_ids(m.tok, name)) == 1, name          # one scoring row per question
    return meta, specs, qs, groups


def name(c):
    kind, l, h = c
    return f"L{l}H{h}" if kind == "head" else f"MLP{l}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, few questions, small selections; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/components)")
    a = ap.parse_args()
    out, sd, stt, k_main, k_all, k_first, n_boot = (ROOT / "results" / "components", dict(SIZES_DISC), dict(SIZES_TEST),
                                                    K_MAIN, K_ALL, K_FIRST, 1000)
    if a.smoke:
        import tempfile
        a.model, out, n_boot = "tiny", Path(tempfile.mkdtemp()), 50
        sd, stt = {c: 8 for c in CELLS}, {c: 8 for c in CELLS}
        k_main, k_all, k_first = 2, (1, 2, 3), 2
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_components"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[components] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[components] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    L, (H, hd, _) = m.n_layers, head_shape(m)
    tok1 = lambda names: [candidate_ids(m.tok, n)[0] for n in names]

    # ---- discovery: direct attribution ----
    dm, dspecs, dqs, dgroups = build(m, sd, SEED_DISC)
    log(f"discovery: {len(dm)} questions")
    notify(f"[Components {a.model}] started: {len(dm)} discovery questions")
    path = out / f"{run}_dla.npz"
    if path.exists():
        d = dict(np.load(path))
    else:
        d = component_dla(m, [s.input_ids for s in dspecs], dgroups, len(GROUPS), tok1(dm.child), tok1(dm.root),
                          a.batch_size)
        np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in d.items()})
    total = d["emb"] + d["heads"].sum((1, 2)) + d["mlp"].sum(1)
    chk_r, chk_max = float(np.corrcoef(total, d["ld"])[0, 1]), float(np.abs(total - d["ld"]).max())
    chk_src = float(np.abs(d["src"].sum(-1) - d["heads"]).max())
    cap = getattr(getattr(m.hf.config, "text_config", m.hf.config), "final_logit_softcapping", None)
    log(f"attribution: sum vs model r = {chk_r:.4f}, max |diff| = {chk_max:.3f}; sources vs heads {chk_src:.4f}")

    F = design(dm)
    comps = [("emb", -1, -1)] + [("head", l, h) for l in range(L) for h in range(H)] + [("mlp", l, -1) for l in range(L)]
    Y = np.c_[d["emb"][:, None], d["heads"].reshape(len(dm), -1), d["mlp"]]
    B = betas(Y, F)
    bt = betas(d["ld"][:, None], F)[0]
    C = pd.DataFrame(dict(kind=[c[0] for c in comps], layer=[c[1] for c in comps], head=[c[2] for c in comps],
                          mean=Y.mean(0), mean_abs=np.abs(Y).mean(0),
                          **{f"b_{f}": B[:, i + 1] for i, f in enumerate(FEATURES)}))
    C.to_csv(out / f"{run}_components.csv", index=False)
    cand = C[C.kind != "emb"]
    key = lambda r: (r.kind, int(r.layer), int(r.head))
    own_rank = [key(r) for r in cand.sort_values("b_own_yes", ascending=False).itertuples()]
    first_rank = [key(r) for r in cand.sort_values("b_d1", ascending=False).itertuples()]
    own = {k: own_rank[:k] for k in k_all}
    first = first_rank[:k_first]
    idx = {c: i for i, c in enumerate(comps)}
    share_own = float(B[[idx[c] for c in own[k_main]], 1].sum() / bt[1])
    rest = [i for c, i in idx.items() if c not in set(own[k_main])]
    rest_b = B[rest].sum(0)

    # where each top head's signal comes from (by source-token group)
    src_rows = []
    for c in dict.fromkeys(own[max(k_all)][:15] + first[:15]):
        if c[0] != "head":
            continue
        Bs = betas(d["src"][:, c[1], c[2], :], F)
        for g, gname in enumerate(GROUPS):
            src_rows.append(dict(component=name(c), group=gname, b_own_yes=Bs[g, 1], b_d1=Bs[g, 2], b_d2=Bs[g, 3],
                                 mean=float(d["src"][:, c[1], c[2], g].mean())))
    S = pd.DataFrame(src_rows)
    S.to_csv(out / f"{run}_sources.csv", index=False)

    def top_sources(c, col):
        q = S[S.component == name(c)]
        if not len(q) or abs(q[col].sum()) < 1e-9:
            return "–"
        q = q.assign(share=q[col] / q[col].sum()).sort_values("share", ascending=False)
        return ", ".join(f"{r.group} {r.share:.2f}" for r in q.head(2).itertuples())
    log("discovery analysed")

    # ---- test: mean ablations on new questions ----
    tm, tspecs, tqs, _ = build(m, stt, SEED_TEST)
    finals = [len(s.input_ids) - 1 for s in tspecs]
    zt, mt = torch.tensor(d["z_mean"]), torch.tensor(d["mlp_mean"])

    def refs(sel):
        heads, mlps = {}, {}
        for kind, l, h in sel:
            if kind == "head":
                heads.setdefault(l, {})[h] = zt[l, h * hd:(h + 1) * hd]
            else:
                mlps[l] = mt[l]
        return heads, mlps

    rng = np.random.default_rng(0)
    excl = set(own[max(k_all)]) | set(first)
    rand = [pick_matched(own[k_main], L, H, rng, excl) for _ in range(N_RAND)]
    conds = {"clean": None, **{f"own{k}": (own[k], None) for k in k_all},
             **{f"random{k_main}_{i}": (r, None) for i, r in enumerate(rand)},
             f"first{k_first}": (first, None), f"first{k_first}_x{ALPHA:g}": (first, ALPHA)}
    res = {}
    cpath = out / f"{run}_test.csv"
    T = pd.read_csv(cpath) if cpath.exists() else None
    if T is None:
        for cname, spec in conds.items():
            ctx = None if spec is None else ablation_ctx(m, finals, *refs(spec[0]), alpha=spec[1])
            sc = score(m, list(zip(tspecs, tqs)), batch_size=a.batch_size, batch_ctx=ctx)
            res[cname] = [x[r.child] - x[r.root] for x, r in zip(sc, tm.itertuples())]
            log(f"test: {cname} scored")
        T = tm.copy()
        for cname, v in res.items():
            T[f"ld_{cname}"] = v
        T.to_csv(cpath, index=False)
    for cname in conds:
        T[f"ok_{cname}"] = (np.sign(T[f"ld_{cname}"]) == T.truth).astype(float)
    T["ok_random"] = T[[f"ok_random{k_main}_{i}" for i in range(N_RAND)]].mean(1)
    T["ld_random"] = T[[f"ld_random{k_main}_{i}" for i in range(N_RAND)]].mean(1)

    # ---- verdict ----
    dec, sr = T.cell.isin(DECISIVE).to_numpy(), T.cell.isin(SHORTCUT_RIGHT).to_numpy()
    o, rnd = f"ok_own{k_main}", "ok_random"
    drop = float(T.loc[sr, rnd].mean() - T.loc[sr, o].mean())
    acc, acc_lo, acc_hi = bootstrap_ci(T.loc[dec, o].to_numpy(), n_boot=n_boot, groups=T.scene[dec].to_numpy())
    gain, gain_lo, gain_hi = bootstrap_ci((T.loc[dec, o] - T.loc[dec, rnd]).to_numpy(), n_boot=n_boot,
                                          groups=T.scene[dec].to_numpy())
    v = verdict_components(drop, acc, acc_hi, gain, gain_lo)
    Ft = design(T)
    show = ["clean"] + [f"own{k}" for k in k_all] + ["random", f"first{k_first}", f"first{k_first}_x{ALPHA:g}"]
    reg = {c: betas(T[f"ld_{c}"].to_numpy()[:, None], Ft)[0] for c in show}

    # ---- report ----
    lines = [f"Model **{a.model}**. Discovery: {len(dm)} new C–R axis questions (seed {SEED_DISC}); test: {len(tm)} "
             f"(seed {SEED_TEST}). Direct contributions to logit(C) − logit(R) at the final position; each "
             "component's contribution regressed on own_yes (C's own step points the asked way), d1 (M's step, per "
             "unit), d2 (C's step, per unit) and the question order. Logit units.", "",
             f"Checks: components sum to the model's logit difference r = {chk_r:.4f} (max |diff| {chk_max:.2f}); "
             f"head parts by source token sum to the head {chk_src:.4f}; final logit softcap {cap}.", "",
             f"**The model's answer** (discovery, same regression): own_yes {bt[1]:+.1f}, d1 {bt[2]:+.2f}/unit, "
             f"d2 {bt[3]:+.2f}/unit, order {bt[4]:+.2f}.", "",
             f"**Own-direction writers** (top {max(k_all)} by own_yes; the top {k_main} carry {share_own:.0%} of the "
             "total own_yes weight):", "",
             "| component | own_yes | d1 | d2 | mean | own_yes comes from |", "|---|---|---|---|---|---|"]
    for c in own[max(k_all)]:
        r = B[idx[c]]
        lines.append(f"| {name(c)} | {r[1]:+.2f} | {r[2]:+.3f} | {r[3]:+.3f} | {Y[:, idx[c]].mean():+.2f} | "
                     f"{top_sources(c, 'b_own_yes')} |")
    lines += ["", f"**First-step writers** (top {k_first} by d1):", "",
              "| component | own_yes | d1 | d2 | d1 comes from |", "|---|---|---|---|---|"]
    for c in first:
        r = B[idx[c]]
        lines.append(f"| {name(c)} | {r[1]:+.2f} | {r[2]:+.3f} | {r[3]:+.3f} | {top_sources(c, 'b_d1')} |")
    lines += ["", f"Everything except the top {k_main} own-direction writers, summed: own_yes {rest_b[1]:+.1f}, "
              f"d1 {rest_b[2]:+.2f}/unit, d2 {rest_b[3]:+.2f}/unit.", "",
              "**Test set, mean ablation at the final position** (accuracy):", "",
              "| condition | " + " | ".join(CELLS) + " | regression: own_yes / d1 / d2 |",
              "|---" * (len(CELLS) + 2) + "|"]
    for c in show:
        accs = T.groupby("cell")[f"ok_{c}"].mean()
        lines.append(f"| {c} | " + " | ".join(f"{accs[x]:.2f}" for x in CELLS)
                     + f" | {reg[c][1]:+.1f} / {reg[c][2]:+.2f} / {reg[c][3]:+.2f} |")
    lines += ["", f"(random = {N_RAND} draws of {k_main} matched components, averaged per question.)", "",
              f"Shortcut-right questions lost beyond the matched control: {drop:+.2f}. Decisive questions "
              f"(*conflict, M larger* and *first step only*) with the top {k_main} ablated: {acc:.2f} [{acc_lo:.2f}, "
              f"{acc_hi:.2f}]; minus the control {gain:+.2f} [{gain_lo:+.2f}, {gain_hi:+.2f}].", "",
              f"Preregistered verdict: **{v}**. Rules: docs/preregistration_components.md."]
    if not a.smoke:
        append_entry(f"Stage 2: which components write the answer? ({a.model}) — preregistered", "\n".join(lines), [])
    done.write_text(v)
    notify(f"[Components {a.model}] COMPLETE. verdict: {v}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
