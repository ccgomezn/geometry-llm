"""Stage 2b (docs/preregistration_components2.md): remove the shortcut from its writers while keeping the names (twin
mean: the average of each component's value on the question and on its twin, the same question with C's own step
negated), ask whether the rest of the model answers from both steps, and measure which components rewrite the
shortcut (exact attribution under the removal)."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.analysis import bootstrap_ci
from geomap.axis import axis_bulk_items
from geomap.checkpoint import notify
from geomap.components import (GROUPS, ablation_ctx, betas, capture_final, component_dla, design_truth, group_ids,
                               head_shape, pick_matched, repair_groups, verdict_components2)
from geomap.flow import axis_of, flip_step, steps
from geomap.models import load
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry
from geomap.scoring import candidate_ids, score

SEED = 1901
SIZES = {"own step only": 300, "agree": 300, "conflict, C larger": 300, "conflict, M larger": 600}
CELLS2 = tuple(SIZES)
# Stage 2's discovery ranking (results/components/gemma3-12b_components_components.csv), fixed in the preregistration
OWN20 = [("head", 44, 10), ("head", 47, 2), ("head", 45, 1), ("head", 41, 0), ("head", 41, 11), ("head", 42, 0),
         ("head", 41, 5), ("head", 41, 3), ("head", 35, 8), ("head", 47, 10), ("head", 47, 8), ("head", 35, 3),
         ("head", 41, 4), ("mlp", 40, -1), ("head", 35, 12), ("head", 46, 4), ("head", 47, 7), ("mlp", 43, -1),
         ("head", 41, 1), ("head", 32, 7)]
FIRST10 = [("head", 41, 0), ("head", 44, 10), ("head", 45, 1), ("head", 47, 2), ("head", 41, 11), ("head", 42, 0),
           ("head", 47, 10), ("head", 35, 8), ("head", 43, 0), ("head", 47, 8)]
N_RAND, DLA_PER_CELL, DECISIVE = 3, 150, "conflict, M larger"


def build(m, sizes, seed):
    """Test questions and their twins (C's own step on the asked axis negated); questions whose twin collides are
    dropped."""
    rows, specs, twins, qs, groups, dropped = [], [], [], [], [], 0
    for si, (s, mid, child, qq) in enumerate(axis_bulk_items(m, sizes, seed=seed)):
        for q, cell in qq:
            d1, d2, truth = steps(s, mid, child, q)
            try:
                tw = flip_step(s, child, axis_of(q))
            except ValueError:
                dropped += 1
                continue
            assert steps(tw, mid, child, q)[:2] == (d1, -d2)
            sp = build_prompt(s, q, m.tok, m.family)
            rows.append(dict(scene=si, cell=cell, axis=axis_of(q), c_first=q.args[0] == child, d1=d1, d2=d2,
                             truth=truth, child=child, root=s.root, mid=mid))
            specs.append(sp)
            twins.append(build_prompt(tw, q, m.tok, m.family))
            qs.append(q)
            groups.append(group_ids(sp, mid, child, s.root))
    meta = pd.DataFrame(rows)
    for name in set(meta.child) | set(meta.root):
        assert len(candidate_ids(m.tok, name)) == 1, name
    return meta, specs, twins, qs, groups, dropped


def name(c):
    kind, l, h = c
    return f"L{l}H{h}" if kind == "head" else f"MLP{l}"


def own_coef(ld, F):
    return float(betas(np.asarray(ld, float)[:, None], F)[0, 1])


def own_coef_ci(ld, F, scenes, n_boot, seed=0):
    """own_yes coefficient with a scene-bootstrap 95% CI."""
    rng, ld = np.random.default_rng(seed), np.asarray(ld, float)
    uniq = np.unique(scenes)
    members = {g: np.flatnonzero(scenes == g) for g in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        boots.append(own_coef(ld[idx], F[idx]))
    return own_coef(ld, F), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, few questions, small selections; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/components)")
    a = ap.parse_args()
    out, sizes, own20, n_boot, dla_per_cell = ROOT / "results" / "components", dict(SIZES), list(OWN20), 1000, DLA_PER_CELL
    first10 = list(FIRST10)
    if a.smoke:
        import tempfile
        a.model, out, n_boot, dla_per_cell = "tiny", Path(tempfile.mkdtemp()), 50, 4
        sizes = {c: 6 for c in CELLS2}
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_components2"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[components2] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[components2] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    m.hf.set_attn_implementation("eager")             # the same attention code for capture, scoring and attribution
    L, (H, hd, _) = m.n_layers, head_shape(m)
    if a.smoke:
        own20 = [("head", L - 1, 0), ("head", L - 1, 1), ("mlp", L - 1, -1), ("head", 0, 2)]
        first10 = [("head", L - 1, 3)]
    own10 = own20[:max(1, len(own20) // 2)]
    tok1 = lambda names: [candidate_ids(m.tok, n)[0] for n in names]

    # ---- questions, twins, matched controls ----
    tm, specs, twins, qs, groups, dropped = build(m, sizes, SEED)
    N = len(tm)
    log(f"{N} test questions with twins ({dropped} dropped: twin collides)")
    notify(f"[Components2 {a.model}] started: {N} questions")
    rng = np.random.default_rng(0)
    excl = set(own20) | set(first10)
    rand = {20: [pick_matched(own20, L, H, rng, excl) for _ in range(N_RAND)],
            10: [pick_matched(own10, L, H, rng, excl) for _ in range(N_RAND)]}
    sets = {"top20": own20, "top10": own10, **{f"rand20_{i}": r for i, r in enumerate(rand[20])},
            **{f"rand10_{i}": r for i, r in enumerate(rand[10])}}
    allc = list(dict.fromkeys(c for s in sets.values() for c in s))

    # ---- clean values on the question and on its twin ----
    cpath = out / f"{run}_capture.npz"
    if cpath.exists():
        z = np.load(cpath)
        cap = {c: (z[f"o_{name(c)}"], z[f"t_{name(c)}"]) for c in allc}
    else:
        co = capture_final(m, [s.input_ids for s in specs], allc, a.batch_size)
        ct = capture_final(m, [s.input_ids for s in twins], allc, a.batch_size)
        cap = {c: (co[c], ct[c]) for c in allc}
        np.savez_compressed(cpath, **{f"o_{name(c)}": v[0] for c, v in cap.items()},
                            **{f"t_{name(c)}": v[1] for c, v in cap.items()})
    log("captured clean values on questions and twins")
    finals = [len(s.input_ids) - 1 for s in specs]

    def refs(sel, mode, idx=None):
        heads, mlps = {}, {}
        for c in sel:
            o, t = cap[c]
            v = (o + t) / 2 if mode == "mean" else t
            v = torch.tensor(v if idx is None else v[idx], dtype=torch.float32)
            if c[0] == "head":
                heads.setdefault(c[1], {})[c[2]] = v
            else:
                mlps[c[1]] = v
        return heads, mlps

    conds = {"clean": None, **{f"twinmean_{k}": (s, "mean") for k, s in sets.items()},
             "swap_top20": (own20, "swap"), "swap_top10": (own10, "swap")}
    tpath = out / f"{run}_test.csv"
    if tpath.exists():
        T = pd.read_csv(tpath)
    else:
        T = tm.copy()
        for cname, spec in conds.items():
            ctx = None if spec is None else ablation_ctx(m, finals, *refs(*spec))
            sc = score(m, list(zip(specs, qs)), batch_size=a.batch_size, batch_ctx=ctx)
            T[f"ld_{cname}"] = [x[r.child] - x[r.root] for x, r in zip(sc, tm.itertuples())]
            log(f"scored: {cname}")
        T.to_csv(tpath, index=False)
    for k in (20, 10):
        T[f"ld_twinmean_rand{k}"] = T[[f"ld_twinmean_rand{k}_{i}" for i in range(N_RAND)]].mean(1)
    lds = [c for c in T.columns if c.startswith("ld_")]
    for c in lds:
        T["ok_" + c[3:]] = (np.sign(T[c]) == T.truth).astype(float)
    for k in (20, 10):
        T[f"ok_twinmean_rand{k}"] = T[[f"ok_twinmean_rand{k}_{i}" for i in range(N_RAND)]].mean(1)

    # ---- rule ----
    F = design_truth(T)
    scenes = T.scene.to_numpy()
    own_yes = F[:, 1]
    dec = (T.cell == DECISIVE).to_numpy()

    def rule(k):
        b_abl, b_ctl = own_coef(T[f"ld_twinmean_top{k}"], F), own_coef(T[f"ld_twinmean_rand{k}"], F)
        ratio = b_abl / b_ctl
        acc, acc_lo, acc_hi = bootstrap_ci(T.loc[dec, f"ok_twinmean_top{k}"].to_numpy(), n_boot=n_boot,
                                           groups=scenes[dec])
        gain, gain_lo, gain_hi = bootstrap_ci((T.loc[dec, f"ok_twinmean_top{k}"] - T.loc[dec, f"ok_twinmean_rand{k}"])
                                              .to_numpy(), n_boot=n_boot, groups=scenes[dec])
        return dict(b_abl=b_abl, b_ctl=b_ctl, ratio=ratio, acc=acc, acc_lo=acc_lo, acc_hi=acc_hi, gain=gain,
                    gain_lo=gain_lo, gain_hi=gain_hi, verdict=verdict_components2(ratio, acc, acc_hi, gain, gain_lo))
    R20, R10 = rule(20), rule(10)
    v = R20["verdict"]
    show = ["clean", "twinmean_top20", "twinmean_rand20", "twinmean_top10", "twinmean_rand10", "swap_top20",
            "swap_top10"]
    reg = {c: betas(T[f"ld_{c}"].to_numpy()[:, None], F)[0] for c in show}
    ci = {c: own_coef_ci(T[f"ld_{c}"], F, scenes, n_boot) for c in ("clean", "twinmean_top20", "twinmean_rand20")}
    lin = T.ld_clean - reg["clean"][1] * own_yes                     # shortcut removed, nothing else changed
    T["ok_linear"] = (np.sign(lin) == T.truth).astype(float)
    follows = {c: ((T[f"ld_{c}"] > 0) == (own_yes == 0)).groupby(T.cell).mean() for c in ("swap_top20", "swap_top10",
                                                                                          "clean")}
    log("test analysed")

    # ---- who rewrites the shortcut: attribution on a subset, clean and under removal ----
    sub = np.sort(np.concatenate([np.flatnonzero((T.cell == c).to_numpy())[:dla_per_cell] for c in CELLS2]))
    Fs = F[sub]
    comps = [("emb", -1, -1)] + [("head", l, h) for l in range(L) for h in range(H)] + [("mlp", l, -1) for l in range(L)]

    def dla(tag, sel):
        p = out / f"{run}_dla_{tag}.npz"
        if p.exists():
            return dict(np.load(p))
        ctx = None if sel is None else ablation_ctx(m, [finals[i] for i in sub], *refs(sel, "mean", sub))
        d = component_dla(m, [specs[i].input_ids for i in sub], [groups[i] for i in sub], len(GROUPS),
                          tok1(T.child.iloc[sub]), tok1(T.root.iloc[sub]), a.batch_size, batch_ctx=ctx)
        d = {k: d[k] for k in ("emb", "heads", "mlp", "ld", "rf")}
        np.savez_compressed(p, **{k: x.astype(np.float32) for k, x in d.items()})
        log(f"attribution: {tag}")
        return d

    Yof = lambda d: np.c_[d["emb"][:, None], d["heads"].reshape(len(sub), -1), d["mlp"]]
    dc = dla("clean", None)
    Yc = Yof(dc)
    chk = float(np.abs(Yc.sum(1) - dc["ld"]).max())
    repair = {}
    for k, sel in (("top20", own20), ("top10", own10)):
        d = dla(k, sel)
        Ya = Yof(d)
        Yfix = Ya * (dc["rf"] / d["rf"])[:, None]                    # at the clean run's final-norm scale
        Bc, Bf = betas(Yc, Fs)[:, 1], betas(Yfix, Fs)[:, 1]
        g = repair_groups(Bf - Bc, comps, sel)
        g["final_norm"] = own_coef((Ya - Yfix).sum(1), Fs)
        total = own_coef(d["ld"], Fs) - own_coef(dc["ld"], Fs)
        delta = pd.Series(Bf - Bc, index=[name(c) if c[0] != "emb" else "emb" for c in comps])
        top = delta.drop([name(c) for c in sel]).sort_values(ascending=False).head(10)
        repair[k] = dict(groups=g, total=total, clean=own_coef(dc["ld"], Fs), abl=own_coef(d["ld"], Fs), top=top,
                         scale=float(np.mean(d["rf"] / dc["rf"])), chk=float(np.abs(Ya.sum(1) - d["ld"]).max()),
                         removed_clean=float(Bc[[comps.index(c) for c in sel]].sum()))
        pd.DataFrame(dict(component=delta.index, b_own_clean=Bc, b_own_removed=Bf, change=Bf - Bc)).to_csv(
            out / f"{run}_repair_{k}.csv", index=False)

    # ---- report ----
    pct = lambda x: f"{x:.2f}"
    lines = [f"Model **{a.model}**. {N} new C–R axis questions (seed {SEED}; {dropped} dropped because the twin "
             "collides), each with a twin: the same question with C's own step on the asked axis negated. Components "
             "fixed from Stage 2. *Twin mean* = each component's value at the final position replaced by the average "
             "of its clean values on the question and the twin; *swap* = by its value in the twin. Shortcut = "
             "own_yes coefficient of logit(C) − logit(R), regressed on own_yes, truth, d1, d2, order. Matched "
             f"controls: {N_RAND} draws each (same kind and layer).", "",
             "| condition | " + " | ".join(CELLS2) + " | shortcut | truth | d1 | d2 |", "|---" * (len(CELLS2) + 5) + "|"]
    for c in show:
        accs = T.groupby("cell")[f"ok_{c}"].mean()
        r = reg[c]
        lines.append(f"| {c} | " + " | ".join(pct(accs[x]) for x in CELLS2)
                     + f" | {r[1]:+.1f} | {r[2]:+.1f} | {r[3]:+.2f} | {r[4]:+.2f} |")
    accs = T.groupby("cell").ok_linear.mean()
    lines.append("| linear prediction | " + " | ".join(pct(accs[x]) for x in CELLS2) + " | 0 | | | |")
    lines += ["", "Shortcut with scene-bootstrap 95% CI: " + "; ".join(
        f"{c} {x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]" for c, x in ci.items()) + ".", "",
        "*Linear prediction* = the clean logit difference minus the clean shortcut term: the answers if the shortcut "
        "were removed and nothing else changed.", "",
        "**Swap**, share of answers that follow the twin's shortcut (C iff the twin's C step points the asked way): "
        + "; ".join(f"{c}: " + ", ".join(f"{x} {follows[c][x]:.2f}" for x in CELLS2) for c in follows) + ".", ""]
    for k, R in (("top 20", R20), ("top 10", R10)):
        lines.append(f"**{k}** (twin mean vs matched control): shortcut {R['b_abl']:+.1f} vs {R['b_ctl']:+.1f} (ratio "
                     f"{R['ratio']:.2f}); *conflict, M larger* accuracy {R['acc']:.2f} [{R['acc_lo']:.2f}, "
                     f"{R['acc_hi']:.2f}], minus control {R['gain']:+.2f} [{R['gain_lo']:+.2f}, {R['gain_hi']:+.2f}]; "
                     f"rule gives *{R['verdict']}*.")
    lines += ["", f"**Who rewrites the shortcut** (exact attribution on {len(sub)} test questions, {dla_per_cell} per "
              f"cell; attribution sums to the model, max |diff| {chk:.3f} clean). Change in the own_yes coefficient "
              "by group, with the removal; components taken at the clean run's final-norm scale, the rescaling by "
              "the final norm counted separately:", "",
              "| removed | shortcut clean → removed | removed components | other heads, same layers | later "
              "components | earlier (should be 0) | final norm | final-norm scale removed ÷ clean |",
              "|---|---|---|---|---|---|---|---|"]
    for k, r in repair.items():
        g = r["groups"]
        lines.append(f"| {k} | {r['clean']:+.1f} → {r['abl']:+.1f} | {g['removed']:+.1f} | {g['same_layer_heads']:+.1f}"
                     f" | {g['later']:+.1f} | {g['earlier']:+.2f} | {g['final_norm']:+.1f} | {r['scale']:.3f} |")
    for k, r in repair.items():
        lines += ["", f"Components whose shortcut grows most when the {k} are removed: "
                  + ", ".join(f"{i} {x:+.2f}" for i, x in r["top"].items()) + "."]
    lines += ["", f"Preregistered verdict (top 20): **{v}**. Rules: docs/preregistration_components2.md."]
    if not a.smoke:
        append_entry(f"Stage 2b: removing the shortcut while keeping the names; who rewrites it? ({a.model}) — "
                     "preregistered", "\n".join(lines), [])
    done.write_text(v)
    notify(f"[Components2 {a.model}] COMPLETE. verdict: {v}")
    print("\n".join(lines))
    for k, r in repair.items():
        print(f"[check] {k}: group sum {sum(r['groups'].values()):+.3f} vs total change {r['total']:+.3f}; "
              f"attribution under removal max |diff| {r['chk']:.3f}")


if __name__ == "__main__":
    main()
