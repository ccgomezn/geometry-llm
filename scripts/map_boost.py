"""Map boost at the answer position (docs/preregistration_map_boost.md).

Locate: ridge probes for M's step (d1) and C's own step (d2), per axis, at the final position of the flow set's chat
prompts (seed 1601), at two layers (the first layer of the final window, and 4 later). Amplify: on new scenes (seed
1701), h ← h + (α − 1) · P(h − μ) at one layer, final position only, for the whole map (4 directions) or M's step
only (2 directions), α ∈ {2, 4, 8}; three variance-matched random subspaces per condition as control. Measure: balanced
accuracy per question kind, the decisive contrast with the control, and transfer to other name questions.

Resumable: residuals, subspaces and every condition's answers are cached under results/map_boost."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.axis import CELLS, axis_bulk_items
from geomap.boost import (DEC, EASY, FIRST, ba_boot, boost_edit, condition_works, feasible_k, map_directions,
                          matched_random_subspaces, orthonormal, transfer_items, transfer_verdict, verdict_boost)
from geomap.cache import cache_prefix
from geomap.checkpoint import notify
from geomap.flow import axis_of, axis_r2, layer_grid, steps
from geomap.models import load
from geomap.probes import scene_split
from geomap.prompts import build_prompt
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scenes import relation_text
from geomap.scoring import score

SEED_FIT, SEED_TEST = 1601, 1701
SIZES_FIT = {"own step only": 400, "agree": 400, "first step only": 600, "conflict, C larger": 400,
             "conflict, M larger": 600}
SIZES_TEST = {"own step only": 200, "agree": 200, "first step only": 300, "conflict, C larger": 200,
              "conflict, M larger": 300}
TRANSFER_PER_KIND = 150
ALPHAS = (2.0, 4.0, 8.0)
ARMS = {"map": [0, 1, 2, 3], "M's step": [0, 2]}       # columns of map_directions: d1-east, d2-east, d1-north, d2-north
N_CTRL, K_PCS = 3, 64
TKINDS = ("C vs M", "M vs R", "farthest", "closest")


def final_residuals(m, specs, layers, batch_size) -> torch.Tensor:
    sp = [type(s)(**{**s.__dict__, "list_pos": {"final": len(s.input_ids) - 1}}) for s in specs]
    return cache_prefix(m, sp, layers, batch_size)["list"][:, :, 0].clone()


def fit_set(m, sizes):
    items = axis_bulk_items(m, sizes, seed=SEED_FIT)
    rows, specs = [], []
    for si, (s, mid, child, qq) in enumerate(items):
        for q, cell in qq:
            d1, d2, truth = steps(s, mid, child, q)
            rows.append(dict(scene=si, cell=cell, axis=axis_of(q), d1=d1, d2=d2))
            specs.append(build_prompt(s, q, m.tok, m.family))
    return pd.DataFrame(rows), specs, items


def test_set(m, sizes, fit_items, per_kind):
    items = axis_bulk_items(m, sizes, seed=SEED_TEST)
    key = lambda s: tuple(relation_text(r) for r in s.relations)
    assert not {key(s) for s, *_ in items} & {key(s) for s, *_ in fit_items}, "test scenes overlap the fitting set"
    rows, qs = [], []
    for si, (s, mid, child, qq) in enumerate(items):
        for q, cell in qq:
            d1, d2, truth = steps(s, mid, child, q)
            rows.append(dict(scene=si, kind="C-R", cell=cell, axis=axis_of(q), d1=d1, d2=d2, truth=truth,
                             child=child, root=s.root, mid=mid, answer=answer_for(q, s.coords), question=q.text))
            qs.append((s, q))
    tr = [(si, s, mid, child, q, kind) for si, (s, mid, child, _) in enumerate(items)
          for q, kind in transfer_items(s, mid, child)]
    rng = np.random.default_rng(SEED_TEST)
    keep = []
    for kind in TKINDS:
        idx = [i for i, t in enumerate(tr) if t[5] == kind]
        keep += sorted(rng.permutation(idx)[:per_kind].tolist())
    for i in sorted(keep):
        si, s, mid, child, q, kind = tr[i]
        rows.append(dict(scene=si, kind=kind, cell="", axis=-1, d1=0, d2=0, truth=0, child=child, root=s.root,
                         mid=mid, answer=answer_for(q, s.coords), question=q.text))
        qs.append((s, q))
    specs = [build_prompt(s, q, m.tok, m.family) for s, q in qs]
    return pd.DataFrame(rows), specs, [q for _, q in qs], len(items)


def locate(m, a, out, layers, sizes, log):
    """Directions, means, held-out R² and control subspaces per layer and arm (cached)."""
    path = out / f"{a.model}_subspaces.pt"
    if path.exists():
        return torch.load(path, weights_only=False)      # our own cache: it also holds the fitting scenes
    meta, specs, items = fit_set(m, sizes)
    apath = Path(a.acts_dir or out) / f"{a.model}_fit_acts.pt"
    acts = torch.load(apath) if apath.exists() else final_residuals(m, specs, layers, a.batch_size)
    torch.save(acts, apath)
    log(f"fitting residuals {tuple(acts.shape)}")
    tr = np.isin(meta.scene.to_numpy(), scene_split(len(items))[0])
    g, ax = meta.scene.to_numpy(), meta.axis.to_numpy()
    res = {"items": items, "layers": {}}
    for li, layer in enumerate(layers):
        X = acts[:, li].float().numpy()
        D = map_directions(X, meta.d1, meta.d2, ax, g)
        lay = dict(mu=torch.as_tensor(X.mean(0), dtype=torch.float32),
                   r2_d1=axis_r2(X, meta.d1, ax, g, tr), r2_d2=axis_r2(X, meta.d2, ax, g, tr),
                   h_norm=float(np.linalg.norm(X, axis=1).mean()), arms={})
        for ai, (arm, cols) in enumerate(ARMS.items()):
            Q = orthonormal(D[:, cols]).float()
            k = feasible_k(X, Q, K_PCS)
            ctrl = matched_random_subspaces(X, Q, k=k, n=N_CTRL, tol=0.1, seed=100 * li + ai)
            Xc = X - X.mean(0)
            sub_norm = lambda B: float(np.linalg.norm(Xc @ B.double().numpy() @ B.double().numpy().T, axis=1).mean())
            lay["arms"][arm] = dict(Q=Q, ctrl=[c.float() for c in ctrl], k=k, norm=sub_norm(Q),
                                    ctrl_norm=float(np.mean([sub_norm(c) for c in ctrl])))
        res["layers"][layer] = lay
        log(f"layer {layer}: held-out R² d1 {lay['r2_d1']:.2f}, d2 {lay['r2_d2']:.2f}; control components "
            + ", ".join(f"{k}: {v['k']}" for k, v in lay["arms"].items()))
    torch.save(res, path)
    return res


def conditions(layers):
    out = [("clean", None, None, None, None)]
    for arm in ARMS:
        for layer in layers:
            for al in ALPHAS:
                out.append((f"{arm}|L{layer}|a{al:g}|real", arm, layer, al, None))
                out += [(f"{arm}|L{layer}|a{al:g}|ctrl{j}", arm, layer, al, j) for j in range(N_CTRL)]
    return out


def _acc_boot(hit_a, hit_b, groups, n_boot, seed=0):
    """Paired scene bootstrap of mean(hit_a) − mean(hit_b): (point, samples)."""
    u, gi = np.unique(groups, return_inverse=True)
    G = len(u)
    W = np.stack([np.bincount(c, minlength=G) for c in np.random.default_rng(seed).integers(0, G, (n_boot, G))])
    n = np.bincount(gi, minlength=G).astype(float)
    sa, sb = np.bincount(gi, weights=hit_a.astype(float), minlength=G), np.bincount(gi, weights=hit_b.astype(float),
                                                                                    minlength=G)
    return float(hit_a.mean() - hit_b.mean()), (W @ sa - W @ sb) / (W @ n)


def sign_of(cr, c):
    """+1 where the answer under condition c is C (the child), −1 where it is R."""
    return np.where(cr[f"pred:{c}"] == cr.child, 1, -1)


def judge(cr: pd.DataFrame, layers, alphas, n_boot: int, arms=ARMS) -> list[dict]:
    """Every (arm, layer, α) condition against clean and the mean of its controls, per question kind."""
    rows = []
    for arm in arms:
        for layer in layers:
            for al in alphas:
                base = f"{arm}|L{layer}|a{al:g}"
                names = ["clean", f"{base}|real"] + [f"{base}|ctrl{j}" for j in range(N_CTRL)]
                st = {}
                for cell in CELLS:
                    k = (cr.cell == cell).to_numpy()
                    b = ba_boot(cr.truth[k], {n: sign_of(cr, n)[k] for n in names}, cr.scene[k], n_boot=n_boot)
                    clean, real = b["clean"], b[f"{base}|real"]
                    ctrl_pt = np.mean([b[f"{base}|ctrl{j}"][0] for j in range(N_CTRL)])
                    ctrl_s = np.mean([b[f"{base}|ctrl{j}"][1] for j in range(N_CTRL)], 0)
                    st[cell] = dict(clean=clean[0], real=real[0], gain=real[0] - clean[0], ctrl_gain=ctrl_pt - clean[0],
                                    contrast=real[0] - ctrl_pt, gain_s=real[1] - clean[1], contrast_s=real[1] - ctrl_s)
                d, f = st[DEC], st[FIRST]
                c = dict(arm=arm, layer=layer, alpha=al, dec_gain=d["gain"], contrast=d["contrast"],
                         contrast_lo99=float(np.percentile(d["contrast_s"], 0.5)),
                         contrast_hi99=float(np.percentile(d["contrast_s"], 99.5)),
                         first_gain=f["gain"], first_lo99=float(np.percentile(f["gain_s"], 0.5)),
                         easy={e: st[e]["gain"] for e in EASY}, st=st)
                c["works"] = condition_works(c)
                rows.append(c)
    return rows


f2 = lambda x: f"{x:+.2f}"


def arm_table(cs: list[dict]) -> list[str]:
    lines = ["| layer | α | decisive | gain | control gain | contrast [99% CI] | first step only gain [99% lower] | "
             "easy gains (own / agree / C larger) | works |", "|---|---|---|---|---|---|---|---|---|"]
    for c in cs:
        d = c["st"][DEC]
        lines.append(f"| {c['layer']} | {c['alpha']:g} | {d['real']:.2f} | {f2(c['dec_gain'])} | "
                     f"{f2(d['ctrl_gain'])} | {f2(c['contrast'])} [{f2(c['contrast_lo99'])}, {f2(c['contrast_hi99'])}] "
                     f"| {f2(c['first_gain'])} [{f2(c['first_lo99'])}] | "
                     + " / ".join(f2(c["easy"][e]) for e in EASY) + f" | {'yes' if c['works'] else 'no'} |")
    return lines


def transfer(T: pd.DataFrame, base: str, n_boot: int) -> tuple[float, dict, list[str]]:
    """Transfer questions under condition `base` against clean and the mean of its controls."""
    trn = T[T.kind != "C-R"]
    hit = lambda c: (trn[f"pred:{c}"] == trn.answer).to_numpy()
    ctrl_hit = np.mean([hit(f"{base}|ctrl{j}") for j in range(N_CTRL)], 0)
    overall = float(hit(f"{base}|real").mean() - hit("clean").mean())
    gains, tl = {}, []
    for kind in TKINDS:
        k = (trn.kind == kind).to_numpy()
        if not k.any():
            continue
        g, _ = _acc_boot(hit(f"{base}|real")[k], hit("clean")[k], trn.scene[k], n_boot)
        cg, cs = _acc_boot(hit(f"{base}|real")[k], ctrl_hit[k], trn.scene[k], n_boot)
        gains[kind] = (g, float(np.percentile(cs, 2.5)))
        tl.append(f"| {kind} | {int(k.sum())} | {hit('clean')[k].mean():.2f} | {f2(g)} | {f2(cg)} "
                  f"[{f2(np.percentile(cs, 2.5))}, {f2(np.percentile(cs, 97.5))}] |")
    lines = ["| question | n | clean accuracy | gain | contrast with control [95% CI] |", "|---|---|---|---|---|", *tl]
    return overall, gains, lines


def follow_shares(cr: pd.DataFrame, c: str) -> tuple[float, float]:
    """On the decisive questions: share of answers following C's sentence, and M's sentence."""
    d1, d2 = cr.d1.to_numpy(float), cr.d2.to_numpy(float)
    dk = (cr.cell == DEC).to_numpy()
    sg = sign_of(cr, c)[dk]
    return float(np.mean(sg == np.where(d2 > 0, 1, -1)[dk])), float(np.mean(sg == np.where(d1 > 0, 1, -1)[dk]))


def analyse(T: pd.DataFrame, res, layers, n_boot: int) -> tuple[str, str, dict]:
    cr = T[T.kind == "C-R"]
    rows = judge(cr, layers, ALPHAS, n_boot)
    by_arm = {arm: [c for c in rows if c["arm"] == arm] for arm in ARMS}
    verdicts = {arm: verdict_boost(cs) for arm, cs in by_arm.items()}

    lines = ["Model **Gemma 3 12B** (chat format). Locate: ridge probes for M's step and C's own step (per axis) at the "
             f"final position of the flow set (seed {SEED_FIT}, 2400 questions); amplify on new scenes (seed "
             f"{SEED_TEST}): h ← h + (α − 1)·P(h − μ) at one layer, final position only. Control: mean of "
             f"{N_CTRL} random subspaces of the same rank and projected variance (±10%) in the top principal "
             "components. Balanced accuracy; gains are boosted − clean; contrast = gain − control gain, paired "
             f"scene-bootstrap 99% CI ({n_boot} resamples).", ""]
    for layer in layers:
        L = res["layers"][layer]
        moves = "; ".join(f"{arm} {v['norm']:.3g} (control {v['ctrl_norm']:.3g}, top {v['k']} components)"
                          for arm, v in L["arms"].items())
        lines.append(f"- Layer {layer}: held-out R² of M's step {L['r2_d1']:.2f}, of C's own step {L['r2_d2']:.2f}; "
                     f"mean residual norm {L['h_norm']:.3g}; mean size of P(h − μ), i.e. the change per unit of "
                     f"α − 1: {moves}.")
    lines += ["", f"Clean balanced accuracy on the decisive questions: {rows[0]['st'][DEC]['clean']:.2f}; first step only "
              f"{rows[0]['st'][FIRST]['clean']:.2f}; " + ", ".join(f"{e} {rows[0]['st'][e]['clean']:.2f}" for e in EASY)
              + ".", ""]
    for arm in ARMS:
        lines += [f"**Arm: {arm}** — preregistered verdict: **{verdicts[arm]}**", "", *arm_table(by_arm[arm]), ""]

    # secondary: transfer, for the working condition with the largest decisive gain
    working = [c for c in rows if c["works"]]
    pick = max(working, key=lambda c: c["dec_gain"]) if working else max(rows, key=lambda c: c["contrast"])
    base = f"{pick['arm']}|L{pick['layer']}|a{pick['alpha']:g}"
    overall, gains, tl = transfer(T, base, n_boot)
    tv = transfer_verdict(overall, gains)
    lines += [("Transfer (secondary, preregistered), for the working condition with the largest decisive gain: "
               if working else "Transfer (exploratory: no condition works), for the condition with the largest "
               "decisive contrast: ") + f"{pick['arm']}, layer {pick['layer']}, α {pick['alpha']:g}.", "", *tl, "",
              f"Overall transfer accuracy change {f2(overall)}. " + (f"Preregistered transfer verdict: **{tv}**."
                                                                     if working else f"(Would read: {tv}.)"), ""]

    # exploratory
    d1, d2 = cr.d1.to_numpy(float), cr.d2.to_numpy(float)
    ratio = lambda c: (lambda w: w[0] / w[1])(np.linalg.lstsq(np.c_[d1, d2, np.ones(len(cr))],
                                                              cr[f"ld:{c}"].to_numpy(float), rcond=None)[0])
    (cc, mc), (cb, mb) = follow_shares(cr, "clean"), follow_shares(cr, base + "|real")
    lines += [f"Exploratory: weight of M's step relative to C's own step in the margin, clean {ratio('clean'):.2f}, "
              f"{base} {ratio(base + '|real'):.2f}. On the decisive questions the answer follows C's sentence in "
              f"{cc:.2f} (clean) and {cb:.2f} (boosted), M's sentence in {mc:.2f} and {mb:.2f}. Logit lens: not "
              "computed.", ""]
    lines.append("Preregistered verdicts: " + "; ".join(f"{arm}: **{v}**" for arm, v in verdicts.items())
                 + ". Rules: docs/preregistration_map_boost.md.")
    return "; ".join(f"{arm}: {v}" for arm, v in verdicts.items()), "\n".join(lines), dict(rows=rows)


def run(a):
    out, fit_sizes, test_sizes, per_kind, n_boot = ROOT / "results" / "map_boost", SIZES_FIT, SIZES_TEST, \
        TRANSFER_PER_KIND, 2000
    if a.smoke:
        import tempfile
        a.model = "tiny"
        fit_sizes, test_sizes, per_kind, n_boot = ({c: 20 for c in CELLS}, {c: 6 for c in CELLS}, 4, 100)
        out = Path(a.out) if a.out else Path(tempfile.mkdtemp())
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    done = out / f"{a.model}.done"
    if done.exists():
        print(f"[map-boost] {a.model} already complete; delete {done.name} to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[map-boost] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    la = layer_grid(m.n_layers)[2]["final"][0]
    layers = [1, 2] if a.smoke else [la, la + 4]
    notify(f"[Map boost {a.model}] started (layers {layers})")
    res = locate(m, a, out, layers, fit_sizes, log)
    T, specs, qs, n_scenes = test_set(m, test_sizes, res["items"], per_kind)
    log(f"test set: {int((T.kind == 'C-R').sum())} C-R questions in {n_scenes} scenes, "
        f"{int((T.kind != 'C-R').sum())} transfer questions")
    ipath = out / f"{a.model}_items.csv"
    if ipath.exists():
        old = pd.read_csv(ipath, keep_default_na=False)
        assert len(old) == len(T) and (old.question == T.question).all() and (old.scene == T.scene).all()
        T = old
    pos = [len(s.input_ids) - 1 for s in specs]
    todo = [c for c in conditions(layers) if f"pred:{c[0]}" not in T.columns]
    for i, (name, arm, layer, al, j) in enumerate(todo):
        edit = None
        if arm is not None:
            L = res["layers"][layer]
            Q = L["arms"][arm]["Q"] if j is None else L["arms"][arm]["ctrl"][j]
            edit = boost_edit(layer, Q, L["mu"], al, pos)
        sc = score(m, list(zip(specs, qs)), edit=edit, batch_size=a.batch_size)
        T[f"pred:{name}"] = [max(x, key=x.get) for x in sc]
        T[f"ld:{name}"] = [x[r.child] - x[r.root] if r.kind == "C-R" else np.nan for x, r in zip(sc, T.itertuples())]
        T.to_csv(ipath, index=False)
        log(f"[{i + 1}/{len(todo)}] {name}")
    v, report, _ = analyse(T, res, layers, n_boot)
    print(report, flush=True)
    title = f"Map boost at the answer position ({a.model}) — preregistered"
    if not a.smoke:
        append_entry(title, report)
    (out / f"{a.model}_report.md").write_text(f"## {title}\n\n{report}\n")
    done.write_text(v + "\n")
    notify(f"[Map boost {a.model}] COMPLETE: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, few questions, layers 1-2; temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/map_boost)")
    ap.add_argument("--acts-dir", default=None, help="where to cache the fitting residuals (default: output folder)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
