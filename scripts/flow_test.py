"""Flow map (docs/preregistration_flow.md): on new "which is farther east / north, C or R?" questions about chains
R → M → C, where is the first step (R → M) present on the way to the answer, where does each step move the answer,
and which answer does each layer lean to?

Resumable: the activations are cached under results/flow and reused on a rerun."""

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.analysis import bootstrap_ci
from geomap.axis import CELLS, axis_bulk_items
from geomap.cache import cache_prefix
from geomap.checkpoint import notify
from geomap.flow import (PATCH_SITES, SITES, axis_of, axis_predict, axis_r2, flip_step, layer_grid, patch_positions,
                         site_positions, steps, verdict_flow, where_lost)
from geomap.models import load
from geomap.patching import full_acts, patch_edit
from geomap.probes import scene_split
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry
from geomap.scoring import candidate_ids, score

SEED = 1601
SIZES = {"own step only": 400, "agree": 400, "first step only": 600, "conflict, C larger": 400,
         "conflict, M larger": 600}
FLOW_N = 100


def lens_ld(m, H: torch.Tensor, c_ids, r_ids) -> np.ndarray:
    """Logit lens: final norm + unembedding applied to residuals H [N, d]; returns logit(C) − logit(R)."""
    base = m.layers_path.rsplit(".", 1)[0]
    norm = functools.reduce(getattr, (base + ".norm").split("."), m.hf)
    W = m.hf.get_output_embeddings().weight
    cap = getattr(getattr(m.hf.config, "text_config", m.hf.config), "final_logit_softcapping", None)
    out = []
    with torch.no_grad():
        for b in range(0, len(H), 256):
            z = norm(H[b:b + 256].to(W.device, W.dtype))
            lc = (z * W[c_ids[b:b + 256]]).sum(-1).float()
            lr = (z * W[r_ids[b:b + 256]]).sum(-1).float()
            if cap:
                lc, lr = cap * torch.tanh(lc / cap), cap * torch.tanh(lr / cap)
            out.append((lc - lr).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, few questions, layers 0-2; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/flow)")
    ap.add_argument("--acts-dir", default=None, help="where to cache the ~3 GB of residuals (default: the output folder)")
    a = ap.parse_args()
    out, sizes, flow_n, n_boot = ROOT / "results" / "flow", dict(SIZES), FLOW_N, 1000
    if a.smoke:
        import tempfile
        a.model, out, sizes = "tiny", Path(tempfile.mkdtemp()), {c: 10 for c in CELLS}
        flow_n, n_boot = 2, 50
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    run = f"{a.model}_flow_seed{SEED}"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[flow] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[flow] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    # layers and rule windows scaled to the depth (docs/preregistration_replication_27b.md; 12B: as preregistered)
    layers, players, win = layer_grid(m.n_layers)
    if a.smoke:
        layers, players, win = [0, 1, 2], [0, 1], {k: (0, 2) for k in win}

    # ---- questions ----
    items = axis_bulk_items(m, sizes, seed=SEED)
    rows, specs, qs = [], [], []
    for si, (s, mid, child, qq) in enumerate(items):
        for q, cell in qq:
            d1, d2, truth = steps(s, mid, child, q)
            rows.append(dict(scene=si, cell=cell, axis=axis_of(q), c_first=q.args[0] == child, d1=d1, d2=d2,
                             truth=truth, child=child, root=s.root, mid=mid))
            specs.append(build_prompt(s, q, m.tok, m.family))
            qs.append(q)
    meta = pd.DataFrame(rows)
    N = len(meta)
    tr_s, _ = scene_split(len(items))
    tr = np.isin(meta.scene.to_numpy(), tr_s)
    g, ax = meta.scene.to_numpy(), meta.axis.to_numpy()
    log(f"{N} questions in {len(items)} scenes ({tr.sum()} train / {(~tr).sum()} test rows)")
    notify(f"[Flow {a.model}] started: {N} questions")

    # ---- clean answers ----
    sc = score(m, list(zip(specs, qs)), batch_size=a.batch_size)
    meta["ld"] = [x[r.child] - x[r.root] for x, r in zip(sc, meta.itertuples())]
    meta["correct"] = np.sign(meta.ld) == meta.truth
    log("clean answers scored")

    # ---- residuals at the six sites ----
    path = Path(a.acts_dir or out) / f"{run}_acts.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        acts = torch.load(path)
    else:
        site_specs = []
        for sp, r, q in zip(specs, meta.itertuples(), qs):
            pos = site_positions(sp, r.mid, r.child, q, r.root)
            assert tuple(pos) == SITES
            site_specs.append(type(sp)(**{**sp.__dict__, "list_pos": pos}))
        acts = cache_prefix(m, site_specs, layers, a.batch_size)["list"]          # [N, L, 6, d]
        torch.save(acts, path)
    log(f"residuals cached {tuple(acts.shape)}")

    # ---- presence probes ----
    d1, d2, truth = (meta[k].to_numpy(float) for k in ("d1", "d2", "truth"))
    te_cm = ~tr & (meta.cell == "conflict, M larger").to_numpy()
    te_fo = ~tr & (meta.cell == "first step only").to_numpy()
    pr = []
    for k, site in enumerate(SITES):
        for li, layer in enumerate(layers):
            X = acts[:, li, k].float().numpy()
            p = axis_predict(X, truth, ax, g, tr)
            ok = np.sign(p) == truth
            pr.append(dict(site=site, layer=layer, r2_d1=axis_r2(X, d1, ax, g, tr), r2_d2=axis_r2(X, d2, ax, g, tr),
                           truth_all=ok[~tr].mean(), truth_conflict_m=ok[te_cm].mean(), truth_first=ok[te_fo].mean()))
        log(f"probes done: {site}")
    P = pd.DataFrame(pr)
    P.to_csv(out / f"{run}_probes.csv", index=False)

    # ---- logit lens at the final position ----
    c_ids = torch.tensor([candidate_ids(m.tok, c)[0] for c in meta.child])
    r_ids = torch.tensor([candidate_ids(m.tok, c)[0] for c in meta.root])
    lens = []
    for li, layer in enumerate(layers):
        ld = lens_ld(m, acts[:, li, SITES.index("final")], c_ids, r_ids)
        meta[f"lens{layer}"] = ld
        for cell in CELLS:
            c = (meta.cell == cell).to_numpy()
            lens.append(dict(layer=layer, cell=cell, toward_truth=float((ld[c] * truth[c]).mean()),
                             acc=float((np.sign(ld[c]) == truth[c]).mean())))
    L = pd.DataFrame(lens)
    L.to_csv(out / f"{run}_lens.csv", index=False)
    lens_check = float(np.corrcoef(meta[f"lens{layers[-1]}"], meta.ld)[0, 1])
    meta.to_csv(out / f"{run}_items.csv", index=False)
    log(f"logit lens done (last layer vs model: r = {lens_check:.3f})")

    # ---- flow: copy one site at one layer from the run where one step points the other way ----
    fpath = out / f"{run}_flow.csv"
    if fpath.exists():
        F = pd.read_csv(fpath)
    else:
        jobs = []
        for cell in CELLS:
            for i in np.flatnonzero((meta.cell == cell).to_numpy())[:flow_n]:
                r, (s, mid, child, _) = meta.iloc[i], items[meta.scene.iloc[i]]
                for kind, who, step in (("own", child, r.d2), ("parent", mid, r.d1)):
                    try:
                        cf = flip_step(s, who, int(r.axis))
                    except ValueError:
                        continue
                    cspec = build_prompt(cf, qs[i], m.tok, m.family)
                    if (len(cspec.input_ids) != len(specs[i].input_ids) or cspec.relation_spans != specs[i].relation_spans
                            or cspec.question_pos != specs[i].question_pos):
                        continue
                    jobs.append((i, kind, -np.sign(step), cspec))
        log(f"flow: {len(jobs)} counterfactual jobs")
        recs = []
        for c0 in range(0, len(jobs), 40):
            chunk, its, mta, info = jobs[c0:c0 + 40], [], [], []
            for j, (i, kind, orient, cspec) in enumerate(chunk):
                A = full_acts(m, cspec, players)
                pos = patch_positions(specs[i], meta.mid.iloc[i], meta.child.iloc[i])
                its += [(specs[i], qs[i]), (cspec, qs[i])]
                mta += [None, None]
                info += [(j, "clean", -1), (j, "counterfactual", -1)]
                for site in PATCH_SITES:
                    for l in players:
                        its.append((specs[i], qs[i]))
                        mta.append((l, pos[site], A[l][pos[site]]))
                        info.append((j, site, l))
            res = score(m, its, edit=patch_edit(mta), batch_size=a.batch_size)
            for (j, site, l), x in zip(info, res):
                i, kind, orient, _ = chunk[j]
                r = meta.iloc[i]
                recs.append(dict(item=i, scene=r.scene, cell=r.cell, kind=kind, orient=orient, site=site, layer=l,
                                 ld=x[r.child] - x[r.root]))
            log(f"flow: {min(c0 + 40, len(jobs))}/{len(jobs)} jobs")
        F = pd.DataFrame(recs)
        F.to_csv(fpath, index=False)

    # oriented effects: + means the answer moved the way the flipped step points
    base = F[F.site == "clean"].set_index(["item", "kind"]).ld
    F["effect"] = (F.ld - base.loc[list(zip(F["item"], F.kind))].to_numpy()) * F.orient
    full = F[F.site == "counterfactual"]
    eff = []
    for kind in ("own", "parent"):
        for cell in CELLS:
            q = full[(full.kind == kind) & (full.cell == cell)]
            if len(q):
                e, lo, hi = bootstrap_ci(q.effect.to_numpy(), n_boot=n_boot, groups=q.scene.to_numpy())
                eff.append(dict(kind=kind, cell=cell, n=len(q), full_effect=e, lo=lo, hi=hi))
    E = pd.DataFrame(eff)
    E.to_csv(out / f"{run}_flow_full.csv", index=False)
    patched = F[F.site.isin(PATCH_SITES)].groupby(["kind", "site", "layer"]).effect.mean().rename("effect").reset_index()
    fulls = full.groupby("kind").effect.mean()
    patched["share_of_full"] = patched.effect / patched.kind.map(fulls)
    patched.to_csv(out / f"{run}_flow_summary.csv", index=False)
    log("flow summarised")

    # ---- verdict ----
    def best(site_list, target, lo_hi):
        q = P[P.site.isin(site_list) & P.layer.between(*lo_hi)]
        return float(q.groupby("layer")[target].max().max()) if len(q) else float("nan")

    p1, p2 = best(["final"], "r2_d1", win["final"]), best(["final"], "r2_d2", win["final"])
    v = verdict_flow(p1, p2)
    hops = {"C_end": best(["C_end"], "r2_d1", win["C_end"]),
            "question": best(["Q_C", "Q_R", "Q_after"], "r2_d1", win["question"]), "final": p1}
    lost = where_lost(hops)
    fq = P[(P.site == "final") & P.layer.between(*win["final"])]
    tb = fq.loc[fq.truth_conflict_m.idxmax()]
    model_cm = float(meta.correct[te_cm].mean())

    # ---- report ----
    acc = meta.groupby("cell").correct.mean()
    lines = [f"Model **{a.model}**. {N} new C–R axis questions (seed {SEED}) on {len(items)} chains R → M → C; d1 = "
             "M's step from R on the asked axis, d2 = C's own step, truth = sign(d1 + d2). Probes: ridge, fitted "
             "separately for east and north questions, held-out scenes (80/20).", "",
             "**Clean accuracy:** " + ", ".join(f"{c} {acc[c]:.2f}" for c in CELLS), "",
             "**Where each step is present** (held-out R², first step d1 / own step d2):", "",
             "| layer | " + " | ".join(SITES) + " |", "|---" * (len(SITES) + 1) + "|"]
    for layer in layers[::2]:
        q = P[P.layer == layer].set_index("site")
        lines.append(f"| {layer} | " + " | ".join(f"{q.r2_d1[s]:.2f} / {q.r2_d2[s]:.2f}" for s in SITES) + " |")
    lines += ["", f"Best d1 R² per hop: C's sentence end (layers {win['C_end'][0]}–{win['C_end'][1]}) {hops['C_end']:.2f}, "
              f"question tokens ({win['question'][0]}–{win['question'][1]}) {hops['question']:.2f}, final position "
              f"({win['final'][0]}–{win['final'][1]}) {p1:.2f}; d2 at the final position {p2:.2f}.", "",
              f"**Right answer at the final position** (truth probe, held-out *conflict, M larger*): best layer "
              f"{int(tb.layer)}, {tb.truth_conflict_m:.2f}; the model on the same questions {model_cm:.2f}. Truth "
              f"probe on held-out *first step only*: {fq.truth_first.max():.2f}.", "",
              f"**Logit lens** (share of questions where the layer's C-vs-R logit points to the right answer; check: "
              f"last layer vs the model's answer r = {lens_check:.3f}):", "",
              "| layer | " + " | ".join(CELLS) + " |", "|---" * (len(CELLS) + 1) + "|"]
    for layer in [x for x in layers if x >= layers[-1] // 2][::2]:          # 24, 28, ..., 48
        q = L[L.layer == layer].set_index("cell")
        lines.append(f"| {layer} | " + " | ".join(f"{q.acc[c]:.2f}" for c in CELLS) + " |")
    lines += ["", "**How much the answer moves when one step points the other way** (C-vs-R logit difference, "
              "oriented so + = the way the flipped step points; scene-bootstrap 95% CI):", "",
              "| flipped step | cell | n | full effect |", "|---|---|---|---|"]
    lines += [f"| {r.kind} | {r.cell} | {r.n} | {r.full_effect:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] |" for r in E.itertuples()]
    for kind in ("own", "parent"):
        lines += ["", f"Share of the full effect carried by copying one site at one layer from the run with the "
                  f"{'own' if kind == 'own' else 'first'} step flipped (all cells pooled; full effect "
                  f"{fulls.get(kind, float('nan')):+.2f}):", "",
                  "| layer | " + " | ".join(PATCH_SITES) + " |", "|---" * (len(PATCH_SITES) + 1) + "|"]
        q = patched[patched.kind == kind]
        for l in players:
            z = q[q.layer == l].set_index("site")
            lines.append(f"| {l} | " + " | ".join(f"{z.share_of_full[s]:.2f}" if s in z.index else "–"
                                                    for s in PATCH_SITES) + " |")
    lines += ["", f"Preregistered verdict: **{v}**" + (f"; first hop below {0.25}: **{lost}**" if lost else "")
              + ". Rules: docs/preregistration_flow.md."]
    if not a.smoke:
        append_entry(f"Flow map: where is the first step lost on the way to the answer? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(v + (f"; lost at {lost}" if lost else ""))
    notify(f"[Flow {a.model}] COMPLETE. verdict: {v}" + (f"; lost at {lost}" if lost else ""))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
