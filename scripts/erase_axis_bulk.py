"""Bulk C–R axis test (docs/preregistration_axis_bulk.md): 5,250 new "which is farther east / north, C or R?" questions,
split by how the two steps combine on the asked axis, scored clean, with the map erased (the confirmatory test's
bases) and with the fair random control (the amendment's bases). Nothing is refitted.

Resumable: each scoring run is saved under results/erase and skipped on a rerun."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geomap.analysis import bootstrap_ci
from geomap.axis import CELLS, axis_bulk_items, verdict_behaviour, verdict_erasure
from geomap.cache import cache_prefix
from geomap.checkpoint import df_step, notify
from geomap.followups import erase_edit, erase_summary
from geomap.models import load
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene
from geomap.scoring import predict, score

SEED, START, SIZES = 1301, 1, {c: (1500 if c.startswith("conflict") else 750) for c in CELLS}
DECISIVE = "conflict, M larger"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", action="store_true", help="tiny model, layers 1-2, random bases; writes to a temp dir")
    ap.add_argument("--out", default=None, help="output folder (default results/erase)")
    a = ap.parse_args()
    out, sizes = ROOT / "results" / "erase", dict(SIZES)
    if a.smoke:
        import tempfile
        a.model, out, sizes = "tiny", Path(tempfile.mkdtemp()), {c: 6 for c in CELLS}
    out = Path(a.out) if a.out else out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"Axis bulk {a.model}"
    run = f"{a.model}_axis_bulk"
    done = out / f"{run}.done"
    if done.exists():
        print(f"[bulk] already complete ({done}); delete it to rerun", flush=True)
        return
    t0 = time.time()
    log = lambda msg: print(f"[bulk] {msg} ({time.time() - t0:.0f}s)", flush=True)
    m = load(a.model)
    window = [1, 2] if a.smoke else list(range(12, m.n_layers + 1))
    d = m.d_model

    def as_t(bases):
        return {l: (torch.as_tensor(B, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32))
                for l, (B, c) in bases.items()}

    # ---- the two erasures, from earlier runs ----
    spath, fpath = out / f"{a.model}_erase_axis_state.pt", out / f"{a.model}_erase_fair_control.pt"
    if a.smoke and not spath.exists():
        pools = [single_token_names(m.tok, NAME_POOLS[k]) for k in ("people", "objects")]
        rng = np.random.default_rng(0)
        sc = [generate_scene(rng, pools[i % 2], 3) for i in range(30)]
        acts = cache_prefix(m, [build_prompt(s, None, m.tok, m.family) for s in sc], window)["list"]
        mu = {l: np.nanmean(acts[:, li].float().numpy().reshape(-1, d), 0) for li, l in enumerate(window)}
        rb = lambda l, s_: np.linalg.qr(np.random.default_rng(s_ + l).standard_normal((d, 3)))[0]
        torch.save(dict(bases={l: (rb(l, 0), mu[l] @ rb(l, 0)) for l in window}), spath)
        torch.save(dict(control={l: (rb(l, 99), mu[l] @ rb(l, 99)) for l in window}), fpath)
    probe = as_t({l: torch.load(spath, weights_only=False)["bases"][l] for l in window})
    fair = as_t({l: torch.load(fpath, weights_only=False)["control"][l] for l in window})

    # ---- the questions ----
    items = axis_bulk_items(m, sizes, seed=SEED)
    flat = [(si, qi, s, q, cell) for si, (s, mid, child, qs) in enumerate(items) for qi, (q, cell) in enumerate(qs)]
    specs = [(build_prompt(s, q, m.tok, m.family), q) for _, _, s, q, _ in flat]
    truth = [answer_for(q, s.coords) for _, _, s, q, _ in flat]
    first = ["C first" if q.args[0] == items[si][2] else "R first" for si, _, _, q, _ in flat]
    log(f"{len(flat)} questions in {len(items)} scenes")

    def scored(cond, edit):
        preds = [predict(x) for x in score(m, specs, edit=edit, batch_size=a.batch_size)]
        return pd.DataFrame([dict(scene=si, qi=qi, qtype=q.qtype.value, cell=cell, first=f, cond=cond, pred=p,
                                  truth=t, correct=float(p == t))
                             for (si, qi, _, q, cell), f, p, t in zip(flat, first, preds, truth)])

    dfs = []
    for cond, edit in [("clean", None), ("probe", erase_edit(probe, start=START)), ("fair", erase_edit(fair, start=START))]:
        path = out / f"{run}_test_{cond}.parquet"
        fresh = not path.exists()
        dd = df_step(path, lambda c=cond, e=edit: scored(c, e))
        dfs.append(dd)
        msg = f"{cond}: accuracy {dd.correct.mean():.3f}; {DECISIVE}: {dd[dd.cell == DECISIVE].correct.mean():.3f}"
        log(msg)
        if fresh:
            notify(f"[{tag}] {msg}")
    df = pd.concat(dfs, ignore_index=True)
    df.to_parquet(out / f"{run}_test.parquet")

    # ---- rules ----
    d2 = df.replace({"cond": {"fair": "random"}}).assign(config="fair random", pool="all")
    by_cell = erase_summary(d2, group="cell")
    pooled = erase_summary(d2, group="pool").iloc[0]
    extra = pd.concat([erase_summary(d2, group=g) .rename(columns={g: "split"}).assign(by=g) for g in ("qtype", "first")])
    by_cell.to_csv(out / f"{run}_summary.csv", index=False)
    extra.to_csv(out / f"{run}_summary_splits.csv", index=False)
    cl = df[(df.cond == "clean") & (df.cell == DECISIVE)]
    b_est, b_lo, b_hi = bootstrap_ci(cl.correct, groups=cl.scene, n_boot=1000, seed=0)
    dec = by_cell[by_cell.cell == DECISIVE].iloc[0]
    v1 = verdict_behaviour(b_lo, b_hi)
    v2 = verdict_erasure(dec.excess, dec.excess_lo, dec.excess_hi)
    v3 = "the C–R axis effect replicates" if pooled.excess_lo > 0 else "not replicated"

    # ---- report ----
    lines = [f"Model **{a.model}**. {len(flat)} new C–R axis questions (*which is farther east / north, C or R?*) in "
             f"{len(items)} chains R → M → C (seed {SEED}), split by the steps on the asked axis (d1 = R→M, d2 = M→C). "
             "Erasures from earlier runs, not refitted: the map (confirmatory test's bases, layers 12–48, every token "
             "except the first) and the fair random control (variance-matched). Excess drop = fair-control accuracy − "
             "map-erased accuracy; 95% scene-bootstrap CIs. Chance 0.50.", "",
             "| cell | n | clean | map erased | fair random | excess drop [95% CI] |", "|---|---|---|---|---|---|"]
    for c in CELLS:
        x = by_cell[by_cell.cell == c].iloc[0]
        lines.append(f"| {c}{' **(decisive)**' if c == DECISIVE else ''} | {x.n} | {x.clean:.2f} | "
                     f"{x.clean - x.drop_probe:.2f} | {x.clean - x.drop_random:.2f} | {x.excess:+.3f} "
                     f"[{x.excess_lo:+.3f}, {x.excess_hi:+.3f}] |")
    lines.append(f"| all (balanced pool) | {pooled.n} | {pooled.clean:.2f} | {pooled.clean - pooled.drop_probe:.2f} | "
                 f"{pooled.clean - pooled.drop_random:.2f} | {pooled.excess:+.3f} [{pooled.excess_lo:+.3f}, "
                 f"{pooled.excess_hi:+.3f}] |")
    lines += ["", "By question type and by which object is named first:", "",
              "| split | n | clean | map erased | fair random | excess drop [95% CI] |", "|---|---|---|---|---|---|"]
    for x in extra.itertuples():
        lines.append(f"| {x.split} | {x.n} | {x.clean:.2f} | {x.clean - x.drop_probe:.2f} | "
                     f"{x.clean - x.drop_random:.2f} | {x.excess:+.3f} [{x.excess_lo:+.3f}, {x.excess_hi:+.3f}] |")
    lines += ["", f"Clean accuracy in *{DECISIVE}*: {b_est:.3f} [{b_lo:.3f}, {b_hi:.3f}].", "",
              f"Preregistered verdicts: behaviour **{v1}**; erasure **{v2}**; replication **{v3}**. "
              "Rules: docs/preregistration_axis_bulk.md."]
    if not a.smoke:
        append_entry(f"Bulk C–R axis test: composed map or own-step shortcut? ({a.model}) — preregistered",
                     "\n".join(lines), [])
    done.write_text(f"behaviour: {v1}; erasure: {v2}; replication: {v3}")
    notify(f"[{tag}] COMPLETE. behaviour: {v1}; erasure: {v2} ({dec.excess:+.3f} [{dec.excess_lo:+.3f}, "
           f"{dec.excess_hi:+.3f}]); replication: {v3}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
