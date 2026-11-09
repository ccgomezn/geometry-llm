"""P4b: the slider. Move one object along the decoded x/y directions and watch every answer move."""

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from geomap.cache import load_cache
from geomap.das import SubspaceIntervention
from geomap.models import load
from geomap.patching import prefix_acts
from geomap.probes import centered_xy, probe_directions
from geomap.prompts import NAME_POOLS, build_prompt, single_token_names
from geomap.questions import all_questions
from geomap.report import ROOT, append_entry
from geomap.scenes import generate_scene, leaves
from geomap.steering import natural_range, predicted_answers, slider, slider_agreement

FIG = ROOT / "results" / "figures"
OUT = ROOT / "results" / "p4"


def das_directions(model, m, cache, scenes):
    """If DAS k=2 met the criterion, map coordinates into its subspace; else None."""
    path = OUT / f"{model}_das.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[(d.method == "DAS k=2") & (d.qtype != "ALL")]
    if d.empty or (d.acc < 0.7).any():
        return None
    ckpt = next(OUT.glob(f"{model}_L*_list_k2.pt"))
    layer = int(ckpt.name.split("_L")[1].split("_")[0])
    iv = SubspaceIntervention(m.d_model, 2)
    iv.load_state_dict(torch.load(ckpt, map_location="cpu"))
    Q = iv.Q.detach().numpy()
    X, Y, _ = centered_xy(cache["list"][:, layer].float().numpy(), scenes, np.arange(len(scenes)))
    B, *_ = np.linalg.lstsq(X @ Q, Y, rcond=None)  # subspace coords -> (x, y)
    return layer, Q @ np.linalg.pinv(B), "DAS subspace"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-scenes", type=int, default=50)
    ap.add_argument("--span", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--max-q", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="smoke tests only: ignore the skip branch")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(ROOT / "results" / "p2" / f"{a.model}_real.csv")
    m = load(a.model)
    cache, cscenes, _ = load_cache(ROOT / "cache" / a.model / "main.pt")
    src = das_directions(a.model, m, cache, cscenes)
    if src is None:
        if p2.r2.max() < 0.3 and not a.force:
            msg = ("Skipped (preregistered branch): no usable direction. "
                   f"Best probe R² {p2.r2.max():.2f} < 0.3 and DAS k=2 did not meet its criterion.")
            append_entry(f"P4 slider ({a.model})", msg)
            print(msg)
            return
        with open(ROOT / "results" / "p2" / f"{a.model}_probe.pkl", "rb") as fh:
            pk = pickle.load(fh)
        src = (pk["layer"], probe_directions(pk["probe"]), "probe directions")
    layer, dirs, label = src

    rng = np.random.default_rng(7)
    names = single_token_names(m.tok, NAME_POOLS["people"])
    grid = np.arange(-a.span, a.span + 1e-9, a.step)
    sweeps = {"x": np.stack([grid, np.zeros_like(grid)], 1), "y": np.stack([np.zeros_like(grid), grid], 1)}
    ref = cache["list"][:, layer].float().numpy().reshape(-1, m.d_model)
    ref_norms = np.linalg.norm(ref[~np.isnan(ref).any(1)], axis=1)
    dfs, steered_norms = [], []
    for sc in range(a.n_scenes):
        s = generate_scene(rng, names, 4)
        leaf = leaves(s)[0]
        qs = all_questions(s, min_margin=0.0, min_angle=0.0)
        moving = [q for q in qs if any(len(set(predicted_answers(s, leaf, d, q))) > 1 for d in sweeps.values())]
        still = [q for q in qs if q not in moving]
        pick = [moving[i] for i in rng.permutation(len(moving))[:a.max_q - 2]] + \
               [still[i] for i in rng.permutation(len(still))[:2]]
        h = prefix_acts(m, build_prompt(s, None, m.tok, m.family), [layer])[layer]
        h_leaf = h[build_prompt(s, None, m.tok, m.family).list_pos[leaf]].float().numpy()
        for axis, deltas in sweeps.items():
            d = slider(m, s, leaf, layer, dirs, deltas, pick)
            d["scene"], d["axis"] = sc, axis
            dfs.append(d)
            steered_norms += [np.linalg.norm(h_leaf + dirs @ dl) for dl in deltas]
        print(f"scene {sc}/{a.n_scenes}", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    df.drop(columns=["probs"]).to_parquet(OUT / f"{a.model}_slider.parquet")
    agg = slider_agreement(df)
    nr = natural_range(np.array(steered_norms), ref_norms)

    # demo figure: the scene/axis with the most predicted flips
    flips = df.groupby(["scene", "axis"]).apply(
        lambda g: sum(g.sort_values("step").groupby("qid").predicted_ans.apply(lambda s: (s != s.shift()).sum() - 1)))
    sc, axis = flips.idxmax()
    g = df[(df.scene == sc) & (df.axis == axis)]
    qids = g.qid.unique()[:4]
    fig, axs = plt.subplots(1, len(qids), figsize=(4 * len(qids), 3.2), sharey=True)
    for ax, qi in zip(np.atleast_1d(axs), qids):
        gq = g[g.qid == qi].sort_values("step")
        x = gq.dx if axis == "x" else gq.dy
        cands = list(gq.probs.iloc[0].keys())
        for c in cands:
            ax.plot(x, [p[c] for p in gq.probs], label=c)
        pa = gq.predicted_ans.values
        for i in range(1, len(pa)):
            if pa[i] != pa[i - 1]:
                ax.axvline(x.iloc[i] - a.step / 2, color="k", ls="--", lw=1)
        ax.set_title(gq.qtype.iloc[0], fontsize=9)
        ax.set_xlabel(f"moved object's {axis} offset (units)")
        ax.legend(fontsize=6)
    np.atleast_1d(axs)[0].set_ylabel("model probability")
    fig.suptitle("P4 slider: model answers as one object is moved inside the model (dashed = geometric threshold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "p4_slider_demo.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(agg["flip_offsets"], bins=np.arange(-a.span, a.span + a.step, a.step))
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("observed flip − geometric threshold (units)")
    ax.set_ylabel("sweeps")
    ax.set_title("P4 slider: where answers flip, relative to the geometry")
    fig.tight_layout()
    fig.savefig(FIG / "p4_slider_offsets.png", dpi=150)
    plt.close(fig)

    off = np.array(agg["flip_offsets"])
    lines = [f"Model **{a.model}**, layer {layer}, using {label}; {a.n_scenes} scenes, x and y sweeps of ±{a.span} units "
             f"in {a.step}-unit steps (including positions no sentence states).", "",
             f"- Agreement with geometry along the sweeps: {agg['agreement']:.2f} [{agg['agreement_lo']:.2f}, {agg['agreement_hi']:.2f}]",
             f"- Sweeps where geometry predicts a flip: {agg['n_sweeps_with_flip']}; the model also flips in "
             f"{agg['frac_observed_flip']:.0%} of them",
             f"- Predicted vs observed flip position: r = {agg['flip_r']:.2f}; median offset "
             f"{np.median(off) if len(off) else float('nan'):+.2f} units",
             f"- Steered activations within the natural norm range (1st–99th percentile): {nr:.0%}"]
    append_entry(f"P4 slider ({a.model})", "\n".join(lines), [FIG / "p4_slider_demo.png", FIG / "p4_slider_offsets.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
