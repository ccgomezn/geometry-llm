"""P2: can each object's (x, y) be read from the model's activations? Real probe vs controls."""

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geomap.cache import load_cache
from geomap.probes import joint_eol_probe, probe_by_layer, rsa_by_layer, scene_split
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p2"
FIG = ROOT / "results" / "figures"


def fmt(r, k="r2"):
    return f"{r[k]:.2f} [{r[k + '_lo']:.2f}, {r[k + '_hi']:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--eol", default="eol4", help="end-of-list cache name, e.g. eol3")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache, scenes, layers = load_cache(ROOT / "cache" / a.model / "main.pt")
    split = scene_split(len(scenes))

    real, probes = probe_by_layer(cache, scenes, split, n_boot=a.n_boot, return_probes=True)
    shuf = probe_by_layer(cache, scenes, split, control="shuffled", n_boot=a.n_boot)
    stated = probe_by_layer(cache, scenes, split, control="stated_relation", n_boot=a.n_boot)
    rand_path = ROOT / "cache" / f"{a.model}-rand" / "main.pt"
    rand = None
    if rand_path.exists():
        rc, rs, _ = load_cache(rand_path)
        rand = probe_by_layer(rc, rs, scene_split(len(rs)), n_boot=a.n_boot)
    best = int(real.r2.idxmax())
    best_row = probe_by_layer(cache, scenes, split, n_boot=1000, layers=[best]).iloc[0]
    rsa = rsa_by_layer(cache, scenes, n_boot=a.n_boot)
    eol = None
    eol_path = ROOT / "cache" / a.model / f"{a.eol}.pt"
    if eol_path.exists():
        ec, es, _ = load_cache(eol_path)
        eol = joint_eol_probe(ec["eol"], es, scene_split(len(es)), n_boot=a.n_boot)
    nonlin = None
    if best_row.r2 < 0.3:  # preregistered branch: try a small nonlinear probe before concluding
        cand = sorted(set(real.r2.nlargest(3).index.tolist() + list(range(0, len(real), 4))))
        nonlin = probe_by_layer(cache, scenes, split, nonlinear=True, n_boot=a.n_boot, layers=cand)

    for name, df in [("real", real), ("shuffled", shuf), ("stated_relation", stated), ("rand", rand),
                     ("rsa", rsa), ("eol", eol), ("nonlinear", nonlin)]:
        if df is not None:
            df.to_csv(OUT / f"{a.model}_{name}.csv", index=False)
    with open(OUT / f"{a.model}_probe.pkl", "wb") as fh:
        pickle.dump({"layer": best, "probe": probes[best]}, fh)

    # figures
    fig, ax = plt.subplots(figsize=(8, 4))
    for df, lab in [(real, "real probe"), (shuf, "shuffled control"), (rand, "random-init model")]:
        if df is not None:
            ax.plot(df.layer, df.r2, label=lab)
            ax.fill_between(df.layer, df.r2_lo, df.r2_hi, alpha=0.2)
    ax.axhline(stated.r2.item(), color="k", ls="--", lw=1, label="stated-relation baseline")
    if eol is not None:
        ax.plot(eol.layer, eol.r2, ls=":", label="joint end-of-list probe")
    ax.axhline(0.6, color="g", lw=0.8, ls="-.", label="preregistered 0.6")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out R² (centred x, y)")
    ax.set_ylim(-0.1, 1)
    ax.legend(fontsize=8)
    ax.set_title(f"P2: decoding object coordinates ({a.model})")
    fig.tight_layout()
    fig.savefig(FIG / "p2_r2_by_layer.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k, lab in [("r2_hop1", "1 step from root"), ("r2_hop2", "2 steps"), ("r2_hop3p", "3+ steps")]:
        ax.plot(real.layer, real[k], label=lab)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out R²")
    ax.set_ylim(-0.1, 1)
    ax.legend(fontsize=8)
    ax.set_title("P2: coordinates by chain length (stated directly vs implied)")
    fig.tight_layout()
    fig.savefig(FIG / "p2_r2_by_hop.png", dpi=150)
    plt.close(fig)

    controls = {"shuffled": shuf.r2.iloc[best], "stated relation": stated.r2.item(), "layer 0": real.r2.iloc[0]}
    if rand is not None:
        controls["random-init (same layer)"] = rand.r2.iloc[best]
    margin_ok = all(best_row.r2 - v >= 0.3 for v in controls.values())
    hop_gap = best_row.r2_hop1 - best_row.r2_hop2p
    lines = [
        f"Model **{a.model}**, {len(scenes)} scenes (80/20 split by scene), per-object readout in the object list.",
        "",
        f"- **Best layer {best}: R² = {fmt(best_row)}** (1000 bootstrap resamples over test scenes)",
        "- Controls at that layer: " + ", ".join(f"{k} {v:.2f}" for k, v in controls.items()),
        f"- By chain length at layer {best}: 1 step {best_row.r2_hop1:.2f}, 2+ steps {best_row.r2_hop2p:.2f} "
        f"(gap {hop_gap:+.2f})",
        f"- RSA (activation vs true distances, Spearman) at layer {best}: {fmt(rsa.iloc[best], 'rho')}; "
        f"peak {rsa.rho.max():.2f} at layer {int(rsa.rho.idxmax())}",
    ]
    if eol is not None:
        lines.append(f"- Joint end-of-list probe (all objects from one position): peak R² {eol.r2.max():.2f} "
                     f"at layer {int(eol.r2.idxmax())}")
    if nonlin is not None:
        lines.append(f"- Nonlinear (MLP) probe, run because linear R² < 0.3: peak {nonlin.r2.max():.2f}")
    lines += ["", "Against the preregistered P2 criteria:",
              f"- R² ≥ 0.6: {'yes' if best_row.r2 >= 0.6 else 'no'}",
              f"- beats every control by ≥ 0.3: {'yes' if margin_ok else 'no'}",
              f"- 2+ step objects within 0.1 of 1-step: {'yes' if hop_gap <= 0.1 else 'no'}",
              f"- H-rel signature (R² < 0.3 or a clear drop with hops): "
              f"{'yes' if best_row.r2 < 0.3 or hop_gap > 0.2 else 'no'}"]
    append_entry(f"P2 probes ({a.model})", "\n".join(lines), [FIG / "p2_r2_by_layer.png", FIG / "p2_r2_by_hop.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
