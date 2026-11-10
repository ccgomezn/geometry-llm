"""P2d (exploratory): did the model *compose* positions, or does it hold the stated offsets side by side?

Coordinates are sums of stated offsets, so a linear probe can read them from a "bag of offsets". Distance to the
root is not additive in the offsets. We probe it at each object's list mention and compare with the best additive
baseline (one-hot codes of the object's own and its parent's stated offsets). Beating that baseline on two-step
objects means the activations carry an interaction of the two offsets, i.e. a composed position."""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geomap.cache import load_cache
from geomap.probes import GroupRidge, distance_rows, scene_split
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p2"
FIG = ROOT / "results" / "figures"


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_eval(Xtr, ytr, gtr, Xte, yte, hop_te):
    mx, my = Xtr.mean(0), ytr.mean()
    pr = GroupRidge().fit(Xtr - mx, (ytr - my)[:, None], gtr)
    p = pr.predict(Xte - mx)[:, 0] + my
    return {f"hop{h}": r2(yte[hop_te == h], p[hop_te == h]) for h in (1, 2)} | {"all": r2(yte, p)}


def boot_ci(y, p, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    mem = {g: np.flatnonzero(groups == g) for g in u}
    vals = [r2(y[i], p[i]) for i in (np.concatenate([mem[g] for g in rng.choice(u, len(u))]) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    cache, scenes, layers = load_cache(ROOT / "cache" / a.model / "main.pt")
    tr, te = scene_split(len(scenes))
    mtr, atr = distance_rows(scenes, tr)
    mte, ate = distance_rows(scenes, te)
    ytr, yte = mtr.dist.to_numpy(), mte.dist.to_numpy()
    gtr, hte = mtr.scene.to_numpy(), mte.hop.to_numpy()

    base = fit_eval(atr, ytr, gtr, ate, yte, hte)

    def pos_feats(meta):
        out = []
        for sc, name in zip(meta.scene, meta.name):
            s = scenes[sc]
            x, y = np.subtract(s.coords[name], s.coords[s.root])
            out.append([x, y, x * x, y * y, x * y])
        return np.asarray(out, dtype=float)

    ref = fit_eval(pos_feats(mtr), ytr, gtr, pos_feats(mte), yte, hte)

    rows = []
    L = cache["list"]
    for li, layer in enumerate(layers):
        Xtr = L[mtr.scene.to_numpy(), li, mtr.slot.to_numpy()].float().numpy()
        Xte = L[mte.scene.to_numpy(), li, mte.slot.to_numpy()].float().numpy()
        rows.append({"layer": layer} | fit_eval(Xtr, ytr, gtr, Xte, yte, hte))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{a.model}_distance.csv", index=False)
    b = int(df.hop2.idxmax())

    # CI for the best layer, two-step objects
    li = b
    Xtr = L[mtr.scene.to_numpy(), li, mtr.slot.to_numpy()].float().numpy()
    Xte = L[mte.scene.to_numpy(), li, mte.slot.to_numpy()].float().numpy()
    mx, my = Xtr.mean(0), ytr.mean()
    p = GroupRidge().fit(Xtr - mx, (ytr - my)[:, None], gtr).predict(Xte - mx)[:, 0] + my
    h2 = hte == 2
    lo, hi = boot_ci(yte[h2], p[h2], mte.scene.to_numpy()[h2])

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(df.layer, df.hop2, color="#2b57c9", label="activations, two-step objects")
    ax.plot(df.layer, df.hop1, color="#2b57c9", ls="--", lw=1, label="activations, one-step objects")
    ax.axhline(base["hop2"], color="#a8432f", ls="-.", lw=1, label=f"best additive baseline, two-step ({base['hop2']:.2f})")
    ax.axhline(ref["hop2"], color="gray", ls=":", lw=1, label=f"true position (quadratic), two-step ({ref['hop2']:.2f})")
    ax.set_ylim(-0.05, 1)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out R², distance to root")
    ax.set_title(f"Is distance (not additive in the stated offsets) readable? ({a.model})", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "p2_distance.png", dpi=170)
    plt.close(fig)

    lines = [f"Model **{a.model}**, same 3,000 scenes and split as P2, readout at the list mention. Exploratory. Target: "
             "each non-root object's distance to the root. Distance is not a sum of the stated offsets, so the best "
             "*additive* baseline (one-hot codes of the object's own and its parent's offsets) cannot fully fit it "
             "for two-step objects. Activations that beat it carry an interaction of the two offsets.", "",
             f"- Best layer for two-step objects: {layers[b]}, R² {df.hop2[b]:.2f} [{lo:.2f}, {hi:.2f}]",
             f"- Best additive baseline: two-step {base['hop2']:.2f}, one-step {base['hop1']:.2f}",
             f"- Reference from the true position (x, y, x², y², xy): two-step {ref['hop2']:.2f}", "",
             "| layer | two-step | one-step |", "|---|---|---|"]
    lines += [f"| {r.layer} | {r.hop2:.2f} | {r.hop1:.2f} |" for r in df.itertuples() if r.layer in (0, 8, 12, 14, 16, 18, 21, 24, 30, 40, 48)]
    append_entry(f"P2d distance probe ({a.model}) — exploratory", "\n".join(lines), [FIG / "p2_distance.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
