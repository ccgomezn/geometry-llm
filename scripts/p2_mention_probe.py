"""P2m (exploratory): is an object's composed position readable where the *question* names it?

Same 3,000 scenes and split as P2. One prompt per object: "Which object is farthest from X?"; the readout is X's
token in the question (it has read the whole description and list). Probes: centred (x, y) as in P2, and
distance to the root against the best additive baseline as in P2d. The list-mention curves are overlaid."""

import argparse
import time
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from geomap.cache import MAX_OBJS, cache_prefix, load_cache, save_cache
from geomap.models import load
from geomap.probes import GroupRidge, distance_rows, probe_by_layer, scene_split
from geomap.prompts import build_prompt
from geomap.questions import QType, make_question
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p2"
FIG = ROOT / "results" / "figures"


def mention_cache(m, scenes, layers, batch_size):
    """{"list": [N, L, 6, d]}: slot k holds object k's token in "Which object is farthest from <object k>?"."""
    N, L = len(scenes), len(layers)
    out = torch.full((N, L, MAX_OBJS, m.d_model), float("nan"), dtype=torch.bfloat16)
    for k in range(MAX_OBJS):
        idx = [i for i, s in enumerate(scenes) if len(s.names) > k]
        if not idx:
            break
        specs = []
        for i in idx:
            s = scenes[i]
            x = s.names[k]
            sp = build_prompt(s, make_question(QType.FARTHEST, (x,), s.names), m.tok, m.family)
            specs.append(replace(sp, list_pos={x: sp.question_pos[x]}))
        c = cache_prefix(m, specs, layers, batch_size)["list"]
        out[idx, :, k] = c[:, :, 0]
        print(f"[mention] slot {k}: {len(idx)} prompts", flush=True)
    return {"list": out}


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def distance_curve(L, scenes, split):
    tr, te = split
    mtr, atr = distance_rows(scenes, tr)
    mte, ate = distance_rows(scenes, te)
    ytr, yte, gtr, h2 = mtr.dist.to_numpy(), mte.dist.to_numpy(), mtr.scene.to_numpy(), mte.hop.to_numpy() == 2

    def fit(Xtr, Xte):
        mx, my = Xtr.mean(0), ytr.mean()
        p = GroupRidge().fit(Xtr - mx, (ytr - my)[:, None], gtr).predict(Xte - mx)[:, 0] + my
        return r2(yte[h2], p[h2])

    base = fit(atr, ate)
    rows = [dict(layer=li, hop2=fit(L[mtr.scene.to_numpy(), li, mtr.slot.to_numpy()].float().numpy(),
                                     L[mte.scene.to_numpy(), li, mte.slot.to_numpy()].float().numpy()))
            for li in range(L.shape[1])]
    return pd.DataFrame(rows), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    _, scenes, layers = load_cache(ROOT / "cache" / a.model / "main.pt")
    path = ROOT / "cache" / a.model / "mention.pt"
    if path.exists():
        cache, _, _ = load_cache(path)
    else:
        m = load(a.model)
        t0 = time.time()
        cache = mention_cache(m, scenes, layers, a.batch_size)
        save_cache(path, cache, scenes, layers)
        print(f"[mention] cached in {time.time() - t0:.0f}s", flush=True)
        del m
    split = scene_split(len(scenes))
    real = probe_by_layer(cache, scenes, split, n_boot=a.n_boot)
    shuf = probe_by_layer(cache, scenes, split, control="shuffled", n_boot=a.n_boot)
    dist, base = distance_curve(cache["list"], scenes, split)
    real.to_csv(OUT / f"{a.model}_mention_real.csv", index=False)
    shuf.to_csv(OUT / f"{a.model}_mention_shuffled.csv", index=False)
    dist.to_csv(OUT / f"{a.model}_mention_distance.csv", index=False)
    lst = pd.read_csv(OUT / f"{a.model}_real.csv")
    ldist = pd.read_csv(OUT / f"{a.model}_distance.csv")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axs[0]
    ax.plot(lst.layer, lst.r2_hop2p, color="#b4761d", label="list mention")
    ax.plot(real.layer, real.r2_hop2p, color="#2b57c9", label="name in the question")
    ax.plot(shuf.layer, shuf.r2, color="gray", ls=":", lw=0.8, label="shuffled labels")
    ax.set_title("(x, y), two-step objects", fontsize=10)
    ax.set_ylabel("held-out R²")
    ax = axs[1]
    ax.plot(ldist.layer, ldist.hop2, color="#b4761d", label="list mention")
    ax.plot(dist.layer, dist.hop2, color="#2b57c9", label="name in the question")
    ax.axhline(base, color="#a8432f", ls="-.", lw=1, label=f"best additive baseline ({base:.2f})")
    ax.set_title("distance to root, two-step objects", fontsize=10)
    for ax in axs:
        ax.set_ylim(-0.1, 1)
        ax.set_xlabel("layer")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Is the composed position readable where the question names the object? ({a.model})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "p2_mention.png", dpi=170)
    plt.close(fig)

    b, bd = int(real.r2_hop2p.idxmax()), int(dist.hop2.idxmax())
    r = real.iloc[b]
    lines = [f"Model **{a.model}**, same 3,000 scenes and split as P2. Exploratory. Readout: the object's name in "
             "\"Which object is farthest from X?\" (one prompt per object).", "",
             f"- (x, y): best layer for two-step objects {b}, R² {r.r2_hop2p:.2f} (all objects {r.r2:.2f} "
             f"[{r.r2_lo:.2f}, {r.r2_hi:.2f}]); list mention best {lst.r2_hop2p.max():.2f}; shuffled {shuf.r2.iloc[b]:.2f}",
             f"- Distance to root, two-step: best layer {bd}, R² {dist.hop2[bd]:.2f}; list mention best "
             f"{ldist.hop2.max():.2f}; best additive baseline {base:.2f}", "",
             "| layer | (x, y) two-step | (x, y) all | distance two-step |", "|---|---|---|---|"]
    lines += [f"| {l} | {real.r2_hop2p[l]:.2f} | {real.r2[l]:.2f} | {dist.hop2[l]:.2f} |"
              for l in (0, 8, 12, 14, 16, 18, 21, 24, 28, 32, 40, 48) if l < len(real)]
    append_entry(f"P2m question-mention probe ({a.model}) — exploratory", "\n".join(lines), [FIG / "p2_mention.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
