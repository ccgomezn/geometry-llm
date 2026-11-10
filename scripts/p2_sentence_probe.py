"""P2b (exploratory, not preregistered): are the composed coordinates readable at the end of each object's own
sentence, the site whose patching does move the answers (P3)?

Same 3,000 scenes, split and probe as P2; only the readout position changes (list mention -> last token of the
object's sentence; the root, which has no sentence, uses the end of the first sentence)."""

import argparse
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from geomap.cache import cache_prefix, load_cache, sentence_end_positions
from geomap.models import load
from geomap.probes import probe_by_layer, scene_split
from geomap.prompts import build_prompt
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p2"
FIG = ROOT / "results" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    _, scenes, layers = load_cache(ROOT / "cache" / a.model / "main.pt")
    m = load(a.model)
    specs = []
    for s in scenes:
        sp = build_prompt(s, None, m.tok, m.family)
        specs.append(replace(sp, list_pos=sentence_end_positions(sp, s)))
    cache = cache_prefix(m, specs, layers, batch_size=a.batch_size)
    split = scene_split(len(scenes))
    real = probe_by_layer(cache, scenes, split, n_boot=a.n_boot)
    shuf = probe_by_layer(cache, scenes, split, control="shuffled", n_boot=a.n_boot)
    real.to_csv(OUT / f"{a.model}_sentence_real.csv", index=False)
    shuf.to_csv(OUT / f"{a.model}_sentence_shuffled.csv", index=False)
    lst = pd.read_csv(OUT / f"{a.model}_real.csv")
    stated = pd.read_csv(OUT / f"{a.model}_stated_relation.csv").r2_hop2p.item()

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(lst.layer, lst.r2, color="#b4761d", label="list mention (P2)")
    ax.plot(lst.layer, lst.r2_hop2p, color="#b4761d", ls="--", lw=1, label="list mention, two-step objects")
    ax.plot(real.layer, real.r2, color="#2b57c9", label="end of the object's own sentence")
    ax.plot(real.layer, real.r2_hop2p, color="#2b57c9", ls="--", lw=1, label="sentence end, two-step objects")
    ax.plot(shuf.layer, shuf.r2, color="gray", ls=":", lw=0.8, label="shuffled labels")
    ax.axhline(stated, color="#a8432f", ls="-.", lw=0.8, label=f"stated-sentence baseline, two-step ({stated:.2f})")
    ax.set_ylim(-0.05, 1)
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out R² of (x, y)")
    ax.set_title(f"Where are the composed coordinates readable? ({a.model})", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "p2_sentence_vs_list.png", dpi=170)
    plt.close(fig)

    b = int(real.r2.idxmax())
    r = real.iloc[b]
    lb = int(lst.r2.idxmax())
    lines = [f"Model **{a.model}**, same 3,000 scenes and split as P2. Exploratory: added after P3 showed that patching "
             "the object's sentence moves the answers while patching its list mention does not.", "",
             f"- Sentence end: best layer {b}, R² {r.r2:.2f} [{r.r2_lo:.2f}, {r.r2_hi:.2f}]; one-step objects "
             f"{r.r2_hop1:.2f}, two-step objects {r.r2_hop2p:.2f} (stated-sentence baseline {stated:.2f}); shuffled "
             f"{shuf.r2.iloc[b]:.2f}",
             f"- List mention (P2): best layer {lb}, R² {lst.r2.iloc[lb]:.2f}; two-step {lst.r2_hop2p.iloc[lb]:.2f}",
             "", "| layer | sentence end | sentence end, two-step | list mention | list, two-step |", "|---|---|---|---|---|"]
    for l in [4, 8, 10, 12, 14, 16, 18, 21, 24, 30, 40, 48]:
        if l < len(real):
            lines.append(f"| {l} | {real.r2.iloc[l]:.2f} | {real.r2_hop2p.iloc[l]:.2f} | {lst.r2.iloc[l]:.2f} | "
                         f"{lst.r2_hop2p.iloc[l]:.2f} |")
    append_entry(f"P2b sentence-end probe ({a.model}) — exploratory", "\n".join(lines), [FIG / "p2_sentence_vs_list.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
