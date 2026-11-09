"""P5: collect the four core figures and write the E1 verdict against the preregistration."""

import argparse
import json
import shutil

import pandas as pd

from geomap.report import ROOT, append_entry

R = ROOT / "results"
FIG = R / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    model = a.model or json.loads((R / "p1" / "chosen.json").read_text())["chosen"]

    figs = []
    for src, dst in [("p1_accuracy.png", "fig1_behavior.png"), ("p2_r2_by_layer.png", "fig2_probes.png"),
                     ("p3_sweep.png", "fig3_consistency.png"), ("p4_slider_demo.png", "fig4_slider.png")]:
        if (FIG / src).exists():
            shutil.copy(FIG / src, FIG / dst)
            figs.append(FIG / dst)

    crit, notes = {}, []
    p2 = R / "p2" / f"{model}_real.csv"
    if p2.exists():
        real = pd.read_csv(p2)
        shuf = pd.read_csv(R / "p2" / f"{model}_shuffled.csv")
        stated = pd.read_csv(R / "p2" / f"{model}_stated_relation.csv")
        b = int(real.r2.idxmax())
        r = real.iloc[b]
        ctrl = [shuf.r2.iloc[b], stated.r2.item(), real.r2.iloc[0]]
        randp = R / "p2" / f"{model}_rand.csv"
        if randp.exists():
            ctrl.append(pd.read_csv(randp).r2.iloc[b])
        crit["map_r2"] = r.r2 >= 0.6 and all(r.r2 - c >= 0.3 for c in ctrl)
        crit["map_hops"] = (r.r2_hop1 - r.r2_hop2p) <= 0.1
        crit["rel_r2"] = r.r2 < 0.3 or (r.r2_hop1 - r.r2_hop2p) > 0.2
        notes.append(f"P2: best layer {b}, R² {r.r2:.2f}; 1-step {r.r2_hop1:.2f} vs 2+-step {r.r2_hop2p:.2f}")
    p3 = R / "p3" / f"{model}_confirm_consistency.csv"
    if p3.exists():
        c = pd.read_csv(p3)
        ro = c[c.site.isin(["list", "eol"])].joint_consistency.max()
        rel = c[c.site == "relation"].joint_consistency.max()
        crit["map_jc"] = ro >= 0.6
        crit["rel_jc"] = ro < 0.2 and rel > ro
        notes.append(f"P3: joint consistency at readout positions {ro:.2f}, at relation tokens {rel:.2f}")
    p4 = R / "p4" / f"{model}_das.csv"
    if p4.exists():
        d = pd.read_csv(p4)
        d = d[(d.method == "DAS k=2") & (d.qtype != "ALL")]
        crit["map_das"] = bool((d.acc >= 0.7).all())
        notes.append(f"P4: DAS k=2 lowest per-type accuracy {d.acc.min():.2f}")

    map_keys = ["map_r2", "map_hops", "map_jc", "map_das"]
    rel_keys = ["rel_r2", "rel_jc"]
    if all(crit.get(k, False) for k in map_keys):
        verdict = "**Support for H-map**: the model builds and uses a per-object coordinate map."
    elif all(crit.get(k, False) for k in rel_keys):
        verdict = "**Support for H-rel**: no usable map; answers come from the relation sentences."
    else:
        verdict = "**Mixed**: see the per-criterion breakdown."
    missing = [k for k in map_keys + rel_keys if k not in crit]
    lines = [f"Model **{model}**. Verdict against `docs/preregistration.md`: {verdict}", "",
             "| criterion | met |", "|---|---|"] + [f"| {k} | {'yes' if v else 'no'} |" for k, v in crit.items()]
    if missing:
        lines.append(f"\nNot yet evaluated: {', '.join(missing)}")
    lines += ["", *[f"- {n}" for n in notes], "",
              "Prior work to engage in the write-up: Tehenan et al. 2025 (linear spatial models, single-answer "
              "steering) and An et al. 2026 (weak, fragmented coordinate probes in 7–8B models)."]
    append_entry("E1 summary", "\n".join(lines), figs)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
