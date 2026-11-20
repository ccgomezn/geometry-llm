"""Cross-family check (docs/preregistration_cross_family.md): applies the preregistered criteria to the flow runs of
Qwen3-8B and Llama 3.1 8B (scripts/flow_test.py, unchanged) and appends one entry to results/RESULTS.md.

Run after the flow runs: python scripts/cross_family_summary.py  (models without a finished flow run are skipped)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geomap.axis import CELLS
from geomap.crossfamily import overall_cross_family, verdict_cross_family, wilson
from geomap.flow import layer_grid
from geomap.probes import scene_split
from geomap.report import ROOT, append_entry

MODELS = ("qwen3-8b", "llama3.1-8b")
RUN = "{}_flow_seed1601"
DEC = "conflict, M larger"


def summarise(flow_dir: Path, model: str) -> dict | None:
    run = RUN.format(model)
    if not (flow_dir / f"{run}.done").exists():
        return None
    meta = pd.read_csv(flow_dir / f"{run}_items.csv")
    P = pd.read_csv(flow_dir / f"{run}_probes.csv")
    n_layers = int(P.layer.max())
    _, _, win = layer_grid(n_layers)
    tr_s, _ = scene_split(int(meta.scene.max()) + 1)
    held = ~np.isin(meta.scene.to_numpy(), tr_s) & (meta.cell == DEC).to_numpy()
    n = int(held.sum())
    dec_acc = float(meta.correct[meta.cell == DEC].mean())
    model_ci = wilson(int(meta.correct[held].sum()), n)
    fq = P[(P.site == "final") & P.layer.between(*win["final"])].sort_values("layer")
    best = fq.loc[fq.truth_conflict_m.idxmax()]
    probe = wilson(int(round(best.truth_conflict_m * n)), n)
    flow_verdict = (flow_dir / f"{run}.done").read_text().strip()
    return dict(model=model_name(model), key=model, n_layers=n_layers, win=win["final"], n=n, dec_acc=dec_acc,
                acc={c: float(meta.correct[meta.cell == c].mean()) for c in CELLS},
                r2_d1=float(fq.r2_d1.max()), r2_d2=float(fq.r2_d2.max()), flow=flow_verdict,
                best_layer=int(best.layer), probe=probe, model_ci=model_ci,
                probe_first=float(fq.truth_conflict_m.iloc[0]), probe_mean=float(fq.truth_conflict_m.mean()),
                verdict=verdict_cross_family(dec_acc, flow_verdict, probe, model_ci))


def model_name(key: str) -> str:
    return {"qwen3-8b": "Qwen3-8B", "llama3.1-8b": "Llama 3.1 8B"}.get(key, key)


def report(rows: list[dict]) -> tuple[str, str]:
    ci = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    lines = ["Models from two other families, run through `scripts/flow_test.py` unchanged (seed 1601, same question "
             "counts, layers scaled to depth). Criteria: decisive accuracy < 0.5; flow verdict *reaches the answer "
             "position*; truth probe's Wilson 95% lower bound above the model's upper bound on the held-out decisive "
             "questions. Gemma 3 for comparison: decisive accuracy 12B 0.40, 27B 0.23; probe vs model 12B "
             "0.86 [0.78, 0.91] vs 0.32 [0.24, 0.41], 27B 0.98 [0.93, 0.99] vs 0.25 [0.18, 0.34].", "",
             "| model | layers | decisive accuracy | R² d1 / d2 at the final position | probe (best layer) | model, "
             "same questions | n | verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['n_layers']} | {r['dec_acc']:.2f} | {r['r2_d1']:.2f} / {r['r2_d2']:.2f} | "
                     f"{ci(r['probe'])} (layer {r['best_layer']}) | {ci(r['model_ci'])} | {r['n']} | "
                     f"**{r['verdict']}** |")
    lines += ["", "**Clean accuracy by kind:**", "", "| model | " + " | ".join(CELLS) + " |",
              "|---" * (len(CELLS) + 1) + "|"]
    lines += [f"| {r['model']} | " + " | ".join(f"{r['acc'][c]:.2f}" for c in CELLS) + " |" for r in rows]
    lines += [""] + [f"{r['model']}: flow verdict *{r['flow']}*; probe at the first layer of the final window "
                     f"({r['win'][0]}–{r['win'][1]}) {r['probe_first']:.2f}, mean over the window {r['probe_mean']:.2f}."
                     for r in rows]
    overall = overall_cross_family({r["key"]: r["verdict"] for r in rows})
    lines += ["", f"Preregistered verdict: **{overall}**. Rules: docs/preregistration_cross_family.md; full flow reports "
              "in the flow entries of each model."]
    return "\n".join(lines), overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-dir", default=str(ROOT / "results" / "flow"))
    ap.add_argument("--no-append", action="store_true", help="print only")
    a = ap.parse_args()
    rows = [r for m in MODELS if (r := summarise(Path(a.flow_dir), m))]
    if not rows:
        raise SystemExit("no finished flow run of " + ", ".join(MODELS))
    text, overall = report(rows)
    print(text)
    if not a.no_append:
        append_entry("Cross-family check: Qwen3-8B and Llama 3.1 8B — preregistered", text)
    (Path(a.flow_dir) / "cross_family.done").write_text(overall + "\n")


if __name__ == "__main__":
    main()
