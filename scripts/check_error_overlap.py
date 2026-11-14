"""Check A (docs/preregistration_map_use_checks.md): where the probe-decoded map answers wrongly, does the model give the
same wrong answer more often than with a blurred true map that errs as often on the same kinds of question?

Uses P9's stored answers (results/p9/gemma3-12b_p9.parquet); regenerates P9's scenes from the seed (tokenizer only, no
model) and checks them against the stored truth, question type and class. CPU only."""

import argparse
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from geomap.checks import blurred_rates, calibrate_sd, error_overlap, verdict_overlap
from geomap.followups import chain_items, p7_class
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry

LAYERS, PRIMARY = [18, 21, 24, 28], 21


def phi(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-log", action="store_true", help="do not append to RESULTS.md")
    a = ap.parse_args()
    out = ROOT / "results" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
    items = chain_items(SimpleNamespace(tok=tok), 300, seed=602, exclude_qtypes=("closer_than",))
    flat = {(si, qi): (q, s, p7_class(q, s.root, mid, child))
            for si, (s, mid, child, qs) in enumerate(items) for qi, (q, _) in enumerate(qs)}
    p9 = pd.read_parquet(ROOT / "results" / "p9" / "gemma3-12b_p9.parquet")
    for r in p9[p9.layer == PRIMARY].itertuples():
        q, s, cls = flat[(r.scene, r.qi)]
        if (answer_for(q, s.coords), q.qtype.value, cls) != (r.truth, r.qtype, r.cls):
            raise ValueError(f"regenerated question ({r.scene}, {r.qi}) does not match P9")
    print(f"[check A] {len(flat)} questions regenerated and matched ({time.time() - t0:.0f}s)", flush=True)

    recs, rows, cal = [], [], []
    for layer in LAYERS:
        d = p9[p9.layer == layer].sort_values(["scene", "qi"]).reset_index(drop=True)
        d["map_wrong"] = (d.map_ans != d.truth).astype(float)
        d["map_same"] = ((d.map_ans != d.truth) & (d.model_ans == d.map_ans)).astype(float)
        d["blur_wrong"], d["blur_same"] = np.nan, np.nan
        for (qt, cls), g in d.groupby(["qtype", "cls"]):
            its = [(r.scene, flat[(r.scene, r.qi)][0], flat[(r.scene, r.qi)][1].coords, r.truth, r.model_ans)
                   for r in g.itertuples()]
            target = 1 - g.map_wrong.mean()
            sd = calibrate_sd(its, target, a.n_draws, seed=layer)
            w, sm = blurred_rates(its, sd, a.n_draws, seed=layer)
            d.loc[g.index, "blur_wrong"], d.loc[g.index, "blur_same"] = w, sm
            cal.append(dict(layer=layer, qtype=qt, cls=cls, n=len(g), map_acc=target, sd=sd, blur_acc=1 - w.mean()))
        print(f"[check A] layer {layer} calibrated ({time.time() - t0:.0f}s)", flush=True)
        for subset, g in [("all", d)] + [(c, d[d.cls == c]) for c in ("CM", "CR", "MR")]:
            r = error_overlap(g.map_wrong, g.map_same, g.blur_wrong, g.blur_same, g.scene, n_boot=a.n_boot)
            r.update(layer=layer, subset=subset, n=len(g),
                     phi_map=phi(g.model_ans == g.truth, g.map_ans == g.truth),
                     phi_blur=float(np.corrcoef(g.model_ans == g.truth, 1 - g.blur_wrong)[0, 1]))
            rows.append(r)
        recs.append(d.assign(layer=layer))

    s = pd.DataFrame(rows)
    s.to_csv(out / "gemma3-12b_error_overlap_summary.csv", index=False)
    pd.DataFrame(cal).to_csv(out / "gemma3-12b_error_overlap_calibration.csv", index=False)
    pd.concat(recs).to_parquet(out / "gemma3-12b_error_overlap.parquet")
    p = s[(s.layer == PRIMARY) & (s.subset == "all")].iloc[0]
    v = verdict_overlap(p.effect, p.lo)

    lines = ["Model **gemma3-12b**, P9's 300 test chains (5,350 questions; scenes regenerated from the seed and matched). "
             "*Follows the map's mistakes* = among questions the probe-decoded map gets wrong, the share where the model "
             "gives that same answer. *Blurred map* = the true positions plus Gaussian noise, with the noise calibrated "
             f"per question type × class so it is as accurate as the decoded map ({a.n_draws} draws). 95% scene-bootstrap "
             "CIs.", "",
             "| layer | questions | n | map wrong | follows decoded map | follows blurred map | difference [95% CI] | "
             "phi (decoded / blurred) |", "|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.layer} | {r.subset} | {r.n} | {int(r.n_map_wrong)} | {r.S:.2f} | {r.S_blur:.2f} | "
                     f"{r.effect:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.phi_map:.2f} / {r.phi_blur:.2f} |")
    lines += ["", f"Preregistered verdict (layer {PRIMARY}, all questions): **{v}**. "
              "Rules: docs/preregistration_map_use_checks.md (check A)."]
    if not a.no_log:
        append_entry("Check A: do the model's mistakes follow its decoded map? (gemma3-12b) — preregistered",
                     "\n".join(lines), [])
    (out / "gemma3-12b_error_overlap.done").write_text(v)
    print("\n".join(lines))
    print(f"verdict: {v} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
