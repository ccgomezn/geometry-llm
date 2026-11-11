"""CPU-only follow-ups on saved results (no model needed).

1. Stability: the 12B relation-site subspace was trained twice on disjoint scenes (exploratory: seeds 5/6;
   preregistered confirm: seeds 105/106). Principal angles between the two learned 2-D subspaces, against the
   null distribution for random 2-D subspaces of the same space.
2. Failures of the full edit (alpha = 1, held-out pairs): when the answer does not follow the edit, where does it go?
   By question type, and by whether the model itself answers the moved scene correctly.
3. Snap direction: at each alpha, on questions whose two stated layouts give different answers, the share that keep
   the original-layout answer vs take the moved-layout answer."""

import numpy as np
import pandas as pd
import torch

from geomap.das import SubspaceIntervention
from geomap.report import ROOT

P4 = ROOT / "results" / "p4"
OUT = ROOT / "results" / "free"
RUNS = {"gemma3-12b": "gemma3-12b_interp_relation_L14_confirm.parquet",
        "gemma3-27b": "gemma3-27b_interp_relation_L18_confirm.parquet"}


def md(df, index=True):
    """Markdown table without the optional `tabulate` dependency."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                      ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)])


def load_Q(path, d):
    iv = SubspaceIntervention(d, 2)
    iv.load_state_dict(torch.load(path, map_location="cpu"))
    with torch.no_grad():
        return iv.Q.double().numpy()


def principal_cosines(A, B):
    """Cosines of the principal angles between span(A) and span(B) (columns orthonormal)."""
    return np.linalg.svd(A.T @ B, compute_uv=False)


def stability(n_null=2000, seed=0):
    d = 3840
    a = load_Q(P4 / "gemma3-12b_L14_relation_k2.pt", d)
    b = load_Q(P4 / "gemma3-12b_L14_relation_confirm_k2.pt", d)
    cos = principal_cosines(a, b)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_null):
        r, _ = np.linalg.qr(rng.standard_normal((d, 2)))
        null.append(principal_cosines(a, r)[0])
    null = np.array(null)
    return dict(cos_1=float(cos[0]), cos_2=float(cos[1]), angle_1=float(np.degrees(np.arccos(min(cos[0], 1)))),
                angle_2=float(np.degrees(np.arccos(min(cos[1], 1)))), null_cos1_p99=float(np.quantile(null, 0.99)),
                null_cos1_max=float(null.max()), n_null=n_null)


def failures(df):
    x = df[(df.alpha == 1.0) & (df.model_src != df.model_base)].copy()
    x["outcome"] = np.where(x.pred == x.model_src, "follows edit",
                            np.where(x.pred == x.model_base, "stays at original", "something else"))
    x["model_right_on_moved"] = x.model_src == x.geo_src
    by_q = pd.crosstab(x.qtype, x.outcome, normalize="index").round(3)
    by_q["n"] = x.groupby("qtype").size()
    by_right = pd.crosstab(x.model_right_on_moved, x.outcome, normalize="index").round(3)
    by_right["n"] = x.groupby("model_right_on_moved").size()
    overall = x.outcome.value_counts(normalize=True).round(3)
    return overall, by_q, by_right


def snap(df):
    x = df[df.model_src != df.model_base]
    rows = []
    for al, g in x.groupby("alpha"):
        rows.append(dict(alpha=al, n=len(g), keeps_original=(g.pred == g.model_base).mean(),
                         takes_moved=(g.pred == g.model_src).mean(),
                         other=((g.pred != g.model_base) & (g.pred != g.model_src)).mean()))
    return pd.DataFrame(rows).round(3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Free follow-up analyses (CPU only, saved results)", ""]
    s = stability()
    pd.DataFrame([s]).to_csv(OUT / "stability_12b.csv", index=False)
    lines += ["## 1. Do two independent trainings find the same 2-D code? (gemma3-12b, layer 14)", "",
              f"- Principal angles between the exploratory (seeds 5/6) and confirmation (seeds 105/106) subspaces: "
              f"**{s['angle_1']:.1f}° and {s['angle_2']:.1f}°** (cosines {s['cos_1']:.3f}, {s['cos_2']:.3f}).",
              f"- Random 2-D subspaces of the same 3,840-d space: largest cosine {s['null_cos1_p99']:.3f} at the 99th "
              f"percentile, {s['null_cos1_max']:.3f} at most ({s['n_null']} draws).", ""]
    for model, f in RUNS.items():
        df = pd.read_parquet(P4 / f)
        overall, by_q, by_right = failures(df)
        sn = snap(df)
        by_q.to_csv(OUT / f"{model}_failures_by_qtype.csv")
        by_right.to_csv(OUT / f"{model}_failures_by_model_correct.csv")
        sn.to_csv(OUT / f"{model}_snap.csv", index=False)
        lines += [f"## {model}", "", "### 2. Full edit (α = 1): where do the answers go?",
                  "Questions whose answer differs between the two stated layouts (model-relative).", "",
                  "| | share |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in overall.items()] + [
                  "", "By question type:", "", md(by_q), "",
                  "By whether the model answers the moved scene correctly when it reads it as text:", "",
                  md(by_right), "", "### 3. Snap direction by α", "", md(sn, index=False), ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
