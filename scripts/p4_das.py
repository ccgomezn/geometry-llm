"""P4a: is there a small (k = 2) subspace whose swap moves an object and every answer with it?"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from geomap.das import SubspaceIntervention, eval_das, train_das
from geomap.models import load
from geomap.patching import make_pairs
from geomap.report import ROOT, append_entry

OUT = ROOT / "results" / "p4"
FIG = ROOT / "results" / "figures"


def best_layer(model):
    c = pd.read_csv(ROOT / "results" / "p3" / f"{model}_sweep_consistency.csv")
    c = c[(c.site == "list") & (c.layer > 0)]
    col = "joint_iia" if c.joint_iia.notna().any() else "joint_consistency"
    return int(c.loc[c[col].idxmax(), "layer"]) if c[col].notna().any() else int(c.layer.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--site", default="list")
    ap.add_argument("--ks", type=int, nargs="+", default=[2])
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-objects", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    layer = a.layer if a.layer is not None else best_layer(a.model)
    # the preregistered run (list site, layer from the P3 sweep) keeps the default names; anything else is exploratory
    expl = a.site != "list" or a.layer is not None
    tag = f"_{a.site}_L{layer}" if expl else ""
    m = load(a.model)
    kw = dict(n_objects=tuple(a.n_objects), exclude_qtypes=tuple(a.exclude))
    train, test = make_pairs(m, a.n_train, seed=5, **kw), make_pairs(m, a.n_test, seed=6, **kw)

    results = {}
    torch.manual_seed(123)
    rnd = SubspaceIntervention(m.d_model, 2)  # untrained random 2-D subspace: control
    results["random k=2"] = eval_das(m, rnd, test, layer, a.site)
    for k in a.ks:
        iv = train_das(m, train, layer, a.site, k=k, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size)
        torch.save(iv.state_dict(), OUT / f"{a.model}_L{layer}_{a.site}_k{k}.pt")
        results[f"DAS k={k}"] = eval_das(m, iv, test, layer, a.site)
        print(f"k={k} loss history {iv.history}", flush=True)
    full = pd.concat([d.assign(method=k) for k, d in results.items()])
    full.to_csv(OUT / f"{a.model}_das{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    meths = list(results)
    qts = [q for q in results[meths[0]].qtype if q != "ALL"] + ["ALL"]
    w = 0.8 / len(meths)
    for i, meth in enumerate(meths):
        d = results[meth].set_index("qtype").reindex(qts)
        ax.bar([j + i * w for j in range(len(qts))], d.acc, w, label=meth,
               yerr=[d.acc - d.acc_lo, d.acc_hi - d.acc], capsize=2)
    ax.axhline(0.7, color="g", ls="-.", lw=0.8, label="preregistered 0.7")
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(qts))], qts, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("interchange accuracy")
    ax.legend(fontsize=8)
    ax.set_title(f"P4 DAS at layer {layer}, {a.site} position ({a.model})")
    fig.tight_layout()
    fig.savefig(FIG / f"p4_das{tag}.png", dpi=150)
    plt.close(fig)

    lines = [f"Model **{a.model}**, layer {layer}, site {a.site}; {a.n_train} training pairs, {a.n_test} test pairs. "
             "Interchange accuracy = on questions where the model answers differently when the object is really "
             "elsewhere, the share whose answer after swapping only the learned subspace equals that answer "
             "(ceiling 1). Geometric accuracy (answers equal the moved layout) in the last column.", "",
             "| method | " + " | ".join(qts) + " | |", "|---|" + "---|" * (len(qts) + 1)]
    for meth, d in results.items():
        d = d.set_index("qtype").reindex(qts)
        lines.append(f"| {meth} | " + " | ".join(f"{r.acc:.2f} [{r.acc_lo:.2f}, {r.acc_hi:.2f}]" for r in d.itertuples())
                     + f" | geometric {d.loc['ALL', 'acc_geom']:.2f} |")
    d2 = results.get("DAS k=2")
    if d2 is not None:
        per = d2[d2.qtype != "ALL"]
        ok = bool(per.acc.notna().all() and (per.acc >= 0.7).all())
        low = (f"lowest: {per.acc.min():.2f} on {per.loc[per.acc.idxmin(), 'qtype']}" if per.acc.notna().any()
               else "no question where the model's answer changes")
        lines += ["", f"Preregistered criterion (k = 2, ≥ 0.7 on every question type): **{'met' if ok else 'not met'}** ({low})."]
    append_entry(f"P4 DAS ({a.model})" + (f" — exploratory: {a.site} site, layer {layer}" if expl else ""), "\n".join(lines), [FIG / f"p4_das{tag}.png"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
