"""Exploratory follow-up to P8 (added after seeing its data; CPU only, no model forward passes).

In each P8 pair, C ends up at the same new absolute position in two scenes: the source (only M's sentence changed, so
M and C moved) and the target (only C's own sentence changed, M unmoved). On questions that involve C and R but not
M, the geometric answer is therefore the same in source and target. Does the model update its answer equally often
in both? If it fully composes C's position from the chain, yes; if it leans on C's own stated offset and under-uses
M's, it updates far less when only M's sentence changed.

The pairs are regenerated with the Gemma 3 tokenizer (shared across sizes; the cached 4B copy is used) and checked
against the saved P8 answers."""

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from geomap.analysis import bootstrap_ci
from geomap.questions import answer_for
from geomap.report import ROOT, append_entry
from geomap.transfer import make_transfer_pairs


def cls(q, root, mid, child):
    inv = (set(q.args) | set(q.candidates)) & {root, mid, child}  # objects only, never compass words
    if child in inv and root in inv and mid not in inv:
        return "C&R only"
    if inv == {child, mid}:
        return "C&M only"
    if child not in inv:
        return "no C"
    return "all three"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-12b")
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-it")
    a = ap.parse_args()
    out = ROOT / "results" / "p8"
    df = pd.read_parquet(out / f"{a.model}_p8.parquet")
    n = int(df.pair.max()) + 1
    m = SimpleNamespace(tok=AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True), family="gemma3")
    pairs = make_transfer_pairs(m, n, seed=501, exclude_qtypes=("closer_than",))
    clean = df[df.layer == -1].pivot_table(index=["pair", "qid"], columns="site", values="model_ans", aggfunc="first")
    # consistency check: same question counts and types as the saved run
    qt = df[df.site == "none"].set_index(["pair", "qid"]).qtype
    regen = {(p, i): q.qtype.value for p, tp in enumerate(pairs) for i, q in enumerate(tp.qs)}
    assert set(regen) == set(qt.index) and all(regen[k] == qt[k] for k in regen), "regenerated pairs do not match"

    recs = []
    for p, tp in enumerate(pairs):
        root = tp.base.root
        for i, q in enumerate(tp.qs):
            gb, gs, gt = answer_for(q, tp.base.coords), answer_for(q, tp.src.coords), answer_for(q, tp.tgt.coords)
            r = clean.loc[(p, i)]
            recs.append(dict(pair=p, qid=i, qtype=q.qtype.value, cls=cls(q, root, tp.mid, tp.child),
                             geo_base=gb, geo_src=gs, geo_tgt=gt, base=r["none"], src=r["source"], tgt=r["target"]))
    x = pd.DataFrame(recs)
    x.to_csv(out / f"{a.model}_p8_behavior_records.csv", index=False)

    rows = []
    for c, g in x.groupby("cls"):
        # questions whose geometric answer changes when C moves (same change in source and target for C&R-only)
        ch = g[(g.geo_base != g.geo_tgt)]
        if not len(ch):
            continue
        row = dict(cls=c, n=len(ch))
        for name, col, geo in (("updates_when_C_sentence_changes", "tgt", "geo_tgt"),
                               ("updates_when_M_sentence_changes", "src", "geo_src")):
            v = ((ch[col] != ch.base) & (ch[col] == ch[geo])).astype(float)  # model moves to the correct new answer
            est, lo, hi = bootstrap_ci(v, groups=ch.pair, n_boot=1000)
            row[name], row[name + "_lo"], row[name + "_hi"] = est, lo, hi
        row["src_equals_base"] = float((ch.src == ch.base).mean())
        row["tgt_equals_base"] = float((ch.tgt == ch.base).mean())
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(out / f"{a.model}_p8_behavior.csv", index=False)

    lines = [f"Exploratory (added after P8's data). Model **{a.model}**, the {n} P8 chain pairs (R → M → C), clean "
             "answers only. In the *source* only M's sentence changed (M and C moved); in the *target* only C's own "
             "sentence changed (M unmoved); C sits at the same new position in both. On questions whose geometric "
             "answer changes when C moves, the share where the model moves to the correct new answer:", "",
             "| questions | n | when C's own sentence changes | when only M's sentence changes | answer unchanged (M changed) |",
             "|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.cls} | {r.n} | {r.updates_when_C_sentence_changes:.2f} [{r.updates_when_C_sentence_changes_lo:.2f}, "
                     f"{r.updates_when_C_sentence_changes_hi:.2f}] | {r.updates_when_M_sentence_changes:.2f} "
                     f"[{r.updates_when_M_sentence_changes_lo:.2f}, {r.updates_when_M_sentence_changes_hi:.2f}] | "
                     f"{r.src_equals_base:.2f} |")
    lines += ["", "For C&R-only questions the geometric change is identical in both scenes, so a model that composes C's "
              "position from the chain should update equally often in both columns."]
    append_entry(f"P8b does the model use M's offset for C at all? ({a.model}) — exploratory", "\n".join(lines), [])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
