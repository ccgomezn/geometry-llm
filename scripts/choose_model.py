"""Apply the preregistered P1 model-choice rule to the (rescored) P1 results and record the choice."""

import argparse
import json

import pandas as pd

from geomap.analysis import choose_model, gate
from geomap.report import ROOT, append_entry

ORDER = ["qwen3-1.7b", "llama3.2-3b", "qwen3-4b", "gemma3-4b", "qwen3-8b", "gemma3-12b", "qwen3-14b", "gemma3-27b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER, help="ascending parameter count")
    a = ap.parse_args()
    res = {k: pd.read_parquet(ROOT / "results" / "p1" / f"{k}.parquet") for k in a.models
           if (ROOT / "results" / "p1" / f"{k}.parquet").exists()}
    order = [k for k in a.models if k in res]
    c = choose_model(res, order)
    lines = ["| model | 4-object accuracy | 3-object accuracy | 4-object gate |", "|---|---|---|---|"]
    for k in order:
        d = res[k][res[k]["set"] == "main"]
        g = gate(res[k])
        lines.append(f"| {k} | {d[d.n_objects == 4].correct.mean():.2f} | {d[d.n_objects == 3].correct.mean():.2f} | "
                     f"{'pass' if g['pass'] else 'fail'} |")
    verdict = (f"**Chosen: {c['model']}** on {c['n_objects'][0]}-object scenes; excluded question types (at chance): "
               f"{', '.join(c['exclude']) or 'none'}." if c["model"] else "**No model passes even on 3-object scenes.**")
    append_entry("P1 rescored: model choice", verdict + "\n\n" + "\n".join(lines))
    (ROOT / "results" / "p1" / "chosen.json").write_text(json.dumps(c))
    print(json.dumps(c))


if __name__ == "__main__":
    main()
