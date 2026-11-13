"""Study 2, exploratory (not preregistered): in the step-by-step replies, are the (x, y) coordinates the model writes
for each object correct, and how does accuracy on "inside" depend on that? Separates building the map (composing the
stated relations into coordinates) from using it (the geometric inside test). Positions are compared up to a shift,
since the model may put a different object at the origin. No GPU needed: rebuilds the scenes from the tokenizer."""

import argparse
import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from geomap.prompts import NAME_POOLS, single_token_names
from geomap.report import ROOT
from geomap.triangles import generate_tri_scene, in_triangle, ladder

NUM = r"(-?\d+(?:\.\d+)?)"


def written_coords(reply: str, name: str):
    """Where the model first puts the object: the first line that starts with the object's name (after list markers and
    bold), or that says 'Let/with/place NAME (be) at', and that contains a numeric (x, y); its last such pair, so
    'Rock is at (-2 - 2, -6) = (-4, -6)' reads (-4, -6). None if no line qualifies."""
    n = re.escape(name)
    pair = re.compile(rf"\(\s*{NUM}\s*,\s*{NUM}\s*\)")
    for line in reply.splitlines():
        head = re.sub(r"^[\s*\-\d.)]*", "", line).lstrip("*").lstrip()
        own = re.match(rf"{n}\b", head) or re.search(rf"\b(?:[Ll]et|with|[Pp]lace|[Pp]ut)\s+{n}\s+(?:be\s+)?at\b", line)
        hits = pair.findall(line)
        if own and hits:
            return float(hits[-1][0]), float(hits[-1][1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-27b")
    ap.add_argument("--tokenizer", default="google/gemma-3-12b-it", help="Gemma 3 sizes share one tokenizer")
    ap.add_argument("--run", default="gemma3-27b_s2_reasoning_n400_seed702_tok1536")
    a = ap.parse_args()
    df = pd.read_parquet(ROOT / "results" / "s2" / "reasoning" / f"{a.run}.parquet")
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pools = [single_token_names(tok, NAME_POOLS[k]) for k in ("people", "objects")]
    rng = np.random.default_rng(702)
    tss = []
    for i in range(len(df)):
        ts = generate_tri_scene(rng, pools[i % 2])
        q = next(it for it in ladder(ts, rng) if it.rung == 4).question
        assert q.text == df.text.iloc[i], "scenes do not match the run"
        tss.append(ts)
    recs = []
    for ts, r in zip(tss, df.itertuples()):
        names = [*ts.corners, ts.point]
        w = [written_coords(r.reply, n) for n in names]
        true = np.array([ts.scene.coords[n] for n in names], float)
        if any(x is None for x in w):
            status, model_geom = "missing", None
        else:
            w = np.array(w)
            ok = np.allclose(w - w[0], true - true[0])
            status = "all correct" if ok else "some wrong"
            model_geom = "yes" if in_triangle(tuple(w[3]), tuple(w[0]), tuple(w[1]), tuple(w[2])) else "no"
        recs.append(dict(stratum=r.stratum, truth=r.truth, pred=r.pred, status=status, implied=model_geom))
    d = pd.DataFrame(recs)
    d = d[d.stratum.isin(("inside", "box_hard"))]
    print(f"hard subset, n={len(d)}")
    for st, g in d.groupby("status"):
        line = f"  coordinates {st}: {len(g)} ({len(g) / len(d):.0%}); accuracy {np.mean(g.pred == g.truth):.2f}"
        if st != "missing":
            line += (f"; the final answer matches what its own coordinates imply in {np.mean(g.pred == g.implied):.0%}")
        print(line)
    ok = d[d.status == "all correct"]
    print(f"  with correct coordinates: inside answered yes {np.mean(ok[ok.truth == 'yes'].pred == 'yes'):.2f}, "
          f"box_hard answered no {np.mean(ok[ok.truth == 'no'].pred == 'no'):.2f}")


if __name__ == "__main__":
    main()
