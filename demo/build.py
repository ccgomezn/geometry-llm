"""Build the self-contained demo page (demo/move-the-map.html) from results/demo/demo_data.json and the P4 result
files. Usage: uv run python demo/build.py [--data path]"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def r2(v):
    return [round(float(x), 2) for x in v]


def compact(d):
    for sc in d["scenes"]:
        sc["base_decoded"] = {k: r2(v) for k, v in sc["base_decoded"].items()}
        for t in sc["targets"]:
            t["text_decoded"] = {k: r2(v) for k, v in t["text_decoded"].items()}
            for st in t["steps"]:
                st["decoded"] = {k: r2(v) for k, v in st["decoded"].items()}
                st["pos"] = [round(float(x), 2) for x in st["pos"]]
                st["probs"] = [{k: round(v, 2) for k, v in p.items()} for p in st["probs"]]
    return d


def stats(model="gemma3-12b"):
    das = pd.read_csv(ROOT / "results" / "p4" / f"{model}_das_relation_L14.csv")
    allq = das[das.qtype == "ALL"].set_index("method")
    it = pd.read_csv(ROOT / "results" / "p4" / f"{model}_interp_relation_L14.csv").set_index("alpha")
    df = pd.read_parquet(ROOT / "results" / "p4" / f"{model}_interp_relation_L14.parquet")
    nov = df[(df.alpha == 0.5) & df.margin_ok & (df.geo != df.geo_base) & (df.geo != df.geo_src)]
    chance = float(np.mean([1 / 8 if q == "direction" else 1 / 2 for q in nov.qtype]))
    return dict(n_pairs=300, das_acc=float(allq.loc["DAS k=2", "acc"]), random_acc=float(allq.loc["random k=2", "acc"]),
                halfway_novel=float(it.loc[0.5, "novel"]), halfway_chance=chance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "results" / "demo" / "demo_data.json"))
    ap.add_argument("--out", default=str(ROOT / "demo" / "move-the-map.html"))
    ap.add_argument("--first", type=int, default=4, help="index of the scene to show first")
    a = ap.parse_args()
    d = compact(json.loads(Path(a.data).read_text()))
    d["scenes"].insert(0, d["scenes"].pop(a.first))  # the scene shown in the GIF opens first
    d["stats"] = stats()
    page = (ROOT / "demo" / "template.html").read_text().replace("/*DATA*/null", json.dumps(d, separators=(",", ":")))
    Path(a.out).write_text(page)
    print(f"{a.out}: {len(page) / 1024:.0f} KB, {len(d['scenes'])} scenes")
    codepen(page, Path(a.out).parent)


FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"


def codepen(page: str, out_dir: Path):
    """Split the page into CodePen's HTML / CSS / JS panels, and write open-in-codepen.html: a local page whose
    button pre-fills a new pen (CodePen prefill API) for the user to save under their own account."""
    css = page.split("<style>", 1)[1].split("</style>", 1)[0]
    js = page.split("<script>", 1)[1].split("</script>", 1)[0]
    body = page.split("<body>", 1)[1].split("<script>", 1)[0].strip()
    cp = out_dir / "codepen"
    cp.mkdir(exist_ok=True)
    (cp / "index.html").write_text(body)
    (cp / "style.css").write_text(f"@import url('{FONTS}');\n{css}")
    (cp / "script.js").write_text(js)
    data = dict(title="Move the map: editing two numbers inside a language model", html=body,
                css=f"@import url('{FONTS}');\n{css}", js=js, editors="001", layout="top")
    val = json.dumps(data).replace("&", "&amp;").replace('"', "&quot;")
    (out_dir / "open-in-codepen.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Open in CodePen</title>"
        "<body style='font:16px system-ui;margin:40px'><h1 style='font-size:20px'>Move the map → CodePen</h1>"
        "<p>Click the button: CodePen opens a new pen with the demo filled in. Save it under your account, then paste "
        "the pen's URL on its own line in your Medium post.</p>"
        "<form action='https://codepen.io/pen/define' method='POST' target='_blank'>"
        f"<input type='hidden' name='data' value=\"{val}\">"
        "<button style='font:inherit;padding:10px 18px;border-radius:8px;border:0;background:#111;color:#fff;cursor:pointer'>"
        "Open in CodePen</button></form></body>")
    print(f"{cp}/ and {out_dir / 'open-in-codepen.html'} written")


if __name__ == "__main__":
    main()
