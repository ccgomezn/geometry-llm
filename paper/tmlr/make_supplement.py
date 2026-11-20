"""Anonymized code-and-data supplement for the TMLR submission.

Packs the committed tree (code, tests, preregistrations, Colab notebooks, result files) into
paper/tmlr/build/supplementary_material.zip, with every identifying string removed: names, accounts, paths, the
repository URL, the earlier preprint's DOI and every commit hash of this repository (a hash can be searched on GitHub).
It then scans every file in the archive, text and binary, and refuses to write the zip if anything identifying is left.

Run: python3 paper/tmlr/make_supplement.py
"""

import io
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "build" / "supplementary_material.zip"

INCLUDE = ("src/", "scripts/", "tests/", "colab/", "results/", "docs/preregistration", "pyproject.toml", "uv.lock",
           ".python-version")
EXCLUDE_SUFFIX = (".pt",)                       # ~200 MB of intermediate tensors; every script regenerates its own
EXCLUDE = {"colab/make_bundle.sh", "results/run_commands.sh", "scripts/medium_page.py", "scripts/blog_figures.py"}
TEXT_SUFFIX = {".py", ".md", ".ipynb", ".sh", ".toml", ".txt", ".json", ".csv", ".cfg", ".yaml", ".yml", ".done", ""}

FORBIDDEN = re.compile(rb"ccgomezn|cristhian|cris_|g[o\xc3\xb3]+\S{0,2}mez\b|neira|anemona|authpay|gmail|zenodo|"
                       rb"/Users/|/home/", re.I)          # the Colab folder MyDrive/geomap is generic and kept

README = """# Supplementary material: code, preregistrations and results

Anonymized for review. The full repository, with the commit history that dates each preregistration before its data,
will be public with the camera-ready version.

## Contents

- `src/geomap/`: the library (scene and question generator, prompts, scoring, probes, activation patching, direct
  attribution through Gemma's normalisations, interventions, decision rules).
- `tests/`: automated tests (`uv run pytest`), including the preregistered decision rules.
- `scripts/`: one script per test; each appends its report to `results/RESULTS.md` and writes its files under
  `results/`.
- `docs/preregistration_*.md`: the preregistrations, each written and committed before its data.
- `colab/`: the notebooks that ran each test on a single Colab GPU.
- `results/`: every result file and the results log `RESULTS.md`, from which the paper's appendix tables are generated.
  Intermediate tensors (`*.pt`, about 200 MB) are omitted; the scripts regenerate them.

## The paper's tests

| test (paper, Appendix A) | preregistration | script |
|---|---|---|
| mistakes vs map (check A) | `preregistration_map_use_checks.md` | `scripts/check_error_overlap.py` |
| one-shot erasure (check B) | `preregistration_map_use_checks.md` | `scripts/check_nonlinear_erasure.py` |
| closed-loop erasure | `preregistration_erase_closed_loop.md` | `scripts/erase_closed_loop.py` |
| confirmatory erasure | `preregistration_erase_axis_confirm.md` | `scripts/erase_axis_confirm.py` |
| fair random control | `preregistration_erase_fair_control.md` | `scripts/erase_fair_control.py` |
| rule and erasure (5,250 questions) | `preregistration_axis_bulk.md` | `scripts/erase_axis_bulk.py` |
| composition | `preregistration_compose.md` | `scripts/compose_test.py` |
| flow | `preregistration_flow.md` | `scripts/flow_test.py` |
| writers, mean ablation | `preregistration_components.md` | `scripts/components_test.py` |
| twin-mean removal | `preregistration_components2.md` | `scripts/components2_test.py` |
| order and voice | `preregistration_order_voice.md` | `scripts/order_voice_test.py` |
| replication (27B) | `preregistration_replication_27b.md` | the flow and component scripts with `--model gemma3-27b` |

## Running

Python 3.11+ and [uv](https://docs.astral.sh/uv/): `uv sync`, then `uv run pytest` for the tests. Each script takes
`--smoke` to run on a tiny random model in a few seconds, and `--model gemma3-12b` or `--model gemma3-27b` for the real
runs, which need the (gated) Gemma 3 weights from Hugging Face and one GPU (A100 40 GB for 12B, 80 GB for 27B).
"""

LICENSE = """MIT License

Copyright (c) 2026 Anonymous authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def scrub(path: str, text: str, commits: set[str]) -> str:
    text = re.sub(r"https?://github\.com/ccgomezn/geometry-llm\S*", "[anonymized repository]", text)
    text = re.sub(r"(doi:?\s*)?(https?://doi\.org/)?10\.5281/zenodo\.\d+", "[earlier preprint]", text)
    text = re.sub(r"/home/[\w.-]+/\.local/bin/uv", "uv", text)
    text = re.sub(r"/(home|Users)/[\w.-]+/", "~/", text)
    if path == "pyproject.toml":
        text = re.sub(r"authors\s*=\s*\[[^\]]*\]", 'authors = [{ name = "Anonymous" }]', text)
    # commit hashes of this repository: any 7-40 hex string that is a prefix of a real commit
    def hide(m: re.Match) -> str:
        h = m.group(0)
        return "[commit]" if any(c.startswith(h) for c in commits) else h
    return re.sub(r"\b[0-9a-f]{7,40}\b", hide, text)


def leaks(data: bytes, commits: set[str]) -> list[str]:
    found = [m.group(0).decode(errors="replace") for m in FORBIDDEN.finditer(data)]
    for m in re.finditer(rb"\b[0-9a-f]{7,40}\b", data):
        h = m.group(0).decode()
        if re.search(r"[a-f]", h) and any(c.startswith(h) for c in commits):
            found.append(h)
    return found


def main():
    commits = set(git("rev-list", "--all").decode().split())
    paths = [p for p in git("ls-files").decode().splitlines()
             if p.startswith(INCLUDE) and not p.endswith(EXCLUDE_SUFFIX) and p not in EXCLUDE]
    files: dict[str, bytes] = {}
    for p in paths:
        data = git("show", f"HEAD:{p}")
        if Path(p).suffix in TEXT_SUFFIX:
            try:
                data = scrub(p, data.decode("utf-8"), commits).encode("utf-8")
            except UnicodeDecodeError:
                pass
        files[p] = data
    files["README.md"], files["LICENSE"] = README.encode(), LICENSE.encode()

    bad = {p: sorted(set(v))[:5] for p, d in files.items() if (v := leaks(d, commits))}
    if bad:
        for p, v in bad.items():
            print(f"LEAK {p}: {v}")
        raise SystemExit(f"{len(bad)} files still identify the author; nothing written")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(files):
            info = zipfile.ZipInfo(f"supplementary_material/{p}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(info, files[p])
    OUT.write_bytes(buf.getvalue())
    print(f"{OUT}: {len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB, no identifying strings found")


if __name__ == "__main__":
    main()
