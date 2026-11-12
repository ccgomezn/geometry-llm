"""Build paper/arxiv/main.tex (pdflatex-compatible, for arXiv) from paper/draft.md, copy figures, and compile.

Usage: python paper/arxiv/build.py   (needs pandoc; pdflatex optional)"""

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
PANDOC = shutil.which("pandoc") or "/opt/homebrew/bin/pandoc"

FIGS = {  # draft path -> arXiv file name
    "../results/figures/paper_fig1_behaviour.png": "fig1_behaviour.png",
    "../results/figures/paper_fig2_mechanism.png": "fig2_mechanism.png",
    "figures/blog_6_nothing_happened.png": "fig3_erasure.png",
    "figures/paper_fig3_code.png": "fig4_code.png",
    "figures/blog_7_map_beats_model.png": "fig5_map_vs_model.png",
}

# characters pdflatex cannot take from UTF-8 input directly
UNICODE = {"²": r"\textsuperscript{2}", "≈": r"\ensuremath{\approx}", "≤": r"\ensuremath{\le}",
           "≥": r"\ensuremath{\ge}", "−": r"\ensuremath{-}", "×": r"\ensuremath{\times}", "→": r"\ensuremath{\to}",
           "α": r"\ensuremath{\alpha}", "Δ": r"\ensuremath{\Delta}", "ℝ": r"\ensuremath{\mathbb{R}}",
           "∈": r"\ensuremath{\in}", "ᵀ": r"\ensuremath{^{\top}}", "′": r"\ensuremath{'}", "⁻": r"\ensuremath{^{-}}",
           "¹": r"\textsuperscript{1}", "³": r"\textsuperscript{3}", "½": r"\ensuremath{\tfrac12}",
           "¼": r"\ensuremath{\tfrac14}", "¾": r"\ensuremath{\tfrac34}", "°": r"\ensuremath{^\circ}",
           "⁰": r"\textsuperscript{0}", "₂": r"\ensuremath{_2}", "·": r"\ensuremath{\cdot}", "✓": r"\checkmark"}


def preprocess(md: str) -> tuple[str, str, str, str]:
    lines = md.splitlines()
    title = lines[0].lstrip("# ").strip()
    author = lines[2].strip("* ").strip()
    body = "\n".join(lines[3:])
    m = re.search(r"## Abstract\n\n(.*?)\n\n## ", body, re.S)
    abstract = m.group(1).strip()
    body = body[:m.start()] + body[m.end() - 3:]

    def fig(mm):
        path, cap = mm.group(1), mm.group(2)
        cap = re.sub(r"^Figure \d+\.\s*", "", cap.strip())
        return f"![{cap}](figures/{FIGS[path]}){{width=100%}}\n"

    body = re.sub(r"!\[Figure \d+\]\((.*?)\)\n\*(.*?)\*\n", fig, body, flags=re.S)
    return title, author, abstract, body


def main():
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    for src, dst in FIGS.items():
        shutil.copy((PAPER / src).resolve(), HERE / "figures" / dst)
    title, author, abstract, body = preprocess((PAPER / "draft.md").read_text())
    header = "\n".join([r"\usepackage{amssymb}", r"\usepackage{newunicodechar}"] +
                       [rf"\newunicodechar{{{k}}}{{{v}}}" for k, v in UNICODE.items()])
    (HERE / "header.tex").write_text(header + "\n")
    meta = HERE / "meta.yaml"
    meta.write_text("---\n" + f"title: {title!r}\nauthor: {author!r}\n" + "abstract: |\n" +
                    "\n".join("  " + l for l in abstract.splitlines()) + "\n---\n")
    src = HERE / "body.md"
    src.write_text(body)
    subprocess.run([PANDOC, str(meta), str(src), "-s", "-o", str(HERE / "main.tex"), "--from",
                    "markdown+implicit_figures+pipe_tables", "-V", "documentclass=article", "-V", "geometry:margin=1in",
                    "-V", "fontsize=11pt", "-V", "colorlinks=true", "-H", str(HERE / "header.tex"),
                    "--shift-heading-level-by=-1"], check=True)
    for f in (meta, src):
        f.unlink()
    print(HERE / "main.tex")
    if shutil.which("pdflatex"):
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=HERE,
                               capture_output=True, text=True)
        tail = "\n".join(r.stdout.splitlines()[-15:])
        print(tail if r.returncode else f"built {HERE / 'main.pdf'}")


if __name__ == "__main__":
    main()
