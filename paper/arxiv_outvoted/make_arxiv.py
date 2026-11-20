"""arXiv version of "The Outvoted Map": the named paper (paper/short) plus the methods appendix and result tables of the
TMLR version, with all LaTeX comments removed (arXiv publishes the source), \\pdfoutput=1, and the .bbl included
(arXiv does not run BibTeX).

Run: python3 paper/arxiv_outvoted/make_arxiv.py
Writes paper/arxiv_outvoted/build/ (compiled, for checking) and paper/arxiv_outvoted/build/arxiv_source.tar.gz (upload).
"""

import re
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
BUILD = HERE / "build"
SRC = BUILD / "src"

AI_NOTE = ("Claude (Anthropic) was used throughout this work: to write the experiment and analysis code, to discuss the "
           "design of the experiments and the interpretation of the results, to search the literature, and to draft "
           "and edit the text from the author's dictation. The author checked the code with an automated test suite, "
           "every number against the result files and every citation against its source, and takes full "
           "responsibility for the content.")


UNICODE = "\n".join([r"\usepackage{newunicodechar}", r"\newunicodechar{−}{\ensuremath{-}}",
                     r"\newunicodechar{→}{\ensuremath{\to}}", r"\newunicodechar{²}{\textsuperscript{2}}",
                     r"\newunicodechar{÷}{\ensuremath{\div}}"])


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        line = re.sub(r"(?<!\\)%.*$", "", line)          # trailing comments (not \%)
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def main():
    shutil.rmtree(BUILD, ignore_errors=True)
    (SRC / "figures").mkdir(parents=True)
    tex = (PAPER / "short" / "main.tex").read_text()

    code_end = "/tree/outvoted-map-v1}."
    assert tex.count(code_end) == 1, "Code and data sentence not found"
    tex = tex.replace(code_end, code_end + "\n" + AI_NOTE)
    assert tex.count("\\end{document}") == 1
    tex = tex.replace("\\end{document}", "\\input{appendix_methods}\n\n\\end{document}")
    hyper = "\\usepackage[hidelinks]{hyperref}"
    assert tex.count(hyper) == 1
    tex = tex.replace(hyper, hyper + "\n" + UNICODE)
    tex = "\\pdfoutput=1\n" + strip_comments(tex)
    (SRC / "main.tex").write_text(tex)

    for f in ("appendix_methods.tex", "appendix_tables.tex"):
        (SRC / f).write_text(strip_comments((PAPER / "tmlr" / f).read_text()))
    shutil.copy(PAPER / "short" / "refs.bib", SRC / "refs.bib")
    used = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    for g in used:
        p = Path(g)
        src = PAPER / "short" / (p if p.suffix else p.with_suffix(".pdf"))
        dst = SRC / (p if p.suffix else p.with_suffix(".pdf"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=SRC, check=True,
                   capture_output=True)
    log = (SRC / "main.log").read_text(errors="replace")
    problems = [l for l in log.splitlines() if re.search(r"Overfull|undefined|Citation .* undefined", l)]

    keep = ["main.tex", "main.bbl", "appendix_methods.tex", "appendix_tables.tex"] + \
           [str(Path(g) if Path(g).suffix else Path(g).with_suffix(".pdf")) for g in used]
    with tarfile.open(BUILD / "arxiv_source.tar.gz", "w:gz") as t:
        for k in keep:
            t.add(SRC / k, arcname=k)
    shutil.copy(SRC / "main.pdf", BUILD / "Gomez-Neira-2026-The-Outvoted-Map-arxiv.pdf")
    print(f"{BUILD / 'arxiv_source.tar.gz'}: {', '.join(keep)}")
    print("LaTeX problems:" if problems else "LaTeX: clean", *problems[:10], sep="\n  ")


if __name__ == "__main__":
    main()
