"""Appendix tables for the TMLR version, rendered straight from the tables in results/RESULTS.md so that no number is
retyped. Run: python3 paper/tmlr/make_appendix.py  (writes paper/tmlr/appendix_tables.tex)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "appendix_tables.tex"

# (entry heading substring, table index in that entry, caption, label)
TABLES = [
    ("Bulk C–R axis test", 0,
     "Behaviour on the 5,250-question set (Gemma 3 12B), clean and under two earlier preregistered erasures: the "
     "probe-readable coordinate directions (layers 12--48, every token but the first) and a variance-matched random "
     "erasure. Excess drop = random-erasure accuracy minus map-erased accuracy; 95\\% scene-bootstrap CIs.", "bulk"),
    ("Sum-of-parts test", 0,
     "Composition test (12B; 3,000 scenes of 3--6 objects, no question). Probes for each two-step object's distance "
     "and absolute coordinates relative to the first object, at its mention in the object list and at the end of its "
     "own sentence, against the best additive code of the two sentences (one-hot offsets).", "compose"),
    ("(gemma3-12b) — preregistered\nModel **gemma3-12b**. 2400", 0,
     "Flow set, Gemma 3 12B: held-out $R^2$ of M's step $d_1$ / C's step $d_2$ by site and layer.", "flow12-r2"),
    ("(gemma3-27b) — preregistered\nModel **gemma3-27b**. 2400", 0,
     "Flow set, Gemma 3 27B: held-out $R^2$ of $d_1$ / $d_2$ by site and layer.", "flow27-r2"),
    ("(gemma3-12b) — preregistered\nModel **gemma3-12b**. 2400", 1,
     "Logit lens, 12B: share of questions on which the layer's C-versus-R logit points to the correct answer.", "lens12"),
    ("(gemma3-27b) — preregistered\nModel **gemma3-27b**. 2400", 1,
     "Logit lens, 27B.", "lens27"),
    ("(gemma3-12b) — preregistered\nModel **gemma3-12b**. 2400", 2,
     "Flow set, 12B: change in the logit difference when one step is flipped (oriented so that + is the way the "
     "flipped step points).", "flip12"),
    ("(gemma3-27b) — preregistered\nModel **gemma3-27b**. 2400", 2, "Flow set, 27B: as Table~\\ref{tab:app-flip12}.",
     "flip27"),
    ("(gemma3-12b) — preregistered\nModel **gemma3-12b**. 2400", 3,
     "12B: share of the effect of flipping C's own step carried by copying one site at one layer.", "copy12own"),
    ("(gemma3-12b) — preregistered\nModel **gemma3-12b**. 2400", 4,
     "12B: share of the effect of flipping M's step carried by copying one site at one layer.", "copy12first"),
    ("(gemma3-27b) — preregistered\nModel **gemma3-27b**. 2400", 3, "27B: as Table~\\ref{tab:app-copy12own}.",
     "copy27own"),
    ("(gemma3-27b) — preregistered\nModel **gemma3-27b**. 2400", 4, "27B: as Table~\\ref{tab:app-copy12first}.",
     "copy27first"),
    ("Stage 2: which components write the answer? (gemma3-12b)", 0,
     "12B, discovery set: the 20 largest writers of the shortcut (own\\_yes coefficient of each component's direct "
     "contribution), their weights on $d_1$ and $d_2$, and the token groups their shortcut comes from.", "writers12"),
    ("Stage 2: which components write the answer? (gemma3-27b)", 0, "27B: as Table~\\ref{tab:app-writers12}.",
     "writers27"),
    ("Stage 2: which components write the answer? (gemma3-12b)", 2,
     "12B, writers' test set: mean ablation at the final position (accuracy and regression own\\_yes / $d_1$ / $d_2$).",
     "meanabl12"),
    ("Stage 2: which components write the answer? (gemma3-27b)", 2, "27B: as Table~\\ref{tab:app-meanabl12}.",
     "meanabl27"),
    ("Stage 2b: removing the shortcut while keeping the names; who rewrites it? (gemma3-12b)", 0,
     "12B, removal set: accuracy and regression coefficients under twin-mean removal, swap and controls.", "removal12"),
    ("Stage 2b: removing the shortcut while keeping the names; who rewrites it? (gemma3-27b)", 0,
     "27B: as Table~\\ref{tab:app-removal12}.", "removal27"),
    ("Stage 2b: removing the shortcut while keeping the names; who rewrites it? (gemma3-12b)", 1,
     "12B: where the removed shortcut comes back (change in the own\\_yes coefficient by group, exact attribution on "
     "600 questions).", "repair12"),
    ("Stage 2b: removing the shortcut while keeping the names; who rewrites it? (gemma3-27b)", 1,
     "27B: as Table~\\ref{tab:app-repair12}.", "repair27"),
    ("Order and voice", 0, "Order-and-voice control (12B): rule coefficients from the pooled regression.", "ov-coef"),
    ("Order and voice", 1, "Order-and-voice control: accuracy by kind of question and rendering (MC = C's sentence "
     "last; CM = first; subj / obj = C as subject / object of its sentence).", "ov-acc"),
    ("Order and voice", 2, "Order-and-voice control: share of answers C by own\\_yes, word\\_yes and last\\_yes.",
     "ov-split"),
    ("Order and voice", 3, "Order-and-voice control: decisive questions by rendering.", "ov-dec"),
]


def entry(text: str, key: str) -> str:
    starts = [m.start() for m in re.finditer(r"^## ", text, re.M)] + [len(text)]
    hits = [(a, b) for a, b in zip(starts, starts[1:]) if key in text[a:b][:400]]
    assert hits, key
    a, b = hits[-1]                                   # the latest entry with this heading
    return text[a:b]


def cell(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    for a, b in (("%", r"\%"), ("_", r"\_"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def render(md: str, caption: str, label: str) -> str:
    rows = [r for r in md.strip().splitlines() if not re.match(r"^\|\s*:?-", r)]
    cells = [[cell(c) for c in r.strip().strip("|").split("|")] for r in rows]
    n = len(cells[0])
    lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", rf"\caption{{{caption}}}",
             rf"\label{{tab:app-{label}}}", r"\begin{tabular}{@{}l" + "r" * (n - 1) + "@{}}", r"\toprule",
             " & ".join(cells[0]) + r" \\", r"\midrule"]
    lines += [" & ".join(r + [""] * (n - len(r))) + r" \\" for r in cells[1:]]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    text = (ROOT / "results" / "RESULTS.md").read_text()
    out = ["% Generated by make_appendix.py from results/RESULTS.md; do not edit by hand.", ""]
    for key, k, cap, lab in TABLES:
        tabs = re.findall(r"(?:^\|.*\|[ \t]*$\n?)+", entry(text, key), re.M)
        out.append(render(tabs[k], cap, lab))
    OUT.write_text("\n".join(out))
    print(OUT, len(TABLES), "tables")


if __name__ == "__main__":
    main()
