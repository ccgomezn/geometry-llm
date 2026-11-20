# arXiv submission: The Outvoted Map

Build: `python3 paper/arxiv_outvoted/make_arxiv.py`. Upload `paper/arxiv_outvoted/build/arxiv_source.tar.gz` (the
source; arXiv compiles it). The compiled PDF to check against is
`paper/arxiv_outvoted/build/Gomez-Neira-2026-The-Outvoted-Map-arxiv.pdf` (20 pages).

The tarball was checked on 2026-11-20: it compiles alone with pdflatex and no BibTeX, as arXiv does. The source has no
LaTeX comments, since arXiv publishes the source.

This version is the named paper plus the methods appendix and the 24 result tables of the TMLR version, with the
repository link and the preregistration commits. It also has the generative-AI note that arXiv's policy requires
("report in their work any significant use of ... text-to-text generative AI"); it is two sentences at the end of
"Code and data".

## Form

| Field | Value |
|---|---|
| License | CC BY 4.0 |
| Primary category | cs.CL (Computation and Language). It must be the category your endorsement is for |
| Cross-list | cs.LG (Machine Learning). Optionally cs.AI |
| Title | The Outvoted Map: LLMs Carry Spatial Geometry to the Answer, and a One-Sentence Rule Outvotes It |
| Authors | Cristhian Gómez Neira |
| Abstract | the plain-text abstract below |
| Comments | 20 pages, 3 figures. Code, preregistrations and results: https://github.com/ccgomezn/geometry-llm/tree/outvoted-map-v1 |

Do not mention TMLR anywhere (anonymity during review). After acceptance, add "Published in Transactions on Machine
Learning Research" to the comments and upload a new version.

## Abstract (plain text)

People often solve geometry problems with the help of mental images. We ask whether large language models do something comparable: whether they build an internal geometric representation from text, and whether their answers depend on it. In our task, a short text places three objects on a plane, each relative to the previous one, and asks which of the first and the last lies farther east or north, a question that requires both steps. Gemma 3 12B, however, answers mainly from the last object's own sentence, with a default towards the first object. As a result, when the step to the middle object is larger and points the other way, the model is right on only 34% of these questions. Yet the information needed for the right answer reaches the position where the answer is produced: on held-out questions of this kind, a linear probe at that position is right 86% of the time, whereas the model is right 32% of the time (in Gemma 3 27B, 98% versus 25%). What overrides it is a set of about twenty late components, mostly attention heads, which impose the one-sentence rule while others counteract it. When we remove what these components carry about the last object's direction, a weak signal in favour of the correct answer appears, but it is too weak to overcome the default. In short, the map is carried to the answer and outvoted there. All main tests were preregistered.

## After posting

Add the arXiv link to the correction note on the old Zenodo record (`paper/zenodo/correction_note_v1.md`).
