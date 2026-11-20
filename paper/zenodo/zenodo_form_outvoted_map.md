# Zenodo: new record for "The Outvoted Map", and the note on v1

## Before uploading

Done: the branch `study2-pilot` and the tag `outvoted-map-v1` are on GitHub, so every commit the paper cites is public,
and the PDF is built: `paper/zenodo/Gomez-Neira-2026-The-Outvoted-Map.pdf` (from `paper/short`, linking to the tag).

## New record

Go to https://zenodo.org → **New upload**. This is a **new record**, not a new version of v1: it is a different paper
with a different claim.

| Field | Value |
|---|---|
| Files | `Gomez-Neira-2026-The-Outvoted-Map.pdf` |
| Resource type | Publication → **Preprint** |
| Title | The Outvoted Map: LLMs Carry Spatial Geometry to the Answer, and a One-Sentence Rule Outvotes It |
| Creators | Gómez Neira, Cristhian (Personal); add your ORCID if you have one |
| Publication date | the upload date |
| Description | the abstract below |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Keywords | mechanistic interpretability; language models; spatial reasoning; world models; probing; activation patching; direct logit attribution; shortcut learning; self-repair; preregistration |
| Related works | *Is supplemented by*: https://github.com/ccgomezn/geometry-llm/tree/outvoted-map-v1 (Software). *Obsoletes*: 10.5281/zenodo.24178043 (Publication); if "Obsoletes" is not in the list, use *Continues* |
| Language | English |

### Description (copy-paste)

People often solve geometry problems with the help of mental images. We ask whether large language models do something
comparable: whether they build an internal geometric representation from text, and whether their answers depend on it.
In our task, a short text places three objects on a plane, each relative to the previous one, and asks which of the
first and the last lies farther east or north, a question that requires both steps. Gemma 3 12B, however, answers
mainly from the last object's own sentence, with a default towards the first object. As a result, when the step to the
middle object is larger and points the other way, the model is right on only 34% of these questions. Yet the
information needed for the right answer reaches the position where the answer is produced: on held-out questions of
this kind, a linear probe at that position is right 86% of the time, whereas the model is right 32% of the time (in
Gemma 3 27B, 98% versus 25%). What overrides it is a set of about twenty late components, mostly attention heads, which
impose the one-sentence rule while others counteract it. When we remove what these components carry about the last
object's direction, a weak signal in favour of the correct answer appears, but it is too weak to overcome the default.
In short, the map is carried to the answer and outvoted there. All main tests were preregistered.

Code, preregistrations (each committed before its data) and all result files: https://github.com/ccgomezn/geometry-llm/tree/outvoted-map-v1

## Note on v1 (10.5281/zenodo.24178043)

Open the v1 record → **Edit** → add this paragraph at the **top of the description** → **Save** → **Publish**. Editing
the description does not create a new version and does not change the PDF. Under *Related works*, add
*Is obsoleted by*: the new record's DOI (or *Is continued by* if that type is not offered).

> **Superseded.** This preprint is superseded by "The Outvoted Map: LLMs Carry Spatial Geometry to the Answer, and a
> One-Sentence Rule Outvotes It" (DOI: [new DOI]). Two claims in this version do not hold. Erasing the map "costs no more
> accuracy than erasing random directions" and "after erasure the probe's R² is at most 0.09" (Section 3.2) both rest on
> a one-shot erasure that does not hold in the running model, and on a random control that also erased the first token.
> With a verified erasure and a fair control, the erased directions do change the answers, mainly by feeding a
> one-sentence shortcut. The new paper reports what the model does instead and why its map goes unused. All corrected
> results are in the repository's results log.
