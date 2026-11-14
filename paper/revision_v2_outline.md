# Revision v2: what changes after the map-use checks and the verified erasure

*Outline for the author (2026-11-14). Facts, numbers and structure only; the prose is yours. Every number is from
`results/RESULTS.md` (entries of 2026-11-14 unless marked).*

## 1. Statements in the v1 preprint (Zenodo) that no longer hold

| Where in `paper/draft.md` | v1 says | What we now know | Source |
|---|---|---|---|
| Abstract; §3.2 "Erasure"; §4 summary | "Erasing it everywhere costs no more accuracy than erasing random directions" | P5c's random control erased the first token (`<bos>`), which damages the model whatever is erased. Sparing it cuts the random erasure's cost from 0.237 to 0.083 (stated-only) and from 0.095 to 0.002 (composed). Against that fair control, P5c's map erasure *does* cost more: stated-only +0.081 [+0.056, +0.106], composed +0.024 [+0.014, +0.034] | Closed-loop entry, "P5c bases, first token spared" rows |
| §3.2 "Erasure" | "After erasure the probe's R² is at most 0.09" | True only on the activations the basis was fitted on. A probe refitted on the erased forward pass still reads R² 0.57 at layer 21 (0.68 clean). The map reappears in other directions downstream | Check B |
| §3.2, 4–5 objects (P6) | "did not hurt deep questions more than random" | Same one-shot erasure and same first-token control as P5c, so the same two problems apply. Not rerun | P6 entry (2026-11-12) |
| §3.4 | "Erasing it everywhere costs nothing, even when chains are long" | With an erasure verified to remove the map (below), composed accuracy drops 0.77 → 0.71; it is not "nothing" | Confirmatory-erasure entry |
| §5 Limitations, "Linear erasure" | nonlinear information may survive | Now tested: after the verified erasure the MLP probe reads nothing (≤ −0.02 at layers 14/21/28/36/44) | Confirmatory-erasure entry |
| Fig. 3 | P5c excess drops | Built on the flawed control; replace or relabel | — |

The other v1 pillars are untouched: copying 0.00, probe steering 2%, DAS at the list 0.07, DAS at the sentence 0.72 /
0.83, halfway test (switch), attention knockout (the question does not read the list), map answers 0.80 vs 0.73.

## 2. The revised story in five lines

1. The map exists and is composed (unchanged).
2. The answers do not read it: copying, steering and DAS at the list are null, and the question does not attend to the
   list (unchanged).
3. **New, methods:** erasing once does not hold in the running model, and erasing the first token breaks random
   controls. Both affect the v1 erasure claim and, plausibly, other erasure studies.
4. **New:** an erasure checked in closed loop until fresh linear *and* MLP probes read nothing leaves composed distance
   answers intact (0.75 → 0.74). Axis and direction answers lose 0.06–0.13; stated-only answers lose 0.19–0.24.
   Without a fair random control these are upper bounds (see §6, open decision 1).
5. **New, weak positive sign:** where the decoded map is wrong, the model copies its wrong answer slightly more than
   chance (Check A, +0.053 [+0.023, +0.086] at layer 21; on C–R questions +0.081 [+0.040, +0.123]).

## 3. Section plan

**Abstract.** Replace the erasure sentence. Points to carry: the map is decodable and composed; the answers do not read
it (copy/steer/attention); a verified erasure leaves composed distance answers intact; methods warning about
one-shot erasure and first-token controls.

**§3.2 The answers do not use the map.** Keep copying, steering and DAS-at-list as they are. Rebuild the erasure part:
- (a) One-shot erasure (P5c, P6) as v1 ran it, briefly.
- (b) It does not hold: Check B table, layer 21 linear 0.68 → 0.57 under the erasure, MLP 0.73 → 0.60; layer 28 MLP
  0.64 → 0.35; layer 36 linear 0.54 → 0.43.
- (c) The first-token problem: the numbers in row 1 of the table above.
- (d) Closed-loop erasure: refit on activations captured under the erasure until a fresh probe fails; layers 12–48,
  every token but the first. First attempt: 7 rounds on 2,000 scenes, fresh-scene R² still 0.13 at layers 20 and 24
  (failed its own check). Second: 4,000 scenes, stop at 0.02, dev check; 8 rounds, rank 2–40 per layer (960 in all),
  fresh-scene linear R² ≤ 0.009 at every layer, MLP ≤ −0.02.
- (e) Answers under the verified erasure (600 new chains, 10,700 questions; clean → erased):

  | questions | clean | map erased |
  |---|---|---|
  | composed, all | 0.77 | 0.71 |
  | C–R distance | 0.75 | 0.74 |
  | C–R axis | 0.73 | 0.67 |
  | C–R direction | 0.51 | 0.38 |
  | C–M axis | 0.95 | 0.86 |
  | stated R–M axis | 0.89 | 0.70 |
  | stated R–M direction | 0.89 | 0.65 |

- (f) Report the failed confirmatory test honestly: the preregistered verdict was "uninformative", because the
  "matched" random control removed far more variance than the map erasure (the map sits in low-variance directions;
  random directions of equal rank remove a median 10× more). Both random controls in that run are harsher than the
  map erasure, so its excess drops are not interpretable.
- (g) The exploratory axis finding from the first closed-loop run (C–R more east +0.06 [+0.02, +0.10] against isotropic
  random) was not confirmed. Mention it only as exploratory.

**§3.4 Why the map is not used.** Drop "erasing it everywhere costs nothing". What survives: the question does not
attend to the list (P7), boosting attention to it does not help (P10), and composed distance answers survive a
verified erasure. Note that the stated-only questions lose most under the erasure, which suggests the erased
directions overlap the stated-relation code; that is a hypothesis, not a result.

**§4 Discussion.** Add the methods lesson as its own paragraph: verify erasure on the erased forward pass with freshly
fitted linear and nonlinear probes; spare the first token and report a random control that does not break the model;
match random controls in removed variance, in both directions. Check A as a weak sign that the map and the answers
are not fully independent.

**§5 Limitations.** Replace "Linear erasure" with: verified erasure on 12B only; no fair random control for it yet;
P6 (4–5 objects) not rerun with the verified erasure.

**§6 Preregistration table, rows to add.**

| Test | Prediction | Result | Verdict |
|---|---|---|---|
| Check A, error overlap | model copies the map's wrong answers beyond a blurred map | +0.053 [+0.023, +0.086] | mistakes follow the map |
| Check B, probe after P5c erasure | MLP R² < 0.10 at L21 | 0.60 | nonlinear map survives |
| Closed-loop erasure | erasure effective, then composed excess drop | fresh-scene R² 0.135 | could not be erased |
| Confirmatory axis erasure | C–R axis excess drop vs matched random | control broke composed accuracy (0.56 < 0.57) | uninformative |

Also annotate the P5c and P6 rows: "control flawed (first token); see Check B and closed-loop entries".

**Title.** "…Its Answers Never Read" still fits the copy/steer/attention evidence, but "never" is stronger than the
erasure evidence now allows (upper bounds of 0.06–0.13 on composed axis/direction answers). Your call; an option is
to keep the title and qualify in the abstract.

## 4. Figures

- **New figure, "erasure that holds":** probe R² by layer (12–48) for clean, P5c one-shot erasure (in the running
  model, from Check B: layers 14/21/28/36) and verified closed-loop erasure (`results/erase/
  gemma3-12b_erase_axis_validation.parquet`), linear and MLP. Shows the map reappearing after one-shot erasure and
  gone after closed-loop.
- **Replace Fig. 3:** accuracy clean vs verified erasure by question class × type (table (e) above), from
  `results/erase/gemma3-12b_erase_axis_test.parquet`. Add the random bars only if the fair control is run.
- I can make both with the existing `paper/make_paper_figs.py` style.

## 5. Related work to check before claiming anything as new

- **LEACE "concept scrubbing"** (Belrose et al., NeurIPS 2023, arXiv:2306.03819): I believe it fits each layer's eraser
  on representations computed with the earlier layers' erasers applied, which is close to our closed loop. Read the
  method section before calling the closed-loop fit new; our additions would then be iterating to convergence across
  all layers and validating with fresh linear and MLP probes on held-out scenes.
- **The first-token effect** relates to work on attention sinks and very large activations at the first token. I have
  not verified the exact references; look them up before citing.
- Amnesic probing (Elazar et al., TACL 2021) used one-shot INLP with random controls; our Check B bears directly on
  that design. Phrase it as a caution, not a refutation: we only tested our setting.

## 6. Open decisions (yours)

1. **Fair random control for the verified erasure.** One scoring pass on the saved bases with random directions
   matched *downward* in removed variance (about 30 minutes on an A100). Without it, the table in (e) can only be
   reported as upper bounds, and a reviewer will ask whether the 0.06–0.13 losses are map use or generic damage. With
   it, either outcome is reportable. I recommend running it for the revision.
2. **Correct the public preprint.** v1 on Zenodo states the two erasure facts in §1 that no longer hold. Zenodo
   supports new versions under the same concept DOI; a v2 with a short correction note would fix the record.
3. **Scope of v2:** whether the triangle study (Study 2) goes in this paper or stays separate.
4. **P6 rerun** (4–5 objects) with the verified erasure: optional, for completeness.
