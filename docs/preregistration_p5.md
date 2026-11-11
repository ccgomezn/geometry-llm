# Preregistration: P5, is the 2-D code a genuine encoding, and is the decodable map used?

*Committed 2026-11-11, before any P5 data exists. Gemma 3 12B only (Colab, one GPU). 3-object scenes, `closer_than`
excluded, as in P4c.*

## Why

P4c confirmed that swapping a learned 2-D subspace of the moved object's sentence (layer 14) moves the answers (0.72 on
12B, 0.83 on 27B; random 0.00). Two questions remain open.

1. **Illusion.** Subspace interventions can work through directions that do not carry the variable in ordinary runs
   (Makelov et al., 2023). Two independent trainings found overlapping subspaces (principal angles 47° and 56°;
   random 2-D subspaces: cosine ≤ 0.07), which argues against an arbitrary solution but does not settle it.
2. **Use of the map.** Probes read composed coordinates (R² 0.74 at the object list, layer 21), but no tested site
   carries them into the answers (P3, E3). Is that information used anywhere?

## P5a: does the code change linearly with the stated offset? (natural encoding)

- Subspace: the preregistered P4c 12B subspace (`results/p4/gemma3-12b_L14_relation_confirm_k2.pt`), layer 14.
- Data: 600 new aligned pairs (seed 107), built as in P4c. For each pair and each token *j* of the moved object's
  sentence (indexed from the sentence end), Δz_j = Qᵀ(h_src − h_base) ∈ ℝ², and Δoff = the change in that sentence's
  stated (dx, dy).
- Model: Δz_j ≈ W_j Δoff (no intercept), one 2×2 map per token index *j* < 8 counted from the end, one shared map for
  earlier tokens. Pair-grouped 5-fold cross-validated R².
- Null: the same fit for 200 random 2-D subspaces.
- **Rule.** *Linear encoding* if CV R² ≥ 0.5 and above the null's 99th percentile. *No linear encoding* if CV R² <
  0.2 or within the null's 95th percentile. Otherwise *partial*.

## P5b: does writing the code from the offset move the answers? (write-in test)

- Fit W_j on the P5a pairs. On the 300 P4c test pairs (seed 106), instead of copying the source run's subspace
  coordinates, set them to base + W_j Δoff at every token of the moved object's sentence. Same evaluation as P4c
  (interchange accuracy, model-relative, questions whose clean base and source answers differ).
- Also evaluated on the same pairs: the copy edit (P4c procedure; expected ≈ 0.72) and a write-in through a random 2-D
  subspace (its own W_j fitted the same way).
- **Rule** (ratio = write-in ALL accuracy / copy-edit ALL accuracy, same pairs):
  - *genuine encoding* if ratio ≥ 0.5 and random write-in ≤ 0.05;
  - *consistent with an illusion* if ratio < 0.2;
  - *partial* otherwise.

## P5c: does erasing the decodable coordinates hurt answers about worked-out positions? (erase the map)

- Fit: activations at the object-list positions for 2,000 new 3-object scenes (seed 201), every layer 12–30.
  Scene-centred, as in P2. At each layer, iterative nullspace projection (INLP; Ravfogel et al., 2020): fit the P2
  ridge probe for (x, y), add its two directions to the basis, project them out, repeat until the probe's
  cross-validated R² < 0.1 or 20 rounds (rank ≤ 40).
- Erasure: remove that basis from the residual stream at **every token position**, keeping each direction's mean
  (h ← h − ((h − μ)·B)Bᵀ, μ = mean list-position activation at that layer). Configurations: single layer 14, 21 or
  28; and a window, every layer 12–30 with its own basis. Control: a random orthonormal basis of the same rank at the
  same layers.
- Test: 300 new chain scenes R → M → C (seed 202) with all questions passing the 1-unit / 10° margins. *Composed*
  questions involve the two-step object C; *stated-only* questions involve only R and M.
- Metric: accuracy against the layout; drop = clean accuracy − erased accuracy; *excess drop* = drop with the probe
  basis − drop with the random basis (same configuration). Pair-bootstrap 95% CIs over scenes.
- **Rule** (over the four configurations):
  - *map used* if, at some configuration, the composed questions' excess drop ≥ 0.10 and exceeds the stated-only
    questions' excess drop by ≥ 0.05;
  - *map not used (in this subspace)* if the composed excess drop ≤ 0.03 at every configuration and the probe's R²
    after erasure is ≤ 0.1 on held-out scenes at the erased layers;
  - *unclear* otherwise.
- Caveat stated in advance: the basis is fitted where probes read the map (the object list). Information coded
  differently elsewhere, or nonlinearly, is not removed; "not used" means not used in this linear subspace.

## Not decided by these rules

P5 does not test the 27B model (it does not fit one Colab GPU at full precision).
