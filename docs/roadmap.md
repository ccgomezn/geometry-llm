# Roadmap: from points to geometry to seeing

The core question: when a model solves a geometry problem from text, does it build an internal spatial model that it *uses*, and what format does that model take?

The method stays fixed across experiments: behaviour gate → probe → causal edit (joint consistency) → preregistration. Only the objects get richer.

## E1 — Points (done)

Positions of points described by relations. Coordinates are decodable and composed, but not used at the list mention. See the paper draft.

## E2 — Where is the used information, and is it map-like? (running)

- **Necessity test.** Copy everything except the list positions.
- **Sentence-end probe.** Are coordinates readable at the site that patching shows is used?
- **Chain test.** Move B in A → B → C; does C follow? This tells *when* (if ever) positions are composed and stored.

Outcome decides the next step. If a used map exists (C stays after some layer; coordinates readable at the sentence), E3–E5 look for geometry *in that map*. If positions are rebuilt from sentences on demand, E3–E5 test whether richer geometry forces a map.

## E3 — Derived geometry from points

Same scenes, but the questions need derived quantities: the angle at B in triangle ABC (acute, right or obtuse), which side is longest, whether P lies inside ABC, which triangle has the larger area, and whether ABCD is a rectangle.

- **Probe:** are side lengths, angles, area and inside/outside linearly decodable, and where (at the vertices, or at the question)?
- **Edit:** move one vertex inside the model. Do *all* derived answers (angle, longest side, area, inside/outside) update together? That is the joint-consistency test for shapes.

## E4 — Textbook geometry problems

Problems stated with measurements, not coordinates: "In triangle ABC, AB = 5, BC = 12, angle B = 90°. Find AC."

- Does the model hold a configuration (vertex positions in some frame), or retrieve formulas?
- **Edit:** change BC inside the model. Do AC, the area and angle A all update consistently, as they would in a drawn figure?

## E5 — Transformations (mental rotation)

"Rotate the layout 90° clockwise around A; where is B now?"

- **Behaviour:** does error grow with rotation angle, as human mental rotation time does (Shepard & Metzler, 1971)?
- **Internals:** does the coordinate code rotate with the scene, or is the answer recomputed from the text?

## E6 — Seeing

Gemma 3 is a vision-language model. Show the *same* layouts as images.

- **Shared code:** does a probe trained on text-built coordinates read image-built ones (and vice versa)?
- **Cross-modal patching:** move an object in the image representation; do text-question answers follow?

This is the bridge from "imagining from words" to "seeing".

## Cross-cutting

- Replicate key results on a second model (Gemma 3 27B, Qwen3-14B).
- 4+ object scenes once a model passes the 4-object gate.
- Preregister each experiment before its data exists.
