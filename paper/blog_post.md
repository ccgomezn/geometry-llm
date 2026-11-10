# The AI Worked Out Where Things Are, Even When Nobody Told It

*Notes from an ongoing experiment on whether language models "picture" a scene they only read about.*

---

Read this: *"Ana is 3 units north of Bo. Cy is 2 units east of Ana."* Now answer: who is farthest from Bo?

You probably didn't reread the sentences. You placed three dots in your head and looked. Notice that nobody told you where Cy is relative to Bo: you worked it out by chaining the two sentences. Psychologists have argued for decades about what that inner picture is and what it does. I want to ask the same of AI: **when a language model reads a scene described only in words, does it build something like a map inside?**

## The setup

I generated thousands of small layouts that no model could have seen before, described only in words. Each came with questions the text never answers directly: farthest, closest, which direction, who is further north.

First, a filter: there's no point looking for a map inside a model that can't do the task. Eight open models (Qwen, Llama and Gemma, from 1.7 to 27 billion parameters) had to answer instantly, without "thinking out loud". Most struggled. The smallest that handled three-object layouts reliably was **Gemma 3 12B**, so that's the one I opened up. Before looking inside, I wrote down in advance what would count as evidence, so I couldn't move the goalposts later.

## What we found: positions nobody stated

Each object is mentioned once more after the description, in a list ("The objects are: Ana, Bo, Cy"). At each mention, I trained a simple linear decoder to read that object's x and y coordinates from the model's internal activations.

- **It works, and well:** R² = 0.74 on layouts the decoder never saw. Shuffled labels, the model's first layer and an untrained copy of the model all score zero.
- **The interesting part is objects like Cy,** whose position is never stated relative to the scene: it only follows from combining two sentences. The decoder reads those as well as directly stated ones (0.74 vs 0.76). A decoder that only gets what each object's own sentence says reaches 0.22 on them.
- **It appears suddenly.** The first 11 layers carry nothing readable. The coordinates show up around layer 12 and peak at layer 21, as if the model spends its early layers reading and then assembles the layout.

Earlier work has read real-world coordinates of famous places out of language models, which is memorised geography. For scenes described in the prompt, recent probes found categorical relations (left of, above) or weaker, fragmented coordinate signals. Here, a model works out a coordinate for every object in a scene it has never seen, including positions it had to compute.

## What's next

Being able to read a map out of a model is not the same as the model *relying* on it. That's the question we're testing now. We edit the model's internal state, moving one object inside it, and check whether all of its answers move consistently, the way they would if it were consulting a picture.

The sharpest version uses a chain: A → B → C, where C is placed relative to B. Move B inside the model. If the model has already worked out where C is, C stays put. If it rebuilds positions from the sentences every time it's asked, C follows B. Doing this layer by layer tells us whether, and when, the model "draws the picture".

The long-term goal is the step from *reading* to *seeing*: what changes when the same model is shown an actual image.

*Code, preregistration and results will be released with the paper.*
