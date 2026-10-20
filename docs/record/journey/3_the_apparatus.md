# Research journey — Part 3: The Apparatus

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §11–§13. [← Part 2](2_building_the_tokenizer.md) · [index](README.md) · [Part 4 →](4_the_audits.md)

*Building the thing that measures. The transformer is deliberately small and
standard: it is the microscope, not the object of study. This part also settles
the metric question, which turned out to carry more weight than anything else
in the document.*

**Established:** frozen facts concatenated to trained dimensions, with the open design questions
turned into a phased comparison (§11); bits per byte as the metric across tokenizers, and a small
fixed transformer as the instrument (§12); calibration on the full corpus placed the BPE baseline at
1.438 bits per byte, learning well with headroom left (§13).

---

## 11. Designing the Frozen Embedding Layer

With the preprocessing pipelines complete, attention turned to the theoretical centerpiece: the hybrid frozen/trained embedding. Before writing any code, the design space was mapped through four questions. The guiding principle, borrowed from the three-pipeline decision: where a design choice is genuinely uncertain, turn it into a comparison rather than a guess.

### The Four Design Questions

**Q1 — What goes in the frozen dimensions?** Three philosophies were considered:
- **(A) One-hot categorical:** each frozen fact is a one-hot block (category, role, entity label). Clean indicator flags — says *that* a token is a suffix, but nothing about how `-ed` relates to `-ing`.
- **(B) Linguistic feature vectors:** encode *what kind* of morpheme using real features (`-ed → [+inflectional, +past]`, `un- → [+derivational, +negation]`). Richer — shared features create real relationships between morphemes.
- **(C) Pretrained semantic vectors:** freeze a slice of GloVe/word2vec for each root. **Discarded** — it muddles the thesis (are gains from our segmentation or from GloVe leaking in?) and breaks the "we inject *certain* facts" story, since GloVe vectors are themselves learned and noisy.

**Q2 — How many dimensions frozen vs trained?** Not a free hyperparameter — T falls out of the encoding scheme in Q1 (one-hot → T≈23). What is free is total width N. The concern: at N=512, a 23-dim frozen block is only 4.5% of the vector — possibly too weak to matter. This produced a set of escalating "impact levers" (below).

**Q3 — Concatenation vs addition?** **Concatenation fixed, addition discarded.** Addition (like positional encoding) lets the trained part learn to cancel the frozen signal — which defeats the entire premise of *guaranteed* presence. Concatenation keeps the regions cleanly separate.

**Q4 — The ragged-structure problem.** Different tokens carry different facts: a prefix has a role but no entity label; an entity has a label but no morphological role; a character has neither. Yet all need a frozen vector of identical width. **Solution:** a fixed schema with reserved slots for every possible fact; each token fills the slots it has and zeros elsewhere. The "none" slots are explicit, meaningful positions — a character's empty entity block is a real zero-pattern, not missing data.

### The Impact Levers (Q2 Mitigations)

Acknowledged that 4.5% frozen signal at N=512 is weak, with four escalating responses ready:
1. **Small N** — at N=128 the frozen block is ~18% of the vector. Default first choice. Also faster to train across many variant combinations.
2. **Amplification** — tile/scale the frozen block to occupy more real estate. Implemented as `freeze_amplify`.
3. **Frozen random projection** — multiply the fact vector by a fixed random T×K matrix, spreading the signal across K dims in a fixed pattern.
4. **Auxiliary probe loss** — force the model to keep frozen dims legible by requiring a probe to predict category from them. Dovetails with the probing evaluation.

### The Annotation Cost of Option B

The hidden cost of B is *annotation* — assigning feature vectors to morphemes. The cost splits sharply by morpheme type:
- **Affixes (381):** closed class, hand-annotatable in an afternoon of review. The linguistically richest part — this is where morphological theory does real work.
- **Roots (~15,000):** hand annotation is infeasible. Three sub-options: **B1** (roots get one-hot flag only, no features), **B2** (coarse hand-annotated root classes — **discarded**, 15k roots is infeasible), **B3** (WordNet supersenses, 45 discrete semantic classes, looked up automatically).
- **Entities (11):** trivial, just one-hot.

This split Option B into two honest sub-variants: **B-affix** (rich affix features + one-hot roots, the B1 choice) and **B-full** (B-affix + WordNet root supersenses, the B3 choice).

### The Decision Tree and Phased Experiment Matrix

The final comparison ladder, each rung adding one idea:
```
baseline (no freezing — control)
  → onehot (categorical flags)
    → B-affix (rich affix features + root/entity flags)
      → B-full (B-affix + WordNet root supersenses)
```

Crossed with the three preprocessing pipelines:

| | baseline | onehot | B-affix | B-full |
|---|---|---|---|---|
| **bpe** | ✓ control | — | — | — |
| **morph** | ✓ | ✓ | soon | later |
| **det** | ✓ | ✓ | soon | later |

BPE only gets `baseline` (no morpheme metadata to freeze) — that cell is the ultimate reference point. **Phase 1** runs the 5 ✓ cells, which already answer two questions: does morpheme segmentation beat BPE (the baseline column), and does freezing flags help (baseline vs onehot). **Phase 2** adds B-affix; **Phase 3** adds B-full if earned.

### Implementation

`frozen_embedding.py` was written scoped to Phase 1 with hooks for the rest:
- `HybridEmbedding` — produces `[frozen (T) | trained (N−T)]` by concatenation. The frozen rows are a registered **buffer** (not a Parameter), so they never receive gradients.
- `FrozenSchema` ladder: `NullSchema` (T=0, baseline control — collapses to a plain `nn.Embedding`), `OneHotSchema` (T=23, implemented), `FeatureSchema` and `WordNetSchema` (stubbed, raise `NotImplementedError` so they can't be silently selected).
- `freeze_amplify` for Lever 2; default `embedding_dim=128` for Lever 1.

**Validation confirmed the core guarantee:** after a full optimizer step with a loss pulling on every output dimension, the frozen slice was byte-for-byte unchanged while the trained part moved (grad norm 0.17). The freeze is literally enforced, not nominal. All six token categories produced exactly the predicted 23-dim frozen vectors (3 nonzero bits each — one per block).

**One deferred decision:** the frozen matrix is currently built from vocabulary rows, which don't carry NER context — so entities currently get their category flag but a `none` entity-label slot. Baking entity *type* into the frozen vector will require either storing it in the entity vocab rows or moving frozen-vector construction to encode-time. Noted for when the entity-type signal is wanted.

---

---

## 12. The Dataset, Model, and the Fairness Metric

With the embedding written, three more pieces were needed to make Phase 1 runnable: the dataset, the model, and the training loop. Before writing them, we settled how they depend on each other.

### Coupling Analysis — Dataset vs Model

The two modules are mostly separable, touching at only three points:
- **`seq_len`** — a shared constant both obey (set to 256).
- **`vocab_size`** — owned by the pipeline/vocabulary upstream; both modules read it. BPE ≈ 10k, morpheme pipelines ≈ 16.7k.
- **Embedding ownership** — the one real architectural decision. **The model owns `HybridEmbedding`; the dataset emits integer IDs only.** Decisive because the embedding *is* the thesis contribution and its trained portion has gradients — if the dataset owned it, we'd be backpropagating into the dataloader. IDs are also tiny to pass around; vectors are not.

Once these three are pinned, the modules decouple cleanly: the dataset is a pure IDs→IDs machine, the model a IDs→logits machine.

### The Fairness Metric — the one decision with thesis weight

The pipelines expand text differently (one word → variable token count), so **per-token perplexity is not comparable across pipelines** — a model that splits words into more, individually-easier tokens looks artificially better. Three positions were considered: equal token count (simple, unfair on context), equal word count (fairer, complex), and **equal token count + a tokenization-agnostic metric (chosen)**.

**Decision: per-token perplexity WITHIN a pipeline (same tokenization, fair); bits-per-byte (BPB) ACROSS pipelines.** This keeps the dataset simple (fixed L, concatenate-and-chop) and moves the fairness burden to the metric, where the literature says it belongs. BPB chosen over BPC as the more robust normalization (handles non-ASCII like `é` cleanly). The citation lineage: Mielke (2019) → The Pile (BPB) → Paloma. The "Neural Polysynthetic Language Modelling" paper provides the exact computation recipe and a directly analogous precedent (comparing segmentation strategies via character-level perplexity). The dataset therefore tracks, per cached stream, the UTF-8 byte count of the original text, so BPB is computable at eval time.

### Other Dataset Decisions (standard answers)

- **Document boundaries:** ignored — concatenate-and-chop into fixed windows (standard GPT-2 practice).
- **Caching:** tokenize once per pipeline, save the integer stream as `.npy`, memory-map for training. spaCy + morpheme segmentation is far too slow to repeat every epoch. Cache validated: miss-then-hit reproduces identical ids.
- **Corpus cap:** a parameter (`max_articles`), value deferred until training speed was known.

### The Model — a Fixed Apparatus, Not the Variable of Interest

The transformer is **load-bearing but not the object of study.** The thesis claim is about embeddings; the transformer is the microscope through which we measure them. This flips normal priorities: we want the *fairest, most standard, most controlled* model, not the best one — identical across every matrix cell, with only the embedding varying.

Two reasons the model must be **small**:
1. **Hypothesis testability.** Frozen linguistic facts can only demonstrably help when the model can't trivially learn morphology itself. A large model on lots of data learns that `-ing` and `-ed` are related without help, washing out our effect. Small/data-limited is the regime where the hypothesis is testable.
2. **The matrix multiplies.** Many pipeline×variant runs; each must be cheap enough to iterate.

**Pinned and standard choices:**
- `d_model = N = 128` — pinned by the frozen-fraction argument (Lever 1: 23-dim frozen block is ~18% of 128).
- 4 layers (shallow on purpose — keeps reliance on the input representation, where our hypothesis lives), 4 heads (head_dim 32), `d_ff` = 4×d_model = 512.
- Learned absolute positional encoding, pre-norm blocks, GELU. Textbook GPT-2.
- **`tie_output` configurable.** Phase 1 uses `'none'` (untied, clean — frozen role stays purely on input). Option C (`tie_output='frozen'` — tie frozen columns into the output projection, a richer "frozen facts directly shape predictions" hypothesis) is reserved as a hook that refuses rather than silently misbehaving.

The result is a ~3.4M-parameter model. Everything is plain config so re-probing a different size is a one-line change.

### Calibration-First

Rather than committing to the architecture by argument, we adopted a proposal: **use the BPE-baseline cell as a calibration instrument.** Run it first, read its learning curve, and confirm the apparatus sits in the right regime — learns the text well, but with visible headroom for an embedding advantage to surface. Too-low final loss → model/corpus too easy; too-high plateau → too weak; decent-but-imperfect → the Goldilocks zone. The architecture gets *calibrated against the baseline* rather than chosen by judgment.

`train.py` reports both metrics, tracks **best-val** (not last-val, so overfitting can't contaminate the recorded number), and **early-stops** when validation plateaus, restoring the best checkpoint.

---

---

## 13. Calibration and Literature Grounding

### The Two-Run Calibration Story

**First run (3,000 articles, fast probe):** val loss bottomed at step 750 then climbed while train kept dropping — textbook **overfitting**. This was a data-quantity signal, not an architecture problem. It also confirmed the model has enough capacity to overfit, i.e. it is *not* too weak.

**Second run (full corpus, ~130M tokens):** overfitting vanished entirely. Train and val tracked together monotonically across all ~40k steps, never diverging — the signature of sufficient data. Result at the step ceiling (still slowly improving):

| Metric | Start | Best (step 39,500) |
|---|---|---|
| val loss | 9.24 | **4.17** |
| val perplexity | 10,124 | **64.9** |
| val BPB | 3.18 | **1.438** |

This is the **anchor cell** of the entire experiment — every other cell is measured against it. The regime is confirmed Goldilocks: a 156× perplexity reduction from random (genuinely learns), but BPB 1.44 is far from the ~0.8–1.0 SOTA floor (clear headroom for an embedding advantage).

For the real matrix, the step ceiling should be raised (the run was still improving at 40k) with the same patience-based early stop catching the true plateau.

### Literature Grounding (for the eventual paper)

Reference numbers were gathered with proper caveats, to avoid the central trap: **word-level vs subword/BPE perplexity are different units and not directly comparable.**

- **Word-level WikiText-103 SOTA** (≈267k vocab, adaptive softmax): Transformer-XL 18.3 test ppl (Dai et al., 2019), previous SOTA Baevski & Auli 20.5; fuller leaderboard ranges ~15.8–18.7 (Compressive Transformer, Routing Transformer, Shortformer, Perceiver AR). **Our 64.9 must NOT be compared to these** — different units.
- **BPE perplexity** (our regime): GPT-2 Small (117M, zero-shot) ≈ 29; GPT-2 XL (1.5B) ≈ 16. These are the relevant comparison class, though imprecise (GPT-2 trained on WebText, not WikiText).
- **Our placement:** a 3.4M-param model (~1/35th of GPT-2 Small) reaching 64.9 BPE ppl is exactly where a tiny model should land — sensible, with clear headroom to the 29 of a 35×-larger model and the ~18 of word-level SOTA.
- **WikiText-103 is known to overfit:** "a small dataset where strong regularization is required to prevent severe overfitting" (Perceiver AR paper). This validates our 3k-article overfitting as a documented dataset property, not a stumble. Dataset facts: 103M training tokens, 28K articles, ~3.6K tokens/article (Dai et al., 2019).

**Reporting rules locked in:** never compare raw perplexity to the 18.3 leaderboard (different units); headline cross-pipeline metric is BPB; the honest peer class is "small models trained from scratch on WikiText-103 with BPE," with GPT-2 Small's ~29 as a "what 35× more parameters buys" reference point.

*Corrected 7 October 2026 (`docs/reference/related_work.md`, section 4): the GPT-2
paper's zero-shot WikiText-103 perplexities are 37.50 (117M) and 17.48 (1.5B).
The ≈29 above is its 117M figure for WikiText-2 (29.41), and ≈16 a sliding-window
WikiText-2 figure for GPT-2 Large (16.44, the Hugging Face perplexity guide). Our
64.9 placement reads the same against 37.50: a 35× smaller model, well behind.*

---

