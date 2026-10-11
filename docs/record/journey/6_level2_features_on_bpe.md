# Research journey — Part 6: Level 2: Features on a BPE Model

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §23–§29. [← Part 5](5_grounding_and_controls.md) · [index](README.md) · [Part 7 →](7_the_causal_reckoning.md)

*(In the single-file journey this part was headed "Part VI — The Synthesis" and ran to the end of the document.)*

*The two negative findings of Parts IV and V, taken together, point at an
experiment neither of them tested: use the tokenizer that wins, and spend the
morphology somewhere else.*

---

## 23. Level 2 — Morphology as Features on a BPE-Tokenized Model

§22 concluded with a decision: stop competing with BPE over segmentation
and instead use BPE tokens while injecting morphological knowledge as embedding
features. This section records the build and the result.

**The result is the strongest in the project, and the first cell that beats its
own best baseline.**

### 23.1 A bug found before anything ran

The BPE + context-fact path had been written in §17 but never executed.
Reviewing it before committing compute turned up a defect that would have
silently voided the entire experiment.

`ByteLevel` folds the preceding space INTO the following token, so a
word-initial token spans `" cat"` and its **start offset lands on whitespace** —
which no spaCy token covers. The code indexed annotations by that start offset:

```
old logic (arr[start]) : aligned 1/6 = 17%
fixed logic (scan span): aligned 6/6 = 100%
```

Word-initial tokens are **78% of the BPE stream** (211,300 of 270,848, measured
in the category bit accounting). Every BPE feature-injection run would have
received `none` for nearly every token, trained normally, and produced a
convincing null result. The cache would then have needed rebuilding.

This is the same class of error as the region-attribution bug in
`coverage_economics` (§19) and the grouping bug in the collision audit
(§17). **Alignment logic is now treated as the default suspect in this
codebase**, and a regression test guards it.

### 23.2 Design decisions taken to avoid re-runs

Three, all chosen specifically because a mistake would have cost a multi-hour
cache pass.

**One cache serves every BPE experiment.** As soon as any context fact is
requested, BPE now computes all six — `dep`, `pos`, `ectx`, `mstruct`, `mrole`,
`mroot` — in a single pass and writes one codes file. Computing only the
requested fact would have meant a fresh spaCy + MorphoLex pass for every new
experiment.

**Cardinality separated from width.** Root identity has ~16k classes and cannot
one-hot into 16 dimensions, so blocks now declare `n_classes` and `width`
independently. When `width < n_classes` the frozen representation becomes a
**fixed random projection** — every class gets a distinct, unchanging vector
with no parameters. That is **Lever 3 from §11**, unbuilt since then.

**Entities carry their own root.** A preflight found 39.1% of BPE tokens sitting
in words for which no root was returned. An entity's lexical identity is itself
(`london` is its own lemma), so entity surface forms became root classes:
15,105 roots + 1,259 entities + `none` = **16,365 classes**. This only moved the
figure to 37.6%, because most of the remainder is character-fallback words, not
entities — the same coverage limit that sank Q1, resurfacing at the feature
layer. A test now asserts the block's class count matches the analyser's
inventory exactly, since a mismatch of one would make every cached code decode
to the wrong root.

### 23.3 The preflight, and what it predicted

A pre-flight script measured, on 300 validation lines and in six seconds,
everything that could void the run:

| check | result |
|---|---|
| alignment rate | **99.99%** (the old bug would show ~20%) |
| code distribution | all six facts vary; none constant |
| tokens carrying a root | **42.9%** |
| within-token-id variance, `mroot` | **11.1%** of ids receive >1 code |
| int16 safety | 16,365 < 32,767 |
| parameter arithmetic | every cell has FEWER parameters than baseline |

Two of those numbers set expectations honestly in advance. The root is
informative for only 42.9% of tokens, and it is genuinely *contextual* for only
11.1% of token ids — for the rest, `Ġrunning` always has root `run`, which a
static embedding table could already learn. **The prediction recorded before
running was 1–3%, not the 6.4% seen for `dep`.**

The cache build then took **4h37m** for 120,625,719 training tokens, at a
measured alignment rate of 99.98%.

### 23.4 The results

Baseline is the parameter-matched `bpe_17k` cell from §21: 1.3477 BPB.

| cell | ppl | BPB | vs base | params vs base |
|---|---|---|---|---|
| + `mroot` **SHUFFLED** (learned) | 68.26 | 1.3519 | **+0.31%** | −0.24% |
| baseline (bpe 17k) | 67.36 | 1.3477 | — | — |
| + `mstruct` **SHUFFLED** (frozen) | 66.60 | 1.3440 | −0.27% | −2.63% |
| + `mroot` frozen **projection** | 63.66 | 1.3296 | −1.34% | −5.26% |
| + `mstruct` frozen one-hot | 62.25 | 1.3225 | −1.87% | −2.63% |
| + `mroot` learned | 59.42 | 1.3076 | −2.98% | −0.24% |
| **+ `mroot`+`mstruct`+`mrole` learned** | **56.72** | **1.2927** | **−4.08%** | **−4.51%** |

Monotonic, nothing overfitted, every cell still descending at the step ceiling,
and **every cell carries fewer parameters than the baseline it beats.**

### 23.5 Seed replication — the last thing being asserted rather than shown

| cell | seed 0 | seed 1 | spread | mean |
|---|---|---|---|---|
| baseline | 1.3477 | 1.3482 | **0.0005** | 1.3480 |
| `mroot` frozen projection | 1.3296 | 1.3280 | **0.0016** | 1.3288 |
| all three facts, learned | 1.2927 | 1.2837 | 0.0090 | 1.2882 |

The frozen cell's spread (0.0016) is four times tighter than the learned cell's
(0.0090), and the reason is structural: **a fixed random projection has no
factor parameters to vary with the seed.** Only the trained 112 dimensions
differ between runs. That is a small piece of evidence for the frozen arm that
is not about perplexity at all — it is more reproducible by construction.

- **seed noise (max spread): 0.0090 BPB**
- **effect (mean difference): 0.0598 BPB**
- **effect / noise: 6.6×**
- **mean improvement: 4.44% BPB**

All four cross-seed pairings hold: every treatment run beats every baseline run.
The baseline's own spread of 0.0005 is remarkably tight, which is itself
reassuring about the apparatus.

### 23.6 Three findings beyond the headline

**The facts compose.** Root alone gives 0.0401 BPB; structure alone 0.0252;
combined with role, 0.0550 — **84% of what pure additivity would predict** *(revised to 93% in §26.4, once `mrole`'s contribution was measured rather than assumed)*. So
structural morphology carries information the root does not; it is not a coarse
proxy for lexical identity.

**Lever 3 works, and is now replicated.** The frozen random projection —
16,365 classes into 16 fixed dimensions, zero trainable factor parameters —
beats baseline by **0.0192 BPB (1.42%), a two-seed mean at 2.1x the noise
floor** *while
carrying a 5.26% parameter handicap*. Learned still wins (−2.98%), but that
comparison is confounded by capacity, so the honest statement is "learned is
better, though by less than the raw gap suggests."

**Both shuffled controls remove the gain, and the direction of the residual is
NOT interpretable.**

| | shuffled vs baseline | real vs shuffled |
|---|---|---|
| `mroot` (learned) | +0.0042 BPB | **−0.0443 (4.9x noise)** |
| `mstruct` (frozen) | −0.0037 BPB | **−0.0215 (2.4x noise)** |

The seed replication of section 23.5 established a noise floor of **0.0090 BPB**.
Both *shuffled vs baseline* differences fall below it, so neither the claim that
shuffled roots hurt nor that shuffled structure helps is supported, and no
mechanism should be inferred from their difference.

What the controls DO establish is the claim they were built for: **real beats
shuffled** by 2.4x and 4.9x the noise floor respectively. Same distribution,
same width, same parameter count, only the alignment to tokens destroyed — so
the gain is linguistic content rather than the mere presence of a structured
input.

*(An earlier draft of this section read the shuffled-vs-baseline signs as a real
asymmetry and explained them via the section 21 scale finding. That was an
over-interpretation of differences inside the noise floor, and it was caught
only when the seed replication supplied a floor to compare against. Recorded
here rather than silently removed: the measurement that makes an effect
claimable is the same one that makes a smaller effect unclaimable.)*

### 23.7 The oracle question, substantially answered

§21.6 flagged that `dep` comes from a full-sentence parse and therefore
leaks information about tokens not yet predicted. The concern was that most of
the Q2 gain was this oracle effect.

| feature | loss reduction | lookahead? |
|---|---|---|
| `dep` on morph | −3.75% | **yes** — full-sentence parse |
| `mroot` on BPE | **−2.98%** | **no** — the root depends only on the word |

**About 80% of the effect survives with no lookahead at all.** These are
different features on different tokenizers, so this is strong evidence rather
than proof — but the worry was that the gain was *mostly* oracle, and it plainly
is not. Morphological features are causally available during generation in a way
the dependency parse is not.

### 23.8 What the thesis now claims

> **Morphological knowledge improves English language modeling — as embedding
> features, not as segmentation.** Injected into a parameter-matched BPE model
> it reduces bits-per-byte by **4.44%** (mean of two seeds, 6.6× the seed
> noise), using **4.5% fewer trainable parameters**, with a shuffled control
> confirming the gain is linguistic content and no lookahead in the features.

The project's arc is now coherent end to end:

1. Morphological segmentation was tested against BPE in English and lost by
   13.5% under a parameter-matched comparison (§18, §21).
2. The loss was localised precisely: OOV coverage, unconditional decomposition
   of frequent words, and the fact that BPE has already absorbed English's eight
   productive affixes (§19, §22).
3. The same morphological knowledge, delivered as **features** on the better
   tokenizer, improves it by 4.44% (this section).

The negative result is not a failure that was later rescued; it is the
measurement that told us where to spend the morphology.

---

---

---

## 24. Where the Project Stands

### The finding

> **Morphological knowledge improves English language modeling as embedding
> features, not as segmentation.**

Injected into a parameter-matched BPE model, root identity, word structure and
morpheme role together reduce bits-per-byte by **4.44%** — the mean of two
seeds, against a seed noise of 0.0090 BPB, an effect-to-noise ratio of **6.6×** —
using **4.5% fewer trainable parameters** than the baseline they beat. A
shuffled-code control removes the gain — and for learned factors performs
*worse* than baseline — so what helps is the linguistic content rather than the
presence of extra structure. Unlike the dependency feature, none of these
facts requires lookahead.

> **Three results sit near 4.5%, and they are not the same result.** They come
> from different experiments, on different metrics, with different tokenizers,
> and conflating them in a short talk is the easiest mistake available here.
>
> | figure | experiment | metric | tokenizer | section |
> |---|---|---|---|---|
> | −4.46% / −4.58% | frozen `dep` vs baseline | per-token perplexity | morph / det | §18 |
> | −4.47% / −4.64% | the same, bucketed overall | per-token perplexity | morph / det | §19 |
> | **−4.44%** | Level-2 morphological facts vs baseline | **bits-per-byte** | **BPE** | §23 |
>
> **Only the third is the thesis result.** The first two measure a syntactic
> feature on a tokenizer that loses to BPE; the third measures morphological
> features on the tokenizer that wins. Quote the third, name the metric, and
> name the baseline.

### How each claim is supported

| claim | evidence | section |
|---|---|---|
| morpheme segmentation loses to BPE in English | 13.5% BPB under matched parameters | §18, §21 |
| the loss is coverage plus unconditional decomposition | regional decomposition of the gap | §19 |
| the loss is expected for English specifically | Hou et al. find the same; English is the floor | §20, §22 |
| linguistic features in the embedding help | 4.44% BPB, replicated across two seeds | §23 |
| the gain is linguistic content, not structure | shuffling removes the gain: real beats shuffled by 2.4-4.9x the seed noise | §21, §23 |
| the gain concentrates where data is scarce | rare tokens benefit ~7.5× more | §19, §23 |
| injection supplies precision, not information | a baseline that never saw the labels reads them at 0.802 vs 0.371 chance | §25 |
| freezing preserves legibility through depth | frozen 0.959 vs learned 0.938 at the final block, replicated | §25 |
| freezing is not the mechanism | learned factors beat frozen under matched capacity | §21 |
| cross-tokenizer BPB needs auditing | a 6.3% "win" reversed to a 9.7% loss | §17, §18 |

### The unified observation

Two independent experiments, using different mechanisms, produced the same
shape:

| | rare items | frequent items |
|---|---|---|
| injected dependency role vs none | −14.5% | −1.9% |
| morpheme segmentation vs BPE | −12.8% | +10.5% |

**Linguistic structure helps where statistical learning is data-starved and
costs where statistics are already sufficient.** This is the claim the project
actually supports, and it explains both the positive and the negative results
rather than treating one as a success and the other as a failure.

### What is not established

Each of these is measured rather than speculated, and all are tabulated with
magnitudes in `docs/reference/apparatus.md` §7.

- **The dependency feature is an oracle.** It comes from a full-sentence parse,
  so it leaks information about tokens not yet predicted. Roughly 80% of the
  effect survives with a feature that has no lookahead (§23), but the
  dependency number itself is an upper bound.
- **Most cells ran at a single seed.** Only the baseline and the best Level-2
  cell were replicated.
- **No downstream task was evaluated.** Every comparable paper has one; this is
  the standard rebuttal to a perplexity-only result.
- **The probing evaluation was never run.** It was promised in §4 as the
  mechanistic half of the story and the script exists, but no classifier was
  trained. This is the largest piece of unfinished evaluation.
- **No run reached convergence.** All stopped still improving at the 50,000-step
  ceiling, so every comparison is at fixed compute rather than at convergence.
- **English only, at 3.4–5.2M parameters.** The literature suggests both choices
  work against the hypothesis.

### What would be worth doing next

In rough order of value per hour.

**The legibility probe.** §14 and §11 argued that freezing buys *guaranteed
legibility* as well as lower perplexity. The perplexity half was refuted in §21;
the legibility half has never been tested, and it is the one place freezing can
still win. Checkpoints exist, so this needs no training.

**Entity vocabulary rebuild.** 77.5% of character fallback is capitalised words
absent from a vocabulary built on 6,000 articles with a single-word threshold of
10, tuned once and never revisited (§6, §19). This is the largest single
component of the Q1 deficit and the cheapest to attack.

**A causal dependency feature.** Derive the syntactic annotation from a
prefix-only parse, removing the lookahead. This converts the §23 upper bound
into a claim about causally available features.

**A downstream task.** A linear probe for POS or topic classification on frozen
representations would corroborate that the bits-per-byte differences mean
something practically.

**Level 1 — the routed tokenizer.** §22 laid out a design that routes words to
BPE or to morphological decomposition by frequency, with the crossover as a
free parameter and both baselines as its endpoints. The measurements motivating
it exist; it was never built.

---

## 25. The Legibility Probe — What Injection Actually Provides

Section 2 promised two things from freezing: lower perplexity, and probing
showing the injected dimensions are used. Section 21 refuted the first —
learned factors beat frozen by 1.7–2.3% under matched capacity. The second had
never been tested, and it was the one place freezing could still win by
construction: a frozen one-hot is trivially decodable, while a learned table is
free to collapse classes together whenever the loss does not punish it.

This section closes that promise. The answer splits the original hypothesis in
two, and only half of it survives.

### 25.1 What was run

A linear probe on the hidden state after every block, against three checkpoints
per pipeline — baseline, frozen `dep`, learned `dep` — predicting the coarse
dependency label. Chance is reported as the **majority-class rate (0.371)**, not
1/n, because the label distribution is skewed and a probe scoring 0.90 on a 90%
skewed label has learned nothing.

The **baseline arm is the informative one**: it never saw the labels, so
whatever it scores is what the model inferred from the language-modeling
objective alone.

Linear rather than non-linear on purpose: linear probing asks whether the
information is present in the form attention heads can actually consume. A
strong non-linear probe can extract information the model itself cannot easily
exploit, which answers a less useful question.

No training was required; the probe reads saved checkpoints.

### 25.2 Results, and their replication

| model | chance | emb | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| morph baseline | 0.371 | 0.779 | 0.801 | 0.809 | 0.812 | **0.802** |
| morph + frozen dep | 0.371 | 0.999 | 1.000 | 0.999 | 0.985 | **0.959** |
| morph + learned dep | 0.371 | 1.000 | 0.998 | 0.988 | 0.966 | **0.938** |
| det baseline | 0.371 | 0.775 | 0.798 | 0.809 | 0.814 | **0.803** |
| det + frozen dep | 0.371 | 0.999 | 0.999 | 0.997 | 0.982 | **0.955** |
| det + learned dep | 0.371 | 1.000 | 0.998 | 0.988 | 0.965 | **0.935** |

**Across all fifteen layer-by-arm pairs, the largest morph-vs-det discrepancy is
0.004.** Two independently trained pipelines, six checkpoints, agreement to the
third decimal. Whatever these numbers measure, they are not noise.

### 25.3 Finding 1 — injection supplies precision, not information

> **Scope correction (see §26.6).** The conclusion of this subsection holds for
> `dep` and **does not generalise to `mroot`**, the feature the headline result
> actually uses. A BPE baseline probed for root identity reaches only 42% of the
> headroom and *decays* with depth. For that feature injection supplies genuinely
> novel information, not merely precision. The finding below is correct as
> stated about dependency role; it was over-generalised when first written.

**A model that never saw a dependency label recovers it at 0.802–0.803**, which
is 70% of the headroom above chance. It does so **at the embedding layer**
(0.775–0.779), before any attention has run: token identity alone predicts
syntactic role well, because most words have a strongly preferred role.

So the injected fact is **not information the model lacks**. Injection raises
0.812 → 0.959, closing **76–78% of the remaining gap**. What it supplies is
*certainty*.

This reframes the Q2 result and, more importantly, explains it. Section 23
measured the gain concentrating in rare tokens: −14.5% for tokens seen 100–1,000
times against −1.9% for the most frequent. That is the same phenomenon from the
other side. **Where the model has enough data to infer syntax itself, injection
adds little; where it does not, injection substitutes for the missing data.**
The probe and the frequency stratification are one finding measured two ways.

It also means section 2's original framing — "guaranteed information channels
the model would otherwise lack" — was wrong in a specific and documentable way.
The model does not lack the information. It lacks confidence in it, on rare
items. Recorded rather than quietly corrected: the hypothesis was falsifiable
and this is the measurement that falsified part of it.

### 25.4 Finding 2 — freezing preserves legibility, modestly

Frozen leads learned at every layer past the first, and the lead grows
monotonically with depth:

| | L2 | L3 | L4 |
|---|---|---|---|
| morph, frozen − learned | +0.011 | +0.019 | **+0.021** |
| det, frozen − learned | +0.009 | +0.017 | **+0.020** |

The decay is the mechanism. From embedding to final block, frozen loses 0.040
(morph) / 0.044 (det); learned loses 0.062 / 0.065 — **about 50% more
degradation**. A learned table may compress the eight classes whenever the loss
does not punish it; a fixed one-hot cannot.

So section 2's second promise is **partially met**. Freezing does preserve the
injected signal better through depth, replicated across both pipelines. But the
advantage is 0.02 accuracy, bought at a 1.7–2.3% perplexity cost.

**Verdict: a real but modest trade-off, not a decisive win.** Freezing is
defensible when interpretability matters more than perplexity, and not
otherwise. That is a narrower claim than the project began with, and it is the
one the evidence supports.

### 25.5 Limitations of this measurement

The probe uses `dep` codes from the same full-sentence spaCy parse used to train
the injected models, so the baseline's 0.802 partly reflects the model inferring
*lookahead-informed* labels. All three arms share that, so the comparison is
sound, but the absolute number should not be read as "the model discovers syntax
causally".

One probe seed per checkpoint. Given agreement to 0.004 across two independently
trained pipelines, seed sensitivity is unlikely to matter here, but it was not
measured.

---

---

## 26. Closing the Reviewer Gaps — Held-Out Evaluation and the Four Missing Cells

Sections 23 and 25 established the Level-2 result and the legibility probe. An
adversarial audit of the claims against the evidence (a review register, R-01 to R-12, since retired; every item is closed here or carried into `docs/GAPS.md`) then found
eleven gaps, three of which a committee would raise immediately. This section
records closing them, because two of the outcomes **change claims made earlier
in this document**.

### 26.1 What the audit found

The audit read each claim and asked what measurement supported it. Three
findings mattered:

- **The headline cell had no control of its own.** Shuffled controls existed for
  `mroot` alone and `mstruct` alone, but never for the three-fact combination
  actually being claimed.
- **Morphology was never separated from word identity.** `mroot` supplies both
  at once, and the shuffled control destroys both, so it cannot distinguish
  them. (Still open — see 26.6.)
- **There was no test set.** Every number came from validation, and the training
  loop selects the reported checkpoint by validation loss. Selection and
  reporting on the same split, 101 evaluations per run.

### 26.2 Held-out evaluation — every claim survives

All 25 checkpoints were re-scored on the WikiText-103 **test** split, which the
project had never touched. No retraining: `scripts/eval/evaluate_split.py` loads each
checkpoint and evaluates it.

| comparison | validation | **test** | sign holds |
|---|---|---|---|
| all-three vs baseline | −0.0598 | **−0.0562** | yes |
| `mroot` learned | −0.0404 | −0.0373 | yes |
| `dep` learned (BPE) | −0.0389 | −0.0367 | yes |
| `mstruct` frozen | −0.0255 | −0.0262 | yes |
| `mroot` frozen projection | −0.0192 | −0.0151 | yes |
| `ectx` | −0.0133 | −0.0112 | yes |
| `mroot` real vs shuffled | −0.0444 | **−0.0445** | yes |
| `mstruct` real vs shuffled | −0.0216 | **−0.0216** | yes |

**The headline result on held-out test: 1.3710 → 1.3148, a reduction of 0.0562
BPB (4.10%), 6.6x the test-set noise floor of 0.0086** — the same ratio as on
validation.

Two details are worth reporting. The **control comparisons are nearly identical
across splits** (−0.0444/−0.0445 and −0.0216/−0.0216) because both arms of a
control share the same selection bias, which therefore cancels exactly. And the
validation→test shift, though it looks variable overall (sd 0.0062), is **tight
within each tokenizer**:

| group | n | mean shift | sd |
|---|---|---|---|
| bpe_17k | 13 | +0.0247 | 0.0019 |
| morph | 5 | +0.0166 | 0.0015 |
| det | 5 | +0.0100 | 0.0009 |

The variation is *between* pipelines — a property of the tokenizer, not of any
arm — so selection bias cancels in every reported comparison. Q1 on test:
BPE-17k 1.3710 against morph 1.5480 (+12.9%) and det 1.5361 (+12.0%), slightly
narrower than on validation but unchanged in direction.

### 26.3 The headline control

| cell | val | test | vs baseline (test) |
|---|---|---|---|
| baseline | 1.3480 | 1.3710 | — |
| all-three **shuffled** | 1.3314 | 1.3542 | −0.0167 |
| all-three **real** | 1.2882 | **1.3148** | **−0.0562** |
| **real vs shuffled** | −0.0432 | **−0.0395** | **4.6x noise** |

The claimed cell is now directly controlled.

One finding must be reported honestly: **shuffled all-three is itself 1.8x noise
better than baseline**, unlike the single-fact shuffles, which sat below the
floor. Three frozen dimensions of consistent-but-meaningless structure appear to
act as mild regularisation at T=29. This does not threaten the result — real
still beats shuffled by 4.6x — but the control should be described as
*"shuffling removes 72% of the gain"*, not *"shuffling destroys it"*.

### 26.4 `mrole` contributes nothing, and the composition becomes exact

`mrole` had only ever appeared inside the three-fact combination, so its
individual contribution was unknown — which also confounded the composition
claim.

| fact | test BPB | gain vs baseline | vs noise |
|---|---|---|---|
| `mrole` alone | 1.3741 | **+0.0031** (worse) | 0.4x |
| `mstruct` alone | 1.3448 | −0.0262 | 3.0x |
| `mroot` alone | 1.3337 | −0.0373 | 4.3x |

**`mrole` is flat.** Knowing whether a token sits in the prefix, root or suffix
region of its word adds nothing once root identity and word structure are
present.

With its contribution measured, the composition arithmetic is exact:
`mroot` + `mstruct` + `mrole` gains sum to 0.0644 (validation) against 0.0598
actual — **93% of additive**, revised up from the 84% reported in §23, where
`mrole`'s contribution was unmeasured and wrongly assumed positive. `mroot` and
`mstruct` are therefore close to independent.

**Consequence for the design:** a two-fact configuration (`mroot` + `mstruct`,
T=24 instead of 29) should give essentially the headline result with fewer
dimensions and one less moving part. Recorded as the next configuration to test.

### 26.5 The oracle argument becomes a direct measurement

§18 and §23 argued that the Q2 gain was not mostly a lookahead artefact by
comparing `dep` on **morph** against `mroot` on **BPE** — different features on
different tokenizers, which the text conceded was "strong evidence, not proof".

`dep` on BPE closes that gap:

| feature, BPE-17k, learned | test BPB | gain | lookahead |
|---|---|---|---|
| `dep` | 1.3343 | 0.0367 | **yes** — full-sentence parse |
| `mroot` | 1.3337 | **0.0373** | **no** — word-local |

Same tokenizer, same baseline, same training configuration. **The
no-lookahead morphological feature slightly outperforms the oracle syntactic
one.** *(Retracted in §30: `mroot` is NOT lookahead-free — it carries within-word lookahead, so this compared one oracle against another.)* The hedged inference is retired: the oracle is not doing the work.

### 26.6 The probe result does NOT generalise — and that helps

§25 concluded, from `dep`, that *"injection supplies precision, not
information"*: a baseline that never saw the labels already read them at 0.802
against 0.371 chance.

Running the same probe for `mroot` on BPE gives the opposite picture.

| model | chance | emb | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| bpe baseline | 0.571 | 0.806 | 0.803 | 0.803 | 0.784 | **0.751** |
| bpe + `mroot` frozen | 0.571 | 0.892 | 0.876 | 0.869 | 0.851 | **0.816** |
| bpe + `mroot` learned | 0.571 | 0.887 | 0.865 | 0.860 | 0.833 | **0.792** |

| | headroom used at final block | trend with depth |
|---|---|---|
| `dep` on morph, baseline | **68.5%** | 0.779 → 0.802, **improves** |
| `mroot` on BPE, baseline | **42.0%** | 0.806 → 0.751, **decays** |

**A BPE model actively loses root information as it goes deeper**, whereas a
morpheme model sharpens its implicit syntax. So for the feature the headline
result actually uses, injection is **not** merely supplying precision — the
baseline genuinely lacks the information, and loses what little it has.

§25's conclusion therefore holds for `dep` and **not** for `mroot`. It is
corrected there rather than deleted: the two features differ, and the difference
explains why `mroot` produces a larger BPB gain than `dep` despite `dep` having
lookahead.

Frozen also leads learned at the final block by **+0.024**, replicating §25's
legibility finding (+0.021 for `dep`) on a different feature and a different
tokenizer.

### 26.7 Where the register stands

| audit finding | status |
|---|---|
| R-01 headline control | closed, 26.3 |
| R-03 no test set | closed, 26.2 |
| R-04 oracle tokenizer-confound | closed, 26.5 |
| R-05 probe generalisation | closed, 26.6 — and it changed §25 |
| R-07 `mrole` untested | closed, 26.4 |
| R-12 thin run records | closed — provenance now captured per run |
| **R-02 morphology vs word identity** | **open — the last substantive question** |
| R-06, R-08, R-10, R-11 | documentation items, recorded as limitations |
| R-09 projection seed | optional |

R-02 is now more interesting than when it was raised. §26.6 showed the baseline
genuinely lacks root information; R-02 asks whether the information injection
supplies is **morphological** or merely **word-boundary**. The `mhash` control —
identical cardinality and width, per-word consistent, morphological grouping
destroyed — answers it. It requires a cache rebuild, which is why it is last.

> **Answered in §27, and the answer changes the thesis.** The control BEATS the
> treatment at every frequency: the effect was word identity, and morphological
> grouping is an actively worse way to deliver it. What survives is coarse
> morphological *structure*, which cannot be word identity at 8 classes and
> contributes independently on top of it.

---

---

## 27. The Word-Identity Control — What the Effect Actually Was

The audit's second critical finding (R-02) was that morphology had never been
separated from word identity. `mroot` gives each BPE fragment the root of its
word, which supplies two things at once: **word identity** — which fragments
belong together, something BPE's frequency-driven merges genuinely lack — and
**morphological relatedness**. The thesis claimed the second; nothing run so far
could distinguish it from the first.

This section closes that. The result falsifies the morphology framing, replaces
it with something sharper, and identifies which morphological claim survives.

### 27.1 The control

`mhash` assigns each surface word a code by hashing it into **the same
inventory** the root feature uses (16,365 classes). Identical cardinality,
identical width, identical projection path, per-word consistency preserved —
morphological grouping destroyed.

| condition | word identity | morphological grouping |
|---|---|---|
| `mroot` | yes | yes |
| `mhash` | yes | **no** |
| shuffled | no | no |

The existing shuffled control could not do this job: it permutes codes *within a
window*, destroying both properties together. It proves the signal must be
aligned; it says nothing about which aligned property matters.

### 27.2 The control beats the treatment

| cell | T | n | val BPB | val gain | **test BPB** | **test gain** |
|---|---|---|---|---|---|---|
| baseline | 0 | 2 | 1.3480 | — | 1.3710 | — |
| `mrole` | 5 | 1 | 1.3492 | −0.0012 | 1.3741 | −0.0031 |
| `ectx` | 12 | 2 | 1.3347 | 0.0132 | 1.3600 | 0.0109 |
| `mroot` frozen | 16 | 2 | 1.3288 | 0.0192 | 1.3559 | 0.0151 |
| `mstruct` | 8 | 1 | 1.3225 | 0.0255 | 1.3448 | 0.0262 |
| `dep` | 8 | 1 | 1.3090 | 0.0389 | 1.3343 | 0.0367 |
| `mroot` learned | 16 | 1 | 1.3076 | 0.0404 | 1.3337 | 0.0373 |
| `mroot`+`mstruct` | 24 | 1 | 1.2908 | 0.0571 | 1.3183 | 0.0527 |
| all three | 29 | 3 | 1.2902 | 0.0578 | 1.3170 | 0.0540 |
| `mhash` | 16 | 2 | 1.2878 | 0.0602 | 1.3169 | 0.0541 |
| **`mhash`+`mstruct`** | 24 | 2 | **1.2805** | **0.0675** | **1.3068** | **0.0641** |

**Random grouping beats morphological grouping by 0.0198 BPB on validation and
0.0168 on test**, and `mhash` alone at T=16 beats every morphological
configuration on validation.

The bucketed comparison closes the obvious defence — that morphological sharing
should help rare words even if it loses overall:

| train frequency | `mhash` ppl | `mroot` ppl | `mroot` worse by |
|---|---|---|---|
| 4–3,911 | 529.3 | 568.7 | +7.45% |
| 3,911–13,504 | 272.2 | 288.1 | +5.84% |
| 13,504–45,502 | 94.7 | 101.8 | +7.42% |
| 45,502–312,674 | 42.9 | 48.4 | +12.93% |
| 312,674–1,988,657 | 11.0 | 11.4 | +3.97% |
| 1,988,657+ | 5.14 | 5.23 | +1.83% |

**There is no frequency band in which morphological grouping wins.**

*(These bands come from equal-mass adaptive edges. The fixed thresholds
inherited from the morpheme pipelines put 64–70% of BPE tokens in a single
bucket, which made this question unanswerable; edges are now derived once from
the baseline and applied to both models.)*

### 27.3 Why random beats morphological

Both features compress roughly 50,000 word types into 16,365 classes — same
cardinality, same lossiness. The only difference is *which* words get merged.

`mroot` merges `running`, `runner` and `ran`: words that are **distributionally
similar**, appearing in similar contexts. Collapsing those destroys distinctions
the model needs and that context cannot recover.

`mhash` merges `running` with `telephone`: words that context separates
trivially, so the collapse costs almost nothing.

**Morphological structure is worse than chance as an organising principle for
word identity, precisely because it groups the things the model must tell
apart.** This is a mechanism, not a curiosity, and it explains the whole
pattern: the gain was never morphological, and morphology was actively
interfering with the gain that was there.

### 27.4 The mechanism is confirmed — for identity, not morphology

Against baseline, `mroot` shows the sample-efficiency gradient the project
predicted from the start:

| train frequency | delta |
|---|---|
| 4–3,911 | **−20.11%** |
| 3,911–13,504 | −18.17% |
| 13,504–45,502 | −17.60% |
| 45,502–312,674 | −11.31% |
| 312,674–1,988,657 | −1.06% |
| 1,988,657+ | **−0.49%** |

Rare tokens benefit **42x** more than frequent ones. Section 19's theory was
right about the mechanism and wrong about the feature: what is scarce for rare
tokens is evidence about the word itself, and injecting word identity supplies
it. Morphology was a proxy that happened to carry some identity information.

### 27.5 What survives — the cardinality test

Sorting features by whether they *could* encode word identity is the move that
separates the claims:

| feature | classes | T | gain | can encode word identity? |
|---|---|---|---|---|
| `mhash` | 16,365 | 16 | 0.0601 | **yes — it is** |
| `mroot` | 16,365 | 16 | 0.0403 | yes, lossily |
| `dep` | 8 | 8 | 0.0389 | **no** |
| `mstruct` | 8 | 8 | 0.0254 | **no** |
| `ectx` | 12 | 12 | 0.0132 | no |
| `mrole` | 5 | 5 | −0.0013 | no, and gives nothing |

**Eight classes cannot index sixteen thousand words.** `dep` and `mstruct` are
therefore not word identity under any reading, and both produce real gains. So
low-cardinality linguistic features contribute something identity cannot.

The decisive test is compositional: does `mstruct` still add *on top of*
identity?

| seed | `mhash` | `mhash`+`mstruct` | val increment | **test increment** |
|---|---|---|---|---|
| 0 | 1.2875 | 1.2794 | +0.0081 | **+0.0104** |
| 1 | 1.2881 | 1.2816 | +0.0065 | **+0.0097** |
| mean | 1.2878 | **1.2805** | **+0.0073** | **+0.0100** |

**The increment is LARGER on held-out data than on validation** — the one
comparison in the project that improves on transfer.

**Yes.** `mstruct` retains 29% of its standalone value on top of word identity,
and the paired comparison agrees in sign and magnitude across both seeds
(range 0.0016).

A note on how this is tested. The project's usual noise floor is the largest
seed spread over replicated cells, currently 0.0105 — but that figure is
dominated by the all-three cell, which is the only one with **three** seeds, and
max-minus-min grows with n by construction. Five of six replicated cells have a
spread of 0.0022 or less. More importantly, the increment is a **paired**
comparison: same seed means same initialisation and the same data order, so the
relevant noise is the variability of the *difference* (0.0016), not of the
individual cells. On that basis the increment is supported; measured against the
conservative global floor alone it would not be, and both readings are reported.

### 27.6 The two mechanisms have different frequency signatures

`mstruct`'s contribution on top of `mhash`, by band:

| train frequency | delta |
|---|---|
| 4–3,911 | −3.08% |
| 3,911–13,504 | −3.66% |
| 13,504–45,502 | −3.55% |
| 45,502–312,674 | −3.87% |
| 312,674–1,988,657 | −1.05% |
| 1,988,657+ | +0.16% |

**Flat at roughly 3.5% across five of six bands**, against word identity's steep
20% → 0.5% gradient. The profiles are qualitatively different, which is
independent evidence that these are two mechanisms and not one:

- **word identity** is *sample efficiency* — it substitutes for evidence the
  model lacks, so it helps precisely where evidence is scarce
- **morphological structure** is *structural information* — eight bits about how
  a word is built, useful at any frequency until the word is so common that
  nothing helps

### 27.6b Transfer tells the two mechanisms apart a second time

The held-out evaluation revealed something the validation numbers alone could
not. Validation→test shift, by cell:

| cell | n | shift |
|---|---|---|
| `mstruct` | 1 | **+0.0223** |
| baseline | 2 | +0.0230 |
| `mroot` learned | 1 | +0.0261 |
| `mhash`+`mstruct` | 2 | +0.0263 |
| all three | 3 | +0.0268 |
| **`mhash`** | 2 | **+0.0291** |

**`mhash` transfers worst of any cell; `mstruct` transfers best.** And the
consequence matters:

| comparison | validation | test |
|---|---|---|
| `mhash` vs all three | −0.0024 | **−0.0001** |
| `mhash` vs `mroot`+`mstruct` | −0.0030 | −0.0014 |
| `mstruct` increment on `mhash` | +0.0073 | **+0.0100** |

On held-out data **`mhash` alone is no better than the three morphological
facts** — its validation advantage evaporates. Meanwhile the `mstruct`
increment *grows*.

This is exactly what the two mechanisms predict. Word identity is a
memorisation-like signal: it substitutes for evidence about specific words, so
it is tied to the vocabulary distribution it was fitted on and degrades when
that distribution changes. Morphological structure is eight bits about how a
word is built — a property of the language, not of the corpus — so it carries
across splits intact.

The direct comparison still holds: `mroot` loses to `mhash` on both splits, so
morphological *grouping of identity* remains counterproductive. But the broader
reading changes. On held-out data, pure word identity is not superior to
morphological features; it is merely *equal* to them, and it needs morphological
structure alongside it to reach the best result.

### 27.7 What the thesis claims now

Three findings, each measured, each controlled:

1. **Word-identity injection is the largest single effect.** Telling a subword
   model which word its fragments belong to reduces bits-per-byte by 0.0602 on
   validation and 0.0541 on test, concentrated in rare tokens with a 42x
   frequency gradient. It is also the **least transferable** feature measured:
   on held-out data its advantage over the morphological configurations
   disappears entirely.
2. **Morphological grouping of identity is counterproductive.** *(Narrowed in
   §29.9: this holds for DERIVATIONAL grouping. Inflectional grouping — lemma —
   is nearly free, keeping ~81% of the available word identity.)* `mroot` loses to
   random hashing at every frequency and on both splits, because it merges
   distributionally similar words that context cannot separate.
3. **Coarse morphological structure contributes independently and transfers
   best.** It retains 29% of its standalone value on top of identity, has a flat
   frequency profile where identity has a steep one, and its increment *grows*
   on held-out data (+0.0073 → +0.0100) while identity's advantage shrinks.

Best configuration: **`mhash`+`mstruct`** — 1.2805 BPB validation and **1.3068
on held-out test against a 1.3710 baseline: 0.0641 BPB, 4.68%, at 2.87% fewer
trainable parameters.** Neither component reaches this alone, and the
composition is what the transfer analysis says should be preferred.

This is a stronger result than the one it replaces. "Morphology helps" would
have been a claim with a confound the project had not tested. What replaced it
is a negative result with a mechanism, a positive result with a mechanism, and a
control that distinguishes them.

### 27.8 Outstanding

- ~~Held-out test scores for the new cells.~~ **Done** — all 32 checkpoints
  scored on test; every claim survives, and the transfer analysis in §27.6b
  turned out to be informative in its own right.
- **The `mstruct` increment is below the conservative global floor.** A third
  seed on both cells would settle it on either reading.
- **`mhash` is not a linguistic feature.** It is a control that turned out to be
  the best-performing cell. Whether a *better* word-identity encoding exists —
  the obvious candidate being one that avoids collisions entirely — is
  unexplored, and it is a different research question from the one this project
  set out to answer.

---

## 28. Turning the Instrument on Ourselves

A paper draft (`paper_v2`) went to review, and the reviewers asked for the
evidence behind numbers the draft asserted. Answering them took eleven
experiments and about thirty GPU-hours. The result is worth recording as a
section, because what it found was **three unaudited numbers inside a paper
about unaudited numbers**.

### 28.1 What was asked, and what it cost

The article's revision register (since retired; the article was accepted) listed sixteen items. The ones needing
compute:

| item | question | cost |
|---|---|---|
| C-01 | how large is the coverage bias, measured rather than assumed? | ~15h |
| C-02 | does the collision correction predict the cost it claims? | ~5h |
| C-05, C-06 | do the headline cells survive a second seed? | ~3h |
| C-10 | is the result specific to BPE, or to statistical tokenizers generally? | ~2h |
| C-16 | does one shared learning rate disadvantage a pipeline? | ~3h |

Estimates in the register were roughly 4x low, because changing a surface
channel changes the **id stream**, not merely the codes: every ablation needs
its own four-hour spaCy pass. That was worth learning early — it is the kind of
thing that turns a "quick check" into a lost weekend.

### 28.2 The audit predicts costs, not just violations

The strongest single outcome. Removing the inflection channel reintroduces the
`cat`/`cats` collapse, so the collision correction can be checked against a
measurement instead of trusted.

| quantity | value |
|---|---|
| predicted by `collision_audit.py` | 0.0440 BPB |
| observed cost of the collapse | **0.0321 BPB** (n=2) |
| ratio | **0.73** |
| overshoot when the correction is applied | +0.0132 (1.5x noise) |

**The correction is confirmed as an upper bound, and the bound is 27% loose** —
in exactly the direction predicted in advance, because expression (2) is
context-free while a running model disambiguates from context.

This upgrades the contribution. Before, the instrument could say *this
comparison is unfair*. Now it can say *and here is what that is worth, to within
27%, erring conservatively*.

### 28.3 Three numbers in our own paper were wrong

**Table 3's ledger.** Two entries were estimates from an assumed 2–4 bits per
character. Measured:

| component | paper | measured | ratio |
|---|---|---|---|
| coverage (punct+digits) | 0.092 | **0.0592** | 0.64 |
| collision (inflection) | 0.0440 | **0.0308** | 0.70 |

Both about 1.5x too high. Worse, the table accounted for 0.136 of a 0.1975
effect and had no account of the rest: **whitespace, case and channel
interaction were never counted at all.** The complete series is in §28.4.

**The alpha-coverage row.** `coverage_audit.py` lowercased the source text but
not the emitted tokens, so every capital in the original failed to match a BPE
token that had faithfully preserved it. BPE's alpha coverage read 95.0% where
the truth is 99.9%, and the figure carried a footnote explaining that its own
headline row was an artifact.

**Byte-fallback invisibility.** Scaling the audit to the full split exposed a
second artifact: byte tokens are written `<b00>`, so counting characters inside
token strings made every byte-fallback character read as *uncovered*. det
appeared to decline 2,153 punctuation bytes it encodes losslessly. Fixed by
decoding byte runs before counting; the asymmetry then returned to **−0.01% on
2.7x the original sample**, confirming the paper's claim on more data than it
originally used.

Three defects, all of the same family: **an instrument that disagreed with the
thing it measured about what counts as encoded.** The project has now had four
of these (journey §17, §21, and both here). Alignment and accounting logic is
the recurring failure mode, and the recurring cure is to exercise the artifacts
rather than reason about them.

### 28.4 The complete channel decomposition

Every entry measured. Baseline `det_baseline_Spdiwc` = **1.5252 (two-seed mean)**.

| channel removed | BPB | marginal | tokens lost | bits/token |
|---|---|---|---|---|
| punctuation + digits (jointly) | 1.4647 | 0.0605 | 56,451 | 1.192 |
| inflection | 1.4930 | 0.0321 | 34,440 | 1.017 |
| whitespace | 1.5008 | 0.0244 | 204,051 | 0.131 |
| case | 1.5149 | 0.0103 | 29,327 | 0.380 |
| **sum of marginals** | | **0.1273** | | |
| **joint removal (legacy, n=1)** | 1.3271 | **0.1981** | | |
| **interaction residual** | | **+0.0708** | | 36% |

**Baseline choice.** Every cell in this table is now **two-seeded**, so the
baseline is the two-seed mean and the comparison is paired by construction. The
legacy joint cell remains n=1, which is noted in its row.

*(An earlier draft computed these marginals against a two-seed baseline while
the ablations were still seed 0 — arithmetically consistent but unpaired. The
discrepancy was caught by a reviewer comparing two of the project's own
documents; `scripts/ops/check_claims.py` now detects that structure automatically.
The ablations were then replicated, which removed the question at the source
rather than resolving it by convention.)*

*(An earlier draft of this table used the two-seed mean and was internally
consistent but unpaired. The discrepancy was caught by comparing two of the
project's own documents against each other — a check nothing automated was
performing.)*

Marginals do not sum to the joint effect, and should not be presented as though
they do. Removing whitespace alone still leaves case and punctuation to segment
on; removing everything leaves total ambiguity. The channels are partly
redundant, so joint removal is **superadditive**.

**A mechanism falls out of this.** Comparing each channel's in-stream cost (what
it costs to carry) against its marginal cost (what it costs to remove):

| channel | carried | costs to remove | ratio |
|---|---|---|---|
| whitespace | 0.245 | 0.131 | 0.53 |
| case | 1.739 | 0.380 | 0.22 |
| **inflection** | 0.807 | **1.017** | **1.26** |

Below 1.0 the channel was **redundant encoding** — neighbouring tokens absorb
its information when it goes. Above 1.0 it was **doing disambiguating work**, so
removing it makes the rest of the stream harder to predict.

Inflection is the only channel above 1.0. The coverage/collision distinction is
therefore a **measurable signature** rather than a definition, which is the
empirical basis the ledger's two categories previously lacked.

### 28.5 The reversal is robust, but the credit was misassigned

The most uncomfortable finding. The paper attributes the apparent 6.31%
morphological advantage to information unfairness. That measurement was taken
with the morphological arm carrying **1.54x the parameters** — vocabulary 17,155
against 10,004.

Parameter fairness is a known axis; the paper's own §2 cites Hou et al. for it.
The *after* comparison has a parameter-matched control. The *before* comparison
does not.

And there is now a direct measurement of what vocabulary alone is worth: two
cells differing **only** in vocabulary size, same ByteLevel tokenizer, both
lossless, both two-seeded — bpe 10k at 1.3957 against bpe 17k at 1.3480, so
**0.0477 BPB**.

| | apparent advantage | final deficit | total swing |
|---|---|---|---|
| as written | +6.29% | −9.28% | **0.2186** |
| fully controlled | **+3.03%** | **−13.15%** | **0.2186** |

**The swing is identical to four decimal places.** The reversal was never at
risk: vocabulary sizes were the same before and after the repair, so the
parameter asymmetry is constant across the comparison and cannot explain a sign
change. What changes is only where the credit goes.

The fully controlled version is the better claim. The apparent advantage
shrinks, the deficit roughly doubles, and the demonstration becomes immune to
the obvious objection.

### 28.6 The result is not specific to BPE

A two-arm comparison invites one objection above all: *maybe the morpheme
pipeline only loses to this particular tokenizer.* Bostrom & Durrett (2020)
found Unigram LM recovers subword units aligning more closely with morphology
than BPE's greedy merging, so it is the strongest available third arm rather
than an arbitrary one.

| tokenizer, 17k vocabulary | BPB | morph deficit |
|---|---|---|
| BPE | 1.3480 | +13.46% |
| **Unigram** | **1.3583** | **+12.60%** |

**The morpheme pipeline loses to both statistical families at matched
vocabulary.** The objection is closed.

A secondary result worth reporting: **BPE beats Unigram by 0.0103 BPB, 4.5x the
noise floor.** Bostrom & Durrett found Unigram matching or beating BPE — but on
downstream tasks at RoBERTa scale, not on bits-per-byte at 5M parameters.
Different metric, different scale; a scale caveat rather than a disagreement.

### 28.7 One shared learning rate does not favour either arm

The draft records the shared-hyperparameter choice as an asymmetry and reads the
morphological deficit as an upper bound. Seven learning rates on both pipelines,
at a matched 10,000-step horizon:

| lr | bpe_17k | det | bpe vs its best | det vs its best |
|---|---|---|---|---|
| 1e-4 | 1.7277 | 1.9716 | +22.0% | +24.0% |
| 3e-4 | 1.5266 | 1.7766 | +7.8% | +11.7% |
| 1e-3 | 1.4294 | 1.6229 | +1.0% | +2.0% |
| **3e-3** | **1.4158** | **1.5904** | **0.0%** | **0.0%** |

**Identical optimum, near-identical response** — 17.3% versus 17.7% improvement
across the range, and the curve has flattened, so the optimum is bracketed.
There is no pipeline-specific optimum, and a shared learning rate does not
advantage either arm.

**A caveat that must travel with this.** The sweep is at a 10,000-step horizon,
and short runs systematically favour higher learning rates because the schedule
anneals over fewer steps. So this does **not** show that 3e-4 was wrong at
50,000 steps. It shows the two pipelines respond to learning rate the same way,
which is the actual confound.

### 28.8 Every headline cell replicated

| cell | seed 0 | seed 1 | spread | mean |
|---|---|---|---|---|
| morph | 1.5292 | 1.5297 | 0.0005 | 1.5294 |
| det | 1.5246 | 1.5258 | 0.0012 | 1.5252 |
| bpe ByteLevel 10k | 1.3946 | 1.3967 | 0.0021 | 1.3957 |
| bpe 17k | 1.3477 | 1.3482 | 0.0005 | 1.3480 |
| unigram 17k | 1.3594 | 1.3571 | 0.0023 | 1.3583 |

Noise floor **0.0021 BPB**. The Q1 effect is 0.1338 BPB — **63.7x the floor**.

*(This 0.0021 is the spread over the four Q1 headline cells, all n=2. Earlier
sections quote 0.0090 or 0.0105 — the spread over families that include the
three-seed all-three cell, where max-minus-min is inflated by the extra draw.
Both are correct for their own comparisons; `docs/record/results.md` §5.23 states
which applies where.)*

### 28.9 What this episode is worth

Eleven experiments found three wrong numbers, two measurement artifacts and one
misassigned attribution, in a paper arguing that unchecked numbers mislead.

That is not an embarrassment; it is the argument. A methods paper that turns its
own instrument on its own work, finds defects, and reports them with magnitudes
is making its case in the strongest available way. The alternative — a paper
whose own numbers were never audited — would have been the weaker document, and
would have failed the same review one stage later.

The core claim is untouched and better supported than before:

- the conditions are still necessary, and still not implied by parameter fairness
- the reversal still holds, with an identical swing, under full control
- the instrument now predicts costs to within a **measured** 27%
- the result is robust across two tokenizer families, two seeds and a held-out
  split

What changed is that the supporting numbers are now measured rather than
assumed. That is the difference between a claim and a result.

---

## 29. Lemma Grouping, and a Ruler for Word Identity

After the paper was sent, the thesis still carried one uncomfortable fact:
**its best-performing feature was a random hash.** §27 had established that the
Level-2 gain is word identity rather than morphology, and that morphological
grouping loses to random grouping at matched cardinality. A thesis about
morphology whose winner is non-linguistic is the first thing a committee would
name.

This section closes that gap. It also contains a bug that invalidated ten runs,
found by questioning a result that looked too clean.

### 29.1 The middle term nobody had tested

`mroot` groups words by MorphoLex root. That merges two different things:

| word | MorphoLex root | spaCy lemma |
|---|---|---|
| running | run | run |
| cats | cat | cat |
| beautiful | **beauty** | beautiful |
| reconfiguring | **configure** | reconfigure |
| demonstrated | **monster** | demonstrate |

Root grouping merges **inflectional** variants (`cats` → `cat`) *and*
**derivational** families (`beautiful` → `beauty`). Lemma grouping merges only
the inflectional ones.

§27.3 explained `mroot`'s loss by a mechanism: it merges distributionally
similar words that context cannot separate. That mechanism makes a prediction.
Inflectional variants are substitutable in context; derivational relatives
often are not. **If the mechanism is right, lemma grouping should lose much less
than root grouping** — and it might match random grouping outright.

`mlemma` hashes the spaCy lemma into the same 16,365-class inventory. Three
features now differ in exactly one respect, verified by a test: what they merge.

### 29.2 The prediction held

| feature | merges | n | BPB | gain |
|---|---|---|---|---|
| `mroot` | derivational + inflectional | 1 | 1.3076 | 0.0403 |
| `mlemma` | inflectional only | 2 | **1.2905** | 0.0575 |
| `mhash` | nothing (random) | 2 | 1.2878 | 0.0601 |

Lemma beats root by **0.0171 BPB**, 9x the noise floor. Lemma trails the hash
by 0.0027, which is 1.4x the floor — within measurement error.

Combined with morphological structure:

| configuration | n | mean BPB | gain |
|---|---|---|---|
| `mhash` + `mstruct` | 2 | 1.2805 | 0.0675 |
| **`mlemma` + `mstruct`** | **3** | **1.2820** | **0.0660** |

A difference of 0.0015, **0.7 standard deviations**. Standard deviation rather
than max-minus-min, because the comparison is n=3 against n=2 and max-minus-min
grows with n. The best single run in the project is `mlemma`+`mstruct` seed 0
at 1.2790.

**The best configuration is now fully linguistic**, and it matches the
non-linguistic ceiling within measurement error. A lemma also generalises to
unseen words and can be inspected, where a hash does neither — so it is the
feature a real system would use, at no measured cost.

### 29.3 How precise does identity need to be?

If word identity is the mechanism, gain should depend on how finely words are
distinguished. `frozen.root_classes` folds the hash inventory down to k classes
at model-build time, so the same cached codes can be tested at any k without a
cache rebuild.

### 29.4 The first sweep was invalid

The first ten runs produced a striking shape: a flat plateau from k=32 to 2,048,
then a sharp rise. It was about to be written up as a threshold effect.

It was an artifact. Coarsening had been implemented inside `FactBlock.index()`,
which runs when the cache is **built**. The cache was built once, at the full
inventory, and the model reads the stored codes directly through `.clamp()` —
it never calls `index()`. A clamp does not fold; it sends every code above the
ceiling to **one** class.

| k | words collapsed into a single class |
|---|---|
| 64 | 99.6% |
| 512 | 96.9% |
| 4,096 | 75.0% |

So the sweep measured how many words kept any identity at all, not how
precisely they were distinguished. Its plateau was simply the region where
almost none did.

A test had verified that coarsening left the cache spec unchanged. That was
true, and irrelevant: it checked the configuration, not what the model saw.

**Three fixes.** The fold moved to model time, applied to the range the cache
was actually built with, with the `none` sentinel preserved rather than folded
into class 0. The ten runs were **quarantined, not deleted** — moved to
`results_invalid/` with a manifest recording why. And because the fix changed
behaviour without changing the configuration, a fixed run would have the same
run_id as a broken one; coarsened runs now record `coarsen_semantics: 2` in
their provenance, so validity is checked rather than remembered.

### 29.5 The check for the fix failed first — correctly

`scripts/audits/check_fold.py` runs the real torch fold on the real cached codes.
Its first version required the largest class to hold under 5% of tokens.
It **failed**: about 7% at every k.

That was a flaw in the criterion, not the fold. The criterion assumed words
occur about equally often; the simulation used to test it had used uniformly
random codes, silently building in that assumption. Real word frequencies are
Zipfian, and one word alone holds 7.07% of word tokens. Its class holds at
least that much however finely everything else is folded.

The revised criterion was derived rather than guessed: with f the measured
share of the largest class, a correct fold into k classes should put about
f + (1 − f)/k in its largest class. It also applies itself to the
known-broken clamp and requires that to **fail** — so a pass means something.

| k | observed | expected | old clamp |
|---|---|---|---|
| 64 | 8.21% | 8.53% | 99.7% |
| 512 | 7.15% | 7.26% | 97.8% |
| 4,096 | 7.08% | 7.10% | 81.2% |

Within 0.31 points at every k; the clamp rejected at every k.

### 29.6 The valid curve

Every run carries `coarsen_semantics: 2`. The prediction recorded beforehand —
*gain rises steadily with k* — held.

| classes | BPB | gain | invalid run at same k |
|---|---|---|---|
| 64 | 1.3251 | 0.0229 | 1.3361 |
| 256 | 1.3174 | 0.0306 | 1.3388 |
| 1,024 | 1.3125 | 0.0354 | 1.3376 |
| 4,096 | 1.3006 | 0.0473 | 1.3283 |
| 8,192 | 1.2966 | 0.0513 | 1.3225 |
| 16,365 | 1.2878 | 0.0601 | — |

Monotonic; about **0.0045 BPB per doubling** of the class count (R² = 0.971).
Every valid point beats its invalid counterpart.

**It does not saturate.** The largest single step is the last, 8,192 to 16,365.
So 16,365 classes is not the ceiling for word identity.

### 29.7 Continuity, verified exactly

The 16,365 point came from a run *without* coarsening; the others went through
the new fold. Any difference between the two code paths would distort the top
of the curve — the region the lemma estimate below depends on.

At k = 16,365 the fold should produce the same codes as no fold at all, so a run
with coarsening switched on ought to reproduce the uncoarsened result. The
threshold for "continuous" was set in advance at 0.002.

**The result was bit-identical — 1.2875168991005173 in both — at every one of
101 evaluation points across training.** Not close: identical. The two paths are
the same computation.

### 29.8 The curve as a ruler

`mroot`, `mlemma` and `mhash` share one inventory, so the random-hash curve can
measure the linguistic groupings: *how many random classes give the same gain?*
That is how much word identity each grouping keeps.

A straight-line fit failed a sanity check — it placed the random hash at 163% of
itself, because the curve steepens at the top. Interpolating between measured
neighbours instead:

| grouping | behaves like | of nominal | range given seed noise |
|---|---|---|---|
| root (`mroot`) | ~1,800 random classes | **11%** | 1,436 – 2,288 |
| lemma (`mlemma`) | ~13,200 random classes | **81%** | 11,309 – 15,489 |
| random (`mhash`) | 16,365 | 100% | — |

**Root grouping keeps about a tenth of the available word identity; lemma
grouping keeps most of it.** The ranges are far apart even at their edges. This
puts a number on what §27.3 argued in words: merging derivational families
discards most of what makes the feature useful.

### 29.9 What changes in earlier sections

§27.7's second claim read *"morphological grouping of identity is
counterproductive."* That is too broad. The accurate statement:

> **Derivational grouping is counterproductive; inflectional grouping is nearly
> free.** Root grouping keeps about 11% of the available word identity and loses
> 0.0171 BPB to lemma grouping. Lemma grouping keeps about 81%, and matches a
> random hash within measurement error.

The thesis claim becomes:

1. **Word identity is the dominant mechanism**, and gain rises with how finely
   words are distinguished *(superseded by §30: 86-97% of this gain is
   within-word lookahead. The causally valid benefit is 0.68%, and it is
   morphological STRUCTURE, not identity — see §31.)* — about 0.0045 BPB per doubling, with no ceiling
   reached in the tested range.
2. **A linguistic grouping costs what it destroys.** Merging inflectional
   variants, which are substitutable in context, loses almost nothing. Merging
   derivational families, which are not, loses most of the identity.
3. **The best configuration is linguistic**: lemma identity plus morphological
   structure, matching the non-linguistic ceiling within 0.7 standard
   deviations.

### 29.10 What this episode shows about the process

The clamp bug was found by looking at a clean-looking result and asking what
would have to be true for it to be real — then checking the code path rather
than the configuration. The fold check then failed on its first attempt, and
that failure was *useful*: it exposed an assumption hidden in synthetic test
data. And the continuity check could easily have gone the other way.

Each of these became a working principle, tied to the incident that taught it.


### 29.11 The part-of-speech probe, and a probe that fed the model its answers

With the precision curve closed, G-07 asked whether the lower BPB came with
better internal representations — the nearest thing to a downstream test that
is meaningful at 5M parameters. The legibility probe was pointed at part of
speech on the baseline and the identity-plus-structure models.

**It crashed, and the crash was lucky.** The probe loaded one dataset holding
only the fact being probed, and fed those codes to the model *as its input*.
Every earlier probe (§25, §26.6) had asked about the very fact the model was
trained on, so the single loaded row happened to be exactly what each model
needed. The first probe of a *different* fact exposed it:

| model reads | probe fed it | outcome |
|---|---|---|
| nothing (baseline) | nothing | correct |
| `mlemma`, `mstruct` | one row of POS | crashed — two rows expected |
| `mlemma` alone | one row of POS | **silently wrong** |

The single-fact case would not have failed. POS tags are numbered 0–17, which
are valid rows of a 16,365-row lemma table, so the model would have read part
of speech as if it were lemma identity and the probe would have reported a
plausible accuracy for meaningless hidden states. The published probes were
checked and are unaffected — in each, the probed fact and the model's fact were
the same. The fix loads the model's own facts plus the probed one, locates each
row by *name* in what the loader returned, and refuses to run if the names do
not match. A count check would have passed the silent case.

**The result.** The baseline already reads part of speech at 0.883–0.890 against
a 0.206 majority rate, and no feature moves that by more than about 0.005. So
the BPB gain was not coming through better grammatical representation.

A narrower question — whether lemma grouping reads POS better than hash grouping
— was tested against a bar fixed in advance: the gap at L3 and L4 had to reach
twice the spread between two baseline seeds.

| layer | lemma − hash | 2× baseline spread | clears |
|---|---|---|---|
| L3 | +0.0065 | 0.0080 | no |
| L4 | +0.0060 | 0.0040 | yes |
| combined | +0.0063 | 0.0060 | by 0.0003 |

Borderline, so not claimed. Every lemma seed beat every hash seed, but a clean
3-against-2 split occurs by chance about one time in ten. Read against the
baseline instead, lemma sits *on* it and hash slightly *below* it — so if
anything the hash muddies part of speech rather than lemma sharpening it.

**These probes ran on leaky checkpoints** (§30 had not yet found the leak), so
they describe the representations of models that were partly reading the
answer. G-07 was reclassified as scope: standard benchmarks are uninformative at
this scale, and this probe is recorded as the representation-level evidence
available.

---

