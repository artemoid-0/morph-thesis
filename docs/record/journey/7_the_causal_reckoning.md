# Research journey — Part 7: The Causal Reckoning

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §30–§32. [← Part 6](6_level2_features_on_bpe.md) · [index](README.md) · [Part 8 →](8_the_controlled_programme.md)

## 30. The Lookahead in Word-Form Features

§29 closed with the identity-precision curve and a thesis claim resting on it:
word identity is the dominant mechanism, and a linguistic grouping costs what
it destroys. This section records why most of that claim did not survive, what
replaced it, and the measurement that decided it.

### 30.1 A claim asserted, never checked

§26.5 and §29 described `mroot`, `mlemma` and `mhash` as having **no
lookahead**, on the reasoning that a root depends only on its own word. That
reasoning was never checked against what "its own word" means once BPE splits
it.

Every BPE fragment of a word carries the code of the **whole** word:

| input | code attached | predicting | leak? |
|---|---|---|---|
| `Ġthe` | hash("the") | `Ġrecon` | no — "the" is complete |
| `Ġrecon` | hash("reconfiguring") | `fig` | **yes** |
| `fig` | hash("reconfiguring") | `uring` | **yes** |
| `uring` | hash("reconfiguring") | `Ġand` | no — the word is complete |

At `recon` the model has seen one fragment, yet its input already names the
whole word, and with it the fragments still to come. During generation that
code cannot exist. It is the same oracle problem `dep` was flagged for, within
a word rather than across a sentence.

This is the project's first principle exactly ("closed" means a check that could have failed, didn't): a claim that felt closed because
it had been reasoned, not because anything had been checked.

### 30.2 Measuring it

The leak can only assist predictions of fragments 2, 3 and so on of a word.
When a new word starts, the previous word is already finished and its code is
fair context. So classifying every prediction separates the two cases, on
existing checkpoints, with no training:

- **word-initial** — a new word begins; the leak cannot reach it
- **continuation** — the next fragment of the same word; the leak can
- **other** — punctuation, digits, line breaks

`scripts/eval/lookahead_check.py` does this, and reproduces each model's known BPB
gain exactly once seeds are paired, so its breakdown is trustworthy.

### 30.3 The result, and a clean split

| feature | describes | total gain | on continuation | ratio | non-leak gain |
|---|---|---|---|---|---|
| `mroot` | word form | 0.0401 | 97% | 7.22 | 0.0012 |
| `mhash` | word form | 0.0602 | 91% | 6.76 | 0.0055 |
| `mlemma` | word form | 0.0582 | 91% | 6.80 | 0.0050 |
| `mstruct` | word form | 0.0252 | 92% | 6.86 | 0.0020 |
| `mlemma`+`mstruct` | word form | 0.0687 | 87% | 6.43 | 0.0093 |
| `ectx` | sentence role | 0.0131 | 32% | 2.34 | 0.0090 |
| `dep` | sentence role | 0.0387 | **−7%** | −0.49 | 0.0412 |

Continuation targets are **13.4% of predictions**.

**Features describing a word's FORM** — its identity and its structure — place
86–97% of their gain there. Knowing a word's form tells you which fragments
come next, so they leak. Even `mstruct`, with eight classes, leaks: knowing a
word is "suffixed" tells you a suffix fragment is coming.

**Features describing a word's ROLE in the sentence** do not. `dep` puts 87% of
its gain on word-initial targets and slightly *hurts* continuations; `ectx` is
mostly word-initial. A syntactic role says little about which fragment of the
current word is next — its information concerns the *following* word.

That split was not obvious in advance, and it is mechanistic rather than
empirical bookkeeping: it follows from what each kind of feature describes.

**A caveat that must travel with `dep`.** This test looks only for leaks WITHIN
a word. `dep` and `ectx` come from spaCy, which reads the whole sentence in
both directions, so they may still carry information about later words. `dep`
is clear of one kind of leak, not of all of them. Its 0.0387 remains an upper
bound until a parse restricted to earlier words is tested (G-08).

### 30.4 What this cost

The claims in §26.5, §27 and §29 rest on FORM features, and therefore mostly on
the leak:

- **§26.5's oracle argument** compared `dep` (lookahead) against `mroot`
  ("none"). Both leak; the comparison was oracle against oracle.
- **§27.4's 42x rare-word concentration**: rare words are split into more
  fragments, so they carry more leak positions. The leak predicts exactly that
  pattern.
- **§29's precision curve**: a more precise identity is also a more informative
  leak.

Q1 is untouched — it uses no features, so the leak cannot reach it. The paper's
audit framework is untouched.

### 30.5 The fix, and what survives

The causally available information is not the current word — at any position
that word is unfinished, and we cannot even tell whether the next token
continues it. It is the **previous** word, whose end was observed when the
current one began.

`embedding.causal_shift` supplies that, derived from the cached codes at model
time with no rebuild. A word boundary is where the code changes, so runs are
found from the codes alone:

```
tokens   the   recon  fig   uring  and
codes     A      B     B      B     C
result  none     A     A      A     B
```

Every source position is strictly earlier than its target, asserted at runtime
rather than assumed.

| configuration | BPB | gain | |
|---|---|---|---|
| baseline (seed 0) | 1.3477 | — | |
| `mlemma`+`mstruct`, leaky | 1.2790 | 0.0687 | 5.10% |
| `mlemma`+`mstruct`, causal | **1.3392** | **0.0085** | **0.63%** |

**88% of the headline was lookahead.** The 0.0085 that remains is 4x the
0.0021 noise floor — real, and small.

> **Superseded.** This 1.3392 run inferred word boundaries from the codes
> (`CAUSAL_SEMANTICS = 2`) and was quarantined once that proved unreproducible
> — re-scored under tokenizer boundaries, the same checkpoint gave a different
> answer. Re-run under `CAUSAL_SEMANTICS = 3` and paired against each seed's
> own baseline, this cell is **0.0075 (n=2)**. And §31 shows the gain belongs
> to `mstruct`, not `mlemma`: structure alone gives **0.0092**. Use §31.

#### Two wrong predictions, and a bug underneath them

Three things were recorded in advance and all three failed. They are kept here
because the sequence is more instructive than the number.

**Prediction 1 — the size.** 0.005–0.010, taken from the non-leak share of the
leaky model. The first causal run returned 0.0257, apparently 2.7x too high. An
explanation was offered on the spot: a model handed the current word never
learns to use previous-word context, so the share measured on a leaky model
understates what a causal one achieves.

**Prediction 2 — the verification.** `lookahead_check` should show the
continuation ratio fall from 6.43 to about 1.0. It fell to 3.66. A second
explanation followed: continuation targets are fragments of rarer words, so a
leak-free feature should still help them more than average.

**That second explanation was tested and killed.** Adding baseline difficulty
to the tool showed continuation targets are **easier** (3.40 nats against 4.77),
not harder. A third explanation was then constructed — an asymmetry in the
shift making context stale at word-final positions.

**All of it was scaffolding over a bug.** The shift was gated on an attribute
only root-family blocks carry. `mstruct` — equally a per-word form feature, and
92% within-word leak on its own — was never shifted. The "causal" run still
leaked. `mstruct` alone leaky is worth 0.0252; the contaminated run gained
0.0257, and the difference between it and the clean run is **0.0172 of pure
`mstruct` leak**.

With every per-word fact shifted, and word boundaries taken from the
highest-cardinality fact (an eight-class feature cannot locate them — adjacent
words share codes constantly), the honest figure is **0.0085**.

The lesson, kept as a principle: two plausible explanations were
built on top of a defect instead of the defect being looked for. A story that
fits the data is not a checked story, and each new patch made the next one
easier to believe.

#### The leak-free signature

| feature | word-initial | continuation |
|---|---|---|
| `mlemma`+`mstruct`, causal | **+0.0483** | **−0.0698** |
| `dep` | +0.1487 | −0.0593 |
| `mhash`+`mstruct`, leaky | +0.0198 | **+1.4186** |

The fixed feature now behaves like `dep`: it **helps predict the next word** and
slightly **hurts** within-word continuations, where a stale previous-word code
is a distraction. That is what a word-level context feature should do, and it is
the opposite of the leaky signature. It was not designed to produce this
pattern; it produced it once the bug was removed.

### 30.6 Where the tokenizer route stands

Q1's frequency analysis uses no features and cannot leak, so it was the natural
place to look for an untainted result. On the full validation split:

| word frequency | morph vs BPE |
|---|---|
| 1–3 | **morph wins 11.4%** |
| 3–10 | **morph wins 6.7%** |
| 10–20 | tie (+0.3%) |
| 20–30 | morph loses 5.6% |
| 100–300 | morph loses 12.3% |

Morphology wins below frequency 10 and loses above 20; the crossover sits
inside that gap. The earlier 12.8% rare-word figure, from 400 lines and 5,475
bytes, becomes **9.0%** on 47,862 bytes — smaller, and confirmed.

A frequency-routed tokenizer — morphemes below the threshold, BPE above — was
then estimated from per-word costs already measured, before building anything:

| threshold | to morph | routed BPB | vs pure BPE |
|---|---|---|---|
| none (pure BPE) | 0% | 1.3471 | — |
| **freq < 10** | 6.7% | **1.3391** | **−0.0080 (0.60%)** |
| freq < 20 | 10.1% | 1.3392 | −0.0079 |
| freq < 100 | 24.7% | 1.3537 | +0.0066 |
| pure morph | 100% | 1.5241 | +0.1770 |

The estimate is an **upper bound**: it grants the router each specialist's cost
without the cost of combining them. A finer sweep was run specifically to see
whether the ceiling could be raised, and it could not — the coarse edge at 10
already sat on the crossover.

**The router was therefore not built**, by a rule fixed before the sweep: worth
building above 1%, marginal at 0.6%. It stands as an upper-bound analysis,
which costs nothing and is still a result.

**The two agree.** Routing at the tokenizer gives 0.0080 BPB as an idealised
upper bound; the causal feature at the embedding gives 0.0085 measured. They
share no code path and no failure mode, and they land within 0.0005 of each
other. That convergence is the strongest thing in this chapter: the same answer
from two directions, neither of which can leak.

### 30.6b Is `dep` an oracle? Two measurements, one of which failed

`dep` survived the within-word test — +0.1487 on word-initial targets, −0.0593
on continuations — but it comes from a bidirectional parse, so it may still
carry information about later words. At 0.0412 BPB it is the largest surviving
effect in the project, and building a causal version means re-parsing every
prefix of every sentence: roughly a billion word-parses, about thirty hours.
Worth paying only if the leak is large.

**The first attempt failed.** Comparing each word's full-sentence label against
a parse of the sentence prefix ending at that word gave 39.39% disagreement on
coarse labels — apparently damning. The structure said otherwise:

| words after it | coarse agreement |
|---|---|
| 0 | 100.00% |
| 1 | 94.70% |
| 2 | **45.83%** |
| 3 | 62.76% |
| 4 | **66.50%** |
| 5+ | 56.44% |

Genuine dependence on the future would fall monotonically as more of it is
withheld. This drops to 45.8% and climbs back to 66.5% — a 17-point swing at
n≈3,000 per bucket, about 19 standard errors. And 28% of disagreements end in
`root`: a parser must assign exactly one root to any input, so handing it a
truncated fragment makes it re-root the fragment and relabel whatever sits at
the cut. spaCy was never trained on fragments. **The number measures an
out-of-distribution artifact, not `dep`.**

**The second attempt asked the question properly.** The baseline model is
already causal — its representation at any position encodes only what came
before. So probing `dep` from the *baseline's* hidden states asks exactly what
is needed, and whatever it reads is causally available by construction.

| model | chance | emb | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| baseline (causal) | 0.455 | 0.701 | 0.736 | 0.754 | **0.785** | 0.778 |
| `dep`-injected | 0.455 | 1.000 | 0.999 | 0.998 | 0.995 | 0.975 |

A left-to-right model with no access to `dep` recovers **78.5%** of its labels,
**60.6% of the headroom** above chance. Only the remaining 21.5% is even
eligible to be future-dependent — and since a linear probe on a 5M model
failing to read something does not mean the information is absent from the
past, 78.5% is a *lower* bound on what is causally determined and 21.5% an
*upper* bound on the leak.

Set against the word-form features, which took 86–97% of their **gain** from
positions a leak could reach, `dep` is not the same kind of object.

**Status: G-22 largely resolved, residual flagged.** `dep`'s 0.0412 is
supported as mostly genuine. It is not proved, and the thirty-hour prefix-parsed
rebuild would settle it — but `dep` is a syntactic feature, and the thesis's
morphological claim rests on the 0.0085 measured in §30.5, not on this. The
residual is recorded as a limitation rather than bought out.

### 30.7 The claim now

1. **Morphological information helps English language modelling by about 0.6%
   at this scale**, measured with a feature carrying only what is available at
   prediction time — and confirmed independently by a routing estimate at the
   tokenizer level (0.60%) that shares no code path with it. *(Refined in §31:
   isolated, the effect is morphological STRUCTURE alone, 0.0092 BPB / 0.68%;
   word identity adds nothing to it.)*
2. **Most of the apparent benefit in this area is lookahead.** Features
   describing a word's form leak its unwritten fragments, inflating BPB by
   **8x** here — 0.0687 against an honest 0.0085. Features describing its
   sentence role do not. *(At three paired seeds the inflation is 5.8×: 0.0691
   against 0.0120 — §31.7.)*
3. **A third fairness condition.** The paper requires injectivity and coverage
   parity for BPB to mean what it claims. This adds: side information must be
   available at the moment of prediction. It is not implied by the other two,
   and it was worth 63% of a headline.

Item 3 is the most general result in the thesis. Anyone injecting word-level
features into a subword model faces the same trap, and standard practice does
not catch it.


### 30.8 The prefix hash, and a third boundary bug

The previous-word shift is conservative: at a word's final fragment the whole
word has been seen, yet the model still receives the word *before* it, because
it cannot know that fragment is the last. G-23 proposed a prefix hash — a hash
of the current word's fragments *so far* — which is causal everywhere and, at
the final fragment, equals the true word identity without the model ever needing
to know the word has ended.

**Built from the token stream alone.** Prefix codes depend only on the BPE ids,
so they were derived from the cache rather than re-running spaCy and MorphoLEX:
100,977,477 of 120,625,719 training positions carry a word code, and the same
word yielded the same final-fragment code across all 21,983 distinct words
checked. Causality was *demonstrated* rather than argued — changing tokens after
position *i* left every code at or before *i* unchanged, whereas a whole-word
code would have moved.

**A cache-resolution bug.** The codes key is computed from the pipeline's
free-ride fact list, not from what a cell requests. `mprefix` is never on that
list, so the key resolved to an existing file that predated it; the file
*existed*, so content-based reuse was skipped, and the run died reporting that
its own cache lacked a fact that file was never going to hold. Reuse is now
attempted whenever the exact file does not *satisfy* the request.

**The first run was contaminated — the third boundary bug.** Word boundaries
were still inferred from "where the code changes", taken from the
highest-cardinality fact. That silently required the reference fact to be
constant across a word. `mprefix` varies fragment by fragment *by design*, so
every position became its own run, the shift moved back one **token** instead of
one word, and inside a word `mstruct` received the current word's own structure
— the leak the shift existed to remove. The run reported 1.3298 and was not a
causal measurement.

Boundaries now come from the **tokenizer**, which knows them. That was the third
bug from inferring word boundaries (after the `n_full` gating and the
low-cardinality reference in §30.5), and it produced the principle *never infer
what you can look up*.

**A reproducibility failure it exposed.** Re-scoring the earlier `mlemma` causal
checkpoint under the new code gave **0.0194** nats/token where it had given
**0.0266** — the same checkpoint, different answer. The boundary rule lives in
the model's forward pass, so a checkpoint trained under one rule and evaluated
under another is not merely suspect but unreproducible. `CAUSAL_SEMANTICS` was
bumped to 3 with `boundary_source: tokenizer` recorded, and both runs were
quarantined. Re-run, `mprefix` reproduced **1.3417 exactly**.

**The result was negative.** At seed 0 the prefix hash gains 0.0060 against
0.0073 for the previous-word shift, within noise. On word-initial targets — the
only place freshness could pay — the stale feature did slightly *better*
(+0.0483 against +0.0432). Freshness buys nothing measurable. §31 then showed
why none of this mattered much: identity was not the active ingredient at all.

---

## 31. It Was the Structure All Along

§30 ended with an honest figure — about 0.0073 BPB from causal identity-plus-
structure features — and a conclusion carried over from §27: that word identity
is the mechanism and morphology the lesser contributor. This section records
the experiment that tested that conclusion on clean measurements, and the
finding that it had the mechanism backwards.

### 31.1 Every causal cell so far confounded two things

Each causal run paired a word-identity feature (`mlemma`, `mhash`, `mprefix`)
with `mstruct`, the eight-class morphological structure of the word. So none of
them could say which of the two carried the gain.

`mstruct` could not be run alone before, because eight classes cannot locate
word boundaries: adjacent words share codes constantly and their runs merge.
Once boundaries came from the tokenizer (§30.5, `CAUSAL_SEMANTICS = 3`), a
feature's cardinality stopped mattering, and each could be isolated.

### 31.2 The isolated contributions

Every figure below pairs a seed with its **own** baseline seed (1.3477 for seed
0, 1.3482 for seed 1). Earlier causal figures used the seed-0 baseline for
everything; they are superseded by these.

| cell | seed 0 | seed 1 | mean | spread |
|---|---|---|---|---|
| **`mstruct` alone** | **+0.0073** | **+0.0111** | **+0.0092** | 0.0037 |
| `mhash` alone | -0.0017 | +0.0077 | +0.0030 | 0.0094 |
| `mlemma` + `mstruct` | +0.0073 | +0.0077 | +0.0075 | 0.0003 |
| `mhash` + `mstruct` | +0.0043 | +0.0097 | +0.0070 | 0.0053 |

**Does adding identity to structure help?** Paired by seed:

| addition to `mstruct` | seed 0 | seed 1 |
|---|---|---|
| + `mlemma` | -0.0000 | -0.0034 |
| + `mhash` | -0.0030 | -0.0014 |

It never helps, and it costs something in three of four comparisons.

**Structure against identity, each alone:** `mstruct` ahead by +0.0091 at seed 0
and +0.0034 at seed 1. Same direction both times.

### 31.3 What this establishes, and what it does not

**Established:** morphological structure produces a genuine causal gain of
**0.0092 BPB (0.68%)**, positive at both seeds and above three times the noise
floor at each. Word identity adds nothing on top of it.

**Not established:** that word identity is worth nothing on its own. `mhash`
alone swung from −0.0017 to +0.0077 between seeds — a spread of 0.0094, the
largest of any two-seed cell in the project. Causal `mhash` is unusually
unstable across seeds (its combination with `mstruct` spreads 0.0053 as well),
so its own value is unresolved. What is resolved is that it contributes nothing
once structure is present.

### 31.4 The reversal, and why it is coherent

Through §27, §29 and §30 the reading was: identity dominant, structure
secondary. Causally it is the reverse. The reason the leaky measurements pointed
the wrong way is now clear — lookahead rewards whatever names the word, and an
identity code names it far more precisely than eight structural classes do. The
leak inflated identity and made it look like the mechanism.

The causal result is also coherent on its own terms. The causal feature carries
the *previous* word's property, and the model already holds that word's tokens.
So the previous word's **identity** is redundant — there is nothing to add. Its
**structure** is not: that it was suffixed, compound, or a named entity is an
abstraction over MorphoLEX the model cannot recover from BPE fragments. That is
genuine linguistic information the model does not otherwise have, and it is the
only thing that helps.

### 31.5 A fifth wrong prediction, caught in time

At n=1, `mlemma` beat `mhash` by 0.0030 and it looked like the first clean
evidence that a linguistic grouping beats an arbitrary one. The second seed
flipped the sign: `mhash` ahead by 0.0020. The means differed by 0.0005. That
was the fifth wrong prediction about these features, and the pre-registered
second seed was the only reason it did not reach a document.

It was also the wrong question. Lemma against hash compared two groupings of
identity; the isolation shows identity was not the active ingredient at all.
The linguistic effect was in the structure feature the whole time, sitting in
every cell as a supposed constant.

### 31.6 The claim now

1. **Morphological structure helps English language modelling by 0.68% at this
   scale**, measured causally and replicated across seeds. *(Revised in §31.7:
   at three paired seeds, 0.89%; and the effect is shared equally between
   morphological classes and name classes, so "structure", not "morphology".)*
2. **Word identity contributes nothing once structure is present.** The
   long-standing reading that identity was the mechanism was an artifact of
   within-word lookahead.
3. **The measurement determined the conclusion.** The same features, measured
   leakily, gave the opposite ranking of mechanisms. The three fairness
   conditions are not a technicality; here they decided which mechanism the
   thesis would have reported.

Item 3 is the most useful result for anyone else: a study injecting word-level
features into a subword model, measured without a causality check, would have
reached our §27 conclusion and been wrong about which feature does the work.

### 31.7 Three seeds, and a noise floor that was never measured

§31 rested on two seeds, and §31.3 promised the isolated cells would be
replicated. Replication came as one overnight block — baseline, full `mstruct`,
and both halves of G-25, all at three seeds — so no claim could form on a
partial set (the practice adopted after §31.5).

**The seed-2 baseline came back at 1.3573**, against 1.3477 and 1.3482. That
exposed something missed all along: baseline seeds 0 and 1 predate environment
recording, while every causal cell ran under torch 2.11.0. If the environment
had shifted, every causal gain had been measured against a stale baseline and
the headline was roughly double.

Re-running seeds 0 and 1 under torch 2.11.0 settled it: **both reproduced to
thirteen decimal places.** Training is deterministic, the originals are valid,
and seed 2 is simply an unlucky seed — it tracks the others until ~20k steps,
then falls steadily behind, with no spike.

**G-25 was answered cell to cell,** without any baseline, because all nine
cells shared one environment:

| paired difference | s0 | s1 | s2 |
|---|---|---|---|
| morphology − names | −0.0001 | −0.0012 | +0.0019 |
| full − morphology | −0.0024 | −0.0003 | −0.0011 |
| full − names | −0.0025 | −0.0015 | +0.0008 |

The morphological classes and the name classes are **tied**, each recovers most
of the effect, and neither matches the full feature alone. Both kinds of
information contribute. So the accurate word is **structure**: how a word is
built and whether it is a name, not morphology alone.

**The headline at three paired seeds:**

| | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| full `mstruct` | +0.0073 | +0.0111 | +0.0176 | **+0.0120 (0.89%)** |
| leaky `mlemma`+`mstruct` | +0.0687 | +0.0650 | +0.0737 | +0.0691 (5.12%) |

Positive at every seed; mean/se 4.0. The leak inflated the apparent gain
**5.8×**, and 17% of it was real — revising §31's 0.68%, and §30's "eightfold"
and 13%, which rested on two seeds.

**The correction that matters most is about noise.** The 0.0021 floor cited
throughout this document was the widest spread among cells with two seeds
each. Baseline seeds 0 and 1 happen to agree within 0.0005; with a third, the
baseline spreads **0.0096** — four and a half times the floor. Two seeds that
agree are one sample of the spread, and a lucky one. Every conclusion that a
difference was *real* because it exceeded 0.0021 was resting on that luck.

Most survive for other reasons. Q1 is 0.18 BPB, about 19× even the new spread.
The claims that differences were *not* real — lemma against hash, prefix against
previous-word — only get stronger. And the headline now stands on its own paired
evidence rather than a floor: three seeds, one sign. Small effects from here on
are reported as paired means with a standard error and a count of seeds agreeing
in sign.

---

## 32. A Fresh Look: What the Records Already Knew

With the register clear of claim-changing gaps (§31.7), the project was read
again as an outsider would read it — the title, the question, the raw
materials — to ask what had been missed. Three findings came out of the
existing records before any new experiment was proposed. They cost nothing.
Each had been sitting in the logs for weeks.

### 32.1 The structure gain grows through training

Every run records validation BPB every 500 steps, so the gain of any cell over
the baseline is available as a function of training time. For causal `mstruct`
(three paired seeds):

| step | baseline BPB | gain | relative |
|---|---|---|---|
| 1,000 | 1.9121 | +0.0032 | 0.17% |
| 5,000 | 1.5758 | +0.0037 | 0.23% |
| 10,000 | 1.4765 | +0.0062 | 0.42% |
| 20,000 | 1.4075 | +0.0081 | 0.57% |
| 30,000 | 1.3765 | +0.0110 | 0.80% |
| 49,999 | 1.3512 | +0.0121 | 0.89% |

The gain **rises monotonically and has not saturated**: 0.86% at 40k, 0.89% at
50k, still climbing. This is the opposite of a sample-efficiency effect. The
feature is not a crutch for a data-starved model; it is information a model can
exploit only once it has learned the basics. **The headline is a floor.**

By token frequency the gain is broad — 0.7–1.15% across every band except the
most frequent tokens (0.41%), where function words and punctuation are already
easy. It is not a rare-word effect.

### 32.2 Three temporal profiles, and why structure survives

Applied to every learned cell, the same analysis separates three shapes
(`scripts/analysis/temporal_profile.py`):

| profile | cells | at 5k steps | shape |
|---|---|---|---|
| **fades** | causal identity: `mhash`, `mlemma`, `mprefix` (+`mstruct`) | 270–600% of final | peaks at several times its final value, then shrinks toward zero |
| **plateaus** | every leaky word-form feature; `dep` | 92–130% | full gain within the first 10% of training, overshoots, holds |
| **grows** | causal structure: `mstruct` and both G-25 halves | 30–36% | still rising at 50k |

This explains what §31 could only assert. The model **learns the previous
word's identity for itself** from that word's tokens, so a feature carrying it
is worth a great deal early and nothing later — it fades. The model **never
learns "this word is suffixed, or a compound, or a name"** as well as MorphoLEX
tells it, so that feature keeps paying as the model gets better — it grows.
Identity is redundant in the limit; structure is complementary.

The plateau is what a **shortcut** looks like. A leaky code that names the
unwritten fragments of the current word is a simple, perfectly predictive
feature, and shortcut-learning theory says such features dominate the output
early in training (Geirhos et al. 2020; Shah et al. 2020; Yang & Dziugaite
2023 show spurious features are separable from the model's output in the first
epochs). The coarsened identity cells make the same point as a gradient: at
full precision the leaky code reaches 90% of its gain by 5k steps, at 64
classes only 59%. A finer identity is a stronger shortcut and saturates
faster. *(The third citation is Yang, Gan, Dziugaite & Mirzasoleiman, AISTATS
2024 — `docs/reference/related_work.md`.)*

**Two caveats travel with this.** `dep` also plateaus, and `dep` is mostly
legitimate (§30.6b) — so a plateau means *exploitable immediately*, not
*leaky*. And frozen cells are slow whatever they carry, because the projection
bottleneck limits how fast anything can be used. The profile is a third lens on
injected information, alongside the positional check and the causal
construction; it is not a leak detector on its own.

### 32.3 Morpheme tokenization learns slower

The same lens on Q1:

| step | bpe | det | gap |
|---|---|---|---|
| 500 | 2.046 | 2.428 | **18.7%** |
| 5,000 | 1.576 | 1.843 | 17.0% |
| 20,000 | 1.407 | 1.591 | 13.1% |
| 49,999 | 1.348 | 1.525 | 13.2% |

The deficit is **largest when the model has seen least data**, and shrinks as
training proceeds. `det` needs 18,750 steps to reach 1.6 BPB where BPE needs
4,500 — four times as many. The sample-efficiency premise that motivated Q1
fails on its own training curve: morphological segmentation does not learn
faster from less data, it learns slower, because it spends more tokens on the
same text.

### 32.4 What a fresh reader asks

Reading the title against the results, the questions that stand out — and what
each becomes — are recorded in `docs/DIRECTIONS.md` with costs and expected
ceilings. The ones that follow directly from this section:

- **"Your best feature is eight classes. Where is the inflection?"** The
  inflection of the context — plural, past, third-singular, comparative — has
  never been used as a feature, though spaCy computes it on every parse and
  `features/inflection.py` already classifies it for the morpheme tokenizer.
  Registered as `minfl` (G-28), one cache rebuild away. The fade-vs-grow
  result predicts it behaves like structure, not identity.
- **"Any downstream evidence?"** BLiMP minimal pairs are the standard
  evaluation at exactly this scale and test exactly the phenomena morphology
  governs. `scripts/eval/blimp_eval.py` scores existing checkpoints on the 18
  agreement and irregular-form paradigms with no training (G-29).
- **"Does the gain survive more training?"** §32.1 says it should grow.
  A convergence pair is the direct test (G-30).
- **"You chose the language where morphology matters least."** The bold
  direction. Everything but the corpus and a spaCy-only annotation adapter
  transfers; MultiBLiMP provides the evaluation across 101 languages
  (Jumelet et al. 2026), and spaCy's Ukrainian pipeline carries a
  morphologizer at 94.7% feature accuracy. Recorded as G-31.

### 32.5 What this section adds to the claim

Nothing here changes a number in §31.7. It changes what the numbers mean:

1. The 0.89% is a **lower bound**, unsaturated at the training ceiling.
2. Structure survives and identity does not for a **mechanistic** reason —
   redundancy with what the model learns — visible in the training curve.
3. Leakage has a **temporal** signature consistent with shortcut learning,
   which any run's log can show without a causality check.
4. The Q1 premise — morphology as sample efficiency — is contradicted by the
   curve, not merely by the endpoint.

