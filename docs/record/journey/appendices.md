# Research journey — Appendices

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** A, B, C. [index](README.md)

## Appendix A — Decision Record

*Grouped by the part in which each decision was taken. Where a decision was
later reversed or refuted, the section that did so is named.*

### Part I — the question

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Intervention point | Embedding layer | Tokenizer only; architecture change | Transformer unchanged; clean separation |
| Unit granularity | Morpheme-level | Word-level; phrase-level | Smaller vocabulary; closer to BPE; better OOV |
| Named entity handling | Atomic units | Morpheme-segment them | Proper nouns are not compositionally meaningful |
| Frozen/trained combination | Concatenation | Addition | Addition lets training cancel the frozen signal |
| Evaluation method | Perplexity + probing | Generation-based metrics | No generation needed; probing is more informative |

### Part II — the tokenizer

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Vocabulary validity | Source-based (MorphoLex) | Frequency threshold | Frequency conflates noise with rare-but-real |
| Entity threshold | Split by structure (multi-word ≥3, single ≥10) | Flat threshold; MorphoLex exclusion; curated list | Token structure is a reliable signal of misclassification risk |
| Morfessor role | Affix-only fallback | Full segmentation; removal | Roots unreliable; affixes more trustworthy — later shown to add nothing (§8) |
| Capitalisation guard | Rejected | Applied temporarily | Typographic heuristic, not a linguistic category |
| Number of pipelines | Three (bpe, morph, det) | Two | Isolates Morfessor's contribution; established the compare-don't-guess method |
| Compound splitting | Implemented, with a particle exception | Not implemented; minimum length only | Concrete wins; false-positive risk negligible for genuine OOV |
| Codebase structure | Flat, seven files | Deep hierarchy | Communication overhead — **reversed in §16** when that premise expired |

### Part III — the apparatus

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Model role | Fixed apparatus, small and standard | Best or largest model | The transformer measures the embedding; small keeps the hypothesis testable |
| `d_model` | 128 | 256/512 | Pinned by the frozen-fraction argument |
| Model depth | 4 layers | 6+ | Shallow keeps reliance on the input representation |
| Cross-pipeline metric | Bits-per-byte | Per-token perplexity | Perplexity is not comparable across tokenizations |
| Within-pipeline metric | Per-token perplexity | BPB only | Same tokenization, and more sensitive |
| Embedding ownership | Model owns it; dataset emits ids | Dataset emits vectors | The embedding is the contribution and carries gradients — **narrowed in §15** |
| Reported metric point | Best-val, early-stopped | Last-val | WikiText-103 overfits; the minimum must not be contaminated |
| Architecture selection | Calibrate against the BPE curve | Choose by argument | Measurement beats judgement |

### Part IV — the audits

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Frozen block content | Composed blocks, width derived | Fixed 23-dim schema | Sized by label inventory, not information; 12 of 23 columns were constant |
| Frozen fact axis | Static vs context source | Treat all frozen facts alike | Static is parameter tying; context is information a lookup table cannot hold |
| Dataset payload | ids + integer codes | ids only | Frozen metadata has no gradients, so the §12 reasoning never applied |
| Validity criterion | Injectivity (collisions) | Reconstruction fidelity | Non-colliding spelling variation loses no information |
| Information channels | All five emitted; each a config flag | Hardcode the fixes | Flags allow per-channel ablation |
| Unknown characters | UTF-8 byte fallback | ASCII punctuation layer | UNK is a collision sink; byte fallback makes the encoding total |
| `max_steps` | Split into ceiling and `lr_horizon` | One field for both | One field doing both made different-length runs incomparable |
| Seeding | Seeded; seed in the run identity | Unseeded single runs | A small expected effect cannot be told from initialisation noise |
| Codebase structure | Package | Flat (§10) | The premise expired, and flat was about to cause two silent bugs |

### Part V — grounding and controls

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| BPE vocabulary size | Matched to morph (17,151) | Left at 10,000 | The +51% parameter advantage was a real confound |
| Frozen vs learned factors | Both built and compared | Assume freezing is better | Factored NLMs are the direct ancestor and learn their factors |
| Factor signal scale | Exposed and tested at 0.22 | Leave one-hot at norm 1.0 | A frozen one-hot entered 4.6× louder, confounding trainability with magnitude |
| Freezing claim | Refuted for perplexity; legibility still open | Claim freezing wins | Learned beats frozen by 1.7–2.3% under a controlled comparison |
| Data-starvation sweep | Rejected as the main path | Run it as the headline | It changes the conditions rather than the method |

### Part VI — the synthesis

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Q1 strategy | Cooperate: BPE segments, morphology informs | Keep competing with BPE | The frequency table is an assignment, not a comparison |
| BPE context alignment | Scan the token's span | Index by start offset | ByteLevel folds the space into the token; 78% of the stream would have received `none` |
| BPE fact computation | Build all six facts in one pass | Build only what was requested | Each new fact would otherwise need its own multi-hour pass |
| High-cardinality facts | Separate cardinality from width; frozen becomes a random projection | One-hot only | 16,365 root classes cannot one-hot into 16 dims |
| Entity lexical identity | Entities occupy their own root classes | Fall to the `none` sentinel | An entity's lemma is itself |
| Seed replication | Two seeds on baseline and best cell | Single runs | The effect-to-noise ratio is what makes a large effect a demonstrated one |

---

## Appendix B — Section Mapping

*This document was restructured after the work was complete. Earlier drafts,
and some docstrings in the codebase, refer to the previous numbering.*

| old | new | note |
|---|---|---|
| 1, 2, 3 | 1 | merged: framing and intervention point |
| 4 | 2 | |
| 5 | 3 | |
| 6, 7 | 4 | merged: granularity and evaluation |
| 8, 10, 11, 12 | 5 | merged: the vocabulary thread, previously split |
| 22, 23 | 6 | merged: entity thresholds, moved to sit beside the vocabulary |
| 9, 14, 15, 16 | 7 | merged: tools, segmenter, format-B bug; minor bugs compressed |
| 17, 18, 19 | 8 | merged: Morfessor's null and the three-pipeline decision |
| 24, 25 | 9 | merged: compound splitting |
| 13, 20, 21, 26 | 10 | merged: first comparison, consolidation; stale status removed |
| 27 | 11 | |
| 28 | 12 | |
| 29 | 13 | |
| 30 | 14 | |
| 31 | 15 | |
| 32 | 16 | |
| 33 | 17 | |
| 34 | 18 | |
| 35 | 19 | |
| 36 | 20 | |
| 37 | 21 | |
| 38 | 22 | |
| 39 | 23 | |
| — | 24 | new: closing status and next steps |
| Probe design | linear, every layer, majority-class chance | non-linear or final layer only | linear asks whether the signal is in the form attention can consume; majority-class stops a skewed label faking a good score |
| Probe baseline arm | include a model that never saw the labels | probe only the injected models | it is the arm that reveals injection supplies precision rather than information |
| Freezing claim, final form | modest interpretability gain at a perplexity cost | decisive win, or no effect | frozen leads learned by 0.02 at the final block, replicated, but costs 1.7-2.3% perplexity |
| Thesis title | *Context-Aware Morphological Embeddings for Language Models* | original *...for Tokenizer-Free Language Models* | "tokenizer-free" was never true — every arm uses a tokenizer and the best one is BPE; "morphological" describes the headline result |
| Lever 3 claim | reported with a two-seed margin | single-seed estimate | 2.1x the noise floor is thin enough that replication was required before quoting it |
| Held-out evaluation | re-score checkpoints on test, no retraining | retrain everything, or report validation | selection and reporting shared a split; 25 checkpoints re-scored in under an hour |
| `mrole` in the headline config | measured, found flat, flagged for removal | keep it because it is morphological | +0.0031 BPB, 0.4x noise — an unused feature is an easy criticism |
| Oracle argument | direct within-tokenizer measurement | cross-tokenizer inference | `dep` on BPE removes the confound; the no-lookahead feature wins anyway |
| §25 scope | corrected in place, with the over-generalisation recorded | silently narrowed | the probe conclusion holds for `dep` and not for `mroot` |
| Word-identity control (`mhash`) | matched-cardinality random grouping | rely on the shuffled control | shuffling destroys identity AND morphology together; only a random regrouping isolates them |
| The morphology claim | falsified and replaced, not narrowed | keep it for the rare-token tail | `mroot` loses to random hashing in all six frequency bands |
| Testing the `mstruct` increment | paired across seeds | global max-spread floor | max-minus-min grows with n, and the floor is set by the only 3-seed cell |
| Headline configuration | `mhash`+`mstruct` (T=24) | `mroot`+`mstruct`+`mrole` (T=29) | 1.2805 vs 1.2902, fewer dimensions, and the claim it supports is the one the evidence supports |
| Reviewer-driven verification | run all sixteen register items | answer the cheap ones, defer the rest | eleven experiments found three wrong numbers and two measurement artifacts; deferring would have moved the failure one review stage later |
| Table 3 ledger | five measured entries plus an interaction residual | two estimated entries | the estimates were 1.5x high and covered half the effect |
| The apparent-advantage figure | report the fully controlled version (+3.03% -> -13.15%) | keep +6.31% -> -9.65% | the before-measurement was parameter-unfair; the swing is identical either way, so the controlled version costs nothing and closes an objection |
| Third tokenizer arm | Unigram LM | WordPiece, or none | Bostrom & Durrett found Unigram aligns MORE closely with morphology than BPE, so it is the strongest available competitor rather than an arbitrary one |
| Learning-rate confound | measured at a matched horizon | left as a stated asymmetry | both pipelines share an optimum and respond near-identically, so a shared LR favours neither |
| Linguistic middle term | test lemma grouping | accept that a random hash wins | §27.3's mechanism predicted lemma would lose far less than root; it did, and ties the hash |
| Invalid cardinality sweep | quarantine ten runs with a manifest | delete them, or re-run over them | same config gives the same run_id, so a re-run overwrites one seed and leaves invalid siblings to be averaged in |
| Marking fixed runs | record coarsen_semantics in provenance | rely on remembering which runs came after the fix | the fix changed behaviour without changing config, so nothing else distinguishes valid from invalid |
| Fold-check criterion | derive expected share from measured f, and require the clamp to fail | a fixed threshold | the fixed threshold assumed uniform word frequencies; Zipf put 7.07% in one class |
| Using the curve as a ruler | interpolate between measured points | fit a straight line | the line placed the random hash at 163% of itself |
| The "no lookahead" claim | measured, then retracted | left as reasoning | a root depends only on its word — but a word spans several BPE fragments, and the code names all of them |
| Where to measure the leak | split predictions by target type on existing checkpoints | build a causal feature first | free, and it decided whether the expensive fix was even needed |
| The causal feature | previous completed word | mask non-final fragments | at any position we cannot know whether the current word has ended; only the previous one is certainly complete |
| The router | estimated, not built | build at the best threshold | the idealised ceiling was 0.60% and a finer sweep could not raise it; the rule for building was set before the sweep |
| Isolating structure from identity | run each alone | keep pairing them | every causal cell confounded the two; the isolation reversed which one was the mechanism |
| Baselines for causal cells | each seed against its own baseline seed | seed-0 baseline throughout | pairing is what §27.5 and the Table 3 correction established; earlier causal figures are superseded |
| Reading `mhash` alone | unresolved, not zero | declare identity worthless | 0.0094 spread across seeds; what IS resolved is that it adds nothing to structure |
| Where to look for new directions | the existing records first | new experiments first | three findings came out of the eval logs at no cost, and each reshaped the direction list |
| The 0.89% headline | reported as a floor | reported as the effect | the gain is unsaturated at the training ceiling (§32.1) |

---

*The project asked whether injecting linguistic structure into a language
model's embedding layer helps. It does — by 4.10% bits-per-byte on held-out test
data, replicated across seeds, at 6.6x the noise floor, using 4.5% fewer
trainable parameters, though the word-identity control showed the mechanism is
not the one originally proposed. Alongside it, the measurement work became a
contribution in its own right: two checkable conditions for cross-tokenizer
comparison, an instrument that predicts the cost of violating them to within a
measured 27%, and a case study in which correcting the measurement reverses the
sign of the result. Getting there required discovering that the project's first
result was an artifact of its own measurement, building the instruments to prove
it, turning those instruments on the claims that survived, and finally turning
them on the paper itself — where they found three more unaudited numbers. That
last step is not an embarrassment. It is the argument.*

## Appendix C — The thesis's list of 15 October and its outcome sentences

*Moved verbatim from `docs/DIRECTIONS.md` ("The queue", as at commit `dd1357d`) on 20 October 2026, when DIRECTIONS was cut to the open queue. It was the only home of each item's outcome sentences. Every item is done or dropped: journey §33.98–§33.116. The section numbers in it are the thesis text's.*

| # | candidate | each outcome's sentence | GPU | ready |
|---|---|---|---|---|
| 1 | **Width 512 at 20k, seeds 1–3** (§3.3.2; conclusion 2) | *Settles near +0.011 at four seeds:* the gain does not fade with learning at any of the three widths. *Keeps falling:* at 30M parameters the tokenizer's gain fades with learning, as a feature's does | — | **done** (§33.102): +0.0102 ± 0.0006, 4/4; MultiBLiMP at seeds 2–3 prepared |
| 2 | **English part of speech without the right-context leak**, over the null, 10k and 20k, four seeds — every word tagged from its left context alone, since in English the leak is spread over thousands of words (§33.99; the thesis's §3.10, conclusion 7) | *Leak-free ≈ word form:* in both languages what a context-aware feature adds over the lexical one is mostly not context. *Leak-free clearly above the word form:* for English part of speech the sentence's context adds information of its own; for Ukrainian case it does not | ~2.3 h | its codes built and checked (§33.102); a piece on the thesis machine, where its comparisons are |
| 3 | **Fiction at full training**: both fiction nulls at fraction 0.4 (~133M tokens), 50k, seeds 0–3 (§3.3.3; conclusion 2) | *Holds:* on both corpora the gain does not fade with training. *Fades:* on fiction the tokenizer's gain belongs to short training | ~3.5 h on the second machine | prepared again (§33.103): its data stopped on a paragraph past spaCy's limit, fixed (§33.102) |
| 4a | **Part of speech switched off early at 50k**, four seeds (§3.8; conclusion 5) | *The deficit persists:* the model relies on the feature at full training too. *It closes:* at full training the model makes up the removed feature; the reliance is temporary | — | **done** (§33.102): +0.0019 ± 0.0011, 3/4 |
| 4b | **Ukrainian case switched off mid-training**, under endings apart at 20k (+0.0038) or on a small corpus | *A deficit:* in both languages. *None:* the deficit belongs to a strong feature, not to features in general | — | **done** (§33.102): no deficit, −0.0008 ± 0.0006 |
| 5 | **The one-seed probes at more seeds**: after removal, with the auxiliary target, across widths (§3.7, §3.8) | *They agree:* the qualifiers go, and §3.7's share is stated with its spread. *They disagree:* the sentence says how far the seeds differ | ~6–9 h, and probing on the CPU | no: re-runs with checkpoints along training |
| 6 | **English at full training, seeds 4–7** of both tokenizers (§3.3.6; conclusion 1) | *7/8 or more:* at full training, at 7 of 8 seeds. *Fewer:* at full training the English gain is within the seed spread, and conclusion 1 loses its full-training clause | — | **done** (§33.102): 7/8, median +0.0053; BLiMP at seeds 4–7 prepared |
| 7 | **The leak at full training under BPE 17k**: `ucaseprep` at 151M, 50k, four seeds (§3.10; conclusion 7) | *All of the lead (+0.0006 ± 0.0004) is leak:* under both tokenizers. *Some is not:* under BPE 17k part of the lead is context | ~3.7 h | no: its codes at 0.2 under BPE 17k (CPU, under an hour) |
| 8 | **The two-seed rows at 50k**: frequency band and part of speech from the word form, seeds 2–3 (§3.5; conclusion 4) | *Still within 0.002:* the sentence stands on four seeds for every row. *Either exceeds it:* the claim names the exception | ~3.7 h | a piece |

About 50 hours of GPU in all: five or six long chains. The section numbers are the thesis's.

**Not requested by the thesis** (every outcome gives the same sentence, or none): more seeds for
English morpheme segmentation (the 12–13% gap is about nineteen times the widest seed spread); the
shape effect by language; the law across model sizes, until there is a hypothesis for the half the
previous word's tag leaves open.

**Dropped by the sentence test:**
- **Why fiction's share of bits per byte is half Wikipedia's** (§33.88). The thesis states the
  size on each corpus, and no outcome changes a claim.
- **Case and its controls on fiction.** Morphology in the embedding is answered (§33.83).

**Dependencies.** None between the items. 2, 3, 4b, 5 and 7 need preparation first — a design,
or a build on the CPU — which can overlap the item before. Each item's predictions are recorded when it is
launched.
