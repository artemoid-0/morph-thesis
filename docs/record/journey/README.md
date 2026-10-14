# Research journey — index

> **Kind:** index of a record. **Authoritative for:** how each result was
> reached, in order, including the mistakes. For what the thesis claims now,
> read [`docs/FINDINGS.md`](../../FINDINGS.md); for the numbers, the
> [results record](../results.md).

The journey is chronological. Earlier sections record what was believed at the
time, and many beliefs were later corrected; corrections are annotated where
the original claim stands, so the reasoning behind each wrong turn stays
visible. Section numbers are permanent — other documents cite "journey §33.24"
— and Appendix B maps an earlier numbering.

Until 1 October 2026 this was one file, `research_journey.md`; its front matter (a
summary that had gone stale, and a contents list that stopped at §32) was retired then.

**Short of time:** Parts 8 and 9 hold the controlled programme and the thesis's
current results; Part 7 the measurement turn that preceded them; Parts 1–2 why
the tokenizer was built as it was.

| part | title | sections | in one line |
|---|---|---|---|
| Part 1 | [The Question](1_the_question.md) | §1–§4 | What was asked, where to intervene, what is knowable at each scope. Design only. |
| Part 2 | [Building the Morphological Tokenizer](2_building_the_tokenizer.md) | §5–§10 | Vocabulary, named entities, the segmenter and its bug, Morfessor, compounds; the first comparison. |
| Part 3 | [The Apparatus](3_the_apparatus.md) | §11–§13 | The frozen embedding layer, the dataset and model, the fairness metric, calibration. |
| Part 4 | [The Audits](4_the_audits.md) | §14–§19 | Auditing the embedding and the information channels; the information-fair rerun: Q1 falls. |
| Part 5 | [Grounding and Controls](5_grounding_and_controls.md) | §20–§22 | Literature grounding, the missing controls, why English resists. |
| Part 6 | [Level 2: Features on a BPE Model](6_level2_features_on_bpe.md) | §23–§29 | Morphology as features on BPE; held-out evaluation; the word-identity control. |
| Part 7 | [The Causal Reckoning](7_the_causal_reckoning.md) | §30–§32 | The within-word lookahead; "it was the structure"; a fresh look at the records. |
| Part 8 | [The Controlled Programme](8_the_controlled_programme.md) | §33.1–§33.19 | BLiMP and the agreement test; the same-shape null; the additive design; position. |
| Part 9 | [Ukrainian, and Before the Model Has Learned It](9_ukrainian_and_learning.md) | §33.20–§33.96 | The Ukrainian pipeline; Q2 in Ukrainian; early help; little data and the training budget; morphology in the tokenizer. |
| — | [Appendices](appendices.md) | A, B | The decision record; the mapping from an earlier section numbering. |

## Every section

- [1. The Starting Point and the Intervention Point](1_the_question.md#1-the-starting-point-and-the-intervention-point)
- [2. The Hybrid Frozen/Trained Embedding](1_the_question.md#2-the-hybrid-frozentrained-embedding)
- [3. What Can Be Known, and Where](1_the_question.md#3-what-can-be-known-and-where)
- [4. Choosing the Unit, and How to Evaluate It](1_the_question.md#4-choosing-the-unit-and-how-to-evaluate-it)
- [5. The Vocabulary: Why Frequency Is the Wrong Criterion](2_building_the_tokenizer.md#5-the-vocabulary-why-frequency-is-the-wrong-criterion)
- [6. Named Entities: The Threshold Wars and a Structural Answer](2_building_the_tokenizer.md#6-named-entities-the-threshold-wars-and-a-structural-answer)
- [7. The Segmenter, and the Bug That Cost Thousands of Prefixes](2_building_the_tokenizer.md#7-the-segmenter-and-the-bug-that-cost-thousands-of-prefixes)
- [8. Morfessor's Null Result and the Three-Pipeline Decision](2_building_the_tokenizer.md#8-morfessors-null-result-and-the-three-pipeline-decision)
- [9. Compound Splitting and the Particle Fix](2_building_the_tokenizer.md#9-compound-splitting-and-the-particle-fix)
- [10. What the First Comparison Showed — and What It Concealed](2_building_the_tokenizer.md#10-what-the-first-comparison-showed--and-what-it-concealed)
- [11. Designing the Frozen Embedding Layer](3_the_apparatus.md#11-designing-the-frozen-embedding-layer)
- [12. The Dataset, Model, and the Fairness Metric](3_the_apparatus.md#12-the-dataset-model-and-the-fairness-metric)
- [13. Calibration and Literature Grounding](3_the_apparatus.md#13-calibration-and-literature-grounding)
- [14. The Frozen Embedding Audit](4_the_audits.md#14-the-frozen-embedding-audit)
- [15. Parameter Tying vs Information Injection](4_the_audits.md#15-parameter-tying-vs-information-injection)
- [16. Restructuring — Reversing the Flat-Layout Decision](4_the_audits.md#16-restructuring--reversing-the-flat-layout-decision)
- [17. The Information Channel Audit — Making the Comparison Fair](4_the_audits.md#17-the-information-channel-audit--making-the-comparison-fair)
- [18. The Information-Fair Rerun — Q1 Falls, Q2 Holds](4_the_audits.md#18-the-information-fair-rerun--q1-falls-q2-holds)
- [19. Localising the Q1 Deficit — Coverage, Decomposition, and the Unified Finding](4_the_audits.md#19-localising-the-q1-deficit--coverage-decomposition-and-the-unified-finding)
- [20. Literature Grounding — Who Has Done This Already](5_grounding_and_controls.md#20-literature-grounding--who-has-done-this-already)
- [21. Completing the Controls — Matched Vocabulary, Frozen vs Learned, and Signal Scale](5_grounding_and_controls.md#21-completing-the-controls--matched-vocabulary-frozen-vs-learned-and-signal-scale)
- [22. Why English Resists — and the Turn from Confrontation to Cooperation](5_grounding_and_controls.md#22-why-english-resists--and-the-turn-from-confrontation-to-cooperation)
- [23. Level 2 — Morphology as Features on a BPE-Tokenized Model](6_level2_features_on_bpe.md#23-level-2--morphology-as-features-on-a-bpe-tokenized-model)
- [24. Where the Project Stands](6_level2_features_on_bpe.md#24-where-the-project-stands)
- [25. The Legibility Probe — What Injection Actually Provides](6_level2_features_on_bpe.md#25-the-legibility-probe--what-injection-actually-provides)
- [26. Closing the Reviewer Gaps — Held-Out Evaluation and the Four Missing Cells](6_level2_features_on_bpe.md#26-closing-the-reviewer-gaps--held-out-evaluation-and-the-four-missing-cells)
- [27. The Word-Identity Control — What the Effect Actually Was](6_level2_features_on_bpe.md#27-the-word-identity-control--what-the-effect-actually-was)
- [28. Turning the Instrument on Ourselves](6_level2_features_on_bpe.md#28-turning-the-instrument-on-ourselves)
- [29. Lemma Grouping, and a Ruler for Word Identity](6_level2_features_on_bpe.md#29-lemma-grouping-and-a-ruler-for-word-identity)
- [30. The Lookahead in Word-Form Features](7_the_causal_reckoning.md#30-the-lookahead-in-word-form-features)
- [31. It Was the Structure All Along](7_the_causal_reckoning.md#31-it-was-the-structure-all-along)
- [32. A Fresh Look: What the Records Already Knew](7_the_causal_reckoning.md#32-a-fresh-look-what-the-records-already-knew)
- [33. The Agreement Test](8_the_controlled_programme.md#33-the-agreement-test)
  - [33.1 BLiMP at one seed](8_the_controlled_programme.md#331-blimp-at-one-seed)
  - [33.2 The inflection feature was blind to agreement](8_the_controlled_programme.md#332-the-inflection-feature-was-blind-to-agreement)
  - [33.3 A tagger feature is not causal, even shifted](8_the_controlled_programme.md#333-a-tagger-feature-is-not-causal-even-shifted)
  - [33.4 Predictions, recorded before the night of 26 September](8_the_controlled_programme.md#334-predictions-recorded-before-the-night-of-26-september)
  - [33.5 Tools this required](8_the_controlled_programme.md#335-tools-this-required)
  - [33.6 What the night found](8_the_controlled_programme.md#336-what-the-night-found)
  - [33.7 Next, with predictions recorded first](8_the_controlled_programme.md#337-next-with-predictions-recorded-first)
  - [33.8 The anaphor gain was a preference, and what that changes](8_the_controlled_programme.md#338-the-anaphor-gain-was-a-preference-and-what-that-changes)
  - [33.9 The shuffled control matches structure](8_the_controlled_programme.md#339-the-shuffled-control-matches-structure)
  - [33.10 The null helps as much as structure](8_the_controlled_programme.md#3310-the-null-helps-as-much-as-structure)
  - [33.11 The headline is the slot](8_the_controlled_programme.md#3311-the-headline-is-the-slot)
  - [33.12 A narrower token embedding is the whole effect](8_the_controlled_programme.md#3312-a-narrower-token-embedding-is-the-whole-effect)
  - [33.13 Where the fresh look's theories stand](8_the_controlled_programme.md#3313-where-the-fresh-looks-theories-stand)
  - [33.14 The additive null gains too](8_the_controlled_programme.md#3314-the-additive-null-gains-too)
  - [33.15 Families that share an initialization](8_the_controlled_programme.md#3315-families-that-share-an-initialization)
  - [33.16 The architecture effect, measured against eight starts](8_the_controlled_programme.md#3316-the-architecture-effect-measured-against-eight-starts)
  - [33.17 The literature, audited — and the case for Ukrainian](8_the_controlled_programme.md#3317-the-literature-audited--and-the-case-for-ukrainian)
  - [33.18 The Ukrainian path, built](8_the_controlled_programme.md#3318-the-ukrainian-path-built)
  - [33.19 Two routes to the same gain](8_the_controlled_programme.md#3319-two-routes-to-the-same-gain)
- [33. The Agreement Test — continued](9_ukrainian_and_learning.md#33-the-agreement-test--continued)
  - [33.20 What we know, and Ukrainian begins](9_ukrainian_and_learning.md#3320-what-we-know-and-ukrainian-begins)
  - [33.21 A gate that held: words glued by split punctuation](9_ukrainian_and_learning.md#3321-a-gate-that-held-words-glued-by-split-punctuation)
  - [33.22 The real cause: paragraphs glued together](9_ukrainian_and_learning.md#3322-the-real-cause-paragraphs-glued-together)
  - [33.23 Two tracks in one night](9_ukrainian_and_learning.md#3323-two-tracks-in-one-night)
  - [33.24 The first Ukrainian family: early help, late cost](9_ukrainian_and_learning.md#3324-the-first-ukrainian-family-early-help-late-cost)
  - [33.25 A day for G-65: does case cost the position channel?](9_ukrainian_and_learning.md#3325-a-day-for-g-65-does-case-cost-the-position-channel)
  - [33.26 D2: with a clean channel beside it, case gains](9_ukrainian_and_learning.md#3326-d2-with-a-clean-channel-beside-it-case-gains)
  - [33.27 D3: seed 0 was the favourable draw](9_ukrainian_and_learning.md#3327-d3-seed-0-was-the-favourable-draw)
  - [33.28 D4, and the one effect every informative feature shares](9_ukrainian_and_learning.md#3328-d4-and-the-one-effect-every-informative-feature-shares)
  - [33.29 Toward the demo: little data](9_ukrainian_and_learning.md#3329-toward-the-demo-little-data)
  - [33.30 Ukrainian at four seeds, and a chain that ran twice](9_ukrainian_and_learning.md#3330-ukrainian-at-four-seeds-and-a-chain-that-ran-twice)
  - [33.31 With little data, case helps](9_ukrainian_and_learning.md#3331-with-little-data-case-helps)
  - [33.32 Not scarce data — an unfinished lesson](9_ukrainian_and_learning.md#3332-not-scarce-data--an-unfinished-lesson)
  - [33.33 After the demo: English at a short budget, and a switch](9_ukrainian_and_learning.md#3333-after-the-demo-english-at-a-short-budget-and-a-switch)
  - [33.34 A crutch, not a teacher](9_ukrainian_and_learning.md#3334-a-crutch-not-a-teacher)
  - [33.35 A programme after the demo](9_ukrainian_and_learning.md#3335-a-programme-after-the-demo)
  - [33.36 The first programme night](9_ukrainian_and_learning.md#3336-the-first-programme-night)
  - [33.37 The follow-on of 2 October](9_ukrainian_and_learning.md#3337-the-follow-on-of-2-october)
  - [33.38 The law holds; a critical period](9_ukrainian_and_learning.md#3338-the-law-holds-a-critical-period)
  - [33.39 Day block D: the confound, and a wider model](9_ukrainian_and_learning.md#3339-day-block-d-the-confound-and-a-wider-model)
  - [33.40 Two pivots](9_ukrainian_and_learning.md#3340-two-pivots)
  - [33.41 The chain widened: four directions in one night](9_ukrainian_and_learning.md#3341-the-chain-widened-four-directions-in-one-night)
  - [33.42 The night of four directions](9_ukrainian_and_learning.md#3342-the-night-of-four-directions)
  - [33.43 A bit-for-bit reproduction](9_ukrainian_and_learning.md#3343-a-bit-for-bit-reproduction)
  - [33.44 A rule written for one generation of runs](9_ukrainian_and_learning.md#3344-a-rule-written-for-one-generation-of-runs)
  - [33.45 Tests the suite never ran](9_ukrainian_and_learning.md#3345-tests-the-suite-never-ran)
  - [33.46 The wide model's crossing, and the slot as consistency](9_ukrainian_and_learning.md#3346-the-wide-models-crossing-and-the-slot-as-consistency)
  - [33.47 A screen before the long chain](9_ukrainian_and_learning.md#3347-a-screen-before-the-long-chain)
  - [33.48 Building D1 and D2 while the screen runs](9_ukrainian_and_learning.md#3348-building-d1-and-d2-while-the-screen-runs)
  - [33.49 The screen's verdicts, and time budgets](9_ukrainian_and_learning.md#3349-the-screens-verdicts-and-time-budgets)
  - [33.50 The teacher fades faster than the crutch](9_ukrainian_and_learning.md#3350-the-teacher-fades-faster-than-the-crutch)
  - [33.51 An empty slot under rotary positions, and resumable chains](9_ukrainian_and_learning.md#3351-an-empty-slot-under-rotary-positions-and-resumable-chains)
  - [33.52 What the input needs, and what the probes found](9_ukrainian_and_learning.md#3352-what-the-input-needs-and-what-the-probes-found)
  - [33.53 Tools for other conditions](9_ukrainian_and_learning.md#3353-tools-for-other-conditions)
  - [33.54 Tools for other models](9_ukrainian_and_learning.md#3354-tools-for-other-models)
  - [33.55 What the tokenizers expose, and codes read across corpus sizes](9_ukrainian_and_learning.md#3355-what-the-tokenizers-expose-and-codes-read-across-corpus-sizes)
  - [33.56 What the nulls read, and a null that was not constant](9_ukrainian_and_learning.md#3356-what-the-nulls-read-and-a-null-that-was-not-constant)
  - [33.57 The constants replayed, and the night of 6 October](9_ukrainian_and_learning.md#3357-the-constants-replayed-and-the-night-of-6-october)
  - [33.58 The night of 6 October: the vector, the exact null, five tokenizers](9_ukrainian_and_learning.md#3358-the-night-of-6-october-the-vector-the-exact-null-five-tokenizers)
  - [33.59 What the feature carries, probed — and the thesis kept in scope](9_ukrainian_and_learning.md#3359-what-the-feature-carries-probed--and-the-thesis-kept-in-scope)
  - [33.60 The previous word's tag: half the width gap, and what the switch leaves](9_ukrainian_and_learning.md#3360-the-previous-words-tag-half-the-width-gap-and-what-the-switch-leaves)
  - [33.61 Morphology in the tokenizer: the ending rule, checked, and BPE that keeps endings apart](9_ukrainian_and_learning.md#3361-morphology-in-the-tokenizer-the-ending-rule-checked-and-bpe-that-keeps-endings-apart)
  - [33.62 The sixth tokenizer, measured — and nothing outside the project](9_ukrainian_and_learning.md#3362-the-sixth-tokenizer-measured--and-nothing-outside-the-project)
  - [33.63 The sixth condition, ready — and what its numbers showed about the tools](9_ukrainian_and_learning.md#3363-the-sixth-condition-ready--and-what-its-numbers-showed-about-the-tools)
  - [33.64 A long chain for 7 October: the sixth tokenizer at four seeds, and D7 at four](9_ukrainian_and_learning.md#3364-a-long-chain-for-7-october-the-sixth-tokenizer-at-four-seeds-and-d7-at-four)
  - [33.65 Morphology in the tokenizer helps — and case helps more where endings are exposed](9_ukrainian_and_learning.md#3365-morphology-in-the-tokenizer-helps--and-case-helps-more-where-endings-are-exposed)
  - [33.66 The same text under every tokenizer — and the work moves to one machine](9_ukrainian_and_learning.md#3366-the-same-text-under-every-tokenizer--and-the-work-moves-to-one-machine)
  - [33.67 A long chain for 8 October: does morphology in the tokenizer last?](9_ukrainian_and_learning.md#3367-a-long-chain-for-8-october-does-morphology-in-the-tokenizer-last)
  - [33.68 Morphology in the tokenizer lasts with training, and is larger in a wider model](9_ukrainian_and_learning.md#3368-morphology-in-the-tokenizer-lasts-with-training-and-is-larger-in-a-wider-model)
  - [33.69 Why exposure helps case: the loss by what each token is](9_ukrainian_and_learning.md#3369-why-exposure-helps-case-the-loss-by-what-each-token-is)
  - [33.70 Case helps by predicting endings — and the exposure effect lies elsewhere](9_ukrainian_and_learning.md#3370-case-helps-by-predicting-endings--and-the-exposure-effect-lies-elsewhere)
  - [33.71 The manuscript begins](9_ukrainian_and_learning.md#3371-the-manuscript-begins)
  - [33.72 paper/ becomes exports/, kept current](9_ukrainian_and_learning.md#3372-paper-becomes-exports-kept-current)
  - [33.73 The queue reviewed: what each candidate settles for the thesis](9_ukrainian_and_learning.md#3373-the-queue-reviewed-what-each-candidate-settles-for-the-thesis)
  - [33.74 What the sentence's case adds: prepositions, and words the form never saw](9_ukrainian_and_learning.md#3374-what-the-sentences-case-adds-prepositions-and-words-the-form-never-saw)
  - [33.75 Ten hours, and two decisions](9_ukrainian_and_learning.md#3375-ten-hours-and-two-decisions)
  - [33.76 Morphology in the tokenizer lasts at full training — and case on top of it, in part through the leak](9_ukrainian_and_learning.md#3376-morphology-in-the-tokenizer-lasts-at-full-training--and-case-on-top-of-it-in-part-through-the-leak)
  - [33.77 The queue reviewed after the chain of 9 October](9_ukrainian_and_learning.md#3377-the-queue-reviewed-after-the-chain-of-9-october)
  - [33.78 Ten hours, 10 October](9_ukrainian_and_learning.md#3378-ten-hours-10-october)
  - [33.79 A grammatical test in Ukrainian: MultiBLiMP](9_ukrainian_and_learning.md#3379-a-grammatical-test-in-ukrainian-multiblimp)
  - [33.80 Figures for the thesis, in Ukrainian](9_ukrainian_and_learning.md#3380-figures-for-the-thesis-in-ukrainian)
  - [33.81 Case at full training without the leak: the word's own case lasts, and the sentence's lead was the leak](9_ukrainian_and_learning.md#3381-case-at-full-training-without-the-leak-the-words-own-case-lasts-and-the-sentences-lead-was-the-leak)
  - [33.82 One more hour, 10 October](9_ukrainian_and_learning.md#3382-one-more-hour-10-october)
  - [33.83 Is a direction already dead? The thesis reviewed, 11 October](9_ukrainian_and_learning.md#3383-is-a-direction-already-dead-the-thesis-reviewed-11-october)
  - [33.84 Context-aware morphology inside the models](9_ukrainian_and_learning.md#3384-context-aware-morphology-inside-the-models)
  - [33.85 How far the tokenizer's gain travels: English, and Ukrainian fiction (prepared)](9_ukrainian_and_learning.md#3385-how-far-the-tokenizers-gain-travels-english-and-ukrainian-fiction-prepared)
  - [33.86 The wider model's lead settles: +0.022 at width 256 and 20k](9_ukrainian_and_learning.md#3386-the-wider-models-lead-settles-0022-at-width-256-and-20k)
  - [33.87 English with endings apart: it helps, a fifth as much](9_ukrainian_and_learning.md#3387-english-with-endings-apart-it-helps-a-fifth-as-much)
  - [33.88 A second Ukrainian corpus: fiction keeps the gain, at half the size in bits per byte and in full in grammar](9_ukrainian_and_learning.md#3388-a-second-ukrainian-corpus-fiction-keeps-the-gain-at-half-the-size-in-bits-per-byte-and-in-full-in-grammar)
  - [33.89 Ten hours in batches, 11 October: what ran, and the queue after](9_ukrainian_and_learning.md#3389-ten-hours-in-batches-11-october-what-ran-and-the-queue-after)
  - [33.90 English at full training (prepared), 12 October](9_ukrainian_and_learning.md#3390-english-at-full-training-prepared-12-october)
  - [33.91 FINDINGS checked against its record, 12 October](9_ukrainian_and_learning.md#3391-findings-checked-against-its-record-12-october)
  - [33.92 English at full training: the gain holds on the median, at three seeds of four](9_ukrainian_and_learning.md#3392-english-at-full-training-the-gain-holds-on-the-median-at-three-seeds-of-four)
  - [33.93 Does the tokenizer's gain keep growing with model size? Width 512 (prepared), 14 October](9_ukrainian_and_learning.md#3393-does-the-tokenizers-gain-keep-growing-with-model-size-width-512-prepared-14-october)
  - [33.94 Width 512: the gain in bits per byte peaks at width 256; the grammatical gain keeps growing](9_ukrainian_and_learning.md#3394-width-512-the-gain-in-bits-per-byte-peaks-at-width-256-the-grammatical-gain-keeps-growing)
  - [33.95 Does the width-512 lead settle? Seed 0 at 20k (prepared), 14 October](9_ukrainian_and_learning.md#3395-does-the-width-512-lead-settle-seed-0-at-20k-prepared-14-october)
  - [33.96 Width 512 at 20k, seed 0: the lead levels off near +0.011](9_ukrainian_and_learning.md#3396-width-512-at-20k-seed-0-the-lead-levels-off-near-0011)
- [Appendix A — Decision Record](appendices.md#appendix-a--decision-record)
- [Appendix B — Section Mapping](appendices.md#appendix-b--section-mapping)
