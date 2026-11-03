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
On 19 October 2026 the former Part 9, which had grown to cover §33.20–§33.118, was divided
at its seams into Parts 9–15; each section moved whole, with its number and heading.

Each part is one stretch of the work with one theme, and opens with what it established.
The index [by question](#by-question) crosses the parts, and [every section](#every-section)
is listed after it. New sections are added at the end of the last part.

**Short of time:** [`docs/FINDINGS.md`](../../FINDINGS.md) states the claims. Parts 11–15
hold the results they rest on now: morphology in the tokenizer (11); full training, grammar
and the leak (12); how far the gain travels (13); the two machines and the null (14–15).
Part 10 holds the learning law; Parts 7–8 the measurement turn and the controlled programme
before it; Parts 1–2 why the tokenizer was built as it was.

## The parts

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
| Part 9 | [Ukrainian, and Before the Model Has Learned It](9_ukrainian_and_learning.md) | §33.20–§33.32 | The Ukrainian pipeline; Q2 in Ukrainian: nothing at the end of training, help before the model has learned case; little data and short budgets. |
| Part 10 | [The Learning Law: A Crutch, Not a Teacher](10_the_learning_law.md) | §33.33–§33.60 | The learning law, within one model size; the switch: a crutch, not a teacher; the slot as a shared vector; the null made constant. |
| Part 11 | [Morphology in the Tokenizer Helps, and Lasts](11_morphology_in_the_tokenizer.md) | §33.61–§33.70 | BPE that keeps endings apart beats plain BPE, lasts with training and gains more in a wider model; case helps by predicting endings. |
| Part 12 | [Full Training, Grammar, and the Leak](12_full_training_and_the_leak.md) | §33.71–§33.82 | The tokenizer's gain at full training and in grammar (MultiBLiMP); what the sentence adds over the word form is the tagger's leak. |
| Part 13 | [How Far the Gain Travels](13_how_far_the_gain_travels.md) | §33.83–§33.96 | The sentence test; morphology inside the models; the gain in English, on fiction, at widths 256 and 512. |
| Part 14 | [Two Machines, and a Null That Is Not Neutral](14_two_machines.md) | §33.97–§33.116 | The second machine and its twins; a same-shape null that is not neutral (G-78); English's leak; width 512 and fiction at full training. |
| Part 15 | [The List of 19 October: Case and the Null at 30M Parameters](15_the_list_of_19_october.md) | §33.117 onward | Open. BLiMP's agreement at eight seeds; exposure on the law's axis; at 30M parameters the null costs nothing; at 85M the tokenizer's gain is smaller again and lasts, split by seed; a removed feature's deficit closes by 100k; gender's and number's leak is small. |
| — | [Appendices](appendices.md) | A, B, C | The decision record; the mapping from an earlier section numbering; the thesis's list of 15 October with its outcome sentences. |

## By question

Where each question of the thesis was worked on, across the parts. A section may appear under more than one question.

### Q1 in English: morphemes instead of BPE

- [§5–§10](2_building_the_tokenizer.md#5-the-vocabulary-why-frequency-is-the-wrong-criterion) — the morpheme tokenizer built, and the first comparison's apparent 6.31% lead over BPE
- [§12](3_the_apparatus.md#12-the-dataset-model-and-the-fairness-metric) — bits per byte, the metric that compares tokenizers
- [§17–§19](4_the_audits.md#17-the-information-channel-audit--making-the-comparison-fair) — the information-channel audit; made fair, morphemes lose to BPE; the deficit localised
- [§20](5_grounding_and_controls.md#20-literature-grounding--who-has-done-this-already), [§21.1](5_grounding_and_controls.md#211-matched-vocabulary--the-q1-confound-finally-removed), [§22](5_grounding_and_controls.md#22-why-english-resists--and-the-turn-from-confrontation-to-cooperation) — the negative is expected in the literature; with vocabularies matched, BPE ahead by 13.5%; why English resists
- [§28](6_level2_features_on_bpe.md#28-turning-the-instrument-on-ourselves) — the reversal re-checked under full control
- [§30.6](7_the_causal_reckoning.md#306-where-the-tokenizer-route-stands), [§32.3](7_the_causal_reckoning.md#323-morpheme-tokenization-learns-slower) — routing between the tokenizers, estimated and not built; morpheme tokenization learns more slowly
- [§33.85](13_how_far_the_gain_travels.md#3385-how-far-the-tokenizers-gain-travels-english-and-ukrainian-fiction-prepared), [§33.87](13_how_far_the_gain_travels.md#3387-english-with-endings-apart-it-helps-a-fifth-as-much), [§33.90](13_how_far_the_gain_travels.md#3390-english-at-full-training-prepared-12-october), [§33.92](13_how_far_the_gain_travels.md#3392-english-at-full-training-the-gain-holds-on-the-median-at-three-seeds-of-four) — English BPE that keeps endings apart: a fifth of the Ukrainian gain, lasting to full training on the median seed
- [§33.101–§33.102](14_two_machines.md#33101-a-second-machine-and-the-night-of-16-october-prepared), [§33.107](14_two_machines.md#33107-in-english-too-the-sentences-lead-over-the-word-form-is-the-taggers-leak-full-trainings-twins-drift-apart), [§33.117](15_the_list_of_19_october.md#33117-the-texts-list-of-19-october-blimps-agreement-at-eight-seeds-the-exposure-comparison-on-the-laws-axis-and-item-4-prepared) — eight seeds at full training; twins that drift; BLiMP's agreement at eight seeds

### Q1 in Ukrainian: morphology in the tokenizer

- [§33.53](10_the_learning_law.md#3353-tools-for-other-conditions), [§33.55](10_the_learning_law.md#3355-what-the-tokenizers-expose-and-codes-read-across-corpus-sizes), [§33.58](10_the_learning_law.md#3358-the-night-of-6-october-the-vector-the-exact-null-five-tokenizers) — tokenizers as conditions; how many endings each exposes; case under five tokenizers
- [§33.59–§33.60](10_the_learning_law.md#3359-what-the-feature-carries-probed--and-the-thesis-kept-in-scope), [§33.61–§33.62](11_morphology_in_the_tokenizer.md#3361-morphology-in-the-tokenizer-the-ending-rule-checked-and-bpe-that-keeps-endings-apart) — morphology in the tokenizer proposed; the ending rule, checked against two dictionaries; the sixth tokenizer measured
- [§33.63–§33.68](11_morphology_in_the_tokenizer.md#3363-the-sixth-condition-ready--and-what-its-numbers-showed-about-the-tools) — endings kept apart beat BPE 17k on the same text, last at twice the budget, and gain more at width 256
- [§33.73](12_full_training_and_the_leak.md#3373-the-queue-reviewed-what-each-candidate-settles-for-the-thesis), [§33.76](12_full_training_and_the_leak.md#3376-morphology-in-the-tokenizer-lasts-at-full-training--and-case-on-top-of-it-in-part-through-the-leak) — at full training, on four times the text
- [§33.79–§33.80](12_full_training_and_the_leak.md#3379-a-grammatical-test-in-ukrainian-multiblimp) — in grammar: MultiBLiMP's subject–verb agreement
- [§33.84](13_how_far_the_gain_travels.md#3384-context-aware-morphology-inside-the-models) — what the models build for themselves
- [§33.86](13_how_far_the_gain_travels.md#3386-the-wider-models-lead-settles-0022-at-width-256-and-20k), [§33.93–§33.96](13_how_far_the_gain_travels.md#3393-does-the-tokenizers-gain-keep-growing-with-model-size-width-512-prepared-14-october), [§33.97](14_two_machines.md#3397-width-512-at-20k-seed-1-prepared-15-october), [§33.100](14_two_machines.md#33100-width-512-at-20k-seed-1-levelled-off-too-at-00097), [§33.102](14_two_machines.md#33102-the-night-of-16-october-width-512-settles-a-removed-features-deficit-halves-by-50k-and-case-leaves-none) — width 256 settles; width 512 gains less and levels off
- [§33.85](13_how_far_the_gain_travels.md#3385-how-far-the-tokenizers-gain-travels-english-and-ukrainian-fiction-prepared), [§33.88](13_how_far_the_gain_travels.md#3388-a-second-ukrainian-corpus-fiction-keeps-the-gain-at-half-the-size-in-bits-per-byte-and-in-full-in-grammar), [§33.101–§33.102](14_two_machines.md#33101-a-second-machine-and-the-night-of-16-october-prepared), [§33.108–§33.109](14_two_machines.md#33108-fiction-at-full-training-the-tokenizers-gain-holds-and-reaches-wikipedias-share) — a second corpus, fiction, at 10k and at full training
- [§33.118](15_the_list_of_19_october.md#33118-item-4-at-30m-parameters-case-gains-alike-under-both-tokenizers-and-the-null-costs-nothing) — over the empty slots the 5M model's figure is conservative
- [§33.119](15_the_list_of_19_october.md#33119-item-1-prepared-the-tokenizers-gain-at-width-1024), [§33.120](15_the_list_of_19_october.md#33120-item-1s-10k-half-at-85m-parameters-the-gain-halves-on-average-because-two-seeds-of-four-lose-it-early) — beyond 30M parameters: at 85M the gain halves at 10k, split by seed
- [§33.121](15_the_list_of_19_october.md#33121-multiblimp-at-width-1024-and-a-finer-look-at-g-81-prepared-20-october), [§33.122](15_the_list_of_19_october.md#33122-multiblimp-at-width-1024-the-gain-holds-at-every-seed-and-stops-growing-g-81s-fall-is-no-spike) — at 85M parameters the grammatical gain holds at every seed and stops growing; the fall is no spike
- [§33.124](15_the_list_of_19_october.md#33124-item-1s-20k-half-at-85m-parameters-the-gain-is-smaller-again-and-lasts-and-the-split-stays) — at 85M parameters the gain is smaller again and lasts; the split stays at 20k
- [§33.130](15_the_list_of_19_october.md#33130-the-texts-request-of-22-october-and-g-81s-recipe-question-prepared) — G-81's recipe question prepared: a 1,000-step warm-up at widths 1024 and 512
- [§33.132](15_the_list_of_19_october.md#33132-item-1-with-a-longer-warm-up-the-gain-at-width-1024-holds-at-every-seed) — with a longer warm-up the gain at width 1024 holds at every seed, smaller than at width 512
- [§33.136](15_the_list_of_19_october.md#33136-the-texts-request-of-24-october-and-the-empty-slot-at-width-1024-prepared) — the text's request of 24 October; the empty slot at width 1024 prepared
- [§33.137](15_the_list_of_19_october.md#33137-item-1-prepared-width-1024-without-repeating-the-text) — width 1024 at 20k without repeating the text, prepared
- [§33.139](15_the_list_of_19_october.md#33139-item-2-at-width-1024-the-tokenizers-gain-over-the-empty-slots-is-smaller-than-between-the-nulls) — at width 1024 the null costs BPE 17k and helps endings apart: the tokenizer's gain over the empty slots is +0.0079, 0.003 below the nulls'
- [§33.140](15_the_list_of_19_october.md#33140-the-queue-reviewed-full-training-at-30m-parameters-and-english-at-30m-prepared) — the queue reviewed: full training at 30M parameters, and English at 30M, prepared
- [§33.143](15_the_list_of_19_october.md#33143-the-texts-list-of-limits-of-30-october-the-learning-rate-the-plain-baselines-the-warm-up-at-width-128-and-width-1024-at-full-training-prepared) — the text's list of limits of 30 October: the learning rate, the plain baselines, the warm-up at width 128 and width 1024 at full training, prepared
- [§33.144](15_the_list_of_19_october.md#33144-the-first-tier-on-the-thesis-machine-prepared-shuffled-case-the-switch-to-a-constant-and-where-the-cut-falls) — the first tier on the thesis machine prepared: shuffled case, the switch to a constant, where the cut falls
- [§33.145](15_the_list_of_19_october.md#33145-the-learning-rate-the-plain-baselines-and-the-warm-up-at-width-128-at-the-best-of-three-rates-the-gain-is-smaller-without-the-slot-it-is-larger) — at the best of three learning rates the 5M model's gain is smaller; between plain baselines it is larger; the warm-up changes nothing at width 128
- [§33.146](15_the_list_of_19_october.md#33146-items-7-and-13-prepared-the-modern-block-and-depth-after-width-1024) — the modern block and depth prepared
- [§33.147](15_the_list_of_19_october.md#33147-item-6-prepared-minimal-pairs-for-agreement-in-the-noun-phrase-and-for-government) — minimal pairs for agreement in the noun phrase and for government, prepared
- [§33.148](15_the_list_of_19_october.md#33148-items-9-10-and-11-the-morpheme-boundary-matters-cases-content-acts-and-the-deficit-is-not-the-inputs-change) — the morpheme boundary matters; case's content acts; the deficit is not the input's change
- [§33.149](15_the_list_of_19_october.md#33149-items-2-7-and-13-at-85m-parameters-the-gain-at-full-training-is-the-30m-models-it-holds-under-a-modern-block-and-falls-with-depth) — at 85M parameters the gain at full training is the 30M model's; it holds under a modern block and falls with depth
- [§33.150](15_the_list_of_19_october.md#33150-item-6-the-tokenizers-grammatical-gain-covers-agreement-in-the-noun-phrase-and-government) — the tokenizer's grammatical gain covers agreement in the noun phrase and government
- [§33.151](15_the_list_of_19_october.md#33151-item-5s-intervals-and-the-softwares-size-the-tokenizers-gains-hold-at-95-the-features-bound-is-0003) — item 5's intervals and the software's size: the tokenizer's gains hold at 95%, the features' bound is 0.003
- [§33.152](15_the_list_of_19_october.md#33152-the-test-split-and-every-vocabulary-size-on-held-out-articles-the-gain-is-smaller-a-larger-vocabulary-gains-more) — the test split and every vocabulary size: on held-out articles the gain is smaller; a larger vocabulary gains more
- [§33.153](15_the_list_of_19_october.md#33153-item-8-on-gold-labels-the-tagger-reads-case-where-context-decides) — on gold labels the tagger reads case where context decides
- [§33.141](15_the_list_of_19_october.md#33141-item-1-without-repeating-the-text-the-gain-at-width-1024-is-smaller-and-still-rising) — without repeating the text the gain at width 1024 is smaller, and still rising
- [§33.142](15_the_list_of_19_october.md#33142-full-training-at-30m-parameters-the-tokenizers-gain-is-smaller-and-lasts-case-stays-within-the-bound-english-gains-nothing) — full training at 30M parameters: the tokenizer's gain is smaller and lasts, case stays within the bound, English gains nothing

### Q2: injected features against a same-shape null

- [§11](3_the_apparatus.md#11-designing-the-frozen-embedding-layer), [§14–§15](4_the_audits.md#14-the-frozen-embedding-audit) — the frozen embedding designed, found nearly empty, and turned to context facts
- [§18](4_the_audits.md#18-the-information-fair-rerun--q1-falls-q2-holds), [§21.2–§21.4](5_grounding_and_controls.md#212-frozen-vs-learned-factors--the-thesis-claim-tested-directly) — frozen dependency role helps, as an oracle; a learned table beats a frozen one
- [§23–§29](6_level2_features_on_bpe.md#23-level-2--morphology-as-features-on-a-bpe-tokenized-model) — features on a BPE model, later found mostly lookahead; the word-identity control
- [§30–§31](7_the_causal_reckoning.md#30-the-lookahead-in-word-form-features) — the within-word lookahead; causally, structure rather than identity
- [§33.1–§33.11](8_the_controlled_programme.md#331-blimp-at-one-seed) — BLiMP; tagger features read the right context; against the null, no feature adds information
- [§33.13–§33.16](8_the_controlled_programme.md#3313-where-the-fresh-looks-theories-stand), [§33.19](8_the_controlled_programme.md#3319-two-routes-to-the-same-gain) — the earlier theories re-read; features added instead of slotted; initialization families
- [§33.20](9_ukrainian_and_learning.md#3320-what-we-know-and-ukrainian-begins), [§33.24–§33.32](9_ukrainian_and_learning.md#3324-the-first-ukrainian-family-early-help-late-cost) — Ukrainian case: nothing at the end of training, help before the model has learned it
- [§33.56](10_the_learning_law.md#3356-what-the-nulls-read-and-a-null-that-was-not-constant) — the Ukrainian gains restated against a constant null
- [§33.65](11_morphology_in_the_tokenizer.md#3365-morphology-in-the-tokenizer-helps--and-case-helps-more-where-endings-are-exposed), [§33.69–§33.70](11_morphology_in_the_tokenizer.md#3369-why-exposure-helps-case-the-loss-by-what-each-token-is) — case on top of six tokenizers; case helps by predicting endings
- [§33.73–§33.74](12_full_training_and_the_leak.md#3373-the-queue-reviewed-what-each-candidate-settles-for-the-thesis), [§33.76–§33.77](12_full_training_and_the_leak.md#3376-morphology-in-the-tokenizer-lasts-at-full-training--and-case-on-top-of-it-in-part-through-the-leak), [§33.81](12_full_training_and_the_leak.md#3381-case-at-full-training-without-the-leak-the-words-own-case-lasts-and-the-sentences-lead-was-the-leak) — context or the tagger's leak (G-38, G-77)
- [§33.99](14_two_machines.md#3399-a-causal-tagger-for-english-part-of-speech-the-lists-item-2-prepared), [§33.104](14_two_machines.md#33104-a-day-on-both-machines-the-comparison-g-78-on-each-and-item-2-prepared-17-october), [§33.107](14_two_machines.md#33107-in-english-too-the-sentences-lead-over-the-word-form-is-the-taggers-leak-full-trainings-twins-drift-apart), [§33.114](14_two_machines.md#33114-the-width-probes-at-five-seeds-items-7-and-8-and-131-twins) — English part of speech from a causal tagger; Ukrainian case without the leak under BPE 17k
- [§33.110](14_two_machines.md#33110-g-78-mapped-the-nulls-handicap-is-endings-aparts-at-every-budget--and-it-confounds-exposure-makes-case-more-useful), [§33.113](14_two_machines.md#33113-g-78-across-tokenizers-and-corpus-sizes-the-slot-costs-where-endings-are-exposed-helps-under-bpe--and-exposure-makes-case-partly-redundant), [§33.115–§33.116](14_two_machines.md#33115-two-follow-ups-the-exposure-comparison-at-20k-and-where-the-nulls-help-sits-prepared-18-october), [§33.117–§33.118](15_the_list_of_19_october.md#33117-the-texts-list-of-19-october-blimps-agreement-at-eight-seeds-the-exposure-comparison-on-the-laws-axis-and-item-4-prepared) — exposure: case against the best model without it
- [§33.127](15_the_list_of_19_october.md#33127-item-2-genders-and-numbers-lead-over-the-word-form-is-small-and-little-of-it-is-the-leak) — gender and number: the lead over the word form is small, the leak a small part of it
- [§33.129](15_the_list_of_19_october.md#33129-g-82-what-the-causal-tagger-adds-over-the-word-form-is-not-context) — and what the causal tagger adds over the word form is not context
- [§33.131](15_the_list_of_19_october.md#33131-item-3-prepared-gender-and-number-at-full-training) — gender and number at full training prepared: the stream moved, the codes derived
- [§33.133](15_the_list_of_19_october.md#33133-item-3-at-full-training-genders-and-numbers-lead-over-the-word-form-is-small-and-not-context) — at full training gender's and number's lead over the word form is small, and not context
- [§33.134](15_the_list_of_19_october.md#33134-item-2-prepared-english-part-of-speech-at-full-training) — English part of speech at full training prepared
- [§33.135](15_the_list_of_19_october.md#33135-item-2-at-full-training-english-part-of-speechs-causal-tagger-keeps-nothing-reliable-over-the-word-form) — at full training English part of speech's causal tagger keeps nothing reliable over the word form

### The learning law: a feature helps until the model has learned it

- [§33.24](9_ukrainian_and_learning.md#3324-the-first-ukrainian-family-early-help-late-cost), [§33.28](9_ukrainian_and_learning.md#3328-d4-and-the-one-effect-every-informative-feature-shares) — early help and late cost, in every informative feature
- [§33.29–§33.32](9_ukrainian_and_learning.md#3329-toward-the-demo-little-data), [§33.33](10_the_learning_law.md#3333-after-the-demo-english-at-a-short-budget-and-a-switch) — little data and short budgets; what the early help is worth
- [§33.34–§33.38](10_the_learning_law.md#3334-a-crutch-not-a-teacher) — a crutch, not a teacher; one curve against the null's bits per byte
- [§33.39–§33.42](10_the_learning_law.md#3339-day-block-d-the-confound-and-a-wider-model) — late features integrated slowly, removed ones leave a deficit; no single curve across widths
- [§33.46–§33.52](10_the_learning_law.md#3346-the-wide-models-crossing-and-the-slot-as-consistency) — the wide model's crossing; the rare tail; the teacher fades faster; the law per feature
- [§33.59–§33.60](10_the_learning_law.md#3359-what-the-feature-carries-probed--and-the-thesis-kept-in-scope) — the previous word's tag accounts for about half of the width gap
- [§33.58](10_the_learning_law.md#3358-the-night-of-6-october-the-vector-the-exact-null-five-tokenizers), [§33.91](13_how_far_the_gain_travels.md#3391-findings-checked-against-its-record-12-october), [§33.98](14_two_machines.md#3398-the-thesiss-list-and-its-ninth-item-the-law-in-ukrainian-is-within-its-seed-spread) — the law in Ukrainian on the constant null, within the seed spread
- [§33.102](14_two_machines.md#33102-the-night-of-16-october-width-512-settles-a-removed-features-deficit-halves-by-50k-and-case-leaves-none), [§33.111](14_two_machines.md#33111-the-probes-at-five-seeds-width-512s-twins-and-fifty-two-twins-in-all), [§33.114](14_two_machines.md#33114-the-width-probes-at-five-seeds-items-7-and-8-and-131-twins) — removal at full training; Ukrainian case leaves no deficit; the probes at five seeds
- [§33.117–§33.118](15_the_list_of_19_october.md#33117-the-texts-list-of-19-october-blimps-agreement-at-eight-seeds-the-exposure-comparison-on-the-laws-axis-and-item-4-prepared) — exposure read on the law's axis
- [§33.126](15_the_list_of_19_october.md#33126-items-3-and-6-english-part-of-speechs-deficit-closes-by-100k-and-at-30m-parameters-unigrams-shortfall-goes) — English part of speech's deficit closes by 100k; at 30M parameters Unigram's shortfall goes

### Controls and nulls

- [§18](4_the_audits.md#18-the-information-fair-rerun--q1-falls-q2-holds), [§21](5_grounding_and_controls.md#21-completing-the-controls--matched-vocabulary-frozen-vs-learned-and-signal-scale) — shuffled codes; frozen against learned
- [§27](6_level2_features_on_bpe.md#27-the-word-identity-control--what-the-effect-actually-was), [§29](6_level2_features_on_bpe.md#29-lemma-grouping-and-a-ruler-for-word-identity) — the word-identity control; a random-hash ruler
- [§33.9–§33.12](8_the_controlled_programme.md#339-the-shuffled-control-matches-structure) — the shuffled control, the null, the zero and flat slots
- [§33.14–§33.16](8_the_controlled_programme.md#3314-the-additive-null-gains-too), [§33.19](8_the_controlled_programme.md#3319-two-routes-to-the-same-gain) — the additive null; initialization families; eight starts of the baseline
- [§33.20–§33.21](9_ukrainian_and_learning.md#3320-what-we-know-and-ukrainian-begins), [§33.32](9_ukrainian_and_learning.md#3332-not-scarce-data--an-unfinished-lesson) — case from the word form, the leak-free control
- [§33.25–§33.27](9_ukrainian_and_learning.md#3325-a-day-for-g-65-does-case-cost-the-position-channel), [§33.30](9_ukrainian_and_learning.md#3330-ukrainian-at-four-seeds-and-a-chain-that-ran-twice) — G-65: a zero slot beside case, not established
- [§33.41–§33.43](10_the_learning_law.md#3341-the-chain-widened-four-directions-in-one-night), [§33.46](10_the_learning_law.md#3346-the-wide-models-crossing-and-the-slot-as-consistency), [§33.49–§33.52](10_the_learning_law.md#3349-the-screens-verdicts-and-time-budgets), [§33.57–§33.58](10_the_learning_law.md#3357-the-constants-replayed-and-the-night-of-6-october) — the slot's own effect: it builds with training, survives rotary positions, and is a shared vector (G-71)
- [§33.55–§33.58](10_the_learning_law.md#3355-what-the-tokenizers-expose-and-codes-read-across-corpus-sizes) — G-72 and G-73: the null made constant
- [§33.73–§33.74](12_full_training_and_the_leak.md#3373-the-queue-reviewed-what-each-candidate-settles-for-the-thesis), [§33.76](12_full_training_and_the_leak.md#3376-morphology-in-the-tokenizer-lasts-at-full-training--and-case-on-top-of-it-in-part-through-the-leak), [§33.81](12_full_training_and_the_leak.md#3381-case-at-full-training-without-the-leak-the-words-own-case-lasts-and-the-sentences-lead-was-the-leak), [§33.114](14_two_machines.md#33114-the-width-probes-at-five-seeds-items-7-and-8-and-131-twins) — prepositions read from their form; the word form's unseen words (G-76)
- [§33.99](14_two_machines.md#3399-a-causal-tagger-for-english-part-of-speech-the-lists-item-2-prepared), [§33.104](14_two_machines.md#33104-a-day-on-both-machines-the-comparison-g-78-on-each-and-item-2-prepared-17-october), [§33.107](14_two_machines.md#33107-in-english-too-the-sentences-lead-over-the-word-form-is-the-taggers-leak-full-trainings-twins-drift-apart) — a causal tagger for English part of speech
- [§33.102–§33.106](14_two_machines.md#33102-the-night-of-16-october-width-512-settles-a-removed-features-deficit-halves-by-50k-and-case-leaves-none), [§33.109–§33.110](14_two_machines.md#33109-why-twins-differ-only-dropouts-masks--and-the-nulls-handicap-is-endings-aparts), [§33.112–§33.113](14_two_machines.md#33112-one-long-chain-on-the-second-machine-g-78-across-tokenizers-and-sizes-the-width-probes-items-7-and-8-prepared-18-october), [§33.115–§33.116](14_two_machines.md#33115-two-follow-ups-the-exposure-comparison-at-20k-and-where-the-nulls-help-sits-prepared-18-october), [§33.118](15_the_list_of_19_october.md#33118-item-4-at-30m-parameters-case-gains-alike-under-both-tokenizers-and-the-null-costs-nothing) — G-78: the empty slot against the null
- [§33.125](15_the_list_of_19_october.md#33125-item-2-prepared-genders-and-numbers-leak-is-spread-over-word-classes-so-their-control-is-a-causal-tagger) — gender's and number's leak spread over word classes: a causal tagger as their control

### The mechanism: probes and the loss by kind

- [§25](6_level2_features_on_bpe.md#25-the-legibility-probe--what-injection-actually-provides), [§26.6](6_level2_features_on_bpe.md#266-the-probe-result-does-not-generalise--and-that-helps), [§29.11](6_level2_features_on_bpe.md#2911-the-part-of-speech-probe-and-a-probe-that-fed-the-model-its-answers) — the legibility probe and where it does not generalise; the part-of-speech probe
- [§30.2–§30.3](7_the_causal_reckoning.md#302-measuring-it) — where gains sit within a word
- [§33.8](8_the_controlled_programme.md#338-the-anaphor-gain-was-a-preference-and-what-that-changes) — BLiMP's items split by the direction of change
- [§33.48–§33.50](10_the_learning_law.md#3348-building-d1-and-d2-while-the-screen-runs), [§33.52](10_the_learning_law.md#3352-what-the-input-needs-and-what-the-probes-found), [§33.59–§33.60](10_the_learning_law.md#3359-what-the-feature-carries-probed--and-the-thesis-kept-in-scope) — probes across widths, after removal, and of the auxiliary target
- [§33.69–§33.70](11_morphology_in_the_tokenizer.md#3369-why-exposure-helps-case-the-loss-by-what-each-token-is), [§33.74](12_full_training_and_the_leak.md#3374-what-the-sentences-case-adds-prepositions-and-words-the-form-never-saw), [§33.76](12_full_training_and_the_leak.md#3376-morphology-in-the-tokenizer-lasts-at-full-training--and-case-on-top-of-it-in-part-through-the-leak), [§33.81](12_full_training_and_the_leak.md#3381-case-at-full-training-without-the-leak-the-words-own-case-lasts-and-the-sentences-lead-was-the-leak) — the loss by what each token is
- [§33.84](13_how_far_the_gain_travels.md#3384-context-aware-morphology-inside-the-models) — case, number and gender inside the models
- [§33.105–§33.106](14_two_machines.md#33105-why-twins-differ-item-5s-probes-at-four-seeds-and-each-machines-afternoon-prepared-17-october), [§33.111–§33.112](14_two_machines.md#33111-the-probes-at-five-seeds-width-512s-twins-and-fifty-two-twins-in-all), [§33.114](14_two_machines.md#33114-the-width-probes-at-five-seeds-items-7-and-8-and-131-twins), [§33.116](14_two_machines.md#33116-the-follow-ups-by-20k-the-two-tokenizers-are-level-the-slot-costs-at-word-starts-or-helps-on-a-words-ending-by-tokenizer) — the probes at five seeds; where the slot's effect sits
- [§33.128](15_the_list_of_19_october.md#33128-g-82-prepared-the-causal-taggers-rest-split-by-the-word-it-describes), [§33.129](15_the_list_of_19_october.md#33129-g-82-what-the-causal-tagger-adds-over-the-word-form-is-not-context) — gender's and number's causal rest, split by the word it describes: after unseen words, not context

### Method and infrastructure: bugs, the two machines, twins

- [§7](2_building_the_tokenizer.md#7-the-segmenter-and-the-bug-that-cost-thousands-of-prefixes), [§14](4_the_audits.md#14-the-frozen-embedding-audit), [§17](4_the_audits.md#17-the-information-channel-audit--making-the-comparison-fair), [§28.3](6_level2_features_on_bpe.md#283-three-numbers-in-our-own-paper-were-wrong), [§30.8](7_the_causal_reckoning.md#308-the-prefix-hash-and-a-third-boundary-bug) — bugs and audits that changed numbers
- [§12–§13](3_the_apparatus.md#12-the-dataset-model-and-the-fairness-metric), [§16](4_the_audits.md#16-restructuring--reversing-the-flat-layout-decision) — the metric and calibration; the code restructured
- [§33.5–§33.6](8_the_controlled_programme.md#335-tools-this-required), [§33.21–§33.23](9_ukrainian_and_learning.md#3321-a-gate-that-held-words-glued-by-split-punctuation), [§33.30](9_ukrainian_and_learning.md#3330-ukrainian-at-four-seeds-and-a-chain-that-ran-twice) — tools and gates; the exported comparison table; the Ukrainian word rule; a chain that ran twice
- [§33.39](10_the_learning_law.md#3339-day-block-d-the-confound-and-a-wider-model), [§33.43–§33.45](10_the_learning_law.md#3343-a-bit-for-bit-reproduction), [§33.48](10_the_learning_law.md#3348-building-d1-and-d2-while-the-screen-runs) — run identity: labels, a bit-for-bit reproduction, a quarantine rule, tests that never ran
- [§33.47](10_the_learning_law.md#3347-a-screen-before-the-long-chain), [§33.49](10_the_learning_law.md#3349-the-screens-verdicts-and-time-budgets), [§33.51](10_the_learning_law.md#3351-an-empty-slot-under-rotary-positions-and-resumable-chains), [§33.53–§33.54](10_the_learning_law.md#3353-tools-for-other-conditions), [§33.64](11_morphology_in_the_tokenizer.md#3364-a-long-chain-for-7-october-the-sixth-tokenizer-at-four-seeds-and-d7-at-four) — screens and time budgets; resumable chains; tools for other conditions and models; time boxes
- [§33.62–§33.63](11_morphology_in_the_tokenizer.md#3362-the-sixth-tokenizer-measured--and-nothing-outside-the-project), [§33.65–§33.66](11_morphology_in_the_tokenizer.md#3365-morphology-in-the-tokenizer-helps--and-case-helps-more-where-endings-are-exposed) — nothing outside the project; the tagger recorded (G-74); one text under every tokenizer (G-75); one machine
- [§33.71–§33.73](12_full_training_and_the_leak.md#3371-the-manuscript-begins), [§33.77](12_full_training_and_the_leak.md#3377-the-queue-reviewed-after-the-chain-of-9-october), [§33.83](13_how_far_the_gain_travels.md#3383-is-a-direction-already-dead-the-thesis-reviewed-11-october), [§33.91](13_how_far_the_gain_travels.md#3391-findings-checked-against-its-record-12-october) — the manuscript and its exports; the queue review and the sentence test; FINDINGS checked against its record
- [§33.87](13_how_far_the_gain_travels.md#3387-english-with-endings-apart-it-helps-a-fifth-as-much), [§33.102](14_two_machines.md#33102-the-night-of-16-october-width-512-settles-a-removed-features-deficit-halves-by-50k-and-case-leaves-none) — the fiction pipeline: a cost that grew with the square of a paragraph's length; a paragraph past the tagger's limit
- [§33.101](14_two_machines.md#33101-a-second-machine-and-the-night-of-16-october-prepared), [§33.103–§33.107](14_two_machines.md#33103-twins-and-the-empty-slot-on-one-machine-prepared-17-october), [§33.109](14_two_machines.md#33109-why-twins-differ-only-dropouts-masks--and-the-nulls-handicap-is-endings-aparts), [§33.111](14_two_machines.md#33111-the-probes-at-five-seeds-width-512s-twins-and-fifty-two-twins-in-all), [§33.114](14_two_machines.md#33114-the-width-probes-at-five-seeds-items-7-and-8-and-131-twins), [§33.118](15_the_list_of_19_october.md#33118-item-4-at-30m-parameters-case-gains-alike-under-both-tokenizers-and-the-null-costs-nothing) — the second machine: calibration, twins, dropout's masks, pairs across machines
- [§33.116](14_two_machines.md#33116-the-follow-ups-by-20k-the-two-tokenizers-are-level-the-slot-costs-at-word-starts-or-helps-on-a-words-ending-by-tokenizer) — models copied between machines (G-79)
- [§33.122](15_the_list_of_19_october.md#33122-multiblimp-at-width-1024-the-gain-holds-at-every-seed-and-stops-growing-g-81s-fall-is-no-spike) — grammar moves more between machines than bits per byte; a rerun on the second machine is close, not bit for bit

## Every section

### Part 1: The Question

- [1. The Starting Point and the Intervention Point](1_the_question.md#1-the-starting-point-and-the-intervention-point)
- [2. The Hybrid Frozen/Trained Embedding](1_the_question.md#2-the-hybrid-frozentrained-embedding)
- [3. What Can Be Known, and Where](1_the_question.md#3-what-can-be-known-and-where)
- [4. Choosing the Unit, and How to Evaluate It](1_the_question.md#4-choosing-the-unit-and-how-to-evaluate-it)

### Part 2: Building the Morphological Tokenizer

- [5. The Vocabulary: Why Frequency Is the Wrong Criterion](2_building_the_tokenizer.md#5-the-vocabulary-why-frequency-is-the-wrong-criterion)
- [6. Named Entities: The Threshold Wars and a Structural Answer](2_building_the_tokenizer.md#6-named-entities-the-threshold-wars-and-a-structural-answer)
- [7. The Segmenter, and the Bug That Cost Thousands of Prefixes](2_building_the_tokenizer.md#7-the-segmenter-and-the-bug-that-cost-thousands-of-prefixes)
- [8. Morfessor's Null Result and the Three-Pipeline Decision](2_building_the_tokenizer.md#8-morfessors-null-result-and-the-three-pipeline-decision)
- [9. Compound Splitting and the Particle Fix](2_building_the_tokenizer.md#9-compound-splitting-and-the-particle-fix)
- [10. What the First Comparison Showed — and What It Concealed](2_building_the_tokenizer.md#10-what-the-first-comparison-showed--and-what-it-concealed)

### Part 3: The Apparatus

- [11. Designing the Frozen Embedding Layer](3_the_apparatus.md#11-designing-the-frozen-embedding-layer)
- [12. The Dataset, Model, and the Fairness Metric](3_the_apparatus.md#12-the-dataset-model-and-the-fairness-metric)
- [13. Calibration and Literature Grounding](3_the_apparatus.md#13-calibration-and-literature-grounding)

### Part 4: The Audits

- [14. The Frozen Embedding Audit](4_the_audits.md#14-the-frozen-embedding-audit)
- [15. Parameter Tying vs Information Injection](4_the_audits.md#15-parameter-tying-vs-information-injection)
- [16. Restructuring — Reversing the Flat-Layout Decision](4_the_audits.md#16-restructuring--reversing-the-flat-layout-decision)
- [17. The Information Channel Audit — Making the Comparison Fair](4_the_audits.md#17-the-information-channel-audit--making-the-comparison-fair)
- [18. The Information-Fair Rerun — Q1 Falls, Q2 Holds](4_the_audits.md#18-the-information-fair-rerun--q1-falls-q2-holds)
- [19. Localising the Q1 Deficit — Coverage, Decomposition, and the Unified Finding](4_the_audits.md#19-localising-the-q1-deficit--coverage-decomposition-and-the-unified-finding)

### Part 5: Grounding and Controls

- [20. Literature Grounding — Who Has Done This Already](5_grounding_and_controls.md#20-literature-grounding--who-has-done-this-already)
- [21. Completing the Controls — Matched Vocabulary, Frozen vs Learned, and Signal Scale](5_grounding_and_controls.md#21-completing-the-controls--matched-vocabulary-frozen-vs-learned-and-signal-scale)
  - [21.1 Matched vocabulary — the Q1 confound, finally removed](5_grounding_and_controls.md#211-matched-vocabulary--the-q1-confound-finally-removed)
  - [21.2 Frozen vs learned factors — the thesis claim, tested directly](5_grounding_and_controls.md#212-frozen-vs-learned-factors--the-thesis-claim-tested-directly)
  - [21.3 The scale confound — found in our own implementation](5_grounding_and_controls.md#213-the-scale-confound--found-in-our-own-implementation)
  - [21.4 What survives, and what it means](5_grounding_and_controls.md#214-what-survives-and-what-it-means)
  - [21.5 Convergence speed — a free finding](5_grounding_and_controls.md#215-convergence-speed--a-free-finding)
  - [21.6 One claim from §11 still untested](5_grounding_and_controls.md#216-one-claim-from-11-still-untested)
- [22. Why English Resists — and the Turn from Confrontation to Cooperation](5_grounding_and_controls.md#22-why-english-resists--and-the-turn-from-confrontation-to-cooperation)
  - [22.1 Why English BPE is so hard to beat](5_grounding_and_controls.md#221-why-english-bpe-is-so-hard-to-beat)
  - [22.2 How much of the deficit is ours, and how much is the language](5_grounding_and_controls.md#222-how-much-of-the-deficit-is-ours-and-how-much-is-the-language)
  - [22.3 The path considered and rejected — data starvation](5_grounding_and_controls.md#223-the-path-considered-and-rejected--data-starvation)
  - [22.4 The reframe — stop confronting BPE](5_grounding_and_controls.md#224-the-reframe--stop-confronting-bpe)
  - [22.5 Two levels of cooperation](5_grounding_and_controls.md#225-two-levels-of-cooperation)
  - [22.6 Where this leaves the project](5_grounding_and_controls.md#226-where-this-leaves-the-project)

### Part 6: Level 2: Features on a BPE Model

- [23. Level 2 — Morphology as Features on a BPE-Tokenized Model](6_level2_features_on_bpe.md#23-level-2--morphology-as-features-on-a-bpe-tokenized-model)
  - [23.1 A bug found before anything ran](6_level2_features_on_bpe.md#231-a-bug-found-before-anything-ran)
  - [23.2 Design decisions taken to avoid re-runs](6_level2_features_on_bpe.md#232-design-decisions-taken-to-avoid-re-runs)
  - [23.3 The preflight, and what it predicted](6_level2_features_on_bpe.md#233-the-preflight-and-what-it-predicted)
  - [23.4 The results](6_level2_features_on_bpe.md#234-the-results)
  - [23.5 Seed replication — the last thing being asserted rather than shown](6_level2_features_on_bpe.md#235-seed-replication--the-last-thing-being-asserted-rather-than-shown)
  - [23.6 Three findings beyond the headline](6_level2_features_on_bpe.md#236-three-findings-beyond-the-headline)
  - [23.7 The oracle question, substantially answered](6_level2_features_on_bpe.md#237-the-oracle-question-substantially-answered)
  - [23.8 What the thesis now claims](6_level2_features_on_bpe.md#238-what-the-thesis-now-claims)
- [24. Where the Project Stands](6_level2_features_on_bpe.md#24-where-the-project-stands)
- [25. The Legibility Probe — What Injection Actually Provides](6_level2_features_on_bpe.md#25-the-legibility-probe--what-injection-actually-provides)
  - [25.1 What was run](6_level2_features_on_bpe.md#251-what-was-run)
  - [25.2 Results, and their replication](6_level2_features_on_bpe.md#252-results-and-their-replication)
  - [25.3 Finding 1 — injection supplies precision, not information](6_level2_features_on_bpe.md#253-finding-1--injection-supplies-precision-not-information)
  - [25.4 Finding 2 — freezing preserves legibility, modestly](6_level2_features_on_bpe.md#254-finding-2--freezing-preserves-legibility-modestly)
  - [25.5 Limitations of this measurement](6_level2_features_on_bpe.md#255-limitations-of-this-measurement)
- [26. Closing the Reviewer Gaps — Held-Out Evaluation and the Four Missing Cells](6_level2_features_on_bpe.md#26-closing-the-reviewer-gaps--held-out-evaluation-and-the-four-missing-cells)
  - [26.1 What the audit found](6_level2_features_on_bpe.md#261-what-the-audit-found)
  - [26.2 Held-out evaluation — every claim survives](6_level2_features_on_bpe.md#262-held-out-evaluation--every-claim-survives)
  - [26.3 The headline control](6_level2_features_on_bpe.md#263-the-headline-control)
  - [26.4 `mrole` contributes nothing, and the composition becomes exact](6_level2_features_on_bpe.md#264-mrole-contributes-nothing-and-the-composition-becomes-exact)
  - [26.5 The oracle argument becomes a direct measurement](6_level2_features_on_bpe.md#265-the-oracle-argument-becomes-a-direct-measurement)
  - [26.6 The probe result does NOT generalise — and that helps](6_level2_features_on_bpe.md#266-the-probe-result-does-not-generalise--and-that-helps)
  - [26.7 Where the register stands](6_level2_features_on_bpe.md#267-where-the-register-stands)
- [27. The Word-Identity Control — What the Effect Actually Was](6_level2_features_on_bpe.md#27-the-word-identity-control--what-the-effect-actually-was)
  - [27.1 The control](6_level2_features_on_bpe.md#271-the-control)
  - [27.2 The control beats the treatment](6_level2_features_on_bpe.md#272-the-control-beats-the-treatment)
  - [27.3 Why random beats morphological](6_level2_features_on_bpe.md#273-why-random-beats-morphological)
  - [27.4 The mechanism is confirmed — for identity, not morphology](6_level2_features_on_bpe.md#274-the-mechanism-is-confirmed--for-identity-not-morphology)
  - [27.5 What survives — the cardinality test](6_level2_features_on_bpe.md#275-what-survives--the-cardinality-test)
  - [27.6 The two mechanisms have different frequency signatures](6_level2_features_on_bpe.md#276-the-two-mechanisms-have-different-frequency-signatures)
  - [27.6b Transfer tells the two mechanisms apart a second time](6_level2_features_on_bpe.md#276b-transfer-tells-the-two-mechanisms-apart-a-second-time)
  - [27.7 What the thesis claims now](6_level2_features_on_bpe.md#277-what-the-thesis-claims-now)
  - [27.8 Outstanding](6_level2_features_on_bpe.md#278-outstanding)
- [28. Turning the Instrument on Ourselves](6_level2_features_on_bpe.md#28-turning-the-instrument-on-ourselves)
  - [28.1 What was asked, and what it cost](6_level2_features_on_bpe.md#281-what-was-asked-and-what-it-cost)
  - [28.2 The audit predicts costs, not just violations](6_level2_features_on_bpe.md#282-the-audit-predicts-costs-not-just-violations)
  - [28.3 Three numbers in our own paper were wrong](6_level2_features_on_bpe.md#283-three-numbers-in-our-own-paper-were-wrong)
  - [28.4 The complete channel decomposition](6_level2_features_on_bpe.md#284-the-complete-channel-decomposition)
  - [28.5 The reversal is robust, but the credit was misassigned](6_level2_features_on_bpe.md#285-the-reversal-is-robust-but-the-credit-was-misassigned)
  - [28.6 The result is not specific to BPE](6_level2_features_on_bpe.md#286-the-result-is-not-specific-to-bpe)
  - [28.7 One shared learning rate does not favour either arm](6_level2_features_on_bpe.md#287-one-shared-learning-rate-does-not-favour-either-arm)
  - [28.8 Every headline cell replicated](6_level2_features_on_bpe.md#288-every-headline-cell-replicated)
  - [28.9 What this episode is worth](6_level2_features_on_bpe.md#289-what-this-episode-is-worth)
- [29. Lemma Grouping, and a Ruler for Word Identity](6_level2_features_on_bpe.md#29-lemma-grouping-and-a-ruler-for-word-identity)
  - [29.1 The middle term nobody had tested](6_level2_features_on_bpe.md#291-the-middle-term-nobody-had-tested)
  - [29.2 The prediction held](6_level2_features_on_bpe.md#292-the-prediction-held)
  - [29.3 How precise does identity need to be?](6_level2_features_on_bpe.md#293-how-precise-does-identity-need-to-be)
  - [29.4 The first sweep was invalid](6_level2_features_on_bpe.md#294-the-first-sweep-was-invalid)
  - [29.5 The check for the fix failed first — correctly](6_level2_features_on_bpe.md#295-the-check-for-the-fix-failed-first--correctly)
  - [29.6 The valid curve](6_level2_features_on_bpe.md#296-the-valid-curve)
  - [29.7 Continuity, verified exactly](6_level2_features_on_bpe.md#297-continuity-verified-exactly)
  - [29.8 The curve as a ruler](6_level2_features_on_bpe.md#298-the-curve-as-a-ruler)
  - [29.9 What changes in earlier sections](6_level2_features_on_bpe.md#299-what-changes-in-earlier-sections)
  - [29.10 What this episode shows about the process](6_level2_features_on_bpe.md#2910-what-this-episode-shows-about-the-process)
  - [29.11 The part-of-speech probe, and a probe that fed the model its answers](6_level2_features_on_bpe.md#2911-the-part-of-speech-probe-and-a-probe-that-fed-the-model-its-answers)

### Part 7: The Causal Reckoning

- [30. The Lookahead in Word-Form Features](7_the_causal_reckoning.md#30-the-lookahead-in-word-form-features)
  - [30.1 A claim asserted, never checked](7_the_causal_reckoning.md#301-a-claim-asserted-never-checked)
  - [30.2 Measuring it](7_the_causal_reckoning.md#302-measuring-it)
  - [30.3 The result, and a clean split](7_the_causal_reckoning.md#303-the-result-and-a-clean-split)
  - [30.4 What this cost](7_the_causal_reckoning.md#304-what-this-cost)
  - [30.5 The fix, and what survives](7_the_causal_reckoning.md#305-the-fix-and-what-survives)
  - [30.6 Where the tokenizer route stands](7_the_causal_reckoning.md#306-where-the-tokenizer-route-stands)
  - [30.6b Is `dep` an oracle? Two measurements, one of which failed](7_the_causal_reckoning.md#306b-is-dep-an-oracle-two-measurements-one-of-which-failed)
  - [30.7 The claim now](7_the_causal_reckoning.md#307-the-claim-now)
  - [30.8 The prefix hash, and a third boundary bug](7_the_causal_reckoning.md#308-the-prefix-hash-and-a-third-boundary-bug)
- [31. It Was the Structure All Along](7_the_causal_reckoning.md#31-it-was-the-structure-all-along)
  - [31.1 Every causal cell so far confounded two things](7_the_causal_reckoning.md#311-every-causal-cell-so-far-confounded-two-things)
  - [31.2 The isolated contributions](7_the_causal_reckoning.md#312-the-isolated-contributions)
  - [31.3 What this establishes, and what it does not](7_the_causal_reckoning.md#313-what-this-establishes-and-what-it-does-not)
  - [31.4 The reversal, and why it is coherent](7_the_causal_reckoning.md#314-the-reversal-and-why-it-is-coherent)
  - [31.5 A fifth wrong prediction, caught in time](7_the_causal_reckoning.md#315-a-fifth-wrong-prediction-caught-in-time)
  - [31.6 The claim now](7_the_causal_reckoning.md#316-the-claim-now)
  - [31.7 Three seeds, and a noise floor that was never measured](7_the_causal_reckoning.md#317-three-seeds-and-a-noise-floor-that-was-never-measured)
- [32. A Fresh Look: What the Records Already Knew](7_the_causal_reckoning.md#32-a-fresh-look-what-the-records-already-knew)
  - [32.1 The structure gain grows through training](7_the_causal_reckoning.md#321-the-structure-gain-grows-through-training)
  - [32.2 Three temporal profiles, and why structure survives](7_the_causal_reckoning.md#322-three-temporal-profiles-and-why-structure-survives)
  - [32.3 Morpheme tokenization learns slower](7_the_causal_reckoning.md#323-morpheme-tokenization-learns-slower)
  - [32.4 What a fresh reader asks](7_the_causal_reckoning.md#324-what-a-fresh-reader-asks)
  - [32.5 What this section adds to the claim](7_the_causal_reckoning.md#325-what-this-section-adds-to-the-claim)

### Part 8: The Controlled Programme

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

### Part 9: Ukrainian, and Before the Model Has Learned It

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

### Part 10: The Learning Law: A Crutch, Not a Teacher

- [33.33 After the demo: English at a short budget, and a switch](10_the_learning_law.md#3333-after-the-demo-english-at-a-short-budget-and-a-switch)
- [33.34 A crutch, not a teacher](10_the_learning_law.md#3334-a-crutch-not-a-teacher)
- [33.35 A programme after the demo](10_the_learning_law.md#3335-a-programme-after-the-demo)
- [33.36 The first programme night](10_the_learning_law.md#3336-the-first-programme-night)
- [33.37 The follow-on of 2 October](10_the_learning_law.md#3337-the-follow-on-of-2-october)
- [33.38 The law holds; a critical period](10_the_learning_law.md#3338-the-law-holds-a-critical-period)
- [33.39 Day block D: the confound, and a wider model](10_the_learning_law.md#3339-day-block-d-the-confound-and-a-wider-model)
- [33.40 Two pivots](10_the_learning_law.md#3340-two-pivots)
- [33.41 The chain widened: four directions in one night](10_the_learning_law.md#3341-the-chain-widened-four-directions-in-one-night)
- [33.42 The night of four directions](10_the_learning_law.md#3342-the-night-of-four-directions)
- [33.43 A bit-for-bit reproduction](10_the_learning_law.md#3343-a-bit-for-bit-reproduction)
- [33.44 A rule written for one generation of runs](10_the_learning_law.md#3344-a-rule-written-for-one-generation-of-runs)
- [33.45 Tests the suite never ran](10_the_learning_law.md#3345-tests-the-suite-never-ran)
- [33.46 The wide model's crossing, and the slot as consistency](10_the_learning_law.md#3346-the-wide-models-crossing-and-the-slot-as-consistency)
- [33.47 A screen before the long chain](10_the_learning_law.md#3347-a-screen-before-the-long-chain)
- [33.48 Building D1 and D2 while the screen runs](10_the_learning_law.md#3348-building-d1-and-d2-while-the-screen-runs)
- [33.49 The screen's verdicts, and time budgets](10_the_learning_law.md#3349-the-screens-verdicts-and-time-budgets)
- [33.50 The teacher fades faster than the crutch](10_the_learning_law.md#3350-the-teacher-fades-faster-than-the-crutch)
- [33.51 An empty slot under rotary positions, and resumable chains](10_the_learning_law.md#3351-an-empty-slot-under-rotary-positions-and-resumable-chains)
- [33.52 What the input needs, and what the probes found](10_the_learning_law.md#3352-what-the-input-needs-and-what-the-probes-found)
- [33.53 Tools for other conditions](10_the_learning_law.md#3353-tools-for-other-conditions)
- [33.54 Tools for other models](10_the_learning_law.md#3354-tools-for-other-models)
- [33.55 What the tokenizers expose, and codes read across corpus sizes](10_the_learning_law.md#3355-what-the-tokenizers-expose-and-codes-read-across-corpus-sizes)
- [33.56 What the nulls read, and a null that was not constant](10_the_learning_law.md#3356-what-the-nulls-read-and-a-null-that-was-not-constant)
- [33.57 The constants replayed, and the night of 6 October](10_the_learning_law.md#3357-the-constants-replayed-and-the-night-of-6-october)
- [33.58 The night of 6 October: the vector, the exact null, five tokenizers](10_the_learning_law.md#3358-the-night-of-6-october-the-vector-the-exact-null-five-tokenizers)
- [33.59 What the feature carries, probed — and the thesis kept in scope](10_the_learning_law.md#3359-what-the-feature-carries-probed--and-the-thesis-kept-in-scope)
- [33.60 The previous word's tag: half the width gap, and what the switch leaves](10_the_learning_law.md#3360-the-previous-words-tag-half-the-width-gap-and-what-the-switch-leaves)

### Part 11: Morphology in the Tokenizer Helps, and Lasts

- [33.61 Morphology in the tokenizer: the ending rule, checked, and BPE that keeps endings apart](11_morphology_in_the_tokenizer.md#3361-morphology-in-the-tokenizer-the-ending-rule-checked-and-bpe-that-keeps-endings-apart)
- [33.62 The sixth tokenizer, measured — and nothing outside the project](11_morphology_in_the_tokenizer.md#3362-the-sixth-tokenizer-measured--and-nothing-outside-the-project)
- [33.63 The sixth condition, ready — and what its numbers showed about the tools](11_morphology_in_the_tokenizer.md#3363-the-sixth-condition-ready--and-what-its-numbers-showed-about-the-tools)
- [33.64 A long chain for 7 October: the sixth tokenizer at four seeds, and D7 at four](11_morphology_in_the_tokenizer.md#3364-a-long-chain-for-7-october-the-sixth-tokenizer-at-four-seeds-and-d7-at-four)
- [33.65 Morphology in the tokenizer helps — and case helps more where endings are exposed](11_morphology_in_the_tokenizer.md#3365-morphology-in-the-tokenizer-helps--and-case-helps-more-where-endings-are-exposed)
- [33.66 The same text under every tokenizer — and the work moves to one machine](11_morphology_in_the_tokenizer.md#3366-the-same-text-under-every-tokenizer--and-the-work-moves-to-one-machine)
- [33.67 A long chain for 8 October: does morphology in the tokenizer last?](11_morphology_in_the_tokenizer.md#3367-a-long-chain-for-8-october-does-morphology-in-the-tokenizer-last)
- [33.68 Morphology in the tokenizer lasts with training, and is larger in a wider model](11_morphology_in_the_tokenizer.md#3368-morphology-in-the-tokenizer-lasts-with-training-and-is-larger-in-a-wider-model)
- [33.69 Why exposure helps case: the loss by what each token is](11_morphology_in_the_tokenizer.md#3369-why-exposure-helps-case-the-loss-by-what-each-token-is)
- [33.70 Case helps by predicting endings — and the exposure effect lies elsewhere](11_morphology_in_the_tokenizer.md#3370-case-helps-by-predicting-endings--and-the-exposure-effect-lies-elsewhere)

### Part 12: Full Training, Grammar, and the Leak

- [33.71 The manuscript begins](12_full_training_and_the_leak.md#3371-the-manuscript-begins)
- [33.72 paper/ becomes exports/, kept current](12_full_training_and_the_leak.md#3372-paper-becomes-exports-kept-current)
- [33.73 The queue reviewed: what each candidate settles for the thesis](12_full_training_and_the_leak.md#3373-the-queue-reviewed-what-each-candidate-settles-for-the-thesis)
- [33.74 What the sentence's case adds: prepositions, and words the form never saw](12_full_training_and_the_leak.md#3374-what-the-sentences-case-adds-prepositions-and-words-the-form-never-saw)
- [33.75 Ten hours, and two decisions](12_full_training_and_the_leak.md#3375-ten-hours-and-two-decisions)
- [33.76 Morphology in the tokenizer lasts at full training — and case on top of it, in part through the leak](12_full_training_and_the_leak.md#3376-morphology-in-the-tokenizer-lasts-at-full-training--and-case-on-top-of-it-in-part-through-the-leak)
- [33.77 The queue reviewed after the chain of 9 October](12_full_training_and_the_leak.md#3377-the-queue-reviewed-after-the-chain-of-9-october)
- [33.78 Ten hours, 10 October](12_full_training_and_the_leak.md#3378-ten-hours-10-october)
- [33.79 A grammatical test in Ukrainian: MultiBLiMP](12_full_training_and_the_leak.md#3379-a-grammatical-test-in-ukrainian-multiblimp)
- [33.80 Figures for the thesis, in Ukrainian](12_full_training_and_the_leak.md#3380-figures-for-the-thesis-in-ukrainian)
- [33.81 Case at full training without the leak: the word's own case lasts, and the sentence's lead was the leak](12_full_training_and_the_leak.md#3381-case-at-full-training-without-the-leak-the-words-own-case-lasts-and-the-sentences-lead-was-the-leak)
- [33.82 One more hour, 10 October](12_full_training_and_the_leak.md#3382-one-more-hour-10-october)

### Part 13: How Far the Gain Travels

- [33.83 Is a direction already dead? The thesis reviewed, 11 October](13_how_far_the_gain_travels.md#3383-is-a-direction-already-dead-the-thesis-reviewed-11-october)
- [33.84 Context-aware morphology inside the models](13_how_far_the_gain_travels.md#3384-context-aware-morphology-inside-the-models)
- [33.85 How far the tokenizer's gain travels: English, and Ukrainian fiction (prepared)](13_how_far_the_gain_travels.md#3385-how-far-the-tokenizers-gain-travels-english-and-ukrainian-fiction-prepared)
- [33.86 The wider model's lead settles: +0.022 at width 256 and 20k](13_how_far_the_gain_travels.md#3386-the-wider-models-lead-settles-0022-at-width-256-and-20k)
- [33.87 English with endings apart: it helps, a fifth as much](13_how_far_the_gain_travels.md#3387-english-with-endings-apart-it-helps-a-fifth-as-much)
- [33.88 A second Ukrainian corpus: fiction keeps the gain, at half the size in bits per byte and in full in grammar](13_how_far_the_gain_travels.md#3388-a-second-ukrainian-corpus-fiction-keeps-the-gain-at-half-the-size-in-bits-per-byte-and-in-full-in-grammar)
- [33.89 Ten hours in batches, 11 October: what ran, and the queue after](13_how_far_the_gain_travels.md#3389-ten-hours-in-batches-11-october-what-ran-and-the-queue-after)
- [33.90 English at full training (prepared), 12 October](13_how_far_the_gain_travels.md#3390-english-at-full-training-prepared-12-october)
- [33.91 FINDINGS checked against its record, 12 October](13_how_far_the_gain_travels.md#3391-findings-checked-against-its-record-12-october)
- [33.92 English at full training: the gain holds on the median, at three seeds of four](13_how_far_the_gain_travels.md#3392-english-at-full-training-the-gain-holds-on-the-median-at-three-seeds-of-four)
- [33.93 Does the tokenizer's gain keep growing with model size? Width 512 (prepared), 14 October](13_how_far_the_gain_travels.md#3393-does-the-tokenizers-gain-keep-growing-with-model-size-width-512-prepared-14-october)
- [33.94 Width 512: the gain in bits per byte peaks at width 256; the grammatical gain keeps growing](13_how_far_the_gain_travels.md#3394-width-512-the-gain-in-bits-per-byte-peaks-at-width-256-the-grammatical-gain-keeps-growing)
- [33.95 Does the width-512 lead settle? Seed 0 at 20k (prepared), 14 October](13_how_far_the_gain_travels.md#3395-does-the-width-512-lead-settle-seed-0-at-20k-prepared-14-october)
- [33.96 Width 512 at 20k, seed 0: the lead levels off near +0.011](13_how_far_the_gain_travels.md#3396-width-512-at-20k-seed-0-the-lead-levels-off-near-0011)

### Part 14: Two Machines, and a Null That Is Not Neutral

- [33.97 Width 512 at 20k, seed 1 (prepared), 15 October](14_two_machines.md#3397-width-512-at-20k-seed-1-prepared-15-october)
- [33.98 The thesis's list, and its ninth item: the law in Ukrainian is within its seed spread](14_two_machines.md#3398-the-thesiss-list-and-its-ninth-item-the-law-in-ukrainian-is-within-its-seed-spread)
- [33.99 A causal tagger for English part of speech (the list's item 2, prepared)](14_two_machines.md#3399-a-causal-tagger-for-english-part-of-speech-the-lists-item-2-prepared)
- [33.100 Width 512 at 20k, seed 1: levelled off too, at +0.0097](14_two_machines.md#33100-width-512-at-20k-seed-1-levelled-off-too-at-00097)
- [33.101 A second machine, and the night of 16 October (prepared)](14_two_machines.md#33101-a-second-machine-and-the-night-of-16-october-prepared)
- [33.102 The night of 16 October: width 512 settles, a removed feature's deficit halves by 50k, and case leaves none](14_two_machines.md#33102-the-night-of-16-october-width-512-settles-a-removed-features-deficit-halves-by-50k-and-case-leaves-none)
- [33.103 Twins and the empty slot, on one machine (prepared, 17 October)](14_two_machines.md#33103-twins-and-the-empty-slot-on-one-machine-prepared-17-october)
- [33.104 A day on both machines: the comparison, G-78 on each, and item 2 (prepared, 17 October)](14_two_machines.md#33104-a-day-on-both-machines-the-comparison-g-78-on-each-and-item-2-prepared-17-october)
- [33.105 Why twins differ, item 5's probes at four seeds, and each machine's afternoon (prepared, 17 October)](14_two_machines.md#33105-why-twins-differ-item-5s-probes-at-four-seeds-and-each-machines-afternoon-prepared-17-october)
- [33.106 The machines agree, and the null is handicapped in Ukrainian: case's own gain sits on the endings](14_two_machines.md#33106-the-machines-agree-and-the-null-is-handicapped-in-ukrainian-cases-own-gain-sits-on-the-endings)
- [33.107 In English too the sentence's lead over the word form is the tagger's leak; full training's twins drift apart](14_two_machines.md#33107-in-english-too-the-sentences-lead-over-the-word-form-is-the-taggers-leak-full-trainings-twins-drift-apart)
- [33.108 Fiction at full training: the tokenizer's gain holds, and reaches Wikipedia's share](14_two_machines.md#33108-fiction-at-full-training-the-tokenizers-gain-holds-and-reaches-wikipedias-share)
- [33.109 Why twins differ: only dropout's masks — and the null's handicap is endings apart's](14_two_machines.md#33109-why-twins-differ-only-dropouts-masks--and-the-nulls-handicap-is-endings-aparts)
- [33.110 G-78 mapped: the null's handicap is endings apart's at every budget — and it confounds "exposure makes case more useful"](14_two_machines.md#33110-g-78-mapped-the-nulls-handicap-is-endings-aparts-at-every-budget--and-it-confounds-exposure-makes-case-more-useful)
- [33.111 The probes at five seeds, width 512's twins, and fifty-two twins in all](14_two_machines.md#33111-the-probes-at-five-seeds-width-512s-twins-and-fifty-two-twins-in-all)
- [33.112 One long chain on the second machine: G-78 across tokenizers and sizes, the width probes, items 7 and 8 (prepared, 18 October)](14_two_machines.md#33112-one-long-chain-on-the-second-machine-g-78-across-tokenizers-and-sizes-the-width-probes-items-7-and-8-prepared-18-october)
- [33.113 G-78 across tokenizers and corpus sizes: the slot costs where endings are exposed, helps under BPE — and exposure makes case partly redundant](14_two_machines.md#33113-g-78-across-tokenizers-and-corpus-sizes-the-slot-costs-where-endings-are-exposed-helps-under-bpe--and-exposure-makes-case-partly-redundant)
- [33.114 The width probes at five seeds, items 7 and 8, and 131 twins](14_two_machines.md#33114-the-width-probes-at-five-seeds-items-7-and-8-and-131-twins)
- [33.115 Two follow-ups: the exposure comparison at 20k, and where the null's help sits (prepared, 18 October)](14_two_machines.md#33115-two-follow-ups-the-exposure-comparison-at-20k-and-where-the-nulls-help-sits-prepared-18-october)
- [33.116 The follow-ups: by 20k the two tokenizers are level; the slot costs at word starts or helps on a word's ending, by tokenizer](14_two_machines.md#33116-the-follow-ups-by-20k-the-two-tokenizers-are-level-the-slot-costs-at-word-starts-or-helps-on-a-words-ending-by-tokenizer)

### Part 15: The List of 19 October: Case and the Null at 30M Parameters

- [33.117 The text's list of 19 October: BLiMP's agreement at eight seeds, the exposure comparison on the law's axis, and item 4 prepared](15_the_list_of_19_october.md#33117-the-texts-list-of-19-october-blimps-agreement-at-eight-seeds-the-exposure-comparison-on-the-laws-axis-and-item-4-prepared)
- [33.118 Item 4: at 30M parameters case gains alike under both tokenizers, and the null costs nothing](15_the_list_of_19_october.md#33118-item-4-at-30m-parameters-case-gains-alike-under-both-tokenizers-and-the-null-costs-nothing)
- [33.119 Item 1 prepared: the tokenizer's gain at width 1024](15_the_list_of_19_october.md#33119-item-1-prepared-the-tokenizers-gain-at-width-1024)
- [33.120 Item 1's 10k half: at 85M parameters the gain halves on average, because two seeds of four lose it early](15_the_list_of_19_october.md#33120-item-1s-10k-half-at-85m-parameters-the-gain-halves-on-average-because-two-seeds-of-four-lose-it-early)
- [33.121 MultiBLiMP at width 1024, and a finer look at G-81 (prepared, 20 October)](15_the_list_of_19_october.md#33121-multiblimp-at-width-1024-and-a-finer-look-at-g-81-prepared-20-october)
- [33.122 MultiBLiMP at width 1024: the gain holds at every seed and stops growing; G-81's fall is no spike](15_the_list_of_19_october.md#33122-multiblimp-at-width-1024-the-gain-holds-at-every-seed-and-stops-growing-g-81s-fall-is-no-spike)
- [33.123 The chain of 21 October prepared: MultiBLiMP at 20k, items 3 and 6](15_the_list_of_19_october.md#33123-the-chain-of-21-october-prepared-multiblimp-at-20k-items-3-and-6)
- [33.124 Item 1's 20k half: at 85M parameters the gain is smaller again and lasts, and the split stays](15_the_list_of_19_october.md#33124-item-1s-20k-half-at-85m-parameters-the-gain-is-smaller-again-and-lasts-and-the-split-stays)
- [33.125 Item 2 prepared: gender's and number's leak is spread over word classes, so their control is a causal tagger](15_the_list_of_19_october.md#33125-item-2-prepared-genders-and-numbers-leak-is-spread-over-word-classes-so-their-control-is-a-causal-tagger)
- [33.126 Items 3 and 6: English part of speech's deficit closes by 100k, and at 30M parameters Unigram's shortfall goes](15_the_list_of_19_october.md#33126-items-3-and-6-english-part-of-speechs-deficit-closes-by-100k-and-at-30m-parameters-unigrams-shortfall-goes)
- [33.127 Item 2: gender's and number's lead over the word form is small, and little of it is the leak](15_the_list_of_19_october.md#33127-item-2-genders-and-numbers-lead-over-the-word-form-is-small-and-little-of-it-is-the-leak)
- [33.128 G-82 prepared: the causal tagger's rest, split by the word it describes](15_the_list_of_19_october.md#33128-g-82-prepared-the-causal-taggers-rest-split-by-the-word-it-describes)
- [33.129 G-82: what the causal tagger adds over the word form is not context](15_the_list_of_19_october.md#33129-g-82-what-the-causal-tagger-adds-over-the-word-form-is-not-context)
- [33.130 The text's request of 22 October, and G-81's recipe question prepared](15_the_list_of_19_october.md#33130-the-texts-request-of-22-october-and-g-81s-recipe-question-prepared)
- [33.131 Item 3 prepared: gender and number at full training](15_the_list_of_19_october.md#33131-item-3-prepared-gender-and-number-at-full-training)
- [33.132 Item 1: with a longer warm-up the gain at width 1024 holds at every seed](15_the_list_of_19_october.md#33132-item-1-with-a-longer-warm-up-the-gain-at-width-1024-holds-at-every-seed)
- [33.133 Item 3: at full training gender's and number's lead over the word form is small, and not context](15_the_list_of_19_october.md#33133-item-3-at-full-training-genders-and-numbers-lead-over-the-word-form-is-small-and-not-context)
- [33.134 Item 2 prepared: English part of speech at full training](15_the_list_of_19_october.md#33134-item-2-prepared-english-part-of-speech-at-full-training)
- [33.135 Item 2: at full training English part of speech's causal tagger keeps nothing reliable over the word form](15_the_list_of_19_october.md#33135-item-2-at-full-training-english-part-of-speechs-causal-tagger-keeps-nothing-reliable-over-the-word-form)
- [33.136 The text's request of 24 October, and the empty slot at width 1024 prepared](15_the_list_of_19_october.md#33136-the-texts-request-of-24-october-and-the-empty-slot-at-width-1024-prepared)
- [33.137 Item 1 prepared: width 1024 without repeating the text](15_the_list_of_19_october.md#33137-item-1-prepared-width-1024-without-repeating-the-text)
- [33.138 Item 3: the work's size, and a term in the Ukrainian figures](15_the_list_of_19_october.md#33138-item-3-the-works-size-and-a-term-in-the-ukrainian-figures)
- [33.139 Item 2: at width 1024 the tokenizer's gain over the empty slots is smaller than between the nulls](15_the_list_of_19_october.md#33139-item-2-at-width-1024-the-tokenizers-gain-over-the-empty-slots-is-smaller-than-between-the-nulls)
- [33.140 The queue reviewed: full training at 30M parameters, and English at 30M, prepared](15_the_list_of_19_october.md#33140-the-queue-reviewed-full-training-at-30m-parameters-and-english-at-30m-prepared)
- [33.141 Item 1: without repeating the text the gain at width 1024 is smaller, and still rising](15_the_list_of_19_october.md#33141-item-1-without-repeating-the-text-the-gain-at-width-1024-is-smaller-and-still-rising)
- [33.142 Full training at 30M parameters: the tokenizer's gain is smaller and lasts, case stays within the bound, English gains nothing](15_the_list_of_19_october.md#33142-full-training-at-30m-parameters-the-tokenizers-gain-is-smaller-and-lasts-case-stays-within-the-bound-english-gains-nothing)
- [33.143 The text's list of limits of 30 October: the learning rate, the plain baselines, the warm-up at width 128 and width 1024 at full training, prepared](15_the_list_of_19_october.md#33143-the-texts-list-of-limits-of-30-october-the-learning-rate-the-plain-baselines-the-warm-up-at-width-128-and-width-1024-at-full-training-prepared)
- [33.144 The first tier on the thesis machine prepared: shuffled case, the switch to a constant, and where the cut falls](15_the_list_of_19_october.md#33144-the-first-tier-on-the-thesis-machine-prepared-shuffled-case-the-switch-to-a-constant-and-where-the-cut-falls)
- [33.145 The learning rate, the plain baselines and the warm-up at width 128: at the best of three rates the gain is smaller; without the slot it is larger](15_the_list_of_19_october.md#33145-the-learning-rate-the-plain-baselines-and-the-warm-up-at-width-128-at-the-best-of-three-rates-the-gain-is-smaller-without-the-slot-it-is-larger)
- [33.146 Items 7 and 13 prepared: the modern block and depth, after width 1024](15_the_list_of_19_october.md#33146-items-7-and-13-prepared-the-modern-block-and-depth-after-width-1024)
- [33.147 Item 6 prepared: minimal pairs for agreement in the noun phrase and for government](15_the_list_of_19_october.md#33147-item-6-prepared-minimal-pairs-for-agreement-in-the-noun-phrase-and-for-government)
- [33.148 Items 9, 10 and 11: the morpheme boundary matters, case's content acts, and the deficit is not the input's change](15_the_list_of_19_october.md#33148-items-9-10-and-11-the-morpheme-boundary-matters-cases-content-acts-and-the-deficit-is-not-the-inputs-change)
- [33.149 Items 2, 7 and 13: at 85M parameters the gain at full training is the 30M model's; it holds under a modern block and falls with depth](15_the_list_of_19_october.md#33149-items-2-7-and-13-at-85m-parameters-the-gain-at-full-training-is-the-30m-models-it-holds-under-a-modern-block-and-falls-with-depth)
- [33.150 Item 6: the tokenizer's grammatical gain covers agreement in the noun phrase and government](15_the_list_of_19_october.md#33150-item-6-the-tokenizers-grammatical-gain-covers-agreement-in-the-noun-phrase-and-government)
- [33.151 Item 5's intervals and the software's size: the tokenizer's gains hold at 95%, the features' bound is 0.003](15_the_list_of_19_october.md#33151-item-5s-intervals-and-the-softwares-size-the-tokenizers-gains-hold-at-95-the-features-bound-is-0003)
- [33.152 The test split and every vocabulary size: on held-out articles the gain is smaller; a larger vocabulary gains more](15_the_list_of_19_october.md#33152-the-test-split-and-every-vocabulary-size-on-held-out-articles-the-gain-is-smaller-a-larger-vocabulary-gains-more)
- [33.153 Item 8: on gold labels the tagger reads case where context decides](15_the_list_of_19_october.md#33153-item-8-on-gold-labels-the-tagger-reads-case-where-context-decides)

### Appendices

- [Appendix A — Decision Record](appendices.md#appendix-a--decision-record)
- [Appendix B — Section Mapping](appendices.md#appendix-b--section-mapping)
- [Appendix C — The thesis's list of 15 October and its outcome sentences](appendices.md#appendix-c--the-thesiss-list-of-15-october-and-its-outcome-sentences)
