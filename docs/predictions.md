# Every prediction and its outcome

> **Kind:** record — every prediction, written down before its result, with its outcome.

Each prediction was recorded in `docs/record/journey/README.md` **before** its result
existed, usually with the outcome that would make it wrong. This table
collects them all; the verdict and its evidence are in the section given.

**Tally:** 109 right, 8 half right, 74 wrong, 8 never tested (3 deferred, 2 parked as outside the thesis, 3 dropped by the sentence test), 1 withdrawn, 0 pending — of 200. The wrong ones carried
the information: the null's gain (P18), the shuffled control (P15), the
leak's absence (P11), the training-budget reframing (P61), the tokenizer
hypothesis (P119, P120), Unigram's early lead (P130), the wider model's larger
tokenizer gain (P135), where the exposure effect sits (P144–P146), and what the sentence's case adds
over the word form's (P148–P150), and how much of it the tagger's leak is (P164, P166), and what context adds inside the models (P184, P187).

| # | prediction | outcome | journey |
|---|---|---|---|
| P1 | Word structure does not move BLiMP agreement | right | §33.6 |
| P2 | Structure's anaphor-gender jump does not replicate | wrong | §33.6 |
| P3 | The tagger's right-context leak is small (95–99% agreement) | wrong — 91.8–92.6% | §33.6 |
| P4 | A word's majority feature bundle matches its in-context bundle 93–98% of the time | right — 94.08% | §33.6 |
| P5 | Part of speech helps within a factor of two of structure | right — tied | §33.6 |
| P6 | Lexical inflection helps less than structure (+0.002 to +0.008) | half right | §33.6 |
| P7 | Lexical inflection lifts determiner–noun agreement by ≥2 points | wrong | §33.6 |
| P8 | Structure plus inflection beats structure | wrong | §33.6 |
| P9 | Inflection's gain grows during training, like structure's | wrong — it fades | §33.6 |
| P10 | The structure headline replicates at seed 3 | right | §33.6 |
| P11 | Strictly causal part of speech keeps 50–80% of the tagger version's gain | wrong — 90%: no measurable leak | §33.9 |
| P12 | Structure plus part of speech beats both alone | not tested | §33.6 |
| P13 | Structure's anaphor gain appears in both directions of change (agreement) | wrong — a preference | §33.8 |
| P14 | Part of speech gains more on word-continuation targets than structure | wrong — reversed | §33.9 |
| P15 | Structure shuffled across words keeps under a third of structure's gain | wrong — all of it | §33.9 |
| P16 | Structure's gain at 100k steps exceeds its 50k gain | not tested (deferred) | §33.9 |
| P17 | Part of speech falls below structure at 100k steps | withdrawn | §33.9 |
| P18 | The null (same slot, no information) gains under +0.003 | wrong — +0.0076 | §33.10 |
| P19 | Frequency band keeps ≥2/3 of structure's gain | right on its number, wrong on its reason | §33.10 |
| P20 | Shuffled part of speech grows like structure | not tested (deferred) | §33.10 |
| P21 | A 32-dim slot gains ≥ +0.0106 | right | §33.11 |
| P22 | The null is within ±0.002 of structure at every seed | right — the headline is the slot | §33.11 |
| P23 | The gain rises from 8 to 32 dims and falls by 64 | partly right — nearly flat | §33.11 |
| P24 | The null reproduces structure's BLiMP profile | wrong by its own rule | §33.12 |
| P25 | The zero slot is within 0.003 of the null | right, at its boundary | §33.12 |
| P26 | Removing the word/punctuation signal keeps the null's gain | right | §33.12 |
| P27 | The empty slot (no position) keeps under half the zero slot's gain | right | §33.19 |
| P28 | Position only in the slot keeps at least the zero slot's gain | half right | §33.19 |
| P29 | At four seeds no feature beats the null by more than 0.002 | right | §33.30 |
| P30 | Structure added to a full-width embedding stays within ±0.002 of the baseline | wrong by its rule (+0.0046) | §33.14 |
| P31 | The null added the same way stays within ±0.002 of the baseline | badly wrong | §33.14 |
| P32 | Zeros added (the baseline's function) stay within ±0.002 of the baseline | wrong, just past its line | §33.15 |
| P33 | One learned vector added at every input gains ≥ +0.006 | right by its letter | §33.15 |
| P34 | Zeros added, seeds 1–3: within ±0.002 of the baseline at ≥2 of 3 | right | §33.16 |
| P35 | Additive family, seed 1: structure within ±0.002 of its control | wrong about structure | §33.19 |
| P36 | Additive family, seeds 2–3: structure within ±0.002 of its control | wrong about structure | §33.19 |
| P37 | Over 10% of Ukrainian case values depend on the right context | wrong — 5.0% | §33.21 |
| P38 | A form's majority case matches its in-context case under 85% | wrong — 89.74% | §33.24 |
| P39 | Contextual case beats its null by ≥ +0.002 (seed 0) | wrong — −0.0025 | §33.24 |
| P40 | Lexical case within ±0.002 of its null (seed 0) | wrong — −0.0034 | §33.24 |
| P41 | The fixed word rule makes every feature ≥99.9% constant within words | wrong — paragraph glue | §33.22 |
| P42 | The rebuilt pilot is ≥99.9% constant and ~11k tokens longer | right | §33.22 |
| P43 | Both position controls fall short of the zero slot at seeds 2–3, with stated shares | half right | §33.24 |
| P44 | Both case features below the null on average, neither above at >1 seed | narrowly wrong | §33.30 |
| P45 | Case leads its null at step 2,000 at every seed | right | §33.30 |
| P46 | The null beats the Ukrainian baseline at every seed, by ≥ +0.003 | direction right, size wrong | §33.30 |
| P47 | Contextual and lexical case within ±0.002 of each other | right | §33.30 |
| P48 | Beside a zero slot, case still trails its null by ≥0.0012 | wrong | §33.26 |
| P49 | Added to the full embedding, case trails its null by ≥0.001 | right, barely | §33.26 |
| P50 | The early help appears in every design | right | §33.26 |
| P51 | Beside a zero slot, case beats its null at seeds 1 and 2 | wrong | §33.27 |
| P52 | …and gains on mid-frequency tokens at both | wrong | §33.27 |
| P53 | A second empty slot costs ≥0.002 | wrong | §33.30 |
| P54 | A zero slot shrinks case's deficit on average | right in sign, within noise | §33.30 |
| P55 | Beside a zero slot, case within ±0.002 of its null over four seeds | right | §33.28 |
| P56 | Added case trails its null by ≥0.001 at seed 1 | wrong | §33.28 |
| P57 | At ~1.5M tokens case beats its null at ≥3 of 4 seeds by ≥ +0.002 | right | §33.31 |
| P58 | The gain is smaller at ~7.6M than at ~1.5M | right | §33.31 |
| P59 | English inflection leads early and ends below its null at seeds 2–3 | right | §33.32 |
| P60 | Lexical case keeps at least half of contextual case's little-data gain | half right restated — 66% at ~1.5M, 43% at ~7.6M against the constant null (65–73% as first recorded) | §33.32, §33.56 |
| P61 | All the data at 10k steps: case within ±0.002 of its null | wrong — +0.0044 | §33.32 |
| P62 | At ~38M tokens the gain lies between zero and +0.0051 | wrong by a hair — +0.0052 | §33.32 |
| P63 | At little data the null beats the baseline | right | §33.32 |
| P64 | All the data at 20k steps: the gain between +0.0044 and −0.0009 | right — +0.0012 | §33.32 |
| P65 | English part of speech at a 10k-step budget beats its null by ≥ +0.003 at seeds 0 and 1 | right — +0.0136 | §33.33 |
| P66 | Switched off at step 2,500, part of speech keeps less than half its 10k-budget gain | right — it keeps none (−19%, −7%) | §33.34 |
| P67 | Just after the switch (step 3,000), the model switched off at 2,500 trails the one switched off from the start (K = 0) | right — −0.0092 (−0.0294 at the switch) | §33.34 |
| P68 | Switched off at step 7,500, part of speech ends ≥0.003 below off-from-start | right — −0.0118 (4 seeds) | §33.38 |
| P69 | Switched on at step 5,000, it ends with ≥ half the kept-throughout gain over off-from-start | wrong — +0.0010, a ninth of the kept gain | §33.38 |
| P70 | English part of speech at a 10k budget beats its null by ≥ +0.003 at seeds 2 and 3 | right — +0.0129, +0.0119 | §33.38 |
| P71 | Switched off at 2,500, part of speech ends at or below off-from-start (≤ +0.0005) at seeds 2 and 3 | right — −0.0033, −0.0030 | §33.38 |
| P72 | Part of speech at 5k and 20k budgets lies within ±0.002 of the 50k run's gain at the same null BPB | right — worst deviation 0.0010 | §33.38 |
| P73 | Ukrainian case at a 20k budget ends between its 10k and 50k results at seeds 2 and 3 | right — +0.0008, +0.0027 | §33.38 |
| P74 | At seeds 2–3 each C1 cell (off at 7,500; on at 5,000) has the same sign of gain over off-from-start as at seeds 0–1 | right — every seed, both cells | §33.38 |
| P75 | English inflection at a 10k budget beats its null by ≥ +0.003 on average, positive at every seed | right — +0.0067 ± 0.0014, 4/4 | §33.38 |
| P76 | Ukrainian case at a 5k budget beats its null by more than its 10k-budget gain (+0.0044) | right — +0.0074 ± 0.0007 | §33.38 |
| P77 | English structure at a 10k budget stays within ±0.002 of its null at every seed | wrong — +0.0028 at seed 2 (mean +0.0008) | §33.38 |
| P78 | Frequency band and word-form part of speech each beat the null by ≥ +0.002 at a 10k budget | right — +0.0046 and +0.0099 | §33.38 |
| P79 | Switched on at 5,000 of a 20k budget, part of speech ends with less than half the kept gain over off-from-start | wrong — 63% (seed 0) | §33.40 |
| P80 | A 4× wider model's part-of-speech gain at a 10k budget is within ±0.002 of the base curve at the same null BPB | wrong — falls away early, worst −0.0111 | §33.40 |
| P81 | Switched on at 5,000 of a 20k budget, part of speech keeps more than a third of the kept gain at each of seeds 1–3 | right — 54%, 82%, 104% | §33.42 |
| P82 | The 256-wide model's gain at null BPB 1.55 is below +0.006 at each of seeds 1–3 | wrong — +0.0082 at seed 1 (mean +0.0036) | §33.42 |
| P83 | The 64-wide model's end gain at a 10k budget is at least the base curve's at the same null BPB | wrong — 0.0028 below the base curve | §33.42 |
| P84 | At a 10k budget the architecture effect is below +0.003 on average over seeds 0–3 | wrong — +0.0035 (seed 0 the outlier) | §33.42 |
| P85 | At a 20k budget it lies between the 10k-budget value and the 50k value (+0.0112) | right — +0.0078 | §33.42 |
| P86 | Switched off at 5,000 of a 20k budget, part of speech ends within ±0.0015 of off-from-start on average | wrong — −0.0042, every seed | §33.42 |
| P87 | The 256-wide model's part of speech ends a 20k budget at or below its null (≤ +0.0005) at each of seeds 1–3 | right — −0.0002, −0.0008, −0.0002 | §33.46 |
| P88 | The architecture effect at 256 wide and a 20k budget is positive at each of seeds 1–3 | wrong — −0.0033 at seed 2; the baseline varies 9× more than the null | §33.46 |
| P89 | At null BPB 1.55 each of four English features' 20k-budget gain is within ±0.001 of its 50k value | wrong — inflection +0.0020 (2 seeds); the others within 0.0007 | §33.46 |
| P90 | Over four seeds, inflection's 20k-budget gain at null BPB 1.55 is within ±0.001 of its 50k value | right — +0.0070 against +0.0069 | §33.49 |
| P91 | Over seeds 0–7, the 256-wide plain baseline's spread of final BPB is ≥ 4× the null's | not tested — parked: about the shared vector, outside the thesis | §33.46, §33.59 |
| P92 | Over seeds 0–7, the 256-wide null beats the plain baseline on average, at ≥ 6 of 8 seeds | not tested — parked: about the shared vector, outside the thesis | §33.46, §33.59 |
| P93 | Under rotary positions the architecture effect at a 20k budget is below +0.002 (seed 0) | wrong — +0.0073 under RoPE | §33.49 |
| P94 | Weaned to zero by step 5,000, part of speech ends < 0.0021 below off-from-start at a 20k budget (seed 0) | wrong — −0.0023 (half the deficit) | §33.49 |
| P95 | Part of speech as an auxiliary target ends ≥ +0.002 better than the shuffled target and the plain baseline (seed 0) | wrong — +0.0018 and +0.0014 | §33.49 |
| P96 | At null BPB ~1.65 the 256-wide null's deepest-layer part-of-speech accuracy is ≥ 2 points above the 128-wide null's | wrong — 1.3 points, the direction right | §33.49 |
| P97 | With its slot doubled to the base share, the 256-wide model's gain at null BPB 1.55 stays below +0.006 | right — +0.0024 | §33.49 |
| P98 | Over seeds 0–3 the real auxiliary target beats the matched shuffled one at 20k by ≥ +0.001, at ≥ 3 seeds | wrong — +0.0003 ± 0.0007 (2/4) | §33.50 |
| P99 | Over seeds 0–3 the architecture effect under RoPE is positive at every seed, mean ≥ +0.004 | right — +0.0057 ± 0.0006, 4/4 | §33.50 |
| P100 | Over seeds 0–3 weaning to 5,000 leaves at most 70% of the abrupt switch's deficit | right — 70% | §33.50 |
| P101 | Along 20k budgets the 256-wide null's part-of-speech accuracy exceeds the 128-wide's by ≥ 1 point at matched null BPB 1.75–1.45 | wrong — +0.8 to +1.3 points | §33.50 |
| P102 | Tiled ×2, the 256-wide model's gain at null BPB 1.55 stays below +0.006 at seed 1 | right — +0.0037 | §33.50 |
| P103 | Under rotary positions the zero slot keeps at least half the constant slot's benefit (≥ +0.0028 over the plain baseline; seed 0) | wrong — −0.0002 | §33.51 |
| P104 | Under rotary positions the flat slot gains ≥ +0.0037 over the plain baseline (seed 0) | right — +0.0076 | §33.52 |
| P105 | Under rotary positions one vector added to every input dim beats zeros added by ≥ +0.004 (seed 0) | right — +0.0093 | §33.52 |
| P106 | Under rotary positions the constant slot beats the zero slot by ≥ +0.004 at each of seeds 1–3 | wrong — +0.0001 at seed 1 (4-seed mean +0.0044) | §33.52 |
| P107 | At step 6,000 the model switched off at 5,000 reads part of speech ≥ 1 point less accurately than off-from-start (deepest layer, seed 4) | wrong — −0.3 points; level from then on | §33.52 |
| P108 | At step 2,000 the real target's layer-2 part-of-speech accuracy exceeds the shuffled target's by ≥ 1 point (seed 4) | right — +4.6 points | §33.52 |
| P109 | Over seeds 0–3 at 20k, word form, frequency band and structure each lie within ±0.001 of their 50k gain at null BPB 1.55 | right — all within 0.0004 | §33.52 |
| P110 | On its own fraction's codes the Ukrainian null scores better than recorded at every seed at ~38M tokens and at 151M with 5k, 10k and 20k budgets, and worse at ~1.5M | right — better by 0.0010–0.0014, worse by 0.0019, every seed | §33.56 |
| P111 | On the constant itself every Ukrainian null scores better than on its own codes, at every seed | right — by 0.0005 (151M) to 0.0039 (~1.5M) | §33.56 |
| P112 | For the 151M Ukrainian null at 50k that improvement is smaller than 0.001 | right — 0.0005 | §33.56 |
| P113 | For every English constant null it is smaller than 0.0005 | right — at most 0.0002 (71 runs) | §33.56 |
| P114 | Without the causal shift the flat slot under rotary positions keeps at least half its gain over the plain baseline (≥ +0.0038; seed 0) | right — +0.0068, 89% | §33.58 |
| P115 | Without the causal shift the flat vector added keeps at least half its gain over the plain baseline (≥ +0.0036; seed 0) | right — +0.0059, 82% | §33.58 |
| P116 | Re-run on the constant, each Ukrainian null ends within 0.0002 of the audit's re-score of its quarantined checkpoint | right — to the sixth decimal (one by 0.000003) | §33.58 |
| P117 | At step 2,000 the re-run 151M null evaluates 0.0002–0.0008 below its quarantined record at every seed re-run | right — 0.00038–0.00047 | §33.58 |
| P118 | Under each of the four new tokenizers case beats its null at the end of a 10k budget on ~38M tokens (seed 0) | right — +0.0017 to +0.0047 | §33.58 |
| P119 | Case gains less under Unigram 17k than under BPE 17k (seed 0) | wrong — +0.0034 against +0.0028 | §33.58 |
| P120 | Case gains more under BPE 32k than under BPE 4k (seed 0) | wrong — +0.0035 against +0.0047 | §33.58 |
| P121 | On the previous unit's tag (deepest layer) the 128- and 256-wide models line up along 20k budgets: gains within 0.0015 at every common accuracy level | wrong — up to 0.0035 apart (the current word's tag: 0.0063) | §33.59, §33.60 |
| P122 | At every matched null BPB 1.75–1.45 the wider null's lead on the previous unit's tag exceeds its lead on the current word's | right — +3.9 to +10.4 points against +0.8 to +1.3 | §33.59, §33.60 |
| P123 | At the end of a 20k budget the model switched off at 5,000 reads the previous unit's tag within 0.5 points of the never-fed model | wrong by its rule — +2.0 points, the other way | §33.59, §33.60 |
| P124 | At the end of its 20k budget the 128-wide null reads the previous unit's tag below 0.85 (deepest layer) | right — 0.619 | §33.59, §33.60 |
| P125 | BPE 17k that keeps endings apart reads 5–15% fewer characters per token than BPE 17k (3.05–3.41) | right — 3.21, 10.6% fewer | §33.61, §33.62 |
| P126 | Its null ends below BPE 17k's null at ~38M tokens and a 10k budget (seed 0) | right — 1.0714 against 1.0891 | §33.61, §33.65 |
| P127 | At null BPB 1.25 case's gain under it lies within 0.002 of the five tokenizers' mean (+0.0056) | right — +0.0040 (seed 0); +0.0051 over four seeds | §33.61, §33.65 |
| P128 | Over seeds 0–3, paired by seed, the endings-apart null ends below BPE 17k's by more than 0.001 on average | right — by 0.0174 ± 0.0014 (4/4) | §33.64, §33.65 |
| P129 | Case beats its own null under endings apart at all four seeds | right — +0.0048 ± 0.0006 (4/4) | §33.64, §33.65 |
| P130 | Over seeds 0–3, Unigram 17k's case gain at null BPB 1.40 lies within 0.003 of BPE 17k's | wrong — +0.0094 against +0.0045: the early lead is real | §33.64, §33.65 |
| P131 | Over seeds 0–3, case's gain exceeds its word form's by more than 0.001 under at least five of the six tokenizers | right — under all six, +0.0019 to +0.0033 | §33.64, §33.65 |
| P132 | At 20k steps the endings-apart null ends below BPE 17k's by more than 0.010 on average, at every seed | right — +0.0169 ± 0.0007 (4/4) | §33.67, §33.68 |
| P133 | The endings-apart lead at 20k lies within 0.004 of its 10k value (+0.0134 to +0.0214) | right — +0.0169 | §33.67, §33.68 |
| P134 | At width 256 (10k) the endings-apart null ends below BPE 17k's by more than 0.008 on average, at every seed | right — +0.0254 ± 0.0012 (4/4) | §33.67, §33.68 |
| P135 | The endings-apart lead at width 256 is smaller than at width 128 (+0.0174) | wrong — larger, +0.0254 | §33.67, §33.68 |
| P136 | Under BPE 17k, case's gain over its null at the end of a 20k budget is below +0.002 | right — +0.0010 ± 0.0008 (3/4) | §33.67, §33.76 |
| P137 | Under endings apart, case beats its null at 20k on average and at three seeds or more | right — +0.0038 ± 0.0008 (4/4) | §33.67, §33.76 |
| P138 | At 20k case gains more under endings apart than under BPE 17k by more than 0.001, paired by seed | right — +0.0029 ± 0.0016 (4/4) | §33.67, §33.76 |
| P139 | At 20k endings apart and case together end below BPE 17k's null by more than 0.015 | right — +0.0208 ± 0.0011 | §33.67, §33.76 |
| P140 | Under BPE 32k at seed 3 case beats its null by more than 0.0015 | right — +0.0027 (four seeds +0.0029 ± 0.0002) | §33.67, §33.81 |
| P141 | The endings-apart condition at fraction 0.2 holds 167.8M–171.2M training tokens | right — 169,569,389 | §33.67, §33.68 |
| P142 | Its tagging pass takes 3.5–5.5 h beside the GPU runs | right — 4 h 43 min | §33.67, §33.68 |
| P143 | Under endings apart, ending tokens carry a share of case's gain at least 1.5 times their share of the targets (10k) | right — 58% of the gain on 19.7% of the targets, 2.9 times | §33.69, §33.70 |
| P144 | More than half of case's extra gain under endings apart over BPE 17k at 10k comes from words the rule cuts | wrong — 25%; the extra is in uncut words | §33.69, §33.70 |
| P145 | More than half of case's extra gain under Unigram 17k over BPE 17k at 10k comes from words the rule cuts | wrong — 24% | §33.69, §33.70 |
| P146 | At 20k, more than half of the extra gain under endings apart comes from cut words, and within them more from ending tokens than from stem tokens | wrong — 37% from cut words (within them, endings over stems as predicted) | §33.69, §33.70 |
| P147 | Every re-scored model reproduces its recorded bits per byte within 0.00001 | right — within 1.9 × 10⁻⁸, 50 models | §33.69, §33.70 |
| P148 | Under endings apart, more than half of what case from the sentence adds over case from the word form on ending tokens sits where the described word is a preposition whose case in context differs from its form's | wrong by its rule — 46% (prepositions as a whole 71%) | §33.73, §33.74 |
| P149 | Under each tokenizer at 10k, targets where both models get the same input carry less than a fifth of the sentence's whole extra | wrong — 21%, 36%, 24% | §33.73, §33.74 |
| P150 | Under endings apart, words whose form gives no case (unseen in train) carry less than a quarter of the sentence's extra | wrong — 28% (on ending tokens 14%) | §33.73, §33.74 |
| P151 | At 151M tokens and 50k steps under BPE 17k, the preposition class's part of the sentence's extra is positive at three seeds or more | right — +0.0005, 4/4 | §33.73, §33.74 |
| P152 | At 151M tokens and 50k steps the endings-apart null ends below BPE 17k's at every seed, by more than 0.010 on average | right — +0.0165 ± 0.0008 (4/4) | §33.73, §33.76 |
| P153 | That lead lies within 0.004 of the 20k lead at ~38M (+0.0129 to +0.0209) | right — +0.0165 | §33.73, §33.76 |
| P154 | Along the 50k schedule the final lead is at least three quarters of the lead at step 10,000 | right — 99% of the lead at step 10,000 | §33.73, §33.76 |
| P155 | Under endings apart, case beats its null at 151M and 50k on average and at three seeds or more | right — +0.0030 ± 0.0009 (4/4) | §33.73, §33.76 |
| P156 | At 151M and 50k case gains more under endings apart than under BPE 17k by more than 0.002, paired by seed | right — +0.0044 ± 0.0011 (4/4) | §33.73, §33.76 |
| P157 | At 151M and 50k endings apart with case ends below BPE 17k's null by more than 0.012 | right — +0.0195 ± 0.0013 | §33.73, §33.76 |
| P158 | At width 256 and 20k the endings-apart null ends below BPE 17k's by more than 0.010 at every seed, and by more than +0.0169 on average | right — +0.0218 ± 0.0029, every seed above +0.017 | §33.73, §33.86 |
| P159 | The endings-apart lead at width 256 and 20k is smaller than at width 256 and 10k (+0.0254) | right — +0.0218 | §33.73, §33.86 |
| P160 | Every re-run Ukrainian null reproduces the audit's re-score of its quarantined checkpoint within 0.00001 where its best step is its twin's; at ~1.5M tokens, where the best step lies mid-schedule, it ends at or below that re-score (refined before any run) | right — 12 of 20 read, each as predicted; the other 8 dropped by the sentence test | §33.73, §33.74, §33.81–§33.83 |
| P161 | At width 256 and 10k, case under BPE 17k ends less than +0.002 above its null | not tested (dropped by the sentence test) | §33.73, §33.83 |
| P162 | At width 256 and 10k case gains more under endings apart than under BPE 17k by more than 0.001, paired by seed | not tested (dropped by the sentence test) | §33.73, §33.83 |
| P163 | Switched off at 5,000 of a 50k budget, English part of speech ends more than 0.002 below the model never fed it, at both seeds | not tested (dropped by the sentence test) | §33.73, §33.83 |
| P164 | Under endings apart, case with prepositions read from their form keeps less than half of case's lead over the word form (ucaseprep over ucaselex below +0.0016) | wrong — +0.0018 ± 0.0002: it keeps 57% | §33.74, §33.76 |
| P165 | Under endings apart, ucaseprep over ucaselex is above +0.0006 (a fifth of case's lead over the word form) | right — +0.0018 ± 0.0002 | §33.74, §33.76 |
| P166 | Under BPE 17k, ucaseprep over ucaselex is below +0.0011 (half of case's lead) | wrong — +0.0015 ± 0.0003: it keeps 71% | §33.74, §33.76 |
| P167 | Under endings apart, ucaseprep's gain over the null on ending tokens is below +0.0017 | right — +0.0012 ± 0.0002 | §33.74, §33.76 |
| P168 | Under endings apart, the preposition classes hold less than a fifth of ucaseprep's lead over ucaselex | right — none of it (+0.0000) | §33.74, §33.76 |
| P169 | Under endings apart at 151M and 50k, case's word form beats its null on average and at three seeds or more | right — +0.0016 ± 0.0009 (3/4) | §33.77, §33.81 |
| P170 | There, the word form gains more under endings apart than under BPE 17k by more than 0.002, paired by seed | right — +0.0036 ± 0.0010 (4/4) | §33.77, §33.81 |
| P171 | Under endings apart at 151M and 50k, case with prepositions read from their form beats its null on average and at three seeds or more | right — +0.0015 ± 0.0010 (3/4) | §33.77, §33.81 |
| P172 | It keeps more than half of case's gain there (more than +0.0015) | right on its letter only — 50.05%: half | §33.77, §33.81 |
| P173 | Case beats it there by more than +0.0005 on average | right — +0.0015 ± 0.0001 (4/4) | §33.77, §33.81 |
| P174 | On MultiBLiMP Ukrainian at 151M and 50k, the endings-apart nulls beat BPE 17k's by more than 0.5 points on average and at three seeds or more | right — +4.9 ± 0.4 (4/4) | §33.79 |
| P175 | Every null at 151M and 50k scores between 65% and 92% on MultiBLiMP Ukrainian | right — 70.7–71.8% and 75.4–77.0% | §33.79 |
| P176 | Under BPE 17k at 151M and 50k, case moves MultiBLiMP accuracy by less than 1 point on average | right — −0.2 ± 0.4 | §33.79 |
| P177 | At ~38M and 10k, endings apart beats BPE 17k on MultiBLiMP by more than 0.5 points at three seeds or more | right — +4.3 ± 0.3 (4/4) | §33.79 |
| P178 | At ~38M and 10k, BPE 4k beats BPE 32k on MultiBLiMP by more than 1 point on average | wrong — BPE 4k 1.0 ± 0.2 below BPE 32k (0/3) | §33.79 |
| P179 | At 151M and 50k, endings apart's MultiBLiMP advantage is largest in gender of the three phenomena | right — gender +5.9, person +4.5, number +4.0 | §33.79 |
| P180 | On MultiBLiMP pairs whose two sentences have the same number of tokens under both tokenizers, at 151M and 50k, endings apart beats BPE 17k by more than 2 points at every seed | right — +5.0 ± 1.1, every seed above 2 | §33.79 |
| P181 | On MultiBLiMP at width 256 (10k, ~38M), endings apart's advantage over BPE 17k is larger than at width 128 (+4.3 points) | right — +5.3 ± 0.3 (4/4) | §33.79 |
| P182 | On MultiBLiMP at 20k (~38M, width 128), endings apart's advantage lies within 1.5 points of its 10k value | right — +4.5 ± 0.5 (4/4) | §33.79 |
| P183 | Under both tokenizers at full training, a probe at the best layer reads case in context more accurately than the form's majority case does | wrong — 83.2% and 84.7% against the form's 84.1% and 83.8% | §33.84 |
| P184 | On words whose case in context is not their form's, the endings-apart models' best-layer accuracy exceeds BPE 17k's by more than 2 points, paired, at three seeds or more | wrong — reversed, −2.5 ± 0.5 (0/4) | §33.84 |
| P185 | For number and for gender, the endings-apart models' best-layer accuracy exceeds BPE 17k's at three seeds or more | right — +2.5 and +5.7 points, 4/4 each | §33.84 |
| P186 | On words whose case in context is not their form's, the best-layer accuracy exceeds that subset's majority class by more than 15 points under both tokenizers | wrong — 12.7 and 10.4 points | §33.84 |
| P187 | Read one unit later, case on the words where only context can be right rises by more than 5 points over its reading at the word, under both tokenizers | wrong — it falls by 8.5 and 7.0 points | §33.84 |
| P188 | Read one unit later, the endings-apart models' accuracy on those words is at least BPE 17k's, paired, at three seeds or more | wrong — −1.0 ± 0.8 (1/4) | §33.84 |
| P189 | At 10k the English endings-apart baselines end below BPE 17k's at four seeds of four | right — +0.0046 ± 0.0010 (4/4) | §33.85, §33.87 |
| P190 | By more than 0.005 BPB on average | wrong — +0.0046 | §33.85, §33.87 |
| P191 | The relative gain is under half the Ukrainian one at 10k (under 0.8%) | right — 0.30% | §33.85, §33.87 |
| P192 | On fiction the endings-apart nulls end below BPE 17k's by more than 0.010 at four seeds of four | wrong — +0.0097 ± 0.0010, above 0.010 at 2/4 (below BPE 17k at 4/4) | §33.85, §33.88 |
| P193 | The relative gain on fiction lies within a factor of 1.5 of ukwiki's 1.6% (1.07–2.4%) | wrong — 0.81% | §33.85, §33.88 |
| P194 | On MultiBLiMP the fiction endings-apart nulls beat BPE 17k's by more than 2 points at four seeds of four | right — +5.3 ± 0.6 (4/4) | §33.85, §33.88 |
| P195 | On English BLiMP the endings-apart baselines beat BPE 17k's by more than 0.5 points on average | wrong — +0.3 ± 0.3 (determiner–noun agreement +4.4, 4/4) | §33.85, §33.87 |
| P196 | At 20k the English endings-apart baselines end below BPE 17k's at four seeds of four | right — +0.0055 ± 0.0008 (4/4) | §33.87 |
| P197 | Their lead at 20k lies within 0.002 of the 10k lead (+0.0026 to +0.0066) | right — +0.0055 | §33.87 |
| P198 | At 50k the English endings-apart baselines end below BPE 17k's at four seeds of four | wrong — three of four (seed 3 −0.0003) | §33.90, §33.92 |
| P199 | The median of their four paired leads at 50k lies within 0.002 of the 20k lead (+0.0035 to +0.0075) | right — +0.0053 | §33.90, §33.92 |
| P200 | On BLiMP's determiner–noun agreement the 50k endings-apart baselines beat BPE 17k's at four seeds of four | wrong — +3.0 ± 1.3, three of four (seed 0 −0.3) | §33.90, §33.92 |
