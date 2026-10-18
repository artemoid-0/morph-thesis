# Results — every number, with run ids

> **Kind:** record — the numbers, with the run ids behind them. **Authoritative for:**
> numbers; §0 holds the headline figures, superseded rows marked, never deleted. For what
> they mean, [`docs/FINDINGS.md`](../FINDINGS.md); for how they were reached, the
> [journey](journey/README.md).

*Until 1 October 2026 this was `results_reference.md`, which also held the
apparatus (§1, §2, §7 → [`docs/reference/apparatus.md`](../reference/apparatus.md)),
a script inventory (§6 → [`scripts/README.md`](../../scripts/README.md)), the
literature numbers (§8 → [`docs/reference/literature_analysis.md`](../reference/literature_analysis.md))
and the bug record (§9 → [`docs/record/bugs.md`](bugs.md)). Section numbers are kept, so
citations such as "results §5.36" still resolve.*

## 0. Current headline — read this first

The numbers the thesis stands on. Everything after this section is the full
chronological record; where a later figure supersedes an earlier one, the
earlier is marked, not deleted. **Feature claims are the rows against the
same-shape null; rows "over the baseline" from before §33.10 measure
architecture and information together.** `docs/FINDINGS.md` states each claim as
it now stands.

| claim | figure | n | where |
|---|---|---|---|
| Word structure over the baseline, causally | 0.0114 BPB (0.85%), 4/4 seeds — **matched by the null below: an architecture effect** | 4 paired | §5.34, §5.36 |
| **The null** (`mconst`: the same 8-dim slot, no information) over the baseline | +0.0111 ± 0.0025 paired by seed — but the baseline's start differs; against **eight starts of the baseline's function: +0.0097 ± 0.0015 (0.72%)**, no overlap | 4 + 8 | §5.36, journey §33.16 |
| **Structure over the null** — its information effect | **+0.0004 ± 0.0005**, 2/4 seeds same sign: **none** | 4 paired | §5.36 |
| Structure **added** to a full-width embedding (seed 0) | +0.0046 — but the **added null** gains +0.0094: structure is 0.0048 below its null | 1 | journey §33.14 |
| **Additive family**, gain over the matched control, 4 seeds | two vectors +0.0090 ± 0.0014, flat vector +0.0085 ± 0.0022, structure +0.0073 ± 0.0023 — structure −0.0013 against the flat null | 4 matched | journey §33.19 |
| **Position test, 4 seeds**, against the zero slot | no position in the slot −0.0082 ± 0.0012, no position in the token dims −0.0062 ± 0.0022 (0/4 better, both); zero slot over baseline +0.0105 ± 0.0035 | 4 matched | journey §33.24 |
| **Shuffled structure, 4 seeds** | over the null +0.0015 ± 0.0005 (4/4); real structure over shuffled −0.0012 ± 0.0008; zero slot over the null −0.0006 ± 0.0013 | 4 matched | journey §33.24 |
| **Early help**, feature minus matched null at step 2k vs end | English pos +0.0132/+0.0170 → −0.0004/+0.0008; poslex +0.0097/+0.0143 → +0.0023/−0.0037; inflection +0.0085/+0.0102 → −0.0074/−0.0051; mfreq +0.0038/+0.0025 → +0.0004/+0.0009; structure ≈ 0 → ≈ 0; Ukrainian case +0.006…+0.011 → −0.0025…+0.0035 | 2–7 | journey §33.28 |
| **G-65, Ukrainian**: case minus its matched null *(G-65 closed: not established)* | one slot −0.0025 (seed 0); **beside a zero slot +0.0011 ± 0.0011 (seeds 0–3: +0.0017, +0.0012, −0.0018, +0.0035)**; additive −0.0006 ± 0.0007 (seeds 0–1) | 1–4 | journey §33.26–§33.28 |
| **English part of speech at a 10k-step budget** *(superseded at four seeds: +0.0130 ± 0.0004, the D1 row; journey §33.38)* | over its null +0.0140, +0.0133: **+0.0136 ± 0.0004** (2/2); runs 3e7d5e2e10b8, c02a49956d0f, a1ca381fcca2, 0d863918f748 | 2 | journey §33.33 |
| **Six Ukrainian tokenizers measured** (`tokenizer_stats_ukwiki_20261007T040311`; 3,000 held-out paragraphs) | characters per token: BPE 4k 2.73, 8k 3.13, 17k 3.59, 32k 3.96, Unigram 17k 3.07, BPE 17k with endings apart 3.21; ending exposure at the rule's cut 20.2%, 13.9%, 8.8%, 5.5%, 43.3%, 100.0% (common prefix: 21.5%, 14.5%, 9.1%, 5.7%, 41.1%, 80.3%). Ending rule against the dictionaries (`ending_gold_20261007T041142`): Polyga 82.7% of words cut, 99.7% of cuts on a morpheme boundary; Yatsenko 73.5%, 97.9% | — | journey §33.62 |
| **The sixth condition prepared** (results_73): BPE 17k with endings apart at 0.05 | 42,600,731 training tokens, 363,158 validation (BPE 17k: 37,972,206 and 325,867 — +12.2%, +11.4%) for the same 235,673,205 bytes; 147,694 validation words; aligned 99.831% / 99.794%; lexicon 1,138,979 word types, majority case matches context 90.02%; 5.59% of validation words unseen in train; every gate passed; tagging 58 min (the other four at 0.05: 1 h 14 min – 1 h 31 min beside a GPU job). Words after a double space: 356–358 validation word tokens without part of speech under BPE 4k, 8k, 32k and this one | — | journey §33.63 |
| **Morphology in the tokenizer** (results_74): BPE 17k with endings apart against plain BPE 17k, null against null, paired by seed (same initial weights; ~38M tokens, 10k) | **+0.0174 ± 0.0014 BPB (1.6%)**, 4/4 — nulls 1.0714, 1.0715, 1.0739, 1.0722 against 1.0891, 1.0910, 1.0874, 1.0911; with case on top, +0.0222 ± 0.0017 below BPE 17k's null; Unigram 17k against BPE 17k −0.0005 ± 0.0004 | 4 paired | journey §33.65 |
| **Every tokenizer's stream the same text** (G-75; `stream_roundtrip_20261007T211736`) | validation split at 0.05: BPE 4k, 8k, 17k and 32k, Unigram 17k and endings apart (423,931 to 296,325 tokens) each decode to the 1,989,866 bytes recorded at build time, all six to one SHA-256 | 6 tokenizers | journey §33.66 |
| **The tokenizer at twice the budget** (the chain of 8 October; `tokenizer_table_20261008T211258`): BPE 17k with endings apart against BPE 17k, null against null, ~38M tokens, 20k steps, paired by seed | **+0.0169 ± 0.0007** (4/4) — nulls 0.9974–0.9978 against 1.0132–1.0161; along training +0.0454 at step 500, +0.0173 at 3,000, +0.0154 to +0.0170 from 5,000 to the end, while BPE 17k's null falls from 1.21 to 1.015 | 4 paired | journey §33.68 |
| **The tokenizer in a 4× wider model** *(twice as wide: width 256 against 128, four times the core's parameters; journey §33.91)* (`tokenizer_table_20261008T211300`): the same pair at width 256, 10k steps | **+0.0254 ± 0.0012** (4/4), 2.6% — nulls 0.9590–0.9605 against 0.9826–0.9879; at matched null BPB +0.0383 (1.18), +0.0336 (1.11), +0.0296 (1.04), +0.0267 (1.006); +0.0254 from step 8,000 to the end | 4 paired | journey §33.68 |
| **Case on top at 20k** (seeds 0–2; seed 3 left) *(superseded by the four-seed row)* | under BPE 17k +0.0005 ± 0.0008 (2/3), under endings apart +0.0041 ± 0.0011 (3/3); paired difference +0.0037 ± 0.0019 (3/3); the two together +0.0205 ± 0.0015 below BPE 17k's null; on the law's axis, BPE 17k +0.0007 at null BPB 1.02, endings apart +0.0042 at 1.00 | 3 paired | journey §33.68 |
| **Case on top at 20k, four seeds** (`tokenizer_table_20261009T143907`) | under BPE 17k +0.0010 ± 0.0008 (3/4), under endings apart **+0.0038 ± 0.0008** (4/4); paired difference +0.0029 ± 0.0016 (4/4); the two together +0.0208 ± 0.0011 below BPE 17k's null; on the law's axis BPE 17k +0.0012 at null BPB 1.02, endings apart +0.0043 at 1.02 and +0.0039 at 1.00 | 4 paired | journey §33.76 |
| **The tokenizer at full training, on 4× the text** (the chain of 9 October; `tokenizer_table_20261009T143624`): BPE 17k with endings apart against BPE 17k, null against null, 151M tokens, 50k steps, paired by seed | **+0.0165 ± 0.0008 (1.75%)**, 4/4 — nulls 0.9266, 0.9286, 0.9286, 0.9269 against 0.9419, 0.9464, 0.9437, 0.9448; along training +0.0456 at step 500, +0.0252 at 2,000, +0.0190 at 3,000, +0.0161 to +0.0176 from 5,000 to the end, while BPE 17k's null falls from 1.126 to 0.944 | 4 paired | journey §33.76 |
| **Case on top at full training** (the same table): case over each tokenizer's null, 151M tokens, 50k steps | under endings apart **+0.0030 ± 0.0009** (4/4; +0.0010, +0.0049, +0.0042, +0.0019), under BPE 17k −0.0013 ± 0.0008 (1/4); paired difference +0.0044 ± 0.0011 (4/4); the two together +0.0195 ± 0.0013 below BPE 17k's null; on the law's axis under endings apart +0.0046 at null BPB 1.00, +0.0033 at 0.94, under BPE 17k −0.0010 at 1.00, −0.0014 at 0.94 | 4 paired | journey §33.76 |
| **Where case's full-training gain sits** (`loss_by_kind_20261009T143815`; 16 models re-scored within 9.4 × 10⁻⁹): case over its null by the word the feature describes, 151M, 50k | endings apart: after prepositions +0.0009 (case in context not the form's) and +0.0008 (the same), 4/4 each; another word, case not the form's +0.0003; unseen +0.0000; the same input +0.0010 ± 0.0008 (3/4); on ending tokens +0.0021 ± 0.0001 of it. BPE 17k: prepositions +0.0005 and +0.0005; the same input −0.0023 ± 0.0007 (0/4) | 4 paired | journey §33.76 |
| **G-38's control in trained models** (the short block of 9 October; `loss_by_kind_20261009T143714`; 32 models re-scored within 1.7 × 10⁻⁸): case with prepositions read from their form (`ucaseprep`), ~38M, 10k | over the null: BPE 17k +0.0026 ± 0.0007, endings apart +0.0035 ± 0.0009 (4/4 each); case over it +0.0006 ± 0.0002 and +0.0014 ± 0.0004 (4/4) — 29% and 43% of case's lead over the word form; it over the word form +0.0015 ± 0.0003 and +0.0018 ± 0.0002 (4/4): after prepositions +0.0000, after words unseen in train +0.0008 and +0.0009, another word whose case differs +0.0002 and +0.0003, the same input +0.0004 and +0.0005; on ending tokens under endings apart +0.0012 ± 0.0002; against BPE 17k, endings apart +0.0008 ± 0.0012 (3/4) at the end, +0.0035 ± 0.0015 (4/4) at null BPB 1.15 | 4 paired | journey §33.76 |
| **The reproduction check** (the long chain of 9 October, piece 0): BPE 17k's null at 151M tokens and 5k steps, seed 0, re-run on the constant | 1.1961031 against the audit's re-score of the quarantined checkpoint, 1.196103 (`codes_audit_20261006T151153`); best step 4,999 both | 1 | journey §33.76 |
| **A second Ukrainian corpus: UberText fiction** (`long_2026-10-11.sh` piece 2; `tokenizer_table_20261011T151443`, `multiblimp_eval_20261011T151505`): the nulls of `bpe_ukf_17k` and `bpeend_ukf_17k` (trained on fiction at 0.4), fraction 0.1 (33.4M BPE tokens), 10k, holdout 500, paired by seed from identical initial weights | BPE 17k 1.2023, 1.1999, 1.2025, 1.2037 (1.2021); endings apart 1.1908, 1.1927, 1.1935, 1.1924 (1.1924); **+0.0097 ± 0.0010 (0.81%, 4/4)** — +0.0116, +0.0071, +0.0113, +0.0090; along training +0.042 at step 500, +0.025 at 1,000–3,000, +0.014 at 5,000, +0.0099 at 7,500, +0.0098 at 9,500. MultiBLiMP: 79.6% and **84.9%, +5.3 ± 0.6 (4/4)** — number +3.3, gender +5.5, person +6.0 (4/4 each). Exposure on fiction: 9.6% and 73.6% of inflected words; 11.8% more tokens; streams the same 4,594,258 bytes. The corpus: 25,565 texts, 1.25 billion characters (`ubertext_fiction_20261011T074342`) | 4 paired | journey §33.85, §33.88 |
| **English with endings apart** (`long_2026-10-11.sh` piece 1; tokenizer `bpeend_17k`, rule `en_endings_v1`): plain baselines on WikiText-103, 10k steps, against BPE 17k's plain baselines at 10k (3 October); paired by seed, initial weights not shared (vocabularies 17,000 and 17,151) | endings apart 1.5246, 1.5253, 1.5233, 1.5218 (1.5238); BPE 17k 1.5266, 1.5298, 1.5289, 1.5281 (1.5283); **+0.0046 ± 0.0010 (0.30%, 4/4)** — +0.0019, +0.0045, +0.0056, +0.0063; every endings-apart run below every BPE 17k run. The rule on validation text: 10.2% of words cut, 89.6% of them inflected in context, 49.6% of inflected words reached; the tokenizer: stem|ending boundary at 58.9% of inflected words (BPE 17k 3.8%), 4.4% more tokens; streams the same 1,136,862 bytes (`stream_roundtrip_20261011T074937`) | 4 paired | journey §33.85, §33.87 |
| **English with endings apart at 20k** (`long_2026-10-11.sh` piece 3): plain baselines at 20k against BPE 17k's (3 October), paired by seed | endings apart 1.4280, 1.4286, 1.4275, 1.4293 (1.4284); BPE 17k 1.4325, 1.4342, 1.4351, 1.4335 (1.4338); **+0.0055 ± 0.0008 (0.38%, 4/4)** — +0.0045, +0.0056, +0.0076, +0.0042; every endings-apart run below every BPE 17k run | 4 paired | journey §33.87 |
| **English with endings apart on BLiMP** (`blimp_eval_20261011T133856`; 67 paradigms, plain baselines at 10k, paired by seed) | all pairs 56.7% → 57.1% (+0.3 ± 0.3, 3/4); morphology +1.2 ± 0.8 (3/4); determiner–noun agreement 61.4% → 65.8% (**+4.4 ± 0.9**, 4/4); subject–verb agreement 54.9% → 53.3% (−1.6 ± 0.4, 4/4); anaphor agreement 68.8% → 66.8% (−2.0 ± 1.0, 4/4) | 4 paired | journey §33.87 |
| **English with endings apart at full training** (`long_2026-10-11.sh` piece 4; `blimp_eval_20261012T043938`): plain baselines at 50k against BPE 17k's (3 October), paired by seed | endings apart 1.3426, 1.3428, 1.3400, 1.3481 (1.3434); BPE 17k 1.3477, 1.3482, 1.3573, 1.3478 (1.3503); +0.0051, +0.0055, +0.0173, −0.0003: **median +0.0053 (0.39%)**, mean +0.0069 ± 0.0037 (0.51%, 3/4); 14 of 16 cross pairs; the median lead +0.0048 to +0.0055 from step 10,000 to the end. One run per tokenizer drifts from ~20k (BPE 17k seed 2 ends 0.009, endings apart seed 3 0.005 above its siblings); no spike above 0.0007 after 10k. BLiMP 59.0% → 60.0%, **+1.0 ± 0.2 (4/4)**; morphology +2.1 ± 0.8 (4/4); determiner–noun +3.0 ± 1.3 (3/4), subject–verb +1.3 ± 0.7 (3/4), anaphor +4.4 ± 2.9 (3/4); mean margin 0.02–0.11 nats lower (4/4) | 4 paired | journey §33.92 |
| **Width 256 at a 20k budget** (`tokenizer_table_20261011T125318`; 11 October, `long_2026-10-10.sh` piece 4): the nulls of BPE 17k and endings apart, ~38M tokens, paired by seed from identical initial weights | BPE 17k 0.9000, 0.9110, 0.9069, 0.8987 (0.9041); endings apart 0.8812, 0.8806, 0.8864, 0.8811 (0.8823); lead **+0.0218 ± 0.0029 (2.4%, 4/4)** — +0.0176, +0.0304, +0.0188, +0.0205. Along the schedule +0.0382 at step 2,000, +0.0276 at 5,000, +0.0229 at 7,500, +0.0234 at 10,000, +0.0224 at 15,000, +0.0220 at 19,500 | 4 paired | journey §33.86 |
| **Width 512** (`scale_2026-10-14.sh`; `tokenizer_table_20261014T094052`, `tokenizer_widths_20261014T094349`): the nulls of BPE 17k and endings apart at width 512 (30,017,760 parameters each), ~38M tokens, 10k, paired by seed from identical initial weights | BPE 17k 0.8803, 0.8790, 0.8770, 0.8777 (0.8785); endings apart 0.8664, 0.8655, 0.8672, 0.8649 (0.8660); **+0.0125 ± 0.0009 (1.4%, 4/4)** — +0.0139, +0.0136, +0.0099, +0.0128 (width 128 +0.0174, 1.6%; width 256 +0.0254, 2.6%). Along training +0.0452 at step 500, +0.0491 at 1,000, +0.0273 at 2,000, +0.0203 at 3,000, +0.0164 at 5,000, +0.0138 at 6,000, +0.0120 at 8,000 and 9,000. At matched null BPB 1.30 / 1.18 / 1.11 / 1.04 / 0.98 / 0.95 / 0.92 / 0.89: width 512 +0.0475 / +0.0408 / +0.0303 / +0.0218 / +0.0180 / +0.0163 / +0.0137 / +0.0121; width 256 at 20k +0.0320 / +0.0381 / +0.0333 / +0.0279 / +0.0223 / +0.0233 / +0.0228 / —; width 128 at 151M, 50k +0.0263 / +0.0178 / +0.0162 / +0.0159 / +0.0174 / +0.0168 / — / —. Validation BPB falls at every evaluation; every best step the last | 4 paired | journey §33.94 |
| **Width 512 at 20k, seed 0** (`scale_2026-10-14.sh`, `STEPS=20000 SEEDS=0`; `tokenizer_widths_20261014T193329`, `multiblimp_eval_20261014T193238`): the nulls of BPE 17k and endings apart at width 512, ~38M tokens, 20k | BPE 17k 0.8149, endings apart 0.8035: **+0.0115 (1.4%)**. Along training +0.0467 at step 1,000, +0.0249 at 2,000, +0.0172 at 5,000, +0.0117 at 7,500, +0.0132 at 10,000, +0.0094 at 12,500, +0.0119 at 15,000, +0.0107 at 17,500 — between +0.0101 and +0.0119 over the last 5,000 steps (−0.0004). At matched null BPB within 0.0006 of the 10k run's seed 0 down to 0.95, 0.0021–0.0025 below it at 0.92–0.89; +0.0105, +0.0129, +0.0094, +0.0109 at 0.88, 0.86, 0.84, 0.82. MultiBLiMP 77.3% → 82.9%, +5.6 (number +7.9, gender +6.5, person +3.9) | 1 paired | journey §33.96 |
| **Width 512 at 20k, seed 1** (`STEPS=20000 SEEDS=1`; `tokenizer_widths_20261015T184449`, `multiblimp_eval_20261015T184315`) | BPE 17k 0.8135, endings apart 0.8038: **+0.0097**; along training +0.0266 at step 2,000, +0.0163 at 5,000, +0.0100 at 7,500, +0.0110 at 10,000, +0.0084 at 12,500, +0.0094 at 15,000, +0.0083 at 17,500 — +0.0080 to +0.0103 over the last 5,000 steps (+0.0003). Seeds 0–1: +0.0106 (1.3%); at matched null BPB +0.0402 (1.18), +0.0206 (1.04), +0.0168 (0.95), +0.0121 (0.92), +0.0099 (0.88), +0.0097 (0.82). MultiBLiMP 76.9% → 83.3%, +6.4 (seed 0 +5.6) | 2 paired | journey §33.100 |
| **Width 512 at 20k, seeds 2–3** (a second machine, RTX 5090; `night_2026-10-16.sh` piece w512; runs 7c67de57b1be, f518b23725ad, 3836680f746a, 665fb7c3d0bd) | BPE 17k 0.8153, 0.8129; endings apart 0.8046, 0.8040: +0.0107, +0.0088; over the last 5,000 steps −0.0005 and +0.0006. Four seeds (0–1 on the thesis machine): **+0.0102 ± 0.0006 (1.25%, 4/4)**, every seed levelled off | 4 paired (2 per machine) | journey §33.102 |
| **English part of speech switched off at 5,000 of 50k** (the second machine; piece posoff) | against off-from-start at the end: +0.0011, −0.0022, −0.0036, −0.0031, **−0.0019 ± 0.0011** (3/4 below; 20k budget: −0.0039 ± 0.0009, 5/5); along training (mean) −0.0413 at 5k, −0.0110 at 6k, −0.0049 at 10k, −0.0038 at 20k, −0.0032 at 30k, −0.0024 at 40k; seed 0 level by 30k. Off from the start 1.3380, 1.3377, 1.3370, 1.3363; off at 5,000 1.3369, 1.3399, 1.3406, 1.3394 | 4 paired | journey §33.102 |
| **Ukrainian case switched off at 5,000 of 20k**, under endings apart, ~38M (the second machine; piece caseoff) | against off-from-start: +0.0022, +0.0012, +0.0003, −0.0007, **+0.0008 ± 0.0006** (3/4 above: no lasting deficit); −0.0239 at the switch (step 5,000), −0.0002 at 7,500, +0.0008 at 10,000, +0.0010 at 15,000. Off from the start 0.9956, 0.9932, 0.9940, 0.9931; off at 5,000 0.9935, 0.9920, 0.9937, 0.9938 | 4 paired | journey §33.102 |
| **English with endings apart at full training, seeds 4–7** (the second machine; piece english) | endings apart 1.3415, 1.3457, 1.3429, 1.3474; BPE 17k 1.3575, 1.3473, 1.3494, 1.3483; +0.0160, +0.0016, +0.0065, +0.0009 (+0.0062 ± 0.0035, 4/4). Eight seeds (0–3 on the thesis machine): **median +0.0053 (0.39%)**, mean +0.0066 ± 0.0024, 7/8 | 8 paired (4 per machine) | journey §33.102 |
| **The causal tagger over WikiText-103's training split** (`derive_causal_pos_20261016T001413`; the second machine, 69 min) | 120,625,719 tokens, the ids identical to the cache; the full pipeline reproduces the cached `pos` on the first 20,000 paragraphs (2,044,348 tokens); the causal tag equals `pos` at 92.09% of tokens (validation 92.22% of 252,255); codes copied with SHA-256 checks | — | journey §33.102 |
| **An empty slot in Ukrainian, across machines** (G-78): case switched off from the start — its slot at zero — under endings apart, ~38M, 20k (the second machine), against the null and case there (the thesis machine) | the empty slot 0.9956, 0.9932, 0.9940, 0.9931 (0.99398); the null 0.9974, 0.9976, 0.9978, 0.9974 (0.99756): **the empty slot 0.0036 better**; case 0.99373, 0.0002 better than the empty slot. English at 20k, one machine: part of speech off from the start 1.4252, the null 1.4261 (the empty slot 0.0008 better) | 4 + 4, unpaired, two machines | journey §33.102 |
| **Fiction at 0.4, the data** (the second machine; `ubertext_fiction_20261016T000257`, `check_cache_20261016T000349`, `…T000601`) | the download (433,084,760 bytes) and the conversion (2,264,291,319 bytes) SHA-256-identical to the thesis machine's; both validation streams built and checked; the training split stopped at paragraph 1,046,116 of 1,121,272 (92.5M tokens tagged): one line of 2,713,718 characters, past spaCy's 1M limit — the only one at 0.4 (next 587,473); the limit lifted to 4M (~3 GB per million characters, measured) | — | journey §33.102 |
| **The two machines: twins** (`device_twins_20261017T082521`; RTX 4060 Laptop, the thesis machine, against RTX 5090 Laptop; the same software, all fp32) | 15 runs repeated with the same arguments and seeds — the null at 10k (3), the null, case and the empty slot at 20k under endings apart (12): second minus thesis −0.0004 ± 0.0002, 6 above and 9 below (sign test p = 0.61), mean \|difference\| 0.0006, largest 0.0025; at step 0 within 2 × 10⁻⁹. Per cell the twins spread 0.2–4.5 times the thesis machine's seed spread (the null's four seeds lie within 0.0004). Case over the null +0.0038 ± 0.0008 (thesis) and +0.0037 ± 0.0008 (second), seed by seed within 0.0018. The second machine repeats bit for bit across days (the empty slot, 16 and 17 October) | 15 pairs | journey §33.106 |
| **G-78 at 20k: the empty slot** (`paired_cells_20261017T081844`; the null switched off from step 0, under endings apart, ~38M) | the thesis machine: the empty slot 0.9956, 0.9941, 0.9948, 0.9955 against the null's 0.9974, 0.9976, 0.9978, 0.9974: **+0.0025 ± 0.0004 (4/4)**; case over the empty slot +0.0019, +0.0027, −0.0005, +0.0010: **+0.0013 ± 0.0007 (3/4)**. The second machine: the empty slot over the null +0.0034 ± 0.0009 (4/4), case over it +0.0003 ± 0.0006 (2/4) | 4 paired, each machine | journey §33.106 |
| **Where the empty slot and case gain** (`loss_by_kind_20261017T081901`; the thesis machine's 12 models re-scored on the CPU within 1.1 × 10⁻⁶) | over the null, by target: a word's first token (uncut 21%, of a cut word 20%) the empty slot +0.0003, +0.0007, case +0.0006, +0.0007; the ending token (20%) the empty slot +0.0001 ± 0.0001, **case +0.0024 ± 0.0002 (4/4; 64% of its gain)**; the rest of cut words (8%) +0.0004, +0.0001; the rest of uncut words (11%) +0.0001, −0.0002; non-words (20%) **the empty slot +0.0009 ± 0.0001 (4/4)**, case +0.0003 | 4 paired | journey §33.106 |
| **Item 2: English part of speech from a causal tagger** (`paired_cells_20261017T103000`; `poscausal`, §33.99; the thesis machine; from `pos`'s initial weights) | 10k: 1.5159, 1.5141, 1.5143, 1.5130 — over the null **+0.0105 ± 0.0003 (4/4)**, over the word form **+0.0006 ± 0.0004 (3/4)**, `pos` over it (the leak) **+0.0025 ± 0.0004 (4/4)** of `pos`'s +0.0031 over the word form; 20k: 1.4217, 1.4202, 1.4208, 1.4199 — +0.0054 ± 0.0003 (4/4), **+0.0004 ± 0.0003 (3/4)**, **+0.0021 ± 0.0005 (4/4)** of +0.0025 | 4 paired | journey §33.107 |
| **English twins at full training** (`device_twins_20261017T103034`) | both plain baselines at seeds 0–1 on the second machine: BPE 17k 1.3481, 1.3473 (+0.0004, −0.0009); endings apart 1.3490, 1.3451 (+0.0063, +0.0023); the lead at seeds 0–1 −0.0009 and +0.0022 there, +0.0051 and +0.0055 on the thesis machine; seed 0's endings-apart twin +0.0010 at 5k, +0.0027 at 20k, +0.0063 at the end. All 19 twins +0.0001 ± 0.0004 (9/10); largest 0.0005 (10k), 0.0025 (20k), 0.0063 (50k) | 4 pairs | journey §33.107 |
| **The grammar tests of items 6 and 1** (`blimp_eval_20261017T094816`, `multiblimp_eval_20261017T094841`; the second machine) | BLiMP, English seeds 4–7 at 50k, endings apart over BPE 17k: +2.0, +0.6, +0.5, +1.0 points; eight seeds **+1.0 ± 0.2 (8/8)**, morphology +1.1 ± 0.6 (6/8). MultiBLiMP, width 512 at 20k, seeds 2–3: +5.6, +6.1; four seeds **+5.9 ± 0.2 (4/4)** | 4 paired each | journey §33.107 |
| **Fiction at full training** (`tokenizer_table_20261017T124204`; the second machine; both nulls at fraction 0.4, 153.9M training tokens under endings apart, 50k) | BPE 17k 1.0663, 1.0657, 1.0656, 1.0665; endings apart 1.0487, 1.0510, 1.0527, 1.0498: **+0.0155 ± 0.0010 (1.45%, 4/4)**; along training +0.0425 (500), +0.0254 (1k), +0.0145 (5k), +0.0112 (10k), +0.0153 (20k), +0.0149 (30k, 40k), +0.0155 (end) | 4 paired | journey §33.108 |
| **Why twins differ** (`dropout_masks_20261017T103028`, `…T134856`; `device_twins_20261017T135807`) | a matrix product, a softmax and a layer norm on the same inputs: bitwise identical on both GPUs; dropout's masks from the same seed: different on the two GPUs for every shape, the CPU's identical. BPE 17k's null at 10k with dropout off, seeds 0–1, on both machines: 1.052776441 / 1.052776431, 1.054259737 / 1.054259743 — at most 2 × 10⁻⁸ apart at any evaluation (with dropout: 1.0891, 1.0910). With dropout, 23 twins: +0.0001 ± 0.0003, 10 above and 13 below, mean \|difference\| 0.0009 | 2 + 23 pairs | journey §33.109 |
| **G-78 under BPE 17k at 20k** (the second machine) | the null's twins −0.0002, +0.0008, −0.0005, −0.0004 from the thesis machine's; the empty slot over the null +0.0005, 0.0000, +0.0002, +0.0012: **+0.0005 ± 0.0003** (under endings apart +0.0025 and +0.0034) | 4 paired | journey §33.109 |
| **MultiBLiMP on fiction at full training** (`multiblimp_eval_20261017T134827`) | BPE 17k 88.1%; endings apart **+2.3 ± 0.3 points (4/4)**: number +3.4 ± 0.3, gender +3.6 ± 0.6, person +0.7 ± 0.1 (4/4 each); at 10k +5.3 ± 0.6 | 4 paired | journey §33.109 |
| **G-78 at 10k** (`device_twins_20261018T033029`; the second machine, ~38M; the endings-apart null and case as twins of the thesis machine's, the BPE 17k nulls the calibration's twins) | the empty slot over the null: endings apart +0.0007, +0.0018, +0.0025, +0.0039, **+0.0022 ± 0.0007 (4/4)**; BPE 17k −0.0014, −0.0000, −0.0011, +0.0000, **−0.0006 ± 0.0004 (1/4)**. Case over the empty slot (endings apart) +0.0016, +0.0040, +0.0021, +0.0009, **+0.0022 ± 0.0007 (4/4)**; case over the null +0.0044 ± 0.0008 there, +0.0048 ± 0.0006 on the thesis machine | 4 paired | journey §33.110 |
| **G-78 at full training** (`paired_cells_20261017T142732`, `loss_by_kind_20261017T142734`; the thesis machine, 151M, 50k, endings apart) | the empty slot 0.9269, 0.9226, 0.9268, 0.9264 against the null's 0.9266, 0.9286, 0.9286, 0.9269: **+0.0020 ± 0.0014 (3/4)**; case over the empty slot +0.0013, −0.0011, +0.0023, +0.0014, **+0.0010 ± 0.0007 (3/4)**; case with prepositions from their form −0.0005 ± 0.0006 (1/4), case from the word form −0.0004 ± 0.0008 (2/4). Over the null by target: case on the endings +0.0021 ± 0.0001 (4/4, 68%), the empty slot there −0.0001 | 4 paired | journey §33.110 |
| **Item 5: the probes at five seeds** (24 probes on the second machine; runs repeated with a checkpoint every 1,000 steps, seeds 0–3; seed 4 the thesis machine's) | after removal, the word's tag at the deepest layer (switched minus never fed): −0.5 to −1.3 points at step 5,000, then within 0.3 of level from 6,000 to the end, every seed; the previous word's tag at the end +1.8, +2.4, +1.9, +1.7, +2.0: **+2.0 ± 0.1 (5/5)**. The auxiliary target at layer 2 (real minus shuffled): +5.1, +5.0, +4.3, +4.8, +4.6 at step 2,000 (**+4.8 ± 0.1**), +2.0, +1.5, +1.7, +1.5, +1.6 at the end (**+1.7 ± 0.1**, 5/5 each) | 5 | journey §33.111 |
| **Width 512's twins, seed 1** (the second machine) | BPE 17k's null +0.0005, endings apart's +0.0001 from the thesis machine's; the lead +0.0101 against +0.0097; the second machine's seeds 1–3 +0.0099 ± 0.0006 | 1 pair | journey §33.111 |
| **All twins** (`device_twins_20261018T033029`) *(superseded by the 131 pairs below)* | 52 runs repeated across the two machines (2 with dropout off): +0.0001 ± 0.0002, 27 above and 25 below, mean \|difference\| 0.0007; largest 0.0011 (10k), 0.0025 (20k), 0.0063 (50k); effects within each machine agree (case over the null +0.0048 / +0.0044 at 10k, +0.0038 / +0.0037 at 20k) | 52 pairs | journey §33.111 |
| **G-78 across tokenizers at 10k** (`device_twins_20261018T182550`; the second machine, ~38M; the null and case as twins of the thesis machine's, the empty slot new) | the empty slot over the null: Unigram 17k −0.0006, +0.0067, +0.0033, +0.0059, **+0.0039 ± 0.0016 (3/4)**; BPE 4k −0.0027, −0.0041, −0.0074, −0.0007, **−0.0037 ± 0.0014 (0/4)**; BPE 8k −0.0013, −0.0033, −0.0018, −0.0002, **−0.0016 ± 0.0006 (0/4)**; BPE 32k −0.0016, −0.0028, −0.0040, −0.0017, **−0.0025 ± 0.0006 (0/4)**. Case over the empty slot: Unigram 17k +0.0045, −0.0011, +0.0023, +0.0014, **+0.0018 ± 0.0012 (3/4)**; BPE 4k +0.0079, +0.0090, +0.0093, +0.0043, **+0.0076 ± 0.0012**; BPE 8k +0.0042, +0.0076, +0.0101, +0.0042, **+0.0065 ± 0.0014**; BPE 17k +0.0045, +0.0019, +0.0040, +0.0047, **+0.0038 ± 0.0007**; BPE 32k +0.0054, +0.0048, +0.0069, +0.0049, **+0.0055 ± 0.0005** (4/4 each). Case over the null: Unigram 17k +0.0056 ± 0.0007, BPE 4k +0.0039 ± 0.0007, 8k +0.0049 ± 0.0012, 17k +0.0032 ± 0.0006, 32k +0.0030 ± 0.0004 (4/4 each; the thesis machine +0.0054, +0.0044, +0.0040, +0.0032, +0.0029) | 4 paired each | journey §33.113 |
| **G-78 on less data** (`device_twins_20261018T182550`; the second machine, BPE 17k, 10k) | the empty slot over the null: ~1.5M −0.0014, −0.0022, −0.0022, −0.0069, **−0.0032 ± 0.0013 (0/4)**; ~7.6M −0.0014, −0.0003, −0.0026, −0.0017, **−0.0015 ± 0.0005 (0/4)**. Case over the empty slot +0.0097 ± 0.0005 and +0.0040 ± 0.0006; over the null +0.0065 ± 0.0015 and +0.0025 ± 0.0005 (4/4 each; the thesis machine +0.0076, +0.0031) | 4 paired each | journey §33.113 |
| **Item 7: case at full training under BPE 17k, without the leak** (`device_twins_20261018T182550`; the second machine, 151M, 50k) | case 0.9430, 0.9451, 0.9433, 0.9470; with prepositions read from their form 0.9435, 0.9459, 0.9437, 0.9475; from the word form 0.9422, 0.9475, 0.9441, 0.9466. Case over its word form +0.0005 ± 0.0007 (2/4; the thesis machine +0.0006 ± 0.0004, 3/4); the leak **+0.0006 ± 0.0001 (4/4)**; prepositions from their form over the word form **−0.0001 ± 0.0007 (2/4)**. The thesis machine's seed 0: 0.9454, the leak +0.0005, over the word form +0.0004 | 4 paired | journey §33.114 |
| **Item 8: the full-training table's two-seed rows, seeds 2–3** (`device_twins_20261018T182550`; the second machine, English, 50k) | the null 1.3432, 1.3388; the frequency band 1.3468, 1.3385 (−0.0036, +0.0003); part of speech from the word form 1.3439, 1.3370 (−0.0008, +0.0018). With seeds 0–1 from the thesis machine: the frequency band **−0.0005 ± 0.0011 (3/4)**, part of speech from the word form **−0.0001 ± 0.0014 (2/4)** | 2 + 2 paired | journey §33.114 |
| **The width probes at five seeds** (`probe_axis_20261018T182434`, `…T182436`; seeds 0–3 the second machine's twins with a checkpoint every 1,000 steps, seed 4 the thesis machine's; the gains over the thesis machine's seeds 0–3) | the largest gap between the widths' gains at a common level, 10k / 20k: on the previous word's tag **0.0036 ± 0.0003 / 0.0039 ± 0.0001** (per seed 0.0025–0.0042 / 0.0035–0.0043; on the averaged accuracy 0.0038 / 0.0037), on the word's own 0.0063 ± 0.0005 / 0.0068 ± 0.0005, on the null's BPB 0.0099 / 0.0098. Wider over base at matched null BPB 1.75–1.45: +2.5 to +10.4 points on the previous word's tag (each seed's largest 8.7–10.4), +0.4 to +1.5 on the word's own. End of 20k, deepest layer: base null 0.629 ± 0.004, 256 wide 0.732 ± 0.002 (the word's own tag 0.883 ± 0.001, 0.895 ± 0.001); the wider model at step 3,000 0.678–0.686 | 5 seeds | journey §33.114 |
| **All twins** (`device_twins_20261018T182550`) | 131 runs repeated across the two machines (2 with dropout off): −0.0001 ± 0.0001, 65 above and 66 below, mean \|difference\| 0.0010; largest 0.0031 (10k), 0.0044 (20k), 0.0063 (50k). The 36 new at 10k: −0.00001 ± 0.00018, four beyond 0.0015 (the largest BPE 4k's null, −0.0031) | 131 pairs | journey §33.114 |
| **What the models build for themselves** (`probe_words_20261011T070939`; `…T071514`, one unit later): a linear probe on each layer's state at a word's last token, labels the tagger's in context, the nulls at 151M, 50k; trained on even validation windows, tested on odd; best layer per run, paired by seed | case: BPE 17k 83.2%, endings apart 84.7% (+1.5 ± 0.3, 4/4) — where the form's majority case is right 87.2% and 89.6% (**+2.3 ± 0.4**, 4/4); where only context can be right 57.6% and 55.1% (**−2.5 ± 0.5**, 0/4; floor 44.9%, 44.7%); forms unseen in train 76.7% and 75.7%; the form's own majority case right for 84.1% and 83.8%. Number 91.8% and 94.3% (**+2.5 ± 0.1**, 4/4); gender 75.7% and 81.4% (**+5.7 ± 0.2**, 4/4). Best layer: 3 under BPE 17k, 1–2 under endings apart. One unit later: case 65.9% and 64.6%, the context words 49.1% and 48.1%, number 84.4% and 82.7%, gender 62.7% and 62.6% | 4 paired | journey §33.84 |
| **G-77: case at full training without the leak** (the chain of 10 October; `tokenizer_table_20261010T170427`, `…T170434`): over each cell's null, 151M tokens, 50k steps, under endings apart, paired by seed | case with prepositions read from their form +0.0015 ± 0.0010 (3/4; −0.0005, +0.0037, +0.0027, +0.0001); case's word form +0.0016 ± 0.0009 (3/4; −0.0006, +0.0034, +0.0026, +0.0010); case over the control — the leak — **+0.0015 ± 0.0001** (4/4); the control over the word form **−0.0001 ± 0.0003** (+0.0001, +0.0003, +0.0001, −0.0009); case over the word form +0.0014 ± 0.0002 (4/4; under BPE 17k +0.0006 ± 0.0004, 3/4); the word form's gain against BPE 17k's (−0.0020 ± 0.0010, 1/4) +0.0036 ± 0.0010 (4/4), with the tokenizer +0.0181 ± 0.0013 below BPE 17k's null. On the law's axis: the control over the word form +0.0011 to +0.0013 at null BPB 1.25–1.05, +0.0007 at 1.00, +0.0002 at 0.94; case over the control +0.0008 to +0.0020, +0.0014 at 0.94; the word form +0.0053, +0.0029, +0.0017 at 1.15, 1.00, 0.94 (BPE 17k +0.0026, −0.0020, −0.0020) | 4 paired | journey §33.81 |
| **Where G-77's gains sit** (`loss_by_kind_20261010T170915`; 16 models re-scored on the CPU within 1.1 × 10⁻⁶): over the null by what each target is, and case over the control by the word the feature describes; 151M, 50k, endings apart | case − control: after a preposition whose case in context differs from its form's +0.0009, one whose case agrees +0.0005 (4/4 each), another word +0.0000, unseen +0.0000, the same input +0.0001 — +0.0013 ± 0.0000 of it on ending tokens. By target: ending tokens case +0.0021 (68% of its gain), control +0.0008 (51%), word form +0.0005 (33%); stems of cut words +0.0005, +0.0007, +0.0007; uncut words' first tokens +0.0006, +0.0006, +0.0005; non-word targets +0.0004, +0.0001, +0.0003 | 4 paired | journey §33.81 |
| **The second reproduction check, and the re-runs** (the chain of 10 October, pieces 0 and 5): BPE 17k's nulls re-run on the constant, each against the audit's re-score of its quarantined twin (`codes_audit_20261006T151153`) | 151M, 5k: seed 1 (the check) 1.1936354, seeds 2 and 3 1.1940965 and 1.1943150 — each its twin's re-score to 10⁻⁷; ~1.5M, 10k (stopped early at 7,500–8,000 steps): seeds 0–2 1.2661331, 1.2635622, 1.2659286, the same; seed 3 1.2606262 against 1.2607126 — its best step 4,500, its twin's 5,000 | 6 | journey §33.81 |
| **The ~7.6M-token re-runs** (the hour of 10 October, piece 5): BPE 17k's nulls at ~7.6M tokens and 10k, re-run on the constant, against the audit's re-scores of their quarantined twins | seeds 0–3: 1.1037241, 1.1065840, 1.1042292, 1.1063754 — each its twin's re-score to 10⁻⁷, best step 9,999 both | 4 | journey §33.82 |
| **A grammatical test: MultiBLiMP's Ukrainian subject–verb agreement** (`multiblimp_eval_20261010T065127`, 151M; `…T065611`, ~38M; `…T070238`, by length): accuracy over 2,744 pairs, nulls paired by seed | 151M, 50k: BPE 17k 71.4%, endings apart **76.3%, +4.9 ± 0.4** (4/4) — number +4.0, gender +5.9, person +4.5 (4/4 each); on the 1,287 pairs of equal token length +5.0 ± 1.1 (4/4). ~38M, 10k: BPE 4k 63.5%, 8k 64.1%, 17k 64.8%, 32k 64.8% (3 seeds; *at four, 64.6%, −0.1 ± 0.2: `multiblimp_eval_20261010T171508`, journey §33.81*), Unigram 17k 64.7%, endings apart **69.0%, +4.3 ± 0.3** (4/4; number +8.3, gender +7.3, person −0.2). Case over its null: 151M −0.2 ± 0.4 (BPE 17k), +0.2 ± 0.5 (endings apart), word form +0.1 ± 0.4 (BPE 17k) and +0.2 ± 0.4 (endings apart, 3/4; `multiblimp_eval_20261010T171043`, journey §33.81); ~38M +0.4 ± 0.1, −0.3 ± 0.4 | 4 paired (32k: 3) | journey §33.79 |
| **The grammatical test along budget and width** (`multiblimp_eval_20261010T071828`, `…T072101`): nulls at ~38M | 20k: BPE 17k 68.5%, endings apart 73.0%, **+4.5 ± 0.5** (4/4); width 256, 10k: 69.7% against 75.0%, **+5.3 ± 0.3** (4/4; number +5.7, gender +7.1, person +3.7) | 4 paired | journey §33.79 |
| **The grammatical test at width 512** (`multiblimp_eval_20261014T094155`): the width-512 nulls, ~38M, 10k, paired by seed | BPE 17k 73.5%, endings apart 80.3%, **+6.8 ± 0.2** (4/4) — number +8.0 ± 0.5, gender +7.9 ± 0.6, person +5.3 ± 0.7 (4/4 each); at the same budget width 128 +4.3 ± 0.3, width 256 +5.3 ± 0.3 | 4 paired | journey §33.94 |
| **The endings-apart condition at 0.2** (the CPU track of 8 October) | 169,569,389 training tokens for the 938,814,912 bytes of BPE 17k's 151,133,566 (ratio 1.1220; 1.1219 at 0.05); tagging 4 h 43 min beside GPU runs; majority case equals the case in context for 89.75% of occurrences; validation words unseen in train 3.05% | — | journey §33.68 |
| **Where case's gain sits** (`loss_by_kind_20261008T212841`, 10k; `loss_by_kind_20261008T213019`, 20k): case over its null by what each target token is; 50 models re-scored to their records within 1.9 × 10⁻⁸ | tokens holding an ending's start: BPE 17k +0.0035 ± 0.0005 (108% of the gain on 22.0% of the targets), Unigram 17k +0.0032 ± 0.0002 (60%, 18.9%), endings apart +0.0028 ± 0.0002 (58%, 19.7%); cut words +0.0036, +0.0041, +0.0040; uncut words −0.0001, +0.0013, +0.0009 — of the extra over BPE 17k, cut words hold 25% (endings apart) and 24% (Unigram); at 20k under BPE 17k endings +0.0021 ± 0.0006, uncut words −0.0008, non-word targets −0.0007; under endings apart the extra +0.0037 ± 0.0019, 37% in cut words; word form on endings: BPE 17k +0.0021, endings apart +0.0005 | 4 paired (20k: 3) | journey §33.70 |
| **What the sentence's case adds over the word form's, by the word the feature describes** (G-38; `loss_by_kind_20261009T035142`, 10k at ~38M; `loss_by_kind_20261009T035358`, 151M at 50k, BPE 17k): case minus case's word form, each over the null, paired by seed; 48 models re-scored to their records within 1.7 × 10⁻⁸ | 10k, BPE 17k / Unigram 17k / endings apart — whole extra +0.0021 ± 0.0003 / +0.0033 ± 0.0001 / +0.0032 ± 0.0003; a preposition whose case in context is not its form's (1.9–2.0% of targets) +0.0007 / +0.0013 / +0.0013; a preposition, the same case (6.4–7.0%) +0.0004 / +0.0006 / +0.0006; another word, case not its form's (4.8–5.2%) +0.0002 each; a word unseen in train (2.8–3.2%) +0.0008 / +0.0007 / +0.0009, where the word form ends below the null (−0.0006 / −0.0005 / −0.0007, 0/4 each); the same input (83–84%) −0.0000 / +0.0006 / +0.0002 — prepositions hold 55% / 57% / 60% of the extra, 64% / 62% / 71% of it on ending tokens; every cell of the first four classes 4/4. 151M at 50k: whole +0.0006 ± 0.0004 (3/4); prepositions +0.0005 and +0.0003 (4/4 each); unseen +0.0003 (4/4); the same input −0.0005 ± 0.0003 (1/4) | 4 paired | journey §33.74 |
| **G-38's control, derived** (`derive_lexical_codes_20261009T035502`, `…T035553`): `ucaseprep` at 0.05 under BPE 17k and endings apart | 415 and 414 word types recoded (majority tag ADP), 11.4% of word occurrences; 97.6% of them tagged ADP in context, 99.7% of ADP occurrences recoded; value changed at 23.3% of recoded occurrences (2.65% of words, 1.1–1.2% of tokens) | — | journey §33.74 |
| **D7 at four seeds** (results_74; BPE 32k at three — *its cell superseded by the row below*): case / word form over each tokenizer's null, end of a 10k budget, ~38M tokens | BPE 4k +0.0044 ± 0.0004 / +0.0014 ± 0.0004; 8k +0.0040 ± 0.0009 / +0.0011 ± 0.0008; 17k +0.0032 ± 0.0007 / +0.0011 ± 0.0009; 32k +0.0030 ± 0.0002 / +0.0011 ± 0.0003; Unigram 17k +0.0054 ± 0.0007 / +0.0020 ± 0.0007; endings apart +0.0048 ± 0.0006 / +0.0017 ± 0.0007 — case 4/4 under each (3/3 for 32k); case over its word form +0.0019 to +0.0033 under all six. At step 2,000: +0.0090, +0.0071, +0.0051, +0.0048, +0.0055, +0.0045 | 3–4 | journey §33.65 |
| **BPE 32k at four seeds** (the chain of 10 October; `tokenizer_table_20261010T170757`): over its null, end of a 10k budget, ~38M tokens | case **+0.0029 ± 0.0002** (4/4; seed 3 +0.0027), word form +0.0008 ± 0.0003 (3/4), case over its word form +0.0022 ± 0.0003 (4/4); null 1.0447 (seed 3 1.0444), +0.0450 ± 0.0007 below BPE 17k's; at step 2,000 case +0.0049 ± 0.0002 | 4 | journey §33.81 |
| **Case gains more where endings are exposed** (results_74; paired by seed with BPE 17k) | end of budget: Unigram 17k +0.0021 ± 0.0006 (4/4), endings apart +0.0016 ± 0.0010 (4/4); at null BPB 1.25 all six within 0.0006 of +0.0052; at 1.15 Unigram +0.0031 ± 0.0006, endings apart +0.0043 ± 0.0013 (4/4 each) — case's word form at 1.15: +0.0021 ± 0.0008 and +0.0033 ± 0.0017 (4/4 each); at 1.40 Unigram +0.0049 ± 0.0009 (4/4) | 4 paired | journey §33.65 |
| **The previous word's tag, probed** (results_72, seed 4; gains over seeds 0–3) | deepest layer, end of 20k: base null 0.619 (best layer 0.655), 256 wide 0.731 (0.805); current word's tag 0.881, 0.895. Wider over base at matched null BPB 1.75–1.45: +3.9 to +10.4 points (current word's: +0.8 to +1.3). Largest gap between the widths' gains at a common level, 20k / 10k budgets: on the previous word's tag 0.0035 / 0.0025, the current word's 0.0063 / 0.0052, the null's BPB 0.0098 / 0.0099; both +0.0075 at accuracy 0.62. Switched off at 5,000: layer 0 reads the tag at 0.997–1.000 before the switch (never fed: 0.33–0.36); deepest layer at the end 0.635 against 0.615 never fed (+2.0) | 1–4 | journey §33.60 |
| **The night of 6 October** (results_71) | G-71, rotary, 20k, seed 0, without the causal shift: flat slot 1.3840 (+0.0068 over the plain baseline, against +0.0076 with it), flat added 1.3849 (+0.0059, against +0.0072). Ukrainian nulls re-run on the constant: ~38M 1.0891, 1.0910, 1.0874, 1.0911; 151M at 50k 0.9419, 0.9464, 0.9437, 0.9448 — each the audit's re-score to the sixth decimal; step 2,000 below the old record by 0.00038–0.00047; case at step 2,000 over the null **+0.0056 ± 0.0005** (4/4). D7, ~38M, 10k, seed 0, case / word form over the null: BPE 4k +0.0047 / +0.0020 (null 1.2239), 8k +0.0017 / −0.0004 (1.1542), 17k +0.0028 / +0.0002 (1.0891), Unigram 17k +0.0034 / +0.0001 (1.0890), 32k +0.0035 / +0.0011 (1.0449) | 1–4 | journey §33.58 |
| **Ukrainian, restated against a constant null** (G-72, G-73; saved best-step models re-scored, 4 seeds each) | case over the null at 10k steps: ~1.5M **+0.0076 ± 0.0016**, ~7.6M +0.0031 ± 0.0002, ~38M +0.0032 ± 0.0007, 151M **+0.0026 ± 0.0004** (4/4 each); on 151M at 5k +0.0055 ± 0.0006, 20k −0.0003 ± 0.0004, 50k −0.0013 ± 0.0008; word form alone ~1.5M +0.0050 ± 0.0013, ~7.6M +0.0014 ± 0.0004, 50k −0.0020 ± 0.0010; null over the plain baseline ~1.5M +0.0099 ± 0.0016, ~7.6M +0.0085 ± 0.0017, 151M +0.0028 ± 0.0010. The null on the constant against its own codes: −0.0005 (151M, every budget), −0.0010 (~38M), −0.0020 ± 0.0001 (~7.6M), −0.0039 ± 0.0002 (~1.5M); English at most −0.0002 (71 runs) | 4 | journey §33.56 |
| **The Ukrainian law, restated against the constant null** (`learning_law_20261012T003054`; case over its null along training, 4 seeds per condition, to each run's best step) | gain at null BPB 1.30 / 1.25 / 1.20 / 1.15 / 1.10: 151M at 50k +0.0055 / +0.0055 / +0.0051 / +0.0046 / **+0.0029**; 151M at 5k +0.0056 / +0.0058 / +0.0056 / — / —; ~38M at 10k +0.0050 / +0.0049 / +0.0050 / +0.0045 / **+0.0037**; ~7.6M at 10k +0.0060 / +0.0062 / +0.0056 / +0.0045 / —; ~1.5M at 10k +0.0065 / — / — / — / —. Spread 0.0015 at 1.30, 0.0013 at 1.25, 0.0006 at 1.20, 0.0001 at 1.15; at 1.10 the 50k run 0.0008 below the ~38M run (against the first null 0.0015, §33.38). The 151M runs at 10k and 20k have no curve on the constant (their nulls were re-scored, not re-run) | 4 | journey §33.58, §33.91 |
| **The Ukrainian law within its seed spread** (`learning_law_20261015T154512`, `--spread`; per-seed gains at each point, to each seed's best step; 20,000 random deals of the seeds' values to the conditions) | gap between the conditions' means / share of random deals with a gap as large: 1.30 0.0015 / 24%; 1.25 0.0013 / 50%; 1.20 0.0006 / 88%; 1.15 0.0001 / 100%; 1.10 0.0007 / 57% (151M at 50k +0.0030, ~38M at 10k +0.0036, per-seed means). Each condition's seeds: sd 0.0004–0.0015 | 4 per condition | journey §33.98 |
| **What the early help saves in training steps** (`steps_saved_20261012T003033`; the 50k runs, per seed, the null's curve interpolated to the feature's level at steps 1k / 2k / 5k / 10k) | English part of speech 86 / 141 / 434 / 790 steps (8.6%, 7.0%, 8.7%, 7.9%); inflection 53 / 78 / 239 / 326 (5.3%, 3.9%, 4.8%, 3.3%); Ukrainian case on 151M, against the constant null, **22 / 64 / 162 / 118 (2.2%, 3.2%, 3.2%, 1.2%)** — as first recorded 23 / 69 / 181 / 163 (2.3–3.6%, 1.6% at 10k) | 4 | journey §33.33, §33.91 |
| **The long chain of 6 October** (results_70) | RoPE, 20k, against the plain baseline: flat slot +0.0064 ± 0.0004 (4/4), flat added +0.0072 ± 0.0008 (3/3), zero slot +0.0013 ± 0.0015, zeros added +0.0012 ± 0.0018; within families: flat over zero slot +0.0051 ± 0.0017, flat added over zeros added +0.0060 ± 0.0016. Switch off at 5,000 of 20k, five seeds: −0.0039 ± 0.0009. Real over shuffled target, five seeds: +0.0009 ± 0.0008. Probes (seed 4): switched model level with off-from-start from step 6,000; real target +4.6 points at layer 2 by step 2,000. At null BPB 1.55, 20k against 50k: all four features within 0.0004. Probe axis (desk): at accuracy 0.870, base +0.0133, 256 wide +0.0076. 256 wide, five seeds: +0.0055 ± 0.0039 (4/5) | 3–5 | journey §33.52 |
| **The zero slot under rotary positions** (seed 0, run 16a11dd9bf91) | 20k: 1.3910, against the plain baseline's 1.3908 (−0.0002) and the constant slot's 1.3835 (same family: +0.0075). Desk, the RoPE null over the plain baseline by token frequency, rarest bucket first: +0.0197, +0.0195, +0.0166, +0.0199, +0.0165, +0.0151 nats (4/4 each) | 1–4 | journey §33.51 |
| **The long chain of 5 October** (results_69) | 20k, seeds 0–3: real over shuffled target +0.0003 ± 0.0007 (+0.0094 at step 1k); RoPE architecture effect +0.0057 ± 0.0006; RoPE over learned positions +0.0443 ± 0.0011; weaned to 5,000 −0.0029 ± 0.0016 (abrupt −0.0042 ± 0.0011). Probe at 20k (seed 4): 256 over 128 wide +0.8 to +1.3 points, null BPB 1.75–1.45. Tiled ×2, seed 1: +0.0037 at 1.55 | 1–4 | journey §33.50 |
| **The screen and block E** (results_68, seed 0) | 20k: target real 1.4311, shuffled 1.4329, plain baseline 1.4325; RoPE baseline 1.3908, null 1.3835 (+0.0073); weaned to 5,000 1.4278 (−0.0023 against off-from-start); inflection 20k seeds 2–3 1.4216, 1.4248 (four-seed gain at 1.55 +0.0070). 10k, 256 wide: tiled ×2 null 1.3818, part of speech 1.3830 (+0.0024 at 1.55). Probe, deepest layer at null BPB ~1.65: 0.867 (256) against 0.854 (128) | 1–4 | journey §33.49 |
| **The tail at full training** (D5, desk; 50k runs) | loss gain over the null by token-frequency bucket, rarest to most frequent — part of speech +0.0016, +0.0007, −0.0045, −0.0029, +0.0112, +0.0159 nats; inflection −0.0194, −0.0216, −0.0356, −0.0232, +0.0052, +0.0071; Ukrainian case −0.0023, −0.0015, −0.0080, −0.0129, +0.0026, +0.0001 | 4 | journey §33.47 |
| **The wide model at 20k; features at 20k** (results_67) | 256 wide, 20k: part of speech over null −0.0022, −0.0002, −0.0008, −0.0002; null over baseline +0.0057, +0.0200, −0.0033, +0.0028; final BPB spread across seeds: baseline 0.0237, null 0.0026, part of speech 0.0007. At null BPB 1.55, 20k (seeds 0–1) − 50k: inflection +0.0020, word form +0.0001, frequency band +0.0002, structure −0.0007 | 2–4 | journey §33.46 |
| **G-69: a bit-for-bit reproduction** (results_66) | the 10k-budget baseline, seed 0: September run cfbb4ce93538 and current-code run 12e3d62efc84 both BPB 1.5265532254632217, identical at all 21 evaluations; architecture effect at a 10k budget over four seeds +0.0035 ± 0.0011 | 4 | journey §33.43 |
| **The night of four directions** (results_65) | on at 5,000 of 20k: +0.0049 against kept +0.0067 (54–104%); off at 5,000 of 20k: −0.0042 (4/4); architecture effect 10k +0.0044, +0.0043, +0.0050 (seeds 1–3), 20k +0.0078 ± 0.0009, 256 wide at 20k +0.0057; part of speech at null BPB 1.55: 256 wide +0.0036 (base +0.0134); 64 wide −0.0028 below the base curve at its end | 1–4 | journey §33.42 |
| **Day block D** (seed 0) | G-68: on at 5,000 of a 20k budget +0.0039 over off-from-start, 63% of the kept +0.0061 (runs 7ab5b53eb521, cc10ec17d780); width 256, 10k budget: null 1.3839, part of speech 1.3816, gain +0.0022; at null BPB 1.58 +0.0031 against the base curve's +0.0133; worst deviation −0.0111 (runs d2d886d98a54, b883601b7067) | 1 | journey §33.40 |
| **The law across schedules (D1)** *(Ukrainian 5k and 20k superseded by the restated row)* | English part of speech, 5k/10k/20k vs the 50k curve: worst deviation −0.0005, +0.0004, +0.0010; at a 10k budget, against the null: part of speech +0.0130 ± 0.0004, word form +0.0099 ± 0.0005, inflection +0.0067 ± 0.0014, frequency band +0.0046 ± 0.0008, structure +0.0008 ± 0.0008; Ukrainian case at 5k +0.0074 ± 0.0007, 20k +0.0015 ± 0.0004 | 4 | journey §33.38 |
| **The switch family, 4 seeds** (English part of speech, 10k budget, against off-from-start) | kept +0.0111 ± 0.0003; off at 2,500 −0.0023 ± 0.0006; off at 7,500 −0.0118 ± 0.0003; on at 5,000 +0.0010 ± 0.0004 | 4 | journey §33.38 |
| **Gain against learning (D1, preliminary)** — `scripts/analysis/learning_law.py` *(the Ukrainian points move by the null's measured offset, §33.56; exact after re-running the nulls — superseded for Ukrainian by the restated law row, §33.91)* | at a common null BPB: English part of speech 50k +0.0134, 10k +0.0139 (1.55); Ukrainian case 50k +0.0035, 20k +0.0040, 10k +0.0049, 38M +0.0056 (1.10). Zero crossings: structure 1.484, frequency band 1.412, inflection 1.404, part of speech from the word form 1.361; Ukrainian case 1.014 | 2–4 | journey §33.35 |
| **The switch, English part of speech, 10k budget** | end, against off-from-start: kept +0.0112, +0.0117; switched off at 2,500 −0.0022, −0.0008 (keeps −19%, −7%); at the switch −0.0294; runs 203b152df0de, 7a89912d99d0, d70b7dd27eac, 4b8bdbc877f6 | 2 | journey §33.34 |
| **Steps saved by the early help** (null's crossing, interpolated) | part of speech 7–9%, inflection 3–5%, Ukrainian case 2–4% of training steps, steps 1k–10k | 4 | journey §33.33 |
| **Training length, Ukrainian, all 151M tokens** *(superseded by the restated row)* | case over the null: 10k steps +0.0044 ± 0.0004 (4), **20k +0.0012 ± 0.0002 (2)**, 50k −0.0009 ± 0.0008 (4) | 2–4 | journey §33.32 |
| **Training budget, Ukrainian, 10k steps, 4 seeds** *(superseded by the restated row)* | case over the null: 1.5M +0.0096 ± 0.0016, 7.6M +0.0052 ± 0.0002, 38M +0.0052 ± 0.0007, **151M +0.0044 ± 0.0004**; word form only: 1.5M +0.0070 ± 0.0013, 7.6M +0.0034 ± 0.0004; null over baseline: 1.5M +0.0079, 7.6M +0.0065 | 4 | journey §33.32 |
| **Little data, Ukrainian, 4 seeds**, case over its matched null (10k steps, best step) *(superseded by the restated row; reading superseded — it is training progress)* | ~1.5M tokens **+0.0096 ± 0.0016**; ~7.6M **+0.0051 ± 0.0002**; 151M (50k steps) −0.0009 ± 0.0008 | 4 | journey §33.31 |
| **Ukrainian family, 4 seeds** *(the null's numbers superseded by the restated row)* | against the null: contextual case −0.0009 ± 0.0008, lexical −0.0015 ± 0.0010; contextual over lexical +0.0006 ± 0.0004; case at step 2k +0.0060 ± 0.0005 (4/4); null over baseline +0.0023 ± 0.0010 (4/4) | 4 | journey §33.30 |
| **English part of speech, 4 seeds** | over the null +0.0012 ± 0.0007; at step 2k +0.0140 ± 0.0011 (4/4) | 4 | journey §33.30 |
| **Ukrainian family, seed 0** *(superseded by the four-seed row)* | against the null: contextual case −0.0025, lexical −0.0034 (+0.0074 and +0.0051 at step 2k); null over baseline +0.0052 (0.55%); majority case = in-context 89.74% | 1 | journey §33.24 |
| **Position test** (slotted, seeds 0, 1) against the zero slot *(superseded by the four-seed row)* | slot empty −0.0081, −0.0051; position only in the slot −0.0039, −0.0028 — the zero slot is a clean position channel | 2 matched | journey §33.19 |
| The **zero slot** (`mzero`: slot held at exactly zero) over the baseline | +0.0083, +0.0071 — a token embedding narrower than the residual stream is the whole effect | 2 paired | §5.37 |
| … the same classes **shuffled** among words of similar frequency (`mstructshuf`) | +0.0105 (+0.0081, +0.0129) against structure's +0.0092 at the same seeds: **the content is not the ingredient** | 2 paired | §5.35 |
| … strictly causal part of speech (`poslex`) | +0.0081 (+0.0098, +0.0064) against `pos`'s +0.0090: the tagger leak contributed nothing | 2 paired | §5.35 |
| … a coarse **syntactic** class (`pos`, same width) helps as much | 0.0090 (0.67%); tied with structure at both seeds (−0.0002, −0.0002) — tagger leak bound 7.4%, different time course | 2 paired | §5.34 |
| Previous word's **inflection** (`mfeatlex`, strictly causal) | +0.0026 ± 0.0024 — not distinguishable from zero; adds nothing to structure (+0.0003); fades like identity | 2 paired | §5.34 |
| BLiMP, structure vs baseline, all 67 paradigms | **no reliable grammatical effect**: +0.6 ± 0.4 points (3/4); the anaphor gain (+6.9 ± 2.5, 4/4) is a learned preference against *itself* — gains where it is wrong, loses where it is right | 4 paired | §5.34, journey §33.8 |
| … from morphological classes alone | 0.0108 (0.80%) | 3 paired | §5.33 |
| … from name classes alone | 0.0110 (0.81%) | 3 paired | §5.33 |
| … morphology vs names, cell to cell | tied: −0.0001, −0.0012, +0.0019 | 3 paired | §5.33 |
| Word identity adds nothing once structure is present | −0.0000 to −0.0034 | 4 paired | §5.32 |
| Share of the apparent Level-2 gain that was real | **17%** (0.0120 of 0.0691) — the leak inflated it **5.8×** | 3 paired | §5.33 |
| Morpheme segmentation vs BPE, parameter-matched | **+13.46%** worse | 2 paired | §5.18 |
| … on the held-out test split | +12.9% worse | 1 | §5.13 |
| … against a second tokenizer family (Unigram), validation | +12.60% worse | 2 | §5.20 |
| … on the rarest words only | **9.0% better** | full split | §5.30 |
| `dep` labels a causal model already recovers | 78.5% (≤21.5% can leak) | — | §5.31 |
| Training is deterministic | same seed reproduces BPB to 13 decimals, across an environment change | 2 | §5.33 |

**Noise.** The 0.0021 "noise floor" cited through most of this document came
from cells with two seeds each, and baseline seeds 0 and 1 happen to agree
within 0.0005. A third seed spreads the baseline to **0.0096**. Two agreeing
seeds were one lucky sample of the spread, not a measurement of it. Small
effects are therefore reported as **paired means with a standard error, and the
count of seeds with the same sign**, not as multiples of a floor. Q1 is
unaffected (0.18 BPB, ~19× the widest baseline spread).

**Superseded — do not cite:** every Level-2 feature gain in §3 and §5.5–5.12
measured without `frozen.causal_identity` (within-word lookahead, journey §30);
the §5.29 causal figures (inferred boundaries, `CAUSAL_SEMANTICS < 3`); the
two-seed causal figures in §5.32, superseded by §5.33; and the first cardinality
sweep (clamp bug, quarantined).

## 3. Complete run inventory

28 training runs in 11 batches.

**Unit warning.** Perplexity is per-token and comparable ONLY within a pipeline.
A BPE cell at 67.36 and a morpheme cell at 6.25 are not 10× apart — they count
different things. **BPB is the only cross-pipeline metric here.** Three separate
results also land near 4.5% (frozen `dep` on morph/det in per-token perplexity,
the same bucketed, and Level-2 features on BPE in BPB); see journey §24 for
which is which.

### 3.1 Calibration (pre-refactor, not saved as JSON)

| run | result |
|---|---|
| BPE baseline, 3,000 articles | val loss bottomed at step 750 then climbed — overfitting, a data-quantity signal |
| BPE baseline, full corpus, 40k ceiling | val loss 4.17, **ppl 64.9, BPB 1.438** at step 39,500, still improving |

### 3.2 Batch 1 — first matrix (LOSSY pipeline, retired)

Legacy Whitespace BPE; morph/det streams not information-preserving. **These
numbers are void** and retained only because they are what the audits refuted.

| run_id | label | ppl | BPB | val_loss | val_tokens | best_step | wall |
|---|---|---|---|---|---|---|---|
| `6cc3e3789d4d` | bpe_baseline | 60.95 | 1.4162 | 4.1100 | 271,360 | 49,999 | 44.2 min |
| `7b28f2074be6` | det_baseline | 25.75 | 1.3271 | 3.2485 | 321,792 | 49,000 | 54.6 min |
| `7b9dd0fe08ef` | morph_baseline | 25.96 | 1.3268 | 3.2565 | 320,768 | 49,999 | 54.4 min |

Apparent result: morph beat BPE by **6.31% BPB**. Refuted — see §2.

### 3.3 Batch 2 — information-fair rerun

ByteLevel BPE; morph/det with all five surface channels (`Spdiwc`).

| run_id | label | ppl | BPB | val_tokens | bytes/token |
|---|---|---|---|---|---|
| `03b9423bfe83` | bpe_baseline_bpe_bytelevel | 57.69 | 1.3946 | 270,848 | 4.195 |
| *(seed 1)* | bpe_baseline_bpe_bytelevel | 58.06 | 1.3967 | 270,848 | — |
| **two-seed mean** | | | **1.3957** | | |
| `0e7cd8838192` | det_baseline_Spdiwc | 6.25 | 1.5246 | 655,360 | 1.734 |
| `735559832ec1` | morph_baseline_Spdiwc | 6.30 | 1.5292 | 654,592 | 1.737 |

**The sign flipped: morph now 9.7% WORSE.** Implied validation bytes agree to
0.05% across all three (1,136,212 / 1,136,663 / 1,136,794), confirming the same
denominator.

### 3.4 Batch 3 — Q2, frozen dependency role

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `51d794b3f24e` | morph_dep_Spdiwc | 6.02 | 1.4913 | 49,500 |
| `11e9fa7d35b3` | det_dep_Spdiwc | 5.97 | 1.4856 | 49,000 |

vs baseline: morph −4.46%, det −4.58%. Ahead in **91/91** evaluations from step
5,000.

### 3.5 Batch 4 — shuffled-codes control

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `ccd320c0a0a0` | morph_dep-SHUF_Spdiwc | 6.51 | 1.5560 | 49,999 |
| `88b6afa1e676` | det_dep-SHUF_Spdiwc | 6.45 | 1.5507 | 49,000 |

**Shuffled codes are WORSE than baseline** (+3.28% / +3.19%), so the gain is
linguistic content, not the presence of structure.

### 3.6 Batch 5 — learned factors and matched vocabulary

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `96ebe413e6bf` | morph_dep-LEARN_Spdiwc | 5.88 | 1.4721 | 49,999 |
| `873a52464643` | det_dep-LEARN_Spdiwc | 5.87 | 1.4716 | 49,500 |
| `a7175857fdf6` | bpe_baseline_bpe_17k | 67.36 | **1.3477** | 49,999 |

Learned beats frozen. BPE **improved** with the larger vocabulary
(1.3946 → 1.3477), widening the Q1 gap to **13.5%**. `bpe_17k` val_tokens =
252,160 (985 windows × 256).

### 3.7 Batch 6 — factor scale control

| run_id | label | ppl | BPB | best_step |
|---|---|---|---|---|
| `55c394dba2e3` | morph_dep-S0.22_Spdiwc | 6.00 | 1.4883 | 49,999 |
| `c69345978d4b` | det_dep-S0.22_Spdiwc | 5.94 | 1.4824 | 49,000 |

Rescaling the frozen one-hot to match the trained slice's norm closed only
**16% (morph) / 23% (det)** of the gap to learned. Trainability genuinely wins.

### 3.8 Batches 7–8 — Level 2, morphology as features on BPE

All against `bpe_baseline_bpe_17k` = 1.3477 BPB.

| run_id | label | ppl | BPB | vs base | params vs base |
|---|---|---|---|---|---|
| `1e57b24ed4f9` | bpe_mroot-SHUF-LEARN_bpe_17k | 68.26 | 1.3519 | **+0.31%** | −0.24% |
| `eed5d2a11a8a` | bpe_mstruct-SHUF_bpe_17k | 66.60 | 1.3440 | −0.27% | −2.63% |
| `3860406fd2c1` | bpe_mroot_bpe_17k (frozen projection) | 63.66 | 1.3296 | −1.34% | −5.26% |
| `34a9e9b342da` | bpe_mstruct_bpe_17k (frozen one-hot) | 62.25 | 1.3225 | −1.87% | −2.63% |
| `56239ccfaf84` | bpe_mroot-LEARN_bpe_17k | 59.42 | 1.3076 | −2.98% | −0.24% |
| `7b46e1e0fffa` | bpe_ectx_bpe_17k (frozen one-hot) | 64.66 | 1.3346 | −0.97% | −3.94% |
| `4e65750bc990` | bpe_mrole-LEARN_bpe_17k | 67.67 | 1.3492 | +0.09% | −1.64% |
| `c564d066df6b` | bpe_dep-LEARN_bpe_17k | 59.69 | 1.3090 | −2.89% | −2.63% |
| `58d2637e8f69` | bpe_mroot+mstruct+mrole-SHUF-LEARN_bpe_17k | 64.01 | 1.3314 | −1.23% | −4.51% |
| `1fb14ea6f45d` | bpe_mroot+mstruct+mrole-LEARN_bpe_17k | **56.72** | **1.2927** | **−4.08%** | −4.51% |

Wall clock: 55–59 minutes per run (14.0–15.1 steps/s).

### 3.9 Batch 9 — seed replication

| run_id | label | seed | ppl | BPB |
|---|---|---|---|---|
| `a12b91da4049` | bpe_baseline_bpe_17k | 1 | 67.47 | 1.3482 |
| *(new)* | bpe_mroot_bpe_17k (frozen projection) | 1 | 63.33 | 1.3280 |
| `e6b206f9a34a` | bpe_mroot+mstruct+mrole-LEARN_bpe_17k | 1 | 55.15 | 1.2837 |

| cell | seed 0 | seed 1 | spread | mean |
|---|---|---|---|---|
| baseline | 1.3477 | 1.3482 | **0.0005** | 1.3480 |
| `mroot` frozen projection | 1.3296 | 1.3280 | **0.0016** | 1.3288 |
| all three facts | 1.2927 | 1.2837 | 0.0090 | 1.2882 |

- seed noise (max spread across three cells): **0.0090 BPB**
- all-three effect: **0.0597 BPB = 6.6x noise**, mean improvement **4.43%**
- `mroot` frozen projection effect: **0.0192 BPB = 2.1x noise**, **1.42%**
- all four cross-seed pairings hold

---

## 4. Audit results

### 4.1 Reconstruction audit (`reconstruction_audit.py`)

det pipeline, 500 validation lines, 29,333 words.

| stage | exact reconstruction | failures | byte recovery |
|---|---|---|---|
| before multi-root fix | 24,036 (81.94%) | 5,297 (18.06%) | 72.14% |
| after multi-root fix | 24,356 (83.03%) | 4,977 (16.97%) | 73.99% |

Failure classes at the second measurement: inflection dropped ~78%,
orthographic ~22%, vanished morphemes 0% (fixed).

**Superseded by the collision audit** — reconstruction failure overstates the
problem, because non-colliding spelling variation is recoverable in principle.

### 4.2 Collision audit (`collision_audit.py`)

det, 2,000 validation lines. Correction = Σ −log₂ P(surface | token sequence),
an upper bound (context-free).

| stage | occurrences | colliding groups | in collision | BPB correction |
|---|---|---|---|---|
| original (lossy) | 142,732 | 2,064 | 28.46% | **0.0440** |
| after surface channels | 157,935 | 370 | 14.50% | 0.1571 ← *audit artifact* |
| after audit fixes | 165,594 | 111 | 1.24% | 0.0028 |
| after byte fallback | 167,025 | 103 | **0.75%** | **0.0008** |

The 0.1571 reading was measurement error: entity `word`-field inconsistency plus
the audit skipping digit/punct tokens. ~70% of it was artifact.

Residual collision classes (all genuine limitations, not bugs):
dialectal normalisation `{honor, honour}` `{colour, color}` `{travelled,
traveled}`; homograph conflation `{found:57, founded:10}` `{surveillance,
survey}`; multi-word entity spans.

### 4.3 Coverage audit (`coverage_audit.py`)

1,000 validation lines, 422,168 bytes.

| class | in corpus | BPE covered | det BEFORE | det AFTER |
|---|---|---|---|---|
| alpha | 319,600 | 95.0%* | 94.5% | 97.4% |
| digits | 8,098 | 99.7% | 5.5% | **99.9%** |
| punctuation | 15,207 | 99.5% | 1.9% | **100.0%** |
| whitespace | 79,263 | 0.0% | 0.0% | 0.0% |

\* BPE's alpha figure is an audit artifact (the script lowercases the original
while BPE preserves case); the digit and punctuation rows are case-insensitive
and reliable.

Asymmetry: **22,480 bytes (5.32% of corpus)** before → **−89 bytes (−0.02%)**
after. Whitespace is dropped by both, so it never biased Q1 — but it is 18.8% of
corpus bytes, so neither BPB was a true compression rate until ByteLevel.

### 4.4 Multi-root audit (`multiroot_audit.py`)

| metric | value |
|---|---|
| MorphoLex entries parsed | 68,623 |
| entries with >1 root | **8,453 (12.32%)** |
| roots per entry | 0: 38 · 1: 60,132 · 2: 8,375 · 3: 77 · 4: 1 |
| new roots added to vocabulary | 83 (15,022 → 15,105) |

Most frequently recovered morphemes: `log` 267, `man` 212, `men` 162,
`graph` 158, `house` 96, `head` 90, `board` 84, `way` 77, `work` 75, `line` 72.

For comparison, the format-B fix of journey §7 affected 4,941 entries.

### 4.5 OOV audit (`oov_audit.py`)

det, 1,000 validation lines, 43,050 alphabetic words.

| metric | value |
|---|---|
| fell to character fallback | 5,625 (**13.07%**) |
| bytes in those words | 55,481 (**21.58%** of word bytes) |

| reason | count | share |
|---|---|---|
| capitalised (likely entity) | 4,360 | **77.5%** |
| very short | 430 | 7.6% |
| alphanumeric | 416 | 7.4% |
| plain lowercase OOV | 315 | 5.6% |
| non-ascii | 68 | 1.2% |
| very long / compound | 36 | 0.6% |

Most frequent unsegmentable: `'s` 240, `Meridian` 61, `Tikal` 57,
`Mississippi` 28, `km` 27, `Mongol` 27, `Radcliffe` 26, `Richmond` 25, `US` 23.

### 4.6 Level-2 preflight (`preflight_level2.py`)

BPE 17k + morphological analyser, 300 validation lines, 26,315 tokens.

| check | result |
|---|---|
| alignment rate | **99.99%** (start-offset bug would show ~20%) |
| tokens carrying a root | **42.9%** |
| not a word (punct/digit) | 19.4% |
| word with no root | 37.6% |
| distinct roots observed | 2,020 of 16,365 |

Within-token-id variance — the property that makes a fact *contextual*:

| fact | ids receiving >1 code | share |
|---|---|---|
| dep | 1,913 | **38.4%** |
| ectx | 849 | 17.0% |
| pos | 739 | 14.8% |
| mroot | 553 | 11.1% |
| mstruct | 540 | 10.8% |
| mrole | 381 | 7.6% |

Full-corpus cache build: **4h37m**, 120,625,719 train tokens, alignment 99.98%.
Cached code inventories: dep 8, pos 18, ectx 12, mstruct 8, mrole 5,
mroot 10,873 distinct (max index 16,364).

---

## 5. Derived analyses

### 5.1 Category bit accounting (`category_bits.py`)

Where each pipeline spends its bits. Information-fair runs, sampled validation.

**BPE (ByteLevel 10k)** — 270,848 tokens, 1,584,556 bits, sampled BPB 1.3946

| category | tokens | % tok | bits | % bits | bits/token |
|---|---|---|---|---|---|
| word-initial (carries space) | 211,300 | 78.0% | 1,341,007 | 84.6% | 6.346 |
| word-internal | 59,548 | 22.0% | 243,549 | 15.4% | 4.090 |

**det (`Spdiwc`)** — 327,680 tokens, 855,208 bits, sampled BPB 1.5048

| category | tokens | % tok | bits | % bits | bits/token |
|---|---|---|---|---|---|
| root | 37,049 | 11.3% | 314,878 | 36.8% | 8.499 |
| character | 80,813 | 24.7% | 211,799 | **24.8%** | 2.621 |
| function | 34,561 | 10.5% | 137,795 | 16.1% | 3.987 |
| punct | 18,653 | 5.7% | 43,538 | 5.1% | 2.334 |
| digit | 10,221 | 3.1% | 34,505 | 4.0% | 3.376 |
| space | 100,844 | **30.8%** | 24,960 | 2.9% | **0.245** |
| case | 14,281 | 4.4% | 24,837 | 2.9% | 1.739 |
| prefix | 4,804 | 1.5% | 23,303 | 2.7% | 4.851 |
| infl | 17,281 | 5.3% | 13,904 | 1.6% | **0.807** |
| entity | 1,522 | 0.5% | 12,006 | 1.4% | 7.888 |
| suffix | 6,052 | 1.8% | 9,415 | 1.1% | **1.556** |
| byte | 1,599 | 0.5% | 4,269 | 0.5% | 2.670 |

BPB excluding whitespace tokens: det 1.4609, morph 1.4656.

Two findings: **inflection and suffix tokens are the cheapest in the stream**
(0.807 / 1.556 bits), so morphology is demonstrably being exploited; and
**character fallback consumes 24.8% of all bits**, which is where Q1 is lost.

A prior hypothesis that whitespace explained the Q1 gap was **refuted by this
measurement**: `<sp>` costs 0.245 bits, not the 0.721 the hypothesis required.

### 5.2 Coverage economics (`coverage_economics.py`)

det vs BPE-ByteLevel-10k, 400 validation lines.

**Part 1 — tokenization economics**

| word class | words | % | bytes | morph tok/word | BPE tok/word |
|---|---|---|---|---|---|
| analysed cleanly | 19,736 | 87.8% | 93,698 | 1.51 | **1.18** |
| character fallback | 2,744 | 12.2% | 26,117 | **9.54** | 3.20 |

Fallback words: 12.2% of words but **21.8% of word bytes**. morph spends
**2.98×** the tokens BPE does on them.

**Part 2 — restricted bits-per-byte**

| region | bytes | morph BPB | BPE BPB | delta | share of gap |
|---|---|---|---|---|---|
| words analysed cleanly | 113,240 | 1.2672 | 1.1937 | +0.0735 | 35.5% |
| words in char fallback | 28,774 | 2.4076 | 2.0172 | +0.3904 | **47.9%** |
| punct / digits | 13,809 | 1.8004 | 1.5189 | +0.2815 | 16.6% |
| **ALL** | 155,823 | **1.5250** | **1.3746** | **+0.1505** | 100% |

Extrapolations: optimistic (fallback words as cheap as clean ones) → morph
1.3145, **wins** by 0.0601; realistic (fallback words merely match BPE's rate)
→ morph 1.4529, **still loses** by 0.0783.

**Part 3 — clean words stratified by training frequency** *(the U-shape)*

| word frequency | bytes | % clean bytes | morph BPB | BPE BPB | delta | verdict |
|---|---|---|---|---|---|---|
| 1–10 | 5,475 | 5.8% | 1.8925 | 2.1707 | **−0.2782** | **morph wins 12.8%** |
| 10–100 | 16,552 | 17.7% | 1.6822 | 1.6731 | +0.0091 | tie |
| 100–1k | 33,732 | 36.0% | 1.4418 | 1.3049 | +0.1369 | morph loses 10.5% |
| 1k–10k | 18,697 | 20.0% | 1.4687 | 1.4038 | +0.0649 | morph loses |
| 10k–100k | 13,338 | 14.2% | 1.5668 | 1.5660 | +0.0008 | tie |
| 100k+ | 5,904 | 6.3% | 0.7280 | 0.7451 | −0.0171 | morph wins slightly |

Frequency bands from a 40,000-line training sample (~1/30 of corpus), so
full-corpus counts are ~30× higher.

### 5.3 Per-frequency-bucket Q2 effect (`compare_buckets.py`)

Frozen `dep` vs baseline, scored from checkpoints, 491,520 validation tokens.

| train frequency | val tokens | % tok | morph delta | det delta |
|---|---|---|---|---|
| 1–100 | 291 | 0.06% | −15.89% | −5.00%* |
| 100–1,000 | 2,212 | 0.45% | **−14.53%** | **−14.66%** |
| 1,000–10,000 | 14,317 | 2.91% | −14.80% | −15.19% |
| 10,000–100,000 | 47,472 | 9.66% | −13.14% | −14.27% |
| 100,000–1,000,000 | 53,809 | 10.95% | −10.13% | −11.07% |
| 1,000,000+ | 373,419 | **75.97%** | **−1.95%** | −1.87% |
| OVERALL | 491,520 | 100% | −4.47% | −4.64% |

\* 291 tokens — noise, disregard.

**Rare tokens benefit ~7.5× more than frequent ones.** Because 76% of tokens sit
in the highest-frequency bucket, the headline 4.5% substantially understates the
mechanism.

### 5.4 Convergence speed

Steps to first reach a given per-token perplexity.

| cell | ppl ≤ 6.5 | ppl ≤ 6.3 |
|---|---|---|
| det baseline | 29,000 | 41,500 |
| det + frozen dep | **19,500** | **25,500** |
| morph baseline | 30,500 | never reached |
| morph + frozen dep | **21,000** | **27,000** |

**33–39% fewer steps to the same perplexity.**

### 5.5 Level-2 composition and controls

Gains over `bpe_17k` baseline (1.3477 BPB):

| feature set | BPB gain |
|---|---|
| mroot alone | 0.0401 |
| mstruct alone | 0.0252 |
| sum if independent | 0.0653 |
| mroot+mstruct+mrole actual | **0.0550** (84% of additive) |

Controls (noise floor 0.0090 BPB, from the seed replication):

| | shuffled vs baseline | real vs shuffled |
|---|---|---|
| mroot (learned) | +0.0042 — **below noise** | −0.0443 (4.9x noise) |
| mstruct (frozen) | −0.0037 — **below noise** | −0.0215 (2.4x noise) |

Only *real vs shuffled* is claimable. The direction of *shuffled vs baseline*
falls inside the noise floor for both, so no mechanism may be inferred from it.

### 5.6 Factor signal scale (journey §21.3)

| component | L2 norm at initialisation |
|---|---|
| trained embedding slice (120 dims, std 0.02) | 0.219 |
| positional embedding (128 dims, std 0.02) | 0.226 |
| frozen factor (one-hot, 8 dims) | **1.000** |
| learned factor (8 dims, std 0.02) | 0.057 |

### 5.7 Fixed random projection quality

16,365 root classes into 16 dimensions, unit-normalised rows:
max off-diagonal cosine among 2,000 sampled classes **0.878**, mean |cosine|
**0.203**. Near-orthogonal on average, but 16 dimensions cannot keep 16k classes
cleanly separated — a stated limitation of the frozen projection arm.

### 5.8 Legibility probe (`probe_dep.py`)

Linear probe on the hidden state after every block, predicting the coarse
dependency label. Chance = **majority-class rate 0.371**, not 1/n.

| model | chance | emb | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| morph baseline | 0.371 | 0.779 | 0.801 | 0.809 | 0.812 | 0.802 |
| morph + frozen dep | 0.371 | 0.999 | 1.000 | 0.999 | 0.985 | 0.959 |
| morph + learned dep | 0.371 | 1.000 | 0.998 | 0.988 | 0.966 | 0.938 |
| det baseline | 0.371 | 0.775 | 0.798 | 0.809 | 0.814 | 0.803 |
| det + frozen dep | 0.371 | 0.999 | 0.999 | 0.997 | 0.982 | 0.955 |
| det + learned dep | 0.371 | 1.000 | 0.998 | 0.988 | 0.965 | 0.935 |

Derived:

| quantity | morph | det |
|---|---|---|
| baseline reads dep at L4 | 0.802 | 0.803 |
| injection closes, of remaining gap | 78.2% | 75.8% |
| frozen − learned at L4 | +0.021 | +0.020 |
| decay emb→L4, frozen | 0.040 | 0.044 |
| decay emb→L4, learned | 0.062 | 0.065 |
| learned decay as % of frozen | 155% | 148% |

**Largest morph-vs-det discrepancy across all 15 layer-arm pairs: 0.004.**

### 5.9 Held-out test evaluation (`evaluate_split.py`)

All checkpoints re-scored on the WikiText-103 **test** split, never used during
development. No retraining.

| cell (bpe_17k group) | val BPB | test BPB | val delta | test delta |
|---|---|---|---|---|
| baseline | 1.3480 | 1.3710 | — | — |
| `mrole` learned | 1.3492 | 1.3741 | +0.0012 | +0.0031 |
| `ectx` frozen | 1.3346 | 1.3597 | −0.0133 | −0.0112 |
| `mroot` frozen projection | 1.3288 | 1.3559 | −0.0192 | −0.0151 |
| `mstruct` frozen | 1.3225 | 1.3448 | −0.0255 | −0.0262 |
| `dep` learned | 1.3090 | 1.3343 | −0.0389 | −0.0367 |
| `mroot` learned | 1.3076 | 1.3337 | −0.0404 | −0.0373 |
| **all three learned** | **1.2882** | **1.3148** | **−0.0598** | **−0.0562** |
| `mroot` shuffled | 1.3519 | 1.3782 | +0.0040 | +0.0072 |
| `mstruct` shuffled | 1.3440 | 1.3664 | −0.0039 | −0.0046 |
| all three shuffled | 1.3314 | 1.3542 | −0.0166 | −0.0167 |

**Headline on test: 1.3710 → 1.3148 = 0.0562 BPB (4.10%), 6.6x the test noise
floor of 0.0086.** Every one of ten headline comparisons keeps its sign.

Validation→test shift by group (tight within each tokenizer, so selection bias
cancels in every comparison):

| group | n | mean shift | sd |
|---|---|---|---|
| bpe_17k | 13 | +0.0247 | 0.0019 |
| morph | 5 | +0.0166 | 0.0015 |
| det | 5 | +0.0100 | 0.0009 |

Q1 on test: BPE-17k 1.3710, morph 1.5480 (+12.9%), det 1.5361 (+12.0%).

### 5.10 Legibility probe for `mroot` (`probe_dep.py --fact mroot`)

| model | chance | emb | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| bpe baseline | 0.571 | 0.806 | 0.803 | 0.803 | 0.784 | 0.751 |
| bpe + `mroot` frozen | 0.571 | 0.892 | 0.876 | 0.869 | 0.851 | 0.816 |
| bpe + `mroot` learned | 0.571 | 0.887 | 0.865 | 0.860 | 0.833 | 0.792 |

| | headroom used at L4 | trend with depth |
|---|---|---|
| `dep` on morph, baseline | 68.5% | 0.779 → 0.802, improves |
| `mroot` on BPE, baseline | **42.0%** | 0.806 → **0.751**, decays |

frozen − learned at L4: **+0.024** (`dep`: +0.021).

### 5.15 Surface-channel leave-one-out series (det)

Baseline `det_baseline_Spdiwc` = **1.5252 (two-seed mean)**. The ablations and the
legacy cell are all seed 0, so this is the paired comparison; using the
two-seed mean (1.5252) would add 0.0006 of unpaired offset to every marginal.
Each cell removes one channel and
requires its own cache build (~4h), because a surface flag changes the ID
stream, not merely the codes.

| channel removed | cell | run_id | BPB | marginal | tokens lost | bits/token |
|---|---|---|---|---|---|---|
| punctuation + digits | `S--iwc` | `4429c0884116` | 1.4647 | 0.0605 | 56,451 | 1.192 |
| inflection | `Spd-wc` | `fb35ef8a3634` | 1.4930 | 0.0321 | 34,440 | 1.017 |
| whitespace | `Spdi-c` | `c1e0ab32e7fa` | 1.5008 | 0.0244 | 204,051 | 0.131 |
| case | `Spdiw-` | `6302e2a46873` | 1.5149 | 0.0103 | 29,327 | 0.380 |
| **sum of marginals** | | | | **0.1273** | | |
| joint (legacy, all five off, n=1) | — | `7b28f2074be6` | 1.3271 | **0.1981** | | |
| **interaction residual** | | | | **+0.0708** | | 36% |

Marginals do not sum to the joint effect and are not expected to: the channels
are partly redundant, so joint removal is superadditive.

**In-stream cost vs marginal cost** — the empirical signature separating
coverage from collision:

| channel | in-stream bits/token | marginal bits/token | ratio |
|---|---|---|---|
| whitespace | 0.245 | 0.131 | 0.53 |
| case | 1.739 | 0.380 | 0.22 |
| inflection | 0.807 | **1.017** | **1.26** |

A ratio below 1 means the channel was redundant encoding — its information is
absorbed by neighbouring tokens when it is removed. A ratio above 1 means the
channel was doing disambiguating work, and removing it makes the rest of the
stream harder. Inflection is the only channel above 1.

### 5.16 C-02 collision calibration

| quantity | value |
|---|---|
| predicted by `collision_audit.py` | 0.0440 BPB |
| observed cost of the collapse | **0.0321 BPB** (n=2) |
| ratio observed / predicted | **0.73** |
| corrected lossy (1.4930 + 0.0440) | 1.5370 |
| true lossless | 1.5252 |
| overshoot | **+0.0119** (1.5x the noise floor) |

The correction is confirmed as an upper bound, **27% loose**, in the direction
predicted: expression (2) is context-free while a running model disambiguates
from context.

### 5.17 Coverage audit, full split, after the case and byte-fallback fixes

2,461 validation lines, 1,136,862 bytes.

| class | in corpus | bpe covered | det covered |
|---|---|---|---|
| alpha | 866,949 | 99.9% | 97.3% |
| digits | 19,105 | 99.9% | 100.0% |
| punctuation | 39,382 | 99.8% | 100.0% |
| whitespace | 211,426 | 0.0% | 0.0% |

**Asymmetry: −81 bytes, −0.01% of corpus** (det encodes marginally more than
BPE). The 1,000-line figure of −0.02% is confirmed on a 2.7x larger sample.

The alpha row is excluded from the asymmetry: morpheme pipelines store
normalised roots, so the character multiset legitimately changes
(`beautiful` → `beauty`+`ful`). `collision_audit` measures the residual
ambiguity of the same stream at 0.0008 BPB over 0.74% of occurrences, which
settles independently that no information is lost.

### 5.18 Every headline cell, two seeds

| cell | seed 0 | seed 1 | spread | mean |
|---|---|---|---|---|
| morph_baseline_Spdiwc | 1.5292 | 1.5297 | 0.0005 | **1.5294** |
| det_baseline_Spdiwc | 1.5246 | 1.5258 | 0.0012 | **1.5252** |
| bpe_baseline_bpe_bytelevel | 1.3946 | 1.3967 | 0.0021 | **1.3957** |
| bpe_baseline_bpe_17k | 1.3477 | 1.3482 | 0.0005 | **1.3480** |

Noise floor across these four: **0.0021 BPB**.

| comparison | delta | vs noise |
|---|---|---|
| morph vs bpe ByteLevel 10k | +0.1338 (+9.58%) | **63.7x** |
| det vs bpe ByteLevel 10k | +0.1295 (+9.28%) | 61.7x |
| morph vs bpe 17k (param-matched) | +0.1815 (+13.46%) | 86.4x |
| det vs bpe 17k | +0.1773 (+13.15%) | 84.4x |

### 5.19 The parameter confound in the legacy comparison

The original apparent morph advantage was measured with unmatched arms:

| arm | BPB | vocabulary | trainable params |
|---|---|---|---|
| `bpe_baseline` (legacy) | 1.4162 | 10,004 | 3,387,136 |
| `det_baseline` (legacy) | 1.3271 | 17,155 | 5,217,792 |

det carried **1.54x** the parameters. Two cells that differ only in vocabulary
size measure what that is worth:

| | BPB |
|---|---|
| bpe ByteLevel 10k (n=2) | 1.3957 |
| bpe ByteLevel 17k (n=2) | 1.3480 |
| **vocabulary alone** | **0.0477** |

| component of the apparent 0.0891 advantage | value | share |
|---|---|---|
| parameter/vocabulary confound | 0.0477 | **54%** |
| coverage, collision, other channels | 0.0414 | 46% |

So the original comparison was unfair along **three** axes — coverage,
collision, and parameters — and the largest single contributor was the one never
named in the original analysis.

### 5.20 Unigram LM arm (C-10)

Trained with the same ByteLevel pre-tokenizer and the same 17,151 vocabulary as
the BPE control; round-trip verified.

| tokenizer, 17k | seed 0 | seed 1 | mean | morph deficit | det deficit |
|---|---|---|---|---|---|
| BPE | 1.3477 | 1.3482 | 1.3480 | +13.46% | +13.15% |
| **Unigram** | 1.3594 | 1.3571 | **1.3583** | **+12.60%** | **+12.29%** |

BPE beats Unigram by **0.0103 BPB (4.5x noise)**. Bostrom & Durrett (2020) found
Unigram matching or beating BPE, but on downstream tasks at RoBERTa scale rather
than bits-per-byte at 5M parameters.

The morpheme pipeline loses to **both** statistical families at matched
vocabulary, so Q1 is not an artifact of comparing against BPE.

### 5.21 Learning-rate sweep (C-16)

Seven rates, both pipelines, matched 10,000-step horizon (`max_steps` and
`lr_horizon` both 10,000).

| lr | bpe_17k | det | bpe vs its best | det vs its best |
|---|---|---|---|---|
| 1e-4 | 1.7277 | 1.9716 | +22.03% | +23.97% |
| 2e-4 | 1.5881 | 1.8520 | +12.17% | +16.45% |
| 3e-4 | 1.5266 | 1.7766 | +7.83% | +11.71% |
| 5e-4 | 1.4703 | 1.6900 | +3.85% | +6.26% |
| 1e-3 | 1.4294 | 1.6229 | +0.96% | +2.04% |
| 2e-3 | 1.4184 | 1.5948 | +0.18% | +0.28% |
| **3e-3** | **1.4158** | **1.5904** | **0.00%** | **0.00%** |

Identical optimum; response 17.3% (bpe) against 17.7% (det) across the range.
The curve flattens at the top, so the optimum is bracketed.

**Caveat:** a 10,000-step horizon systematically favours higher learning rates,
because the cosine schedule anneals over fewer steps. This establishes that the
two pipelines respond to learning rate the same way; it does **not** establish
that 3e-4 was suboptimal at 50,000 steps.

### 5.22 The fully controlled reversal

The apparent-advantage measurement was parameter-unfair: det carried 1.54x the
parameters (vocab 17,155 against 10,004). Vocabulary alone is worth **0.0477
BPB**, measured from two cells differing only in vocabulary size (bpe ByteLevel
10k 1.3957 vs 17k 1.3480, both n=2).

| | apparent advantage | final deficit | total swing |
|---|---|---|---|
| as published | +6.29% | −9.28% | **0.2186** |
| fully controlled | **+3.03%** | **−13.15%** | **0.2186** |

Identical swing to four decimal places. Vocabulary sizes were unchanged across
the repair, so the parameter asymmetry is constant and cannot explain the sign
change.

### 5.23 Which noise floor applies where

Three floors appear in these documents. Each is the maximum seed spread over a
different family of cells, and each is correct for its own comparison. Stating
this explicitly because two of them differ by 5x and a reader meeting both
without explanation would reasonably call it an inconsistency.

| floor | scope | used for |
|---|---|---|
| **0.0105** | all replicated cells, including the n=3 all-three cell | Level-2 comparisons (§5.11–5.12), the conservative reading |
| **0.0021** | the four Q1 headline cells, all n=2 | Q1 comparisons (§5.18–5.19) |
| **0.0015** | the surface-ablation family, all n=2 | Table 3 marginals (§5.15) |

**Why they differ.** Max-minus-min grows with n by construction, so the single
three-seed cell inflates any floor computed over a family that includes it.
Five of the six originally replicated cells had a spread of 0.0022 or less; the
all-three cell alone had 0.0105.

**The rule applied throughout:** use the floor from the family a comparison is
drawn from, and when in doubt use the more conservative one. Where a single
figure is quoted for the project as a whole, it is 0.0105 — the most
conservative — never 0.0015.

### 5.24 The three grouping features

Same 16,365-class inventory, 16-dim width and projection path; they differ only
in what they merge. Baseline 1.3480 (n=2).

| feature | merges | n | seeds | mean BPB | gain |
|---|---|---|---|---|---|
| `mroot` | derivational + inflectional | 1 | 1.3076 | 1.3076 | 0.0403 |
| `mlemma` | inflectional only | 2 | 1.2895, 1.2914 | 1.2905 | 0.0575 |
| `mhash` | nothing (random) | 2 | 1.2875, 1.2881 | 1.2878 | 0.0601 |
| `mhash` + `mstruct` | | 2 | 1.2794, 1.2816 | 1.2805 | 0.0675 |
| `mlemma` + `mstruct` | | 3 | 1.2790, 1.2833, 1.2836 | 1.2820 | 0.0660 |

`mlemma`+`mstruct` vs `mhash`+`mstruct`: +0.0015, **0.7 sd** (sd used, not
max−min, because n differs).

### 5.25 Identity-precision curve (valid, `coarsen_semantics` = 2)

`mhash` folded to k classes at model time. All runs seed 0 except k=16,365.

| k | BPB | gain |
|---|---|---|
| 64 | 1.3251 | 0.0229 |
| 256 | 1.3174 | 0.0306 |
| 1,024 | 1.3125 | 0.0354 |
| 4,096 | 1.3006 | 0.0473 |
| 8,192 | 1.2966 | 0.0513 |
| 16,365 | 1.2878 (n=2) | 0.0601 |

Slope ~0.0045 BPB per doubling, R² = 0.971. No saturation in the tested range.

**Continuity:** k=16,365 with coarsening on reproduced the uncoarsened seed-0
run at **1.2875168991005173, bit-identical at all 101 evaluation points.**

**Invalid first sweep** (clamp, not fold — quarantined in
`results_invalid/coarsen_clamp/`): k = 32, 64, 128, 256, 512 (×2), 1024, 2048,
4096, 8192. Not to be cited.

### 5.26 The grouping features measured on the curve

Effective random-class count giving the same gain, by interpolation between
measured neighbours. Range reflects ±0.002 seed noise.

| grouping | effective k | of nominal | range |
|---|---|---|---|
| `mroot` | ~1,800 | 11% | 1,436 – 2,288 |
| `mlemma` | ~13,200 | 81% | 11,309 – 15,489 |
| `mhash` | 16,365 | 100% | — |

A global straight-line fit is **not** usable here: it places `mhash` at 163% of
itself, because the curve steepens at the top.

### 5.27 Fold verification on real data (`check_fold.py`)

| k | observed largest class | expected f + (1−f)/k | old clamp |
|---|---|---|---|
| 64 | 8.21% | 8.53% | 99.7% |
| 512 | 7.15% | 7.26% | 97.8% |
| 4,096 | 7.08% | 7.10% | 81.2% |

f = 7.07% (measured). Criterion rejects the clamp at every k.

### 5.28 Within-word lookahead by target type (`lookahead_check.py`)

252,160 validation targets: word-initial 70.44%, continuation 13.44%,
other 16.12%. Baseline `a7175857fdf6`, all treatments seed 0.

| feature | describes | gain (BPB) | continuation share | ratio | non-leak |
|---|---|---|---|---|---|
| `mroot` | form | 0.0401 | 97.0% | 7.22 | 0.0012 |
| `mhash` | form | 0.0602 | 90.8% | 6.76 | 0.0055 |
| `mlemma` | form | 0.0582 | 91.4% | 6.80 | 0.0050 |
| `mstruct` | form | 0.0252 | 92.1% | 6.86 | 0.0020 |
| `mhash`+`mstruct` | form | 0.0683 | 89.4% | 6.65 | 0.0072 |
| `mlemma`+`mstruct` | form | 0.0687 | 86.5% | 6.43 | 0.0093 |
| `ectx` | role | 0.0131 | 31.5% | 2.34 | 0.0090 |
| `dep` | role | 0.0387 | −6.6% | −0.49 | 0.0412 |

The tool reproduces every known BPB gain exactly when seeds are paired
(seed-0 baseline against seed-0 treatment).

**`dep` and `ectx` are clear of the WITHIN-word leak only.** Both come from a
bidirectional spaCy pass, so cross-sentence lookahead remains untested (G-08).

### 5.29 The causal identity feature

`embedding.causal_shift`: every per-word fact takes the code of the previous
word, with boundaries from the highest-cardinality fact. Derived from cached
codes at model time — cache spec unchanged, label carries `-CAUSAL`.

| configuration | BPB | gain | |
|---|---|---|---|
| baseline seed 0 | 1.3477 | — | |
| `mlemma`+`mstruct` leaky | 1.2790 | 0.0687 | 5.10% |
| `mlemma`+`mstruct` causal *(v2, quarantined — see §5.32)* | 1.3392 | 0.0085 | 0.63% |

88% of the headline was lookahead. 0.0085 is 4x the 0.0021 noise floor.

**By target class** (baseline `a7175857fdf6`):

| class | tokens | base NLL | mean gain | share | ratio |
|---|---|---|---|---|---|
| word-initial | 70.44% | 4.7662 | **+0.0483** | 128.0% | 1.82 |
| continuation | 13.44% | 3.4049 | **−0.0698** | −35.2% | −2.62 |
| other | 16.12% | 2.4514 | +0.0120 | 7.3% | 0.45 |

Helps predict the next word, slightly hurts within-word continuations — the
same pattern as `dep` (+0.1487 / −0.0593), and the opposite of the leaky
signature (+0.0198 / +1.4186).

**QUARANTINED — `causal_partial`.** An earlier run of this cell (same run_id,
`a0abed930f58`, config identical) gated the shift on an attribute only
root-family blocks carry, leaving `mstruct` unshifted and still 92% leak. It
reported 1.3220 / 0.0257, of which **0.0172 was `mstruct` leak**. Runs now
record `causal_semantics` so the two are distinguishable. Not to be cited.

### 5.30 Routing estimate, full split (`coverage_economics.py` Part 4)

Clean words routed by training frequency; fallback words and punctuation always
BPE. All BPB over the full 1,136,862 validation bytes.

| threshold | to morph | routed BPB | vs pure BPE |
|---|---|---|---|
| none (pure BPE) | 0.0% | 1.3471 | — |
| freq < 3 | 3.4% | 1.3422 | −0.0050 |
| **freq < 10** | 6.7% | **1.3391** | **−0.0080 (0.60%)** |
| freq < 20 | 10.1% | 1.3392 | −0.0079 |
| freq < 30 | 12.7% | 1.3407 | −0.0064 |
| freq < 100 | 24.7% | 1.3537 | +0.0066 |
| pure morph | 100.0% | 1.5241 | +0.1770 |

Idealised **upper bound** — each specialist's cost without the cost of
combining them. A finer sweep (edges 1,3,10,20,30,50,75,100,300,1000) could not
raise the ceiling: the crossover sits at 10–20, exactly where the coarse edge
already was. **Not built**, by a rule set before the sweep.

Full-split frequency stratification, clean words only:

| band | bytes | morph BPB | bpe BPB | delta |
|---|---|---|---|---|
| 1–3 | 24,502 | 1.7877 | 2.0183 | **−0.2306** |
| 3–10 | 23,360 | 2.0716 | 2.2203 | **−0.1487** |
| 10–20 | 23,846 | 1.8811 | 1.8764 | +0.0047 |
| 20–30 | 18,529 | 1.7705 | 1.6760 | +0.0944 |
| 30–50 | 29,459 | 1.6614 | 1.4935 | +0.1680 |
| 100–300 | 120,988 | 1.5089 | 1.3437 | +0.1652 |
| 1,000+ | 276,888 | 1.4218 | 1.3757 | +0.0461 |

The 400-line figure of 12.8% becomes **9.0%** on 47,862 bytes.

### 5.31 Is `dep` an oracle? (G-22)

**Prefix-parse agreement — FAILED, not usable.** 3,000 validation sentences,
62,919 words. Coarse-label agreement 60.61%, fine 56.26%. Rejected because the
disagreement measures spaCy on fragments, not `dep`:

| words after it | n | coarse agreement |
|---|---|---|
| 0 | 3,000 | 100.00% |
| 1 | 3,000 | 94.70% |
| 2 | 3,000 | 45.83% |
| 3 | 2,978 | 62.76% |
| 4 | 2,949 | 66.50% |
| 5+ | 47,992 | 56.44% |

Non-monotonic (≈19 standard errors), and 28% of disagreements end in `root` —
the signature of a parser re-rooting a truncated fragment.

**Probe from the causal baseline — the usable measurement.**

| model | chance | emb | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|---|
| `a7175857fdf6` baseline | 0.455 | 0.701 | 0.736 | 0.754 | **0.785** | 0.778 |
| `c564d066df6b` dep | 0.455 | 1.000 | 0.999 | 0.998 | 0.995 | 0.975 |

Causal recovery **78.5%** of labels, **60.6%** of headroom. Upper bound on
future-dependent labels: **21.5%**, and that is loose in the safe direction.

Compare the word-form features: 86–97% of their **gain** on leak-reachable
positions.

### 5.32 Isolated causal contributions — the headline

All `CAUSAL_SEMANTICS = 3`, `boundary_source: tokenizer`. Each seed paired with
its own baseline seed (seed 0: 1.3477064872, seed 1: 1.3482297766).

| cell | run_id s0 | run_id s1 | BPB s0 | BPB s1 | gain s0 | gain s1 | mean gain |
|---|---|---|---|---|---|---|---|
| **`mstruct`** | `648675b80ba0` | `93556c318fcc` | 1.3404 | 1.3372 | +0.0073 | +0.0111 | **+0.0092** |
| `mhash` | `05904594143e` | `54b18728cdfd` | 1.3494 | 1.3405 | -0.0017 | +0.0077 | +0.0030 |
| `mlemma`+`mstruct` | `a0abed930f58` | `5658981a9dc3` | 1.3404 | 1.3406 | +0.0073 | +0.0077 | +0.0075 |
| `mhash`+`mstruct` | `1b0847ff7660` | `d5b4bc5ca05c` | 1.3434 | 1.3386 | +0.0043 | +0.0097 | +0.0070 |
| `mprefix`+`mstruct` | `122cf133747d` | — | 1.3417 | — | +0.0060 | — | — |

**Headline: morphological structure, 0.0092 BPB (0.68%), causal, n=2.**

Adding identity to `mstruct`, paired by seed:

| addition | seed 0 | seed 1 |
|---|---|---|
| + `mlemma` | -0.0000 | -0.0034 |
| + `mhash` | -0.0030 | -0.0014 |

`mstruct` alone vs `mhash` alone: +0.0091, +0.0034.

**Caveat.** `mhash` alone spreads 0.0094 across seeds, the largest of any
two-seed cell. Its standalone value is unresolved; that it adds nothing to
`mstruct` is resolved.

**Superseded.** §5.29's causal figures used the seed-0 baseline throughout and
a single seed; use this table.

### 5.33 Three paired seeds — the headline, G-25 and G-26

**G-26 — did the environment change?** Baseline seeds 0 and 1 predate
environment recording; every causal cell ran under torch 2.11.0. Re-run under
torch 2.11.0:

| seed | original | re-run (torch 2.11.0) | difference |
|---|---|---|---|
| 0 | `a7175857fdf6` 1.3477064871870 | `8ff1b23b16d0` 1.3477064871870 | 0 |
| 1 | `a12b91da4049` 1.3482297765577 | `70e02f2481d5` 1.3482297765577 | 0 |

Bit-identical. Training is deterministic and the originals are valid. Seed 2
(`f846a29a25a1`, **1.3573**) tracks seeds 0 and 1 until ~20k steps, then falls
steadily behind to +0.0096 with no spike — an unlucky seed.

**Paired gains**, each cell against the baseline at its own seed
(1.3477, 1.3482, 1.3573; mean 1.3511):

| cell | run ids (s0, s1, s2) | s0 | s1 | s2 | mean | % |
|---|---|---|---|---|---|---|
| **`mstruct`, full** | `648675b80ba0` `93556c318fcc` `c249334b1bdb` | +0.0073 | +0.0111 | +0.0176 | **+0.0120** | **0.89%** |
| `mstruct`, morphology only | `e9e017b480c0` `6388341bd574` `61dba34034ff` | +0.0050 | +0.0108 | +0.0165 | +0.0108 | 0.80% |
| `mstruct`, names only | `1a6059aded6f` `063318a88494` `ff9efc229f5c` | +0.0049 | +0.0096 | +0.0184 | +0.0110 | 0.81% |
| `mlemma`+`mstruct`, **leaky** | — | +0.0687 | +0.0650 | +0.0737 | +0.0691 | 5.12% |

Full `mstruct`: sd 0.0052, se 0.0030, mean/se 4.0, positive at every seed.

**G-25, cell to cell** — all in one environment, so no baseline enters:

| paired difference | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| morphology − names | −0.0001 | −0.0012 | +0.0019 | +0.0002 |
| full − morphology | −0.0024 | −0.0003 | −0.0011 | −0.0012 |
| full − names | −0.0025 | −0.0015 | +0.0008 | −0.0010 |

(Negative means the first cell has lower BPB.) The two halves are tied; each is
slightly below the full feature; the halves overlap rather than add.

**Seed-2 gains are all larger** — leaky and causal alike — because the seed-2
baseline is the unlucky one. That is pairing working as intended: the seed
effect moves every seed-2 comparison together.

---


### 5.34 The night of 26 September: inflection, `pos`, BLiMP at every seed

Records: `results/audits/blimp_eval_20260926T120531+0000.json` (existing
checkpoints, 67 paradigms), `…T213947…` (new cells), `morph_causality_check_
20260926T120401…`, `derive_lexical_codes_20260926T175858…`,
`temporal_profile_20260926T221031…`. Gain = baseline BPB − cell BPB at the
same seed.

| cell | run ids (seeds) | gain per seed | mean ± se | % |
|---|---|---|---|---|
| `mstruct` causal | `648675b80ba0` `93556c318fcc` `c249334b1bdb` `61156897cb7e` (0–3) | +0.0073 +0.0111 +0.0176 +0.0098 | **+0.0114 ± 0.0022** | 0.85% |
| `pos` causal | `06b2a3bb429b` `b0dc64d5199a` (0,1) | +0.0072 +0.0109 | +0.0090 ± 0.0019 | 0.67% |
| `mfeatlex` causal | `a9f4a2434cfd` `599e62772132` (0,1) | +0.0002 +0.0050 | +0.0026 ± 0.0024 | 0.19% |
| `mstruct`+`mfeatlex` | `0b1f6c39cde6` `615cf6796212` (0,1) | +0.0094 +0.0096 | +0.0095 ± 0.0001 | 0.70% |
| … against `mstruct` alone | | +0.0021 −0.0015 | +0.0003 ± 0.0018 | — |
| … `pos` against `mstruct` | | −0.0002 −0.0002 | −0.0002 | — |

Baseline seed 3: `2d8d8fbce97b`, 1.3478. Temporal profiles: `mstruct` GROWS
(34% of final at 5k, n=4); `pos` front-loaded (179% at 5k); `mfeatlex` FADES
(437%). Tagger right-context dependence (left-only vs full tagging, 55,282
words): `tag` 8.2%, `pos` 7.4%, `mfeat` 4.5%, `magr` 3.3%, `minfl` 1.6%.
Lexicon: 642,581 word types; majority bundle = in-context bundle for 94.08%.

BLiMP, paired against the baseline (accuracy points, mean ± se, seeds agreeing):
`mstruct` all paradigms +0.6 ± 0.4 (3/4); anaphor agreement +6.9 ± 2.5 (4/4) —
anaphor gender alone +14.6, +5.6, +21.2 at seeds 0–2, carried by the
morphology half; determiner–noun −0.4 ± 1.6 (2/4); subject–verb +0.5 ± 0.3
(3/4). `pos`: all +1.4 ± 0.1 (2/2), anaphor −5.2 (2/2), determiner–noun +1.2
(2/2). `mfeatlex`: determiner–noun −1.8 ± 0.4 (2/2). Scorer deterministic: 536
paradigm accuracies identical across the two passes.


### 5.35 The day of 27 September: the controls

| cell | run ids (seeds 0, 1) | gain per seed | mean | profile |
|---|---|---|---|---|
| `mstructshuf` | `e189f96ecd25` `ee684acb4e9a` | +0.0081 +0.0129 | +0.0105 | GROWS (8% at 5k) |
| `poslex` | `85fc254b2c6d` `2a508f0432d6` | +0.0098 +0.0064 | +0.0081 | front-loaded (156% at 5k) |

Shuffle fairness (derive record `…20260926T231039…`): class changed for 47.7% of
types and 32.8% of tokens; token-level class shares moved 0.09%; 83.2% of tokens
outside `simple` changed class. `poslex` lexicon: 642,581 types, 93.2%
agreement with in-context tags, known words all correct.

Lookahead split (records `lookahead_check_20260926T2309…`), share of each
cell's gain by target type (token shares: word-initial 70.4%, continuation
13.4%): `mstruct` word-initial 72.1% / 70.7%, continuation 20.8% / 20.1%;
`pos` word-initial 124.9% / 105.4%, continuation −36.3% / −14.7%.


### 5.36 The null at every seed, and its width

Gain over the baseline (BPB), paired by seed:

| cell | run ids | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|---|
| null, 8 dims (`mconst`) | `c8fe0eb7cfe8` `8ba64cd4a185` `0096ec5ce160` `4b388cef56ed` | +0.0076 | +0.0101 | +0.0185 | +0.0082 |
| null, 16 dims | `d261f11e8c7e` `8b7b7ac89cd8` | +0.0092 | +0.0053 | | |
| null, 32 dims | `0e89c4564f5d` `15cc32c37ce9` | +0.0114 | +0.0113 | | |
| null, 64 dims | `ac20641c273b` `ffe63b490e43` | +0.0112 | +0.0088 | | |
| `mfreq` | `10b7adbeef86` `8ba5756b2ccf` | +0.0079 | +0.0110 | | |

Information effect = null BPB − cell BPB at the same seed: structure −0.0002,
+0.0010, −0.0009, +0.0015 (mean +0.0004 ± 0.0005); shuffle +0.0006, +0.0028;
`mfreq` +0.0004, +0.0009; `pos` −0.0004, +0.0008; `poslex` +0.0023, −0.0037;
`mfeatlex` −0.0074, −0.0051. Temporal profile of the null: GROWS, 30% of its
final gain at 5k steps, 54% at 10k (n=4). Training loss, last 10k steps,
null minus baseline: −0.010, −0.021, −0.030, −0.017 nats at seeds 0–3.


### 5.37 What the slot does

| cell | run ids (seeds 0, 1) | over the baseline | over the null |
|---|---|---|---|
| `mzero` (exactly zero) | `070773989bad` `6eede6137dc8` | +0.0083 +0.0071 | +0.0007 −0.0030 |
| `mflat` (one vector everywhere) | `72b0e10a9502` `6b2a1889bbd6` | +0.0069 +0.0114 | −0.0007 +0.0013 |

BLiMP (record `blimp_eval_20260927T172247…`, 4 seeds, accuracy points against
the baseline): the null +0.8 ± 0.3 overall, structure +0.6 ± 0.4; structure
minus null −0.2 overall (ahead at 1/4), anaphor +1.0 +0.3 +0.4 +13.6 (4/4),
ellipsis +4.5 +5.4 +5.4 −2.2. Anaphor-gender item split: the null +15.1 on
*herself > itself*, −26.9 on *itself > herself*; structure +30.6, −38.5.

