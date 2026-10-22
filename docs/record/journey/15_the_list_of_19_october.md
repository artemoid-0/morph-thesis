# Research journey — Part 15: The List of 19 October: Case and the Null at 30M Parameters

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.117 onward. [← Part 14](14_two_machines.md) · [index](README.md)

**Established so far:** over eight seeds of English at full training, keeping endings apart is
better on BLiMP's subject–verb and anaphor agreement at six seeds of eight, on determiner–noun
agreement at four (§33.117). Read on the law's axis, case gains less where endings are exposed only
in a middle stretch of training, in models of 5M and 30M parameters (§33.117–§33.118). At 30M
parameters the same-shape null costs nothing; in the 5M model the tokenizer's gain over the empty
slots is +0.0203, not +0.0175 (§33.118). At 85M parameters (width 1024) and 10k the tokenizer's gain
is half width 512's on average, +0.0067 ± 0.0030: two seeds keep width 512's level, two lose it
early in training (§33.120). Its grammatical gain holds at every seed (+5.3 ± 0.4 points) and is no larger than
30M parameters' on the same machine; the two seeds' fall is a slower stretch, no spike (§33.122). At 20k the gain is smaller again and lasts,
+0.0075 ± 0.0018, and the split stays (§33.124). For gender and number the tagger's leak is spread
over word classes, as for English part of speech, so their control is a causal tagger (§33.125). Removing
English part of speech leaves a deficit that closes by 100k, and at 30M parameters Unigram's shortfall
goes (§33.126). Gender's and number's lead over the word form is about 0.001 at 10k, and little of it
is the leak (§33.127). The rest is not context: most of it sits after words unseen in training,
where the word form misleads its model (§33.128–§33.129).

---

### 33.117 The text's list of 19 October: BLiMP's agreement at eight seeds, the exposure comparison on the law's axis, and item 4 prepared

The thesis's list of 19 October ranks five items. In its order: the tokenizer's gain beyond 30M
parameters (width 1024); number and gender as context-aware features; whether English part of
speech's deficit closes; the exposure redundancy in a wider model; and BLiMP's agreement phenomena
over all eight English seeds. Items 4 and 5 run now. The others are ranked and costed in DIRECTIONS'
queue.

**Item 5: BLiMP's agreement phenomena at eight seeds** (`blimp_eval_20261012T043938`, seeds 0–3;
`blimp_eval_20261017T094816`, seeds 4–7 on the second machine; the full-training plain baselines,
endings apart minus BPE 17k, points):
- determiner–noun −0.3, +3.2, +2.9, +6.0, −4.1, +2.8, −1.2, −0.0: **+1.2 ± 1.1 (4/8)**;
- subject–verb +1.5, +0.7, −0.3, +3.2, +1.3, +0.4, +3.2, −1.2: **+1.1 ± 0.5 (6/8)**;
- anaphor +6.2, +9.5, −3.9, +6.0, +0.7, +7.1, +7.5, −2.4: **+3.8 ± 1.8 (6/8)**.

Every agreement phenomenon is better on average, but determiner–noun agreement only at half the
seeds: seeds 4–7 give −0.6 ± 1.4. All 67 paradigms stay +1.0 ± 0.2 (8/8), morphology +1.1 ± 0.6 (6/8).
The text's rule for this item: the clause stands as «x/8» where the direction holds. So it stands for
subject–verb and anaphor agreement (6/8 each), and not for determiner–noun agreement (4/8).

**The exposure comparison on the law's axis** (`exposure_axis_20261019T035335`, against the empty
slot; `…T035337`, against the null; new tool `scripts/analysis/exposure_axis.py`; the second
machine's runs at width 128, ~38M, 10k, seeds 0–3). Item 4 compares widths on the law's axis, so
width 128 was read the same way first. Each tokenizer's gain is read at its reference's bits per
byte along training, interpolated per seed at common levels, and compared with BPE 17k's, paired by
seed.

| case over the empty slot, BPE 17k minus | reference BPB 1.27–1.28 | 1.18–1.19 | 1.09 (BPE 17k's end) | end of the 10k budget |
|---|---|---|---|---|
| endings apart | +0.0017 ± 0.0010 (3/4) | −0.0001 ± 0.0010 | **+0.0002 ± 0.0011** | +0.0016 ± 0.0013 (3/4) |
| Unigram 17k | +0.0017 ± 0.0005 (4/4) | +0.0016 ± 0.0003 (4/4) | **+0.0020 ± 0.0007 (3/4)** | +0.0020 ± 0.0008 (3/4) |

Earlier in training (reference BPB 1.37–1.57) both exposing tokenizers gain more than BPE 17k, by
0.0005 to 0.0080. Endings apart learns faster, and its models are further along at a given step:
its empty slot ends at ~1.07 where BPE 17k's ends at 1.09. Measured at the same point of learning,
its fixed-budget shortfall is gone late in the budget. Unigram learns no faster than BPE 17k (its
tokenizer gains nothing), and its shortfall stays at the same point of learning. Over the null,
case gains more under both exposing tokenizers late in the budget, by 0.0025–0.0035 at reference
BPB 1.18 and 1.09, as FINDINGS stated first: the null's cost there (§33.113).

**What it changes for the thesis.** The redundancy sentence of §33.113 holds at a fixed budget. At
the same point of learning it holds under Unigram (about 0.002, late in the budget) and not under
endings apart: there the fixed-budget shortfall is the head start, and a feature's gain falls as a
model learns (the law). So the sentence must name the axis.

**Item 4 prepared** (`scripts/chains/second_2026-10-19.sh`; rehearsed with the stub in seven
scenarios, all passing). Width 512, ~38M tokens, 10k, seeds 0–3, under BPE 17k and endings apart:
the null (eight twins of the thesis machine's, its run ids kept), case and the empty slot. 24 runs,
about 16 minutes each on the second machine (6.4 h). Read with `exposure_axis.py --widths 128 512`.

Predictions, before any run:

- **P264** (~60%). At width 512, at the lowest reference BPB both tokenizers' curves share, case's
  gain over the empty slot under endings apart is within 0.001 of BPE 17k's: level at the same
  point of learning, as at width 128 (+0.0002 ± 0.0011).
- **P265** (~70%). At the end of the 10k budget at width 512, case's gains over the empty slot
  under the two tokenizers differ by less than 0.001. At width 128 the gap was +0.0016 at 10k and
  +0.0002 at 20k; width 512 at 10k is further along than either.
- **P266** (~55%). At width 512 and 10k the empty slot ends better than the null under endings
  apart by more than 0.001, and within 0.001 of it under BPE 17k: the null's cost where endings are
  exposed holds at 30M parameters.

**The sentences these decide.**
- *Level at both sizes on the law's axis:* «за однакового поступу навчання відмінок додає однаково
  за обох токенізаторів у моделях на 5 і 30 млн параметрів; за фіксованого бюджету менше там, де
  закінчення відкриті, бо ці моделі вже далі». *Endings apart short at width 512 on that axis too:*
  «у більшій моделі відкриті закінчення частково дублюють відмінок і за однакового поступу».
- *P266:* method point 10 is stated for both sizes, or for the 5M model only.

### 33.118 Item 4: at 30M parameters case gains alike under both tokenizers, and the null costs nothing

Item 4 of §33.117 ran on the second machine on 19 October (`scripts/chains/second_2026-10-19.sh`):
24 runs in 6.5 hours, about 16 minutes each, none failed. Width 512, ~38M tokens, 10k, seeds 0–3,
under BPE 17k and endings apart: the null (twins of the thesis machine's, their run ids kept), case
and the empty slot. Every run's best step was its last. Bits per byte, seeds 0–3:

| width 512, 10k | BPE 17k | endings apart |
|---|---|---|
| the null (the thesis machine's) | 0.8790, 0.8818, 0.8798, 0.8771 (0.8803, 0.8790, 0.8770, 0.8777) | 0.8671, 0.8641, 0.8637, 0.8663 (0.8664, 0.8655, 0.8672, 0.8649) |
| case | 0.8765, 0.8800, 0.8776, 0.8755 | 0.8642, 0.8619, 0.8615, 0.8647 |
| the empty slot | 0.8788, 0.8833, 0.8790, 0.8775 | 0.8670, 0.8647, 0.8638, 0.8663 |

**Case against the empty slot, on the law's axis** (`exposure_axis_20261019T151643`, at each width's
own levels; `…T151927`, at common levels; the second machine). Case's gain over the empty slot under
BPE 17k minus its gain under endings apart, at the same reference BPB, paired by seed:

| reference BPB | width 128 | width 512 |
|---|---|---|
| 1.55 | −0.0080 ± 0.0011 (0/4) | — |
| 1.46 | −0.0039 ± 0.0012 (0/4) | — |
| 1.35–1.37 | −0.0005 ± 0.0009 (1/4) | −0.0037 ± 0.0001 (0/4) |
| 1.27 | +0.0017 ± 0.0010 (3/4) | +0.0008 ± 0.0002 (4/4) |
| 1.18 | −0.0001 ± 0.0010 (3/4) | **+0.0033 ± 0.0003 (4/4)** |
| 1.09 | **+0.0002 ± 0.0011 (3/4)**, the lowest both reach | −0.0000 ± 0.0007 (2/4) |
| 0.98 | — | −0.0006 ± 0.0005 (2/4) |
| 0.88 | — | **+0.0001 ± 0.0004 (2/4)**, the lowest both reach |
| the end of the 10k budget | +0.0016 ± 0.0013 (3/4) | **−0.0001 ± 0.0003 (2/4)** |

At width 512 case over the empty slot ends at **+0.0022 ± 0.0004 under BPE 17k and +0.0024 ± 0.0003
under endings apart** (4/4 each). Along training, under BPE 17k its gain falls slowly, +0.0092 at
reference BPB 1.35, +0.0068 at 1.18, +0.0042 at 1.09; under endings apart it falls early, +0.0128,
+0.0035, +0.0042. **P264 right, P265 right.**

At both widths the order is the same. Early in training case gains more where endings are exposed.
In a middle stretch it gains less: at width 512 by 0.0033 ± 0.0003 at reference BPB 1.18, 4/4; at
width 128 by 0.0017 ± 0.0010 at 1.27, 3/4. From reference BPB 1.09 to the end of the budget the two
are level. The 10k budget takes width 512 past that stretch (its references end at 0.88 and 0.87,
below width 128's 1.09 and 1.07), so at a fixed budget it shows no shortfall either. Width 128's
fixed-budget shortfall at 10k (§33.113) is the stretch, and it is gone by 20k (§33.116). Over the null
(`exposure_axis_20261019T151704`, `…T151930`), case at width 512 gains +0.0020 ± 0.0002 under BPE 17k
and +0.0022 ± 0.0003 under endings apart (4/4 each), level too (−0.0002 ± 0.0001). At width 128 over
the null endings apart gains more late in the budget, by 0.0035 at reference BPB 1.18 and 0.0025 at
1.09 (0/4 each), by the null's cost there.

**The empty slot against the null** (`device_twins_20261019T151746`, `…T152231`; along training,
`exposure_axis_20261019T152210`, the new option `--feature empty`). At width 512 the empty slot ends
level with the null under both tokenizers: under endings apart +0.0001, −0.0006, −0.0001, −0.0000,
**−0.0002 ± 0.0001 (1/4)**; under BPE 17k +0.0002, −0.0015, +0.0008, −0.0004, −0.0002 ± 0.0005 (2/4).
**P266 half right:** within 0.001 under BPE 17k. Under endings apart the empty slot does not beat
the null; at width 128 it did, by +0.0022 ± 0.0007. Along training, the empty slot over the null at
the same null BPB:

| null BPB | endings apart, width 128 | endings apart, width 512 | BPE 17k, width 128 | BPE 17k, width 512 |
|---|---|---|---|---|
| 1.35 | −0.0047 ± 0.0010 (0/4) | −0.0046 ± 0.0009 (0/3) | −0.0049 ± 0.0005 (0/4) | −0.0021 ± 0.0003 (0/4) |
| 1.25 | −0.0023 ± 0.0004 (0/4) | −0.0031 ± 0.0006 (0/4) | −0.0037 ± 0.0003 (0/4) | −0.0024 ± 0.0004 (0/4) |
| 1.15 | **+0.0017 ± 0.0008 (3/4)** | **−0.0015 ± 0.0005 (0/4)** | −0.0019 ± 0.0006 (0/4) | −0.0030 ± 0.0008 (0/4) |
| 1.10 | **+0.0021 ± 0.0007 (4/4)** | **−0.0012 ± 0.0005 (0/4)** | −0.0009 ± 0.0004 (1/4) | −0.0022 ± 0.0005 (0/4) |
| 1.00 | — | −0.0013 ± 0.0003 (0/4) | — | −0.0008 ± 0.0004 (1/4) |
| 0.89 | — | −0.0002 ± 0.0003 (2/4) | — | −0.0001 ± 0.0005 (2/4) |
| the end of the 10k budget | +0.0022 ± 0.0007 (4/4) | −0.0002 ± 0.0001 (1/4) | −0.0006 ± 0.0004 (1/4) | −0.0002 ± 0.0005 (2/4) |

Early in training the null's slot helps under both tokenizers at both widths. In the 5M model under
endings apart that help turns into a cost by null BPB 1.15. In the 30M model it never turns, and fades
to nothing by the end of the budget. At the same null BPB the small model's slot costs and the wide
model's helps: the cost belongs to the small model, not to a stage of learning.

**The tokenizer's gain over each reference** (`device_twins_20261019T152231`; the second machine, ~38M,
10k). Endings apart over BPE 17k:
- width 128: over the nulls +0.0175 ± 0.0015 (the thesis machine's +0.0174 ± 0.0014); over the empty
  slots **+0.0203 ± 0.0013** (4/4), more at every seed, by 0.0028 on average;
- width 512: over the nulls +0.0141 ± 0.0016 (the thesis machine's +0.0125 ± 0.0009); over the empty
  slots +0.0142 ± 0.0017 (4/4).

The Ukrainian tokenizer comparisons in FINDINGS are between the nulls. At width 128 the null's cost
under endings apart makes that gain the smaller one, so the 5M model's figures are conservative. At
width 512 the two references agree, and its gain is below width 128's over either.

**Twins** (`device_twins_20261019T152231`). The eight nulls repeated at width 512 differ from the
thesis machine's by −0.0012, +0.0028, +0.0027, −0.0006 (BPE 17k) and +0.0007, −0.0014, −0.0034,
+0.0014 (endings apart); at step 500, by at most 0.0020. Their differences spread more than the
thesis machine's seeds do (SD 0.0021 and 0.0022, against 0.0014 and 0.0010), so at this width dropout's
draw moves a run about as much as its seed. The effects still agree: the tokenizer's gain +0.0125 ±
0.0009 and +0.0141 ± 0.0016. Over all 143 pairs the difference is −0.0001 ± 0.0001, 71 above and 72
below; the largest is 0.0034 at 10k, 0.0044 at 20k and 0.0063 at 50k.

**What it changes for the thesis.**
1. **The exposure sentence holds at both sizes, by stretch of training.** At the same point of
   learning, late in the budget, case gains as much under endings apart as under BPE 17k in models of
   5M and 30M parameters. Where endings are exposed it gains less only in a middle stretch of
   training, after gaining more early. At a fixed budget the shortfall is the 5M model's at 10k; at
   width 512 a 10k budget is past the stretch. Of §33.117's outcomes this is the first (level at both
   sizes on the law's axis); its fixed-budget clause is the 5M model's.
2. **Method point 10 is the small model's.** In the 30M model the null's slot costs nothing under
   either tokenizer: it helps early and fades. The null is a fair reference there, as the tokenizer
   comparison and width 1024 (queue item 1) use it.
3. **The tokenizer's gain is conservative in the 5M model.** Over the empty slots it is +0.0203, not
   +0.0175. Width 512's gain is the same over either reference, and smaller.
4. **Unigram at width 512 stays a follow-up.** Under Unigram at width 128 the shortfall stays at the
   same point of learning (§33.117). Whether it does at 30M parameters would decide that clause for
   both sizes: 12 runs on the second machine, ~3.2 h, if the text asks for it (DIRECTIONS).

**Predictions** (§33.117). P264 and P265 right; P266 half right. The tally is 157 right, 11 half
right, 89 wrong, 8 never tested and 1 withdrawn, of 266; none pending.

### 33.119 Item 1 prepared: the tokenizer's gain at width 1024

The thesis text's list of 19 October ranks it first (§33.117): does keeping endings apart still help
beyond 30M parameters? At width 512 the gain was smaller than at width 256. It fell along the law's
axis, +0.041 at null BPB 1.18 and +0.012 at 0.89, then levelled off at 20k at +0.0102 ± 0.0006
(1.25%; §33.94–§33.102). Item 1 asks the same at width 1024: the null under BPE 17k and under endings
apart, ~38M tokens, seeds 0–3, at 10k and at 20k, read on the law's axis as at width 512.

**The fit check** (`scripts/ops/fit_check.py` on the second machine): width 1024 has 85.3M parameters
(34.9M in the embeddings and head, 50.4M in the core). It peaks at 7.2 GB and runs 4.0 steps a
second, about 43 minutes per 10k run. Width 512 measures 15.9 minutes there, as its recorded runs
took.

**The chain** (`scripts/chains/second_2026-10-20.sh`, the first on `scripts/chains/lib/chain_v1.sh`):
- **Order:** the 10k half first (8 runs, ~5.8 h), then the 20k half (8 runs, ~11.5 h) until END_BY.
  What END_BY leaves runs on a later night.
- **Guards:** a run starts only if it can end before END_BY, and finished runs are never run again.
  A sanity check sits on the first seed's pair.
- **No twins:** no width-1024 run exists.
- **The two halves are separate runs.** A 10k run cannot be extended to 20k: its learning rate
  anneals to its end at 10k, as at every width.

Predictions, before any run (the second machine; width 512 there is item 4's twins, §33.118):

- **P267** (~85%). At 10k, width 1024's tokenizer gain (endings apart's null over BPE 17k's, paired
  by seed) is positive at every seed.
- **P268** (~65%). It is smaller than width 512's on the same machine at 10k (+0.0141 ± 0.0016,
  §33.118) on average: the gain keeps shrinking with size beyond 256.
- **P269** (~55%). On the law's axis, at every null BPB both widths reach between 0.88 and 1.04,
  width 1024's gain is within 0.004 of width 512's, on average over seeds. Beyond width 512 the gain
  would then depend on how far the model has learned, not on its size.
- **P270** (~70%). Width 1024's BPE 17k null ends its 10k budget at least 0.03 BPB below width
  512's (0.8794 on average, §33.118).
- **P271** (~55%). At 20k the gain levels off. Over the budget's last 5,000 steps it changes by less
  than 0.002 on average, as width 512's did from about step 7,500.

**The sentences these decide** (the text's, §33.117):
- smaller at 10k and level at 20k (P268 and P271) → *smaller again but lasting*;
- smaller and still falling at 20k → *it fades*;
- not smaller → *it holds at 512's level*.

### 33.120 Item 1's 10k half: at 85M parameters the gain halves on average, because two seeds of four lose it early

The chain of 20 October (§33.119) ran on the second machine from 20:40 to 05:32 EDT: the 10k half (8
runs of 44 minutes), then the 20k pair of seed 0 (88 minutes each). END_BY then left the 20k pairs of
seeds 1–3 (6 runs, ~8.8 h) for a later night. No run failed, the sanity check passed in both halves,
and every run's best step was its last. Width 1024, ~38M tokens, 10k, bits per byte:

| width 1024, 10k | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| BPE 17k | 0.8222 | 0.8244 | 0.8204 | 0.8233 | 0.8226 |
| endings apart | 0.8095 | 0.8248 | 0.8162 | 0.8131 | 0.8159 |
| the gain | +0.0127 | −0.0004 | +0.0042 | +0.0103 | **+0.0067 ± 0.0030 (0.81%, 3/4)** |

On the same machine width 512's gain at 10k is +0.0141 ± 0.0016 (1.61%, 4/4; §33.118). Width
1024's BPE 17k null ends 0.0568 below width 512's 0.8794 (6.5%), its endings-apart null 0.0494 below
0.8653. At 20k, seed 0: BPE 17k 0.7700, endings apart 0.7585, **+0.0115 (1.50%)**.

**On the law's axis** (`tokenizer_widths_20261020T131612`; the script's new option `--machine second`
reads the second machine's records, width 512's twins included, and each seed's own curve gives the
spread). The lead at the same null BPB, at 10k:

| null BPB | width 512 | width 1024 | width 1024, seeds 0, 1, 2, 3 |
|---|---|---|---|
| 1.30 | +0.0483 | +0.0587 ± 0.0013 (4/4) | +0.0616, +0.0597, +0.0571, +0.0561 |
| 1.18 | +0.0409 | +0.0404 ± 0.0019 (4/4) | +0.0436, +0.0410, +0.0348, +0.0420 |
| 1.11 | +0.0305 | +0.0276 ± 0.0021 (4/4) | +0.0310, +0.0267, +0.0220, +0.0304 |
| 1.04 | +0.0226 | +0.0158 ± 0.0030 (4/4) | +0.0209, +0.0130, +0.0086, +0.0205 |
| 0.98 | +0.0198 | +0.0061 ± 0.0044 (2/4) | +0.0131, −0.0022, −0.0018, +0.0134 |
| 0.95 | +0.0187 | +0.0041 ± 0.0050 (2/4) | +0.0125, −0.0068, −0.0029, +0.0121 |
| 0.92 | +0.0162 | +0.0036 ± 0.0050 (2/4) | +0.0125, −0.0082, −0.0014, +0.0113 |
| 0.89 | +0.0139 | +0.0056 ± 0.0047 (3/4) | +0.0134, −0.0056, +0.0005, +0.0130 |
| 0.88 | +0.0141 | +0.0054 ± 0.0039 (3/4) | +0.0131, −0.0043, +0.0024, +0.0103 |
| 0.84 | — | +0.0047 ± 0.0034 (3/4) | +0.0118, −0.0030, +0.0014, +0.0087 |

**Two seeds hold, two fall behind.** At seeds 0 and 3 the lead falls to about +0.012 by step 3,000
(null BPB ~0.97) and stays between +0.0087 and +0.0136 to the end. At seeds 1 and 2 it falls below zero
between steps 2,500 and 3,000. Seed 1's reaches −0.0084 at step 4,500 and seed 2's −0.0029 at step
3,000. Over the rest of the budget they climb back by +0.0080 and +0.0071, still rising at its end
(by +0.0020 and +0.0018 over the last 1,000 steps).

**The fall is the endings-apart null's.**
- **Spread:** across seeds that null's BPB spreads over 0.0153 (SD 0.0066), against 0.0040
  (SD 0.0017) under BPE 17k. At width 512 on the same machine the two spread over 0.0034 and 0.0047.
- **Seed 1 against seed 0:** the two runs of a seed start from the same initial weights, and both
  of seed 1's start slower. Its BPE 17k null is behind by up to 0.0203 (step 2,500) and level by step
  6,000 (0.0014; 0.0022 at the end). Its endings-apart null falls further behind, from 0.0126 at step
  1,500 to 0.0320 at step 3,500, and makes up only half by the end (0.0153).
- **Its shape:** no run spiked. Every run's validation BPB fell at every evaluation (every 500
  steps). Seed 1's endings-apart null fell 0.006 less than seed 0's in each of three evaluations in a
  row (steps 1,500–3,000), and from step 3,500 it fell faster.
- **Where:** seed 1's endings-apart model is behind seed 0's on tokens of every frequency, most on
  the rarest (the records' per-frequency buckets): by 0.109 nats on the rarest sixth of the validation
  tokens, by 0.012 on the commonest.

What sets two endings-apart runs back is not known from these records (G-81).

**At 20k, seed 0** levels off: +0.0119 at step 15,000 and +0.0115 at the end, between +0.0090 and
+0.0125 over the last 5,000 steps. At the same null BPB its lead is within 0.003 of its 10k run's down
to 0.84 (+0.0214 and +0.0209 at 1.04, +0.0128 and +0.0125 at 0.95, +0.0092 and +0.0118 at 0.84).
From 0.86 to 0.82 it is within 0.002 of width 512's 20k runs on the same machine (+0.0109, +0.0092,
+0.0110 against +0.0111, +0.0091, +0.0090).

**Predictions** (§33.119).
- **P267 wrong:** at seed 1 the gain is −0.0004.
- **P268 right:** +0.0067 ± 0.0030, against +0.0141 ± 0.0016. The mean is smaller because two seeds
  lose the lead early; the other two hold width 512's level (+0.0127, +0.0103).
- **P269 wrong:** from null BPB 1.04 to 0.88 width 1024's lead is 0.007 to 0.015 below width 512's
  (−0.0068 at 1.04, −0.0147 at 0.95, −0.0088 at 0.88). It starts above it (+0.0104 at 1.30) and is
  level at 1.18. Seeds 0 and 3 alone come within 0.004 of it at 1.04 and at 0.89–0.88, and are
  0.004–0.007 below it between.
- **P270 right:** 0.8226, 0.0568 below 0.8794.
- **P271 pending:** the 20k half has one seed so far.

**Predictions for the rest of the 20k half** (seeds 1–3, before any of its runs). On this machine a
20k run shares its 10k run's initial weights, batches and dropout masks, and differs only in the
learning rate's schedule: at step 500 seed 0's two runs agree to 0.0001.
- **P272** (~55%). In the 20k runs of seeds 1 and 2 the lead again falls below +0.003 at some
  evaluation between steps 2,500 and 5,000. If it does not, the fall at 10k was chance within
  training, not the seed's.
- **P273** (~50%). At 20k the four seeds' gain is within 0.003 of width 512's at 20k on the same
  machine (+0.0099 ± 0.0006, seeds 1–3).

**The sentences the 20k half decides** (§33.119's, now that at 10k the gain is smaller):
- level at 20k and within 0.003 of width 512's (P271, P273) → *by 20k it holds at 30M parameters'
  level; at 10k two seeds of four had not yet regained it*;
- level and below → *smaller again but lasting*;
- still falling → *it fades*.

**What it changes for the thesis.**
1. **Beyond 30M parameters the gain is smaller at a 10k budget, and depends on the run.** At 85M
   parameters it is +0.0067 ± 0.0030 (0.81%, 3/4), half width 512's on the same machine. Two seeds
   keep width 512's level; two lose the gain early in training and regain part of it by the end of the
   budget. A single run at this width would not tell which.
2. **Whether it lasts, and at what level, is the 20k half's.** That is 6 runs, ~8.8 h on the second
   machine. Seed 0 at 20k levels off at +0.0115, width 512's level at the same null BPB.
3. **Width 1024 is not undertrained at 10k.** Its BPE 17k null ends 6.5% below width 512's, so the
   smaller mean is not a model that failed to learn.

**Predictions** (this section): P267 and P269 wrong, P268 and P270 right. The tally is 159 right, 11
half right, 91 wrong, 8 never tested, 1 withdrawn and 3 pending (P271–P273), of 273.

### 33.121 MultiBLiMP at width 1024, and a finer look at G-81 (prepared, 20 October)

Two short jobs on the second machine, under an hour together, while the rest of item 1 waits for its
night:
- **MultiBLiMP on the eight 10k models** of §33.120 (`scripts/eval/multiblimp_eval.py`). The text
  asked for it on the 20k models; at 10k it shows whether the grammatical gain follows the split in
  bits per byte. Width 512's eight 10k nulls on the same machine (item 4's twins) are scored in the
  same job, so the two widths are compared within one machine. About 30 seconds per eight models
  there (17 October).
- **G-81 at 100-step resolution:** seed 1's endings-apart null at width 1024 again, with the 10k
  run's schedule (`lr_horizon` 10,000) stopped at step 5,000, evaluating every 100 steps, early
  stopping off (`train.patience=none`). On this machine it shares the 10k run's initial weights,
  batches and dropout masks, and evaluation draws nothing at random. So it should repeat that run's
  first 5,000 steps exactly and show the fall's shape between its evaluations. Its label,
  `…_st5k_w1024`, comes from `max_steps`; it is not a 5k budget. About 27 minutes (132.9 s per 500
  steps and one evaluation in the 10k run; 50 evaluations).

Predictions, before either job:
- **P274** (~75%). At width 1024 and 10k, endings apart's models prefer the grammatical sentence
  more often than BPE 17k's at every seed (4/4).
- **P275** (~55%). Its mean gain in points is at least width 512's on the same machine at 10k.
- **P276** (~55%). The grammatical gain follows the split in bits per byte: seeds 1 and 2 each gain
  fewer points than seeds 0 and 3.
- **P277** (~80%). The finer run repeats the 10k run bit for bit: its validation BPB at steps 0,
  500, …, 4,500 equals the 10k run's to 1e-6.
- **P278** (~75%). No spike: from step 1,000 to 4,000 its validation BPB never rises by more than
  0.005 from one evaluation to the next.

**What each outcome would say.** If the grammatical gain holds at seeds 1 and 2 (P276 wrong), the
split is in predicting tokens, not in agreement: «за ~85 млн параметрів граматичний виграш не залежить
від запуску». If it follows the split, both measures depend on the run. A spike (P278 wrong) points
G-81 at the training recipe at this width; a smooth, slower stretch points it at the seed or at
chance, which the rest of the 20k half separates (P272).

### 33.122 MultiBLiMP at width 1024: the gain holds at every seed and stops growing; G-81's fall is no spike

The two jobs of §33.121 ran on the second machine on 20 October from 09:37 to 10:04 EDT, 26 minutes
in all (the log: `logs/second/`, the remote log started at 09:37:46).

**MultiBLiMP at 10k** (`multiblimp_eval_20261020T133750`, width 1024; `…T133839`, width 512's
twins, item 4's nulls on the same machine). Accuracy over the 2,744 pairs; endings apart's gain over
BPE 17k in points, paired by seed:

| 10k, the second machine | BPE 17k | endings apart | the gain | seeds 0, 1, 2, 3 |
|---|---|---|---|---|
| width 512 | 73.54% | 79.46% | +5.9 ± 0.3 (4/4) | +6.01, +6.09, +6.45, +5.10 |
| width 1024 | 76.64% | 81.95% | **+5.3 ± 0.4 (4/4)** | +5.07, +4.56, +5.28, +6.34 |

By phenomenon at width 1024: number +5.2 ± 0.8, gender +5.0 ± 0.7, person +5.6 ± 0.3 (4/4 each).
At width 512 they are +6.2 ± 0.7, +6.6 ± 0.4 and +5.2 ± 0.3.
- **Grammar does not follow the split in bits per byte.** Seed 1 has no lead in bits per byte
  (−0.0004) and still gains +4.6 points. Seed 2 (+0.0042) gains +5.3, more than seed 0 (+0.0127,
  +5.1). G-81's fall is in predicting tokens, most of all rare ones (§33.120), not in agreement.
- **The grammatical gain stops growing at 30M parameters.** On one machine, width 1024's gain is no
  larger than width 512's (−0.6 ± 0.5). On the thesis machine it grew from width 128 through 256 to
  512 (+4.3, +5.3, +6.8).
- **Between machines, grammar moves more than bits per byte.** Width 512's twins gain 0.89 ± 0.42
  points less than the thesis machine's models (+5.9 against +6.8), lower at every seed (−0.44,
  −0.44, −0.55, −2.15).
  - Their BPE 17k models are within 0.5 points of the thesis machine's at each seed. Their
    endings-apart models are 0.3 to 1.6 points lower.
  - In bits per byte the same twins differ by −0.0034 to +0.0014, with no consistent sign (§33.118).
  - So the widths are compared within one machine here too: against the thesis machine's +6.8,
    width 1024's +5.3 would have looked like a fall.

**G-81 at 100-step resolution** (run `e50b75e8fdd2`: seed 1's endings-apart null at width 1024, the
10k run's schedule to step 5,000, evaluated every 100 steps; 24 min 49 s, 3.4 steps a second with
51 evaluations; `scratch/g81_finer.py` reads it against its 10k run).
- **The rerun repeats its 10k run closely, but not bit for bit.** At step 0 the two agree exactly
  (2.608084410). From step 500 they differ by 1.3 × 10⁻⁶, and by at most 3.2 × 10⁻⁵ BPB up to step
  4,500 (at 2,000). Their single-batch training losses agree to 10⁻³. The code sets no deterministic
  mode, and the rerun also evaluates five times as often; which of the two moves the arithmetic is
  not separated. G-69's bit-for-bit reproduction (§33.43) was on the thesis machine, of a run with the
  same training settings.
- **No spike.** From step 1,000 to 4,000 the validation BPB falls at every evaluation. Over the
  stretch where seed 1 falls behind seed 0 (steps 1,500–3,000) it falls by 0.0048 to 0.0143 per 100
  steps, the drops shrinking on the whole as a run's curve flattens. No single-batch training loss stands more than
  0.31 nats above the mean of its neighbours (step 4,700; 0.28 at 2,900, 0.21 at 1,500), which is
  within the batch-to-batch scatter of the whole run.
- **So the fall is a slower stretch of learning, and on this machine it is the seed's.** Its
  initial weights, batches and dropout masks repeat it to 3 × 10⁻⁵ BPB, under arithmetic that differs
  from step 500 on: small numerical noise does not grow into the fall or out of it. Two questions stay
  open in G-81. Would another dropout draw (a twin on the thesis machine) fall too? Would another
  recipe at this width (a longer warm-up)?

**Predictions** (§33.121): P274 right; P275, P276 and P277 wrong; P278 right. The tally is 161 right,
11 half right, 94 wrong, 8 never tested, 1 withdrawn and 3 pending (P271–P273), of 278.

**What it changes for the thesis.**
1. **At 85M parameters the grammatical gain holds at every seed** (+5.3 ± 0.4 points at 10k), even
   where the gain in bits per byte is nearly gone. The FINDINGS sentence that the grammatical gain is
   not the code length's now rests on a split within one cell as well as across widths.
2. **It stops growing at 30M parameters**: on one machine 85M's gain is no larger than 30M's.
3. **The method point gains a caveat:** an effect measured on each machine agrees in bits per byte,
   but the grammatical gain moved by 0.9 points between machines at width 512.
4. **G-81 is narrowed, not closed:** no instability shows at 100-step resolution, and the fall belongs
   to the seed on this machine. Whether it belongs to the recipe at this width is still open.

### 33.123 The chain of 21 October prepared: MultiBLiMP at 20k, items 3 and 6

The second machine is ours for 24 hours from 13:38 EDT on 20 October. Item 1's 20k half (seeds 1–3,
6 runs) started then and ends about 23:00 EDT. `scripts/chains/second_2026-10-21.sh` (on
`lib/chain_v1.sh`) starts after it has let go of its lock, so the two never share the GPU. Item 2
(number and gender) follows once its preparation is done. In the text's order, it runs items 3 and
6 before item 2 only because item 2 is not ready.
- **mb20k:** MultiBLiMP on width 1024's eight 20k nulls (minutes), as the text asked.
- **en100k (item 3):** English part of speech never fed and switched off at 5,000, 100k, seeds 0–3:
  8 runs of ~42 min. At 50k every such run was still improving at its last step (0.007 BPB over the
  last 10k steps), so 6.8 passes over WikiText-103 should still be read at their best step.
- **uni512 (item 6):** Unigram 17k at width 512, ~38M, 10k, seeds 0–3: the null, case and the
  empty slot, 12 runs of ~16 min. Read on the law's axis against item 4's BPE 17k cells (§33.118).

PLAN on the second machine: 20 runs, ~9.7 h with margins, ending well before END_BY. The data for
every cell are there (`cells_ready.py`). Rehearsed with the stub in six scenarios
(`scratch/rehearse_1021/`): all of it, a failed sanity check, MultiBLiMP run and already recorded,
END_BY, a cell not ready.

Predictions, before any of its runs:
- **P279** (~65%). At 20k, width 1024's MultiBLiMP gain is positive at every seed and within 1.0
  point of its 10k gain (+5.3).
- **P280** (~50%). At 100k, the switched-off model's deficit against the never-fed one is below
  0.001 on average: the deficit closes. At 50k it was −0.0019 ± 0.0011, 3/4 below (§33.102).
- **P281** (~70%). Every never-fed run at 100k still improves over its last 10,000 steps (its best
  step is its last).
- **P282** (~65%). The deficit at 100k is smaller than at 50k.
- **P283** (~50%). At width 512, at the lowest common reference BPB, case's gain over the empty slot
  under Unigram is within 0.001 of BPE 17k's. At width 128 Unigram stayed +0.0016 ± 0.0003 short
  at 1.19 (§33.117).
- **P284** (~60%). At the end of the 10k budget, case's gains over the empty slot under Unigram and
  under BPE 17k differ by less than 0.001.
- **P285** (~60%). At width 512 the empty slot ends within 0.001 of the null under Unigram.

**The sentences these decide** (the text's): item 3 *persists* (P280 wrong) or *closes* (P280
right). Item 6 *stays*, if under Unigram exposed endings make case partly redundant at both sizes
(P283 wrong), or *goes*, if at 30M parameters the redundancy is gone under every exposing tokenizer
(P283 right).

### 33.124 Item 1's 20k half: at 85M parameters the gain is smaller again and lasts, and the split stays

The 20k pairs of seeds 1–3 ran on the second machine from 13:38 to 22:33 EDT on 20 October.
`second_2026-10-20.sh` was started again and skipped the finished runs. MultiBLiMP on the eight 20k
models then opened the chain of 21 October (§33.123). Bits per byte at 20k, each run at its best step:

| width 1024, 20k | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| BPE 17k | 0.7700 | 0.7703 | 0.7684 | 0.7701 (step 18,500) | 0.7697 |
| endings apart | 0.7585 | 0.7670 | 0.7626 | 0.7608 | 0.7622 |
| the gain | +0.0115 | +0.0033 | +0.0058 | +0.0093 | **+0.0075 ± 0.0018 (0.97%, 4/4)** |

Width 512 at 20k on the same machine: +0.0099 ± 0.0006 (1.21%, seeds 1–3; §33.102, §33.111).
- **It lasts.** Over the last 5,000 steps the gain changes by +0.0002 on average (−0.0004, +0.0005,
  +0.0006, +0.0002 at seeds 0–3).
- **The split stays, and it is the seed's.** At seeds 1 and 2 the 20k runs fall behind again as
  their 10k runs did: to −0.0091 at step 4,500 and to −0.0047 at step 3,000 (the 10k runs reached
  −0.0084 and −0.0029). From there they regain part of it, to +0.0033 and +0.0058. Seeds 0 and 3 end
  at +0.0115 and +0.0093, width 512's level.
- **The last 1,500 steps add about 0.0025.** From step 18,500, BPE 17k's null stops improving at
  every seed: its validation BPB rises at step 19,000, and seed 3's best step is 18,500.
  Endings apart's null keeps falling, by 0.0025 to 0.0031 from 18,500 to the end. By 20k the BPE
  17k model has read its ~38M-token text 4.3 times; endings apart's model, with 12% more tokens for
  the same text, 3.9 times. At 85M parameters this is where repetition begins to tell. At step 18,500,
  the last step every pair shares before a best step, the gain is +0.0049 ± 0.0018 (4/4).

**On the law's axis** (`tokenizer_widths_20261021T030542`; the lead at the same null BPB, at 20k).
From null BPB 1.11 down to 0.82, width 1024's lead is 0.003 to 0.015 below width 512's. It starts
above it.

| null BPB | width 512, 20k | width 1024, 20k | width 1024, seeds 0, 1, 2, 3 |
|---|---|---|---|
| 1.30 | +0.0493 | +0.0587 | +0.0617, +0.0598, +0.0571, +0.0560 |
| 1.18 | +0.0416 | +0.0403 | +0.0436, +0.0409, +0.0346, +0.0419 |
| 1.04 | +0.0224 | +0.0154 | +0.0214, +0.0129, +0.0069, +0.0204 |
| 0.95 | +0.0183 | +0.0036 | +0.0128, −0.0077, −0.0039, +0.0123 |
| 0.88 | +0.0102 | +0.0044 | +0.0115, −0.0055, +0.0029, +0.0091 |
| 0.84 | +0.0091 | +0.0031 | +0.0092, −0.0035, −0.0003, +0.0070 |
| 0.82 | +0.0090 | +0.0063 | +0.0110, +0.0005, +0.0023, +0.0102 |

**MultiBLiMP at 20k** (`multiblimp_eval_20261021T023320`).
- BPE 17k 81.1%, endings apart 85.8%: **+4.7 ± 0.3 points (4/4)**, at seeds 0–3 +4.19, +4.59,
  +4.45 and +5.39.
- By phenomenon: number +3.4 ± 0.6, gender +4.4 ± 0.1, person +5.4 ± 0.5 (4/4 each).
- At 10k the gain was +5.3 ± 0.4. Width 512 at 20k on the same machine gains +5.8 ± 0.3 (seeds 2–3,
  `multiblimp_eval_20261017T094841`).

**Predictions** (§33.119–§33.123).
- **P271 right:** +0.0002 over the last 5,000 steps on average.
- **P272 right:** −0.0091 and −0.0047, again between steps 2,500 and 5,000.
- **P273 right by its letter:** +0.0075 against +0.0099, 0.0024 apart. Not by its intent:
  - 0.0025 of the gain comes in the last 1,500 steps, as BPE 17k's null stops improving;
  - at step 18,500 the gain is 0.0050 below width 512's;
  - on the law's axis it is below width 512's from null BPB 1.11 down.
- **P279 right:** +4.7 ± 0.3 at every seed, 0.6 below the 10k gain.

The tally is 165 right, 11 half right, 94 wrong, 8 never tested, 1 withdrawn and 6 pending
(P280–P285), of 285.

**The sentence** (§33.120's rule). By its letter the rule picks the first sentence, *by 20k it holds
at 30M parameters' level; at 10k two seeds of four had not yet regained it*, because P271 and P273
are right. Neither clause fits what the runs show.
- At 20k, two seeds have still regained only part of the gain.
- What brings the mean within 0.003 of width 512's is the last 1,500 steps, where BPE 17k's null
  stops improving.

The second sentence fits: *smaller again but lasting*, with the split.
- It is smaller than at 30M parameters at both budgets, and at every common point of learning from
  null BPB 1.11 down.
- It is level over the last 5,000 steps.
- Two runs of four keep 30M's level; two lose much of the gain early in training and keep a smaller
  one.

**What it changes for the thesis.**
1. **Beyond 30M parameters the gain is smaller again, lasts, and depends on the run.** At 85M
   parameters it is +0.0067 ± 0.0030 at 10k and +0.0075 ± 0.0018 at 20k (0.97%, 4/4). Width 512 on
   the same machine gains +0.0141 and +0.0099. Two seeds keep width 512's level and two lose part of
   it early, at both budgets.
2. **At this size, ~38M tokens limit a 20k budget.** The 85M model's BPE 17k null stops improving
   after 4.3 passes over its text. A longer budget on this corpus would compare repetition as much as
   tokenizers.
3. **Grammar at 20k:** +4.7 points at every seed, below 30M's +5.8 on the same machine.
4. **G-81 holds at both budgets.** The 20k runs repeat the fall at the same seeds, and on this
   machine it belongs to the seed. Why those seeds fall is still open: another dropout draw, or
   another recipe.

### 33.125 Item 2 prepared: gender's and number's leak is spread over word classes, so their control is a causal tagger

Item 2 of the text's list (§33.117) asks whether Ukrainian number and gender, read as context-aware
features, behave like case. For case, what reading the sentence adds over reading the word form is
mostly the tagger's leak. The item is prepared on the second machine: its CPU is free while its GPU
trains, and the thesis machine is not used.

**Where the leak sits** (`morph_causality_check_20261021T030528`). The audit read 10,000 training
sentences (101,091 alphabetic words), each word tagged by the stream builder's pipeline in full and
from the prefix that ends at it. Two new options made this possible: `--by-type-attrs` counts gender
and number by word class, and `--stream` reads the text from the BPE 17k training stream, decoded.
Gender changes with the right context at 3.23% of words, number at 1.39%; case changes at 4.94% and
part of speech at 3.66%. By the word type's majority class:

| majority class | gender: share of its changes (rate) | number: share of its changes (rate) |
|---|---|---|
| adjectives | 40.8% (8.3%) | 26.3% (2.3%) |
| proper nouns | 24.5% (5.9%) | 16.9% (1.7%) |
| nouns | 18.4% (1.7%) | 25.7% (1.0%) |
| determiners | 10.7% (11.2%) | 13.4% (6.0%) |
| verbs | 0.5% (0.2%) | 5.8% (0.9%) |

- **Gender's changes** fall on 1,635 word types; the twenty commonest hold 24.2% of them.
- **Number's** fall on 554 types; the twenty commonest hold 39.9%.
- **Unlike case,** whose leak sat at prepositions so that a word list removed it (`ucaseprep`), this
  is spread like English part of speech's (§33.98). Reading adjectives and determiners from their
  form would remove 51.5% of gender's changes and 39.7% of number's.

The control is therefore a causal tagger, as `poscausal` is for English.

**The data on the second machine.**
- **The codes:** gender and number were cached for the 0.05 streams only on the thesis machine, in
  its `_ucaselex` codes files. The four files went across (176 MB) and were checked by SHA-256
  against the packing manifest after unpacking.
- **The text:** Ukrainian Wikipedia's text is not on the second machine; a first attempt to load it
  stalled on a download. Every stream decodes to its text byte for byte (G-75), so the audit and
  the causal tagger read the text from the stream.

**The word form** (`ugenderlex`, `unumberlex`; `derive_lexical_codes.py`;
`derive_lexical_codes_20261021T031915`, `…T031956`, `…T032051`, `…T032145`). Each word form takes its
majority value over the 0.05 training split: 1,138,752 word types under BPE 17k, 1,138,979 under
endings apart. That majority is the in-context value at 97.25% of gender's occurrences and 98.00% of
number's. Every known word (the personal pronouns, nouns of one gender) maps to its value.

**The causal tagger** (`ugendercausal`, `unumbercausal`; the new `scripts/data/derive_causal_ud.py`;
`derive_causal_ud_20261021T031855`, after two runs of its checks alone, `…T031542` and `…T031805`).
- **How:** each word is tagged in a window of its last five tokens, by the builder's pipeline
  without the parser. The parser never changed a word's gender or number: 22,208 of 22,208 words were
  the same in full context and in windows. One pass serves both streams.
- **Its checks, all before anything was written:**
  - the window, tagged without the parser, gives the same values as the whole prefix tagged with
    it, at 2,000 of 2,000 words;
  - the decoded paragraphs re-encode to both streams' cached ids, all 37,972,206 and 42,600,731
    training tokens;
  - in full context, the first 20,000 paragraphs of each split reproduce the cached codes exactly:
    the whole validation split (325,867 and 363,158 tokens) and 778,525 and 871,160 training tokens.
- **How often it differs:** the causal value equals the in-context one at 97.42% and 97.51% of
  training tokens for gender (BPE 17k, endings apart), 99.07% and 99.10% for number.
- **Cost:** 18 minutes for the training split, 16 workers, both streams in one pass.

**The chain** (`scripts/chains/second_item2_2026-10-21.sh`, on `lib/chain_v1.sh`).
- **The runs:** gender and number, each in context, from the word form and from the causal tagger,
  under BPE 17k and endings apart. ~38M tokens, 10k, seeds 0–3: 48 runs of ~4.2 min, about 3.4 h.
- **When:** it starts once the chain of 21 October lets go of its lock (§33.123).
- **The references:** the null and the empty slot at 10k under both tokenizers are on this machine
  already (G-78).
- **Rehearsed:** with the stub in five scenarios. Each new cell also ran 30 steps through the real
  loader, on the CPU.

Predictions, before any run:
- **P286** (~70%). At 10k, gender read in context leads gender from the word form under both
  tokenizers (≥3/4 each).
- **P287** (~60%). Number likewise.
- **P288** (~60%). For gender, at least half of that lead is the leak (in context over the causal
  tagger), under both tokenizers. For case the leak was 29% and 43% of the lead at 10k; for English
  part of speech, four fifths.
- **P289** (~55%). For number likewise.
- **P290** (~65%). Gender and number from the causal tagger lead their word forms by less than
  0.001, under both tokenizers. English part of speech led its word form by +0.0006 at 10k.
- **P291** (~65%). At 10k, gender and number in context each gain less over the empty slot under
  endings apart than under BPE 17k: exposed endings carry them, as they did case in the 5M model at
  a fixed budget (§33.113).

**The sentences these decide** (the text's, from its list of 19 October):
- the lead mostly the leak (P288 and P289 right, P290 right) → *for case, number and gender in
  Ukrainian and part of speech in English, what the sentence's feature adds over the word form's is
  mostly the leak*;
- the causal tagger 0.001 or more above the word form, the leak under half (P290 wrong) → *number
  and gender, unlike case, take information of their own from the sentence*: the title's answer then
  depends on the feature;
- P291 gives the exposure sentence for both features.

### 33.126 Items 3 and 6: English part of speech's deficit closes by 100k, and at 30M parameters Unigram's shortfall goes

The chain of 21 October (§33.123) ran on the second machine from 22:33 EDT on 20 October to 07:32
EDT on 21 October: MultiBLiMP on the 20k models (§33.124), item 3's 8 runs and item 6's 12. Item 3's
runs took about 45 minutes each, slowed by item 2's preparation on the same CPU (§33.125); item 6's
took about 16 minutes each. None failed, and both sanity checks passed.

**Item 3: English part of speech switched off at 5,000 of 100k** (`paired_cells_20261021T171341`).
Each run at its best step; the switched model's gain over the one never fed (negative: behind):

| | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| never fed, 100k | 1.2962 | 1.2987 | 1.2977 | 1.2995 | |
| switched off at 5,000, 100k | 1.2992 | 1.2975 | 1.2987 | 1.2987 | |
| the gain, 100k | −0.0030 | +0.0012 | −0.0010 | +0.0008 | **−0.0005 ± 0.0010 (2/4 behind)** |
| the gain, 50k (§33.102) | +0.0011 | −0.0022 | −0.0036 | −0.0031 | −0.0019 ± 0.0011 (3/4 behind) |

- **Along training,** the mean of the 100k runs at each step: −0.0414 at the switch, −0.0110 at step
  6,000, −0.0048 at 10,000, −0.0038 at 20,000, −0.0020 at 30,000, −0.0012 at 50,000, −0.0005 at
  75,000 and −0.0006 at the end. The deficit shrinks over the whole budget and is within its seed
  spread from about step 75,000.
- **No overfitting yet:** every never-fed run still improves over its last 10,000 steps, by 0.0015
  to 0.0017. Three of the four reach their best 500 to 1,000 steps before the end, the last
  evaluation 0.0001 above it. WikiText-103 read 6.8 times does not yet overfit the 5M model.

**Item 6: Unigram 17k at width 512** (`exposure_axis_20261021T171401`, `paired_cells_20261021T171420`,
`…T171453`; the second machine; BPE 17k's cells are item 4's, §33.118). Bits per byte at 10k:

| width 512, 10k, Unigram 17k | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| the null | 0.8841 | 0.8853 | 0.8860 | 0.8849 | 0.8850 |
| case | 0.8841 | 0.8807 | 0.8833 | 0.8831 | 0.8828 |
| the empty slot | 0.8862 | 0.8840 | 0.8853 | 0.8853 | 0.8852 |

- **Case gains as much under Unigram as under BPE 17k.** Over the empty slot it gains
  +0.0024 ± 0.0003 (4/4), against +0.0022 ± 0.0004 under BPE 17k (§33.118). Over the null it gains
  +0.0022 ± 0.0010 (3/4).
- **On the law's axis** case's gain over the empty slot, BPE 17k's minus Unigram's (positive:
  Unigram gains less), at each width's own reference levels:

| width 128 | | width 512 | |
|---|---|---|---|
| 1.38 | −0.0012 ± 0.0006 (1/4) | 1.37 | −0.0044 ± 0.0004 (0/4) |
| 1.28 | +0.0017 ± 0.0005 (4/4) | 1.28 | −0.0000 ± 0.0005 (2/4) |
| 1.19 | +0.0016 ± 0.0003 (4/4) | 1.18 | +0.0011 ± 0.0010 (2/4) |
| 1.09 | +0.0020 ± 0.0007 (3/4) | 1.08 | −0.0006 ± 0.0010 (2/4) |
| — | | 0.98 | −0.0009 ± 0.0004 (0/4) |
| — | | 0.89 | −0.0004 ± 0.0003 (1/4) |
| the end of 10k | +0.0020 ± 0.0008 (3/4) | the end of 10k | −0.0001 ± 0.0002 (1/4) |

  Unigram's shortfall, steady in the 5M model from reference BPB 1.28 on, is not in the 30M model:
  there the two tokenizers' gains differ by no more than their spread from 1.28 down.
- **The null costs nothing here either:** under Unigram at width 512, the empty slot ends
  −0.0001 ± 0.0007 (2/4) against the null, as under the other two tokenizers at that width.
- **And an observation:** Unigram's null is behind BPE 17k's at every seed. On this machine it is
  −0.0056 ± 0.0009 at width 512, against −0.0014 ± 0.0003 at width 128 (0/4 each).

**Predictions** (§33.123).
- **P280 right:** −0.0005 ± 0.0010, under 0.001.
- **P281 half right:** every never-fed run improves over its last 10,000 steps, but three of four
  reach their best 500 to 1,000 steps before the end.
- **P282 right:** −0.0005 against −0.0019.
- **P283 right:** −0.0004 ± 0.0003 at reference BPB 0.89.
- **P284 right:** −0.0001 ± 0.0002.
- **P285 right:** −0.0001 ± 0.0007.

**The sentences** (the text's, from its list of 19 October).
- Item 3 *closes*: given long enough training, the model makes up for the removed feature, so its
  reliance on it is temporary. The deficit is 0.0039 at 20k, 0.0019 at 50k, and within its spread at
  100k.
- Item 6 *goes*: at 30M parameters the redundancy is gone under both exposing tokenizers, and
  Unigram's shortfall belongs to the 5M model, as the null's cost does.

**What it changes for the thesis.**
1. **Removing English part of speech leaves a deficit that closes with training.** The feature is
   still a crutch, not a teacher. The model leans on it while it is there, and needs about 70,000
   more steps to do without it.
2. **The exposure sentence is the 5M model's alone, under every exposing tokenizer.** In the 30M
   model case gains alike where endings are exposed, late in training and at the end of a 10k
   budget, under endings apart and under Unigram.
3. **At width 512 Unigram trails BPE 17k by 0.006** at every seed. Where the cuts fall matters more
   in the larger model, not less.

### 33.127 Item 2: gender's and number's lead over the word form is small, and little of it is the leak

The chain for item 2 (`second_item2_2026-10-21.sh`, §33.125) ran on the second machine from 07:32 to
10:54 EDT on 21 October: 48 runs of ~4.2 min. None failed, and every sanity check passed. Width 128,
~38M tokens, 10k, seeds 0–3, every pair within the machine (`paired_cells_20261021T171235`). Gains
in bits per byte (positive: the first better):

| 10k | gender, BPE 17k | gender, endings apart | number, BPE 17k | number, endings apart |
|---|---|---|---|---|
| in context over the word form: the lead | +0.0010 ± 0.0001 (4/4) | +0.0008 ± 0.0003 (4/4) | +0.0009 ± 0.0001 (4/4) | +0.0010 ± 0.0000 (4/4) |
| in context over the causal tagger: the leak | +0.0003 ± 0.0002 (3/4) | −0.0001 ± 0.0004 (2/4) | +0.0001 ± 0.0000 (4/4) | +0.0002 ± 0.0001 (4/4) |
| the causal tagger over the word form | +0.0007 ± 0.0001 (4/4) | +0.0009 ± 0.0002 (4/4) | +0.0008 ± 0.0001 (4/4) | +0.0008 ± 0.0001 (4/4) |
| in context over the null | +0.0003 ± 0.0007 (2/4) | +0.0027 ± 0.0012 (3/4) | +0.0013 ± 0.0001 (4/4) | +0.0018 ± 0.0004 (4/4) |
| in context over the empty slot | +0.0009 ± 0.0008 (3/4) | +0.0005 ± 0.0013 (2/4) | +0.0019 ± 0.0003 (4/4) | −0.0004 ± 0.0007 (2/4) |

- **The lead over the word form is small:** about +0.001 for both features under both tokenizers,
  at every seed. That is a third to a half of case's at 10k (+0.0021 to +0.0033, §33.65, §33.81).
- **Little of it is the leak:** 30% and none for gender (BPE 17k, endings apart), 10% and 21% for
  number. For case at 10k it was 29% and 43%; for English part of speech, four fifths.
- **The rest is not split yet.** The causal tagger keeps +0.0007 to +0.0009 over the word form (4/4
  each). For case at 10k, about half of the corresponding rest sat at words unseen in training:
  there the word form gives the model nothing, while a tagger reads the word's own ending (G-76,
  §33.74). For gender and number that split is not made yet (G-82). It decides whether the rest is
  the sentence's information or the word's.
- **Exposure:** over the empty slot, number gains less under endings apart than under BPE 17k at
  every seed, +0.0023 ± 0.0008 (4/4). Gender's two gains do not differ, +0.0004 ± 0.0020 (2/4).
  Over its references gender is noisy under endings apart, from −0.0001 to +0.0056 over the null.

**Predictions** (§33.125).
- **P286 right:** +0.0010 and +0.0008, 4/4 each.
- **P287 right:** +0.0009 and +0.0010, 4/4 each.
- **P288 wrong:** the leak is 30% and none of gender's lead.
- **P289 wrong:** 10% and 21% of number's.
- **P290 right:** +0.0007 to +0.0009, all below 0.001.
- **P291 half right:** number gains less under endings apart at every seed; gender's gains are level.

The tally is 173 right, 13 half right, 96 wrong, 8 never tested, 1 withdrawn and none pending, of 291.

**The sentences** (the text's, from its list of 19 October). Neither fits yet.
- *Mostly the leak* does not hold: for both features the leak is a small part of the lead.
- *Number and gender take information of their own from the sentence* waits for G-82. If the causal
  tagger's +0.0007 to +0.0009 sits at words unseen in training, it is the word's ending, which the
  lexicon lacks and the tagger reads. If it sits at seen words, it is context.

**What it changes for the thesis.**
1. **For gender and number, what the sentence's tag adds over the word form is about 0.001 at 10k,
   and the tagger's leak is a small part of it.** That differs from English part of speech (four
   fifths) and is smaller than case's share (29% and 43%).
2. **Whether the rest is context is G-82.** It needs the loss by the target's word, seen and unseen
   in training, as for case (§33.74): the 32 word-form and causal models rescored on the validation
   text, on the CPU.
3. **Number gains less where endings are exposed,** as case did in the 5M model at a fixed budget;
   gender shows no difference.

### 33.128 G-82 prepared: the causal tagger's rest, split by the word it describes

G-82 asks where gender's and number's causal rest sits (§33.127). The causal tagger leads the word
form by +0.0007 to +0.0009 at 10k, at every seed. Either it tells a seen word's value from the words
before it, which is the sentence's information, or it reads the ending of a word the training text
never had, where the form gives the model `none`. For case at 10k, about half of the corresponding
rest sat at such words (§33.74).

**The tool.** `scripts/analysis/loss_by_kind.py --previous` (§33.74) now takes any feature with a
word-form variant.
- `--unseen-lexicon` decides what is unseen by the training lexicon itself: a word the lexicon lacks.
  Under §33.74's rule any word whose form gives `none` counted as unseen; for gender and number
  `none` is also the majority value of many seen words (plural forms have no gender, adverbs no
  number). Such a seen word now joins the class whose value differs from its form's.
- Each fact is loaded from its own codes file and checked against the scored stream: no file holds
  part of speech, the causal tagger's codes and the word form's together.
- A new check that could fail: the word form's codes must be the lexicon's at every position of the
  validation stream.
- The classes are named for any feature ("values differ"); without the option the rule is §33.74's.
- New test, `test_loss_by_kind_unseen_by_lexicon`: the classes with and without a lexicon, every
  position of an unseen word coded as unseen, codes from another lexicon refused.

The tool was tried on the thesis machine on the four validation streams alone, before any model was
scored; the lexicon check passed on all four. The two models' inputs differ at about 5% of targets,
a little over half of them after words unseen in training.

**Predictions** (before any model is scored; the class sizes above were seen when trying the tool):
- **P292** (~60%). For gender, words unseen in training hold at least half of the causal tagger's
  lead over the word form, under both tokenizers.
- **P293** (~60%). For number likewise.
- **P294** (~65%). Seen words whose value from the causal tagger differs from their form's majority
  hold less than a third of that lead, in each of the four cells.
- **P295** (~65%). After unseen words the word-form model ends below the null in each of the four
  cells, at one seed of four above it at most: its `none`, which in training marked a word without
  the feature, misleads it there, as for case (G-76).

**What each outcome says**, for each feature, fixed before the run:
- unseen words hold at least half under both tokenizers, and seen words less than a third: *what the
  sentence's gender (number) adds over the word form is not context: a small part is the tagger's
  leak, most the endings of words the training text never had*. The seen words' part bounds what
  context adds.
- seen words hold at least half under both tokenizers: *gender (number) takes a little information
  of its own from the sentence*, as much as the seen words' part.
- anything else: the split as it falls, each part with its size.

**The run**, on the second machine, which holds the 32 models and their nulls. §33.74's split took
49 s of GPU for 48 models. Every model is re-scored first, and nothing is reported unless each
matches its record within 10⁻⁵ bits per byte.

```bash
python scripts/analysis/loss_by_kind.py --machine second --tokenizers bpe_uk_17k bpeend_uk_17k \
    --facts ugendercausal ugenderlex --previous --previous-facts pos ugendercausal ugenderlex --unseen-lexicon ugender
python scripts/analysis/loss_by_kind.py --machine second --tokenizers bpe_uk_17k bpeend_uk_17k \
    --facts unumbercausal unumberlex --previous --previous-facts pos unumbercausal unumberlex --unseen-lexicon unumber
```

### 33.129 G-82: what the causal tagger adds over the word form is not context

The split of §33.128 ran on the second machine on 22 October (`loss_by_kind_20261022T021826`, gender;
`loss_by_kind_20261022T021845`, number; 18 s each). All 48 models re-scored to their records within
1.6 × 10⁻⁸ bits per byte, and the word form's codes were the lexicon's at every position of both
validation streams. What the causal tagger adds over the word form, by the word the feature
describes, paired by seed (with each class's share of the rest):

| described word | gender, BPE 17k | gender, endings apart | number, BPE 17k | number, endings apart |
|---|---|---|---|---|
| a word unseen in training | +0.00057 ± 0.00002 (4/4), 82% | +0.00057 ± 0.00003 (4/4), 61% | +0.00065 ± 0.00003 (4/4), 80% | +0.00067 ± 0.00002 (4/4), 88% |
| a seen word whose value differs from its form's majority | +0.00004 ± 0.00000 (4/4), 6% | −0.00001 ± 0.00002 (1/4), −1% | +0.00004 ± 0.00001 (4/4), 5% | −0.00003 ± 0.00000 (0/4), −4% |
| a preposition | +0.00001 ± 0.00001 (3/4), 1% | +0.00005 ± 0.00002 (4/4), 6% | −0.00002 ± 0.00002 (3/4), −2% | +0.00000 ± 0.00001 (2/4), 1% |
| every other target: the same input in both models | +0.00007 ± 0.00009 (2/4), 10% | +0.00032 ± 0.00021 (3/4), 34% | +0.00014 ± 0.00007 (3/4), 17% | +0.00012 ± 0.00004 (3/4), 16% |
| **the whole rest** | **+0.00069 ± 0.00011 (4/4)** | **+0.00094 ± 0.00025 (4/4)** | **+0.00081 ± 0.00013 (4/4)** | **+0.00076 ± 0.00007 (4/4)** |

The classes hold 2.64%, 2.53%, 2.95% and 2.82% of the targets (unseen words), 2.21%, 2.18%, 1.62% and
1.55% (seen words whose value differs), 8.4–9.1% (prepositions), 86–87% (the rest). The described
word is unseen in training at 4.63% of targets under BPE 17k and 4.33% under endings apart; at the
rest of those targets the causal tagger also gives `none`. Seen words whose form's majority is
`none`, which §33.74's rule would have counted as unseen, are 0.29–0.36% of targets.

- **The rest sits after words unseen in training**: three fifths to nine tenths of it, at every seed
  of every cell. About a third of that part is on the tokens holding an ending's start (+0.00018 to
  +0.00021, 4/4 each).
- **And there it is mostly the word form's loss, not the tagger's gain.** After unseen words the
  word-form model ends below the null, −0.00048, −0.00045, −0.00054 and −0.00055 (0/4 each), while
  the causal tagger's model gains +0.00009, +0.00013, +0.00011 and +0.00012 over it (4/4 each).
  About four fifths of the part is the form's `none`, which in training marked a word without the
  feature, misleading its model (G-76), as for case (§33.74).
- **Where the sentence changes a seen word's gender or number, the two models do not differ:** −4% to
  +6% of the rest, under 0.00005 either way.
- **The rest of the rest, 10–34%, is spread over targets that both models see alike.** Two trained
  models compared target by target are not a model and its counterfactual (§33.76, §33.81), and this
  part says they differ in what they learned too.

**Predictions** (§33.128).
- **P292 right:** 82% and 61% of gender's rest.
- **P293 right:** 80% and 88% of number's.
- **P294 right:** −4% to +6%, in each of the four cells.
- **P295 right:** −0.00048, −0.00045, −0.00054 and −0.00055 below the null, 0/4 above in each.

The tally is 177 right, 13 half right, 96 wrong, 8 never tested, 1 withdrawn and none pending, of 295.

**The sentence.** For both features unseen words hold over half of the rest under both tokenizers,
and seen words under a third. By the rule fixed in §33.128 that gives: *what the sentence's gender
and number add over the word form is not context: a small part is the tagger's leak, most the
endings of words the training text never had.* The null sharpens its second half: after those words
the word form's model is misled more than the tagger informs. The text's sentence for item 2 is
therefore neither of its two (§33.127): not *mostly the leak*, and not *information of their own
from the sentence*.

**What it changes for the thesis.**
1. **For gender and number, what the sentence adds over the word form is not context.** At 10k their
   lead over the word form is about 0.001. Of that, none to 30% is the tagger's leak (§33.127). Most
   of the rest sits after words unseen in training, where the word form misleads its model. Context
   resolving a known word adds nothing measurable.
2. **The same holds for all three Ukrainian features at 10k.** For case, context in the title's
   sense held under a tenth of the extra over the word form (§33.74); for gender and number, nothing
   measurable.
3. **G-82 is closed.** G-76 is now measured for gender and number as well. The word-form controls'
   handicap after unseen words is about the same for all three features: 0.00045 to 0.00055 below
   the null for gender and number, 0.0005 to 0.0007 for case.

### 33.130 The text's request of 22 October, and G-81's recipe question prepared

The text, checked against `a691568`, asks for four experiments in strict order: (1) the longer
warm-up at width 1024 that G-81 holds, at four seeds under both tokenizers, then MultiBLiMP; (2)
English part of speech at full training from the causal tagger; (3) gender and number at full
training; (4) width 1024 at 20k on the 151M-token corpus, which repeats the text about once. Each
passes the sentence test: its outcomes give the text different sentences. Items 1 and 3 run first,
on the second machine; items 2 and 4 wait.

**Item 1, G-81's recipe question.** At width 1024 and 10k, seeds 1 and 2 lose most of the
tokenizer's gain early in training (−0.0004 and +0.0042, against +0.0127 and +0.0103 at seeds 0 and
3; §33.120), and keep the split at 20k (§33.124). Every run of the thesis warms up for 100 steps to
the same peak learning rate, 3 × 10⁻⁴, at every width; at 85M parameters that may be too short.
- The nulls at width 1024 under BPE 17k and endings apart, ~38M tokens, 10k, seeds 0–3, with a
  1,000-step warm-up: ten times the usual, a tenth of the budget, ending before the fall begins at
  about step 1,500 (§33.122).
- The same at width 512, so that width 1024's gain is set against width 512's under the same
  recipe: a longer warm-up shifts every null.
- Then MultiBLiMP on each width's eight. In all, 16 runs, about 8 h on the second machine (44 and
  16 min a run).

**The label.** `train.warmup` did not enter a run's label, so these runs would have joined the
width-1024 cell as more seeds. It now adds `_wu<steps>` when it is not 100 (`morphlm/config.py`).
The label tests say so, and had been failing since 20 October on the 5k look at G-81 (§33.121): it
evaluated every 100 steps, never stopped early and ran on a 10k run's schedule, three settings the
tests held to one value in every run. It is alone in its cell; the tests now name it as the one
documented exception (`tests/test_labels.py`).

**The chain** (`scripts/chains/second_2026-10-22.sh`; pieces wu1024, wu512, mbwu): each piece's first
seed must pass the pair check. Rehearsed with the stub, five scenarios: everything runs (16 runs,
each with the warm-up, width 1024 first); the pair check fails at width 1024 and its other seeds are
skipped (10 runs); END_BY leaves every run for later; with every run done MultiBLiMP runs once per
width; with its records there it runs not at all. Its cells are the nulls that ran on that machine
on 19–20 October with one setting changed, so no new data or loader path is involved.

**Predictions** (before any run):
- **P296** (~50%). With a 1,000-step warm-up, the tokenizer's gain at width 1024 is at least half of
  width 512's mean under the same warm-up at every seed: the split goes.
- **P297** (~35%). Its mean is within 0.003 of width 512's under the same warm-up.
- **P298** (~65%). The longer warm-up moves width 512's gain by less than 0.002 (+0.0141 ± 0.0016
  with 100 steps, on the same machine).
- **P299** (~70%). Width 1024's MultiBLiMP gain is within 1 point of its value with 100 steps (+5.3),
  and positive at every seed.

**What each outcome says** (the text's sentences, fixed before the run). With 100 steps, seeds 1 and
2 sit below half of width 512's gain and seeds 0 and 3 above it, so the rule separates the old split.
- Every seed at least half of width 512's mean under the same warm-up, and the mean within 0.003 of
  it: *with a longer warm-up the gain at width 1024 holds at every seed, at width 512's level: the
  dependence on the run was the training schedule's, not the model's.*
- Every seed at least half, the mean further below: the same, but *smaller than at width 512*.
- A seed below half: *the dependence on the run does not go with a longer warm-up.* Only then the
  twin on the thesis machine, another dropout draw (G-81).
