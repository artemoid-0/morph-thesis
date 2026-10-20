# Research journey — Part 15: The List of 19 October: Case and the Null at 30M Parameters

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.117 onward. [← Part 14](14_two_machines.md) · [index](README.md)

**Established so far:** over eight seeds of English at full training, keeping endings apart is
better on BLiMP's subject–verb and anaphor agreement at six seeds of eight, on determiner–noun
agreement at four (§33.117). Read on the law's axis, case gains less where endings are exposed only
in a middle stretch of training, in models of 5M and 30M parameters (§33.117–§33.118). At 30M
parameters the same-shape null costs nothing; in the 5M model the tokenizer's gain over the empty
slots is +0.0203, not +0.0175 (§33.118).

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
