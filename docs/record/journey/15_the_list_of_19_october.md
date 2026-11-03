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
where the word form misleads its model (§33.128–§33.129). At 85M parameters
with a 1,000-step warm-up the tokenizer's gain holds at every seed, smaller than at 30M (§33.132). At full training
gender's and number's lead over the word form is about 0.0002, without a leak, and not context (§33.133). English part of
speech's causal tagger keeps nothing reliable over the word form at full training (§33.135). At 85M parameters the
same-shape null costs BPE 17k and helps endings apart, so the tokenizer's gain over the empty slots is +0.0079, 0.003
below its gain between the nulls (§33.139). Without repeating the text, at 20k, that gain is smaller:
+0.0082 (1.09%), still rising (§33.141). At full training at 30M parameters the tokenizer gains +0.0072
(0.97%) and lasts; case stays within 0.002 of its null but keeps +0.0015 at every seed; English endings
apart gains nothing at 30M (§33.142). At the best of three learning rates the 5M model's gain at full training is 1.52%,
smaller than at the default; between plain baselines 2.31%, larger (§33.145). The cut one letter off the boundary loses
to plain BPE, shuffled case gains nothing, and the switch to a constant leaves the deficit the switch to zeros does
(§33.148). At 85M parameters the gain at full training is the 30M model's, +0.0071 (1.02%); it holds under a modern
block and halves at 8 layers (§33.149). Inside the noun phrase and after a preposition, near the ceiling, endings apart
makes about half the errors at 5M parameters and a third at 30M, at every seed (§33.150). Every row of the
tokenizer's table holds at 95%; the features' one-sided 95% bound is 0.003, and four seeds see a feature's effect only
from about 0.002–0.006 (§33.151). On the held-out test articles the gain is 1.3% at 5M parameters and 0.83% at 30M,
positive at every seed; at the same token length plain BPE 8k loses by 0.083, and BPE 32k wins by 0.028 (§33.152). On
gold labels the tagger reads case where context decides: 81% of those words, and 87% where it departs from the form's
usual case (§33.153). XLM-R's, mBERT's and GPT-4o's tokenizers keep a Ukrainian ending apart for a quarter of words or
fewer; Qwen2.5's for half, by cutting every word into 3.5 tokens (§33.154). Near 5M parameters the gain is worth about 15%
more parameters, or 17% more training steps (§33.155). Four fifths of it sits in the words the rule cuts, of middle and
low frequency; over articles the two held-out samples differ by 2.5 standard errors (§33.156). The probes' lead is not in
the ending's token: level at layer 0, it appears at layer 1 (§33.157).

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

### 33.131 Item 3 prepared: gender and number at full training

The text's item 3 (§33.130). At 10k what the causal tagger adds over the word form is not context
(§33.129); at full training, 151M tokens and 50k steps, case's whole lead over the word form is the
tagger's leak (§33.81). Do gender and number go the same way?

**The cells**, under endings apart, where case's full training was measured: gender and number
tagged in context, from the word form and from the causal tagger, and the null; seeds 0–3, 28 runs,
about 10 h on the second machine. Then G-82's split of each feature's causal rest (§33.128).

**The data.** The 151M-token stream under endings apart was on the thesis machine only, packed.
- Unpacked there (`cache_pack.py`; six arrays, 8.84 GiB, each checked against the pack), its token
  ids, its codes in context (part of speech, case, gender, number and the rest) and its null's codes
  went to the second machine: 4.76 GB in twelve files, each checked by SHA-256 on arrival.
- There the word-form codes were derived (`derive_lexical_codes_20261022T051852`, gender;
  `…T052025`, number): 2,646,849 word types from 68.9M words of training text. A word's majority
  value matches its value in context at 97.06% of occurrences for gender and 97.92% for number.
  3.05% of validation words are unseen in training, against 5.59% at ~38M tokens (1,138,979 types).
- And the causal tagger's codes (`derive_causal_ud_20261022T052158`; 73 min on that machine's CPU).
  Its three checks passed: a window of four tokens tags 2,000 of 2,000 words as the whole prefix
  does; the ids encode again identically; the full context reproduces the cached gender and number
  on 805,818 training tokens and on the whole validation stream. The causal tag equals the tag in
  context at 97.50% of tokens for gender and 99.10% for number (97.51% and 99.10% at ~38M tokens).
- A cell of each piece ran 30 steps through the real loader on that machine, on its CPU while the
  GPU trained, in a copy of the code whose records stay in the copy. All seven did, each reading the
  stream's 662,380 training windows and its 1,418 validation windows.

**The chain** (`scripts/chains/second_full_2026-10-22.sh`; pieces gender_f, number_f, null_f,
split_f) starts when the warm-up chain lets go of its lock. Each feature's first seed must pass the
pair check (in context against the word form). The null's first seed repeats the thesis machine's
run `57f76ed51760` and must end within 0.01 of it (0.926563). Rehearsed with the stub, seven scenarios:
everything runs (28 runs, four of each cell); a failed pair check skips that feature's other seeds;
a failed twin check skips the null's; a causal cell not ready skips its feature; END_BY leaves
every run for later; with every run done the split runs once per feature; with its records there it
runs not at all.

**Predictions** (before any run):
- **P300** (~75%). At full training the causal tagger's lead over the word form is smaller than at
  10k for both features (+0.0009 for gender and +0.0008 for number under endings apart, §33.127):
  fewer words are unseen in training, and the word form's handicap after them shrinks.
- **P301** (~75%). By the rule below, both features give *like case*.
- **P302** (~55%). Number's leak (in context over the causal tagger) is positive at three seeds of
  four or more.
- **P303** (~45%). Gender read in context still leads its word form at three seeds of four or more.

**What each outcome says** (the text's sentences, fixed before the run), for each feature:
- the causal tagger's lead over the word form positive at fewer than three seeds of four, or seen
  words whose value the sentence changes holding less than a third of it: *like case: at full
  training what the sentence adds over the word form is not context*;
- that lead positive at three seeds of four or more, and seen words whose value differs holding at
  least half of it: *unlike case: gender (number) takes information of its own from the sentence*;
- anything else: the split as it falls, each part with its size.

### 33.132 Item 1: with a longer warm-up the gain at width 1024 holds at every seed

The warm-up chain (`second_2026-10-22.sh`, §33.130) ran on the second machine from 01:13 to 09:22 EDT
on 22 October: 16 runs of 44 and 16 min, then MultiBLiMP on each width's eight. None failed, and both
pair checks passed. Paired by seed within the machine (`paired_cells_20261022T135826`;
`multiblimp_eval_20261022T132120`, width 1024; `multiblimp_eval_20261022T132214`, width 512):

| ~38M tokens, 10k | warm-up of 100 steps | warm-up of 1,000 steps |
|---|---|---|
| width 1024, the tokenizer's gain at seeds 0–3 | +0.0127, −0.0004, +0.0042, +0.0103 | +0.0095, +0.0143, +0.0100, +0.0103 |
| its mean | +0.0067 ± 0.0030 (3/4) | **+0.0110 ± 0.0011 (1.34%, 4/4)** |
| width 512, the tokenizer's gain | +0.0141 ± 0.0016 (4/4) | +0.0147 ± 0.0022 (1.67%, 4/4) |
| MultiBLiMP, width 1024 | +5.3 ± 0.4 points (4/4) | +5.9 ± 0.4 (4/4) |
| MultiBLiMP, width 512 | +5.9 ± 0.3 (4/4) | +6.0 ± 0.5 (4/4) |

- **The split goes.** Every seed at width 1024 keeps more than half of width 512's gain under the same
  warm-up (+0.0074). The two seeds that fell with 100 steps gain +0.0143 and +0.0100.
- **The warm-up helps the endings-apart null.** Against its 100-step run at the same seed, the
  endings-apart null at width 1024 ends 0.0053 ± 0.0032 lower: 0.0135 at seed 1 and 0.0067 at seed 2,
  the two that fell. The BPE 17k null ends 0.0010 ± 0.0008 lower. At width 512 neither moves: +0.0012 ±
  0.0023 and +0.0006 ± 0.0014 (2/4 each).
- **The gain stays smaller than at width 512:** the mean is 0.0037 below width 512's under the same
  warm-up, beyond the rule's 0.003.
- **In grammar the two widths are level:** +5.9 and +6.0 points under the longer warm-up. At width
  1024: number +5.8 ± 1.3, gender +6.6 ± 0.3, person +5.3 ± 0.5 (4/4 each).

**Predictions** (§33.130).
- **P296 right:** every seed above half of width 512's mean: +0.0095 to +0.0143.
- **P297 wrong:** the mean is 0.0037 below width 512's, not within 0.003.
- **P298 right:** +0.0147 against +0.0141 with 100 steps.
- **P299 right:** +5.9 against +5.3 points, positive at every seed.

The tally is 180 right, 13 half right, 97 wrong, 8 never tested, 1 withdrawn and 4 pending, of 303.

**The sentence** (the rule of §33.130): every seed at least half of width 512's gain, the mean further
below. *With a longer warm-up the gain at width 1024 holds at every seed, smaller than at width 512:
the dependence on the run was the training schedule's, not the model's.* The twin on the thesis
machine, needed only if the split had stayed, is not run. G-81 is closed.

**What it changes for the thesis.**
1. **At 85M parameters the tokenizer's gain is smaller than at 30M under either warm-up, and with a
   1,000-step warm-up it holds at every seed:** +0.0110 ± 0.0011 at 10k. "Depends on the run" goes
   from FINDINGS: it belonged to the 100-step warm-up at this width.
2. **The grammatical gain stops growing at 30M parameters under the longer warm-up too:** +5.9
   against +6.0 points.
3. **The 100-step warm-up stays every other run's recipe.** At 30M parameters the longer one changes
   neither null nor the gain; at 85M it is the endings-apart null that needed it.

### 33.133 Item 3: at full training gender's and number's lead over the word form is small, and not context

The full-training chain (`second_full_2026-10-22.sh`, §33.131) ran on the second machine from 09:22 to
19:08 EDT on 22 October: 28 runs of ~21 min, then G-82's split of each feature. None failed. Both pair
checks passed, and the null's first seed repeated the thesis machine's run to 0.0007 (0.92725 against
0.92656). Endings apart, 151M tokens, 50k, paired by seed within the machine
(`paired_cells_20261023T004525`; the null's four runs are the thesis machine's repeated, kept in
`results/audits/device_twins_20261022/`). Gains in bits per byte (positive: the first better):

| full training, endings apart | gender | number |
|---|---|---|
| in context over the word form: the lead | +0.0002 ± 0.0001 (3/4) | +0.0002 ± 0.0002 (3/4) |
| in context over the causal tagger: the leak | −0.0001 ± 0.0001 (2/4) | −0.0001 ± 0.0001 (1/4) |
| the causal tagger over the word form: the rest | +0.0002 ± 0.0001 (4/4) | +0.0003 ± 0.0003 (3/4) |
| in context over the null | +0.0011 ± 0.0006 (3/4) | −0.0005 ± 0.0005 (1/4) |
| the word form over the null | +0.0010 ± 0.0007 (3/4) | −0.0007 ± 0.0006 (2/4) |

At 10k the leads were +0.0008 and +0.0010, the rests +0.0009 and +0.0008 (§33.127).

**The split** (`loss_by_kind_20261022T230754`, gender; `loss_by_kind_20261022T230806`, number; 24 models
re-scored to their records within 9 × 10⁻⁹). What the causal tagger adds over the word form, by the
word the feature describes:

| described word | gender | number |
|---|---|---|
| a word unseen in training (1.35% and 1.50% of targets) | +0.00011 ± 0.00001 (4/4), 47% | +0.00016 ± 0.00001 (4/4), 50% |
| a seen word whose value differs from its form's majority (2.27%, 1.63%) | +0.00001 ± 0.00001 (3/4), 4% | +0.00001 ± 0.00001 (2/4), 2% |
| a preposition (9.1%) | +0.00002 (3/4), 11% | +0.00001 (3/4), 3% |
| every other target: the same input in both models | +0.00008 ± 0.00008 (2/4), 37% | +0.00015 ± 0.00026 (3/4), 46% |
| **the whole rest** | **+0.00022 ± 0.00010 (4/4)** | **+0.00033 ± 0.00030 (3/4)** |

The described word is unseen in training at 2.38% of targets, against 4.33% at ~38M tokens. After
unseen words the word-form model again ends below the null, −0.00006 and −0.00015 (0/4 each), and the
causal tagger's model gains +0.00004 and +0.00002 over it.

- **What the sentence adds over the word form has nearly gone:** about +0.0002 for both features,
  against about +0.001 at 10k.
- **There is no leak left:** in context over the causal tagger −0.0001 for both.
- **The rest is not context:** half of it sits after words unseen in training, where the word form
  misleads its model; after seen words whose value the sentence changes the models differ by 2–4% of
  it; the other half is spread where both models see the same input.
- **Over the null** gender still gains +0.0011 ± 0.0006 (3/4), from the word form as much as from the
  sentence; number gains nothing, −0.0005 ± 0.0005 (1/4).

**Predictions** (§33.131).
- **P300 right:** the rest is +0.0002 and +0.0003, against +0.0009 and +0.0008 at 10k.
- **P301 right:** both features give *like case*; seen words whose value differs hold 4% and 2% of
  the rest.
- **P302 wrong:** number's leak is positive at one seed of four.
- **P303 right by its letter:** gender in context leads its word form at three seeds of four, by
  +0.0002.

The tally is 183 right, 13 half right, 98 wrong, 8 never tested, 1 withdrawn and none pending, of 303.

**The sentence** (the rule of §33.131): both features *like case*. *At full training what the
sentence adds to gender and number over the word form is not context — and it is small, about 0.0002
bits per byte, with no leak left.* For case at full training the whole lead was the leak (§33.81); for
gender and number the leak has gone as well.

**What it changes for the thesis.**
1. **At full training, as at 10k, the sentence adds no context to gender and number over the word
   form.** Their lead shrinks from about 0.001 to 0.0002; half of what is left is the word form's
   handicap after words unseen in training.
2. **Their leak goes with training;** case's did not (+0.0015 at full training, §33.81), since a
   preposition's case names the next noun's ending.
3. **Number gives nothing over the null at full training, gender a little (+0.0011),** and the word
   form as much as the sentence: once the model has learned, what is left is the word's own form.

### 33.134 Item 2 prepared: English part of speech at full training

The text's item 2 (§33.130). At 10k and 20k English part of speech from a causal tagger leads the word
form by +0.0006 ± 0.0004 and +0.0004 ± 0.0003 (3/4 each), and the tagger's leak is four fifths of
the tagged feature's lead (§33.99, §33.107). At full training case's whole lead is the leak (§33.81).
Does English part of speech go the same way, so that the text's conclusion on context can state full
training for both languages?

**The cells.** WikiText-103, width 128, 50k, seeds 0–3, on the second machine: part of speech tagged
in context, from the word form and from the causal tagger, and the null. The word form's and the
null's seeds 2–3 ran there on 18 October; the rest are 12 runs, ~4.2 h. The thesis machine's runs of
these cells predate fields the configuration gained later, so their repeats take new run ids, and
they are kept as twins. Then G-82's split of the causal tagger's rest.

**The tool.** `loss_by_kind.py` now builds English labels (WikiText-103 leaves the label without a
corpus part, as `config.label()` does). Tried on the thesis machine on the 10k runs' validation
stream, no model scored: the English text decodes; the word form's codes are the lexicon's at every
position; the two models' inputs differ at 22% of targets, and a word unseen in training is described
at 0.6% of them (WikiText-103's vocabulary is closed), so the English split weighs seen words. The
lexicon went to the second machine, checked by SHA-256.

**The chain** (`scripts/chains/second_en_2026-10-23.sh`; pieces enfull, ennull, ensplit). Each piece's
first run repeats a thesis machine run and must end within 0.01 of it: part of speech in context at
seed 0 (1.340535), the null at seed 0 (1.340149). Its causal and in-context cells ran 30 steps through
the real loader on that machine's GPU, in a copy of the code whose records stay in the copy. Rehearsed
with the stub, seven scenarios: everything runs (16 runs, four of each cell); a failed twin check skips
that piece's other seeds; a causal cell not ready skips its piece; END_BY leaves every run for later;
with every run done the split runs once; with its record there it runs not at all.

**Predictions** (before any run):
- **P304** (~55%). At full training English part of speech from the causal tagger ends level with the
  word form: its lead over the word form is positive at fewer than three seeds of four.
- **P305** (~75%). The leak (in context over the causal tagger) is positive at three seeds of four or
  more.
- **P306** (~60%). Part of speech in context leads the word form at three seeds of four or more, and
  the leak is at least three quarters of that lead.

**What each outcome says** (the text's sentences, fixed before the run):
- the causal tagger's lead over the word form positive at fewer than three seeds of four: *all leak:
  at full training all that English part of speech from the sentence adds over the word form is the
  tagger's leak, as for case*; the text's conclusion on context then states full training for both
  languages;
- positive at three seeds or more, seen words whose value differs holding less than a third of it:
  *not context*, the leak and the rest that the split names;
- positive at three seeds or more, those words holding at least half: *English part of speech takes
  information of its own from the sentence*, and the conclusion splits by language;
- anything else: the split as it falls.

### 33.135 Item 2: at full training English part of speech's causal tagger keeps nothing reliable over the word form

The English chain (`second_en_2026-10-23.sh`, §33.134) ran on the second machine from 20:55 EDT on 22
October to 01:09 on 23 October: 12 runs of ~21 min, then the split. None failed. Both twin checks
passed: part of speech in context at seed 0 ended 0.0042 from the thesis machine's run, the null
0.0006. WikiText-103, 50k, four seeds, every pair within the machine (the word form's and the null's
seeds 2–3 ran there on 18 October). Gains in bits per byte (positive: the first better):

| full training, English | seeds 0–3 | mean |
|---|---|---|
| in context over the word form: the lead | +0.0028, +0.0018, +0.0047, −0.0004 | +0.0022 ± 0.0011 (3/4) |
| in context over the causal tagger: the leak | −0.0000, −0.0000, +0.0064, +0.0006 | +0.0017 ± 0.0016 (2/4) |
| the causal tagger over the word form: the rest | +0.0029, +0.0018, −0.0018, −0.0010 | +0.0005 ± 0.0011 (2/4) |
| in context over the null | +0.0044, −0.0014, +0.0039, +0.0014 | +0.0021 ± 0.0013 (3/4) |
| the word form over the null | | −0.0001 ± 0.0012 (2/4) |
| the causal tagger over the null | | +0.0003 ± 0.0015 (2/4) |

At 10k and 20k the lead was +0.0031 and +0.0025, the leak +0.0025 and +0.0021 (4/4 each), the rest
+0.0006 and +0.0004 (3/4 each; §33.99, §33.107).

- **The seeds split in two.** At seeds 0 and 1 the causal tagger's model ends level with the tagged
  feature's (no leak) and ahead of the word form's; at seeds 2 and 3 it ends behind both, at seed 2
  behind the null. Each causal run converged as its siblings did, still improving over its last 5,000
  steps, so none is set aside.
- **By the mean the lead is mostly the leak:** +0.0017 of +0.0022, 77%. The rest, +0.0005 ± 0.0011, is
  positive at two seeds of four.
- **The split** (`loss_by_kind_20261023T050924`; 12 models re-scored within 1.6 × 10⁻⁸). The only part
  of the rest present at every seed sits after words unseen in training, +0.00030 ± 0.00005 (4/4), on
  0.6% of targets, where the word form ends below the null (−0.00030). After seen words whose part of
  speech the causal tagger changes (20.7% of targets) it is +0.00018 ± 0.00034 (2/4).

**Predictions** (§33.134).
- **P304 right:** the causal tagger's lead over the word form is positive at two seeds of four.
- **P305 wrong:** the leak is positive at two seeds of four.
- **P306 right:** in context leads the word form at three seeds of four, and the leak is 77% of that lead.

The tally is 185 right, 13 half right, 99 wrong, 8 never tested, 1 withdrawn and none pending, of 306.

**The sentence** (the rule of §33.134): the causal tagger's lead over the word form is positive at fewer
than three seeds of four, so *all leak: at full training all that English part of speech from the
sentence adds over the word form is the tagger's leak, as for case*. Its caveat: at full training the
seeds disagree on where the lead sits, so the leak's share, 77%, is a mean over seeds that split.

**What it changes for the thesis.**
1. **At full training, in both languages, what the sentence adds over the word form is not context.**
   For English part of speech the causal tagger keeps nothing reliable over the word form (+0.0005 ±
   0.0011, 2/4); the lead in context, +0.0022, is mostly the leak by its mean. For Ukrainian case it is
   all the leak (§33.81); for gender and number the lead is small and has no leak (§33.133).
2. **The text's conclusion on context can state full training for both languages.**

### 33.136 The text's request of 24 October, and the empty slot at width 1024 prepared

The text, checked against `b36c89f`, asks for three things, in order: (1) width 1024 at 20k on the
151M-token corpus, so that the text is not repeated; (2) the empty slot at width 1024; (3) the work's
size: trained models, GPU-hours by machine, training tokens. The text is complete without them; each
lifts a limit it states. The second machine is free over the weekend, so (1) and (2) both run. (2)
goes first: its data is already on that machine, while (1) needs its BPE 17k null's codes there.

**Item 2, the empty slot at width 1024.** At 85M parameters the tokenizer's gain is measured between
the two tokenizers' same-shape nulls (§33.132). Method point 10 asks for the empty slot, the null
switched off from step 0, at every model size. In the 5M model the null costs endings apart about
0.002, so the gain between the empty slots is larger (+0.0203 against +0.0175); at 30M the two
references agree (+0.0142, +0.0141; §33.118).
- The empty slot at width 1024 under BPE 17k and endings apart, ~38M tokens, 10k, the 1,000-step
  warm-up of the nulls it is set against (§33.132), seeds 0–3: 8 runs, about 6 h.
- The chain (`scripts/chains/second_2026-10-24.sh`, piece empty1024): the first seed's two runs must
  pass the pair check. Rehearsed with the stub, four scenarios: everything runs (8 runs, each switched
  off from step 0, with the warm-up); a failed pair check skips the other seeds; END_BY leaves every
  run for later; with every run done nothing runs. Its cells ran at widths 128 and 512 on that
  machine, so no new loader path is involved.

**Predictions** (before any run):
- **P307** (~65%). The tokenizer's gain between the empty slots is within 0.002 of its gain between the
  nulls (+0.0110 ± 0.0011).
- **P308** (~70%). Under each tokenizer the empty slot ends within 0.002 of its null on average.
- **P309** (~75%). The tokenizer's gain between the empty slots is positive at every seed.

**What each outcome says** (the text's sentences, fixed before the run):
- the gain between the empty slots within 0.002 of the gain between the nulls: *at width 1024 the two
  references agree, as at width 512*;
- further apart: *at width 1024 the tokenizer's gain over the empty slots is X*, with the null's cost
  or gain at that size.

### 33.137 Item 1 prepared: width 1024 without repeating the text

The text's item 1 (§33.136). At 20k on ~38M tokens the BPE 17k null stops improving after 4.3 passes,
and the last 1,500 steps add about 0.0025 to the tokenizer's gain (§33.124). On the 151M-token corpus
20k steps are about 1.1 passes.

**The design.** The 20k runs of §33.124 had the 100-step warm-up, which splits the seeds at this width
(§33.132). Set against them, a 151M-token run with the 1,000-step warm-up would change two things. So
both corpora run under the 1,000-step warm-up, and the two differ in the repetition alone.
- The nulls at width 1024, 20k, 1,000-step warm-up, under BPE 17k and endings apart, seeds 0–3, on the
  151M-token corpus: 8 runs.
- The same on ~38M tokens: 8 runs, the repeated text's partner under the same recipe.
- Then MultiBLiMP on each set. In all 16 runs, about 25 h on the second machine (89 min a run), after
  the empty slot.

**The data.** The BPE 17k 151M stream's ids on the second machine equal the thesis machine's pack
(SHA-256). Its null's codes went there from the pack, decompressed on arrival and checked against the
pack's SHA-256. A null under BPE 17k on that stream ran 30 steps through the real loader there (on the
CPU, the GPU being busy).

**The chain** (`scripts/chains/second_wide_2026-10-24.sh`; pieces wide151, wide38, mbwide) starts when
the empty slot's chain lets go of its lock. Each piece's first seed must pass the pair check.
Rehearsed with the stub, five scenarios: everything runs (16 runs, 151M first, each with the warm-up);
a failed pair check on 151M skips its other seeds; END_BY leaves every run for later; with every run
done MultiBLiMP runs once per corpus; with its records there it runs not at all.

**Predictions** (before any run):
- **P310** (~70%). Without repeating the text, the tokenizer's gain at width 1024 at 20k is positive at
  every seed.
- **P311** (~50%). It is more than 0.002 below the gain on the repeated text under the same warm-up:
  repetition adds to the 20k gain.
- **P312** (~65%). Without repetition, the gain over the last 5,000 steps stays within 0.002 of its
  value at step 15,000.
- **P313** (~70%). MultiBLiMP without repetition is positive at every seed and within 1 point of the
  repeated text's.

**What each outcome says** (the text's sentences, fixed before the run), X the gain without repetition
and Y on the repeated text, both with the 1,000-step warm-up:
- X within 0.002 of Y or above: *without repeating the text the gain at width 1024 is X*, and the
  caveat about repetition goes;
- X more than 0.002 below Y: *without repeating the text the gain at width 1024 is smaller: X*.

### 33.138 Item 3: the work's size, and a term in the Ukrainian figures

The text's item 3 (§33.136) asks for the work's size, from the records, for the Introduction.
`scripts/ops/work_size.py` reads every run record (`work_size_20261024T021450`):

| machine | trained models | GPU-hours | training tokens |
|---|---|---|---|
| thesis machine, RTX 4060 Laptop | 569 | 284.1 | 117.6 billion |
| second machine, RTX 5090 Laptop | 376 | 93.3 | 70.3 billion |
| **in all** | **945** | **377.5** | **188.0 billion** |

- 25 of the thesis machine's models come from September records that name no GPU, an older record
  format without an environment block. The second machine first trained on 16 October and every
  record of it names its GPU, so they are the thesis machine's.
- 152 of the second machine's models repeat the thesis machine's runs (twins).
- Tokens are the steps completed times 32 windows of 256 tokens. Toy rehearsals and smoke tests run
  in copies of the code, leave no record, and are not counted.
- A first count a minute earlier (`work_size_20261024T021411`) listed the September records apart, as
  an unnamed GPU; this one gives them to their machine.

The home is the results record, §3. The weekend's runs add to it, and the tool counts again.

**A term.** The thesis now calls the same-shape null «контроль тієї самої архітектури», not
«…тієї самої форми», which readers took for the word form. The Ukrainian Q2 figure (`fig_q2_end`) says
so, its axis label now on three lines. The English figures keep "same-shape null".

### 33.139 Item 2: at width 1024 the tokenizer's gain over the empty slots is smaller than between the nulls

The empty slot's chain (`second_2026-10-24.sh`, §33.136) ran on the second machine from 22:00 EDT on
23 October to 03:55 on 24 October: 8 runs of 44 min, none failed, and the pair check passed. Every
run's best step was its last. Paired by seed within the machine against the nulls of §33.132, which
had the same warm-up (`paired_cells_20261025T061543`; along training `exposure_axis_20261025T061652`,
whose new option `--warmup` reads the labels' `_wu1000`):

| width 1024, ~38M tokens, 10k, 1,000-step warm-up | BPE 17k | endings apart | the tokenizer's gain |
|---|---|---|---|
| the null (§33.132) | 0.8207, 0.8256, 0.8195, 0.8207 | 0.8112, 0.8113, 0.8095, 0.8104 | +0.0110 ± 0.0011 (1.34%, 4/4) |
| the empty slot | 0.8186, 0.8213, 0.8203, 0.8196 | 0.8117, 0.8121, 0.8109, 0.8135 | **+0.0079 ± 0.0008 (0.96%, 4/4)**: +0.0069, +0.0091, +0.0093, +0.0061 |
| the empty slot over the null | **+0.0017 ± 0.0011 (3/4)**: +0.0021, +0.0044, −0.0008, +0.0010 | **−0.0015 ± 0.0006 (0/4)**: −0.0005, −0.0008, −0.0014, −0.0032 | |

- **The two references differ by 0.0032 ± 0.0010, at every seed** (+0.0026, +0.0052, +0.0006, +0.0042).
  The null's slot costs BPE 17k and helps endings apart, so the gain between the nulls is the larger.
  In the 5M model it is the other way round: there the null costs endings apart, and the gain between
  the nulls is the smaller (+0.0175 against +0.0203). At 30M parameters the two agree (§33.118).
- **Each tokenizer's part is small.** Under BPE 17k the empty slot is better at three seeds of four;
  under endings apart it is worse at all four, by 0.0032 at seed 3.

**Along training** (the empty slot over the null at the same null BPB; positive: the empty slot is better):

| null BPB | 1.46 | 1.33 | 1.21 | 1.08 | 0.95 | 0.83 | the end of the 10k budget |
|---|---|---|---|---|---|---|---|
| BPE 17k | −0.0028 ± 0.0006 (0/4) | −0.0044 ± 0.0008 (0/4) | −0.0044 ± 0.0006 (0/4) | −0.0016 ± 0.0005 (0/4) | **+0.0033 ± 0.0020 (4/4)** | +0.0015 ± 0.0012 (3/4) | +0.0017 ± 0.0011 (3/4) |
| endings apart | −0.0024 ± 0.0005 (0/4) | −0.0030 ± 0.0006 (0/4) | −0.0033 ± 0.0009 (0/4) | −0.0031 ± 0.0005 (0/4) | −0.0041 ± 0.0005 (0/4) | −0.0015 ± 0.0007 (0/4) | −0.0015 ± 0.0006 (0/4) |
| the references' gap in the tokenizer's gain | −0.0004 ± 0.0006 (2/4) | −0.0014 ± 0.0008 (1/4) | −0.0011 ± 0.0010 (1/4) | +0.0015 ± 0.0007 (3/4) | **+0.0074 ± 0.0019 (4/4)** | +0.0030 ± 0.0011 (4/4) | +0.0032 ± 0.0010 (4/4) |

Early in training the null's slot helps under both tokenizers, as at widths 128 and 512 (§33.118).
Under BPE 17k the help turns into a cost from about null BPB 1.0; under endings apart the slot still
helps at the end of the budget, less than at 0.95. The gap between the references is largest at
null BPB 0.95 and smaller by the end of the budget; whether it closes with longer training is not
measured.

**Predictions** (§33.136).
- **P307 wrong:** the two gains are 0.0032 apart (+0.0079 against +0.0110), beyond the rule's 0.002.
- **P308 right:** under BPE 17k the empty slot ends 0.0017 better than the null, under endings apart
  0.0015 worse, each within 0.002.
- **P309 right:** +0.0061 to +0.0093, positive at every seed.

The tally is 187 right, 13 half right, 100 wrong, 8 never tested, 1 withdrawn and 4 pending, of 313.

**The sentence** (the rule of §33.136): further apart than 0.002. *At width 1024 the tokenizer's gain
over the empty slots is +0.0079 (0.96%), positive at every seed; at this size the null's slot costs
BPE 17k 0.0017 and helps endings apart by 0.0015, so the gain between the nulls (+0.0110) is the
larger by 0.003.*

**What it changes for the thesis.**
1. **At 85M parameters the tokenizer's gain has its second reference:** +0.0079 ± 0.0008 (0.96%, 4/4)
   over the empty slots, against +0.0110 ± 0.0011 (1.34%) between the nulls. Over either it is below
   width 512's (+0.0142 over its empty slots, with the 100-step warm-up, which changes neither null at
   that width; §33.118, §33.132), so "smaller beyond 30M parameters" holds over both references, and
   at every seed.
2. **Method point 10 holds at 85M too, with the opposite sign.** In the 5M model the gains between the
   nulls are conservative; at 85M they overstate the tokenizer's gain by 0.003. At 30M the null costs
   nothing. The null is a fair reference only where the empty slot shows it.
3. **The 20k figures at width 1024 are between the nulls** (§33.124; item 1, §33.137). At 10k that
   reference gives 0.003 more than the empty slot at this width; the gap at 20k is not measured.

### 33.140 The queue reviewed: full training at 30M parameters, and English at 30M, prepared

Nothing the text asked for is left to run, and the second machine has until Monday, 26 October. The
queue was reviewed against the limits the text states. Two of its headline claims rest on the 5M model
at full training:
- the tokenizer's gain lasts (+0.0165, 1.75%, at 151M tokens and 50k);
- the features' gain is gone at the end of training (no feature beyond about 0.002 over its null).

At 30M parameters (width 512) the tokenizer ran to 20k on ~38M tokens (+0.0102), and case to 10k, where
it still gains +0.0020 over its null at null BPB 0.88 (§33.118). English endings apart (+0.0046 at 10k)
is measured in the 5M model only. The text ranked the candidates on 25 October: first the full
training at 30M, then English at 30M, then what case's lead over the word form is made of, but only
if case keeps a gain. The modern block and widths beyond 85M it does not need.

**The chain** (`scripts/chains/second_2026-10-25.sh`), at 151M tokens and 50k unless noted, seeds 0–3:
- full512: the nulls at width 512 under BPE 17k and endings apart, and case (tagger) under BPE 17k:
  12 runs of about 80 min;
- mb512: MultiBLiMP on the nulls;
- en512: English plain baselines at width 512 under BPE 17k and endings apart, 10k: 8 runs of about
  16 min;
- lex512, **only if case keeps more than 0.002 over its null in full512**: case from the word form and
  case without the leak (prepositions read from their form, as for G-77), under BPE 17k: 8 runs;
- empty512, **only if the tokenizer's gain in full512 is below 0.006**: the empty slot under both
  tokenizers: 8 runs. Below 0.006, a gap like width 1024's (0.0032 ± 0.0010, §33.139) could take the
  gain to zero over the empty slots.

The chain decides the two conditions itself, from the records, once full512's four seeds are in. The
100-step warm-up is kept: at width 512 the longer one changes neither null (§33.132). Every cell's data
is on the second machine except the English endings-apart stream, which goes there before en512 starts.
Rehearsed with the stub in seven scenarios:
- case keeps a gain: 28 runs, the word form and its leak control run, the empty slot does not;
- case's gain is gone and the tokenizer's gain is low: 28 runs, the empty slot runs, the word form does
  not;
- a failed pair check in full512 skips its other seeds;
- END_BY leaves every long set for later;
- with every run done, MultiBLiMP runs once;
- with its record there, MultiBLiMP runs not at all;
- with full512's seeds missing, neither condition is decided and neither piece runs.

**Predictions** (before any run):
- **P314** (~80%). At width 512 and full training the tokenizer's gain is positive at every seed.
- **P315** (~75%). It is more than 0.002 below the 5M model's at full training (+0.0165).
- **P316** (~70%). Case (tagger) under BPE 17k ends within 0.002 of its null on average.
- **P317** (~75%). On MultiBLiMP endings apart is ahead at every seed, by at least the 5M model's lead
  at full training (+4.9 points).
- **P318** (~65%). English endings apart at width 512 and 10k beats BPE 17k at three seeds of four or
  more, with a positive mean.
- **P319** (~60%; only if lex512 runs). The leak (case over case without it) is at least half of case's
  lead over the word form.
- **P320** (~60%; only if empty512 runs). The tokenizer's gain between the empty slots is within 0.002
  of its gain between the nulls.

**What each outcome says** (the text's sentences, fixed before the run):
- the tokenizer, X the gain between the nulls:
  - X within 0.002 of +0.0165 or above: *at full training the tokenizer gains 1.75% in the 5M model and
    X% in the 30M*;
  - more than 0.002 below: *in the 30M model it is smaller: X%*.

  Either way the reference is named: between the nulls, which agree with the empty slots at this width
  at 10k.
- case under BPE 17k:
  - within 0.002 of its null on average: *in the 30M model too, at the end of full training, case
    does not beat the same-architecture control by more than about 0.002*. The full-training claims
    then hold at 5M and 30M parameters.
  - more than 0.002: *in the 30M model case keeps X at full training*. Those claims keep the 5M scope,
    and the law gains a sentence: the gain goes in the 5M model, and at full training in the 30M model
    it is still there.
- English, positive at three seeds of four or more with a positive mean: *in English too, at 30M
  parameters: X%*; otherwise: *only in the small model*.
- what case's lead over the word form is made of, if lex512 runs:
  - the leak at least half of the lead: *at 30M parameters too, case's lead over the word form at full
    training is mostly the leak*;
  - otherwise: *in the 30M model context itself gives X*.

The tally is 187 right, 13 half right, 100 wrong, 8 never tested, 1 withdrawn and 11 pending, of 320.

### 33.141 Item 1: without repeating the text the gain at width 1024 is smaller, and still rising

The chain of 24 October (`second_wide_2026-10-24.sh`, §33.137) ran on the second machine from 03:55 EDT
on 24 October to 03:38 on 25 October: 16 runs of 88–89 min, then MultiBLiMP on each corpus. None failed,
and both pair checks passed. Paired by seed within the machine (`paired_cells_20261026T101353`; at step
15,000, `paired_cells_20261026T101504`, read with the tool's new option `--step`;
`multiblimp_eval_20261025T073703`, 151M tokens; `…T073758`, ~38M):

| width 1024, 20k, 1,000-step warm-up | BPE 17k | endings apart | the gain |
|---|---|---|---|
| ~38M tokens (4.3 passes) | 0.7688, 0.7705, 0.7682, 0.7691 | 0.7583, 0.7593, 0.7590, 0.7581 | +0.0105 ± 0.0004 (1.36%, 4/4) |
| 151M tokens (about 1.1 passes) | 0.7512, 0.7548, 0.7505, 0.7520 | 0.7436, 0.7459, 0.7433, 0.7428 | **+0.0082 ± 0.0005 (1.09%, 4/4)**: +0.0077, +0.0089, +0.0072, +0.0092 |

- **Without repetition the gain is smaller, at every seed,** by 0.0022 ± 0.0002 (−0.0028, −0.0023, −0.0020,
  −0.0018). Both nulls gain from the larger corpus, BPE 17k's more: by 0.0170 ± 0.0005 against 0.0148 ±
  0.0005.
- **It is still rising:** +0.0074 at step 15,000, +0.0008 more by the end (+0.0001 to +0.0011). On the
  repeated text the gain is level over the last 5,000 steps (−0.0003).
- **With the 1,000-step warm-up the repeated text's gain is larger** than with 100 steps (+0.0075 ± 0.0018,
  §33.124), and every seed keeps it (+0.0092 to +0.0112).
- **Grammar holds without repetition:** MultiBLiMP +5.4 ± 0.3 points (4/4): number +6.4, gender +4.3,
  person +5.8. On the repeated text it is +5.1 ± 0.1 (4/4).

**Predictions** (§33.137).
- **P310 right:** +0.0072 to +0.0092, positive at every seed.
- **P311 right:** 0.0022 below the repeated text's gain, beyond 0.002.
- **P312 right:** +0.0008 over the last 5,000 steps.
- **P313 right:** +5.4 against +5.1 points, positive at every seed.

The tally is 191 right, 13 half right, 100 wrong, 8 never tested, 1 withdrawn and 7 pending, of 320.

**The sentence** (the rule of §33.137: X more than 0.002 below Y). *Without repeating the text the gain at
width 1024 is smaller: +0.0082 (1.09%) at 20k, at every seed, and still rising.*

**What it changes for the thesis.**
1. **The 20k figure at width 1024 is measured without repetition:** +0.0082 (1.09%). The caveat about
   the repeated text goes. On that text, with the same warm-up, it is +0.0105 (1.36%).
2. **Repeating the text added to the gain:** BPE 17k's null loses more from the repetition than the
   endings-apart null.
3. **The grammatical gain does not depend on the repetition:** +5.4 points against +5.1.

### 33.142 Full training at 30M parameters: the tokenizer's gain is smaller and lasts, case stays within the bound, English gains nothing

The chain of 25 October (`second_2026-10-25.sh`, §33.140) ran on the second machine from 03:38 to 21:56
EDT on 25 October: 12 runs of about 80 min, MultiBLiMP, and 8 English runs of 16 min. None failed, and both
pair checks passed. The chain decided both conditions from the records:
- case keeps +0.0015 over its null, not more than 0.002, so the split of its lead (lex512) did not run;
- the tokenizer's gain is +0.0072, not below 0.006, so the empty slots (empty512) did not run.

Paired by seed within the machine (`paired_cells_20261026T101353`; along training
`paired_cells_20261026T101648`, `…T101650`, `…T101653`, at steps 10,000, 25,000 and 45,000;
`multiblimp_eval_20261025T234639`). 50k steps read the 151M-token corpus about 2.7 times.

| width 512, 151M tokens, 50k | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| the null, BPE 17k | 0.7447 | 0.7450 | 0.7442 | 0.7438 | 0.7444 |
| the null, endings apart | 0.7371 | 0.7378 | 0.7365 | 0.7374 | 0.7372 |
| case (tagger), BPE 17k | 0.7427 | 0.7424 | 0.7432 | 0.7436 | 0.7430 |
| the tokenizer's gain | +0.0076 | +0.0073 | +0.0076 | +0.0065 | **+0.0072 ± 0.0003 (0.97%, 4/4)** |
| case over its null | +0.0020 | +0.0027 | +0.0009 | +0.0002 | **+0.0015 ± 0.0005 (4/4)** |

- **The tokenizer's gain is smaller than the 5M model's, and it lasts.** It is +0.0078 at step 10,000,
  +0.0072 at 25,000, +0.0077 at 45,000 and +0.0072 at the end. That is less than half of the 5M model's
  +0.0165 (1.75%), and 0.003 below width 512's at 20k on ~38M tokens (+0.0102).
- **Case stays within the bound, but its gain does not go.** It is +0.0017 at step 10,000, +0.0014 at 25,000
  and at 45,000, and +0.0015 at the end, positive at every seed at each point. In the 5M model, on the same
  text, case gains nothing with a 20k budget (−0.0003) and ends full training below its null (−0.0013 ±
  0.0008).
- **Grammar:** MultiBLiMP gives +5.0 ± 0.6 points (4/4, from +3.9 to +6.3): number +5.4, gender +4.2, person
  +5.6. The 5M model's at full training is +4.9, on the thesis machine; at width 512 the second machine's
  models gain 0.9 points less (§33.122).
- **English gains nothing at 30M parameters and 10k.** The plain baselines at width 512:
  - BPE 17k: 1.2605, 1.2598, 1.2625, 1.2610;
  - endings apart: 1.2609, 1.2602, 1.2611, 1.2608;
  - the gain: −0.0003, −0.0004, +0.0014, +0.0002, **+0.0002 ± 0.0004 (2/4)**.

  In the 5M model it is +0.0046 ± 0.0010 (4/4).

**Predictions** (§33.140).
- **P314 right:** +0.0065 to +0.0076, positive at every seed.
- **P315 right:** 0.0093 below the 5M model's.
- **P316 right:** +0.0015 on average, though positive at every seed.
- **P317 right by its letter:** +5.0 on average, at least +4.9. Seeds 0 and 2 gain less (+3.9, +4.2).
- **P318 wrong:** positive at two seeds of four.
- **P319 and P320 not tested:** their conditions were not met.

The tally is 195 right, 13 half right, 101 wrong, 10 never tested and 1 withdrawn, of 320; none pending.

**The sentences** (the rules of §33.140).
- **The tokenizer:** X is more than 0.002 below +0.0165. *In the 30M model it is smaller: 0.97%*, between
  the nulls, which agree with the empty slots at this width at 10k.
- **Case:** within 0.002. *In the 30M model too, at the end of full training, case does not beat the
  same-architecture control by more than about 0.002.* Unlike in the 5M model, it keeps that small gain at
  every seed, level from step 10,000.
- **English:** *only in the small model.*

**What it changes for the thesis.**
1. **Full training at two model sizes:** the tokenizer's gain lasts at 30M parameters too, at 0.97%, smaller
   than the 5M model's 1.75%.
2. **The features' end-of-training answer holds at 30M by its bound, but "gone" is the 5M model's.** At 30M,
   case falls to about +0.0015 by step 10,000 and stays there to the end, at every seed.
3. **English endings apart helps the small model only:** at 30M parameters and 10k it gains nothing.
4. **The grammatical gain at full training is about the same at 30M as at 5M:** +5.0 and +4.9 points, on
   two machines.

### 33.143 The text's list of limits of 30 October: the learning rate, the plain baselines, the warm-up at width 128 and width 1024 at full training, prepared

On 30 and 31 October the thesis text listed fifteen limits a reviewer could fairly call gaps, ranked in
three tiers. The first tier:
- where the cut falls, at matched fertility;
- the learning rate;
- width 1024 at full training;
- shuffled case;
- switching a feature off to a constant instead of to zeros.

Each item is costed in DIRECTIONS' queue from the recorded timings. Four of them need no new code and run
now on the second machine. The others need code, a condition or a download first, and wait.

**The chain** (`scripts/chains/second_2026-10-31.sh`). The 5M model at 151M tokens and 50k, unless noted:
- **lr0**, the learning rate (item 1): the nulls under BPE 17k and endings apart at 1e-3 and 3e-3, and
  BPE 17k's null at the default 3e-4 as the thesis machine's twin, all at seed 0. 5 runs of ~21 min.
- **lrbest**, decided on lr0: the rate at which BPE 17k's null ends lowest at seed 0, of the three. If it
  is not the default, both nulls at that rate, seeds 1–3: 6 runs.
- **plain128**, the tokenizer without a slot (item 12): plain baselines under both tokenizers, seeds 0–3,
  and BPE 17k's null at the default rate, seeds 1–3. 11 runs.
  - BPE 17k's runs are the thesis machine's twins, and this machine already holds endings apart's nulls at
    that rate.
  - So the gain between the nulls at the default rate is paired by seed here, for items 1 and 12 alike.
- **wu128**, the 1,000-step warm-up at width 128 (item 14): the nulls under both tokenizers, ~38M tokens,
  10k, seeds 0–3. 8 runs of ~4 min.

**Then** (`scripts/chains/second_wide_2026-10-31.sh`), item 2:
- **full1024:** the nulls at width 1024 under both tokenizers, 151M tokens, 50k, with the 1,000-step
  warm-up, seeds 0–3. 8 runs of ~3.7 h, each seed's pair together.
- **mb1024:** MultiBLiMP on the eight.

About 8.5 hours, then about 30. Both chains must end by 08:00 New York time on 2 November.

**Two choices.**
- **The rate is searched at the full budget, not at 10k.**
  - The English sweep at 10k (results §5.21) put 3e-3 first and the default 7.8% behind. Its caveat: a
    10k horizon favours higher rates under the cosine schedule.
  - The search at 50k costs about as much as one at 10k followed by full training.
  - The rate is chosen for BPE 17k, the reference, so the run answers whether the reference was held back
    by its rate.
- **Width 1024 runs between the nulls,** as at 5M and 30M parameters at full training, and in width 1024's
  cells at 10k and 20k.
  - So its gain compares like with like along both axes.
  - At this width and 10k the nulls overstate the gain by 0.0032 against the empty slots (§33.139). The text
    states that bound beside it.
  - A second reference would double the 30 hours.

**Rehearsed** with the stub in ten scenarios:
- a new rate wins and lrbest runs; the default wins and it does not;
- lr0's twin is off by more than 0.01, and lrbest does not run;
- plain128's twin is off, and its other seeds are skipped;
- a seed-0 run is missing, and lrbest is not decided;
- END_BY leaves whole sets for later, in both chains;
- the wide chain with all eight runs to do; with all eight done, and MultiBLiMP runs once; with
  MultiBLiMP's record there, and it does not run again.

**Predictions** (before any run):
- **P321** (~80%). At seed 0, BPE 17k's null ends lower at 1e-3 or 3e-3 than at the default 3e-4, so lrbest
  runs.
- **P322** (~80%; only if lrbest runs). At the better rate the tokenizer's gain between the nulls is
  positive at every seed.
- **P323** (~55%; only if lrbest runs). It is within 0.003 of the gain at the default rate on the same
  machine.
- **P324** (~90%). Between the plain baselines at full training the tokenizer's gain is positive at every
  seed.
- **P325** (~65%). It is more than 0.002 larger than between the nulls on the same machine. In the 5M
  model the null costs endings apart about 0.002, and in English a same-shape slot helps a plain baseline
  (FINDINGS, [the same-shape null](../../FINDINGS.md#the-same-shape-null-is-not-neutral)).
- **P326** (~80%). With the 1,000-step warm-up, the 5M model's gain at 10k on ~38M tokens is within 0.002 of
  its gain with 100 steps on the same machine (+0.0175 ± 0.0015).
- **P327** (~85%). At width 1024 and full training the tokenizer's gain between the nulls is positive at
  every seed.
- **P328** (~65%). It is at least 0.87% of BPE 17k's null BPB: within 0.1 point of the 30M model's 0.97%, or
  above it.
- **P329** (~80%). On MultiBLiMP, endings apart at width 1024 and full training is ahead at every seed, by
  at least 4 points.

**What each outcome says** (the text's sentences, fixed before the run):
- **The rate:**
  - the default is the best of the three: *3·10⁻⁴ is the best of three rates for the reference at full
    training*;
  - a new rate wins, and the gain moves by at most 0.003: *the gain does not depend on the learning rate*;
  - the gain is more than 0.003 smaller: the headline is restated at the better rate.

  Either way the text says why 3·10⁻⁴.
- **The plain baselines:**
  - within 0.002 of the nulls' gain: *it does not depend on the slot*;
  - more than 0.002 larger: the figure between the nulls is conservative;
  - more than 0.002 smaller: a correction to the headline.
- **The warm-up:** within 0.002, the 5M results do not depend on it. Otherwise the text names the warm-up
  with them.
- **Width 1024,** X its gain in per cent:
  - within 0.1 point of 0.97%: *from 30M parameters the gain does not fall*;
  - above that: *at 85M it is no smaller than at 30M*;
  - below: *it falls with each size: 1.75 / 0.97 / X%*.

**Waiting:**
- **items needing code, a condition or a download:** 9, 10 and 11 of the first tier, then 7 and 13;
- **the analyses without the GPU;**
- **item 6's minimal pairs,** which the author checks on a random hundred;
- **item 8,** the tagger on gold labels, after the treebank is downloaded.

**Dropped:**
- **item 15,** the word-form control with an unseen-word class: it would make the split of case's lead at
  10k exact, a number rather than the answer (G-76);
- **Morfessor in place of the dictionary,** this week: it needs an install and hours of training before
  any run, and item 9 and Unigram bracket the question.

The tally is 195 right, 13 half right, 101 wrong, 10 never tested, 1 withdrawn and 9 pending, of 329.

### 33.144 The first tier on the thesis machine prepared: shuffled case, the switch to a constant, and where the cut falls

The second machine is busy with width 1024 at full training until Sunday (§33.143). The three cheap items of
the first tier therefore run on the thesis machine, which holds every run each is compared with.

**New code** (`morphlm/`, tests in `tests/test_blocks.py` and `tests/test_endings.py`):
- **`ucaseshuf`**, shuffled case: case from the word form, each form's majority value reassigned at random
  among word types of similar frequency, as `mstructshuf` is for structure.
  - Derived at ~38M tokens under BPE 17k (`derive_lexical_codes_20261031T063236`).
  - The class of 72% of tokens changed; the classes' shares moved by 1.4%.
- **`frozen.feature_const_step`**, the switch to a constant: from step K every value of the feature but
  'none' becomes its block's first class (for part of speech, NOUN).
  - The information goes, and a shared vector stays where the switch to zeros left nothing.
  - As after that switch, only evaluations after the step count.
- **`bpe-endings-shift1`**, a tokenizer named `bpeshift_uk_17k`: BPE 17k trained as endings apart is, on the
  same words cut once each, but one letter off the rule's boundary (endings.ShiftedEndingRule).
  - The direction comes from a stable hash of the form; a one-letter ending can only move earlier.
  - Its data is built beside the chain: the tokenizer, its statistics on held-out text beside BPE 17k's and
    endings apart's, then `prepare_condition.py` at ~38M tokens.
- A test of the tokenizer loader had failed since 21 October: an audit script opened a tokenizer directly.
  It now opens it through the loader.

**The chain** (`scripts/chains/tulpar_2026-10-31.sh`):
- **shuf** (item 10): shuffled case under BPE 17k, ~38M tokens, 10k, seeds 0–3. 4 runs of ~12 min.
- **const** (item 11): English part of speech set to one class at step 5,000 of 20k, the seeds of the switch
  to zeros (0–4). 5 runs of ~23 min.
- **cutshift** (item 9): the null under the shifted cut, ~38M tokens, 10k, seeds 0–3. 4 runs of ~12 min,
  once its data is built.

About 3.7 hours of the GPU.

**The references,** all on this machine at the same budgets:
- the null, case, case from the word form and case without the leak;
- the switch to zeros (OFF5000), the empty slot (OFF0) and the null at 20k in English;
- the nulls under BPE 17k and endings apart.

**Rehearsed** with the stub:
- every run to do;
- a failed first-run check in shuf;
- cutshift's data not ready;
- cutshift waiting for its build and then running;
- END_BY reached during that wait.

**Predictions** (before any run):
- **P330** (~60%). At 10k, shuffled case ends within 0.001 of the null on average.
- **P331** (~75%). It gains less than half of case's +0.0032 over the null on average.
- **P332** (~65%). Set to one class at step 5,000 of 20k, English part of speech ends at least 0.002 below
  the model never fed it (the empty slot) on average, and below it at four seeds of five or more.
- **P333** (~55%). That deficit is within 0.002 of the switch to zeros' (+0.0039).
- **P334** (~75%). The shifted cut's null gains less than half of endings apart's +0.0174 over BPE 17k's
  null at 10k.
- **P335** (~55%). It gains less than +0.004, or loses.
- **P336** (~75%). Its characters per token on held-out text are within 3% of endings apart's.

**What each outcome says** (the text's sentences, fixed before the run):
- **Shuffled case:**
  - about zero (P330): *what acts is the feature's content*;
  - as much as case: *any grouping of words gives the early gain*.
- **The switch to a constant:**
  - the same deficit: the conclusion stands;
  - none: *the deficit comes from the change of the input*, and §3.8 is rewritten.
- **The cut moved:**
  - no gain: *it is the morpheme boundary that matters*;
  - the same gain, within 0.003 of endings apart's: *cutting off a word end of typical length is enough;
    the dictionary is not needed*;
  - in between: the boundary gives the rest.

The tally is 195 right, 13 half right, 101 wrong, 10 never tested, 1 withdrawn and 16 pending, of 336.

### 33.145 The learning rate, the plain baselines and the warm-up at width 128: at the best of three rates the gain is smaller; without the slot it is larger

The chain of 31 October (`second_2026-10-31.sh`, §33.143) ran on the second machine from 18:53 EDT on
30 October to 03:07 on 31 October. It made 30 runs: 22 at full training of 20.5–21.2 min, and 8 at 10k
of ~4.2 min. None failed.
- **The twin checks passed.** BPE 17k's null at the default rate ended 0.0003 from the thesis machine's at
  seed 0, and its plain baseline 0.0050.
- **The twins over all four seeds:** the null's within −0.0001 to +0.0015 of the thesis machine's, the
  plain baseline's within −0.0050 to +0.0022.
- **The rate was decided:** of the three rates, BPE 17k's null at seed 0 ended lowest at 3e-3, so lrbest ran
  there.

Paired by seed within the machine: `paired_cells_20261031T070744`. An earlier record of the same pairs with
each pair's cells the other way round, `…T064544`, gives the same numbers with the opposite sign.

**Item 1, the learning rate** (the 5M model, 151M tokens, 50k, the nulls):

| | BPE 17k | endings apart | the tokenizer's gain |
|---|---|---|---|
| 3e-4, the default | 0.9422, 0.9464, 0.9436, 0.9463 | 0.9273, 0.9279, 0.9285, 0.9259 | +0.0172 ± 0.0013 (1.82%, 4/4) |
| 1e-3, seed 0 | 0.9041 | 0.8948 | +0.0093 |
| 3e-3 | 0.8952, 0.8966, 0.8943, 0.8961 | 0.8819, 0.8808, 0.8838, 0.8814 | **+0.0136 ± 0.0011 (1.52%, 4/4)** |
| 3e-3 over the default | **+0.0491 ± 0.0007 (5.2%, 4/4)** | +0.0454 ± 0.0006 (4/4) | **−0.0037 ± 0.0009 (0/4)** |

- **The default is not the best rate for the 5M model at full training.**
  - 3e-3 is better for both tokenizers, by about 5%.
  - At seed 0, 1e-3 lies between the two rates, for both tokenizers.
- **At 3e-3 the gain is smaller at every seed,** by 0.0017, 0.0026, 0.0046 and 0.0058. In per cent of the
  reference it falls from 1.82% to 1.52%.
- **The better rate helps BPE 17k more than endings apart** (+0.0491 against +0.0454), so the gap closes
  by about a fifth. A reference held back by its rate inflated a fifth of the gain.
- **At 1e-3 (seed 0) the gain is smaller still,** +0.0093 against +0.0133 at 3e-3. One seed, so it is
  read as no trend.

**Item 12, the tokenizer without a slot** (the same models; the plain baselines have all 128 dims for the
token):
- **The plain baselines:**
  - BPE 17k: 0.9425, 0.9485, 0.9484, 0.9475;
  - endings apart: 0.9222, 0.9251, 0.9270, 0.9250;
  - the gain: **+0.0219 ± 0.0007 (2.31%, 4/4).**
- **Against the nulls' +0.0172 the gain is +0.0046 ± 0.0009 larger, at every seed** (+0.0054, +0.0049,
  +0.0063, +0.0020).
- **The null helps BPE 17k and costs endings apart:**
  - BPE 17k's null beats its plain baseline by +0.0021 ± 0.0010 (4/4);
  - endings apart's ends 0.0026 ± 0.0009 behind its own (0/4).

  At 10k the empty slots told the same story (+0.0203 against +0.0175; FINDINGS, method point 10).

**Item 14, the 1,000-step warm-up at width 128** (~38M tokens, 10k, the nulls):
- **The gain:**
  - with 1,000 steps: BPE 17k 1.0894, 1.0929, 1.0913, 1.0903; endings apart 1.0724, 1.0747, 1.0731,
    1.0737; **+0.0175 ± 0.0004 (4/4)**;
  - with 100 steps on the same machine: +0.0175 ± 0.0015.
- **The difference is +0.0000 ± 0.0017** (−0.0016, −0.0012, +0.0051, −0.0024).
- Both nulls end 0.0016 worse with the longer warm-up.

**Verdicts:**
- **P321 right:** 3e-3 and 1e-3 both beat the default at seed 0, by 0.0470 and 0.0381.
- **P322 right:** +0.0105 to +0.0158 at every seed.
- **P323 wrong:** the gain is 0.0037 smaller, not within 0.003.
- **P324 right:** +0.0203 to +0.0233.
- **P325 right:** 0.0046 larger.
- **P326 right:** +0.0175 against +0.0175.

The tally is 200 right, 13 half right, 102 wrong, 10 never tested, 1 withdrawn and 10 pending, of 336.

**The sentences** (the rules of §33.143):
- **The rate:** a new rate wins, and the gain is more than 0.003 smaller, so the headline is restated at the
  better rate.
  - *At the best of three learning rates the 5M model's gain at full training is 1.52%; at the default rate it
    is 1.82% on the same machine.*
  - Why 3·10⁻⁴: it is the recipe every run shares. The English sweep at 10k (results §5.21) did not show it
    to be worse at 50k, and at full training the 5M reference is 5.2% better at 3·10⁻³.
- **The plain baselines:** more than 0.002 larger, so *the figure between the nulls is conservative: without
  a slot the gain is 2.31%.*
- **The warm-up:** within 0.002, so the 5M results do not depend on it.

**What it changes for the thesis.**
1. **The 5M headline is measured at a rate that holds the reference back by 5%.** At the best of three rates
   the gain at full training is 1.52% (+0.0136), smaller at every seed, and still positive at every seed.
   The 30M and 85M models ran at the default rate only: a limitation the text states.
2. **Between plain baselines, with no slot, the gain is larger: 2.31% (+0.0219).** The nulls understate it
   by 0.0046: at full training too, the null helps BPE and costs endings apart.
3. **The 1,000-step warm-up chosen for width 1024 changes nothing at width 128.**

### 33.146 Items 7 and 13 prepared: the modern block and depth, after width 1024

Width and data were varied, but the transformer's block and its depth (4 layers) never were. The text's
items 7 and 13 ask whether the tokenizer's gain holds under a modern block and in a deeper model. The chain
(`scripts/chains/second_block_2026-10-31.sh`) runs on the second machine once width 1024's full training
ends, at ~38M tokens and 10k:
- **block10** (item 7): the nulls under both tokenizers with the LLaMA-style block (rotary positions,
  RMSNorm, SwiGLU, built in §33.54), width 128, seeds 0–3. 8 runs of ~4.5 min.
- **deep10** (item 13): the nulls at 8 layers of width 512, with the 1,000-step warm-up, seeds 0–3. 8 runs
  estimated at ~24 min each, 32 with the margin.

**The references** are on that machine already: the nulls at width 128 under the standard block, and width
512's 4 layers under the same warm-up (§33.132).
- **Rehearsed** with the stub: every run to do, and a failed pair check.
- **A cell of each piece ran 30 steps** through the real loader on that machine's CPU, while its GPU trained
  width 1024. They had 5.0M and 42.6M trainable parameters.

**Predictions** (before any run):
- **P337** (~80%). Under the LLaMA-style block the tokenizer's gain at width 128 and 10k is positive at every
  seed.
- **P338** (~60%). It is within 0.003 of the standard block's on the same machine (+0.0175 ± 0.0015).
- **P339** (~80%). At 8 layers of width 512 and 10k the gain is positive at every seed.
- **P340** (~55%). It is within 0.003 of 4 layers' under the same warm-up on the same machine (+0.0147 ±
  0.0022).

**What each outcome says** (the text's sentences):
- **The block:** within 0.003, *the gain does not depend on the block's architecture*; smaller, *it depends
  on the architecture*.
- **Depth:** within 0.003, *it does not depend on depth*; smaller, *it falls with depth too*.

The tally is 200 right, 13 half right, 102 wrong, 10 never tested, 1 withdrawn and 14 pending, of 340.

### 33.147 Item 6 prepared: minimal pairs for agreement in the noun phrase and for government

The tokenizer's grammatical gain rests on MultiBLiMP's single phenomenon, a verb agreeing with its subject.
Item 6 of the text's list asks whether it covers the two that an inflectional ending carries inside a noun
phrase and after a preposition.

**The pairs** (`scripts/data/make_minimal_pairs_uk.py`; `data/minimal_pairs/uk_np_pp_v1.tsv`, its sidecar
`.json`). They come from the corpus's test split rather than validation: no checkpoint was ever chosen on
it. The tagger finds the phrases:
- **an adjective** whose head is the noun directly after it, the two agreeing;
- **a noun** directly after one of 24 prepositions whose government is not in doubt (в/у, по and о are left
  out).

pymorphy3 then changes one word:
- **AN-C:** the adjective's case;
- **AN-#:** its number;
- **AN-G:** its gender, in the singular;
- **PP-C:** the noun's case, to one its preposition never governs.

**A pair is kept only if the change is wrong by every reading in the dictionary.** No reading of the new
adjective may agree with any reading of the noun, and no reading of the new noun may be in a case the
preposition governs. Both words must be known, wholly Ukrainian, and free of apostrophes.

**What was built:**
- From 14,973 paragraphs: 11,229, 8,865, 8,380 and 4,275 candidates.
- Kept: 1,000 of each, at one sentence each and a fixed seed; AN-C has 999, after one row with a tab.
- A first build kept singular words only by accident. pymorphy3 names the plural but not the singular,
  so the check now reads a reading without a number as singular.

**The check.** A random 100, 25 per phenomenon (`…_sample100.tsv`), goes to a native speaker on a page that
saves each mark. ✓ when the first sentence is fine and the second wrong because of the changed word; ✗
otherwise, with a note. At 95 or more valid the set stands; below, the rule is tightened and a fresh sample
checked.

**The scoring** follows the check. `multiblimp_eval.py` gains the four phenomena and reads the table with
`--data`. It scores the nulls at full training: the 5M model's on the thesis machine, the 30M model's on
the second.

**Predictions** (before any check or score):
- **P341** (~70%). At least 95 of the 100 checked pairs are valid.
- **P342** (~75%). On the three agreement phenomena together, the 5M model's endings-apart nulls at full
  training beat BPE 17k's at every seed.
- **P343** (~65%). On government, endings apart leads at every seed in the 5M model.

**What each outcome says** (the text's sentences): a gain, *it covers agreement in the noun phrase and
government*; none, *only subject–verb agreement*.

The tally is 200 right, 13 half right, 102 wrong, 10 never tested, 1 withdrawn and 17 pending, of 343.

### 33.148 Items 9, 10 and 11: the morpheme boundary matters, case's content acts, and the deficit is not the input's change

The thesis machine's chain (`tulpar_2026-10-31.sh`, §33.144) ran from 08:43 to 12:07 on 31 October: 13 runs,
none failed, and every first-run check passed. cutshift's data was ready before the chain reached it.

Paired by seed on the thesis machine (`paired_cells_20261102T100043`). The switch at and just after step
5,000: `…T100302`, `…T100304`, `…T100306`.

**Item 9, where the cut falls** (the nulls, ~38M tokens, 10k):
- **The tokenizer** (`tokenizer_stats_ukwiki_20261031T064805`, held-out text). The shifted cut splits the
  words endings apart splits, at the same rate:

  | | characters per token | tokens per word | words split | a boundary at the rule's cut |
  |---|---|---|---|---|
  | BPE 17k | 3.59 | 1.73 | 43.3% | 8.8% |
  | endings apart | 3.21 | 1.99 | 64.4% | 100% |
  | the cut moved one letter | 3.18 | 2.02 | 64.9% | 1.4% |

- **The null under the moved cut ends worse than plain BPE 17k:** 1.1089, 1.1114, 1.1105, 1.1104 against
  1.0891, 1.0910, 1.0874, 1.0911, so **−0.0207 ± 0.0008 (−1.90%, 0/4)**.
  - Endings apart beats it by +0.0381 ± 0.0007 (4/4).
  - Endings apart beats BPE 17k by +0.0174 ± 0.0014.
- **A cut placed one letter off the boundary costs** more than the right cut gains.

**Item 10, shuffled case** (BPE 17k, ~38M tokens, 10k; over the null):

| | the gain over the null |
|---|---|
| case in context (tagger) | +0.0032 ± 0.0007 (4/4) |
| case from the word form | +0.0011 ± 0.0009 (3/4) |
| the word form's values shuffled among types of similar frequency | **−0.0011 ± 0.0004 (0/4)**: −0.0013, −0.0000, −0.0020, −0.0010 |

- **The same grouping of word types,** with the case values reassigned, gains nothing and costs a little.
- **The word form beats its shuffled version** by +0.0022 ± 0.0012 (3/4).

**Item 11, the switch to a constant** (English part of speech, 20k, switched at step 5,000, seeds 0–4):
- **Set to one class (NOUN),** the model ends **0.0038 ± 0.0009 below the model never fed it** (the empty slot,
  5/5).
  - Set to zeros, it ends 0.0039 ± 0.0009 below (5/5).
  - It ends 0.0035 ± 0.0007 below the null, whose slot held one class throughout (5/5).
- **The two switches end within 0.0001 of each other at every seed.**
- **The runs are the same until the switch,** and they part there.
  - At step 5,000 the constant costs 0.0264 ± 0.0012 more than zeros (5/5): the jump is larger with the shared
    vector.
  - By step 5,500 the two are within 0.0002, and by 6,000 within 0.0001.
- The input's change is absorbed within 500 steps, and what remains is the lost information.

**Verdicts:**
- **P330 wrong:** it ends 0.0011 ± 0.0004 below the null, just past the bound (0/4).
- **P331 right:** it loses rather than gains.
- **P332 right:** +0.0038, 5/5.
- **P333 right:** 0.0038 against 0.0039.
- **P334 and P335 right:** the moved cut loses 0.0207.
- **P336 right:** 3.18 characters per token against 3.21.

**The sentences** (the rules of §33.144):
- **The cut moved:** no gain, a loss: *it is the morpheme boundary that matters*.
- **Shuffled case:** about zero, a small cost: *what acts is the feature's content*.
- **The switch to a constant:** the same deficit, so the conclusion stands. *The deficit is lost
  information, not the input's change.*

**What it changes for the thesis.**
1. **"Where the cuts fall matters, not how many there are"** no longer rests on Unigram alone. At the same
   words and the same split rate, the cut one letter off the boundary loses 1.9% to plain BPE, where the cut on
   it gains 1.6%.
2. **«Отже діє саме відмінок»** has its shuffled control: the same grouping of words with case reassigned
   gains nothing.
3. **§3.8's deficit after removal** is not the input's sudden change. A shared vector kept in the slot leaves
   the same deficit.

### 33.149 Items 2, 7 and 13: at 85M parameters the gain at full training is the 30M model's; it holds under a modern block and falls with depth

Width 1024's chain (`second_wide_2026-10-31.sh`, §33.143) ran on the second machine from 03:09 EDT on
31 October to 07:53 EST on 1 November: 8 runs of 222–224 min, then MultiBLiMP. The block-and-depth chain
(`second_block_2026-10-31.sh`, §33.146) followed it, from 07:53 to 12:00 EST: 16 runs. None failed, and every
pair check passed.

Paired by seed on the second machine (`paired_cells_20261102T100005`; item 2 at steps 10,000, 25,000 and
45,000: `…T100255`, `…T100258`, `…T100300`; MultiBLiMP: `multiblimp_eval_20261101T125220`).

**Item 2, width 1024 at full training** (151M tokens, 50k, the 1,000-step warm-up, the nulls):
- **The nulls:**
  - BPE 17k: 0.6969, 0.6987, 0.6960, 0.6968;
  - endings apart: 0.6895, 0.6909, 0.6902, 0.6893;
  - the gain: **+0.0071 ± 0.0004 (1.02%, 4/4)**, from +0.0058 to +0.0078.
- **It is level from step 10,000:** +0.0069 there, +0.0073 at 25,000, +0.0076 at 45,000.
- **Width 512 at full training gains +0.0072 (0.97%).** At 20k on the same corpus, width 1024 gained +0.0082
  (1.09%).
- **MultiBLiMP:** endings apart 89.4% against BPE 17k's 86.2%, **+3.1 ± 0.2 points (4/4)**, from +2.5 to +3.5.
  Width 512 at full training: +5.0.

| at full training | the tokenizer's gain | MultiBLiMP |
|---|---|---|
| 5M (width 128; thesis machine) | +0.0165 (1.75%) | +4.9 |
| 30M (width 512; second machine) | +0.0072 (0.97%) | +5.0 |
| 85M (width 1024; second machine) | **+0.0071 (1.02%)** | **+3.1** |

**Item 7, the LLaMA-style block** (width 128, ~38M tokens, 10k):
- **The nulls under the block:**
  - BPE 17k: 1.0288, 1.0267, 1.0253, 1.0258;
  - endings apart: 1.0136, 1.0113, 1.0112, 1.0097;
  - the gain: **+0.0152 ± 0.0004 (1.48%, 4/4)**.
- **Under the standard block on the same machine the gain is +0.0175 ± 0.0015 (1.60%).** The block's gain is
  0.0023 smaller.
- **The block itself is much better:** BPE 17k's null gains 0.0627 ± 0.0011 (5.8%) from it.

**Item 13, depth** (8 layers of width 512, ~38M tokens, 10k, the 1,000-step warm-up):
- **The nulls at 8 layers:**
  - BPE 17k: 0.8550, 0.8537, 0.8545, 0.8541;
  - endings apart: 0.8450, 0.8498, 0.8449, 0.8459;
  - the gain: **+0.0079 ± 0.0014 (0.93%, 4/4)**, from +0.0040 to +0.0101.
- **4 layers under the same warm-up gain +0.0147 ± 0.0022 (1.67%).** At 8 layers the gain is about half.
- **8 layers improve BPE 17k's null by 0.0263 ± 0.0017 (3.0%).**
- As with width at 10k, the gain shrinks as the model grows.

**Verdicts:**
- **P327 right:** +0.0058 to +0.0078.
- **P328 right:** 1.02%.
- **P329 wrong:** ahead at every seed, but by +3.1 points, not 4.
- **P337 right:** +0.0141 to +0.0161.
- **P338 right:** 0.0023 smaller.
- **P339 right:** +0.0040 to +0.0101.
- **P340 wrong:** 0.0068 smaller.

The tally is 211 right, 13 half right, 105 wrong, 10 never tested, 1 withdrawn and 3 pending, of 343.

**The sentences** (the rules of §33.143 and §33.146):
- **Width 1024:** X = 1.02%, within 0.1 point of 0.97%: *from 30M parameters the gain does not fall*.
- **The block:** within 0.003: *the gain does not depend on the block's architecture*.
- **Depth:** smaller: *it falls with depth too*.

**What it changes for the thesis.**
1. **The gain at full training is 1.75% at 5M parameters, 0.97% at 30M and 1.02% at 85M,** all at the default
   rate. It falls from 5M to 30M, then holds, and lasts at every size.
   - The Introduction's and the general conclusions' full-training claims can name 85M parameters.
   - At 10k and 20k, 85M stays below 30M; at full training the two agree.
2. **The grammatical gain is smaller at 85M:** +3.1 points against +5.0 and +4.9. It is positive at every seed.
3. **A modern block keeps the gain,** at 5M parameters and 10k.
4. **Depth reduces it, as width does,** at 10k: two models of 30M-parameter width, 4 and 8 layers.

### 33.150 Item 6: the tokenizer's grammatical gain covers agreement in the noun phrase and government

**The check.** A native speaker marked the hundred pairs of §33.147 on 3 November
(`data/minimal_pairs/uk_np_pp_v1_sample100_check.tsv`).
Each mark was then checked a second time, independently, and none changed. Two rules were settled along the way:
- **Only the changed phrase is judged.** An error elsewhere in the source sentence is in both sentences, so it
  does not decide the pair. The pairs with one (#019, #044, #048, #078, #090) are valid.
- **Disagreeing in two features is still disagreeing.** «рідкісним види» (#056) is dative plural, against the
  noun's case only, or instrumental singular, against its case and number. Neither reading agrees.

**98 of the 100 are valid.** Both errors come from the dictionary, not from the rule:
- **#020, also a noun:** «створення чергового блоку → черговим блоку». «черговим» is also the noun черговий,
  the person on duty, so the wrong sentence reads as "the block's creation by the person on duty".
- **#034, two lexemes:** «один з голів → з головах». «голів» there is of гол, a goal, and the builder inflected
  голова's form.

**The scoring.** `multiblimp_eval.py` now reads the four phenomena, and reports the three of agreement together
as "noun phrase". It scores only the phenomena a table holds. Each null at full training is paired by seed on
its own machine.

**The 5M model** (thesis machine; `multiblimp_eval_20261103T062800`, and `…T062944` with the length split):
- **All 3,999 pairs:** BPE 17k 98.2%, endings apart 99.0%, **+0.8 ± 0.1 points (4/4)**.
- **Agreement in the noun phrase:** 98.6% against 99.2%, **+0.6 ± 0.1 (4/4)**: +0.5, +0.9, +0.5, +0.7.
  - case +0.5 ± 0.0 (4/4), gender +1.2 ± 0.2 (4/4), number +0.2 ± 0.2 (2/4).
- **Government:** 97.0% against 98.5%, **+1.5 ± 0.3 (4/4)**: +1.3, +1.4, +2.3, +0.8.
- **The errors:** BPE 17k makes 67–79, endings apart 36–41, **47% fewer**.
- **On the 2,371 pairs of equal token length under both tokenizers:** +0.6 ± 0.1 (4/4).

**The 30M model** (width 512, 151M tokens, 50k; second machine; `multiblimp_eval_20261103T062855`):
- **All pairs:** 99.4% against 99.8%, **+0.4 ± 0.1 (4/4)**.
- **Agreement:** +0.3 ± 0.0 (4/4): +0.3, +0.2, +0.4, +0.3.
  - case +0.2 ± 0.1 (4/4), number +0.4 ± 0.1 (4/4), gender +0.3 ± 0.1 (3/4).
- **Government:** 99.0% against 99.6%, **+0.6 ± 0.1 (4/4)**.
- **The errors:** 21–26 against 7–11, **63% fewer**.

**Both tokenizers are near the ceiling.** Each pair changes one word next to the word it must agree with or
follow, and the new form is wrong by every reading. On MultiBLiMP's subject–verb pairs BPE 17k's nulls reach
71.4% and 82.2%. Here the gain is in the few errors left, and in errors it is larger than MultiBLiMP's: 47% and
63% fewer here, against 17% and 28% fewer there (76.3% against 71.4%; 87.2% against 82.2%).

**The margin is not compared.** The mean log-probability margin is larger under BPE 17k (9.3–9.5 nats against
8.1–8.4). A wrong form costs BPE 17k more tokens than it costs endings apart, where the ending is a token of its
own. Accuracy is compared, as for MultiBLiMP.

**Without the two kinds of error.** `make_minimal_pairs_uk.py --refine` drops them from the built table, by the
dictionary, and writes `uk_np_pp_v2.tsv`:
- **also a noun:** 459 pairs whose new adjective form is a noun too;
- **two lexemes:** 20 pairs whose noun's readings belong to lexemes with different forms in the new case;
- **kept:** 3,520 pairs, among them 89 of the checked hundred, all valid.

On it (`multiblimp_eval_20261103T063313`), the 5M model:
- **All pairs:** +1.0 ± 0.1 (4/4).
- **Agreement:** +0.7 ± 0.1 (4/4).
- **Government:** +1.5 ± 0.3 (4/4). Its errors are the same pairs as in v1.
- **The errors:** 51% fewer.
- **On the 2,105 pairs of equal length:** +0.6 ± 0.1 (4/4).

**Predictions** for the 30M model and the second table were recorded at 06:27 UTC, before either was scored:
- **P344** (~70%): on agreement, the 30M model's endings-apart nulls lead at every seed.
- **P345** (~60%): on government they lead at every seed.
- **P346** (~90%): without the two kinds of error, the 5M model's result on agreement and on government keeps its
  sign and its count of seeds.

**Verdicts:**
- **P341 right:** 98 of 100.
- **P342 right:** +0.5 to +0.9.
- **P343 right:** +0.8 to +2.3.
- **P344 right:** +0.2 to +0.4.
- **P345 right:** +0.4 to +0.8.
- **P346 right:** +0.7 and +1.5, 4/4 each.

The tally is 217 right, 13 half right, 105 wrong, 10 never tested, 1 withdrawn and none pending, of 346.

**The sentence** (§33.147): a gain, so *it covers agreement in the noun phrase and government*.

**What it changes for the thesis.**
1. **The grammatical gain is no longer one phenomenon's.** Inside the noun phrase and after a preposition,
   endings apart makes about half the errors at 5M parameters and about a third at 30M, at every seed.
2. **Near the ceiling the points are small:** +0.6 and +1.5 at 5M. The claim is about errors.
3. **An adjective's number gains nothing at 5M** (2/4), and does at 30M (4/4).
4. **The set is in the repository, with its builder and its check.** It has 3,999 pairs from the test split, 98
   of 100 valid. The refined table without the check's two kinds of error gives the same result.

### 33.151 Item 5's intervals and the software's size: the tokenizer's gains hold at 95%, the features' bound is 0.003

The text's list of 30 October asks for 95% intervals on the headline effects, one-sided upper bounds for the
features' table, the smallest effect four seeds can detect, and the software's size for §3.1. None of it trains,
and every figure below is a function of records already made, so no prediction was recorded.

**The software's size** (`software_size_20261103T065333`, `scripts/ops/software_size.py`, HEAD's tree at
`76bedf4`):
- **125 Python modules:** 41 in the package, 71 scripts and 13 test files, with 27,039 lines. 18,473 of them are
  code: not blank, not a comment, not a docstring.
- **51 shell scripts,** 47 of them chains, with 4,380 lines of code.
- **140 test functions** in the 13 test files.

**The work's size, recounted** (`work_size_20261103T065344`):
- **1,056 trained models:** 582 on the thesis machine and 474 on the second, 160 of them repeats.
- **470.7 GPU-hours** (287.5 and 183.1), and **212.6 billion training tokens** (119.1 and 93.5).
- On 26 October it was 989 models, 425.3 hours and 196.8 billion tokens.

**The intervals** (`bounds_20261103T065743`, `scripts/analysis/bounds.py`). Each row of the text's two tables is
recomputed from the run records, paired by seed on one machine. Each must reproduce its printed mean and standard
error within rounding, and all do. The check found one row's reference: structure added to a full-width embedding
is measured against the added flat vector (§33.19). Against the null's two vectors it would be −0.0017.

**The tokenizer's gain:**
- **Every row's 95% interval excludes zero,** with t-test p from 0.0001 to 0.005.
- **The 5M model at full training:** +0.0165, interval +0.0141 to +0.0189, at least +0.0147 at the one-sided 95%
  bound.
- **At 30M and 85M parameters at full training:** +0.0072 (+0.0064 to +0.0081) and +0.0071 (+0.0057 to +0.0085).
- **Unigram against BPE 17k:** −0.0005, from −0.0017 to +0.0007, p = 0.27.
- **Four seeds of four alone is p = 1/16,** one-sided, by the sign test. The t-test uses the size as well, and is
  stronger.

**The features against their null:**
- **The one-sided 95% upper bounds run from −0.0016 (inflection) to +0.0031** (part of speech, strictly causal).
- **Four rows are above 0.002:** part of speech strictly causal +0.0031, part of speech from the tagger +0.0029,
  shuffled structure +0.0027, case at 30M parameters +0.0027. The frequency band's is +0.0020.
- **So "none beyond 0.002" holds for the means.** At the one-sided 95% bound it is 0.003.
- **Inflection's −0.0046 is below zero at 95%** (p = 0.036): the identity-like features' loss is real.
- **Case at 30M** is positive at every seed, but p = 0.073 and its interval includes zero.

**What four seeds can detect** (80% power at α = 0.05, each row's own spread, (t₀.₉₇₅ + t₀.₈₀) · se):
- the features' rows, 0.0022 to 0.0058;
- the tokenizer's rows, 0.0011 to 0.0122; the headline's is 0.0032.
- A feature's effect below about 0.002–0.006, row by row, can be missed.

**What it changes for the thesis.**
1. **The tokenizer's gains are significant row by row.** The table can carry the intervals.
2. **The features' bound reads 0.003 at 95%.** The means are within 0.002, and four seeds see a feature's effect
   only from about 0.002–0.006. "No feature helps" stays a statement about effects of that size.
3. **§3.1 can describe the software in figures:** 125 Python modules, 18,473 lines of code and 140 tests. The work
   is 1,056 models and 470.7 GPU-hours.

### 33.152 The test split and every vocabulary size: on held-out articles the gain is smaller; a larger vocabulary gains more

Two more of item 5's formalities. The headline nulls at full training are scored on the corpus's test split. That
split holds articles kept out of training and never used to choose anything; the minimal pairs of §33.147 came from
it, and no model saw them. And endings apart is compared with plain BPE at every vocabulary size.

**The test streams.** The tagger reads the test split under each tokenizer, and the null is derived from that file,
as for validation (`ukrainian_pipeline.md`). The derivation reads the train stream, packed on the thesis machine, so
it ran on the second machine (`derive_lexical_codes_20261103T070704`, `…T070850`). The files both machines built are
identical by SHA-256. The two streams hold 410,864 and 453,882 tokens for the same 2,336,304 bytes.

**The scorer.** `evaluate_split.py` now writes a record and pairs runs by seed (`--pair`). On validation it
reproduces two runs' recorded BPB exactly, 0.9419 and 0.9266 (`evaluate_split_20261103T071148`).

**Predictions** (recorded at 07:01 and 07:03 UTC, before any score):
- **P347** (~85%): on the test split the 5M model's gain at full training is positive at every seed and within 0.002
  of its validation value (+0.0165).
- **P348** (~85%): at 30M parameters too, within 0.002 of +0.0072.
- **P349** (~90%): at ~38M tokens and 10k, endings apart's null beats BPE 8k's, of nearly the same characters per
  token (3.13 against 3.21), at every seed.
- **P350** (~85%): BPE 32k's null beats endings apart's at every seed. Its embedding has 1.9M more parameters, and
  the records had it 0.045 ahead of BPE 17k's.

**The 5M model on the test split** (`evaluate_split_20261103T071109`; the thesis machine):
- **BPE 17k:** 0.9560, 0.9615, 0.9581, 0.9586.
- **Endings apart:** 0.9443, 0.9482, 0.9464, 0.9449.
- **The gain: +0.0126 ± 0.0005 (1.3%, 4/4),** +0.0117 to +0.0137; 95% interval +0.0109 to +0.0142. On validation it
  is +0.0165 (1.75%), so the test split gives 0.0039 less.
- **Every model is worse on the test split, endings apart's more:** +0.0144 against validation for BPE 17k, +0.0183
  for endings apart.

**The 30M model on the test split** (`evaluate_split_20261103T070937`; the second machine):
- **The gain: +0.0062 ± 0.0004 (0.83%, 4/4),** +0.0051 to +0.0068; 95% interval +0.0049 to +0.0074. On validation it
  is +0.0072, so 0.0010 less.
- **The shifts are small here:** +0.0018 for BPE 17k, +0.0029 for endings apart.

**Choosing the checkpoint does not explain it.** Every run's best step is its last evaluation, or step 49,500 at the
same BPB. The two held-out samples differ instead:
- BPE 17k takes 7.4% more tokens per byte on the test articles than on validation (0.176 against 0.164);
- endings apart adds 10.5% more tokens there, against 11.4%, so fewer of the words the rule cuts.

**Every vocabulary size** (`paired_cells_20261103T070353`; the thesis machine, ~38M tokens, 10k, the nulls):

| endings apart's null over | gain | seeds |
|---|---|---|
| BPE 4k | +0.1510 ± 0.0006 | 4/4 |
| BPE 8k (3.13 characters per token) | **+0.0829 ± 0.0007** | 4/4 |
| BPE 17k | +0.0174 ± 0.0014 | 4/4 |
| BPE 32k | **−0.0276 ± 0.0009** | 0/4 |
| Unigram 17k (3.07) | +0.0179 ± 0.0012 | 4/4 |

- **At about the same token length** (BPE 8k, Unigram 17k) plain tokenizers lose to endings apart by 0.083 and 0.018.
- **BPE 32k, with longer tokens and 1.9M more parameters, beats it by 0.028.** Endings apart's gain is 39% of what
  going from 17k to 32k gains (0.0450), and it adds no parameters.
- **The vocabularies differ in size, so the embeddings do too.** BPE 8k's has 1.1M fewer parameters than the 17k
  models'.

**Verdicts:**
- **P347 wrong:** positive at every seed, but 0.0039 smaller than on validation.
- **P348 right:** 0.0010 smaller.
- **P349 right:** +0.0829.
- **P350 right:** −0.0276, 0/4.

The tally is 220 right, 13 half right, 106 wrong, 10 never tested, 1 withdrawn and 3 pending, of 353.

**What it changes for the thesis.**
1. **On held-out articles the headline gain is 1.3%, not 1.75%.** It stays positive at every seed, with its interval
   well above zero. At 30M parameters it is 0.83%, not 0.97%.
   - The validation figures stay the table's: every comparison in the text is on validation.
   - The test split is a second sample of articles. The gain depends on the text, which holds fewer of the words the
     rule cuts.
2. **The gain is not the shorter tokens'.** At nearly the same token length, plain BPE 8k loses by 0.083.
3. **A larger vocabulary gains more, with more parameters:** BPE 32k beats endings apart by 0.028. At the same
   vocabulary and parameters, the endings rule is worth 39% of going from 17k to 32k.

### 33.153 Item 8: on gold labels the tagger reads case where context decides

The contextual case feature is uk_core_news_sm's reading of each word in its sentence. Its published case F1 (0.963)
is measured on Ukr-Synth, whose labels are themselves automatic, and over all words. Item 8 asks how it does on gold
labels, where the context decides. The treebank is UD Ukrainian-IU, test and dev (`external/ud_ukrainian_iu/`). The
tagger 3.8.0 was trained on Ukr-Synth (its model card), not on this treebank.

`scripts/audits/tagger_gold.py` (`tagger_gold_20261103T070729`) tags each gold sentence over its gold tokens. For
every word with a gold case it compares three readings:
- **the tagger's case** in context;
- **the form's majority case** in the training corpus: the lexicon of the strictly causal feature (BPE 17k at 0.2),
  looked up by the word's key as the streams compute it;
- **the dictionary's readings** (pymorphy3): a form is case-ambiguous when its readings carry two cases or more.

**Predictions** (recorded at 07:06 UTC, before the run):
- **P351** (~70%): the tagger's case is right for at least 90% of the words that have one.
- **P352** (~65%): on the words whose gold case differs from their form's majority case, it is right for at least 60%.
- **P353** (~85%): where the tagger and the majority case disagree, the tagger is right more often.

**The test split** (898 sentences, 10,240 words with a gold case):

| words | n | tagger | majority case |
|---|---|---|---|
| all | 10,240 | **94.2%** | 73.3% |
| the dictionary gives one case | 2,547 | 98.4% | 93.4% |
| the dictionary gives two or more | 5,825 | 92.3% | 72.7% |
| unknown to the dictionary | 1,868 | 94.7% | 47.7% |
| nouns and pronouns | 5,689 | 93.1% | 77.6% |
| adjectives, determiners, numerals | 2,962 | 94.3% | 65.7% |
| prepositions | 1,588 | 98.2% | 71.8% |
| **context decides** (gold ≠ the form's majority) | 1,890 | **81.1%** | 0% |
| **the tagger departs from the majority** | 1,762 | **87.0%** | 8.3% |

- **The majority case is known for 91.7% of the words.** A word unseen in the training text counts against it.
- **Among the words it knows, the majority case is right for 79.9%.**
- **Of 6,977 words gold gives no case, the tagger gives one to 93 (1.3%).**
- **Dev agrees** (673 sentences, 7,141 words): 94.7% over all, 83.7% where context decides, and 88.8% where the tagger
  departs from the majority (7.9% for the majority case).

**Verdicts:**
- **P351 right:** 94.2%.
- **P352 right:** 81.1%.
- **P353 right:** 87.0% against 8.3%.

The tally is 223 right, 13 half right, 106 wrong, 10 never tested, 1 withdrawn and none pending, of 353.

**The sentence** (the text's, item 8): accurate, so *the context is read by the tagger; the model does not need it*.

**What it changes for the thesis.**
1. **The contextual feature is accurate where it matters.** One word in five has a case other than its form's usual
   one, and the tagger reads 81% of them. Where it departs from the form's usual case it is right 87% of the time,
   against 8% for the form.
2. **So the negative answer is not the tagger's.** The models gain nothing at full training from what the sentence adds
   (§3.10). That is not because the feature is wrong where context decides.
3. **One bound remains:** this is the tagger's reading of the sentence, including words to its right (the leak, G-38).
   The strictly causal feature is the form's majority case, right for 73% of words with a case on gold labels.

### 33.154 Widely used multilingual tokenizers keep a Ukrainian ending apart for a quarter of words or fewer

The thesis's tokenizer keeps every ending its rule finds apart from the stem. Do the tokenizers of widely used
models already do so? `scripts/audits/multilingual_endings.py` (`multilingual_endings_20261103T071706`) measures
four of them on the project's held-out text: the first 3,000 paragraphs of the validation split, words by
uk_core_news_sm, 18,235 words the rule cuts. The measure is `tokenizer_stats`, now able to take another library's
tokenizer. The four (`external/tokenizers/`, each at a recorded commit, its SHA-256 in the record):
- XLM-R's, a SentencePiece Unigram model of 250k pieces;
- mBERT's, WordPiece, 119k;
- Qwen2.5's, byte-level BPE, 152k;
- GPT-4o's `o200k_base`, byte-level BPE, 200k, through tiktoken in an environment of its own.

The project's three tokenizers give the figures recorded on 7 October (§33.62), so the measure is the same.

**Predictions** (recorded at 07:16 UTC, before the run):
- **P354** (~50%): of the four, XLM-R's (a Unigram model) puts a boundary at the rule's cut for the most words.
- **P355** (~70%): each of the four puts one there for fewer than 40% of those words.

| tokenizer | vocabulary | characters per token | tokens per word | a boundary at the rule's cut | the ending a token of its own |
|---|---|---|---|---|---|
| BPE 17k (this work) | 17,000 | 3.59 | 1.73 | 8.8% | 8.7% |
| **endings apart (this work)** | 17,000 | 3.21 | 1.99 | **100.0%** | 99.8% |
| Unigram 17k (this work) | 17,000 | 3.07 | 1.85 | 43.3% | 43.2% |
| XLM-R | 250,002 | 3.44 | 1.89 | 22.8% | 22.7% |
| mBERT | 119,547 | 3.05 | 2.15 | 16.6% | 16.5% |
| GPT-4o | 200,019 | 2.68 | 2.42 | 25.0% | 24.3% |
| Qwen2.5 | 151,665 | 1.90 | 3.50 | 50.7% | 41.9% |

- **XLM-R, mBERT and GPT-4o keep the ending apart for 17–25% of the words the rule cuts.**
- **Qwen2.5 reaches half only by cutting every word finely:** 3.5 tokens per word, 1.9 characters per token, 81.7%
  of words split.
- **At about endings apart's token length** (mBERT 3.05 characters per token, XLM-R 3.44) the boundary falls at the cut
  for a fifth of the words.

**Verdicts:**
- **P354 wrong:** Qwen2.5's 50.7% is the most, by cutting finely; XLM-R's is 22.8%.
- **P355 wrong:** Qwen2.5's is above 40%. The other three are below 26%.

The tally is 223 right, 13 half right, 108 wrong, 10 never tested, 1 withdrawn and none pending, of 355.

**What it changes for the thesis.**
1. **The finding is news for widely used tokenizers.** A large vocabulary does not learn the stem|ending boundary:
   XLM-R, mBERT and GPT-4o keep a Ukrainian ending apart for a quarter of words or fewer.
2. **Where one comes closer, it is by cutting everything:** Qwen2.5 spends 3.5 tokens on a Ukrainian word.
   Endings apart keeps every ending apart at 1.99 tokens per word, with a vocabulary of 17,000.

### 33.155 Item 4: the gain is worth about 15% more parameters, or 17% more training steps

Item 4 asks what the gain is worth in a form a practitioner reads. `scripts/analysis/practical_value.py`
(`practical_value_20261103T072121`) reads two forms from runs already made, each trained to the end of its own
schedule. A step inside a cosine schedule is not a shorter budget, so none is used. Per seed, paired, BPE 17k's BPB
is interpolated linearly in the logarithm of steps or parameters between two of its runs. The thesis machine's runs
only.

**Predictions** (recorded at 07:20 UTC, before the run):
- **P356** (~65%): at width 128 and ~38M tokens, BPE 17k needs between 1.2 and 2 times the training steps to match
  endings apart's null at 10k (its own 10k and 20k runs).
- **P357** (~60%): at 10k it needs between 1.2 and 2 times the parameters (its widths 128 and 256).

**Steps** (width 128, ~38M tokens): endings apart's null at 10k reaches 1.0714, 1.0715, 1.0739, 1.0722. BPE 17k's
null takes 1.0874–1.0911 at 10k and 1.0132–1.0161 at 20k. To match, it needs **1.17 ± 0.01 times the steps**
(1.13–1.19 by seed): about 11,700 of its own schedule instead of 10,000.

**Parameters** (10k, ~38M tokens): BPE 17k's null at width 256, with 11.8M parameters, reaches 0.9826–0.9879. To match
endings apart at width 128 (5.04M) it needs **1.15 ± 0.01 times the parameters** (1.12–1.17): about 5.8M. At 20k,
1.14 ± 0.00 (1.13–1.15).

**And the vocabulary** (§33.152): the endings rule gains 39% of what going from 17k to 32k gains. That step adds
1.9M parameters.

**Verdicts:**
- **P356 wrong:** 1.17, just below 1.2.
- **P357 wrong:** 1.15.

The tally is 223 right, 13 half right, 110 wrong, 10 never tested, 1 withdrawn and none pending, of 357.

**What it changes for the thesis** («Практична цінність»):
1. **Keeping endings apart is worth about 15% more parameters, or 17% more training steps, at no cost.** At 5M
   parameters plain BPE 17k would need about 5.8M parameters, or about 11,700 steps instead of 10,000, to match it.
   The vocabulary and the parameter count stay the same.
2. **Each form interpolates between two runs.** Between widths 128 and 256, and between 10k and 20k steps, so it is
   read near the 5M model; at 30M parameters the gain is about half (§33.142).

### 33.156 Item 3: four fifths of the tokenizer's gain sits in the words the rule cuts

Item 3 asks where the tokenizer's gain sits. The text places case's gain by target, not the tokenizer's. The
bootstrap over articles asks whether the test articles' smaller gain (§33.152) is the text's sample.

`scripts/analysis/tokenizer_by_word.py` (`tokenizer_by_word_20261103T072522`) scores the 5M model's full-training
nulls again, token by token, on validation and on test. Every model's figure equals its record within 10⁻⁵. Each
target is put in its word:
- **a word the endings rule cuts,** a word it does not, or not a word (punctuation, digits, line breaks), as
  `loss_by_kind.py` classifies;
- **a band by the word's count in training,** from each tokenizer's training lexicon.

Both streams of a split hold the same text, so each paragraph's loss is summed in bits and compared directly. The
paragraphs are mapped back to their articles, and the count is checked. The gain is BPE 17k's loss minus endings
apart's, over the text's bytes, paired by seed.

**Predictions** (recorded at 07:24 UTC, before the run):
- **P358** (~70%): more than half of the gain on validation sits in the words the rule cuts.
- **P359** (~60%): a bootstrap over articles puts the validation and test gains within two standard errors of each
  other.

| bits per byte (share of the gain) | validation, 598 articles | test, 645 articles |
|---|---|---|
| all | +0.0166 ± 0.0008 | +0.0126 ± 0.0005 |
| **words the rule cuts** | **+0.0134 ± 0.0003 (81%)** | **+0.0108 ± 0.0003 (86%)** |
| — seen 1–99 times in training | +0.0065 (39%) | +0.0049 (39%) |
| — seen 100–9,999 times | +0.0069 (41%) | +0.0061 (49%) |
| — seen 10,000 times or more | −0.0002 (−1.5%) | −0.0003 (−2.5%) |
| — unseen | +0.0002 | +0.0001 |
| words the rule does not cut | +0.0038 ± 0.0004 (23%) | +0.0022 ± 0.0005 (18%) |
| — seen 100–9,999 times | +0.0031 (19%) | +0.0031 (25%) |
| not words | −0.0005 ± 0.0003 (−3%) | −0.0004 ± 0.0003 (−3%) |

- **On the cut words every seed gains:** +0.0125 to +0.0141 on validation, +0.0101 to +0.0115 on test.
- **The gain sits in the words of middle and low frequency.** The most frequent words, seen 10,000 times or more, give
  nothing, nor do unseen ones.
- **The uncut words gain a fifth.** That is the context around the cut words, or a better vocabulary for the rest.

**The bootstrap over articles** (10,000 draws of the four seeds' mean gain):
- validation +0.0166, standard error 0.0012, 95% from +0.0142 to +0.0188;
- test +0.0126, standard error 0.0011, from +0.0104 to +0.0147;
- validation minus test **+0.0040, standard error 0.0016 (z = 2.5).** Two random samples of articles differ that much
  about once in a hundred, or the bootstrap understates how long articles spread.
- On the cut words the test gives 0.0026 less, and on the uncut 0.0016 less.

**Verdicts:**
- **P358 right:** 81%.
- **P359 wrong:** 2.5 standard errors apart.

The tally is 224 right, 13 half right, 111 wrong, 10 never tested, 1 withdrawn and none pending, of 359.

**The sentence** (the text's, item 3): mostly on cut words, so *the model predicts endings more accurately*. A fifth of
the gain is on the other words.

**What it changes for the thesis.**
1. **Four fifths of the tokenizer's gain sits in the words the rule cuts,** and almost all of it in words of middle and
   low frequency. A frequent word is learned whole under either tokenizer.
2. **The text's sample matters.** Over articles, the headline's 95% interval is +0.0142 to +0.0188 on validation and
   +0.0104 to +0.0147 on test. Seeds alone give +0.0141 to +0.0189.
   - The two held-out samples differ by more than the bootstrap expects (z = 2.5). Both stay well above zero.

### 33.157 The probes at layer 0: the lead is not in the ending's token; the first layer builds it

Under endings apart the word probes read a word's grammar best at layers 1–2, against layer 3 under BPE 17k (§33.84).
The text asks whether the lead is already in the ending's token: layer 0, the input vector at a word's last token,
which under endings apart is the ending itself. The record of 11 October holds every layer, layer 0 included
(`probe_words_20261011T070939`: the nulls at full training, 151M tokens, 50k, four seeds; trained on even validation
windows, tested on odd). It is read here by layer, paired by seed. No prediction was recorded: the figures were in a
record made before.

| endings apart minus BPE 17k, points | layer 0 | layer 1 | layer 2 | layer 3 | layer 4 |
|---|---|---|---|---|---|
| case, all words | +0.3 ± 0.2 (3/4) | **+3.7 ± 0.5 (4/4)** | +3.0 ± 0.2 | +0.3 ± 0.3 | −1.3 ± 0.4 |
| case, where context decides | −1.5 ± 0.3 (0/4) | +3.1 ± 2.1 (3/4) | **+3.4 ± 1.0 (4/4)** | −1.0 ± 0.2 | −4.1 ± 0.3 |
| number | +0.4 ± 0.2 (4/4) | **+2.9 ± 0.1 (4/4)** | +2.7 ± 0.1 | +2.0 ± 0.1 | +1.5 ± 0.2 |
| gender | +0.4 ± 0.1 (4/4) | **+6.0 ± 0.3 (4/4)** | +4.7 ± 0.3 | +4.3 ± 0.1 | +3.3 ± 0.2 |

- **At layer 0 the two tokenizers are level within half a point.** The ending's own vector says no more of a word's
  case, number or gender than BPE 17k's last token. One ending serves many words and many readings.
- **The lead appears at layer 1:** +3.7 points on case, +2.9 on number, +6.0 on gender. The first layer reads the
  ending with its stem.
- **Under BPE 17k the deeper layers catch up on case,** and pass it where context decides (layers 3–4). Number and
  gender keep endings apart's lead to the last layer.

**What it changes for the thesis.**
1. **The lead is built, not given.** It is not in the ending's token but in the first layer, which joins the ending
   to its stem: the model reads an ending it can see apart.
2. **This fits where the gain sits** (§33.156): in the words the rule cuts, as better predictions of their endings.

### 33.158 The text's figures: the law with five conditions and bands, Figure 3.1 in two, and where the gain sits

The text's list of 30 October asks for three figures. `scripts/analysis/thesis_figures.py` draws them in Ukrainian and
in English into `exports/figures/{uk,en}/`. The figures the text already holds keep their names.
- **`fig_law_ukrainian`:** the Ukrainian law with all five conditions, each with a band of the standard error over
  seeds. The five are 151M tokens at 50k and at 5k, and 10k on ~38M, ~7.6M and ~1.5M. Panel б keeps the six
  tokenizers, now with bands. **`fig_law_english`** has its bands too.
- **`fig_tokenizer_budget` and `fig_tokenizer_width`:** Figure 3.1's lines in two figures. The first holds width 128,
  its data and budgets, with Unigram 17k. The second holds widths 256, 512 and 1024 with width 128 at 10k for
  reference, and adds width 1024 at full training (§33.149). `fig_tokenizer` itself is unchanged.
- **`fig_tokenizer_by_word`:** where the 5M model's lead at full training sits (§33.156). It reads the latest
  `tokenizer_by_word` record: each word class and frequency band, validation and test, with the standard error over
  seeds.

Each was looked at in both languages before it went to `exports/`. One fix: the English bars' axis crowded its labels,
and now steps by 0.002.

**What it changes for the thesis:** the figures the text asked for are drawn. Their captions are the text's; what
each needs is in the channel.
