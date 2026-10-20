# Research journey — Part 10: The Learning Law: A Crutch, Not a Teacher

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §33.33–§33.60. [← Part 9](9_ukrainian_and_learning.md) · [index](README.md) · [Part 11 →](11_morphology_in_the_tokenizer.md)

**Established:** a feature's gain, plotted against the null's own bits per byte, traces one curve
whatever the schedule (the learning law), within one model size; across widths the previous word's
tag accounts for about half of the difference (§33.35, §33.38, §33.52, §33.60). Early help is a
crutch, not a teacher: a removed feature leaves a lasting deficit, a late one is integrated slowly,
and as a target it teaches yet stops paying (§33.34, §33.40, §33.42, §33.50, §33.52). The slot's own
effect is a shared learned vector, not a position channel (§33.49–§33.52, §33.58). The nulls, made
constant, restate the Ukrainian gains smaller (G-72, G-73; §33.55–§33.58), and under five tokenizers
how far the null has learned sets case's gain (§33.58).

---

### 33.33 After the demo: English at a short budget, and a switch

The demo of 1 October is over; experiments resume in small blocks.

**What the early help is worth.** Interpolating each null's curve between
evaluations gives the step at which it reaches the feature's level — the
training the feature saves:

| feature | at step 1k | 2k | 5k | 10k |
|---|---|---|---|---|
| English part of speech | 86 steps (8.6%) | 141 (7.0%) | 434 (8.7%) | 790 (7.9%) |
| English inflection | 53 (5.3%) | 78 (3.9%) | 239 (4.8%) | 326 (3.3%) |
| Ukrainian case | 23 (2.3%) | 69 (3.5%) | 181 (3.6%) | 163 (1.6%) |

The early help is worth **2–9% of training steps** — consistent, and modest. *(Ukrainian case,
restated against the constant null: 22, 64, 162 and 118 steps, 1.2–3.2% — 1–9% in all; §33.91.)*
(The first measurement, read off the 500-step evaluation grid, gave identical
ratios for every feature; it could only resolve "within one interval".)

**Block A1** (results_58): English part of speech against its null at a
10k-step budget, seeds 0–1: **+0.0140, +0.0133 — +0.0136 ± 0.0004**. **P65
right.** The gap holds through the whole short schedule (+0.0159 at step 1k,
+0.0136 at the end), against +0.0096 at step 10k of the 50k run and +0.0002 at
its end. The finding of §33.32 holds in both languages: a feature helps a model
that has not yet learned what it encodes. At the same budget English part of
speech helps three times as much as Ukrainian case.

**The switch.** Does early help leave a lasting trace, or only arrive early?
`frozen.feature_off_step = K` closes a gate on every feature's output at step K
— the slot becomes a zero slot — and resets the best-so-far and the patience
counter, so the reported result describes the model *without* the feature and
the jump after the switch cannot end training. The gate exists only when set
(every other model and checkpoint is unchanged — logits bit-identical to the old
code), is saved with the checkpoint, and adds no parameters, so a switched cell
starts from exactly the weights of the cell that keeps its feature. The label
marks it (`-OFF2500`); the cache key does not change. Tested on the real
embedding (gate, gradient, checkpoint, initial weights) and in a real run.

**Block B1** compares, at the 10k budget, English part of speech switched off
from the start (K = 0) and after the first quarter (K = 2,500), seeds 0–1;
A1's runs are the third arm, the feature kept throughout. All three share
their initial weights.

- **P66** (~65%). Switched off at step 2,500, part of speech keeps less than
  half its gain: at the end, the K = 2,500 model sits closer to K = 0 than to
  the model that kept the feature.
- **P67** (~65%). The switch costs at once: at step 3,000 the K = 2,500 model
  trails the K = 0 model.

### 33.34 A crutch, not a teacher

Block B1 (results_60; the upload was named `results_59.zip`) switched English
part of speech off at the 10k budget, seeds 0–1. With A1's runs as the third
arm, all three share their initial weights:

| at the end of 10k steps | seed 0 | seed 1 |
|---|---|---|
| feature kept throughout | 1.5122 | 1.5121 |
| switched off at step 2,500 | 1.5256 | 1.5246 |
| off from the start | 1.5234 | 1.5238 |

Switched off at 2,500, part of speech keeps **−19% and −7%** of its gain: the
model ends *below* the one that never had the feature (−0.0015 on average).
**P66 right**, beyond its own terms. Step by step, against off-from-start:

| step | 1k | 2k | 2,500 | 3k | 5k | 7,500 | 10k |
|---|---|---|---|---|---|---|---|
| switched at 2,500 | +0.0160 | +0.0157 | **−0.0294** | −0.0092 | −0.0046 | −0.0021 | −0.0015 |
| kept throughout | +0.0160 | +0.0157 | +0.0151 | +0.0134 | +0.0116 | +0.0116 | +0.0114 |

**P67 right**: at the switch the model falls from 0.016 ahead to 0.029 behind
a model that never had the feature — it had routed information through the
slot, and the route is suddenly empty — and seven thousand steps of recovery
never quite close the gap. Before the switch the switched and kept runs follow
the same trajectory, as identical weights and an open gate must.

The early help is **a crutch, not a teacher**: the model does not learn better
representations from the feature, it learns to use it; taken away, the reliance
costs more than the feature gave. Matched within its own family, keeping part of
speech is worth +0.0114 at this budget (+0.0136 against the null, a different
family).

Two follow-ups, one block (C1), seeds 0–1, at the same budget:

- **P68** (~70%). Switched off later, at step 7,500 — more reliance, less time
  to recover — part of speech ends at least 0.003 below off-from-start.
- **P69** (~70%). Switched **on** at step 5,000 (`frozen.feature_on_step`, the
  reverse gate: closed until K, then open), it ends with at least half of the
  kept-throughout gain over off-from-start (≥ +0.0057): the feature is
  information the model can pick up mid-training.

### 33.35 A programme after the demo

With the thesis secure and months to the defence, the question became where
investment could give a strong result. What the project has ruled out is
narrower than "morphology in embeddings does not work": segmentation does not
beat BPE overall (Q1), and linguistic features *added to the input* of a small
model add nothing *at the end*, once it has learned what they encode — and act
as a crutch before. Other routes stay open: morphology as a training target, as
a scaffold removed gradually, in the rare and novel tail that averages hide,
and the question of *when* the model learns what a feature encodes.

**The observation that set the programme.** Plotted against the null's own
validation BPB — how far the model has got — a feature's gain from every run
traces nearly one curve (`scripts/analysis/learning_law.py`): an inverted U,
peaking early and falling as the model learns. English part of speech collapses
across schedules (+0.0134 and +0.0139 at null BPB 1.55 under 50k and 10k
schedules). Ukrainian case collapses approximately, with a residual that grows
with schedule length (+0.0035 at 50k to +0.0056 for a 38M corpus at 10k, at
1.10). The little-data gains are explained: a 1.5M-token model plateaus near
the curve's peak. Each feature has its own zero crossing — structure first
(1.484), then frequency band (1.412), inflection (1.404), part of speech from
the word form (1.361) — while contextual part of speech never reaches it at this
scale. This is exploratory, not pre-registered; it sets direction D1 and its
first registered test.

The programme ([`docs/DIRECTIONS.md`](../../DIRECTIONS.md)) has six directions — the
law (D1), its mechanism by probing (D2), scaffolding (D3), the teacher (D4), the
tail (D5), the position channel (D6) — each with a day test, night-ready work and
a pivot criterion; it runs on two queues: decision-independent work at night, anything that could change
the plan by day.

### 33.36 The first programme night

The night of 2 October runs the night-ready queue
(`scripts/chains/night_2026-10-02.sh`), in priority order: English part of
speech at a 10k budget at seeds 2–3 with its null; the switch at seeds 2–3; D1's
sampling — English part of speech at 5k and 20k budgets, seeds 0–3; Ukrainian case
at a 20k budget, seeds 2–3. The chain waits while any training run is still
going — a process whose command line *starts* `python -m morphlm.cli run`; a
bare substring would also have matched a shell that merely mentions it, which
the rehearsal caught — and the guard refuses a second copy. Rehearsed with the
stub: 28 runs, each consolidation cell extending exactly the seeds on record.

- **P70** (~90%). At seeds 2 and 3, English part of speech at a 10k budget again
  beats its null by at least +0.003 at each seed.
- **P71** (~70%). The crutch replicates: switched off at 2,500, part of speech
  ends at or below off-from-start (gain over it ≤ +0.0005) at seeds 2 and 3.
- **P72** (~70%). English part of speech at 5k and 20k budgets lands on the
  gain-against-learning curve: wherever the curves overlap, within ±0.002 of
  the 50k run's gain at the same null BPB.
- **P73** (~75%). Ukrainian case at a 20k budget ends between its 10k and 50k
  results (+0.0044 and −0.0009) at seeds 2 and 3.

### 33.37 The follow-on of 2 October

At 09:07 the night chain had started 17 of 28 runs, on time for about 13:40;
with the results read near 19:00, a follow-on fills the rest
(`scripts/chains/day_2026-10-02.sh`, ~4.8 h; it waits for the night chain). The
English feature cells at a 10k budget reuse the 10k nulls now run at seeds 0–3.
Structure is the law's negative control: no early help, so no gain predicted at
a short budget either.

- **P74** (~80%). At seeds 2–3 each of C1's cells — off at 7,500, on at 5,000 —
  has the same sign of gain over off-from-start as at seeds 0–1.
- **P75** (~75%). English inflection at a 10k budget beats its null by at least
  +0.003 on average, positive at every seed — though it ends below its null at
  50k steps.
- **P76** (~70%). Ukrainian case at a 5k budget beats its null by more than its
  10k-budget gain (+0.0044) on average.
- **P77** (~75%). English structure at a 10k budget stays within ±0.002 of its
  null at every seed.
- **P78** (~70%). English frequency band and part of speech from the word form
  each beat the null by at least +0.002 on average at a 10k budget.

### 33.38 The law holds; a critical period

Both chains of 2 October completed (results_62; C1's seeds 0–1 in results_61).

**The learning-progress law, in English, holds almost exactly.** English part
of speech traces the same gain-against-learning curve under 5k, 10k, 20k and
50k schedules: the worst deviation from the 50k curve is 0.0010 over 67
compared points, the average 0.0002–0.0005. **P72 right.** Feature by feature,
50k and 10k schedules agree within 0.0004 at null BPB 1.55 (inflection +0.0069
and +0.0073; part of speech from the word form +0.0101 and +0.0105; frequency band
+0.0052 and +0.0050; structure +0.0011 and +0.0007). The curve predicts the
short-budget gains: at a 10k budget, part of speech **+0.0130 ± 0.0004**, part
of speech from the word form **+0.0099 ± 0.0005**, inflection **+0.0067 ±
0.0014** (4/4 — though it ends 0.0046 below its null at 50k), frequency band
**+0.0046 ± 0.0008**, structure +0.0008 ± 0.0008 — the order of their curves.
**P70, P75, P78 right**; **P77 wrong** by one seed (structure +0.0028 at seed
2; its mean is at the null). In Ukrainian, case at 5k, 10k and 20k budgets and
on a 38M corpus cluster on one curve (at null BPB 1.10: +0.0051, +0.0049,
+0.0056); the 50k run sits 0.0015 below them, the one outlier. *(Against the first null.
Against the constant null it is 0.0008 below the ~38M run at 1.10 and level at 1.15; §33.91.)* **P73, P76
right** (20k: +0.0015 ± 0.0004; 5k: +0.0074 ± 0.0007).

**The switch family, four seeds**, English part of speech at a 10k budget,
against off-from-start:

| | mean | share of the kept gain |
|---|---|---|
| kept throughout | +0.0111 ± 0.0003 | 100% |
| off at 2,500 | −0.0023 ± 0.0006 | −7% to −33% |
| off at 7,500 | −0.0118 ± 0.0003 | −94% to −120% |
| on at 5,000 | +0.0010 ± 0.0004 | 0% to 17% |

**P68, P71, P74 right; P69 wrong.** Reliance grows with exposure: switched off
at 7,500 the model falls from 0.011 ahead to 0.041 behind, and ends 0.012 below
a model that never had the feature. And a feature switched *on* halfway is
barely used — +0.0010 against +0.0111 from the start — after its own jump at the
switch (−0.032). Together: **a critical period** — a feature is integrated if it
is present while the representations form; arriving late, it is largely ignored;
removed late, it costs most. One confound is open (G-68): a closed gate passes no
gradient, so a feature switched on meets an embedding table still at its random
initial values, and a cosine schedule has halved the learning rate by step 5,000.

### 33.39 Day block D: the confound, and a wider model

Two day tests. **G-68**: English part of speech switched on at step 5,000 of a
**20k** budget — the learning rate still near its peak, 15,000 steps left —
with off-from-start for comparison; the feature kept throughout at 20k is
already run. **The width pilot** (D1): null and part of speech at a 10k budget
in a model four times wider *(twice as wide, with four times the core's parameters; §33.91)*
(d_model 256; the feature keeps its 8 dims), seed 0.

Preparing the pilot found that a run's label ignored the model's size: a
256-wide run would have been labelled like a base run, and every analysis
groups by label — it would have entered the base cell as a fifth seed. The
rule the code states for training settings (non-default values mark the label)
now covers the model: `_w256`, `_L6`, `_h8`; defaults add nothing. Rebuilt from
their saved configurations, all 392 recorded runs keep their labels (two old Q1
runs differ under the old code and the new alike: they predate the surface
settings).

- **P79** (~60%). Switched on at 5,000 of a 20k budget, part of speech ends
  with less than half of the kept-throughout gain over off-from-start: the
  critical period survives time and learning rate.
- **P80** (~60%). The 4× wider model's part-of-speech gain at a 10k budget lies
  within ±0.002 of the base 50k curve's gain at the same null BPB: the law holds
  across capacity.

### 33.40 Two pivots

Day block D (results_63, results_64), seed 0. Both predictions wrong; both
redirect the programme as the pivot criteria said they would.

**G-68: slow integration, not a critical period.** Switched on at step 5,000
of a **20k** budget, English part of speech ends +0.0039 over off-from-start —
**63%** of the kept gain (+0.0061), against 9% at a 10k budget. Its share
climbs steadily: 14% by step 8k, 40% by 12k, 54% by 15k, 63% at the end. A
late feature is integrated slowly — its table starts untrained, the model
rebuilds around it — not ignored. **P79 wrong.** The removal side of §33.38
stands; "a late feature is largely ignored" was the short schedule.

**The width pilot: the law does not cross model size.** A model four times
wider (d_model 256; 19–20 minutes per 10k run, 1.6× the base) starts on the
base curve (+0.0136 against +0.0140 at null BPB 1.94) — then its gain falls
away early: +0.0031 at null BPB 1.58, where the base model gains +0.0133; it
stays near +0.002, and the base curve comes down to meet it (+0.0022 against
+0.0039 at 1.384). Worst deviation −0.0111, mean 0.0046. **P80 wrong** — the
D1 pivot. At equal overall BPB the wider model has already learned what part of
speech encodes; overall BPB measures "how much the model has learned" across
schedules and corpus sizes, not across capacity. If "learned it" is the model's
own knowledge of the feature, a probe's accuracy should be the axis on which
widths collapse — D2 moves to the centre.

The night of 2–3 October (`scripts/chains/night_2026-10-03.sh`, ~6.8 h) maps
both: G-68 at seeds 1–3; the 256-wide model at seeds 1–3; a 64-wide model at
seeds 0–3 — if capacity shifts the curve, a narrower model should shift it the
other way; and the 256-wide model at a 20k budget, seed 0.

- **P81** (~70%). At each of seeds 1–3, part of speech switched on at 5,000 of
  a 20k budget ends with more than a third of the kept gain over off-from-start.
- **P82** (~75%). The wider model's early fall replicates: at null BPB 1.55 its
  gain is below +0.006 at each of seeds 1–3 (the base curve: +0.013).
- **P83** (~60%). The 64-wide model's gain at the end of a 10k budget is at least
  the base curve's gain at the same null BPB: a narrower model keeps its gain
  longer.

### 33.41 The chain widened: four directions in one night

Before it started, the night chain of §33.40 grew to ~11 hours, so that one
night advances four directions.

**D6 without new code.** During the 50k runs the architecture effect — the
null over the plain baseline — is positive from the start and grows: +0.0056 at
step 2k, +0.0060 at 10k, +0.0095 at 20k, +0.0112 at the end (4/4 seeds at every
point). *(+0.0112 is the mean at the last evaluation, +0.01116; each run taken to its best step, it is +0.0111 ± 0.0025, as §33.11 and the results record have it — seeds 2–3's baselines are best at step 49,500; annotated 19 October 2026.)* But a run *budgeted* for 10k steps, its learning rate annealed by step
10k, shows **+0.0003** (seed 0: the 128-wide baseline from the September
learning-rate sweep, whose training settings match today's field for field; a
plain baseline exercises none of the code changed since, so it serves as seed
0). If that holds, the free dims help optimization at a high learning rate rather
than representation — in line with the slotted runs' fourfold stability across
seeds. Tonight: the plain baseline at a 10k budget (seeds 1–3) and at 20k (seeds
0–3), and a 256-wide baseline at 20k (seed 0) beside the 256-wide null.

**D3, the mirror of G-68.** Part of speech switched off at 5,000 of a 20k
budget (seeds 0–3): given 15,000 steps, can the model shed its reliance?

- **P84** (~70%). At a 10k budget the architecture effect is small: the null
  beats the plain baseline by less than +0.003 on average over seeds 0–3
  (+0.0060 at step 10k of the 50k runs).
- **P85** (~60%). At a 20k budget it lies between: above the 10k-budget value
  and below the 50k value (+0.0112), on average over seeds 0–3.
- **P86** (~60%). Switched off at 5,000 of a 20k budget, part of speech ends
  within ±0.0015 of off-from-start on average over seeds 0–3: given time, the
  model sheds its reliance.

### 33.42 The night of four directions

The chain of §33.41 completed (results_65, 34 runs).

**D3 — adding is reversible in time, removing is not.** Switched on at step
5,000 of a 20k budget, part of speech recovers 63%, 54%, 82% and 104% of the
kept gain at seeds 0–3 (+0.0049 against +0.0067): slow integration, four seeds.
**P81 right; G-68 closed** — the 9% at a 10k budget was the schedule. Switched
*off* at 5,000 of a 20k budget, with 15,000 steps to recover, the model still
ends **0.0042 below** off-from-start, at every seed (−0.0044, −0.0012, −0.0047,
−0.0064) — more than the 0.0023 left by a switch at 2,500 of a 10k budget.
**P86 wrong.** A late feature can be learned; a removed one leaves a lasting
deficit. The likely mechanism: leaning on the feature while learning is fastest
displaces what the model would have learned by itself, and that is hard to make
up later.

**D1 — no single curve across widths.** At null BPB 1.55 the 256-wide model
gains +0.0023, +0.0082, +0.0009, +0.0029 (mean +0.0036; the base curve
+0.0134). **P82 wrong** by one seed; the fall replicates on average. It is
abrupt: the 256-wide curve follows the base curve to null BPB ~1.70, collapses
by 1.60, and crosses zero near 1.37 (20k budget, seed 0: −0.0022 at the end).
The 64-wide model gains *less* than the base model at every common null BPB
(+0.0112 against +0.0135 at 1.70), ending 0.0028 below the base curve at its own
null BPB. **P83 wrong.** The slot is a fixed 8 dims — 12.5% of a 64-wide
embedding, 3% of a 256-wide one — so across widths the feature's share of
capacity differs; within a width the same-shape null keeps each comparison fair.

**D6 — the architecture effect builds with training, and survives width.** The
null over the plain baseline: **+0.0044, +0.0043, +0.0050** at a 10k budget
(seeds 1–3), **+0.0078 ± 0.0009** at 20k (4/4), +0.0112 at 50k; in the 256-wide
model **+0.0057** at 20k (seed 0). The September seed 0 at a 10k budget
(+0.0003) is the outlier on which §33.41's high-learning-rate hypothesis rested;
whether that run differs from current code is G-69. **P84 wrong** (+0.0035 with
seed 0 counted); **P85 right**.

### 33.43 A bit-for-bit reproduction

**G-69 closed.** The 10k-budget plain baseline at seed 0, rerun under current
code (results_66, run 12e3d62efc84), matches the September learning-rate
sweep's run (cfbb4ce93538) to the last digit of float precision — validation
BPB 1.5265532254632217 — at all 21 evaluations, in validation BPB and training
loss alike. A run made weeks of code changes earlier reproduces exactly: the
pipeline is deterministic along this path, and the September run was never a
different code era. Its small architecture effect is seed variation; over four
seeds the effect at a 10k budget is **+0.0035 ± 0.0011** (4/4). That is below
the 50k run's value at step 10k (+0.0060 ± 0.0007) but far from zero: the
high-learning-rate hypothesis of §33.41 falls.

Two runs now held one cell's seed 0. A quarantine rule (`sweep_duplicate`)
moves the September copy aside; dry-run against 234 runs it matches that one
run and nothing else.

**The night of 3–4 October** (`scripts/chains/night_2026-10-04.sh`, ~12 h): the
256-wide model at a 20k budget, seeds 1–3 — null, part of speech, plain
baseline; then English inflection, part of speech from the word form,
frequency band and structure at a 20k budget, seeds 0–3.

- **P87** (~75%). The 256-wide model's part of speech ends a 20k budget at or
  below its null (gain ≤ +0.0005) at each of seeds 1–3.
- **P88** (~75%). The architecture effect at 256 wide and a 20k budget is
  positive at each of seeds 1–3.
- **P89** (~70%). At null BPB 1.55 each of the four features' 20k-budget gain
  lies within ±0.001 of its 50k value.

### 33.44 A rule written for one generation of runs

The dry run of the quarantine (before `--apply`) listed, beside the
September duplicate, **nine valid runs** under an old rule, `causal_partial`:
every plain baseline run since 2 October — today's G-69 rerun among them. The
rule flags runs with `causal_identity` set that lack the current causal-shift
marker. The chains pass the causal flags to every run, plain baselines
included; a baseline has no feature block, so it cannot carry the marker — nor
have the defect, having nothing to shift. Written when only feature runs set the
flag, the rule misfired on a later generation of runs. Fix: a run is in scope
only if it has features; a regression test pins it. Over all 340 runs on hand,
the complete dry run now lists exactly the September duplicate.

The rehearsal had included the G-69 rerun, so it must have flagged it too — but
it was read only around the new rule. The procedure now includes what the near miss
taught: a dry run is read in
full, every rule and every run, and a new rule is checked together with the old
ones. Methodological finding 6 — analysis code needs the scrutiny the models get
— extends to the tools that decide which runs count.

### 33.45 Tests the suite never ran

Checking why a new test's line was missing from a suite's output found that
each test file ran a hand-kept list of its tests — and four tests had never been
added to one: G-63's regression test for the fragment-aware word rule (since 1
October), the feature switch's (2 October), and today's two, the model size in
the label and the quarantine sparing plain baselines. Each passed when called
directly; "PASS" at the end of a file was true of the listed tests only, so none
of the four guarded anything. Each file now runs every test function it defines,
in order — 63 tests, all run, all pass, the four among them. The same lesson as
§33.44, one level up: the tools that certify the code need the scrutiny the code
gets.

### 33.46 The wide model's crossing, and the slot as consistency

The night chain of §33.43 ran shortened to 8.5 hours (results_67, 17 runs):
`FEAT_SEEDS="0 1"` kept the features at a 20k budget to seeds 0–1, so that it
ended within the nine hours available; its log records the seeds it ran.

**P87 right.** At a 20k budget the 256-wide model's part of speech ends at or
below its null at every seed: −0.0022, −0.0002, −0.0008, −0.0002 (mean −0.0008 ±
0.0005), where the 128-wide model still gains +0.0075. The wider model has
passed its zero crossing within 20k steps.

**P88 wrong — and the reason is the finding.** The architecture effect at 256
wide is +0.0057, +0.0200, −0.0033, +0.0028 (mean +0.0063 ± 0.0049). The variance
is all in the plain baseline: its final BPB spans 1.2740–1.2977 across seeds
(spread 0.0237), the null's 1.2751–1.2777 (0.0026), part of speech's 0.0007. No
run spiked — every curve falls smoothly; the baseline's seed 1 is worse from the
start, its seed 2 better. At this width the slot's clearest effect is
**consistency**: without it, where training ends depends on the seed; with it,
every seed lands in the same place — the "4× more stable" of 128-wide 50k runs,
now 9×.

**P89 wrong by one feature.** At null BPB 1.55 and a 20k budget, part of speech
from the word form, frequency band and structure match their 50k values within
0.0007 (+0.0001, +0.0002, −0.0007); inflection sits +0.0020 above (+0.0088
against +0.0069, two seeds — the noisiest feature, ±0.0014 at 10k). Seed noise
or a schedule effect: G-70.

The night of 4–5 October (`scripts/chains/night_2026-10-05.sh`, ~8.1 h): the four
features at a 20k budget, seeds 2–3; the 256-wide baseline and null at a 20k
budget, seeds 4–7.

- **P90** (~65%). Over four seeds, inflection's 20k-budget gain at null BPB
  1.55 lies within ±0.001 of its 50k value (+0.0069): the two-seed deviation was
  noise.
- **P91** (~70%). Over seeds 0–7, the 256-wide plain baseline's spread of final
  BPB is at least four times the null's.
- **P92** (~60%). Over seeds 0–7, the 256-wide null beats the plain baseline on
  average, at six or more of the eight seeds.

### 33.47 A screen before the long chain

The night chain of §33.46 did not run. A better rhythm replaced it:
a short screen first — one sharp test per open direction — then a long chain
shaped by what it shows. Its G-70 runs move into the screen; P91 and P92 (the
wide model at seeds 4–7) into the long chain after it.

**D5, screened at the desk.** Every result records loss by token-frequency
bucket. At the end of 50k runs no benefit hides in the rare tail: English part
of speech keeps its gain only in the two most frequent buckets (+0.0112 and
+0.0159 nats, 4/4 seeds), noise in the rarest; English inflection is *worse*
than its null in rare and mid-frequency tokens (−0.019 to −0.036 nats, 0/4
positive) — where its end deficit lives; Ukrainian case is neutral to negative
across the tail. By BPE-token frequency, not word rarity; with this prior the
unseen-word version is parked.

**Three new modes**, each inert unless set (logits of existing models
bit-identical; labels unchanged), each with a test that runs in the suite and a
real run on the toy corpus:

- `model.position = rope` (D6): rotary positions in attention, none added at the
  input; label `_rope`.
- `frozen.feature_dropout_until = K` (D3): weaning — each token keeps its
  features with a probability falling linearly to 0 at step K; evaluation scales
  by it; best-tracking resets at K; label `-DROP<K>`.
- `frozen.role = target` (D4): the fact as an auxiliary target, predicted from the
  final hidden state with weight `aux_weight` (0.1); nothing injected; its head
  made last, so every language-model weight starts exactly as a baseline's; its
  loss added only to the training step, never to evaluation; label `-TGT0.1`, the
  shuffled-label null `-SHUF-…-TGT0.1`.

The RoPE run exposed the provenance record assuming a learned position table;
it now counts positional parameters as zero when absent and the auxiliary head
apart from the transformer core.

**The screen** (`scripts/chains/screen_2026-10-05.sh`, ~2.7 h, seed 0, 20k budget):

- **P95** (~35%). Part of speech as an auxiliary target ends at least +0.002
  better than both the shuffled target and the plain baseline: a lasting teacher.
- **P93** (~65%). Under rotary positions the architecture effect is below +0.002
  (+0.0078 ± 0.0009 with learned positions): it needs position in the slot.
- **P94** (~55%). Weaned gradually to zero by step 5,000, part of speech ends less
  than 0.0021 below off-from-start — half the abrupt switch's deficit at 5,000.

### 33.48 Building D1 and D2 while the screen runs

**D1 needs no new model code.** Both part of speech and the null occupy 8 dims
(part of speech's 19 classes learned into 8), and the existing tiling lever,
`frozen.amplify`, repeats a block k times: a 256-wide model with its slot tiled
×2 gives the feature 16 of 256 dims — the base model's share. Tiling repeats
values, so it raises the feature's share and prominence, not its capacity. But
`amplify` never reached the label — a third hazard like the model size (§33.39):
a tiled run would have joined its untiled cell as more seeds. All 356 recorded
runs used `amplify=1`, so no collision ever happened; the label now carries
`-x<k>`.

**D2: checkpoints and a probe.** `MORPHLM_CHECKPOINT_EVERY=N` — an environment
switch, not a config field, so a checkpointed run keeps its label — saves
`ckpt/step_NNNNNN.pt` beside `model.pt`; a toy run with and without it is
bit-identical. `scripts/analysis/probe.py` rebuilds each checkpoint, feeds the
model its own features, and trains a linear probe from every layer's hidden
states to the probed feature's labels (alternating windows train and test,
sentinel tokens left out, the two datasets' tokens asserted identical). Layer 0
is the input — what token identity alone says. Its record goes to
`results/audits/` with a UTC timestamp in the filename, the form by which the
packer collects audits.

**A fourth hazard, found preparing the pilot.** Run ids hash the whole
configuration; every field added since a cell ran — the switches, today's three
modes — gives a re-run of that cell a *new* id under the *same* label and seed.
G-69's September duplicate arose this way. The pilot's checkpointed nulls
therefore take a new seed (4), not a re-run of seed 0; and the packer now lists
**DUPLICATE SEEDS** — any packed run sharing its label and seed with another
run on disk — tested on a planted duplicate. `scripts/ops/run_ids.py` gives the
id a command line would produce, checked against a real run.

**Day block E** (`scripts/chains/day_2026-10-05.sh`, ~80 min; it waits for the
screen chain itself, since between a chain's runs there are seconds with no
training): the 128- and 256-wide nulls at a 10k budget, seed 4, with checkpoints
every 500 steps, then the part-of-speech probe on both; and the 256-wide model
with its slot tiled ×2, null and part of speech.

- **P96** (~60%). At null BPB ~1.65 — where the 256-wide gain collapses — the
  256-wide null's deepest-layer part-of-speech accuracy is at least 2 points
  above the 128-wide null's at the same null BPB: the wider model knows part of
  speech earlier.
- **P97** (~55%). With its slot doubled to the base model's share, the 256-wide
  model's part-of-speech gain at null BPB 1.55 stays below +0.006: the width
  effect is not the slot's share.

### 33.49 The screen's verdicts, and time budgets

The screen and block E completed (results_68: 11 runs, two probe records).

| direction | result (seed 0 unless noted) | prediction |
|---|---|---|
| G-70 | inflection at 20k over four seeds, at null BPB 1.55: +0.0070 against +0.0069 at 50k | **P90 right** — G-70 closed |
| D1 | slot tiled ×2 (the base model's share): +0.0024 at null BPB 1.55 (×1: +0.0036) | **P97 right** — the collapse is not the slot's share |
| D2 | at null BPB ~1.65, deepest-layer part-of-speech accuracy 0.867 (256 wide) against 0.854 (128) | **P96 wrong** — 1.3 points, not 2 |
| D6 | RoPE: null over plain baseline +0.0073 (learned positions: +0.0078 ± 0.0009) | **P93 wrong** — the effect survives |
| D4 | the real target over the matched shuffled one +0.0018, over the plain baseline +0.0014 | **P95 wrong** — under +0.002, but positive |
| D3 | weaned to 5,000: −0.0023 against off-from-start (the abrupt switch at 5,000: −0.0044) | **P94 wrong** by 0.0002 — half the deficit |

**D6 changes its question.** Under rotary positions nothing positional enters
the input, and eight content-free dims help as much as before: the effect is not
a clean position channel. What such dims do — a bias- or sink-like channel — is
the new question. RoPE itself is far better than learned positions here (1.3908
against 1.4325 BPB for the plain baseline at 20k).

**D2 locates the width effect in context.** At equal null BPB the two widths
read part of speech from their *input* equally well (0.838 and 0.839); the wider
model's deeper layers add twice as much (+0.029 against +0.015). It learns
contextual part of speech earlier — the knowledge the feature would have supplied.
Probe accuracy narrows the two widths' gap but, on this one comparison, does not
close it: at equal accuracy (~0.867) the 128-wide model still gains +0.0134, the
256-wide +0.0091.

**Time budgets replace day and night.** The hours available are not fixed, and
several long chains may follow one another; a chain is now planned from one
number, the hours until its results are read. Short blocks take the boldest,
plan-changing tests; long chains the quiet, mutually independent work: two
queues, by time budget. "Night-ready" becomes
long-chain-ready, with the same two conditions.

**The long chain of 5 October** (`scripts/chains/long_2026-10-05.sh`, ~7.5 h, 19
runs) gives the screen's promising tests seeds 1–3, extends the probe to 20k
budgets, and adds D1's second seed.

- **P98** (~55%). Over seeds 0–3 the real auxiliary target beats the matched
  shuffled one at a 20k budget by at least +0.001 on average, at three or more
  seeds.
- **P99** (~75%). Over seeds 0–3 the architecture effect under RoPE is positive
  at every seed, with a mean of at least +0.004.
- **P100** (~65%). Over seeds 0–3 weaning to 5,000 leaves at most 70% of the
  abrupt switch's deficit.
- **P101** (~60%). Probed along 20k budgets, the 256-wide null's deepest-layer
  part-of-speech accuracy exceeds the 128-wide null's by at least 1 point at
  every matched null BPB between 1.75 and 1.45.
- **P102** (~65%). With its slot tiled ×2, the 256-wide model's gain at null BPB
  1.55 stays below +0.006 at seed 1 as well.

### 33.50 The teacher fades faster than the crutch

The long chain of §33.49 completed (results_69, 19 runs, two probe records).

| | seeds 0–3, 20k budget unless noted | prediction |
|---|---|---|
| D4 | real target over its shuffled twin: +0.0018, +0.0015, −0.0011, −0.0008 — **+0.0003 ± 0.0007** | **P98 wrong** |
| D6 | under RoPE, null over plain baseline: **+0.0057 ± 0.0006**, 4/4 (learned positions: +0.0078 ± 0.0009) | **P99 right** |
| D3 | weaned to 5,000: −0.0029 ± 0.0016 against the abrupt −0.0042 ± 0.0011 — **70%** of the deficit | **P100 right** |
| D2 | probe along 20k budgets: the 256-wide null leads the 128-wide by +0.8 to +1.3 points at every matched null BPB, 1.75–1.45; input layers equal | **P101 wrong** — below 1 point at the ends |
| D1 | tiled ×2, seed 1: +0.0037 at null BPB 1.55 (seed 0: +0.0024) | **P102 right** |

**D4 answered — the law covers targets.** Along the 20k budget the real target
beats the shuffled one by +0.0094 at step 1k, +0.0087 at 2k, +0.0041 at 5k,
+0.0020 at 10k, +0.0006 at 15k, +0.0003 at the end — early help that fades, like
the feature as an input (+0.0150 at 1k, still +0.0075 at the end), only faster.
The contrast fits §33.34: as a target the feature pushes the model to learn part
of speech itself, and once it has, nothing is left to add; as an input it is
leaned on, so its gap lasts longer. Any auxiliary head helps a little — the
shuffled target too (+0.0006 ± 0.0004 over the plain baseline) — the morphology
in it adds nothing at the end. Seed 0's +0.0018 (§33.49) was within this spread.

**D6:** the effect survives rotary positions at four seeds, a little smaller.
RoPE itself beats learned positions by +0.0443 ± 0.0011 BPB (plain baseline,
4/4). **D3:** weaning's reduction is modest and varies by seed (one seed ends
above off-from-start, one equals the abrupt switch). **D2:** the wider model's
contextual lead holds at every matched null BPB while the input layers agree —
the width effect lives in what context adds. **D1:** the slot's share is ruled
out at two seeds.

### 33.51 An empty slot under rotary positions, and resumable chains

One run as a small piece (6 October): the zero slot under rotary positions,
seed 0 (run 16a11dd9bf91) — 8 input dims held at exactly zero, and nothing
positional anywhere in the input. **1.3910**, against the plain baseline's 1.3908
and the constant slot's 1.3835. **P103 wrong** (stated before the run and recorded
after it, ~55%: at least half the constant slot's benefit). The zero and constant
slots share their initial weights (§33.15), so the matched contrast is clean:
the constant beats the empty slot by **+0.0075**.

Read with the record, the slot needs a **token-independent signal** in its dims.
With learned positions the "empty" slot was never empty — position filled it, and
zero, flat and constant slots gained alike (§33.12); one learned vector *added* to
every input dim beat its initialization-matched control by +0.0085 ± 0.0022 (4/4;
§33.14–§33.15). Under rotary positions a plain model's input holds token identity
and nothing else: an empty slot then gives nothing, a constant one +0.0075. Every
linear layer here has a bias and every LayerNorm a shift, so this is not a missing
bias in the usual sense: the constant enters *before* normalization and changes
what LayerNorm does — one hypothesis is that it softens it, letting each block see
the scale of the token's embedding. At the desk, by token frequency, the RoPE
null's gain is even across the vocabulary (+0.015 to +0.020 nats in every bucket,
4/4), where with learned positions it concentrated in rare tokens (+0.061 in the
rarest bucket, +0.0005 in the most frequent).

**Resumable chains.** The hours available vary, and a long chain may have to stop.
`scripts/ops/run_ids.py --done` tells a chain whether a run's cell and seed already
have a finished result — the identity every analysis groups by, which also finds
an equivalent run recorded under an older id; a half-written result does not
count. The long chain of 6 October (`scripts/chains/long_2026-10-06.sh`, ~9.7 h,
24 runs in 10 pieces) skips every finished run and every recorded probe, and
`PIECES` selects pieces: stopped and restarted, it repeats only the run that was
in progress. Rehearsed against the 386 recorded runs: 24 runs; the 4 with
checkpoints receive the checkpoint setting through the skip function, and no other
run does; 4 probes, each reading its checkpointed run; no duplicate seed. After a
simulated stop, 21 runs and 3 probes; with `PIECES="5 9"`, 4 runs. The 256-wide
null at 20k, seed 4, already existed — the checkpointed run of §33.49 — so that
piece runs only its plain baseline.

- **P104** (~55%, stated before its run and recorded after it). The flat slot under
  rotary positions — one learned vector in the 8 dims at every position, no word
  marker — gains at least +0.0037 over the plain baseline at seed 0, half the
  constant slot's +0.0073.
- **P105** (~70%). Under rotary positions, one learned vector ADDED to every input
  dim beats zeros added the same way — its initialization-matched control — by at
  least +0.004 at seed 0: the signal needs no dims of its own.
- **P106** (~75%). At each of seeds 1–3 under rotary positions, the constant slot
  beats the zero slot of the same family by at least +0.004.
- **P107** (~65%). At step 6,000 — 1,000 steps after the switch — the model
  switched off at 5,000 reads part of speech from its deepest layer at least 1
  point less accurately than the model off from the start (seed 4): early
  reliance displaced the model's own learning of it.
- **P108** (~55%). At step 2,000 the real-target model's layer-2 part-of-speech
  accuracy exceeds the shuffled-target model's by at least 1 point (seed 4): the
  teacher speeds the model's own learning upstream of its head.
- **P109** (~70%). Over seeds 0–3 at a 20k budget, word form, frequency band and
  structure each lie within ±0.001 of their 50k gain at null BPB 1.55.

### 33.52 What the input needs, and what the probes found

The long chain of 6 October ran all ten pieces in 9 hours 5 minutes (results_70:
24 runs, the zero-slot piece of §33.51, and four probe records), with no
duplicate seed.

**D6 — one learned vector shared by every input.** Under rotary positions, final
BPB at a 20k budget:

| seed | plain baseline | zero slot | flat slot | constant slot | zeros added | flat added |
|---|---|---|---|---|---|---|
| 0 | 1.3908 | 1.3910 | 1.3832 | 1.3835 | 1.3929 | 1.3836 |
| 1 | 1.3909 | 1.3853 | 1.3849 | 1.3852 | 1.3867 | 1.3822 |
| 2 | 1.3898 | 1.3886 | 1.3842 | 1.3843 | 1.3884 | 1.3840 |
| 3 | 1.3866 | 1.3879 | 1.3801 | 1.3822 | — | — |

Within a family — identical initial weights — the flat slot beats the zero slot by
**+0.0051 ± 0.0017** (4/4), the constant slot beats it by +0.0044 ± 0.0016 (4/4),
the flat slot equals the constant slot (+0.0007 ± 0.0005), and one vector added
to every input dim beats zeros added the same way by **+0.0060 ± 0.0016** (3/3).
Against the plain baseline: flat slot +0.0064 ± 0.0004 (4/4), flat added +0.0072
± 0.0008 (3/3), constant slot +0.0057 ± 0.0006 (4/4), zero slot +0.0013 ± 0.0015
(2/4), zeros added +0.0012 ± 0.0018 (2/3). **P104 right** (+0.0076 at seed 0);
**P105 right** (+0.0093); **P106 wrong** — at seed 1 the constant slot beats the
zero slot by only +0.0001, because that seed's zero slot is the outlier (0.0056
better than its baseline); over four seeds the margin is +0.0044 ± 0.0016, 4/4.

Under rotary positions, then, a single learned vector shared by every input
helps, whether it has dims of its own or is added to all of them; dims merely kept
free of token identity do not; whether the vector marks words is irrelevant.
Added to every input, the vector is a reparameterization of the token embedding —
a shared vector plus each token's row reaches nothing the rows alone cannot — so
the gain lies in how the model learns, not in what it can represent.

**A confound found while writing this down (G-71).** The causal shift gives the
first word of every window the sentinel class, since no earlier word is known
(`causal_shift`). The flat and constant slots, and the flat vector added,
therefore also mark the start of each window — one to three tokens of 256 —
which a rotary model cannot otherwise see; the zero slot and zeros added mark
nothing. With learned positions the window start is visible anyway, and there
one vector added to every input still beat zeros added by +0.0085 ± 0.0022 (4/4;
§33.15) — evidence for the vector itself. Under rotary positions the two are not
yet separated. Without the causal shift a flat slot is a pure constant, and it
starts from the same initial weights (the shift reads only tokenizer tables, with
no random draw): one run separates them, with no new code.

**D3 — the deficit is not lost knowledge of part of speech.** Seed 4 adds −0.0027:
over five seeds the model switched off at 5,000 of a 20k budget ends **0.0039 ±
0.0009** below off-from-start (0/5 above). The gap levels off — −0.0051 at step
10k, −0.0044 at 12.5k, −0.0039 at the end — while the learning rate is still
1.1e-4 at 12.5k (a third of its peak); weaning's gap is flat at −0.0029 from step
7,500. The probe (seed 4, deepest layer; the model is fed its own features):

| step | 1k | 4k | 5k | 6k | 8k | 12k | 20k |
|---|---|---|---|---|---|---|---|
| off from the start | 0.815 | 0.859 | 0.866 | 0.870 | 0.874 | 0.879 | 0.882 |
| off at 5,000 | 0.831 | 0.867 | 0.854 | 0.867 | 0.873 | 0.879 | 0.882 |

Before the switch the switched model reads part of speech better — its input
carried the previous word's tag. One training step after it, the deepest layer is
1.2 points behind; 1,000 steps later 0.3; from then on level, and at layer 2 a
little ahead (+0.6 at the end) — while its BPB stays behind. **P107 wrong.**
Whatever the lasting deficit is, it is not the model's knowledge of part of speech.

**D4 — the teacher teaches; the knowledge stops paying.** Seed 4: the real target
beats the shuffled one by +0.0029; over five seeds **+0.0009 ± 0.0008** (3/5). The
probe (seed 4):

| step | 1k | 2k | 5k | 10k | 20k |
|---|---|---|---|---|---|
| layer 0, real / shuffled | 0.860 / 0.804 | 0.862 / 0.829 | 0.866 / 0.843 | 0.864 / 0.850 | 0.864 / 0.850 |
| layer 2, real / shuffled | 0.874 / 0.819 | 0.885 / 0.839 | 0.893 / 0.863 | 0.896 / 0.877 | 0.900 / 0.884 |
| layer 4, real / shuffled | 0.875 / 0.819 | 0.892 / 0.838 | 0.911 / 0.867 | 0.916 / 0.879 | 0.921 / 0.884 |

The real-target model encodes part of speech better at every layer from step
1,000 — its input embeddings included (+5.5 points at layer 0) — **+4.6 at layer 2
by step 2,000 (P108 right)**, and still +1.4, +1.6 and +3.7 at layers 0, 2 and 4 at
the end (layer 4 is the one the head is trained to read). The extra knowledge
persists while the advantage in prediction fades: knowing more part of speech than
language modelling itself acquires does not help predict text.

**D1 — the law per feature, at three schedules, four seeds each.** At null BPB
1.55 the 20k-budget gains match the 50k ones within 0.0004: part of speech from the
word form +0.0100 (50k: +0.0101), frequency band +0.0055 (+0.0052), structure
+0.0010 (+0.0011), inflection +0.0070 (+0.0069). **P109 right.** End gains at a
20k budget over seeds 0–3: part of speech +0.0075 ± 0.0008, word form +0.0050 ±
0.0006, inflection +0.0022 ± 0.0011, frequency band +0.0014 ± 0.0005, structure
−0.0008 ± 0.0006 — the order of their curves.

**D2 — probe accuracy does not line up the widths (desk).** With the 20k-budget
probes of both nulls (seed 4) and the part-of-speech gain over seeds 0–3 at the
same steps: at deepest-layer accuracy 0.870 the base model gains +0.0133, the 4×
wider one +0.0076; at 0.880, +0.0096 and +0.0036. The wider model's gain falls
away even at equal knowledge of part of speech — D2's pivot: "learned it" is not
the linear decodability of the current word's part of speech. One candidate
remains: these probes read the *current* word's tag, while the feature carries the
*previous* word's — which is what the model must infer from context for the
feature to become redundant.

**256 wide, seed 4:** plain baseline 1.2757, null 1.2735 (+0.0022). Over five seeds
the null leads by +0.0055 ± 0.0039 (4/5), and the baseline's spread of final BPB
is 5.7 times the null's (0.0237 against 0.0041). P91 and P92 await seeds 5–7.

### 33.53 Tools for other conditions

Every claim so far rests on two conditions — WikiText-103 with a 17k BPE, and
Ukrainian Wikipedia with its own 17k BPE. Other corpora, tokenizers and models
were opened (6 October), and testing how far the claims travel is the
least drifting use of the remaining time: it is the examiner's first question.
Tools first, then chains (`docs/reference/conditions.md`).

**Corpora as specs** (`morphlm/data/corpora.py`). A corpus is a dataset, a text
field, a language and tagger, a word rule, a split rule (`native`, or `hash` —
the Ukrainian rule, now general) and a serving rule (`rows`, or `paragraphs`
with their newline, G-64). `stream.load_corpus` reads only the spec. WikiText
keeps its single historical call and refuses a fraction (its cache key carries
none). An unknown corpus fails when the configuration is checked; a tagger of
another language fails before the first document. On toy corpora — the toy
WikiText, and a toy Ukrainian Wikipedia built from the UD_Ukrainian-IU
treebank's real sentences (`scripts/rehearsal/make_toy_ukwiki.py`) — the new
code rebuilt all 34 cache files of the old code byte for byte, and the
tokenizer and its sidecar; a test reimplements the old Ukrainian loader verbatim
and requires the same rows at every split, size and holdout.
`scripts/data/corpus_info.py` looks at a candidate dataset before it gets a spec.

**Tokenizers** (`scripts/data/train_tokenizer.py`, `morphlm/data/tokenization.py`).
One training path, shared with `cli build bpe` (byte-identical tokenizers for
BPE in both languages). Standard names (`bpe_uk_8k`), and a tokenizer is never
overwritten: its name is in every cache key built from it. That matters because
**Unigram training is not deterministic** in the `tokenizers` library — three
trainings with the old code, even on one thread, gave the same 387 pieces with
scores differing by up to 0.02 — so a Unigram file is an artifact, not a
recipe. The sidecar records the recipe and the file's sha256. The statistics
measure what a tokenizer does to words on held-out text: tokens per word, words
split, and **ending exposure** — how often a token boundary falls exactly
between stem and ending in inflected words (школ|и, студент|ам). On the toy
corpus the gradient is the expected one: case-bearing endings exposed in 54.6%
of words at 600 tokens, 38.6% at 1,500, 26.1% at 4,000, and 51.6% for Unigram at
1,500 — a larger vocabulary hides endings inside whole-word tokens, Unigram
exposes them.

**One command per condition** (`scripts/data/prepare_condition.py`): the Ukrainian
recipe for any corpus × size × tokenizer — a timed validation pilot that
estimates the train build, its check, the tagging pass, the null, the zero slot
and the word-form variants, and a final gate on both splits. It calls the
existing commands, skips what is done (a ready condition: 10 seconds) and stops
at the first failure. Rehearsed end to end on the toy corpus with a new
tokenizer, then four runs trained on the new condition.

**G-66 closed.** Outside WikiText a lexicon is named for the corpus and size it
was built from (`lexicon_ucase_bpe_uk_8k_ukwiki0.05.npz`); readers fall back to
the historical name. All 16 derived codes files stayed byte-identical. While
refactoring, the heading check of G-37 showed again — `startswith(' =')` after
`strip()` never fires — and now says so in the code.

The model tools — a fit check for larger models, RMSNorm and SwiGLU blocks —
follow in §33.54.

### 33.54 Tools for other models

The second half of the tools (§33.53): two block options and a way to size a
model before chaining it.

**RMSNorm and SwiGLU** (`model.norm`, `model.mlp`; `docs/reference/apparatus.md`,
§2.1a). RMSNorm keeps a gain and drops the mean subtraction and the shift;
SwiGLU gates the MLP with a hidden width of 8d/3 (344 at d=128), so its weights
match the GELU MLP's within 1%. Each marks the label (`_rmsnorm`, `_swiglu`);
with rotary positions the block is LLaMA-style. RMSNorm draws no random number,
so at the same seed a model with it starts from exactly the weights of its
LayerNorm twin — a matched pair, and a sharp test for D6: with no mean
subtraction, a constant input vector no longer shifts what the norm removes.
Default models are untouched: their outputs are bit-identical to the old code's
in seven kinds of cell (baseline, causal part of speech, the null, rotary
positions with the zero slot, 256 wide, the auxiliary target, the switch).

**The fit check** (`scripts/ops/fit_check.py`) builds the model a command line
would build, times the training step as the loop runs it — AdamW, clipping, the
real batch shape — and one validation pass as the loop runs it, and projects the
run. Over a grid of widths and depths it sizes a scale study in one call. Writing
it showed that the validation pass runs without `torch.no_grad()`: it builds
graphs it never uses — about 2% of a run's time, no change to any result — and
the fit check measures it as it is.

Rehearsed end to end on the toy Ukrainian condition of §33.53: runs with rotary
positions, RMSNorm and SwiGLU train, record their parameters (core 796,992
against the standard 793,344) and reload from their checkpoints with their
settings. The two new fields change the run ids of every run made from now on, as
each field added since September has (§33.48); labels and cache keys of all 411
recorded configurations are unchanged, and chains match cells by label and seed.


### 33.55 What the tokenizers expose, and codes read across corpus sizes

**The 17k condition at 0.05 is complete**: the zero slot and lexical case
derived — 5.58% of validation words are unseen in its 38M-token train split —
and every gate passed (`prepare_condition.py`, a few minutes, no tagging).

**Four Ukrainian tokenizers** (D7), trained on the 0.2 train split like
`bpe_uk_17k` (5.29M rows, 2.74M distinct words) and lossless by round trip:
BPE at 4k, 8k and 32k, and Unigram at 17k. Measured on 3,000 held-out
paragraphs — 15,740 inflected words, 12,701 of them nouns, proper nouns or
adjectives (`train_tokenizer.py --stats`):

| tokenizer | characters per token | tokens per word | words split | stem\|ending boundary | case-bearing words |
|---|---|---|---|---|---|
| BPE 4k | 2.73 | 2.37 | 64.7% | 21.5% | 20.4% |
| BPE 8k | 3.13 | 2.04 | 55.0% | 14.5% | 13.8% |
| BPE 17k | 3.59 | 1.73 | 43.3% | 9.1% | 9.0% |
| BPE 32k | 3.96 | 1.53 | 34.2% | 5.7% | 5.6% |
| Unigram 17k | 3.07 | 1.85 | 62.9% | **41.1%** | **43.7%** |

BPE hides endings as it grows: each doubling of the vocabulary removes about a
third of the remaining stem|ending boundaries, the words absorbed whole.
**Unigram at 17k exposes four and a half times as many endings as BPE at 17k**
while compressing like BPE at 8k: it splits more words, and splits them where
the morphology is. When a boundary falls at the ending, the ending is nearly
always one token (the "own token" share trails the boundary share by at most
half a point), so the two measures say the same thing. D7 therefore has two
axes: the vocabulary within BPE (exposure 21.5% → 5.7%, compression changing
with it) and the algorithm at a fixed size (9.1% against 41.1% at 17k). The law
predicts that case helps most where endings are hidden — BPE 32k — and least
where they are exposed — Unigram 17k. Predictions are recorded when the chain
is planned; the four conditions at 0.05 are being prepared on the CPU.

**Codes read across corpus sizes (G-72).** The condition's own checks showed
it. Before its derivations, the validation check reported the null firing on
92.9% of word tokens; after them 78.7% — though the null had not been derived
again — and the zero slot 87.8%. A corpus's validation split is the same text
at every fraction, so the Ukrainian validation streams at 0.002, 0.01, 0.05 and
0.2 are the same 325,867 tokens under four cache stems. The guard that keeps
codes with their stream compared token counts, and the check, a derivation's
source lookup and a run's own lookup each took any file with the right count —
in name order, or the fullest. For built codes, spaCy's tags of the same
tokens, that is harmless. For derived codes it is not: a null or a word-form
lexicon is built from its fraction's own train split, and a word unseen there
gets the class `none` — in training, the mark of a non-word. The three numbers
are three lexicons: 0.2's (7.1% of validation word tokens marked unseen),
0.05's (12.2%), and 0.01's (21.3%), which the 0.05 zero slot's derivation had
copied into its file.

Reconstructed from the run records and the derivation records — and reproduced
exactly on the toy corpus — from the night of 1 October, when the 0.01 null was
derived (its stem sorts first), **every Ukrainian null run read the 0.01
fraction's validation null, whatever its own fraction**:

| null runs, 4 seeds each | read | unseen marks against its own lexicon |
|---|---|---|
| ~1.5M tokens (0.002), 10k steps | 0.01's | fewer |
| ~38M (0.05), 10k steps | 0.01's | more |
| 151M (0.2), 5k, 10k and 20k budgets | 0.01's | more |

The chain logs, where every run printed the file it reused, confirm the
reconstruction exactly: 24 runs reused the 0.01 validation null — the 20 above
and the four 0.01 runs — 18 reused 0.2's own files (the null at 50k and added,
the zero slot, lexical case), 8 reused the 0.01 lexical-case file, and every
English reuse in the logs is of its own stream's files.

Unaffected: the ~7.6M runs (0.01, their own), the 151M runs at 50k and every
zero-slot run (all before that night), contextual case (built codes), lexical
case (its files at 0.002, 0.01 and 0.2 all hold the 0.2 lexicon by design —
identical rows), and English (one stream per tokenizer). So the nulls behind
§33.31–§33.32 (case over its matched null at ~1.5M, ~38M, and 151M at 10k and
20k steps; P57, P60–P64), §33.38 (Ukrainian case at 5k and 20k; P73, P76) and
the Ukrainian points of the learning law (§33.35) were scored on another
fraction's lexicon. If unseen marks hurt a null, more of them than its own
inflate the gain (~38M; 151M at 5k–20k) and fewer flatter the null (~1.5M).

**The fix.** A derived file is read only by the stream it was derived for
(`stream.stream_codes_files`, now used by the loader, by a derivation's source
lookup and by the check). Built files are still shared by token count, and an
older build file still lends a fact the exact file lacks. A test builds three
fractions' files and requires every run to read its own; it fails under the old
rule. The check names the file each fact was read from. On the toy corpus the
old code reproduced the symptom exactly (the null at 44.9% both before and after
the derivations, the zero slot 39.0%); after the fix both read 39.0%. The 0.05
zero slot's file must be derived again; its own row was right, the copied null
row was 0.01's.

**The audit** (`scripts/audits/codes_audit.py`) lists, for every recorded run,
the validation file it read under the old rule — over the files that existed
when it started — against its own stream's file, and checks every copied row
against the file that derived it. With `--score` it re-scores the affected
runs' saved models — the best step's weights — on the codes they were scored
on, which must reproduce the recorded BPB or the reconstruction is wrong; on
their own fraction's codes; and on the constant itself. The chain logs give a
second witness: every run printed the file it reused (`[codes reuse] …`).
Rehearsed on the toy corpus: the as-read scores reproduce the records to four
decimals.

**A question under it (G-73).** Even on its own lexicon the null is not constant
at validation: every word unseen in train is marked `none`, a mark the null met
only on non-words in training. The contextual feature's codes carry no such
mark — spaCy tags every word — so the null may carry a handicap that grows as
the data shrinks. On the toy corpus, where 38% of validation words are unseen,
a null trained for 600 steps scores 0.0083 better on the constant itself — its
training codes exactly, every word one class. `--nulls` measures this for every
constant null; if it is not negligible, the null's validation codes become the
constant, the existing checkpoints are re-scored, and the comparisons restated.

Predictions, before the audit:

- **P110** (~85%). On its own fraction's codes the null scores better than
  recorded at every seed at ~38M tokens and at 151M with 5k, 10k and 20k
  budgets, and worse at ~1.5M.
- **P111** (~75%). On the constant itself every Ukrainian null scores better
  than on its own codes, at every seed.
- **P112** (~50%). For the 151M null at 50k that improvement is smaller than
  0.001.
- **P113** (~65%). For every English constant null it is smaller than 0.0005.

### 33.56 What the nulls read, and a null that was not constant

**The audit** (`scripts/audits/codes_audit.py`, 6 October). Of 343 recorded runs
that read context codes, 319 read their own stream's file, 4 another stream's
file with identical rows (lexical case at ~1.5M tokens, whose lexicon is the
0.2 one by design), and 20 read different codes — exactly the 20 null runs of
§33.55. None would miss its codes under the new rule. One file held a foreign
copy, the 0.05 zero slot's null row (7.1% of tokens); derived again, every
copied row equals its original. Re-scored on the codes the audit says they
read, all 105 saved models — the 20, and 85 constant nulls that read their own —
reproduce their recorded BPB to four decimals, so the reconstruction is exact.

**G-72, the size of it.** On its own fraction's codes each affected null scores
differently at every seed: worse by 0.0019 ± 0.0001 at ~1.5M tokens (its own
lexicon marks more words unseen), better by 0.0010 at ~38M and by 0.0013–0.0014
at 151M with 5k, 10k and 20k budgets. **P110 right.**

**G-73: the null was not constant.** On the constant itself — its training
codes' rule, every word one class — every Ukrainian null scores better than on
its own codes, at every seed:

| null runs | constant − own codes | validation word tokens marked unseen |
|---|---|---|
| 151M, at 5k, 10k, 20k and 50k budgets alike | −0.0005 | 7.1% |
| 151M, the null added to the token dims | −0.0007 | 7.1% |
| ~38M | −0.0010 | 12.2% |
| ~7.6M | −0.0020 ± 0.0001 | 21.3% |
| ~1.5M | −0.0039 ± 0.0002 | |
| English, 71 runs | at most −0.0002 (mean −0.0001) | |
| zero slots, both languages | 0.0000 | (codes never reach the model) |

**P111, P112 and P113 right.** The handicap grows with the share of held-out
words unseen in the null's train split, and a morphologically rich language
has many more unseen word forms: in English it is noise, in Ukrainian it is the
size of the effects measured. It contradicts the null's own definition — "every
word given the same class" (`MorphConstBlock`) — so it is a bug, not a design:
the constant was a lexicon lookup, and a word missing from the lexicon got
`none`.

**The fix.** `--transform const` now gives every word the class at every split,
seen in train or not, and the null codes new text (BLiMP) the same way, with no
lexicon. Train codes cannot change — every train word is in its lexicon — and a
derivation that replaces a file now reports how many tokens it changed: on the
toy corpus, 0 in train, the unseen words in validation. A constant's codes file
declares the rule (`rules`); a run of the null refuses codes that do not, with
the command that fixes them; and every run records the codes file and rules it
read per split (`data.<split>.codes_file`), so the next audit needs no
reconstruction. `scripts/data/rederive_constants.py` replays every constant
derivation in the records, in order, resumable. Quarantine rule `const_unseen`
takes the 34 Ukrainian runs with `mconst` among their facts; English runs stay
(at most 0.0002), and so does case beside a zero slot.

**Restated** — Ukrainian, against the constant null, from the saved best-step
models (the audit's printed scores, four decimals):

| | recorded | restated |
|---|---|---|
| case over the null at 10k steps, ~1.5M tokens | +0.0096 ± 0.0016 | **+0.0076 ± 0.0016** (4/4) |
| ~7.6M | +0.0052 ± 0.0002 | **+0.0031 ± 0.0002** (4/4) |
| ~38M | +0.0052 ± 0.0007 | **+0.0032 ± 0.0007** (4/4) |
| 151M | +0.0044 ± 0.0004 | **+0.0026 ± 0.0004** (4/4) |
| 151M at 5k steps | +0.0074 ± 0.0007 | **+0.0055 ± 0.0006** (4/4) |
| 151M at 20k steps | +0.0015 ± 0.0004 | **−0.0003 ± 0.0004** (1/4) |
| 151M at 50k steps | −0.0009 ± 0.0008 | **−0.0013 ± 0.0008** (1/4) |
| word form alone, ~1.5M | +0.0070 ± 0.0013 | +0.0050 ± 0.0013 (66% of case's gain) |
| word form alone, ~7.6M | +0.0034 ± 0.0004 | +0.0014 ± 0.0004 (43%) |
| word form alone, 151M at 50k | −0.0015 ± 0.0010 | −0.0020 ± 0.0010 |
| the null over the plain baseline, ~1.5M | +0.0079 ± 0.0016 | +0.0099 ± 0.0016 |
| ~7.6M | +0.0065 ± 0.0017 | +0.0085 ± 0.0017 |
| 151M at 50k | +0.0023 ± 0.0010 | **+0.0028 ± 0.0010** |

**The claim stands, smaller.** Before the model has learned case, case helps —
at every seed and every corpus size at a 10k budget: about +0.003 from 7.6M to
151M tokens, +0.008 at 1.5M — and on 151M it is gone by 20k steps, not 50k.
Read from the word form alone it keeps two-thirds of the gain at 1.5M and
two-fifths at 7.6M, so **P60 is half right restated** (at least half was
predicted). The other verdicts hold: P57, P58, P63, P64 (−0.0003, between
+0.0026 and −0.0013), P73 and P76 right; P61 and P62 still wrong (+0.0026 is
outside ±0.002; at ~38M +0.0032 against +0.0031 at ~7.6M — still not below).

**What re-scoring cannot restate.** Numbers read from evaluations during
training — Ukrainian case at step 2,000 (+0.0060 ± 0.0005), the law's Ukrainian
points (§33.35), the steps the feature saves (§33.33) — are curves, not saved
models. The handicap was the same 0.0005 at four training lengths on 151M, so
they move by about the measured offset: case at step 2,000 ≈ +0.0055; the law's
points at null BPB 1.10 ≈ +0.0030 (50k), +0.0023 (20k), +0.0031 (10k), +0.0036
(~38M). Re-running the Ukrainian nulls makes them exact: training is
deterministic (G-69: bit-identical at 21 evaluations) and the null's training
codes are unchanged, so the same models come back with every evaluation on the
constant — and their best-step BPB must equal the audit's re-scores, a check.

### 33.57 The constants replayed, and the night of 6 October

**The replay** (`rederive_constants.py`, 18:48–19:00): all 13 constant derivations
in the records — seven English (the null, its three widths, the three zero
slots) and six Ukrainian — derived again. Every train split changed by **0
tokens**, 120.6M English and 151.1M Ukrainian among them; only held-out words
unseen in train changed: 1.44% of English validation tokens (2.60% of test), and
in Ukrainian 5.54% at 151M, 9.47% at ~38M, 16.54% at ~7.6M and 26.89% at ~1.5M.
The audit then found every copied row equal to its original and all 20 constant
files constant on every word, the rule declared. The quarantine's dry run named
exactly the 34 runs expected.

**The night chain** (`scripts/chains/night_2026-10-06.sh`, ~7.7 h, six resumable
pieces, ordered by what each settles):

1. **G-71** — the flat slot and the flat vector added, *without* the causal
   shift, under rotary positions (seed 0): a pure constant, the same initial
   weights, no window-start marker.
2. **G-73** — the ~38M-token Ukrainian null again, seeds 0–3: the 17k reference
   of D7, and the first check that a re-run reproduces the audit's re-score.
3. **G-73** — the 151M null at 50k again, seeds 0–1: early help at step 2,000,
   and Q2's end, on the constant.
4. **D7** — each new tokenizer whose condition is ready: the null, case and
   lexical case at ~38M tokens, 10k steps, seed 0; and lexical case under BPE 17k.
5. **G-73** — the 151M null at 50k, seeds 2–3.
6. **D7** — the tokenizers not ready in piece 4, waiting while they are prepared.

Before piece 2 the chain makes sure the 34 runs are out of `results/` (else
`--done` would take them for finished), moving them only if the dry run is
exactly the one read. If no preparation is running when it starts, the chain
prepares the D7 conditions itself on the CPU. Rehearsed on the toy corpus, real
code end to end: the quarantine moved 34 stand-in runs; every piece trained its
cells; a toy Unigram condition, prepared by the chain's own CPU track, was not
ready in piece 4, waited for in piece 6, then trained; a second pass skipped every
run; an unexpected quarantine skipped pieces 2, 3 and 5. The rehearsal caught one
fault: piece 1's smoke test took the default English tokenizer path, which a toy
has not got — the tokenizer is now a knob (`ENTOK`) the smoke test shares with
the runs.

Predictions:

- **P114** (~65%). Without the causal shift the flat slot under rotary positions
  keeps at least half its gain over the plain baseline at seed 0 (≥ +0.0038; with
  the shift +0.0076): the shared vector, not the window-start marker, carries it.
- **P115** (~65%). Without the causal shift the flat vector added keeps at least
  half its gain over the plain baseline at seed 0 (≥ +0.0036; with the shift
  +0.0072).
- **P116** (~85%). Re-run on the constant, each Ukrainian null ends within
  0.0002 of the audit's re-score of its quarantined checkpoint.
- **P117** (~70%). At step 2,000 the re-run 151M null evaluates 0.0002–0.0008
  below its quarantined record at every seed re-run: the handicap is about the
  same early as late.
- **P118** (~65%). Under each of the four new tokenizers case beats its null at
  the end of a 10k budget on ~38M tokens (seed 0).
- **P119** (~55%). Case gains less under Unigram 17k (endings exposed in 41% of
  inflected words) than under BPE 17k (9%), seed 0.
- **P120** (~55%). Case gains more under BPE 32k (5.7%) than under BPE 4k
  (21.5%), seed 0.

One seed per tokenizer reads a direction, not a size: the per-seed spread of
case's gain at ~38M is about 0.0014.

### 33.58 The night of 6 October: the vector, the exact null, five tokenizers

The chain ran all six pieces in 7 hours 34 minutes (results_71: 23 runs). No
preparation was running when it started, so its CPU track prepared all four
tokenizer conditions itself, each in 1 h 15 min to 1 h 31 min of tagging; the
8k condition was ready only for piece 6, which ran it. (The preparation's pilot
estimate, 2.4–2.7 h, runs about 1.8 times long: the pilot's fixed start-up cost
is counted as tagging. About 1.3 h per condition at 0.05 is the planning figure.)

**G-71 closed: the vector, not the window-start marker.** Without the causal
shift — a pure constant, no sentinel at the window's first word, the same
initial weights — under rotary positions at seed 0 the flat slot ends 1.3840
and the flat vector added 1.3849, against the plain baseline's 1.3908:

| seed 0, rotary, 20k | with the shift (marks the window start) | without it |
|---|---|---|
| flat slot, over the plain baseline | +0.0076 | **+0.0068** (89%) |
| flat vector added, over the plain baseline | +0.0072 | **+0.0059** (82%) |

**P114 and P115 right.** The marker is worth 0.0008 and 0.0013 here — the two
cells of each pair start from the same weights — and the shared vector carries
the rest.

**G-73 closed: the re-runs are the audit's re-scores.** Run again on the
constant, every null reproduced the audit's re-score of its quarantined
checkpoint to the sixth decimal (seven of eight exactly; one 50k seed by
0.000003, its best step 49,500): training is deterministic and the null's
training codes were unchanged, so the restated best-step numbers of §33.56 are
exact. **P116 right.** Along training the handicap grew from 0.0002 at step
1,000 and 0.0004 at step 2,000 to 0.0005 from step 5,000 on, at every seed
(0.00038–0.00047 at step 2,000): **P117 right.** Against the re-run nulls,
Ukrainian case leads at step 2,000 by **+0.0056 ± 0.0005** (4/4; first recorded
+0.0060 ± 0.0005), and the 151M curve now matches the ~38M one at the same null
BPB — +0.0055 against +0.0049 at 1.25, +0.0030 against +0.0036 at 1.10.

**D7, seed 0 — case under five tokenizers** (~38M tokens, 10k steps; gains over
each tokenizer's own null):

| tokenizer | endings exposed | null BPB | case | word form | case at step 2,000 |
|---|---|---|---|---|---|
| BPE 4k | 21.5% | 1.2239 | +0.0047 | +0.0020 | +0.0082 |
| BPE 8k | 14.5% | 1.1542 | +0.0017 | −0.0004 | +0.0052 |
| BPE 17k | 9.1% | 1.0891 | +0.0028 | +0.0002 | +0.0064 |
| Unigram 17k | 41.1% | 1.0890 | +0.0034 | +0.0001 | +0.0066 |
| BPE 32k | 5.7% | 1.0449 | +0.0035 | +0.0011 | +0.0053 |

**P118 right**: case beats its null under every tokenizer, and leads early under
every one. **P119 and P120 wrong**: Unigram, which exposes the most endings,
gains a little more than BPE at the same size, and BPE 32k less than BPE 4k. At
one seed nothing here orders by ending exposure — BPE 17k's four seeds spread
0.0014 — so the exposure hypothesis has no support yet. Read at the same null
BPB (the law's axis), case's gain is about the same under every tokenizer: at
1.25, +0.0055 (4k), +0.0052 (8k), +0.0060 (17k), +0.0057 (Unigram), +0.0054
(32k). The one departure is Unigram early: +0.0112 at null BPB 1.40 against
+0.0046 for BPE 17k — the opposite of what hidden endings would predict. So far,
what sets case's gain is how much the null has learned, not how the tokenizer
cuts words.

Two side observations. At the same 10k steps Unigram 17k reaches the same BPB as
BPE 17k (1.0890 against 1.0891) while reading 15% less text per step (3.07
characters per token against 3.59). And across BPE sizes the null's BPB falls
from 1.2239 to 1.0449 — the larger vocabulary reads more text per step and holds
more parameters (1.8M at 4k, 8.8M at 32k, the same 0.8M core) — which is why the
comparison across tokenizers is made at equal null BPB.

### 33.59 What the feature carries, probed — and the thesis kept in scope

**D2: the previous unit, probed.** At each position a causal feature carries the
class of the *previous* unit — the previous word, or a run of punctuation (its
last token's class) — and the window's first unit gets the sentinel (§30). The
probes so far read the *current* word's tag. `probe.py --previous` labels every
position with what the feature would carry there, shifted by the model's own
functions (`word_runs` over `_word_starts`, then `causal_shift`), word starts from
the model's tables. A test checks the labels three ways, on Ukrainian text whose
letters and dashes the tokenizer splits into byte fragments and in a window that
opens inside a word: against units read independently from the lexicon's word
spans; with the model's tables as with the tokenizer's; and against the model
itself — a causal embedding fed the current labels puts in its slot exactly what
a plain one puts there for the probe's labels. It fails for a shift by one token,
for split characters ignored and for a wrong sentinel. Rehearsed end to end on toy
runs of two widths, with checkpoints.

**The desk half** (`scripts/analysis/probe_axis.py`): every probed null's
accuracy at each checkpoint, paired with the feature's gain at the same step over
the seeds both cells share; the widths compared within each budget at six levels
across the accuracy range both reach; and each condition's accuracy at common
null BPB. On the current-word records it reproduces §33.52 — at 0.870, +0.0133
against +0.0076; at 0.880, +0.0097 against +0.0036 (recorded +0.0096; the
interpolation) — and gives the verdict the pivot needs: along the 20k budgets the
widths' gains differ by up to **0.0063** (at 0.874), along the 10k budgets by up to
0.0052. Up to an accuracy of about 0.85 they agree within 0.0013 in both budgets;
they part where the wider model's gain collapses. At matched null BPB the wider
null leads on the current word's tag by +0.8 to +1.3 points (1.75 to 1.45).

**The block** (minutes of GPU): the previous-unit probe on six checkpointed runs,
seed 4 — the four nulls, 128 and 256 wide at 10k and 20k budgets (D2), and D3's
switched pair, off from the start and off at 5,000. A check before anything is
read: until step 5,000 the switched model's input carries exactly the probe's
label, so at its layer 0 it must read the previous unit's tag far better than the
model never fed it (by 10 points or more) — and about level after the switch.

Predictions:

- **P121** (~15%). The pivot. By `probe_axis.py --previous`, the two widths line
  up along the 20k budgets: at every common level of the deepest layer's
  previous-unit accuracy their part-of-speech gains differ by at most 0.0015 (on
  the current word's tag: up to 0.0063). Against it: the wider model's gain falls
  to the base model's final +0.0075 by step 2,000, so to line up it would have to
  know the previous word's tag at step 2,000 as well as the base model does at
  20,000.
- **P122** (~60%). At each of the seven matched null BPB levels from 1.75 to
  1.45 (20k budgets), the wider null's deepest-layer lead on the previous unit's
  tag exceeds its lead on the current word's (+0.8 to +1.3 points): the width
  effect lies in reading context.
- **P123** (~60%). At the end of the 20k budget the model switched off at 5,000
  reads the previous unit's tag at the deepest layer within 0.5 points of the
  model never given the feature: the lasting deficit (§33.52) is not missing
  knowledge of what the feature carried either.
- **P124** (~55%). At the end of its 20k budget the 128-wide null reads the
  previous unit's tag at the deepest layer below 0.85 (the current word's: 0.881)
  — at the end it still lacks much of what the feature tells it, as the
  feature's remaining +0.0075 would suggest.

**The thesis kept in scope** (7 October: every theory presented
and every test run must serve the thesis — context-aware morphological embeddings
for language models). D1–D5 and D7–D9 ask when, why, and how far morphology in
the embedding helps. D6 asks how a shared input vector helps a small transformer.
For the thesis it matters as the confound it explains: what looked like word
structure helping was the slot itself, which is why every feature is measured
against a same-shape null — and G-71, the one test that could have changed that
reading, is closed (§33.58). Its remaining tests — the fixed against the learned
vector, G-71's seeds 1–3, the vector added under rotary positions at seed 3, the
256-wide pair under rotary positions — and the 256-wide spread at seeds 5–7 (P91,
P92) are about the vector itself: parked, P91 and P92 not tested. FINDINGS now
presents the vector as that confound, and D8 drops its RMSNorm test of D6.

**Other tokenizers** (are BPE variants enough?). A
controlled comparison of 64 models (350M–2.4B parameters) found the best subword
tokenizers — BPE, Unigram, WordPiece and two BPE variants — statistically
indistinguishable on downstream tasks (Schmidt et al. 2024, arXiv 2402.18376);
and morphological alignment did not explain why morphologically complex
languages are modelled worse — data size, counted in bytes, did (Arnett and
Bergen, arXiv 2411.14198). D7 already crosses the algorithm (BPE, Unigram) with
the vocabulary size, and at the same null BPB case gains about the same under
all five (§33.58). Tokenizers that differ in kind are outside the design: superword
tokens span several words, so a word's feature has no token of its own
(SuperBPE, arXiv 2503.13423); byte-level patching needs its own architecture
(BLT, arXiv 2412.09871). What D7 lacks is the far end of its own axis: a
tokenizer that exposes every inflectional ending — morphology in the tokenizer,
the thesis's alternative to morphology in the embedding (morphology-aware BPE:
MorphBPE, arXiv 2502.00894). For Ukrainian, a BPE that never merges across a
stem|ending boundary, the ending found from the word form alone so that it reads
no context. A candidate (DIRECTIONS D7): code first, then a
condition (~1.3 h of tagging) and its runs.

### 33.60 The previous word's tag: half the width gap, and what the switch leaves

The probe ran on all six runs (results_72: six records, about six minutes). The
check passed: until step 5,000 the switched model's input carries the probe's
label, and its layer 0 reads it at 0.997–1.000, against 0.33–0.36 for the model
never fed it; after the switch, 0.362 against 0.359. The labels are what the model
reads.

**What the nulls know of the previous word's tag** (seed 4; majority class 0.181;
from the current token alone, layer 0, 0.44): far less than of the current
word's. At the end of the 20k budgets the deepest layer reads it at **0.619** (128
wide; best layer 0.655) and 0.731 (256 wide; best 0.805), against 0.881 and 0.895
for the current word's tag. **P124 right.** The wider model's lead lies where
context is read: at matched null BPB from 1.75 to 1.45 it leads by **+3.9 to +10.4
points** on the previous word's tag, against +0.8 to +1.3 on the current word's.
**P122 right.**

**The pivot: closer, not lined up.** On the previous word's tag the two widths'
gains differ by up to **0.0035** along the 20k budgets and 0.0025 along the 10k
budgets, above the 0.0015 the pivot set. **P121 wrong.** Yet each axis closer to
what the feature carries cuts the widest gap — on the null's BPB 0.0098 (20k) and
0.0099 (10k), on the current word's tag 0.0063 and 0.0052, on the previous word's
0.0035 and 0.0025 — and the mean gap halves (+0.0034 to +0.0017, 20k). At the top
of the shared range the widths meet exactly: the base model at the end of its 20k
budget and the wider one at step 2,000 both read the tag at 0.62 and both gain
+0.0075 — the coincidence P121's own argument held unlikely. The deepest layer
lines up best (the others: 0.0036–0.0042 along 20k). One probed seed per width.

D2 closes as its pivot said: the law is stated within one model size. It closes
with a mechanism in hand. The width effect is largely how early a model reads the
previous word's part of speech from context — about half of what separates the
widths' curves, not all of it. And the feature stops paying long before the null
knows the tag fully: the wider model's gain has collapsed (+0.0033 at step 3,000)
when its deepest layer reads the tag at 0.68.

**D3 — the switched model rebuilds the tag, and ends knowing more of it.** One
checkpoint after the switch its deepest layer reads the previous word's tag 9.5
points below the model never fed it (0.443 against 0.538); 1,000 steps later level
(0.556, 0.557); ahead from then on. At the end of the 20k budget: +2.0 points at
layer 4 (0.635 against 0.615) and +1.8 at layer 3, but −1.6 at layer 2 and −1.2
at layer 1. **P123 wrong by its rule** — 2 points, the other way. The lasting
deficit (−0.0039 BPB) is not a lack of what the feature carried: the switched
model reads it from context as well as the never-fed one, at its deep layers
better.

**The proposed tokenizer, first look at its ending rule** (desk; the toy
Ukrainian corpus, 74k words). The D7 exposure measure takes the stem to be the
longest common prefix of a word and its lemma. Where the lemma's own ending
begins like the word's, that puts the boundary inside the ending — `книгами`
(lemma `книга`) becomes `книга|ми`, `важливих` becomes `важливи|х` — and citation
forms are not counted at all. A rule that reads the word form alone — the lemma
from pymorphy3's best parse with no context, minus its citation ending by part of
speech (`книга` → `книг`), the ending whatever follows the stem — gives
`книг|ами`, `книг|а`, `важлив|их`, `зібра|лися`, `написа|в`. It splits 54% of word
tokens and declines where the stem changes beyond recognition (`ішов`, `дітьми`);
where both rules split a word, they disagree on 9% of boundaries. Its faults:
alternations shift the boundary (`кни|зі`), and indeclinable words are split
(`метр|о`). Whatever rule the tokenizer uses, D7's exposure must then be measured
by the same rule, for every tokenizer.

