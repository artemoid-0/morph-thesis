# Chains

> **Kind:** record — each chain is a dated script of what ran, kept runnable
> (its paths are current). What each found is in the
> [journey](../../docs/record/journey/README.md). A new chain sources `guard.sh`,
> so it cannot run twice, and is rehearsed before it runs unattended, with the stub
> `python` of `scripts/rehearsal/`.

| chain | what it ran | journey |
|---|---|---|
| `smoke_2026-09-26.sh` | smoke checks on the GPU before the night of 26 September; stops at the first failure | §33.5 |
| `night_2026-09-26.sh` | two lanes: the cache rebuild adding inflectional facts (CPU) beside English feature runs (GPU) | §33.5–§33.9 |
| `night_2026-09-27.sh` | rebuilt after the null: the null at seeds 1–3, frequency band, slot widths 16–64 | §33.10–§33.12 |
| `night_2026-09-28.sh` | the additive family at seeds 2–3; why the zero slot works (position tests) | §33.16–§33.19 |
| `night_2026-09-29.sh` | two tracks: the Ukrainian train cache (CPU) beside English seeds 2–3 (GPU); then the first Ukrainian family, gated | §33.23–§33.24 |
| `night_2026-09-30.sh` | the Ukrainian family at seeds 1–3; English part of speech at seeds 2–3 | §33.30 |
| `night_2026-10-01.sh` | follow-on: the little-data caches and runs; English inflection at seeds 2–3 | §33.29, §33.31 |
| `day_2026-10-01.sh` | the controls before the demo: all data at 10k steps, lexical case at little data, ~38M tokens, baselines | §33.31–§33.32 |
| `night_2026-10-02.sh` | the first programme night: English at a 10k budget and the switch at seeds 2–3; English at 5k and 20k budgets; Ukrainian at 20k | §33.36 |
| `day_2026-10-02.sh` | follow-on: C1 at seeds 2–3; English features at a 10k budget; Ukrainian case at 5k | §33.37 |
| `night_2026-10-03.sh` | four directions: G-68 and shedding reliance (D3); model width (D1); the architecture effect against the schedule (D6) | §33.40–§33.41 |
| `night_2026-10-04.sh` | the 256-wide model at a 20k budget, seeds 1–3; English features at a 20k budget | §33.43 |
| `screen_2026-10-05.sh` | a 3-hour screen across directions: the auxiliary target (D4), RoPE (D6), weaning (D3), inflection at 20k (G-70) | §33.47 |
| `day_2026-10-05.sh` | day block E: the probe pilot (D2) and the tiled slot (D1); starts after the screen | §33.48 |
| `long_2026-10-05.sh` | the screen's promising tests at seeds 1–3; the probe at 20k; D1's second seed | §33.49 |
| `long_2026-10-06.sh` | 10 resumable pieces: what a token-independent input does under RoPE (D6); the switch's and the teacher's mechanisms by the probe (D3, D4); features at 20k, seeds 2–3 (D1); the 256-wide baseline | §33.51 |
| `night_2026-10-06.sh` | 6 resumable pieces: the window-start test (G-71); the Ukrainian nulls re-run on the constant (G-73); the four new tokenizers at seed 0 (D7), with a CPU track preparing their conditions | §33.57 |
| `long_2026-10-07.sh` | 5 resumable pieces in priority order, time-boxed by `END_BY`: BPE 17k with endings apart at seeds 0–3 (D7's sixth tokenizer); D7's seeds 1–3 under the other four tokenizers and BPE 17k's word form | §33.64 |
| `long_2026-10-08.sh` | 4 resumable pieces with guards: the endings-apart and BPE 17k nulls at a 20k budget and at width 256; case under both at 20k; BPE 32k at seed 3; a CPU track preparing the endings-apart condition at 0.2 | §33.67 |
| `long_2026-10-09.sh` | ran 9 October with `PIECES="0 1 2 3" PIECE3=case` (§33.76): a reproduction check, then the quiet candidates in rank order — the tokenizer and case at 151M tokens and 50k steps; the five runs left; width 256 at 20k; the Ukrainian re-runs; case at width 256; D3 at 50k — with `PLAN=1` to show what given hours buy | §33.73 |
| `short_2026-10-09.sh` | ran 9 October (§33.76): G-38's control, case with prepositions read from their form (`ucaseprep`), 10k, seeds 0–3, two tokenizers | §33.74 |
| `long_2026-10-10.sh` | prepared after the chain of 9 October: a reproduction check, then G-77 — case's word form and case with prepositions read from their form under endings apart at 151M tokens and 50k steps — then BPE 32k's last seed, width 256 at 20k, the re-runs, case at width 256, D3 at 50k | §33.77 |
| `long_2026-10-11.sh` | prepared 11 October: how far the tokenizer's gain travels — English with endings apart (plain baselines at 10k; 20k as a named follow-up) and UberText fiction (both tokenizers' nulls at 0.1, 10k), with readiness, smoke and seed-0 sanity guards | §33.85 |
| `scale_2026-10-14.sh` | does the tokenizer's gain keep growing with model size: both Ukrainian nulls (BPE 17k, endings apart) at width 512, 30M parameters, ~38M tokens, 10k, seeds 0–3, a seed's pair from identical initial weights, with readiness, smoke and width-sanity guards and `FILL`; with `STEPS=20000` the same at 20k — seed 0 on 14 October, seed 1 on 15 October (§33.100), seeds 2–3 through `night_2026-10-16.sh` | §33.93–§33.96 |
| `night_2026-10-16.sh` | the second machine's first night (RTX 5090), time-boxed by `END_BY`: the items of the thesis's list of 15 October whose comparisons are all new runs — width 512 at 20k, seeds 2–3 (through `scale_2026-10-14.sh`); English part of speech switched off from the start and at 5,000, 50k; fiction at full training; Ukrainian case switched off the same two ways under endings apart, 20k; English plain baselines at 50k, seeds 4–7 — with readiness (`cells_ready.py`), pair-sanity and GPU guards | §33.101 |
| `night_2026-10-17.sh` | the second machine: twins of the thesis machine's runs — the Ukrainian null and case under endings apart at 20k, seeds 0–3, and both English plain baselines at 50k, seeds 0–1 — with the empty slot beside them (G-78), each twins piece checked against the thesis machine's run (0.01); BLiMP on English seeds 4–7 and MultiBLiMP on width 512 at 20k; fiction at full training once its data was verified; the null's twin and the empty slot under BPE 17k | §33.103 |
| `tulpar_2026-10-17.sh` | the thesis machine's day beside it: the empty slot under endings apart at 20k, seeds 0–3 (G-78 on this machine), and item 2 — part of speech from a causal tagger (`poscausal`) at 10k and 20k, seeds 0–3 — each piece's first run checked against this machine's run (0.05) | §33.104 |
| `tulpar_followup_2026-10-17.sh` | the thesis machine's afternoon, its pieces chosen by `PIECES`: dropout's masks on this GPU; BPE 17k's null with dropout off at 10k, seeds 0–1; the empty slot at full training (151M tokens, 50k) under endings apart and under BPE 17k; item 7 (`ucaseprep` under BPE 17k at full training); item 8 (frequency band and part of speech from the word form, English, 50k, seeds 2–3) | §33.105 |
| `second_followup_2026-10-17.sh` | the second machine's follow-up: MultiBLiMP on fiction's nulls at full training; dropout's masks on this GPU; the null with dropout off as twins; width 512's twins at 20k, seed 1; item 5's probes — part of speech switched off and as an auxiliary target at 20k, seeds 0–3, with a checkpoint every 1,000 steps, each run probed; G-78 at 10k (optional) | §33.105 |
| `second_2026-10-18.sh` | one long chain on the second machine, about 15 hours unattended: G-78 across tokenizers (Unigram 17k, BPE 4k, 8k, 32k) and corpus sizes (~1.5M and ~7.6M tokens) at 10k, the null and case as twins beside the empty slot; the width probes (item 5: the English null at widths 128 and 256, 10k and 20k, probed); item 7 at 151M tokens and 50k; item 8; then the tables | §33.112 |
| `second_followup_2026-10-18.sh` | the exposure comparison at twice the budget: case under BPE 17k at ~38M tokens and 20k, seeds 0–3, as twins of the thesis machine's runs | §33.115 |
| `second_2026-10-19.sh` | item 4 of the thesis's list of 19 October: width 512, ~38M tokens, 10k, seeds 0–3, under BPE 17k and endings apart — the null (twins of the thesis machine's runs), case and the empty slot | §33.117–§33.118 |
| `second_2026-10-20.sh` | item 1 of the thesis's list of 19 October: the tokenizer's gain at width 1024 (85.3M parameters), ~38M tokens, seeds 0–3, the null under BPE 17k and under endings apart — the 10k half, then the 20k half until END_BY; the first chain on `lib/chain_v1.sh`. Ran 20 October: the 10k half and seed 0's 20k pair; started again, it runs the 20k pairs of seeds 1–3 | journey §33.119–§33.120 |
| `second_2026-10-21.sh` | after item 1's chain lets go of its lock: MultiBLiMP on width 1024's 20k nulls; item 3, English part of speech never fed and off at 5,000, 100k, seeds 0–3; item 6, Unigram 17k at width 512, the null, case and the empty slot, 10k, seeds 0–3 | journey §33.123 |
| `second_item2_2026-10-21.sh` | item 2 of the thesis's list of 19 October, after the chain of 21 October lets go of its lock: Ukrainian gender and number, ~38M tokens, 10k, seeds 0–3, under BPE 17k and endings apart — in context, from the word form, and from a causal tagger | journey §33.125 |
| `second_2026-10-22.sh` | the text's request of 22 October, item 1 — G-81's recipe question: the nulls at width 1024 and at width 512, ~38M tokens, 10k, seeds 0–3, under BPE 17k and endings apart, with a 1,000-step warm-up (`_wu1000`); then MultiBLiMP on each width's eight | journey §33.130 |
| `second_full_2026-10-22.sh` | item 3 of the text's request of 22 October, after the warm-up chain lets go of its lock: Ukrainian gender and number at full training (151M tokens, 50k) under endings apart, seeds 0–3 — in context, from the word form and from a causal tagger, and the null (its first seed a twin of the thesis machine's); then G-82's split of each feature's causal rest | journey §33.131 |
| `guard.sh` | not a chain: sourced by every chain so it cannot run twice | §33.30 |
| `lib/chain_v1.sh` | not a chain: the body the chains of 16–19 October repeated — the preamble with `guard.sh` and the log, the helpers, one seed-piece runner (twin, sanity and pair checks), the wait for a training run, the PLAN and STOP ending — sourced by chains written after it; frozen once a dated chain sources it, a change going into `chain_v2.sh`; tested by `tests/test_chainlib.py` | — |

`batches/` holds the input of the earlier batch runner
(`scripts/ops/run_batch.py`), the predecessor of chains.

`lib/` holds the chain library, by version. A chain written after 19 October sources
`lib/chain_v1.sh` (which sources `guard.sh`) instead of carrying its own copy of that body; a
version stays as it is once a dated chain has sourced it, and the chains before it keep their own
copies.
