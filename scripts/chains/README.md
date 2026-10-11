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
| `guard.sh` | not a chain: sourced by every chain so it cannot run twice | §33.30 |

`batches/` holds the input of the earlier batch runner
(`scripts/ops/run_batch.py`), the predecessor of chains.
