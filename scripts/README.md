# Scripts

> **Kind:** reference — every script, one line each. Run from the project root, e.g.
> `python scripts/data/check_cache.py --help`. **active** = in current use; **historical** =
> answered its question and is kept so the answer can be reproduced.

Scripts that read run records read them through one module, `morphlm/runs.py`: every record
under `--results` (default `results/`; `--roots`, `--results-dir` and `--root` are older
spellings), keyed by cell and seed; with `--machine second`, the second machine's records
wherever they are — its own runs in `results/`, its twins' copies in `results/audits/device_*/`
with their models in `results/audits/device_twins_models/`. Runs are paired by seed within one
machine, and the seeds left out are named (FINDINGS, method point 11). Findings are written by
`morphlm/record.py` to `results/audits/<script>_<UTC timestamp>.json`.

## `data/` — build and check caches and derived codes

| script | what it answers | status |
|---|---|---|
| `cache_index.py` | What is in data/cache/, and what would a given cell reuse? | historical |
| `cache_pack.py` | Keep a cache's streams packed, losslessly (zstd; each array checked by SHA-256 before its original goes to the trash), and unpack a stream when it is needed. Nothing reads or rebuilds a packed stream: `morphlm/data/stream.py` stops and names it. | active |
| `check_cache.py` | The health of a built cache, in any language. | active |
| `corpus_info.py` | Look at a dataset before it becomes a corpus: splits, sizes, fields, samples, estimates. | active |
| `fiction_full.sh` | UberText fiction's two conditions at fraction 0.4, built from the public source on a machine that has none of it; CPU only. | historical |
| `make_minimal_pairs_uk.py` | Ukrainian minimal pairs beyond subject–verb agreement, from the held-out test split: an adjective agreeing with its noun in case, number or gender, and a noun in the case its preposition governs; the changed form unambiguously wrong by the dictionary; MultiBLiMP's table format, with a stratified sample for a native speaker's check; `--refine` drops the check's two kinds of error from a built table (an adjective form that is a noun too, a noun of two lexemes). | active |
| `derive_lexical_codes.py` | A strictly causal version of a contextual feature; and the controls derived the same way — the constant null, frequency bands, a shuffle, and `--recode-pos` (G-38: prepositions read from their form, `ucaseprep`). | active |
| `derive_causal_pos.py` | `poscausal`: part of speech from a strictly causal tagger, every word tagged from its left context only. | active |
| `derive_causal_ud.py` | `ugendercausal`, `unumbercausal`: Ukrainian gender and number from a strictly causal tagger, for every given tokenizer's stream; paragraphs decoded from the stream (G-75). | active |
| `derive_prefix_codes.py` | Build `mprefix` codes from the cached token stream. | historical |
| `preflight_level2.py` | Check everything that could void a Level-2 run, BEFORE the multi-hour cache build. | historical |
| `preflight_rebuild.py` | The gate to pass BEFORE the multi-hour cache rebuild. | active |
| `prepare_condition.py` | The data for one corpus × size × tokenizer: pilot, check, tagging, derived facts, final gate; resumable. | active |
| `rederive_constants.py` | Derive every constant null again from the records — one class for every word (G-73); train codes must not change. Resumable. | active |
| `train_tokenizer.py` | Train a tokenizer for any corpus, size and algorithm (`bpe-endings`: BPE that keeps endings apart, by the corpus language's rule — Ukrainian `uk_endings_v1`, English `en_endings_v1`); measure what it does to words. | active |
| `prepare_ubertext_fiction.py` | UberText 2.0's fiction subcorpus as documents of paragraphs (`external/ubertext/fiction.jsonl`, the `ukfiction` corpus): texts split at their separators, hard-wrapped ones unwrapped (journey §33.85). | active |
| `ukwiki_stats.py` | Size the Ukrainian Wikipedia dump and choose the train fraction (docs/reference/ukrainian_pipeline.md, block U0). | active |

## `eval/` — score trained checkpoints

| script | what it answers | status |
|---|---|---|
| `blimp_eval.py` | BLiMP minimal pairs on existing checkpoints, paired by seed; `--machine second` scores the second machine's runs. | active |
| `blimp_item_split.py` | Is a BLiMP gain agreement, or a shifted preference? From a blimp_eval record's per-item margins. | historical |
| `multiblimp_eval.py` | MultiBLiMP's Ukrainian subject–verb agreement pairs on saved checkpoints, paired by seed; the scoring path must first reproduce the cached ids and codes of validation paragraphs; `--machine second` scores the second machine's runs; `--data` scores another table of the same shape, such as the noun-phrase and government pairs. | active |
| `checkpoint_survey.py` | What is in results/, and which checkpoints still load? | historical |
| `evaluate_split.py` | Score saved checkpoints on a HELD-OUT split. | historical |
| `lookahead_check.py` | How much of the word-identity gain is within-word lookahead? | historical |
| `probe_dep.py` | Is the injected fact ACTUALLY IN the representation? | historical |

## `analysis/` — turn run records into tables, profiles and figures; re-score saved models

Most read the run records only; `probe.py`, `probe_words.py`, `loss_by_kind.py` and
`compare_buckets.py` load saved models and score them again.

| script | what it answers | status |
|---|---|---|
| `compare_buckets.py` | Does context freezing help RARE tokens, or everything? | historical |
| `demo_figures.py` | The figures of `docs/README.md`, each computed from raw run records (the thesis's own are `thesis_figures.py`'s). | active |
| `device_twins.py` | Do the two machines give the same results? Every run repeated on the other machine, its difference along training, and chosen effects per seed on each machine. | active |
| `thesis_figures.py` | The thesis's figures in Ukrainian and English (`exports/figures/uk/`, `en/`): no titles, 16 cm at 300 dpi; a figure with planned runs missing is not drawn. | active |
| `export_results.py` | Turn results/ into thesis material; effects paired by seed within one machine. | active |
| `exposure_axis.py` | Does case gain less where the tokenizer exposes endings, at the same point of learning? Each tokenizer's gain over the empty slot (or the null) along training, at common reference levels, per width, within one machine; with `--feature empty`, the empty slot's gain over the null instead; `--warmup` for runs with a longer warm-up (`_wu<steps>` in the label). | active |
| `learning_law.py` | A feature's gain against how much its null has learned (D1). | active |
| `loss_by_kind.py` | Where a feature's gain sits, by what each predicted token is — a word's first token or a later one, holding an inflectional ending or not — re-scoring saved models token by token (each must reproduce its record); the tokenizers' difference by word class; `--previous` places one feature's lead over another — or over the null — by the word the feature describes (G-38, G-77), with `--unseen-lexicon` the training lexicon deciding which words are unseen (G-82); `--machine second` reads the second machine's runs and models. | active |
| `bounds.py` | The thesis's two tables of effects recomputed from the run records, each row checked against its printed mean and standard error: 95% t-intervals, one-sided 95% bounds, t-test p and the minimum detectable effect at their seeds; writes a `bounds` audit record | active |
| `paired_cells.py` | One cell against another, paired by seed, each pair on one machine. | active |
| `probe.py` | When a model comes to know a feature: a linear probe over checkpoints (D2); `--previous` probes for what a causal feature carries, the previous unit's class; `--machine second` probes a second machine's twin by its id. | active |
| `probe_axis.py` | Does a probe's accuracy line model widths up on the law? Pairs each probed null's accuracy with the feature's gain at the same steps (D2); with several probed seeds, compares at each and on their averaged accuracy, finding a second machine's probed twins by id. | active |
| `probe_words.py` | What a model builds for itself: a linear probe reads each word's case, number and gender in context from final checkpoints, at the word's last token (`--after N`: N units later), with case split by whether the form alone decides it; cells paired by seed (journey §33.84). | active |
| `steps_saved.py` | What a feature's early help is worth in training steps. | historical |
| `tokenizer_table.py` | D7's table from the run records: each feature's gain under each tokenizer, paired by seed; tokenizers against a reference (null against null — at the end and along training — and gain against gain); gains at a step and at matched null BPB, per seed. | active |
| `tokenizer_widths.py` | The tokenizer's lead across model widths, at matched null BPB (D8). | active |
| `temporal_profile.py` | When during training does a feature's gain appear, against the plain baseline (journey §32.2)? Superseded by `learning_law.py`'s same-shape nulls. | historical |

## `audits/` — one question each — most answered and kept to reproduce the answer

| script | what it answers | status |
|---|---|---|
| `category_bits.py` | Where does each pipeline SPEND its bits? | historical |
| `check_fold.py` | Does coarsening actually reach the model? On real data. | historical |
| `codes_audit.py` | Which validation codes did each run read — its own stream's, or another corpus size's (G-72)? Are the constant nulls constant (G-73)? With `--score`, what each did to the saved models' BPB. | active |
| `collision_audit.py` | The RIGOROUS bits-per-byte correction. | historical |
| `coverage_audit.py` | What fraction of the corpus does each pipeline ACTUALLY encode? | historical |
| `coverage_economics.py` | Is the Q1 deficit COVERAGE or SEGMENTATION? | historical |
| `dep_causality_check.py` | How much does a `dep` label depend on LATER words? | historical |
| `dropout_masks.py` | Do two GPUs draw the same dropout masks, and compute the same numbers? | active |
| `ending_gold_check.py` | Does the ending rule cut where a morpheme dictionary does? (citation forms of two Ukrainian dictionaries, read from a clone in the gitignored `external/`) | active |
| `morph_causality_check.py` | How much do tagger-derived features depend on the words to their RIGHT? By word class with `--by-type` (any feature with `--by-type-attrs`); text from a cached stream with `--stream`. | active |
| `multiroot_audit.py` | How many MorphoLex entries did the multi-root bug affect? | historical |
| `oov_audit.py` | What falls through to character fallback, and why? | historical |
| `reconstruction_audit.py` | Is the morpheme token stream information-preserving? | historical |
| `stream_roundtrip.py` | Does every tokenizer's cached stream decode to the same text, byte for byte? The check bits per byte across tokenizers rests on (G-75). | active |
| `tagger_gold.py` | The tagger's case against UD Ukrainian-IU's gold labels, where context decides: the tagger in context, the form's majority case from the training lexicon and the dictionary's ambiguity, over all words, by part of speech and where the tagger and the form part; writes a `tagger_gold` audit record | active |
| `verify_channels.py` | Assert that a surface-channel ablation did what it says. | historical |

## `ops/` — run, watch, pack and verify experiments; keep the documents honest

| script | what it answers | status |
|---|---|---|
| `backfill_results.py` | Recover the derivable provenance for runs that predate it. | historical |
| `refresh_exports.sh` | Bring `exports/` up to date with `results/`: every table and figure; run after every chain and every result. Every step runs; `exports/REFRESHED.txt` is written only when every step succeeded and every file was redrawn (or is on the script's `NOT_DRAWN` list), otherwise it prints "exports/ NOT refreshed", names the failed steps and stale files, and exits 1. | active |
| `chain_status.sh` | Where every chain stands, in one command — and the runs finished in the last day, flagging duplicate seeds; changes nothing. | active |
| `cells_ready.py` | Does this machine hold every cache a run needs? A chain's readiness guard. | active |
| `check_claims.py` | Verify every number in the documents against results/. | active |
| `check_docs.py` | Keep the documents honest: links, names, citations, kinds, numbers; the journey's part ranges, the results record's § citations, cited record names and run ids, the predictions tally, and that every script is in an index. | active |
| `fit_check.py` | Will a model fit, and how long will its run take? Measured, over a grid of sizes. | active |
| `project_health.sh` | Diagnose a PyCharm project that shows no files. | historical |
| `quarantine.py` | Move invalid runs out of results/, keeping the record. | active |
| `run_batch.py` | Run a block of experiments unattended, safely. | historical |
| `run_ids.py` | The run id a command line would produce, without running it; `--done` tells a chain whether a cell's seed is already finished, `--bpb` what it scored. | active |
| `run_times.py` | How long runs of a kind took, from their records — the source of every estimate in a queue review. | active |
| `software_size.py` | The software's size from HEAD's tree: Python modules and shell scripts by part, their lines and lines of code (no blanks, comments or docstrings), and the tests; writes a `software_size` audit record | active |
| `work_size.py` | The work's size from the run records: trained models, GPU-hours and training tokens, by machine (twins and September records without a GPU name counted apart, then folded in); writes a `work_size` audit record | active |
| `slim_codes.py` | One fact of a stream's codes in a file of its own, for a machine that holds only what its runs read. | active |
| `smoke_model.py` | Build each cell's REAL model and push one batch through it. | active |
| `tagger_record.py` | Which spaCy tagger, at which version, tagged each corpus's caches? From the installed packages and the earliest evidence of each corpus's caches (G-74). | active |
| `verify.py` | Run this first, before spending GPU time: every module of `morphlm/` (found, not listed), every suite in `tests/`, the model's guarantees, and whether the cached streams resolve from the recorded configurations. | active |

## `chains/` — night and day chains

Each chain is a dated record of what ran, and stays runnable. Index: [`chains/README.md`](chains/README.md).
`chains/lib/chain_v1.sh` is the body a chain written after 19 October sources instead of copying it: the preamble with the guard and the log, the helpers, one seed-piece runner (twin, sanity and pair checks), the wait for a training run, and the PLAN and STOP ending. A version is frozen once a dated chain sources it; a change goes into the next version, beside it. `tests/test_chainlib.py` drives it with the stub `python`.

## `rehearsal/` — testing a chain before it runs

Toy data and a stub `python` that records calls and fails on demand; see
the chain scripts' rehearsals (`scripts/chains/README.md`). `make_toy_data.py`
writes a toy WikiText, `make_toy_ukwiki.py` a toy Ukrainian Wikipedia from the UD_Ukrainian-IU
treebank's real sentences; the stand-in `datasets` package serves both, and any other corpus from
`$REHEARSAL_DATA`. A rehearsal runs from a copy of the code in a folder under `scratch/`, never in
`data/cache/` or `results/`. `tests/test_chainlib.py` drives the chain library with this stub, in a
temporary project.
