# docs/GAPS.md — Gap Register

> **Kind:** claims — every question that could change a claim, and where it stands.
> **Authoritative for:** what is open, closed, out of scope or moot.

Current state only.
Every closed gap is listed below with where its resolution is recorded; how each was reached is in the journey.

Ranked by impact on claims.
A gap is closed when a check that could have failed, did not.
The register is empty when no open gap could change a claim stated in FINDINGS.

---

## Open — could change a claim

| # | gap | status | cost |
|---|---|---|---|
| G-81 | At width 1024 two of four endings-apart nulls (seeds 1 and 2) fall behind their siblings from about step 1,500, at 10k and again at 20k, and regain only part: the tokenizer's gain at those seeds is −0.0004 and +0.0042 at 10k, +0.0033 and +0.0058 at 20k (seeds 0 and 3: +0.0127, +0.0103; +0.0115, +0.0093). No evaluation shows a spike, at 500 steps or at 100: the fall is a slower stretch over steps 1,500–3,000, and the grammatical gain does not follow it (MultiBLiMP +4.6 to +6.3 at every seed). Not known: whether another dropout draw would fall too, and whether the training recipe, the same at every width (a 100-step warm-up, peak learning rate 3e-4), is at fault at this width. It decides whether FINDINGS' "depends on the run" at 85M parameters is the model's or the recipe's. | open, narrowed (§33.122, §33.124): no spike at 100-step resolution — seed 1's endings-apart null falls at every evaluation, and a rerun on the same machine repeats its 10k run to 3 × 10⁻⁵ BPB; the 20k runs repeat the fall at both seeds (P272). On this machine the fall is the seed's, not run-to-run noise. Left: would another dropout draw fall too (a twin on the thesis machine), and would another recipe at this width (a longer warm-up)? | seeds 1 and 2 at width 1024 under both tokenizers with a longer warm-up, 10k: 4 runs, ~3 h on the second machine; their twins on the thesis machine (hours, if width 1024 fits there) |

The research programme's open questions are in [`DIRECTIONS.md`](DIRECTIONS.md).

## Open — would not change a claim

| # | gap | status | cost |
|---|---|---|---|
| G-80 | The morpheme arms' vocabulary (`data/our_final_vocab.csv`) lists 11 tokens twice (ten single letters and `nan`): 17,151 rows, 17,140 distinct. So `UNK_ID = len(token2id)` (17,141; `morphlm/pipelines/base.py`) is also the id of a byte-fallback token, `<bf6>`. Harmless as far as checked: an unknown token never survives `_byte_fallback`, so no stream holds it, and the model's 17,155 embedding rows cover every id. Found by the audit of 19 October. | open; Q1 is settled and no stream is affected; a fix would change the morpheme arms' token ids, which their records depend on | none: leave it, unless the morpheme arms are rebuilt |
| G-79 | Three of the thesis machine's saved models were overwritten by their second-machine reruns: the BPE 17k null at ~38M and 10k, seeds 0–2 (`e9f3f153ec20`, `6f214dfa186f`, `fe50426ea6ef`), the calibration of 16 October (journey §33.101), whose reruns share their run ids and were taken for new runs when the second machine's models were copied over on 18 October. Their records are intact and every number stands; a re-scoring of the three would stop on its check, since they score as the second machine's runs. The copy now keeps the second machine's models apart (`results/audits/device_twins_models/<id>/`), tells runs apart by the GPU each record names, and never overwrites a file. | open; to be re-made when there is time (journey §33.116) | ~36 min on the thesis machine, which reproduces a run; each `model.pt` copied into its folder once its bits per byte matches the record |
| G-24 | Why is causal `mhash` so unstable across seeds? 0.0094 spread alone and 0.0053 with `mstruct`, against 0.0003 for `mlemma`+`mstruct`. A 16k-class code of the previous word's surface form may leave many rows rarely trained. | open; `mhash` adds nothing to structure whichever way it resolves | seeds, or an embedding-norm check |
| G-06 | Entity vocabulary was tuned once and never revisited. | open; affects Q1 coverage only, and Q1 is settled | ~1 day |
| G-37 | Section headings are in every training stream. `build_stream` strips each line, then tests `startswith(' =')` — which a stripped line can never satisfy — so WikiText headings (`= Robert Boulter =`) were never skipped, whatever the intent. | open; identical in every run, so no comparison is affected. Fixing it would change every cache key; left as is and recorded | trivial, but invalidates caches |
| G-38 | Tagger-derived features are not strictly causal even when shifted: the tagger's window reaches ~4 tokens right, so the previous word's `pos` / `mfeat` / `magr` / `minfl` — and Ukrainian `ucase` — can depend on the word being predicted (§33.3). | open; every claim on a tagger feature is stated beside its word-form variant, which reads no context (`mfeatlex`, `poslex`, `ucaselex`). Measured by `morph_causality_check.py` — 7–8% of English tags, 5.0% of Ukrainian case values, the latter concentrated on prepositions, whose case is the next noun's (journey §33.21) — and in trained models, with prepositions read from their form (`ucaseprep`), in both languages and to full training: FINDINGS, [what the sentence adds over the word form](FINDINGS.md#what-the-sentence-adds-over-the-word-form) (journey §33.76, §33.81, §33.107, §33.114) | measured wherever a claim needed it |
| G-76 | The word-form variants (`ucaselex`, and so `poslex` and `mfeatlex`) give a word unseen in train `none` — in training the class of a word without the feature (a caseless word; for part of speech, a non-word), the null's fault of G-73 in another control. | open; measured for Ukrainian case at ~38M and 10k (journey §33.74): on targets after such words (1.9–3.2%) the word-form model ends −0.0005 to −0.0007 below the null (0/4 each), and they hold 21–38% of contextual case's lead over the form. It makes every word form's share of a feature's gain a lower bound, so the claims it supports ("it is case, not only the tagger's right context") only gain; the desk split separates it | a class for unseen words (a new derivation, and runs), or comparisons on seen words only — `loss_by_kind.py --previous` |

---

## Scope — declared, not chased

Limitations to state in the thesis.
Closing any of these would be a different project: a hole in a claim is closed, a question outside its scope is declared.

| # | limitation |
|---|---|
| G-07 | **No downstream task.** Standard benchmarks are uninformative at 5M parameters. The evidence beyond bits per byte is the probe suite (journey §25, §29.11, §30.6b) and minimal pairs: BLiMP in English, MultiBLiMP's subject–verb agreement in Ukrainian (journey §33.79). |
| G-11 | **No tokenizer-free baseline** (ByT5, CANINE, Charformer). The comparison is between tokenizer families at fixed architecture. |
| G-12 | **No run reached convergence.** All stopped still improving at the 50,000-step ceiling, so every comparison is at fixed compute. |
| G-13 | **Small models, two languages: widths 64–512, 1.8M to 30M parameters, most cells about 5M** (`docs/reference/apparatus.md`), in English and Ukrainian. MorphPiece uses 117M, Hou et al. 132–979M. Hou et al. find English the weakest case for morphological approaches, which makes a small effect in English *expected* rather than disappointing. *(Until 19 October: "English only, 3.4–5.2M parameters".)* |
| G-14 | **Residual collisions**: 0.0008 BPB over 0.74% of word occurrences (full split), from dialectal spelling, homographs and entity spans. |
| R-11 | **Effect sizes are paired means with a standard error and the count of seeds that share the sign, not significance tests** (results §0, "Noise": a floor from two seeds was one lucky sample of the spread). Four seeds for most headline cells, one to eight elsewhere; each claim names its count. *(Until 19 October: "against an empirical noise floor … Two seeds for most headline cells, three for some".)* |
| — | **Pre-repair Q1 cells have no checkpoints** (checkpointing postdates them), so they cannot be scored on held-out test. The post-repair half is test-verified. |

---

## Closed

| # | gap | resolution | recorded in |
|---|---|---|---|
| G-78 | Does an empty slot beat the same-shape null in Ukrainian, and does that confound the cross-tokenizer claim "exposed endings make case more useful"? | It depends on the tokenizer and the model's size: the claim and method point 10 are restated against both references — FINDINGS, [the same-shape null is not neutral](FINDINGS.md#the-same-shape-null-is-not-neutral) and [exposed endings](FINDINGS.md#exposed-endings-case-gains-less-for-a-stretch) | journey §33.106–§33.110, §33.113, §33.116, §33.118 |
| G-77 | Case's full-training gain under endings apart (+0.0030 ± 0.0009, 4/4) might be the tagger's right-context leak (G-38): over half of it sat after prepositions, where under endings apart the leak can reach the noun's ending, a token of its own (journey §33.76) | measured at 151M tokens and 50k steps, seeds 0–3: with prepositions read from their form case keeps +0.0015 ± 0.0010 (3/4), the word form alone +0.0016 ± 0.0009 (3/4) — +0.0036 ± 0.0010 (4/4) over the word form's gain under BPE 17k; the leak is +0.0015 ± 0.0001 (4/4), the whole of the sentence's lead over the word form. FINDINGS restated: half the gain is the word's own case, half the leak, and context without the leak adds nothing at full training *Annotated 19 October: against the empty slot, the better reference under endings apart, case keeps +0.0010 ± 0.0007 (3/4) at full training, about the leak, and without the leak ends level with it (journey §33.110).* | journey §33.81, §33.110 |
| G-75 | The tokenizer's 1.6% (BPE that keeps endings apart against BPE 17k) holds only if both streams are the same text: a stream that dropped or altered characters would score fewer bits over the same byte count | `stream_roundtrip.py` on the project's caches: on the validation split at 0.05 all six tokenizers' streams decode to the 1,989,866 bytes recorded at build time, and to one SHA-256 | journey §33.65, §33.66 |
| G-74 | Every run and audit recorded its spaCy model as `en_core_web_sm` whenever that model was installed — every Ukrainian run included — so the Ukrainian tagger's version was never recorded | records now name the tagger used, with its version, and every project tagger installed; the export reads old runs' tagger from their config; `tagger_record.py` on the GPU machine: `uk_core_news_sm` 3.8.0, installed eight minutes before the first Ukrainian cache was checked and never reinstalled, tagged every Ukrainian cache; `en_core_web_sm` 3.8.0 every English one | journey §33.63, §33.65 |
| G-73 | A constant null's held-out codes were a lexicon lookup: every word unseen in train got `none`, a mark the null met only on non-words in training | measured on the saved models (Ukrainian nulls better on the constant by 0.0005–0.0039, English by at most 0.0002); fixed — one class for every word, the rule declared and checked, every constant replayed with train codes unchanged; the re-run nulls reproduce the re-scores to the sixth decimal, so every restated number is exact; early help in Ukrainian +0.0056 ± 0.0005 | journey §33.56–§33.58 |
| G-71 | Under rotary positions every variant that helped also marks each window's first word (the causal shift's sentinel): the gain might be a window-start marker, not the shared vector | without the causal shift — a pure constant, the same initial weights — the flat slot keeps 89% of its gain over the plain baseline (+0.0068) and the vector added 82% (+0.0059), seed 0: the vector; the marker is worth about 0.001 | journey §33.58 |
| G-72 | A corpus's validation split is the same tokens at every fraction, and derived codes were matched by token count and name order across fractions: 20 Ukrainian null runs were scored on the 0.01 fraction's validation null | a derived file is read only by its own stream (loader, derivation, check; a test fails under the old rule); the audit found exactly the 20, every re-scored model reproduced its record, and on their own codes they moved by 0.0010–0.0019; restated with G-73 | journey §33.55–§33.56 |
| G-66 | Lexicon files were named per tokenizer, not per corpus size: deriving at one size overwrote the lexicon built at another | outside WikiText, lexicons are named per corpus and size (`lexicon_<fact>_<tokenizer>_<corpus><fraction>.npz`); loaders try that name, then the historical one; derived codes unchanged (16 of 16 files byte-identical on the toy corpus); pinned by a test | journey §33.53 |
| G-70 | Inflection's 20k-budget gain at null BPB 1.55 sat +0.0020 above its 50k value on two seeds | over four seeds +0.0070 against +0.0069: seed noise; every feature agrees across three schedules | journey §33.49 |
| G-69 | The September 10k-budget baseline (seed 0) gave the architecture effect +0.0003 against +0.0043–0.0050 for current-code seeds | rerun under current code: bit-identical at all 21 evaluations — seed variation, not code; over four seeds +0.0035 ± 0.0011; the September copy quarantined as a duplicate (`sweep_duplicate`) | journey §33.43 |
| G-68 | A feature switched on mid-training meets an untrained table and a halved learning rate | with 15,000 steps left the late feature recovers 54–104% of the kept gain (4 seeds): slow integration, not a critical period; the 9% at a 10k budget was the schedule | journey §33.42 |
| G-67 | The little-data gain could be right-context leakage or short training | not leakage (word form alone keeps 65–73%); it IS training progress — +0.0044 with all data at 10k steps; the claim restated as 'before the model has learned it' — *restated against the constant null (G-73): the word form keeps two-fifths to two-thirds, and all data at 10k steps gives +0.0026* | journey §33.32, §33.56 |
| G-62 | Ukrainian case tagged from the right context could carry the next word's case | contextual over lexical case +0.0006 ± 0.0004 over four seeds: the right context buys nothing — *annotated 19 October: that +0.0006 is the tagger's leak itself (+0.0006 ± 0.0001, 4/4, with prepositions read from their form level with the word form): the sentence's context, without the leak, buys nothing* | journey §33.30, §33.114 |
| G-65 | Why an informative code would end below its same-shape null | the single-slot deficit was mostly seed 0 (−0.0009 ± 0.0008 over four seeds); the zero slot's relief +0.0020 ± 0.0015 is within noise — not established | journey §33.24–§33.30 |
| G-64 | Ukrainian paragraphs lost their newline, so headings and list items ran into the next paragraph's first word (the pilot's 97–98% within-word agreement) | rows carry their newline; the key gains `:nl`; reproduced 92.6% → 100.00%; English streams and keys unchanged | journey §33.22 |
| G-63 | Byte fragments of split punctuation were counted as letters, gluing quotes and dashes into words (97–98% within-word agreement on Ukrainian Wikipedia) | a fragment takes its predecessor's class, a lead byte its Unicode block; words = letter runs on all treebank text but a split ×; model and numpy agree; English identical | journey §33.21 |
| G-01 | §23 control-asymmetry paragraph over-interpreted | corrected in place | journey §23 |
| G-02 | `probe_dep.py` and `return_hidden` lost | rebuilt | journey §25 |
| G-03 | legibility probe never run | run on `dep` and `mroot` | journey §25, §26.6 |
| G-04 | `ectx` never run | run; marginal | journey §23 |
| G-05 | seed replication missing on quoted cells | two or three seeds on every headline cell | journey §26, §28.8 |
| G-15 | checkpoint schema drift blocked legacy analyses | `migrate_state_dict`, format 1→2 | journey §24 |
| G-16 | widening the fact inventory orphaned caches | content-based codes reuse | journey §24 |
| G-17 | identity-precision curve | monotonic, continuity bit-identical — **but it measured leak strength**, see G-21 | journey §29, §30.4 |
| G-18 | `mlemma`+`mstruct` vs `mhash`+`mstruct` | tied leakily and causally; **superseded** — identity is not the mechanism | journey §29, §31 |
| G-21 | within-word lookahead in identity features | 86–97% of form-feature gain was leak; causal shift built | journey §30 |
| G-22 | causal `dep` (cross-sentence lookahead) | **largely resolved**: a causal model recovers 78.5% of `dep` labels, so ≤21.5% can leak; residual stated as a limitation | journey §30.6b |
| G-08 | causal prefix-only `dep` | superseded by G-22 | journey §30.6b |
| G-23 | prefix-hash identity | **negative**: no better than the previous-word shift (seed 0: 0.0060 vs 0.0073) | journey §30.8 |
| G-26 | were baselines trained in the same environment as the cells? | **yes**: re-run under torch 2.11.0, seeds 0 and 1 reproduced to 13 decimal places; seed 2's 1.3573 is an unlucky seed | journey §31.7 |
| G-25 | is `mstruct`'s gain morphology, or name status? | **both, equally**: at three seeds, morphology-only and names-only are tied (paired differences −0.0001, −0.0012, +0.0019); each is slightly below the full feature (5 of 6 paired comparisons); morphology-only is positive at every seed against every baseline. Compared cell-to-cell in one environment, so independent of any baseline | journey §31.7 |
| G-10 | routed tokenizer | **estimated, not built**: idealised ceiling 0.60%, a finer sweep could not raise it; the build rule was fixed before the sweep | journey §30.6 |
| R-10 | shared learning rate might favour one arm | measured: both pipelines share the same optimum across seven rates | journey §28.7 |
| G-39 | cache codes built from the requesting run's config: a coarsened run triggering a build would have poisoned the shared cache with folded codes, and BLiMP would have folded a coarsened model's codes twice (never triggered) | `stream.cache_blocks` builds codes as the cache stores them, for both paths; its test fails without it | journey §33.5 |
| G-27 | `comparisons.md` measured the main group against a 10k-step sweep baseline (whichever feature-free cell came last) and compared means over different seed sets against an unpaired floor | each cell paired by seed with its own baseline (label with features → `baseline`); regression test | journey §33.6 |
| G-40 | structure's anaphor-gender gain: agreement or preference? | preference: gains where *itself* is wrong (+35.1), loses where it is right (−46.2); features identical in both sentences, so learned — P13 wrong, BLiMP retracted | journey §33.8 |
| G-41 | `poslex`: does the `pos` tie survive without the tagger leak? | yes — matches `pos` (P11 wrong) | journey §33.9 |
| G-43 | `mstructshuf`: content or any partition? | the shuffle matches structure in size, time course and frequency profile — not the content (P15 wrong) | journey §33.9 |
| G-45 | lookahead split `pos` vs `mstruct` | `pos` all on word-initial targets, hurts continuations; `mstruct` proportional (P14 wrong, reversed) | journey §33.9 |
| G-46 | the null: is part of every feature's gain the architecture? | all of structure's: null +0.0111 ± 0.0025, structure over null +0.0004 ± 0.0005 (4 seeds) — P22 right | journey §33.11 |
| G-47 | `mfreq`: is frequency the ingredient? | no — +0.0007 ± 0.0003 over the null | journey §33.11 |
| G-49 | width of the null slot | nearly flat from 8 to 64 dims (+0.0073 to +0.0114); peak at 32 | journey §33.11 |
| G-50 | `mzero`: learned vector or free dims? | free dims: the zero slot keeps 87% of the null's gain (P25 right) | journey §33.12 |
| G-51 | `mflat`: is the null's word/non-word signal the ingredient? | no — +0.0003 over the null (P26 right) | journey §33.12 |
| G-52 | BLiMP of the null | overall shift is the slot; structure amplifies the *itself* preference on anaphor (P24 wrong by rule) | journey §33.12 |
| G-58 | Is the gain architecture at all? | yes: the baseline's function over eight random starts averages 1.3489 (sd 0.0039); the slotted null beats every start, +0.0097 ± 0.0015 (P34 right) | journey §33.16 |
| G-57 | Literature audit against four fairness conditions | written: no audited Q2 work reports a same-shape null; positive results cluster in morphologically rich languages | `docs/reference/literature_audit.md`, journey §33.17 |
| G-53 | Why the zero slot helps | a clean position channel: without position in the free dims the gain vanishes; position only there keeps half (P27 right, P28 half) — *revised (journey §33.52): the dims need a token-independent signal, which learned position supplied; under rotary positions an empty slot gives nothing* | journey §33.19 |
| G-55 | Additive injection | its own architecture effect: any learned vector added to every input gains ~+0.009 over the matched control; structure −0.0013 against the flat null (P35, P36 wrong about structure) | journey §33.19 |

The article's revision register (C-01 to C-16) and the adversarial review (R-01 to R-12) are retired; every item in them is closed or appears above.

## Moot

Retired because their premise was a measurement artifact, not because they were answered.

| # | gap | why moot |
|---|---|---|
| G-19 | does Level-2 transfer to the morph pipeline? | Level-2's identity effect was within-word lookahead (§30). Transferring a leak tells us nothing. |
| G-20 | where does word identity saturate? | The precision curve measured how informative the leak was, not how useful identity is (§30.4). |

---

## Standing risk — read before touching alignment or boundary code

The recurring failure mode of this project is **something inferred where it could have been looked up, or two things assumed equal without checking.**
At least eleven bugs belong to that family:

| bug | what was assumed | recorded |
|---|---|---|
| region attribution by start offset | a token's start offset marks its word | journey §19 |
| `freq_bucket` sending unseen words to the last bucket | every word has a training frequency | journey §19 |
| BPE context alignment by start offset | ByteLevel does not fold the preceding space | journey §23 |
| coverage audit case folding | source and emitted text are cased alike | journey §28.3 |
| coverage audit byte fallback | a token's text contains the character it encodes | journey §28.3 |
| codes borrowed across token streams | a matching fact list implies matching tokens | `stream.py`, `test_cache_keys.py` |
| coarsening in `index()` | the model reads codes through `index()` | journey §29.4 |
| probe feeding labels as inputs | the probed fact is the one the model reads | journey §29.11 |
| causal shift gated on `n_full` | every word-form feature descends from `MorphRootBlock` | journey §30.5 |
| boundaries from low-cardinality codes | adjacent words rarely share a code | journey §30.5 |
| boundaries from `mprefix` | the reference fact is constant across a word | journey §30.8 |

**ByteLevel BPE folds the preceding space into the token**, so indexing a word annotation by a token's start offset is wrong — scan the span.
**Word boundaries come from the tokenizer** — never infer them from codes.
Any change to alignment or boundary code ships with an assertion on the resulting distribution, not merely a passing test, and a check that is shown to fail on the known-broken case.

This section maps each general lesson to the incident it came from.
