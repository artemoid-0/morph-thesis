# Bugs found and fixed — a chronological record

> **Kind:** record — append-only. **Authoritative for:** every bug that affected, or
> could have affected, a result: what it was, how it was found, what it changed.
> Later bugs (from 29 September) are recorded in the journey and in
> [`docs/GAPS.md`](../GAPS.md) (G-63 onward; G-72, derived codes read across corpus sizes,
> §33.55; G-73, a constant null that was a lexicon lookup, §33.56). *(Rows 16–28, appended
> 19 October 2026, list those of §29–§33 here too, one line each with its source.)*

*Formerly `results_reference.md` §9; the section number is kept.*

## 9. Bugs found and fixed — a chronological record

Useful for a methods or limitations section, since several would have silently
invalidated results.

| # | bug | impact if unfixed | journey |
|---|---|---|---|
| 1 | MorphoLex format-B prefixes lost | 4,941 entries missing prefixes | §7 |
| 2 | multi-root compounds: `re.search` not `findall` | 8,453 entries (12.32%) losing roots | §17 |
| 3 | dead frozen dimensions | 12 of 23 columns constant; 6 distinct vectors total | §14 |
| 4 | `max_steps` doubling as LR horizon | runs of different length incomparable | §16 |
| 5 | no seeding anywhere | results not reproducible, no noise floor | §16 |
| 6 | results keyed without seed | second seed silently overwrote the first | §16 |
| 7 | O(n) per-token category scan | 2.18 ms/token → ~93 h per pipeline | §17 |
| 8 | pandas destroying `na`/`nan`/`null` morphemes | three real morphemes → UNK | §17 |
| 9 | five information channels dropped | 5.32% of corpus unencoded; 6.3% "win" was an artifact | §17 |
| 10 | UNK as a collision sink | non-ASCII punctuation all collapsed together | §17 |
| 11 | collision audit skipping digits/punct | inflated the measured correction ~70% | §17 |
| 12 | region attribution by start offset | BPE reported at 0.1354 BPB on words | §19 |
| 13 | frozen one-hot 4.6× louder than trained slice | confounded trainability with scale | §21 |
| 14 | BPE context alignment by start offset | **78% of the stream would have received `none`** | §23 |
| 15 | `freq_bucket` sending unseen words to the last bucket | would have inverted the frequency analysis | §19 |
| 16 | coarsening folded codes only when the cache was built; the model clamped them | the first identity-precision sweep put up to 99.6% of words in one class (k = 64): it measured whether words kept any identity, not how precisely; 10 runs quarantined (`results_invalid/coarsen_clamp/`) | §29.4 |
| 17 | every BPE fragment of a word carried the whole word's code: within-word lookahead | word-form features read the rest of the word — 86–97% of their gain sat on continuation fragments, 88% of the 5.10% headline | §30 |
| 18 | the causal shift gated on an attribute only root-family blocks carry: `mstruct` never shifted | a run reported as causal still leaked — 0.0172 of its 0.0257 was `mstruct`'s within-word leak; quarantined (`results_invalid/causal_partial/`) | §30.5 |
| 19 | word boundaries inferred from where a code changes, which `mprefix` does within a word: the third boundary bug | the shift moved one token, not one word, and `mstruct` leaked again; a checkpoint re-scored under another boundary rule gave another answer; with row 18, 2 runs quarantined (`causal_partial`) | §30.8 |
| 20 | byte fragments of split punctuation counted as letters (G-63) | quotes and dashes glued into Ukrainian words, 97–98% within-word agreement; stopped by the gate before the first Ukrainian build | §33.21 |
| 21 | Ukrainian paragraphs lost their newline (G-64) | headings and list items ran into the next paragraph's first word: 92.6% within words on heading-rich articles, 5,004 glued words in the pilot | §33.22 |
| 22 | a quarantine rule (`causal_partial`) written for feature runs, met by plain baselines that set the causal flags | nine valid runs, every plain baseline since 2 October, listed for quarantine; caught in the dry run | §33.44 |
| 23 | test files ran a hand-kept list of their tests; four were never added | four regression tests, G-63's word rule among them, guarded nothing while the files printed "PASS" | §33.45 |
| 24 | lexicon files named per tokenizer, not per corpus size (G-66) | deriving at one size overwrote the lexicon built at another | §33.53 |
| 25 | derived codes matched across corpus sizes by token count and name order (G-72) | 20 Ukrainian null runs scored on the 0.01 fraction's validation null codes; on their own they moved by 0.0010–0.0019 | §33.55–§33.56 |
| 26 | a constant null's held-out codes were a lexicon lookup: words unseen in train got `none` (G-73) | the Ukrainian nulls scored 0.0005–0.0039 worse than on a true constant, inflating every gain over them; restated; 34 runs quarantined (`results_invalid/const_unseen/`), row 25's 20 among them | §33.56–§33.58 |
| 27 | records named the spaCy model `en_core_web_sm` whenever it was installed, Ukrainian runs included (G-74) | the Ukrainian tagger's version went unrecorded; established afterwards by `scripts/ops/tagger_record.py` (3.8.0) | §33.63, §33.65 |
| 28 | copying the second machine's models took three reruns that share the thesis machine's run ids for new runs (G-79) | three of the thesis machine's saved models (BPE 17k's null, ~38M, 10k, seeds 0–2) overwritten; the records are intact and every number stands | §33.116 |

Bugs 12, 14 and 15 were caught by numbers that were *impossible* rather than
merely surprising. **Alignment logic is the recurring failure mode in this
codebase.**
