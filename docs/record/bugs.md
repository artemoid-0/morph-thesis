# Bugs found and fixed — a chronological record

> **Kind:** record — append-only. **Authoritative for:** every bug that affected, or
> could have affected, a result: what it was, how it was found, what it changed.
> Later bugs (from 29 September) are recorded in the journey and in
> [`docs/GAPS.md`](../GAPS.md) (G-63 onward; G-72, derived codes read across corpus sizes,
> §33.55; G-73, a constant null that was a lexicon lookup, §33.56).

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

Bugs 12, 14 and 15 were caught by numbers that were *impossible* rather than
merely surprising. **Alignment logic is the recurring failure mode in this
codebase.**
