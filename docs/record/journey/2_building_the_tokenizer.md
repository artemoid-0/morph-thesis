# Research journey — Part 2: Building the Morphological Tokenizer

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §5–§10. [← Part 1](1_the_question.md) · [index](README.md) · [Part 3 →](3_the_apparatus.md)

*Six months of design compressed: how the vocabulary was decided, how the
segmenter was built, which bugs mattered, and what the first comparison against
BPE appeared to show.*

*A warning that applies to this entire part: the comparisons reported here were
later found to be **invalid**, because the two token streams being compared did
not encode the same information. Part IV establishes this and reverses the
conclusion. The design decisions remain sound and are the reason the later,
valid comparison was possible at all — but no number in this part should be
quoted as a result.*

---

## 5. The Vocabulary: Why Frequency Is the Wrong Criterion

### The first design, and its failure

The initial vocabulary used a frequency threshold: morphemes appearing at least
5 times in 10,000 training articles were kept, others cut. Named entities
required frequency ≥ 10 and length ≥ 5.

**Dataset choice: WikiText-103** — a standard language-modeling benchmark of
verified Good and Featured Wikipedia articles, roughly 103 million tokens across
28,000 articles. Clean prose, high named-entity density, and standard enough
that results are comparable to published work.

The first build over 10,000 articles produced 10,462 unique morphemes against a
target vocabulary of 10,000 — already over budget with zero slots left for
entities. Worse, running a coverage check on the validation set showed **34.8%
of words falling to character-level fallback**, and the content of that fallback
was alarming: `known`, `related`, `closely`, `parts`, `eastern`, `becoming` —
completely ordinary English words with clear morphological structure, present in
MorphoLex, which simply had not appeared often enough in a 10,000-article
sample.

### The diagnosis

The threshold was conflating two different things:

- low frequency because a morpheme is **rare but real** — should keep
- low frequency because it is **segmenter noise** — should discard

**Frequency cannot distinguish these cases.** No choice of threshold can,
because the distinction is not about frequency at all.

### The redesign: source determines validity

The insight was that a hand-curated linguistic database was already in hand.
MorphoLex maps roughly 70,000 English words to their morpheme decompositions.
**Any morpheme appearing anywhere in MorphoLex is, by definition, a real English
morpheme.** Any morpheme not in it is either noise or a genuine OOV that should
fall to characters. No threshold is needed.

This produced the six-layer vocabulary:

| Layer | Category | Source | Threshold | Count |
|---|---|---|---|---|
| 1 | Prefixes | MorphoLex | none | 127 |
| 2 | Suffixes | MorphoLex | none | 211 |
| 3 | Roots | MorphoLex | none | ~15,000 |
| 4 | Function words | hardcoded closed class | none | 112 |
| 5 | Named entities | dataset frequency | see §6 | 1,259 |
| 6 | Characters | a–z | never removed | 26 |

**Only layer 5 uses frequency, and only because the justification is different
there.** A named entity deserves an embedding slot only if the model will see it
enough times to learn from; one that appears once cannot be meaningfully
represented however many dimensions it gets. For every other layer frequency is
irrelevant — a real morpheme deserves an embedding whether it appears once or a
million times.

**Function words as a fourth category.** Function words (`the`, `and`, `of`) are
a closed class with no morphological structure and no compositional meaning.
They exist in MorphoLex as single-morpheme roots, but they deserve their own
flag, so a `function` layer provides that signal for free.

**Characters always survive.** The character layer is never deduplicated even
when a character also appears as an affix. This guarantees 100% coverage: any
string can be spelled out.

**Priority ordering.** Some tokens appear in several layers — `and` is both a
MorphoLex suffix and a function word. Deduplication must therefore respect
priority rather than insertion order: `entity > function > prefix > suffix >
root > character`. Getting this wrong produced a real bug, discussed in §7.

The vocabulary settled at roughly 15,700 entries, later 17,151 after the
information-preserving layers of §17 were added. Larger than BPE's 10,000, but
every entry linguistically justified.

---

## 6. Named Entities: The Threshold Wars and a Structural Answer

Layer 5 was the one place frequency was still doing work, and it took three
failed attempts to get right. The problem: spaCy's named-entity recogniser
occasionally mislabels common words, so a low threshold admits noise while a
high one excludes real entities.

**Attempt 1 — lower the threshold to 3.** This correctly admitted
`north_korea` (5), `european_union` (9) and `san_francisco` (7). It also
admitted `light` (3), `world`, `child` and `group` as entities. The layer grew
to 2,049 entries, many of them noise.

**Attempt 2 — exclude anything that is a MorphoLex root.** The reasoning: a
token already known as an English root is a common word being misclassified.
**This failed badly**, because MorphoLex covers proper nouns too. The filter
correctly removed `light` and `world`, but also removed `london`, `england`,
`france`, `germany`, `australia`, `florida` and `chicago`.

**Attempt 3 — a curated list of ambiguous words.** Rejected before
implementation as incomplete and unmaintainable: there are too many such words
and the list would need permanent curation.

### The structural answer

The distinguishing feature was already present in the token itself.

**Multi-word entity strings contain underscores.** `north_korea`,
`european_union`, `san_francisco` are produced only when spaCy recognises a
multi-word span. There is no common English phrase `north_korea` that could be
a misclassification. Single-word entities are different — `light`, `world`,
`child` are overwhelmingly ordinary words that spaCy occasionally mislabels at
low frequency.

**The rule: split the threshold by structure.**

- multi-word entities (containing `_`): frequency ≥ 3 — they cannot be misclassifications
- single-word entities: frequency ≥ 10 — higher confidence required

| Token | Type | Count | Threshold | Result |
|---|---|---|---|---|
| `north_korea` | multi-word | 5 | 3 | included |
| `european_union` | multi-word | 9 | 3 | included |
| `san_francisco` | multi-word | 7 | 3 | included |
| `london` | single-word | 97 | 10 | included |
| `light` | single-word | 3 | 10 | excluded |

This is not a heuristic. It is a structural observation about how the annotation
is produced, and it made the MorphoLex exclusion filter unnecessary. The entity
vocabulary settled at **1,259 entries**.

**The limitation that remained, and later mattered enormously.** `Goldman
Sachs` never entered the vocabulary at all — it does not appear often enough as
a recognised entity span in a 6,000-article sample. At the time this was
accepted as a corpus-coverage limitation rather than a design flaw. It was the
correct diagnosis, and §19 measures its cost: **77.5% of all character-fallback
words are capitalised**, which is the single largest component of the deficit
that eventually sank the segmentation approach.

---

## 7. The Segmenter, and the Bug That Cost Thousands of Prefixes

### The tools

**MorphoLex** is a hand-curated Excel database of roughly 70,000 English words
with morpheme decompositions in a custom notation: `{(beauty)}>ful>` is root
`beauty` plus suffix `ful`; `<un<{(happy)}` is prefix `un` plus root `happy`.

Its most useful property is that it stores **normalised roots, not surface
forms**: `beautiful` → `beauty`, not `beauti`; `running` → `run`, not `runn`.
This solves morphophonological normalisation for free across the whole database.
It is also, as §17 establishes, the reason the token stream turned out not to be
information-preserving — the same property that helps a tagger destroys
information a language model must predict.

**spaCy** provides named-entity recognition. Multi-word entities are correctly
grouped, possessives are stripped, and leading articles are removed before the
surface form is stored (`the united states` → `united_states`). Entity types
kept: ORG, PERSON, GPE, LOC, FAC, NORP, PRODUCT, EVENT, WORK_OF_ART, LAW,
LANGUAGE. Excluded: DATE, CARDINAL, ORDINAL, PERCENT, TIME, MONEY, QUANTITY —
these are compositional and infinite in variety, so treating them as atomic
units makes no sense.

**Morfessor**, an unsupervised statistical segmenter, was intended as a fallback
for words absent from MorphoLex. Its fate is §8.

### The pipeline

```
sentence → spaCy NER → entity spans
         → for each token:
             entity            → emit atomic unit
             in MorphoLex      → emit its morphemes
             Morfessor valid   → emit those
             otherwise         → emit characters
```

Each emitted token carries its id, surface string, category, morphological role,
entity label if any, and the original word it came from.

### The format-B bug

Running the segmenter over a larger test set revealed that `researchers`
produced only `er[suffix]` — the root `search` and prefix `re` were vanishing.

MorphoLex stores it as `{<re<(search)}>er>`. The parser looked for prefixes in
the string *before* the opening brace, but this entry places the prefix
*inside* it. Two formats exist:

- Format A: `<un<{(happy)}` — prefix before the opening brace
- Format B: `{<re<(search)}>er>` — prefix inside it

**4,941 of roughly 70,000 entries use format B.** Thousands of words had been
silently losing their prefixes. The fix detects prefix position relative to the
parenthesised root and collects both. Recovered examples: `researchers →
re + search + er`, `prosecution → pro + secute + ion`, `universe → uni + verse`.

**Side effect:** the prefix count expanded from 78 to 141, and the 63 new
entries include single-letter Latin prefixes such as `a-`, `e-` and `o-`. Those
single-character prefixes immediately caused a second problem, addressed in §8.

This bug is worth dwelling on because **the same class of error recurred four
more times**, always with the same signature: a regular expression matching one
instance of something that occurs many times. §17 records its exact twin in the
handling of compound roots.

### Smaller fixes, recorded for completeness

Three category-resolution bugs were found and fixed at this stage: `and`
resolving to `suffix` rather than `function` because a flat dictionary took the
last insertion; `london` resolving to `root` because the entity check ran after
the MorphoLex check; and single characters in fallback contexts inheriting a
vocabulary category rather than `character`. All three were consequences of the
priority ordering from §5 not being enforced at lookup time, and all were fixed
by resolving category through minimum priority rather than insertion order. The
full bug record, including the eleven found later, is in
`docs/record/bugs.md` §9.

---

## 8. Morfessor's Null Result and the Three-Pipeline Decision

### The noise problem

A 60-sentence test set across 11 linguistic categories was built to evaluate the
segmenter qualitatively. It showed Morfessor's contribution was close to
entirely negative.

Every useful segmentation in the test set came from MorphoLex or from prefix
stripping. What Morfessor added was noise: `biden → bi[prefix] + den[root]`,
where two real English morphemes (`bi` = two, `den` = lair) were found inside a
proper noun to produce a confident, wrong parse. The single-letter prefixes
recovered by the format-B fix made this worse, producing output like
`e[prefix]`, `o[prefix]` and `n[suffix]` scattered through character soup.

A minimum-length filter was applied — a Morfessor segment must be at least two
characters to be accepted as a morpheme — which removed the single-character
noise but not the `bi+den` class.

### The capitalisation guard, considered and rejected

One proposal was to fall straight to characters for any capitalised word absent
from MorphoLex and the entity vocabulary.

**Rejected.** Sentence-initial words are always capitalised; stylistic
capitalisation exists (`God`, `Internet`); lowercase proper nouns appear in
informal text. The guard relies on a typographic convention rather than a
linguistic category, which violates the design principle established in §5.

The reasoning for accepting `bi+den` instead is worth preserving: at scale, that
segmentation appearing consistently across thousands of Biden mentions will
cluster with political context regardless of being wrong, because the model does
not know morphemes are supposed to be meaningful — it sees consistent
co-occurrence. Character soup is more honestly unknown but equally
uninformative. The guard would have hidden a real limitation behind a brittle
heuristic.

### The fork, and the decision that set the project's method

Three options for Morfessor: remove it entirely; keep it but accept only known
affixes, since its root guesses are unreliable while its affix identification is
more trustworthy; or revert to 78 prefixes to eliminate the noise source.

The third was rejected as unprincipled — discarding half the prefixes and hoping
the problematic ones were among them.

**Decision: build both remaining options as separate pipelines and compare
them.**

| Pipeline | Description |
|---|---|
| `bpe` | standard GPT-2 style BPE, the baseline |
| `morph` | morpheme-based with Morfessor fallback, affix-only rule |
| `det` | morpheme-based, fully deterministic, no Morfessor |

**This decision established the method used for every subsequent uncertainty in
the project: where a design choice is genuinely uncertain, turn it into a
comparison rather than a guess.** §11 applies it to the frozen embedding design,
and §21 applies it to frozen versus learned factors.

It also produced a durable result. The two morpheme pipelines gave identical
output on roughly 55 of 60 test sentences, differing only where Morfessor might
find a valid affix inside character soup. **Morfessor adds essentially nothing
over the deterministic pipeline for English with MorphoLex coverage** — and
every subsequent quantitative comparison confirmed it. In the final
information-fair runs, `det` matched or beat `morph` in **41 of 41** tail
evaluations.

---

## 9. Compound Splitting and the Particle Fix

OOV compounds — `blockchain`, `cryptocurrency`, `livestream`, `speedrun`,
`homepage` — were still producing character soup. BPE handles them because it
has seen them in training data. Compound splitting was the linguistically
principled alternative.

**The safety question came first.** Could splitting produce `therapist → the +
rapist`? A check confirmed that every dangerous-sounding candidate —
`therapist`, `together`, `carpet`, `island`, `before`, `because` — is in
MorphoLex and therefore resolved before the compound splitter is ever reached.
The splitter only ever sees genuinely OOV words, among which such false
positives are rare.

**The algorithm.** Try every split point from longest-left to shortest-left;
accept the first split where both halves are in MorphoLex and both are at least
three characters. Position in the fallback chain:

```
entity → function → MorphoLex → prefix strip → compound split → Morfessor/characters
```

Clear wins followed: `blockchain → block + chain`, `speedrun → speed + run`,
`videogame → video + game`, `spreadsheet → spread + sheet`,
`counterarguments → counter + argue + ment`.

### The star+tup problem

Then `startup → star[root] + tup[root]`. `tup` is a genuine but obscure
MorphoLex root — British English for a male sheep — and the algorithm found
`star + tup` at position 4 before reaching `start + up` at position 5.

The cause was structural rather than accidental. The loop guaranteed a
right-hand part of at least three characters, which **blocked `start + up`
entirely** because `up` is two. And `up`, `in`, `on`, `out`, `off` and `by` are
among the most productive compound-forming elements in English: `startup`,
`backup`, `addon`, `cutoff` and `roundup` were all failing silently.

**The insight:** these are not arbitrary short strings. They are exactly the
phrasal-verb particles that also form nominal compounds — and all of them were
already in the `FUNCTION_WORDS` set.

**The fix:** allow two-character right parts, but only when they are known
function words. For `startup`, position 5 gives `start` + `up`, `up` is a
function word, the split is accepted, and `star + tup` is never reached.

This is the same shape of solution as §6's split threshold: rather than tuning a
numeric threshold, use a structural property of the linguistic material to make
the distinction.

---

## 10. What the First Comparison Showed — and What It Concealed

### The comparison

A GPT-2 style BPE tokenizer was trained on the same 10,000 WikiText-103
articles with a 10,000-token vocabulary, for direct comparability. Side by side
on test words:

| Word | Morpheme pipeline | BPE |
|---|---|---|
| `unbreakable` | `un + break + able` | `un \| break \| able` |
| `beautiful` | `beauty + ful` | `beautiful` |
| `cats` | `cat` | `c \| ats` |
| `Microsoft` | atomic entity [ORG] | `Mic \| ros \| oft` |
| `happiness` | `happy + ness` | `happ \| iness` |
| `whether` | `whether` | `W \| he \| ther` |

Clear apparent wins for inflected forms, function words, morphologically complex
words, named entities and compounds. Clear apparent losses on unknown multi-word
entities (`Goldman Sachs`, `North Korea` → character soup) and on complex OOV
(`cryptocurrency`, `Instagram` → characters), where BPE produces much shorter
sequences.

### Consolidating the codebase

By this point the code was several large files with overlapping responsibilities
and accumulated patches. A deep folder hierarchy was proposed and rejected in
favour of a **flat structure of seven files**, on the explicit grounds that the
communication overhead of a deep hierarchy outweighed its organisational benefit
in a context where files were shared one at a time.

That reasoning was correct for its conditions. §16 records the conditions
changing and the decision being reversed.

### What this comparison could not see

Reading the table above, the morpheme pipeline looks clearly better on the cases
that matter and worse only on genuinely unknown words. That impression survived
into the first quantitative results, where the morpheme pipelines appeared to
beat BPE by **6.31% bits-per-byte**.

**It was an artifact, and every element of it is visible in the table above if
one knows what to look for.**

`cats → cat` loses the plural. `running → run` loses the progressive.
`beautiful → beauty + ful` concatenates to `beautyful`. The morpheme stream was
not encoding the same information as the BPE stream, and bits-per-byte divides
both by the same byte count. On top of that, the pipeline emitted nothing at all
for punctuation, digits or case, which BPE encodes — an asymmetry worth 5.32% of
the corpus.

Part IV establishes this in detail and reverses the result. What matters here is
that **the qualitative comparison in this section was not wrong about
segmentation quality; it was blind to information content**, and no amount of
inspecting example words would have revealed that. It took an explicit audit
against a formal criterion.

---

