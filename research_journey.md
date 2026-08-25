# Context-Aware Embeddings for Tokenizer-Free Language Models
## Research Journey

*The record of how this project reached its conclusions: the decisions, the
forks, the dead ends, and the several occasions on which a result we believed
turned out to be an artifact of our own measurement.*

> **Companion documents.**
> `results_reference.md` — the data: 23 runs with identifiers, every
> configuration, every audit result, the limitations table and the bug record.
> This document explains *why*; that one holds the *numbers*.
> `literature_analysis.md` — full reads of MorphPiece, Hou et al. 2023,
> Bostrom & Durrett 2020, FLOTA and Factored Neural Language Models, compared
> dimension by dimension against this project.
>
> Together with the codebase these three files are the complete source of
> truth. No result quoted anywhere should be absent from them.

**How to read this.** The document is chronological, in six parts. Parts I–II
build a morphological tokenizer and a vocabulary. Part III builds the measuring
apparatus. **Part IV is where the project turns**: a series of audits shows that
the headline result of Parts I–II was an artifact, and reverses its sign. Part V
grounds the work in the literature and completes the controls. Part VI reaches
the finding the project actually stands on.

Readers short of time can get the conclusion from Part VI and the reasoning
behind it from Parts IV–V. Parts I–II matter mainly for *why* the tokenizer was
built the way it was, and for the audit methodology that grew out of finding it
wrong.

A note on the arc: **the negative result in Part IV is not a setback that was
later rescued.** It is the measurement that identified where morphological
knowledge should be spent, and Part VI follows directly from it.

---

## Contents

**Part I — The Question**

1. [The Starting Point and the Intervention Point](#1-the-starting-point-and-the-intervention-point)
2. [The Hybrid Frozen/Trained Embedding](#2-the-hybrid-frozentrained-embedding)
3. [What Can Be Known, and Where](#3-what-can-be-known-and-where)
4. [Choosing the Unit, and How to Evaluate It](#4-choosing-the-unit-and-how-to-evaluate-it)

**Part II — Building the Morphological Tokenizer**

5. [The Vocabulary: Why Frequency Is the Wrong Criterion](#5-the-vocabulary-why-frequency-is-the-wrong-criterion)
6. [Named Entities: The Threshold Wars and a Structural Answer](#6-named-entities-the-threshold-wars-and-a-structural-answer)
7. [The Segmenter, and the Bug That Cost Thousands of Prefixes](#7-the-segmenter-and-the-bug-that-cost-thousands-of-prefixes)
8. [Morfessor's Null Result and the Three-Pipeline Decision](#8-morfessors-null-result-and-the-three-pipeline-decision)
9. [Compound Splitting and the Particle Fix](#9-compound-splitting-and-the-particle-fix)
10. [What the First Comparison Showed — and What It Concealed](#10-what-the-first-comparison-showed--and-what-it-concealed)

**Part III — The Apparatus**

11. [Designing the Frozen Embedding Layer](#11-designing-the-frozen-embedding-layer)
12. [The Dataset, Model, and the Fairness Metric](#12-the-dataset-model-and-the-fairness-metric)
13. [Calibration and Literature Grounding](#13-calibration-and-literature-grounding)

**Part IV — The Audits**

14. [The Frozen Embedding Audit](#14-the-frozen-embedding-audit)
15. [Parameter Tying vs Information Injection](#15-parameter-tying-vs-information-injection)
16. [Restructuring — Reversing the Flat-Layout Decision](#16-restructuring--reversing-the-flat-layout-decision)
17. [The Information Channel Audit — Making the Comparison Fair](#17-the-information-channel-audit--making-the-comparison-fair)
18. [The Information-Fair Rerun — Q1 Falls, Q2 Holds](#18-the-information-fair-rerun--q1-falls-q2-holds)
19. [Localising the Q1 Deficit — Coverage, Decomposition, and the Unified Finding](#19-localising-the-q1-deficit--coverage-decomposition-and-the-unified-finding)

**Part V — Grounding and Controls**

20. [Literature Grounding — Who Has Done This Already](#20-literature-grounding--who-has-done-this-already)
21. [Completing the Controls — Matched Vocabulary, Frozen vs Learned, and Signal Scale](#21-completing-the-controls--matched-vocabulary-frozen-vs-learned-and-signal-scale)
22. [Why English Resists — and the Turn from Confrontation to Cooperation](#22-why-english-resists--and-the-turn-from-confrontation-to-cooperation)

**Part VI — The Synthesis**

23. [Level 2 — Morphology as Features on a BPE-Tokenized Model](#23-level-2--morphology-as-features-on-a-bpe-tokenized-model)
24. [Where the Project Stands](#24-where-the-project-stands)

- [Appendix A — Decision Record](#appendix-a--decision-record)

- [Appendix B — Section Mapping](#appendix-b--section-mapping)

---

# Part I — The Question

*What we were asking, where in a language model to intervene, and what
linguistics says is knowable at each scope. This part is design only; nothing
was built yet.*

---

## 1. The Starting Point and the Intervention Point

**The thesis topic:** *Context-Aware Embeddings for Tokenizer-Free Language
Models.* The initial goal was vague — build some representation of text based
on morphology, syntax or semantics that replaces tokens. No decision had been
made about what the unit should be, where in the model to intervene, or how
success would be judged.

The first question was whether the unit is a word, part of a word, and whether
that part must be meaningful. Four directions were open: morpheme-based
embeddings, character n-grams in the FastText style, syntactic span embeddings,
and continuous CTC-style soft segmentation. Rather than choose immediately, the
first priority was to understand precisely what was being replaced.

### What the standard pipeline actually does

```
text → [tokenizer] → [integer ids] → [embedding lookup] → [vectors] → [transformer]
```

The embedding table is a matrix of shape `[vocab_size × embedding_dim]`. Each
row is one token's vector; looking up token 7 fetches row 7. **There is no
intelligence in this step — it is a matrix indexing operation.**

Two properties of it turned out to drive the entire project.

**Gradient sparsity.** During training, gradients reach the embedding table only
for tokens that appeared in the batch. Rare tokens are updated rarely and
therefore have worse representations than common ones. This is a known and real
problem, and it is the mechanism that every positive result in this project
eventually traces back to.

**Context-independence.** The table produces the *same* vector for a word
regardless of context. `bank` gets one vector whether it means a financial
institution or a river bank; context-sensitivity arrives later, inside
attention. The implication is sharp: because the embedding is static, it must
pre-encode everything useful about a token's identity *before* any context is
available. For common unambiguous words this is fine. For rare, ambiguous or
morphologically complex words the single-row constraint is a real limitation.

### Where to intervene

Three options: replace the tokenizer only, replace the embedding table only, or
replace both.

**Decision: target the embedding layer, not the model architecture.** A
transformer receives a sequence of vectors and does not care where they came
from. Better vectors require no architectural change. This fixed the shape of
every experiment that followed:

- Flow A: text → BPE → embedding table (standard)
- Flow B: text → our preprocessor → our embedding layer

with the transformer held identical between them. Everything in this document
is a comparison of that form.

---

## 2. The Hybrid Frozen/Trained Embedding

The central idea. Instead of an embedding vector whose N dimensions are all
learned freely, split it:

```
[ frozen dims (T) | trained dims (N − T) ]
```

The frozen dimensions encode linguistically certain facts injected at
preprocessing time and are never updated by backpropagation. The rest are
learned normally.

**The argument for it.** Frozen dimensions create guaranteed information
channels. Nothing normally stops a model from learning to ignore or distort any
part of what an embedding encodes. Freezing some dimensions guarantees that
certain information is always present, always readable, at every layer, for any
attention head that wants it.

**The division-of-labour question this raises.** If the trained table already
handles everything, freezing eases the burden of the *embedding layer*, not of
the attention layers. Attention already handles context, syntactic resolution
and word-sense disambiguation well. The target is specifically what the
embedding layer must do before any context exists.

That framing produced the sharp version of the question: *does giving the model
guaranteed correct morphological information as a starting point actually
help?* The answer was not obvious, which is what made it worth investigating.

Two things about this section are worth flagging in advance, because both were
substantially revised later. The claim that frozen dimensions supply
information the model otherwise lacks turns out to be **false for facts that
are a function of the token id** — see §15. And the claim that freezing beats
learning was **tested directly and refuted** — see §21. What survives is the
mechanism, not the original explanation of it.

---

## 3. What Can Be Known, and Where

To decide what to freeze, we needed to know what linguistic information is
actually available at each scope.

### Morphology

A morpheme is the smallest meaningful unit: `unbreakable` = `un` (negation) +
`break` (root) + `able` (capable-of). Free morphemes stand alone (`break`,
`cat`); bound ones do not (`-ing`, `un-`, `-able`). Inflectional morphemes mark
grammatical properties (`-s` plural, `-ed` past); derivational ones change
meaning or category (`-ness` nominalises, `re-` adds repetition).

From a single word in isolation, morphology yields the root, the affixes and
their meanings, and often the part of speech — many suffixes are
category-specific, since `-ness` almost always makes a noun and `-ly` almost
always an adverb.

### Syntax and semantics

**Dependency syntax** describes grammatical relations: in "the cat chased the
dog", `cat` is subject of `chased`. These relations are *relational* — a word's
syntactic role cannot be determined by looking at it alone.

**Constituency syntax** groups words into phrases that behave as units.
Membership requires sentence-level analysis.

**The key insight:** syntactic role is not a property of a word. It is a
property of a word *in a particular sentence*. It cannot be frozen per-word
without sentence context.

Lexical semantics is partly available at word level (`cat` is always an animal)
but polysemous words need context. Compositional semantics is entirely
context-dependent.

### The scope table

| What can be known | Morpheme | Word | Phrase | Sentence |
|---|---|---|---|---|
| Morpheme type | ✓ | ✓ | ✓ | ✓ |
| Inflection (tense, plural) | partial | ✓ | ✓ | ✓ |
| Part of speech | partial | mostly | ✓ | ✓ |
| Syntactic role (subject/object) | ✗ | ✗ | partial | ✓ |
| Phrase membership | ✗ | ✗ | ✓ | ✓ |
| Unambiguous lexical meaning | ✗ | partial | ✓ | ✓ |
| Contextual meaning | ✗ | ✗ | partial | ✓ |

### The observation that mattered most

**Preprocessor scope ≠ unit scope.** The scope of what the preprocessor *looks
at* when making a decision is independent of the scope of the *unit* it
produces. We can examine an entire sentence to decide what tag to attach to one
word; the model still receives word-level vectors. The preprocessor is simply a
smarter tagger.

This vastly expands what can be frozen. Syntactic role — which requires
sentence context to determine — can be frozen at the word level if the
preprocessor runs a dependency parser first.

**This is the most valuable observation in the document, and the
implementation lost it for the entire first half of the project.** It was
recovered only in §15, and the experiment it enables is the one that produced
every positive result the project has.

---

## 4. Choosing the Unit, and How to Evaluate It

### Word-level or morpheme-level?

Word-level units can encode morphological facts, POS, syntactic role and entity
type, but not compositional meaning or sense for ambiguous words. The
vocabulary is large and out-of-vocabulary handling is poor — an unseen word has
no vector at all.

Morpheme-level units encode morpheme type and meaning class. The vocabulary is
small — English has only a few thousand productive morphemes — so each entry
trains more often, and OOV handling is linguistically grounded rather than
statistical, since even an unknown word usually has known affixes. The costs
are that POS and syntactic role attach to words rather than morphemes, and that
phrasal verbs and multiword expressions break compositional assumptions.

**An observation that forced a third category.** Some words cannot be
meaningfully decomposed. `Microsoft` does not break into `micro` + `soft` in any
useful sense. This produced a tripartite theory: every word is either
**decomposable into meaningful morphemes**, a **named entity** to be treated as
an atomic unit, or **unknown**, in which case it falls to characters honestly.

**Decision: morpheme-level units, with named entities atomic.** Smaller
vocabulary, closer structural comparison to BPE, and principled OOV handling.

### Evaluating without generation

**Decision: text generation is not required.** Three strategies avoid it.

**Perplexity** measures how surprised the model is by held-out text. Lower is
better and it is directly comparable between approaches.

**Probing** trains a small classifier on the embeddings to predict a linguistic
property. If the embeddings encode it, the classifier learns easily from little
data. This directly tests whether frozen dimensions are actually *used* rather
than merely present.

**Word similarity benchmarks** correlate embedding cosine similarity against
human judgements.

The intended strategy was perplexity as the headline number and probing as the
mechanistic story: *"our embeddings produce lower perplexity, and probing shows
the frozen dimensions are actively used."*

Two notes on how this aged. The perplexity half was pursued throughout, though
§12 shows that comparing perplexity across tokenizers is invalid and
bits-per-byte must be used instead. **The probing half was never run.** The
extraction code and a full probing script exist; the classifier was deliberately
deferred until the results indicated what was worth probing, and the project
reached its conclusions before that happened. It remains the single largest
piece of unfinished evaluation.

---

# Part II — Building the Morphological Tokenizer

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
`results_reference.md` §9.

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

# Part III — The Apparatus

*Building the thing that measures. The transformer is deliberately small and
standard: it is the microscope, not the object of study. This part also settles
the metric question, which turned out to carry more weight than anything else
in the document.*

---

## 11. Designing the Frozen Embedding Layer

With the preprocessing pipelines complete, attention turned to the theoretical centerpiece: the hybrid frozen/trained embedding. Before writing any code, the design space was mapped through four questions. The guiding principle, borrowed from the three-pipeline decision: where a design choice is genuinely uncertain, turn it into a comparison rather than a guess.

### The Four Design Questions

**Q1 — What goes in the frozen dimensions?** Three philosophies were considered:
- **(A) One-hot categorical:** each frozen fact is a one-hot block (category, role, entity label). Clean indicator flags — says *that* a token is a suffix, but nothing about how `-ed` relates to `-ing`.
- **(B) Linguistic feature vectors:** encode *what kind* of morpheme using real features (`-ed → [+inflectional, +past]`, `un- → [+derivational, +negation]`). Richer — shared features create real relationships between morphemes.
- **(C) Pretrained semantic vectors:** freeze a slice of GloVe/word2vec for each root. **Discarded** — it muddles the thesis (are gains from our segmentation or from GloVe leaking in?) and breaks the "we inject *certain* facts" story, since GloVe vectors are themselves learned and noisy.

**Q2 — How many dimensions frozen vs trained?** Not a free hyperparameter — T falls out of the encoding scheme in Q1 (one-hot → T≈23). What is free is total width N. The concern: at N=512, a 23-dim frozen block is only 4.5% of the vector — possibly too weak to matter. This produced a set of escalating "impact levers" (below).

**Q3 — Concatenation vs addition?** **Concatenation fixed, addition discarded.** Addition (like positional encoding) lets the trained part learn to cancel the frozen signal — which defeats the entire premise of *guaranteed* presence. Concatenation keeps the regions cleanly separate.

**Q4 — The ragged-structure problem.** Different tokens carry different facts: a prefix has a role but no entity label; an entity has a label but no morphological role; a character has neither. Yet all need a frozen vector of identical width. **Solution:** a fixed schema with reserved slots for every possible fact; each token fills the slots it has and zeros elsewhere. The "none" slots are explicit, meaningful positions — a character's empty entity block is a real zero-pattern, not missing data.

### The Impact Levers (Q2 Mitigations)

Acknowledged that 4.5% frozen signal at N=512 is weak, with four escalating responses ready:
1. **Small N** — at N=128 the frozen block is ~18% of the vector. Default first choice. Also faster to train across many variant combinations.
2. **Amplification** — tile/scale the frozen block to occupy more real estate. Implemented as `freeze_amplify`.
3. **Frozen random projection** — multiply the fact vector by a fixed random T×K matrix, spreading the signal across K dims in a fixed pattern.
4. **Auxiliary probe loss** — force the model to keep frozen dims legible by requiring a probe to predict category from them. Dovetails with the probing evaluation.

### The Annotation Cost of Option B

The hidden cost of B is *annotation* — assigning feature vectors to morphemes. The cost splits sharply by morpheme type:
- **Affixes (381):** closed class, hand-annotatable in an afternoon of review. The linguistically richest part — this is where morphological theory does real work.
- **Roots (~15,000):** hand annotation is infeasible. Three sub-options: **B1** (roots get one-hot flag only, no features), **B2** (coarse hand-annotated root classes — **discarded**, 15k roots is infeasible), **B3** (WordNet supersenses, 45 discrete semantic classes, looked up automatically).
- **Entities (11):** trivial, just one-hot.

This split Option B into two honest sub-variants: **B-affix** (rich affix features + one-hot roots, the B1 choice) and **B-full** (B-affix + WordNet root supersenses, the B3 choice).

### The Decision Tree and Phased Experiment Matrix

The final comparison ladder, each rung adding one idea:
```
baseline (no freezing — control)
  → onehot (categorical flags)
    → B-affix (rich affix features + root/entity flags)
      → B-full (B-affix + WordNet root supersenses)
```

Crossed with the three preprocessing pipelines:

| | baseline | onehot | B-affix | B-full |
|---|---|---|---|---|
| **bpe** | ✓ control | — | — | — |
| **morph** | ✓ | ✓ | soon | later |
| **det** | ✓ | ✓ | soon | later |

BPE only gets `baseline` (no morpheme metadata to freeze) — that cell is the ultimate reference point. **Phase 1** runs the 5 ✓ cells, which already answer two questions: does morpheme segmentation beat BPE (the baseline column), and does freezing flags help (baseline vs onehot). **Phase 2** adds B-affix; **Phase 3** adds B-full if earned.

### Implementation

`frozen_embedding.py` was written scoped to Phase 1 with hooks for the rest:
- `HybridEmbedding` — produces `[frozen (T) | trained (N−T)]` by concatenation. The frozen rows are a registered **buffer** (not a Parameter), so they never receive gradients.
- `FrozenSchema` ladder: `NullSchema` (T=0, baseline control — collapses to a plain `nn.Embedding`), `OneHotSchema` (T=23, implemented), `FeatureSchema` and `WordNetSchema` (stubbed, raise `NotImplementedError` so they can't be silently selected).
- `freeze_amplify` for Lever 2; default `embedding_dim=128` for Lever 1.

**Validation confirmed the core guarantee:** after a full optimizer step with a loss pulling on every output dimension, the frozen slice was byte-for-byte unchanged while the trained part moved (grad norm 0.17). The freeze is literally enforced, not nominal. All six token categories produced exactly the predicted 23-dim frozen vectors (3 nonzero bits each — one per block).

**One deferred decision:** the frozen matrix is currently built from vocabulary rows, which don't carry NER context — so entities currently get their category flag but a `none` entity-label slot. Baking entity *type* into the frozen vector will require either storing it in the entity vocab rows or moving frozen-vector construction to encode-time. Noted for when the entity-type signal is wanted.

---

---

## 12. The Dataset, Model, and the Fairness Metric

With the embedding written, three more pieces were needed to make Phase 1 runnable: the dataset, the model, and the training loop. Before writing them, we settled how they depend on each other.

### Coupling Analysis — Dataset vs Model

The two modules are mostly separable, touching at only three points:
- **`seq_len`** — a shared constant both obey (set to 256).
- **`vocab_size`** — owned by the pipeline/vocabulary upstream; both modules read it. BPE ≈ 10k, morpheme pipelines ≈ 16.7k.
- **Embedding ownership** — the one real architectural decision. **The model owns `HybridEmbedding`; the dataset emits integer IDs only.** Decisive because the embedding *is* the thesis contribution and its trained portion has gradients — if the dataset owned it, we'd be backpropagating into the dataloader. IDs are also tiny to pass around; vectors are not.

Once these three are pinned, the modules decouple cleanly: the dataset is a pure IDs→IDs machine, the model a IDs→logits machine.

### The Fairness Metric — the one decision with thesis weight

The pipelines expand text differently (one word → variable token count), so **per-token perplexity is not comparable across pipelines** — a model that splits words into more, individually-easier tokens looks artificially better. Three positions were considered: equal token count (simple, unfair on context), equal word count (fairer, complex), and **equal token count + a tokenization-agnostic metric (chosen)**.

**Decision: per-token perplexity WITHIN a pipeline (same tokenization, fair); bits-per-byte (BPB) ACROSS pipelines.** This keeps the dataset simple (fixed L, concatenate-and-chop) and moves the fairness burden to the metric, where the literature says it belongs. BPB chosen over BPC as the more robust normalization (handles non-ASCII like `é` cleanly). The citation lineage: Mielke (2019) → The Pile (BPB) → Paloma. The "Neural Polysynthetic Language Modelling" paper provides the exact computation recipe and a directly analogous precedent (comparing segmentation strategies via character-level perplexity). The dataset therefore tracks, per cached stream, the UTF-8 byte count of the original text, so BPB is computable at eval time.

### Other Dataset Decisions (standard answers)

- **Document boundaries:** ignored — concatenate-and-chop into fixed windows (standard GPT-2 practice).
- **Caching:** tokenize once per pipeline, save the integer stream as `.npy`, memory-map for training. spaCy + morpheme segmentation is far too slow to repeat every epoch. Cache validated: miss-then-hit reproduces identical ids.
- **Corpus cap:** a parameter (`max_articles`), value deferred until training speed was known.

### The Model — a Fixed Apparatus, Not the Variable of Interest

The transformer is **load-bearing but not the object of study.** The thesis claim is about embeddings; the transformer is the microscope through which we measure them. This flips normal priorities: we want the *fairest, most standard, most controlled* model, not the best one — identical across every matrix cell, with only the embedding varying.

Two reasons the model must be **small**:
1. **Hypothesis testability.** Frozen linguistic facts can only demonstrably help when the model can't trivially learn morphology itself. A large model on lots of data learns that `-ing` and `-ed` are related without help, washing out our effect. Small/data-limited is the regime where the hypothesis is testable.
2. **The matrix multiplies.** Many pipeline×variant runs; each must be cheap enough to iterate.

**Pinned and standard choices:**
- `d_model = N = 128` — pinned by the frozen-fraction argument (Lever 1: 23-dim frozen block is ~18% of 128).
- 4 layers (shallow on purpose — keeps reliance on the input representation, where our hypothesis lives), 4 heads (head_dim 32), `d_ff` = 4×d_model = 512.
- Learned absolute positional encoding, pre-norm blocks, GELU. Textbook GPT-2.
- **`tie_output` configurable.** Phase 1 uses `'none'` (untied, clean — frozen role stays purely on input). Option C (`tie_output='frozen'` — tie frozen columns into the output projection, a richer "frozen facts directly shape predictions" hypothesis) is reserved as a hook that refuses rather than silently misbehaving.

The result is a ~3.4M-parameter model. Everything is plain config so re-probing a different size is a one-line change.

### Calibration-First

Rather than committing to the architecture by argument, we adopted the user's proposal: **use the BPE-baseline cell as a calibration instrument.** Run it first, read its learning curve, and confirm the apparatus sits in the right regime — learns the text well, but with visible headroom for an embedding advantage to surface. Too-low final loss → model/corpus too easy; too-high plateau → too weak; decent-but-imperfect → the Goldilocks zone. The architecture gets *calibrated against the baseline* rather than chosen by judgment.

`train.py` reports both metrics, tracks **best-val** (not last-val, so overfitting can't contaminate the recorded number), and **early-stops** when validation plateaus, restoring the best checkpoint.

---

---

## 13. Calibration and Literature Grounding

### The Two-Run Calibration Story

**First run (3,000 articles, fast probe):** val loss bottomed at step 750 then climbed while train kept dropping — textbook **overfitting**. This was a data-quantity signal, not an architecture problem. It also confirmed the model has enough capacity to overfit, i.e. it is *not* too weak.

**Second run (full corpus, ~130M tokens):** overfitting vanished entirely. Train and val tracked together monotonically across all ~40k steps, never diverging — the signature of sufficient data. Result at the step ceiling (still slowly improving):

| Metric | Start | Best (step 39,500) |
|---|---|---|
| val loss | 9.24 | **4.17** |
| val perplexity | 10,124 | **64.9** |
| val BPB | 3.18 | **1.438** |

This is the **anchor cell** of the entire experiment — every other cell is measured against it. The regime is confirmed Goldilocks: a 156× perplexity reduction from random (genuinely learns), but BPB 1.44 is far from the ~0.8–1.0 SOTA floor (clear headroom for an embedding advantage).

For the real matrix, the step ceiling should be raised (the run was still improving at 40k) with the same patience-based early stop catching the true plateau.

### Literature Grounding (for the eventual paper)

Reference numbers were gathered with proper caveats, to avoid the central trap: **word-level vs subword/BPE perplexity are different units and not directly comparable.**

- **Word-level WikiText-103 SOTA** (≈267k vocab, adaptive softmax): Transformer-XL 18.3 test ppl (Dai et al., 2019), previous SOTA Baevski & Auli 20.5; fuller leaderboard ranges ~15.8–18.7 (Compressive Transformer, Routing Transformer, Shortformer, Perceiver AR). **Our 64.9 must NOT be compared to these** — different units.
- **BPE perplexity** (our regime): GPT-2 Small (117M, zero-shot) ≈ 29; GPT-2 XL (1.5B) ≈ 16. These are the relevant comparison class, though imprecise (GPT-2 trained on WebText, not WikiText).
- **Our placement:** a 3.4M-param model (~1/35th of GPT-2 Small) reaching 64.9 BPE ppl is exactly where a tiny model should land — sensible, with clear headroom to the 29 of a 35×-larger model and the ~18 of word-level SOTA.
- **WikiText-103 is known to overfit:** "a small dataset where strong regularization is required to prevent severe overfitting" (Perceiver AR paper). This validates our 3k-article overfitting as a documented dataset property, not a stumble. Dataset facts: 103M training tokens, 28K articles, ~3.6K tokens/article (Dai et al., 2019).

**Reporting rules locked in:** never compare raw perplexity to the 18.3 leaderboard (different units); headline cross-pipeline metric is BPB; the honest peer class is "small models trained from scratch on WikiText-103 with BPE," with GPT-2 Small's ~29 as a "what 35× more parameters buys" reference point.

---

# Part IV — The Audits

*Where the project turns. A pre-flight review of the frozen embedding found it
was carrying almost no information; pulling that thread led to a reframing of
what freezing can do at all, and then to a series of audits establishing that
the Part II comparison was invalid. The headline result reverses sign here.*

*This part contains the project's most transferable contribution: a method for
checking whether two tokenizers are being compared fairly, which the published
literature does not do.*

---

## 14. The Frozen Embedding Audit

Everything was built, calibrated, and ready. The five Phase-1 cells were about
to be launched. A last pre-flight review of the code — intended to check the
step ceiling and nothing more — instead found that the frozen embedding was
mostly empty.

### What the measurement showed

The frozen matrix was rebuilt from the actual `our_final_vocab.csv` (16,757
rows) and every column inspected:

| block | width | finding |
|---|---|---|
| category | 6 | varies — informative |
| role | 5 | **redundant** — a deterministic function of category |
| entity_label | 12 | **constant** — every row `none` |

`role:root` had exactly 15,134 nonzero entries = 15,022 roots + 112 function
words, confirming role is fully determined by category (6 categories → 5 roles,
`function` and `root` collapsing together).

**Across all 16,757 vocabulary rows there were 6 distinct frozen vectors.** The
23-dimension block encoded one 6-way categorical — about 2.6 bits — and 12 of
those dimensions were provably uninformative. A constant input dimension is
just a bias term: it consumes embedding width and carries nothing.

### Root cause

`FrozenSchema.matrix()` built frozen vectors from static vocabulary rows, and
those rows have no NER context, so `entity_label` was hardcoded to `None` for
every token. This was noted in §11 as "one deferred decision" — but recorded as
a missing *feature*, not as a bug that hollowed out half the block.

The role redundancy has the same root cause. Role is context-dependent at
encode time (the same token can occupy different roles in different words), but
when reconstructed from a vocabulary row it can only be re-derived from
category. Both dead blocks trace to building frozen vectors from static rows.

### Why validation missed it

§11 validated two things, and both passed:
- the freeze is enforced (frozen slice byte-identical after an optimizer step)
- every token category produces exactly the predicted 23-dim vector with 3
  nonzero bits

Both are **per-token** properties. Neither asks what the matrix looks like
across the population. The frozen width was sized by the label inventories
(6 + 5 + 12 = 23) rather than by realized information content, and nobody
computed the rank or entropy of the result.

**The lesson, recorded honestly:** per-instance correctness is not evidence of
population-level correctness. When a design choice cannot be independently
evaluated by the researcher, the load-bearing assumption must be made
*checkable* — a script that fails loudly — not merely stated confidently. This
is now `tests/test_blocks.py`, which asserts no requested block has a constant
column, plus a regression test that reproduces the old 23-dim layout and
confirms all 12 entity columns are constant. If that test ever stops failing,
the check itself has broken.

### Consequence for Q2

As coded, `onehot` gave up 385,503 trainable parameters (23 of 128 dims, ~18%
of embedding width) to buy a 6-way flag. A negative result would have been
uninterpretable: "guaranteed linguistic flags don't help" could not be
separated from "we spent 12 dimensions on a constant."

**Decision: do not launch the two `onehot` cells.** Not because the result
would be wrong, but because it answers a question nobody wants the answer to.

---

---

## 15. Parameter Tying vs Information Injection

The dead dimensions turned out to be a symptom. Digging into *why* the frozen
block was so thin exposed something more fundamental about the whole approach.

### The reframe

The frozen block is a function of the token id. So is the trained embedding.
Both are lookups keyed on the same integer.

Which means the frozen dimensions **cannot inject information the trained table
could not already represent.** If "being a suffix" were useful, the trained row
for that token could encode it. Freezing does not supply a fact the model
lacks; it *constrains* 23 dimensions to equal `f(id)`.

This demotes the §2 story. "Guaranteed information channels, always present,
always readable" reads as though facts arrive from outside the system. They do
not. What is actually happening is **structured parameter tying** — a real
inductive bias, but a different and more modest one than advertised.

Parameter tying can buy exactly three things:

1. **Sample efficiency for rare tokens.** A root seen 3 times gets almost no
   gradient signal, so its trained row is noise; its frozen flag is correct
   from step 0 for free. (This is the §1 rare-token problem, restated.)
2. **Forced similarity structure.** All 15,022 roots share a pattern, so they
   are constrained to be near each other; the trained table would have to
   discover that.
3. **Guaranteed legibility** for the probing story.

All three predict an effect concentrated in the **tail**, not the aggregate —
which is likely why 2.6 bits over 23 dimensions looked so unpromising measured
as "does overall perplexity drop." A single corpus-wide number averages the
tail away. Hence `eval/buckets.py`: per-frequency-bucket perplexity, so
"frozen flags help rare tokens by X% and leave frequent tokens unchanged"
survives an aggregate null result.

### The idea that got lost

§3 contains the most valuable observation in this document: **preprocessor
scope ≠ unit scope.** You may look at the entire sentence to decide what to
stamp on one word. That is what would permit freezing syntactic role, or entity
type in context, or word sense — facts a lookup table **provably cannot
represent**, because they differ across occurrences of the same token id.

The implementation dropped it entirely. Every occurrence of a token received an
identical frozen block. The one idea that would have made freezing genuine
information injection never reached the code.

And the consequence was uncomfortable: the thesis is titled *Context-Aware
Embeddings*, and the implementation was a static lookup table with no context
in it anywhere.

### Why it was dropped, and why that reason was wrong

§12 decided "the dataset emits integer IDs only; the model owns the embedding,"
reasoning that the embedding's trained part carries gradients and
backpropagating into the dataloader would be wrong.

**That reasoning does not extend to frozen metadata, which by definition has no
gradients.** The dataset can emit small integer codes alongside the ids with no
gradient problem at all. The principle ("dataset is a pure int-in/int-out
machine") is fully preserved. A rule was mistaken for a constraint, and the
cost was the §3 insight.

### The new axis

Frozen facts now split by whether a static table could represent them:

| rung | frozen content | representable by a trained table? |
|---|---|---|
| A0 | nothing (control) | — |
| A1 | category flags | **yes** — it is `f(id)` |
| A2 | affix features, entity type, WordNet supersense | yes in principle, hard to learn for rare tokens |
| A3 | dependency role, POS in context | **no** — varies across occurrences |

A1 is parameter tying. A3 is information injection. **"Static freezing does
little; context freezing helps"** is a real, measurable claim, and a better
result than "freezing helps by 2%" whichever way it lands.

**Decision: v1 injects ONE fact — dependency role**, the fact most provably
unavailable to a static table. Coarse granularity (7 classes + none) so each
class is seen often and stays legible in probing; `frozen.dep_granularity`
switches to the full ~45-label set in one line. Injecting the full set at once
was considered and deferred: one fact keeps the result unambiguous and the
ladder legible.

### The structural payoff

Context facts do not require morphology. A BPE token sits inside a word, and
that word has a dependency role. Aligning BPE tokens to spaCy tokens by
character offset means **BPE can take context-freezing too**, which converts
the matrix from an L-shape into a genuine 2×3:

| | no freeze | static freeze | context freeze |
|---|---|---|---|
| **bpe** | ✓ anchor | — | ✓ newly possible |
| **morph** | ✓ | ✓ | ✓ |
| **det** | ✓ | ✓ | ✓ |

This **decouples the two legs of the thesis**. Injection can now be tested
independently of segmentation, and the interaction becomes answerable: does
linguistic freezing help *more* when the units are already linguistic?

### An unresolved fork, deliberately left open

A word maps to several morphemes, so a word-level fact must be distributed:
stamp every morpheme with the word's dependency role, or only the root? §4
flagged this and never resolved it. It is wired into `BasePreprocessor` as
`stamp='all' | 'root'` and left as a genuine comparison rather than a guess.

---

---

## 16. Restructuring — Reversing the Flat-Layout Decision

§10 chose a flat 7-file layout, and its stated justification was
**communication overhead in a collaborative development context**: deep
hierarchies are expensive when files are pasted into a conversation one at a
time.

That premise expired. A packing script reduced sharing to one command, and the
work ahead is a matrix across branches rather than debugging one pipeline.
Recording this as a *premise change* rather than drift matters: the original
reasoning was correct for its conditions, and the conditions changed.

### The concrete trigger

Two bugs the flat structure was about to produce:

1. **Results overwrite on seed.** Results were written to
   `results/{pipeline}_{variant}.json`, with no seed in the identity. Running
   a second seed silently clobbered the first — exactly the run needed for
   effect-size confidence.
2. **Cache staleness on features.** The cache key was
   `(pipeline, split, max_articles)`. Once the stream carries annotation codes,
   that key cannot express which annotation produced it: change the stamping
   rule, get a cache hit on stale data, train on silently wrong features. No
   crash, no error — just an invalid result that looks fine.

A third symptom of the same disease: `max_steps` had **three different
defaults** in one codebase (2000, 40000, 50000), and `max_articles_train` had
two. That is how the learning-rate confound below stayed invisible.

### The learning-rate confound

The cosine schedule was parameterized by `max_steps`:

```python
prog = (step - warmup) / max(max_steps - warmup, 1)
```

So `max_steps` was never merely a ceiling — it defined the entire LR
trajectory. Two cells run to different lengths had **different learning rates
at the same step** and were not comparable. The 40k calibration run and a 50k
matrix cell were silently different experiments.

**Decision: split the field.** `max_steps` decides when to stop; `lr_horizon`
is the cosine denominator and is pinned across all cells. Config validation
refuses `lr_horizon < max_steps`. Cells of different length are now comparable
by construction, which removes the constraint the old code forced on us.

### What changed

| area | before | after |
|---|---|---|
| config | defaults scattered across modules | one `ExperimentConfig` tree, one default per knob |
| LR schedule | cosine tied to `max_steps` | `lr_horizon` separate and pinned |
| seeding | **none anywhere** | seeds torch/numpy/random/dataloader; `train.seed` |
| run identity | `{pipeline}_{variant}.json` | `results/<config-hash>/` with `config.json` |
| cache key | pipeline+split+cap | ids key unchanged; codes keyed separately |
| frozen dims | fixed 23-dim schema classes | composed blocks, T derived not asserted |
| context facts | impossible | `dep` / `pos` / `ectx` per position |
| BPE freezing | impossible | possible via character-offset alignment |

On seeding: there was none, so runs were not reproducible, and — worse — a
single unseeded run per cell cannot distinguish a 1–3% difference from
initialization noise. Note that seeding every cell identically does *not* give
matched initialization across cells, since different vocabulary sizes draw
different numbers of random values. Same seed buys reproducibility; multiple
seeds buy variance control.

### Two efficiency findings

**spaCy was already parsing dependencies.** `spacy.load()` enables tagger,
parser and NER by default, so the morpheme pipelines have always computed
dependency parses and discarded them. Context facts are therefore close to free
for morph/det; only BPE pays a genuinely new spaCy cost. The cache now computes
*all* context facts on whichever pass happens first for spaCy-native pipelines,
so the baseline pass and every later fact request share one traversal.

**MorphoLex was parsed twice per pipeline construction.** Both
`build_morpholex_lookup` and `extract_morpheme_sets` independently parsed the
~70k-row workbook. Now cached in-process and on disk, keyed on the file's mtime
and size.

### Verification

The refactor was validated before any GPU time: 7/7 checks passing, including
the freeze guarantee (frozen buffers byte-identical after an optimizer step
while the trained part moves) and — on the real pipeline — **3 token ids
receiving multiple distinct dependency codes**, the direct empirical
confirmation that context facts carry information a static table cannot.

The pre-refactor state is preserved in git as the calibrated apparatus that
produced the BPE anchor (64.9 ppl / 1.438 BPB), so the three baseline cells can
serve as a behaviour-preserving regression test.

---

---

## 17. The Information Channel Audit — Making the Comparison Fair

The first Phase-1 results arrived and looked like a clean win: morph 1.3268 BPB
and det 1.3271 against a BPE anchor of 1.4162, a 6.3% advantage that appeared by
step 5,000 and held flat to 50k. The parameter asymmetry (+51% for the morpheme
pipelines, from a larger vocabulary hitting both the embedding and the untied
output head) was noted as a limitation, and the result looked ready to write up.

It was an artifact. Finding that out took four audits and reversed the sign of
the headline number.

### The question that started it

Bits-per-byte divides the model's total surprise by the byte count of the
**original text**. That normalisation is only fair if both token streams encode
the same information. BPE is lossless by construction — concatenating its tokens
reproduces the input. The morpheme pipelines might not be, and the journey's own
tables said so:

```
cats       -> cat            the plural -s is gone
running    -> run            the progressive -ing is gone
beautiful  -> beauty + ful   concatenates to "beautyful"
```

§7 celebrated MorphoLex's lemma normalisation as solving morphophonology "for
free". For a tagger that is a feature. For a language model scored in BPB it is
a liability: a stream that cannot reconstruct the bytes is being credited with
bytes it never predicted.

### Audit 1 — reconstruction (and what it got wrong)

The first audit asked whether concatenating a word's morphemes reproduces the
surface form. Result: **18.06% of words could not be rebuilt**, 72% byte
recovery.

That number, and the correction estimated from it, turned out to overstate the
problem — and understanding *why* sharpened the whole investigation. The test
that matters for BPB is not reconstruction but **injectivity**: do two
*different* texts map to the *same* token sequence?

```
cat  -> [cat]  and  cats -> [cat]      COLLISION. Information destroyed.
beautiful -> [beauty, ful]             concatenation fails, but no OTHER word
                                       maps here, so a decode table recovers it
```

BPE's merge table is side information too, and nobody counts it against BPE. A
morpheme→surface table is the same kind of object. What cannot be excused is two
distinct inputs producing one output. This distinction retired the reconstruction
audit in favour of a collision audit, and it is the single most useful conceptual
move in this section.

### A real bug found on the way

The reconstruction failures included a class where whole morphemes vanished:
`highway → high`, `touchdowns → touch`. Investigation found
`parse_morpholex_segm` collecting roots with `re.search` — first match only —
while MorphoLex writes compounds with several parenthesised roots:
`{(high)(way)}`, `{(book)(keep)}>er>`.

Same family as the format-B bug of §7: a regex matching one instance of
something that occurs many times. **8,453 entries affected (12.3% of MorphoLex)
— larger than §7's 4,941.** The fix recovered Greek/Latin combining forms as
well as compounds: `log` (267 occurrences), `graph` (158), `meter`, `amphi`,
`alti`. So `photograph` now shares `graph` with `telegraph` — morpheme sharing
that is exactly the mechanism the thesis argues for, and that was invisible
before. Vocabulary grew by 83 roots.

### Audit 2 — collisions

Grouping surface forms by token-ID sequence and summing −log2 P(surface | seq)
gives the exact context-free correction. First measurement: **0.0440 BPB**, with
2,064 colliding groups covering 28.5% of word occurrences — almost entirely
inflectional paradigms:

```
{include:40, included:88, includes:18, including:140}
{year:120, years:108}
{make:49, made:123, makes:8, making:23}
```

### Audit 3 — coverage, and the channel nobody had looked for

The collision audit's output contained a group that made no linguistic sense:
punctuation and digits appearing as surface forms. Tracing it led to a line in
`encode()`:

```python
if not spacy_tok.is_alpha:
    continue
```

**The morpheme pipelines emitted nothing at all for punctuation, digits or
symbols.** BPE's pre-tokenizer splits on word/non-word boundaries and *does*
emit them. Both were divided by the same byte count.

Measured: **det covered 1.9% of punctuation bytes and 5.5% of digit bytes
against BPE's 99.5% and 99.7% — an asymmetry of 22,480 bytes, 5.32% of the
corpus.** Unlike the collision problem, which is a shared limitation in kind,
this was asymmetric between the two arms of Q1, so it biased the comparison
directly. It was also larger than everything else found.

At that point the ledger said the advantage was gone: coverage bias ~0.092 plus
collision 0.044 = 0.136 BPB against an observed gap of 0.089.

### The six channels

The audits together identified every way the morpheme stream differed from the
text it was scored against:

| # | channel | measured | lossy? | fix |
|---|---|---|---|---|
| 1 | punctuation & digits dropped | 5.32% of corpus | yes | emit as tokens |
| 2 | inflection collapsed | 0.0440 BPB | yes | explicit `<plur>`/`<past>`/… |
| 3 | case lowercased | asymmetric | yes | `<cap>`/`<upper>` marker |
| 4 | whitespace dropped | 18.8% of bytes | by BOTH | `<sp>`/`<nl>`/`<tab>` |
| 5 | orthographic (`beauty+ful`) | injective | **no** | decode table, document it |
| 6 | UNK as collision sink | 645 occurrences | yes | UTF-8 byte fallback |

Channel 5 is the one that needed *no* change, and saying so matters: the
lemma-normalised root is what makes `beauty` shareable across *beauty,
beautiful, beautify, beautician*. That is the thesis mechanism, and it does not
collide with anything.

Channel 4 was symmetric, so it never biased Q1 — but it meant neither BPB was a
true compression rate or comparable to any published figure.

### Building the information-preserving pipeline

Every channel became a `SurfaceConfig` flag rather than a hardcoded fix, so each
one's contribution can be ablated and reported separately; all-off reproduces
the legacy stream byte-for-byte, and a test asserts it. On the BPE side the
tokenizer was retrained with a **ByteLevel** pre-tokenizer (GPT-2 standard,
lossless by construction), with a round-trip check that refuses to save a
tokenizer that cannot reconstruct its input.

Two rounds of measurement then drove the remaining work:

- Inflection from spaCy's POS features alone missed cases where the tag hides
  the inflection — gerunds tagged NOUN (`gaming`), participles ADJ (`seasoned`),
  and MorphoLex root mismatches invisible to any feature (`news` → root `new`).
  A `residue_marker` deriving the marker from the trailing surface difference
  fixed all of them, and — importantly — is deterministic on the surface string,
  so injectivity is guaranteed rather than hoped for. It deliberately does *not*
  fire on channel 5 cases.
- UNK turned out to be a collision sink: every unknown string mapping to one id.
  En-dash, em-dash, degree and pound all collapsed together. A 256-row UTF-8
  byte fallback makes the encoding **total**, so injectivity no longer depends
  on the vocabulary happening to cover the input.

Final vocabulary: 17,151 entries across twelve categories.

### Result

**Collision correction 0.0440 → 0.0008 BPB.** Colliding occurrences 28.5% →
0.75%. Coverage asymmetry 5.32% → −0.02% (det now covers marginally *more* than
BPE). The stream is information-preserving.

The residual collisions are findings rather than bugs, and belong in the thesis
as stated costs of lemma-normalised, database-driven segmentation:

- **dialectal normalisation** — `{honor, honour}`, `{colour, color}`,
  `{travelled, traveled}`: MorphoLex stores American spellings
- **homograph conflation** — `{found:57, founded:10}`, `{surveillance, survey}`:
  one root for two words; no surface rule can separate them
- **multi-word entity spans** — whitespace inside an entity span is dropped

### What this section is actually worth

Every number in the project before this point was void, and roughly a week of
measurement produced no new capability. But the methodological finding
generalises well beyond this pipeline:

**Cross-tokenizer BPB comparisons are invalid unless both streams are
collision-free and cover the same bytes.** Tokenizer papers report BPB routinely
without checking either. Here is a worked case where a plausible morpheme
tokenizer appears to beat BPE by 6.3% and, once the comparison is made
information-fair, loses — with the four audit scripts that find it.

A note on how the estimates went, because the honest record matters: the
correction was estimated three times before it was measured (0.015–0.055, then
0.0275, then "70% artifact"), and each estimate was made against a measurement
that later proved flawed. Only the fourth figure — taken after coverage was
closed *and* the grouping bug in the audit itself was fixed — was trustworthy.
The lesson is the same one §14 recorded: build the check, do not reason about
what the check would say.

---

---

## 18. The Information-Fair Rerun — Q1 Falls, Q2 Holds

With the pipeline made information-preserving (§17), every earlier number was
void and the whole matrix was re-measured. The two legs of the thesis came apart
cleanly, and in opposite directions.

### Q1 — segmentation: the advantage was an artifact

| cell | old (lossy) | new (information-fair) |
|---|---|---|
| bpe | 1.4162 | **1.3946** |
| morph | 1.3268 | 1.5292 |
| det | 1.3271 | 1.5246 |
| morph − bpe | −0.0894 | **+0.1346** |

Morpheme segmentation went from 6.3% better to 9.7% worse. A prediction made
before the run was wrong in an instructive way: ByteLevel BPE was expected to
get *worse*, since it now pays for whitespace the old Whitespace pre-tokenizer
skipped. It improved instead (1.4162 → 1.3946) at essentially unchanged token
count, because folding the space into `Ġthe` encodes 18.8% more information for
free. Morph encodes the same information at **+104% tokens**.

### Locating the loss

A per-category bit accounting was run on the saved checkpoints rather than
guessing. A prior hypothesis — that whitespace tokens explained the gap — was
refuted by measurement: `<sp>` costs **0.245 bits/token**, not the 0.721 the
hypothesis required, and whitespace is only 2.9% of morph's bits. Removing it
entirely still leaves morph behind (1.4656 vs 1.3946).

The real distribution:

| category | % of bits | bits/token |
|---|---|---|
| root | 36.6% | 8.472 |
| **character** | **24.8%** | 2.654 |
| function | 16.1% | 3.996 |
| infl | 1.6% | **0.807** |
| suffix | 1.2% | **1.622** |
| space | 2.9% | 0.245 |

Two findings sit in that table. **Morphological tokens are cheap**: inflection
at 0.807 bits and suffixes at 1.622 are far below any BPE token, so the model
demonstrably learns and exploits the morphological structure. And **character
fallback is the loss**: it handles 14.1% of the corpus at 2.654 bits/byte,
nearly double both morph's own average and BPE's. Priced against plausible BPE
rates on rare text, that excess accounts for 56–155% of the entire gap.

So the correct statement is not "morphological segmentation is worse". It is:
*where the pipeline can analyse a word it does well; where it cannot, it spells
the word out one character at a time and pays double.*

The OOV audit then identified the population precisely: 13.07% of words fall to
characters, and **77.5% of those are capitalised** — `Meridian`, `Tikal`,
`Radcliffe`, `Kawaguchi`. Named entities the frequency-gated entity vocabulary
(§6, built over 6k articles with `freq>=10` for single words) never admitted.
This is a corpus-coverage limit, not a morphological one, and it is the
weakness §8, §10 and §10 already described qualitatively — now quantified and
priced.

### Q2 — freezing: a clean positive, with the control that proves it

The frozen-embedding leg was run for the first time, injecting **one** context
fact: coarse dependency role, 8 dimensions carved OUT of the 128 (not added).

| pipeline | baseline | shuffled dep | real dep |
|---|---|---|---|
| morph | 6.30 | 6.51 | **6.02** |
| det | 6.25 | 6.45 | **5.97** |

| comparison | morph | det |
|---|---|---|
| shuffled vs baseline | +3.28% | +3.19% |
| real dep vs baseline | **−4.46%** | **−4.58%** |
| real dep vs shuffled | **−7.49%** | **−7.53%** |

The effect appears by step 5,000 and holds in **91 of 91 evaluations** in both
pipelines, and it is achieved with **2.63% FEWER trainable parameters**
(5,080,552 vs 5,217,792) — 8 dimensions of guaranteed dependency role beating
8 dimensions of free parameters.

**The shuffled control is what makes this claimable.** Permuting the codes
within each window preserves their distribution, the frozen width and the
parameter count, removing only the alignment to tokens. Shuffled codes make the
model *worse than baseline*, so the gain cannot be attributed to "any fixed
structured input helps" or to a regularisation effect. `dep` vs `dep-SHUF`
isolates the linguistic content exactly: **7.5%**.

### The mechanism is the predicted one

Per-frequency-bucket comparison, scored from checkpoints:

| train frequency | % of val tokens | morph | det |
|---|---|---|---|
| 100–1k | 0.45% | −14.53% | −14.66% |
| 1k–10k | 2.91% | −14.80% | −15.19% |
| 10k–100k | 9.66% | −13.14% | −14.27% |
| 100k–1M | 10.95% | −10.13% | −11.07% |
| 1M+ | **75.97%** | −1.95% | −1.87% |

Monotonic in both pipelines: rare tokens benefit **~7.5× more** than frequent
ones. This is exactly the sample-efficiency argument from §15 — a token seen
rarely gets little gradient signal, so its trained row is noise, while a frozen
fact about it is correct from step 0. Because 76% of validation tokens sit in
the highest-frequency bucket where the effect is smallest, the headline 4.5%
substantially *understates* the mechanism.

### The limitation that must be stated

`spacy_tok.dep_` comes from a parser that has seen the **whole sentence**. The
language model is autoregressive, so the dependency code at position *i* encodes
information about tokens *i+1..N*. When the model sees `cat[nsubj]`, that label
implies a verb is coming.

This is an **oracle-feature condition**, and the result is therefore an upper
bound on what causally-available linguistic features could provide. That is a
standard and defensible framing in the literature, but it has to be declared
rather than discovered by a reader. A causal variant — features derived from a
prefix-only parse — is the natural follow-up and is left as future work.

Note what the shuffled control does and does not settle: it proves the gain is
*information*, not structure or regularisation. It does not separate locally
available linguistic information from lookahead. Comparing `dep` against `pos`
(far more locally determined) is the cheap partial probe, since both code sets
are already cached.

### Where this leaves the thesis

The two legs separated. The **segmentation** claim — borrowed from the tokenizer
literature — does not survive an information-fair comparison in its current
form, and the reason is localised to OOV handling rather than to morphology. The
**representation** claim, which is what the title was always about, produced a
replicated 4.5% overall gain (7.5% against its own control), concentrated in
rare tokens exactly as predicted, at a lower parameter count.

---

---

## 19. Localising the Q1 Deficit — Coverage, Decomposition, and the Unified Finding

§18 established that morpheme segmentation loses to BPE by 0.1346 BPB under an
information-fair comparison, and that character fallback consumed 24.8% of
morph's bits for 14.1% of the corpus. That located the loss but did not explain
it. Three questions remained:

1. How many words actually fall through, and what does each pipeline spend on
   them?
2. On words morph *can* analyse, is it competitive?
3. If not, does the deficit depend on how frequent the word is?

All three are answered below on 400 WikiText-103 validation lines, scoring the
saved baseline checkpoints (det `0e7cd8838192`, bpe `03b9423bfe83`).

### A methodological note on the measurement itself

Restricting a comparison to words one pipeline can handle is **selection on the
outcome**. It cannot support "our tokenizer is better"; it can only support a
claim about *where* a deficit comes from. That framing was fixed before the
measurement was run, and the falsifiable prediction stated in advance: if morph
loses even on covered words, coverage is not the explanation and vocabulary
expansion will not fix it.

Two bugs in the analysis code were found and fixed before the numbers below
were trusted, both of the same kind — **misattribution during alignment**:

- BPE tokens were being assigned to a text region by their *start offset*.
  ByteLevel folds the preceding space into the token (`Ġcat` spans `" cat"`), so
  nearly every word-initial BPE token was charged to the whitespace region. This
  produced the impossible reading of BPE at 0.1354 BPB on words and 4.6757 on
  punctuation. The fix was structural: attach whitespace to the word it
  precedes, making the partition symmetric across the two encoding schemes, and
  distribute each token's bits *proportionally* across its span rather than by a
  single offset.
- `freq_bucket` sent words unseen in the frequency sample to the *last* bucket
  instead of the first, which would have placed every unseen word in the "most
  frequent" band and inverted Part 3 entirely.

Both were caught by numbers that were impossible rather than merely surprising.
The same lesson as §14 and §17: build the check, and treat alignment logic as
the default suspect.

### Part 1 — tokenization economics

| word class | words | % | bytes | morph tok/word | BPE tok/word |
|---|---|---|---|---|---|
| analysed cleanly | 19,736 | 87.8% | 93,698 | 1.51 | 1.18 |
| character fallback | 2,744 | 12.2% | 26,117 | **9.54** | 3.20 |

Fallback words are 12.2% of words but **21.8% of word bytes** — the
unanalysable words are the long ones. At 9.54 tokens for an average 9.5-byte
word, morph is spending roughly one token per byte: pure character spelling. It
uses **2.98× the tokens BPE does** on those words.

### Part 2 — restricted bits-per-byte

Both models see the full text as context; only the scored positions are
partitioned.

| region | bytes | morph BPB | BPE BPB | delta | share of gap | morph worse by |
|---|---|---|---|---|---|---|
| words analysed cleanly | 113,240 | 1.2672 | 1.1937 | +0.0735 | 35.5% | 6.2% |
| words in char fallback | 28,774 | 2.4076 | 2.0172 | +0.3904 | **47.9%** | 19.4% |
| punct / digits | 13,809 | 1.8004 | 1.5189 | +0.2815 | 16.6% | 18.5% |
| **ALL** | 155,823 | **1.5250** | **1.3746** | **+0.1505** | 100% | 10.9% |

**The coverage hypothesis is half right.** Fallback words are 18.5% of bytes but
contribute 47.9% of the gap — disproportionate, exactly as predicted. But morph
loses in *every* region, including words it analyses perfectly, so coverage is
not a complete explanation.

Two extrapolations bracket what a vocabulary rebuild could buy:

- **optimistic** — fallback words become as cheap as clean ones:
  morph 1.3145 vs BPE 1.3746, **morph wins by 0.0601**
- **realistic** — fallback words merely match BPE's own rate on them:
  morph 1.4529 vs BPE 1.3746, **morph still loses by 0.0783**

The truth lies between. Newly covered words would be rare ones, harder than the
average covered word, so the optimistic figure is a ceiling and not a forecast.

### Part 3 — the deficit depends on word frequency, and not monotonically

Clean words only, stratified by frequency in a 40,000-line training sample
(~1/30 of the corpus, so these bands map to roughly 30× higher full-corpus
counts).

| word frequency | bytes | % of clean bytes | morph BPB | BPE BPB | delta | verdict |
|---|---|---|---|---|---|---|
| 1–10 | 5,475 | 5.8% | 1.8925 | 2.1707 | **−0.2782** | **morph wins by 12.8%** |
| 10–100 | 16,552 | 17.7% | 1.6822 | 1.6731 | +0.0091 | tie |
| 100–1k | 33,732 | 36.0% | 1.4418 | 1.3049 | +0.1369 | morph loses by 10.5% |
| 1k–10k | 18,697 | 20.0% | 1.4687 | 1.4038 | +0.0649 | morph loses |
| 10k–100k | 13,338 | 14.2% | 1.5668 | 1.5660 | +0.0008 | tie |
| 100k+ | 5,904 | 6.3% | 0.7280 | 0.7451 | −0.0171 | morph wins slightly |

The shape is a **U**, not a monotone. **On the rarest words morpheme
segmentation beats BPE by 12.8%** — where BPE has no good whole-word merge and
must hack the word into arbitrary fragments, morphological decomposition shares
statistical strength across the family and wins clearly.

The loss is concentrated in the **middle band**: words frequent enough that BPE
has learned a whole-word token, but not so frequent that prediction is trivial.
These are inflected content words — `played`, `including`, `yards`. BPE spends
one token; morph spends `play + <past>`. That band holds 56% of clean-word
bytes.

At the very top (100k+, essentially function words) both pipelines have single
well-trained tokens and the difference vanishes.

### The unified finding

The two legs of the thesis, measured by completely different mechanisms, agree:

| | rare items | frequent items |
|---|---|---|
| **Q2** — frozen dependency role vs none | −14.5% | −1.9% |
| **Q1** — morpheme segmentation vs BPE | −12.8% | +10.5% |

> **Linguistic structure helps exactly where statistical learning is
> data-starved, and costs where statistics are already sufficient.**

This is a stronger and more useful claim than "our tokenizer wins", because it
says *when* and *why* rather than reporting a single number. It also explains
the whole shape of the project's results: the frozen-embedding gain concentrates
in rare tokens (§18), the segmentation gain concentrates in rare words (here),
and both dissolve where the data is plentiful.

### What the finding implies for design

The deficit has **two independent causes with two independent fixes**:

1. **Coverage** (47.9% of the gap) — 77.5% of fallback words are capitalised
   named entities the frequency-gated entity vocabulary never admitted (§18).
   Fix: rebuild entity extraction over more articles and revisit §6's
   single-word threshold, which was tuned on 6k articles and never adjusted.
2. **Unconditional decomposition** (the middle-band loss) — the pipeline
   decomposes every word, including frequent inflected forms that BPE handles
   in a single well-trained token. Fix: **frequency-aware decomposition** —
   admit the most frequent inflected surface forms to the vocabulary as whole
   entries and decompose only below that threshold. Losslessness is preserved,
   because the whole form is stored rather than reconstructed.

The second is the more interesting one: it is what BPE does statistically
without knowing any linguistics, and doing it with a principled threshold plus
linguistic decomposition for the tail is a design that follows *from the
measurement* rather than from intuition.

---

---

---

# Part V — Grounding and Controls

*Having reached a negative result, two things were needed: to find out whether
the field already knew it, and to remove the confounds that remained. Both
changed the conclusions — the literature showed the negative result was
expected for English, and the controls refuted the project's own headline
mechanism while leaving its effect intact.*

---

## 20. Literature Grounding — Who Has Done This Already

Up to this point the project had been developed almost entirely from first
principles, with the literature consulted only for the bits-per-byte lineage
(§12) and the calibration reference points (§13). With both experimental legs
now measured, it became urgent to find out whether the field had already
answered these questions — partly to avoid claiming a known result, partly to
learn from other people's mistakes rather than repeating them.

Five primary sources turned out to matter.

### A. MorphPiece (Jabbar, 2023) — our Q1, built at scale

A tokenizer that looks each word up in a morpheme dictionary and, when the
lookup fails, falls back to BPE.

| | this project | MorphPiece |
|---|---|---|
| morpheme source | MorphoLex, 68,608 words | **MorphyNet, 346,340 words** (134,943 after a freq≥5 trim) |
| OOV handling | character fallback | **BPE fallback** |
| segmentation form | canonical / lemma | canonical / lemma |
| vocabulary | 17,151 | 18,304 morphemes + 32,000 BPE = 50,006 |
| model | 3.4–5.2M, from scratch | GPT-2 Base, 117M |
| corpus | WikiText-103 | OpenWebText |

MorphGPT beat GPT-2 on Penn Treebank perplexity (61.9 → 38.3), on LAMBADA, on
zero-shot GLUE and across all seven MTEB embedding tasks, in roughly half the
training steps.

**Three details in their paper are our findings, arrived at independently:**

1. They use canonical (lemma) segmentation — `batting` → `bat` + `ing`, not
   `bat` + `ting` — and name it as a likely source of their gains. Identical to
   our MorphoLex-based design.
2. Their tokenizer produces **17% more tokens than BPE**. They list this as a
   limitation and leave "fusing morphemes to reduce these extra tokens" to
   derivative work. That is exactly the mid-frequency deficit §19 measured.
3. Their limitations note that MorphyNet does not cover all lexical families —
   our coverage problem, at five times our dictionary size.

**And one detail is a gap we can fill.** Their Section 7 acknowledges that
token-level perplexity is not comparable across vocabularies, citing Cotterell
et al. (2018), and then argues for using it anyway on the grounds that
downstream gains corroborate it. That is honest, but it is precisely the hole
our audits (§17) close.

### B. Hou, Katinskaia, Vu & Yangarber (2023) — the most useful paper for us

Not a new tokenizer, but a careful controlled comparison: BPE against two
unsupervised morphological segmenters (Morfessor and StateMorph), across four
languages, two architectures, several vocabulary sizes and two model sizes.
Their four research questions were perplexity, convergence speed, downstream
performance, and performance at reduced model size.

**The English numbers changed how this project should be read:**

| language | BPE | morphological | outcome |
|---|---|---|---|
| **English**, vocab 151k | 19.82 | Morfessor **20.05** | **BPE wins** |
| **English**, vocab 177k | 20.33 | StateMorph 18.93 | morph wins 7% |
| **English**, vocab 107k | 17.92 | StateMorph-pruned 17.55 | morph wins **2%** |
| **Finnish**, vocab 78k | 23.69 | StateMorph-pruned **16.65** | morph wins **30%** |
| **Turkish**, vocab 60k | 12.93 | StateMorph-pruned 10.67 | morph wins **17%** |

**For English the benefit is small or negative; for morphologically rich
languages it is large.** Our Q1 negative result is therefore not anomalous — it
is the expected outcome for the language we chose, published independently.

Two further items we adopted:

- **Vocabulary matching.** They deliberately tune BPE's vocabulary to match the
  morphological segmenter's, stating it is "only to make the comparison fair."
  We had documented a +51% parameter advantage to morph since §18 and never
  removed it. Acted on in §21.
- **Word-initial marking** (their Appendix A.1). BPE marks word-initial tokens;
  morphological segmenters do not, so they mark all word-initial tokens with a
  symbol, as BPE does. This is precisely the fix for our `<sp>` overhead —
  32.6% of all morph tokens.

They also make convergence speed a first-class metric, which we had never
computed despite having the data (rectified in §21).

### C. Bostrom & Durrett (2020) — the standard citation

Compared BPE against Unigram LM tokenization using identical transformer
models, on English and Japanese. Unigram recovers subword units that align more
closely with morphology, avoiding artifacts of BPE's greedy merging, and
matches or outperforms BPE downstream. This is the reference everyone cites for
"BPE mis-segments morphology" and is the paper our Q1 must be positioned
against.

### D. FLOTA (Hofmann, Schütze & Pierrehumbert, 2022) — the cheap baseline

Uses the **existing BPE vocabulary** but changes how words are split,
recursively taking the longest vocabulary substring, so `undesirable` becomes
`un` + `desirable` rather than `und` + `es` + `irable`. No new vocabulary, no
retraining. A very cheap baseline sitting between our two arms — statistical
vocabulary, linguistic segmentation — and one a reviewer may ask why we omitted.

### E. Alexandrescu & Kirchhoff (2006), *Factored Neural Language Models* — the ancestor of our Q2

A word is represented not as one symbol but as a **set of features** — stem,
affixes, POS, capitalisation — each with its **own embedding**, concatenated to
form the input vector. Introduced specifically to help with rare words, extended
to neural machine translation by Sennrich & Haddow (2016).

**Architecturally this is our Q2 exactly.** The difference is that their factor
embeddings are **learned** and ours are **frozen**. In the sources reviewed no
one appears to have compared the two, nor run a shuffled-feature control. That
observation set the agenda for §21.

### What this review established

| | status |
|---|---|
| Q1, morpheme segmentation vs BPE | **well-trodden** — four papers, several with far more compute |
| Q1's English negative result | **expected and citable**, not an anomaly |
| Our audit methodology | **unoccupied** — MorphPiece names the problem and proceeds anyway |
| Frequency-stratified localisation | **unoccupied** — nobody partitions the comparison by word frequency |
| Q2, frozen vs learned factors | **untested** — the obvious next experiment |
| Shuffled-feature control | **unoccupied** |

The conclusion was uncomfortable but clarifying: **the tokenizer is not the
contribution.** The instrumentation and the frozen-factor question are.

---

---

---

## 21. Completing the Controls — Matched Vocabulary, Frozen vs Learned, and Signal Scale

The literature review (§20) produced three concrete experiments, all cheap, all
addressing confounds we had documented but never removed. Each changed the
conclusion.

### 37.1 Matched vocabulary — the Q1 confound, finally removed

Since §18 the morpheme pipelines had carried a **+51% parameter advantage**: a
17,151-entry vocabulary against BPE's 10,000, hitting both the embedding and the
untied output head. Hou et al. match vocabularies explicitly to make the
comparison fair; we had only documented the asymmetry.

Retraining the ByteLevel BPE tokenizer at 17,151 took under a minute.

| cell | vocab | trainable params | val tokens | bytes/token | BPB |
|---|---|---|---|---|---|
| BPE 10k | 10,004 | 3,387,136 | 270,848 | 4.195 | 1.3946 |
| **BPE 17k** | 17,155 | **5,217,792** | 252,160 | 4.507 | **1.3477** |
| morph | 17,155 | **5,217,792** | 654,592 | 1.737 | 1.5292 |
| det | 17,155 | **5,217,792** | 655,360 | 1.734 | 1.5246 |

Parameter counts are now byte-identical.

**Removing the confound made Q1 worse, not better.** BPE improved with the
larger vocabulary (1.3946 → 1.3477), so the controlled gap widened from +0.1346
(9.7%) to **+0.1815 BPB, 13.5%**.

There is a structural asymmetry worth stating in the write-up: **BPE's
vocabulary size is a free hyperparameter that buys performance; a morpheme
vocabulary is fixed by the linguistics.** BPE can always spend its way up. The
morpheme pipeline cannot.

### 37.2 Frozen vs learned factors — the thesis claim, tested directly

Factored NLMs learn their factor embeddings. This project freezes them. Nobody
had compared the two, so the comparison had to be built: `frozen.trainable`
selects between a fixed identity matrix (a registered buffer) and an
`nn.Embedding` of identical shape. Both index the same codes and occupy the same
8 dimensions.

The learned table adds **64 parameters — 0.0013%** — so the comparison is
controlled by construction, and `verify.py` now asserts it stays under 0.1%.

| condition | morph | det | vs baseline |
|---|---|---|---|
| shuffled dep | 6.51 | 6.45 | **+3.2%** (worse) |
| baseline | 6.30 | 6.25 | — |
| frozen dep @ scale 1.0 | 6.02 | 5.97 | −4.5% |
| frozen dep @ scale 0.22 | 6.00 | 5.94 | −4.9% |
| **learned dep** | **5.88** | **5.87** | **−6.4%** |

**Learned factors beat frozen ones**, by 2.29% (morph) and 1.68% (det).

### 37.3 The scale confound — found in our own implementation

Before accepting that result, an arithmetic check on the implementation:

| component | L2 norm at initialisation |
|---|---|
| trained embedding slice (120 dims, std 0.02) | 0.219 |
| positional embedding (128 dims, std 0.02) | 0.226 |
| **frozen factor (one-hot, 8 dims)** | **1.000** |
| learned factor (8 dims, std 0.02) | 0.057 |

**The frozen one-hot entered the residual stream 4.6× louder than the entire
trained embedding**, and pre-norm LayerNorm then normalised the sum, squashing
everything else. A learned table initialises at std 0.02, is balanced by
construction, and can additionally adjust its own magnitude during training.

So the frozen-vs-learned comparison as first run confounded **trainability**
with **scale** — a defect of our implementation, not of the hypothesis. A
`frozen.scale` parameter was added and the comparison rerun at 0.22, matching
the trained slice.

**Result: the confound was real but minor.** Rescaling gained 0.02 perplexity in
both pipelines, closing **16% (morph) and 23% (det)** of the gap to learned. The
remainder is genuine trainability.

**Conclusion: the specific claim that freezing beats learning is refuted**, and
now refuted under a properly controlled comparison rather than resting on an
initialisation accident.

### 37.4 What survives, and what it means

The ladder is complete, monotonic, and replicated across two pipelines:

```
shuffled dep   +3.2%   worse than baseline
baseline        0.0%
frozen dep     -4.9%
learned dep    -6.4%
```

Read carefully, this is a strong object even though the headline claim fell:

- **Injecting the dependency role helps a great deal** — 6.4% at the top of the
  ladder, and the effect is replicated.
- **The shuffled control proves the gain is linguistic content**, not the mere
  presence of extra structured input. Permuting the codes — same distribution,
  same width, same parameter count — makes the model *worse than baseline*.
- **Freezing captures roughly 75% of the available benefit at zero parameter
  cost**, which is a real trade-off rather than a failure.
- **Trainability adds the remaining 25%**, plausibly because a learned table can
  place the eight dependency classes in a geometry reflecting their similarity,
  whereas one-hot forces them equidistant.

### 37.5 Convergence speed — a free finding

Hou et al. treat convergence speed as a first-class metric (their RQ2). We had
the data in every training log and had never looked.

| cell | steps to ppl ≤ 6.5 | steps to ppl ≤ 6.3 |
|---|---|---|
| det baseline | 29,000 | 41,500 |
| det + frozen dep | **19,500** | **25,500** |
| morph baseline | 30,500 | never reached |
| morph + frozen dep | **21,000** | **27,000** |

**Frozen dependency features reach the same perplexity in 33–39% fewer steps.**
An independent second axis on which Q2 wins, at zero cost, matching a metric the
literature already values.

### 37.6 One claim from §11 still untested

The original argument for freezing (§2, §11) had two parts:

1. **lower perplexity** — refuted above
2. **guaranteed legibility** — the fact is always present and always readable,
   at every layer, for any attention head that wants it

Part 2 has never been measured, and it is the one place freezing can still win
by construction: a frozen one-hot is trivially decodable, whereas a learned 8×8
table is free to collapse classes together if the loss does not punish it. If
frozen dimensions remain linearly readable at depth while learned ones do not,
the trade-off is 2% perplexity for guaranteed interpretability — a defensible
result rather than a loss.

`scripts/probe_dep.py` was written for this: a linear probe on the hidden state
after every layer, run against baseline, frozen and learned checkpoints. The
baseline arm is the informative one — if a probe can already read dependency
role off a model that never saw the labels, then injection is partly redundant
and the gain must come from precision or from earlier availability.

---

---

---

## 22. Why English Resists — and the Turn from Confrontation to Cooperation

By this point both legs had been measured under proper controls, and the picture
was uncomfortable: Q1 negative and widening, Q2 positive but with its
distinguishing mechanism refuted. The obvious question — *is there any point in
offering a morphological inductive bias to English at all?* — had to be answered
before deciding what to build next.

Answering it properly turned out to change the direction of the project.

### 38.1 Why English BPE is so hard to beat

**Measurement first.** From §19's tokenization economics:

| | tokens per word |
|---|---|
| BPE, clean words (88% of words) | **1.18** |
| BPE, fallback words (12%) | 3.20 |
| BPE, overall | 1.43 |
| morph, clean words | 1.51 |
| morph, fallback words | 9.54 |
| morph, overall | 2.49 |

**For 88% of English words BPE emits essentially one token.** English words are
short, the Zipf curve is steep, and a 17k vocabulary holds nearly every common
word whole. BPE is therefore not really a subword tokenizer for English — it is
a **word-level tokenizer with a subword escape hatch for the tail**.

Morphological decomposition takes a representation that already works and
fragments it. It can only help inside the escape hatch. Which is exactly the
U-shape §19 measured: morph wins 12.8% where BPE must fragment, loses 10.5%
where BPE has the whole word.

**Five structural reasons, in order of force:**

1. **Almost nothing to decompose.** Our own pipeline measures **1.51 morphemes
   per clean word**. Roughly half of English words are a single morpheme. In
   Finnish, `taloissanikin` = *talo+i+ssa+ni+kin* ("in my houses too") — one
   word, five morphemes, where English uses five separate words.

2. **Tiny paradigms.** An English noun has 2 forms; a verb 4–5; an adjective 3.
   A Finnish noun has roughly 2,000 (15 cases × number × possessive × clitic).
   This is the sparsity argument: if a noun has 2,000 forms then *every form is
   rare even when the lemma is common*, and decomposition collapses 2,000 sparse
   types into one frequent lemma plus a few frequent affixes. In English `cats`
   is common enough by itself. **There is nothing to share.**

3. **English derivation is lexicalised, not compositional.** §10 already recorded
   `demonstrated → de + monster + ate`. English derivational morphology is
   largely fossilised Latin and Greek borrowing that no longer means what it
   says, so decomposition often injects noise rather than signal. Agglutinative
   affixes mean the same thing every time.

4. **English orthography hides morphology.** `ran` ≠ `run` + past. Irregular
   verbs and suppletion mean that even where morphology exists it is not
   recoverable from the surface string — precisely the `{found, founded}`
   collision the audit caught in §17.

5. **BPE has already learned English morphology.** Our vocabulary contains
   **338 affixes** (127 prefixes + 211 suffixes), of which the productive
   inflectional set is **8**. Those eight are so frequent that BPE's merge
   process learns `-ing`, `-ed`, `-s`, `un-` automatically, with no linguistic
   knowledge at all. **For English, BPE already is a morphological segmenter.**
   Finnish has hundreds of affixes, each individually rarer, and a 32k
   vocabulary cannot absorb them — which is the gap explicit morphology fills,
   and which English does not have.

### 38.2 How much of the deficit is ours, and how much is the language

Separating implementation defects from language properties, using §19's regional
decomposition:

| cause | share of gap | status |
|---|---|---|
| character fallback | **47.9%** | fixable — MorphPiece uses BPE fallback |
| punctuation/digits emitted one per character | 16.6% | fixable — group digit runs as BPE does |
| clean words (whitespace + unconditional decomposition) | 35.5% | partly fixable |

**Most of the deficit is our implementation, not English.** But the budget is
sobering:

```
morph today                                1.5250
+ BPE fallback for OOV                     1.4529
+ word-initial marking (kills <sp> tokens) 1.4087
+ digit grouping                           1.3962
BPE-10k                                    1.3746
BPE-17k (parameter-matched)                1.3477
```

**Every realistic implementation fix lands morph at about 1.40** — level with
BPE-10k, still behind the parameter-matched BPE-17k. Implementation explains
most of the gap but not all of it. The residue is the language.

A useful consistency check fell out of this: the band-weighted clean-word delta
is **+0.0468** on word bytes alone, while §19's Part 2 measured **+0.0735**
including leading whitespace. The 0.027 difference is exactly the `<sp>`
overhead — two independent measurements agreeing.

### 38.3 The path considered and rejected — data starvation

If the mechanism is data starvation rather than Finnish specifically, the same
condition can be created in English by shrinking the corpus. Taking §19's
per-band deltas and shifting the frequency distribution (a 10× smaller corpus
moves every word down one band) gives a quantitative forecast:

| data fraction | predicted clean-word delta | |
|---|---|---|
| 100% | +0.0468 | morph loses *(matches measurement)* |
| 10% | −0.0255 | morph wins |
| 1% | −0.1402 | morph wins decisively |

The crossover is predicted between 10% and 100% of the corpus, and the design
would be sound provided BPE is **retrained at each data size** (its vocabulary
is learned from the corpus; MorphoLex is external and does not shrink).

**Rejected, on the researcher's objection**, and the objection was correct: the
experiment changes *the conditions* rather than *the method*. It would report
"morphology wins when BPE is handicapped" — a weaker claim than "morphology adds
something BPE cannot get". Low-resource modelling is a real setting and this
remains a legitimate future experiment, but it is not the thesis.

### 38.4 The reframe — stop confronting BPE

The researcher's counter-proposal: instead of competing with BPE, find where it
struggles and help it. Both methods have strengths; the question is whether they
can cooperate.

**The decisive observation: §19's frequency table is not a comparison, it is an
assignment.**

| word frequency | % of clean bytes | delta | who should handle it |
|---|---|---|---|
| 1–10 | 5.8% | −0.2782 | **morph** |
| 10–100 | 17.7% | +0.0091 | tie |
| 100–1k | 36.0% | +0.1369 | **BPE** |
| 1k–10k | 20.0% | +0.0649 | **BPE** |
| 10k–100k | 14.2% | +0.0008 | tie |
| 100k+ | 6.3% | −0.0171 | **morph** |

Each method has a measured region of competence. Nothing in the project's data
says one should replace the other.

### 38.5 Two levels of cooperation

**Level 1 — routing at the tokenizer.** Choose the better tokenizer per word:

```
entity                → entity token
frequent word         → BPE (it holds the whole word, well-trained)
rare and analysable   → morphemes (shares strength across the family)
unanalysable          → BPE fallback
```

The threshold is a free parameter, and sweeping it *is* the result: T→0 is pure
BPE, T→∞ is pure morph, so the sweep contains both baselines as endpoints and
measures where the crossover lies. MorphPiece explicitly left this to future
work ("fusing morphemes to reduce these extra tokens"), so it is unoccupied.

Budget, on top of the implementation fixes:

```
+ route the 100–10k band to BPE            1.3588
BPE-10k                                    1.3746
BPE-17k                                    1.3477
```

**Predicted to beat BPE-10k and approach BPE-17k, with nothing handicapped.**

**Level 2 — feature injection at the embedding.** Do not compete over
segmentation at all. Use **BPE tokens**, which the measurement says are better
for English, and inject **morphological facts as embedding factors**.

The logic is the project's own two results, put together:

> Q1: morphology-as-**segmentation** loses in English.
> Q2: linguistic-features-as-**embedding** wins by 6.4%.
> Therefore: **segment with BPE, inform with morphology.**

This also restores the thesis title. *Context-Aware **Embeddings*** was always a
claim about the embedding layer; Q1 was the supporting comparison, and it has now
established that segmentation is the wrong place to spend English morphology.

The mechanism already exists: §17 gave the BPE pipeline character-offset
alignment so it could carry context facts, and the same machinery carries
morphological ones. The strongest candidate is **root identity** — BPE sees
`running` as one token, unrelated to `runner`, `runs`, `ran`, because its merges
are frequency-driven rather than lexical. A root factor ties them together.

Parameter accounting, taking 16 dimensions out of the 128 exactly as `dep` took
8:

| | embedding parameters |
|---|---|
| BPE-17k baseline | 2,195,840 |
| BPE-17k + root factor | 1,921,360 + 241,680 = **2,163,040** |

**32,800 fewer parameters** — the same controlled-substitution design, so the
comparison is clean by construction. Every control built for Q2 transfers:
shuffled roots, frozen versus learned, frequency-bucket stratification.

### 38.6 Where this leaves the project

The arc is now coherent in a way it was not before:

1. Morphological segmentation was tested against BPE in English and lost.
2. The loss was localised precisely — coverage, unconditional decomposition,
   and the fact that BPE has already absorbed English's eight productive
   affixes.
3. The same morphological knowledge, delivered as **features** rather than as
   segmentation, produces a replicated 6.4% gain with a control proving it is
   the linguistic content.
4. The natural synthesis — BPE for segmentation, morphology for features — is
   the next experiment, and it is the one the thesis title always named.

**Decision: pursue Level 2 first.** It builds on the positive result rather than
rescuing the negative one, needs no new vocabulary or tokenizer, and yields the
cleaner claim: *morphological knowledge helps English language models — as
features, not as segmentation.* Level 1 remains available afterwards, and the
two compose.

---

---

---

# Part VI — The Synthesis

*The two negative findings of Parts IV and V, taken together, point at an
experiment neither of them tested: use the tokenizer that wins, and spend the
morphology somewhere else.*

---

## 23. Level 2 — Morphology as Features on a BPE-Tokenized Model

Section 38 concluded with a decision: stop competing with BPE over segmentation
and instead use BPE tokens while injecting morphological knowledge as embedding
features. This section records the build and the result.

**The result is the strongest in the project, and the first cell that beats its
own best baseline.**

### 39.1 A bug found before anything ran

The BPE + context-fact path had been written in section 17 but never executed.
Reviewing it before committing compute turned up a defect that would have
silently voided the entire experiment.

`ByteLevel` folds the preceding space INTO the following token, so a
word-initial token spans `" cat"` and its **start offset lands on whitespace** —
which no spaCy token covers. The code indexed annotations by that start offset:

```
old logic (arr[start]) : aligned 1/6 = 17%
fixed logic (scan span): aligned 6/6 = 100%
```

Word-initial tokens are **78% of the BPE stream** (211,300 of 270,848, measured
in the category bit accounting). Every BPE feature-injection run would have
received `none` for nearly every token, trained normally, and produced a
convincing null result. The cache would then have needed rebuilding.

This is the same class of error as the region-attribution bug in
`coverage_economics` (section 19) and the grouping bug in the collision audit
(section 17). **Alignment logic is now treated as the default suspect in this
codebase**, and a regression test guards it.

### 39.2 Design decisions taken to avoid re-runs

Three, all chosen specifically because a mistake would have cost a multi-hour
cache pass.

**One cache serves every BPE experiment.** As soon as any context fact is
requested, BPE now computes all six — `dep`, `pos`, `ectx`, `mstruct`, `mrole`,
`mroot` — in a single pass and writes one codes file. Computing only the
requested fact would have meant a fresh spaCy + MorphoLex pass for every new
experiment.

**Cardinality separated from width.** Root identity has ~16k classes and cannot
one-hot into 16 dimensions, so blocks now declare `n_classes` and `width`
independently. When `width < n_classes` the frozen representation becomes a
**fixed random projection** — every class gets a distinct, unchanging vector
with no parameters. That is **Lever 3 from section 11**, unbuilt since then.

**Entities carry their own root.** A preflight found 39.1% of BPE tokens sitting
in words for which no root was returned. An entity's lexical identity is itself
(`london` is its own lemma), so entity surface forms became root classes:
15,105 roots + 1,259 entities + `none` = **16,365 classes**. This only moved the
figure to 37.6%, because most of the remainder is character-fallback words, not
entities — the same coverage limit that sank Q1, resurfacing at the feature
layer. A test now asserts the block's class count matches the analyser's
inventory exactly, since a mismatch of one would make every cached code decode
to the wrong root.

### 39.3 The preflight, and what it predicted

A pre-flight script measured, on 300 validation lines and in six seconds,
everything that could void the run:

| check | result |
|---|---|
| alignment rate | **99.99%** (the old bug would show ~20%) |
| code distribution | all six facts vary; none constant |
| tokens carrying a root | **42.9%** |
| within-token-id variance, `mroot` | **11.1%** of ids receive >1 code |
| int16 safety | 16,365 < 32,767 |
| parameter arithmetic | every cell has FEWER parameters than baseline |

Two of those numbers set expectations honestly in advance. The root is
informative for only 42.9% of tokens, and it is genuinely *contextual* for only
11.1% of token ids — for the rest, `Ġrunning` always has root `run`, which a
static embedding table could already learn. **The prediction recorded before
running was 1–3%, not the 6.4% seen for `dep`.**

The cache build then took **4h37m** for 120,625,719 training tokens, at a
measured alignment rate of 99.98%.

### 39.4 The results

Baseline is the parameter-matched `bpe_17k` cell from section 21: 1.3477 BPB.

| cell | ppl | BPB | vs base | params vs base |
|---|---|---|---|---|
| + `mroot` **SHUFFLED** (learned) | 68.26 | 1.3519 | **+0.31%** | −0.24% |
| baseline (bpe 17k) | 67.36 | 1.3477 | — | — |
| + `mstruct` **SHUFFLED** (frozen) | 66.60 | 1.3440 | −0.27% | −2.63% |
| + `mroot` frozen **projection** | 63.66 | 1.3296 | −1.34% | −5.26% |
| + `mstruct` frozen one-hot | 62.25 | 1.3225 | −1.87% | −2.63% |
| + `mroot` learned | 59.42 | 1.3076 | −2.98% | −0.24% |
| **+ `mroot`+`mstruct`+`mrole` learned** | **56.72** | **1.2927** | **−4.08%** | **−4.51%** |

Monotonic, nothing overfitted, every cell still descending at the step ceiling,
and **every cell carries fewer parameters than the baseline it beats.**

### 39.5 Seed replication — the last thing being asserted rather than shown

| cell | seed 0 | seed 1 | spread | mean |
|---|---|---|---|---|
| baseline | 1.3477 | 1.3482 | **0.0005** | 1.3480 |
| all three facts, learned | 1.2927 | 1.2837 | 0.0090 | 1.2882 |

- **seed noise (max spread): 0.0090 BPB**
- **effect (mean difference): 0.0598 BPB**
- **effect / noise: 6.6×**
- **mean improvement: 4.44% BPB**

All four cross-seed pairings hold: every treatment run beats every baseline run.
The baseline's own spread of 0.0005 is remarkably tight, which is itself
reassuring about the apparatus.

### 39.6 Three findings beyond the headline

**The facts compose.** Root alone gives 0.0401 BPB; structure alone 0.0252;
combined with role, 0.0550 — **84% of what pure additivity would predict**. So
structural morphology carries information the root does not; it is not a coarse
proxy for lexical identity.

**Lever 3 works.** The frozen random projection — 16,365 classes into 16 fixed
dimensions, zero trainable factor parameters — beats baseline by 1.34% *while
carrying a 5.26% parameter handicap*. Learned still wins (−2.98%), but that
comparison is confounded by capacity, so the honest statement is "learned is
better, though by less than the raw gap suggests."

**The two shuffled controls behave differently, and the asymmetry is
explainable.**

| | shuffled vs baseline | real vs shuffled |
|---|---|---|
| `mroot` (learned) | **+0.31%** worse | −3.28% |
| `mstruct` (frozen) | **−0.27%** slightly better | −1.61% |

Shuffled roots hurt; shuffled structure marginally helps. This is the section-37
scale finding resurfacing: a **frozen one-hot enters at L2 norm 1.0 against a
trained slice of 0.219**, so even random one-hots inject a loud
constant-magnitude signal that plausibly acts as regularisation. A learned table
initialises at std 0.02, is quiet, and random codes there are simply wasted
dimensions. The control still does its job — real beats shuffled in both cases —
but `mstruct`'s should be reported as *"shuffling removes the gain"* rather than
*"shuffling hurts"*.

### 39.7 The oracle question, substantially answered

Section 37.6 flagged that `dep` comes from a full-sentence parse and therefore
leaks information about tokens not yet predicted. The concern was that most of
the Q2 gain was this oracle effect.

| feature | loss reduction | lookahead? |
|---|---|---|
| `dep` on morph | −3.75% | **yes** — full-sentence parse |
| `mroot` on BPE | **−2.98%** | **no** — the root depends only on the word |

**About 80% of the effect survives with no lookahead at all.** These are
different features on different tokenizers, so this is strong evidence rather
than proof — but the worry was that the gain was *mostly* oracle, and it plainly
is not. Morphological features are causally available during generation in a way
the dependency parse is not.

### 39.8 What the thesis now claims

> **Morphological knowledge improves English language modeling — as embedding
> features, not as segmentation.** Injected into a parameter-matched BPE model
> it reduces bits-per-byte by **4.44%** (mean of two seeds, 6.6× the seed
> noise), using **4.5% fewer trainable parameters**, with a shuffled control
> confirming the gain is linguistic content and no lookahead in the features.

The project's arc is now coherent end to end:

1. Morphological segmentation was tested against BPE in English and lost by
   13.5% under a parameter-matched comparison (sections 34, 37).
2. The loss was localised precisely: OOV coverage, unconditional decomposition
   of frequent words, and the fact that BPE has already absorbed English's eight
   productive affixes (sections 35, 38).
3. The same morphological knowledge, delivered as **features** on the better
   tokenizer, improves it by 4.44% (this section).

The negative result is not a failure that was later rescued; it is the
measurement that told us where to spend the morphology.

---

---

---

## 24. Where the Project Stands

### The finding

> **Morphological knowledge improves English language modeling as embedding
> features, not as segmentation.**

Injected into a parameter-matched BPE model, root identity, word structure and
morpheme role together reduce bits-per-byte by **4.44%** — the mean of two
seeds, against a seed noise of 0.0090 BPB, an effect-to-noise ratio of **6.6×** —
using **4.5% fewer trainable parameters** than the baseline they beat. A
shuffled-code control performs *worse* than baseline, so the gain is the
linguistic content rather than the presence of extra structure. Unlike the
dependency feature, none of these facts requires lookahead.

### How each claim is supported

| claim | evidence | section |
|---|---|---|
| morpheme segmentation loses to BPE in English | 13.5% BPB under matched parameters | §18, §21 |
| the loss is coverage plus unconditional decomposition | regional decomposition of the gap | §19 |
| the loss is expected for English specifically | Hou et al. find the same; English is the floor | §20, §22 |
| linguistic features in the embedding help | 4.44% BPB, replicated across two seeds | §23 |
| the gain is linguistic content, not structure | shuffled control is worse than baseline | §21, §23 |
| the gain concentrates where data is scarce | rare tokens benefit ~7.5× more | §19, §23 |
| freezing is not the mechanism | learned factors beat frozen under matched capacity | §21 |
| cross-tokenizer BPB needs auditing | a 6.3% "win" reversed to a 9.7% loss | §17, §18 |

### The unified observation

Two independent experiments, using different mechanisms, produced the same
shape:

| | rare items | frequent items |
|---|---|---|
| injected dependency role vs none | −14.5% | −1.9% |
| morpheme segmentation vs BPE | −12.8% | +10.5% |

**Linguistic structure helps where statistical learning is data-starved and
costs where statistics are already sufficient.** This is the claim the project
actually supports, and it explains both the positive and the negative results
rather than treating one as a success and the other as a failure.

### What is not established

Each of these is measured rather than speculated, and all are tabulated with
magnitudes in `results_reference.md` §7.

- **The dependency feature is an oracle.** It comes from a full-sentence parse,
  so it leaks information about tokens not yet predicted. Roughly 80% of the
  effect survives with a feature that has no lookahead (§23), but the
  dependency number itself is an upper bound.
- **Most cells ran at a single seed.** Only the baseline and the best Level-2
  cell were replicated.
- **No downstream task was evaluated.** Every comparable paper has one; this is
  the standard rebuttal to a perplexity-only result.
- **The probing evaluation was never run.** It was promised in §4 as the
  mechanistic half of the story and the script exists, but no classifier was
  trained. This is the largest piece of unfinished evaluation.
- **No run reached convergence.** All stopped still improving at the 50,000-step
  ceiling, so every comparison is at fixed compute rather than at convergence.
- **English only, at 3.4–5.2M parameters.** The literature suggests both choices
  work against the hypothesis.

### What would be worth doing next

In rough order of value per hour.

**The legibility probe.** §14 and §11 argued that freezing buys *guaranteed
legibility* as well as lower perplexity. The perplexity half was refuted in §21;
the legibility half has never been tested, and it is the one place freezing can
still win. Checkpoints exist, so this needs no training.

**Entity vocabulary rebuild.** 77.5% of character fallback is capitalised words
absent from a vocabulary built on 6,000 articles with a single-word threshold of
10, tuned once and never revisited (§6, §19). This is the largest single
component of the Q1 deficit and the cheapest to attack.

**A causal dependency feature.** Derive the syntactic annotation from a
prefix-only parse, removing the lookahead. This converts the §23 upper bound
into a claim about causally available features.

**A downstream task.** A linear probe for POS or topic classification on frozen
representations would corroborate that the bits-per-byte differences mean
something practically.

**Level 1 — the routed tokenizer.** §22 laid out a design that routes words to
BPE or to morphological decomposition by frequency, with the crossover as a
free parameter and both baselines as its endpoints. The measurements motivating
it exist; it was never built.

---

## Appendix A — Decision Record

*Grouped by the part in which each decision was taken. Where a decision was
later reversed or refuted, the section that did so is named.*

### Part I — the question

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Intervention point | Embedding layer | Tokenizer only; architecture change | Transformer unchanged; clean separation |
| Unit granularity | Morpheme-level | Word-level; phrase-level | Smaller vocabulary; closer to BPE; better OOV |
| Named entity handling | Atomic units | Morpheme-segment them | Proper nouns are not compositionally meaningful |
| Frozen/trained combination | Concatenation | Addition | Addition lets training cancel the frozen signal |
| Evaluation method | Perplexity + probing | Generation-based metrics | No generation needed; probing is more informative |

### Part II — the tokenizer

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Vocabulary validity | Source-based (MorphoLex) | Frequency threshold | Frequency conflates noise with rare-but-real |
| Entity threshold | Split by structure (multi-word ≥3, single ≥10) | Flat threshold; MorphoLex exclusion; curated list | Token structure is a reliable signal of misclassification risk |
| Morfessor role | Affix-only fallback | Full segmentation; removal | Roots unreliable; affixes more trustworthy — later shown to add nothing (§8) |
| Capitalisation guard | Rejected | Applied temporarily | Typographic heuristic, not a linguistic category |
| Number of pipelines | Three (bpe, morph, det) | Two | Isolates Morfessor's contribution; established the compare-don't-guess method |
| Compound splitting | Implemented, with a particle exception | Not implemented; minimum length only | Concrete wins; false-positive risk negligible for genuine OOV |
| Codebase structure | Flat, seven files | Deep hierarchy | Communication overhead — **reversed in §16** when that premise expired |

### Part III — the apparatus

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Model role | Fixed apparatus, small and standard | Best or largest model | The transformer measures the embedding; small keeps the hypothesis testable |
| `d_model` | 128 | 256/512 | Pinned by the frozen-fraction argument |
| Model depth | 4 layers | 6+ | Shallow keeps reliance on the input representation |
| Cross-pipeline metric | Bits-per-byte | Per-token perplexity | Perplexity is not comparable across tokenizations |
| Within-pipeline metric | Per-token perplexity | BPB only | Same tokenization, and more sensitive |
| Embedding ownership | Model owns it; dataset emits ids | Dataset emits vectors | The embedding is the contribution and carries gradients — **narrowed in §15** |
| Reported metric point | Best-val, early-stopped | Last-val | WikiText-103 overfits; the minimum must not be contaminated |
| Architecture selection | Calibrate against the BPE curve | Choose by argument | Measurement beats judgement |

### Part IV — the audits

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Frozen block content | Composed blocks, width derived | Fixed 23-dim schema | Sized by label inventory, not information; 12 of 23 columns were constant |
| Frozen fact axis | Static vs context source | Treat all frozen facts alike | Static is parameter tying; context is information a lookup table cannot hold |
| Dataset payload | ids + integer codes | ids only | Frozen metadata has no gradients, so the §12 reasoning never applied |
| Validity criterion | Injectivity (collisions) | Reconstruction fidelity | Non-colliding spelling variation loses no information |
| Information channels | All five emitted; each a config flag | Hardcode the fixes | Flags allow per-channel ablation |
| Unknown characters | UTF-8 byte fallback | ASCII punctuation layer | UNK is a collision sink; byte fallback makes the encoding total |
| `max_steps` | Split into ceiling and `lr_horizon` | One field for both | One field doing both made different-length runs incomparable |
| Seeding | Seeded; seed in the run identity | Unseeded single runs | A small expected effect cannot be told from initialisation noise |
| Codebase structure | Package | Flat (§10) | The premise expired, and flat was about to cause two silent bugs |

### Part V — grounding and controls

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| BPE vocabulary size | Matched to morph (17,151) | Left at 10,000 | The +51% parameter advantage was a real confound |
| Frozen vs learned factors | Both built and compared | Assume freezing is better | Factored NLMs are the direct ancestor and learn their factors |
| Factor signal scale | Exposed and tested at 0.22 | Leave one-hot at norm 1.0 | A frozen one-hot entered 4.6× louder, confounding trainability with magnitude |
| Freezing claim | Refuted for perplexity; legibility still open | Claim freezing wins | Learned beats frozen by 1.7–2.3% under a controlled comparison |
| Data-starvation sweep | Rejected as the main path | Run it as the headline | It changes the conditions rather than the method |

### Part VI — the synthesis

| Decision | Chosen | Rejected | Reason |
|---|---|---|---|
| Q1 strategy | Cooperate: BPE segments, morphology informs | Keep competing with BPE | The frequency table is an assignment, not a comparison |
| BPE context alignment | Scan the token's span | Index by start offset | ByteLevel folds the space into the token; 78% of the stream would have received `none` |
| BPE fact computation | Build all six facts in one pass | Build only what was requested | Each new fact would otherwise need its own multi-hour pass |
| High-cardinality facts | Separate cardinality from width; frozen becomes a random projection | One-hot only | 16,365 root classes cannot one-hot into 16 dims |
| Entity lexical identity | Entities occupy their own root classes | Fall to the `none` sentinel | An entity's lemma is itself |
| Seed replication | Two seeds on baseline and best cell | Single runs | The effect-to-noise ratio is what makes a large effect a demonstrated one |

---

## Appendix B — Section Mapping

*This document was restructured after the work was complete. Earlier drafts,
and some docstrings in the codebase, refer to the previous numbering.*

| old | new | note |
|---|---|---|
| 1, 2, 3 | 1 | merged: framing and intervention point |
| 4 | 2 | |
| 5 | 3 | |
| 6, 7 | 4 | merged: granularity and evaluation |
| 8, 10, 11, 12 | 5 | merged: the vocabulary thread, previously split |
| 22, 23 | 6 | merged: entity thresholds, moved to sit beside the vocabulary |
| 9, 14, 15, 16 | 7 | merged: tools, segmenter, format-B bug; minor bugs compressed |
| 17, 18, 19 | 8 | merged: Morfessor's null and the three-pipeline decision |
| 24, 25 | 9 | merged: compound splitting |
| 13, 20, 21, 26 | 10 | merged: first comparison, consolidation; stale status removed |
| 27 | 11 | |
| 28 | 12 | |
| 29 | 13 | |
| 30 | 14 | |
| 31 | 15 | |
| 32 | 16 | |
| 33 | 17 | |
| 34 | 18 | |
| 35 | 19 | |
| 36 | 20 | |
| 37 | 21 | |
| 38 | 22 | |
| 39 | 23 | |
| — | 24 | new: closing status and next steps |

---

*The project asked whether injecting linguistic structure into a language
model's embedding layer helps. It does — by 4.44% bits-per-byte, replicated,
with fewer parameters, concentrated in exactly the tokens where statistical
learning has least to work with. Getting to that answer required discovering
that the project's first result was an artifact of its own measurement, and
building the instruments to prove it.*
