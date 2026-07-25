# Context-Aware Embeddings for Tokenizer-Free Language Models
## Complete Research Journey

*A chronological record of every decision, fork, dead end, and breakthrough from initial idea to working implementation.*

---

## Table of Contents

1. [The Starting Point](#1-the-starting-point)
2. [Understanding the Pipeline](#2-understanding-the-pipeline)
3. [Identifying the Intervention Point](#3-identifying-the-intervention-point)
4. [The Hybrid Frozen/Trained Embedding Idea](#4-the-hybrid-frozentraned-embedding-idea)
5. [Linguistics Crash Course](#5-linguistics-crash-course)
6. [Granularity Decision — Word vs Morpheme](#6-granularity-decision--word-vs-morpheme)
7. [What the Preprocessor Can and Cannot Do](#7-what-the-preprocessor-can-and-cannot-do)
8. [The Vocabulary Design](#8-the-vocabulary-design)
9. [Tool Survey — MorphoLex, Morfessor, spaCy](#9-tool-survey--morpholex-morfessor-spacy)
10. [Dataset Choice and First Vocabulary Build](#10-dataset-choice-and-first-vocabulary-build)
11. [The Frequency Threshold Problem](#11-the-frequency-threshold-problem)
12. [Redesigning the Vocabulary Without Thresholds](#12-redesigning-the-vocabulary-without-thresholds)
13. [BPE Baseline Build and First Comparison](#13-bpe-baseline-build-and-first-comparison)
14. [Building the Preprocessor](#14-building-the-preprocessor)
15. [Bug Hunting — Category Priority](#15-bug-hunting--category-priority)
16. [The MorphoLex Format B Bug](#16-the-morpholex-format-b-bug)
17. [Large-Scale Testing and New Problems](#17-large-scale-testing-and-new-problems)
18. [The Morfessor Noise Problem](#18-the-morfessor-noise-problem)
19. [The Three Pipelines Decision](#19-the-three-pipelines-decision)
20. [Refactoring into Final Structure](#20-refactoring-into-final-structure)
21. [Post-Refactor: Dissatisfaction with BPE Losses](#21-post-refactor-dissatisfaction-with-bpe-losses)
22. [Entity Vocabulary — The Frequency Threshold Wars](#22-entity-vocabulary--the-frequency-threshold-wars)
23. [The Split Threshold Solution](#23-the-split-threshold-solution)
24. [Compound Word Splitting](#24-compound-word-splitting)
25. [The star+tup Problem and the Particle Fix](#25-the-startup-problem-and-the-particle-fix)
26. [Current State and What Comes Next](#26-current-state-and-what-comes-next)
27. [Designing the Frozen Embedding Layer](#27-designing-the-frozen-embedding-layer)
28. [The Dataset, Model, and the Fairness Metric](#28-the-dataset-model-and-the-fairness-metric)
29. [Calibration and Literature Grounding](#29-calibration-and-literature-grounding)

---

## 1. The Starting Point

**The thesis topic:** *Context-Aware Embeddings for Tokenizer-Free Language Models*

The initial goal was vague: create some representation of text based on morphology, syntax, or semantics that would replace tokens. No concrete idea of what the unit should be, where in the model to intervene, or how to evaluate success.

**The first question asked:** Is the unit a word? Part of a word? If a part, does it have to be meaningful?

At this point, several directions were on the table:
- Morpheme-based embeddings (BPE-like but linguistically motivated)
- Character n-gram embeddings (FastText-style)
- Syntactic span embeddings (constituency-based)
- Continuous semantic units (CTC-style soft segmentation)

No decision was made yet. The first priority was to understand what the current pipeline actually does.

---

## 2. Understanding the Pipeline

Before designing anything, we needed to understand what we were replacing. The standard LLM pipeline was mapped out:

```
text → [tokenizer] → [integer IDs] → [embedding table lookup] → [vectors] → [transformer]
```

**Key insight about the embedding table:** It is simply a matrix of shape `[vocab_size × embedding_dim]`. Each row is one token's vector. Looking up token ID 7 means fetching row 7. That's the entire operation. There is no intelligence here — just a matrix lookup.

**How the embedding table learns:** During training, gradients flow backward from the loss through all transformer layers and eventually reach the embedding table. Only the rows corresponding to tokens that appeared in that batch get updated. Rare tokens appear infrequently, get updated rarely, and therefore have worse representations than common tokens. This is a known, real problem.

**The static vs contextual distinction:** The embedding table produces the *same vector* for a word regardless of context. The word "bank" always gets the same starting vector whether it means a financial institution or a river bank. Context-sensitivity only arrives later, inside the attention layers.

**The critical implication:** Because the embedding is static and context-free, it must somehow pre-encode everything useful about a word's identity before context is available. For common, unambiguous words this is fine. For rare words, ambiguous words, or morphologically complex words, the single-row constraint is a real limitation.

---

## 3. Identifying the Intervention Point

**Fork: Where do we intervene?**

Three options were considered:
- Replace the tokenizer only (feed different IDs into the same embedding table)
- Replace the embedding table (keep the tokenizer, change how IDs become vectors)
- Replace both (our own segmentation *and* our own embedding construction)

**Decision: target the embedding layer, not the model architecture.**

The transformer receives a sequence of vectors and does not care where they came from. If we can produce better vectors, the transformer works unchanged. No architecture modification needed.

**What this means practically:** We replace the two-step pipeline (tokenizer → lookup table) with our own pipeline that produces vectors more directly. The transformer remains a standard architecture.

**The comparison experiment design:** Two flows, same transformer:
- Flow A: text → BPE → embedding table (standard)
- Flow B: text → our preprocessor → embedding layer (ours)

The transformer is identical. The only difference is the embedding step.

---

## 4. The Hybrid Frozen/Trained Embedding Idea

**The original insight:** Instead of an embedding vector where all N dimensions are learned freely, split the vector into two parts:

```
[frozen dims (T) | trained dims (N-T)]
```

- **Frozen dims:** Encode linguistically certain facts we inject at preprocessing time. These are never updated by backprop.
- **Trained dims:** Learned normally by the model during training.

**Why this is principled:** The frozen dimensions create guaranteed information channels. Normally, nothing stops the model from learning to ignore or distort any part of what the embedding encodes. By freezing some dimensions, we guarantee that certain information is always present, always readable, at every layer, for every attention head that wants it.

**The burden question:** This raises a clear question about division of labor. If the embedding table handles everything through training, we are easing the *embedding layer's* burden — not the attention layers' burden. The attention layers already handle context, syntax resolution, and word sense disambiguation. We are targeting what the embedding layer struggles with before any context is available.

**The experiment becomes sharper:** Does giving the model guaranteed correct morphological information as a starting point actually help? The answer is not obvious, which makes it worth investigating.

---

## 5. Linguistics Crash Course

To decide what to freeze, we needed to understand what linguistic information is actually available at different scopes.

### Morphology

Morphology is about the internal structure of words. A morpheme is the smallest meaningful unit. `unbreakable` = `un` (negation prefix) + `break` (root) + `able` (capable-of suffix).

Key morpheme types:
- **Free morphemes:** Can stand alone (`break`, `cat`, `run`)
- **Bound morphemes:** Only exist attached to something (`-ing`, `un-`, `-able`)
- **Inflectional morphemes:** Mark grammatical properties (`-s` for plural, `-ed` for past tense)
- **Derivational morphemes:** Change word meaning or category (`-ness` nominalizes, `re-` adds repetition)

What morphology can tell us from a single word: the root, what affixes are attached and what they mean, and often the part of speech (because many suffixes are category-specific: `-ness` almost always makes a noun, `-ly` almost always makes an adverb).

### Syntax

Syntax is about how words combine into sentences. Two key frameworks:

**Dependency syntax:** Describes grammatical relations between words. In "the cat chased the dog," `cat` is the subject of `chased` and `dog` is the object. These relations are *relational* — you cannot determine a word's syntactic role by looking at it in isolation. You need the full sentence.

**Constituency syntax:** Words group into phrases that behave as units. A noun phrase (NP) like "the old cat" can be replaced by "it," can be moved around the sentence as one block, and is a real linguistic object. Phrase type membership requires sentence-level analysis.

**The key insight:** Syntactic role (subject, object, modifier) is *not* a property of a word — it is a property of a word *in a particular sentence*. You cannot freeze it per-word without sentence context.

### Semantics

Two levels:

**Lexical semantics:** The meaning of individual words. Some is available word-level (`cat` is always an animal), but polysemous words (`bank`, `light`, `left`) require context.

**Compositional semantics:** How words combine to create sentence meaning. Entirely context-dependent.

### The Scope Table

| What you can know | Morpheme | Word | Phrase | Sentence |
|---|---|---|---|---|
| Morpheme type | ✓ | ✓ | ✓ | ✓ |
| Inflection (tense, plural) | partial | ✓ | ✓ | ✓ |
| Part of speech | partial | mostly | ✓ | ✓ |
| Syntactic role (subject/object) | ✗ | ✗ | partial | ✓ |
| Phrase membership | ✗ | ✗ | ✓ | ✓ |
| Unambiguous lexical meaning | ✗ | partial | ✓ | ✓ |
| Contextual meaning | ✗ | ✗ | partial | ✓ |

### A Critical Realization — Preprocessor Scope ≠ Unit Scope

**Important insight arrived at during this discussion:** The scope of what the preprocessor *looks at* when making a decision is independent of the scope of the *unit* it produces.

We can look at the whole sentence to decide what tag to put on one word. The model still receives word-level (or morpheme-level) vectors. The preprocessor is just a smarter tagger. This significantly expands what we can freeze — syntactic role, which requires sentence context to determine, can be frozen at the word level if our preprocessor runs a dependency parser first.

---

## 6. Granularity Decision — Word vs Morpheme

**The central design question:** What does one vector represent?

This choice determines what information you can and cannot encode.

**Word-level units:**
- Can encode: morphological facts (derived from internal structure), POS, syntactic role (with sentence-level analysis), named entity type
- Cannot encode: compositional meaning, word sense for ambiguous words
- Vocabulary size: large (50k+ for typical BPE, still large for true words)
- Handles OOV: poorly (a word not seen in training has no vector)

**Morpheme-level units:**
- Can encode: morpheme type (prefix/root/suffix), morpheme meaning class
- Cannot encode: word-level POS (unless whole-word POS is mapped to the head morpheme), syntactic role without extra machinery
- Vocabulary size: small (few thousand productive morphemes in English)
- Handles OOV: better (even unknown words have known affixes)

**The argument for morphemes:**
- Much closer to BPE in pipeline structure → cleaner comparison
- Smaller vocabulary → each entry trains more often → better embeddings
- OOV handling is linguistically grounded rather than statistical

**The argument against morphemes:**
- POS tags and syntactic roles attach to words, not morphemes
- Phrasal verbs and multiword expressions break compositional assumptions

**The named entity insight:** Some words cannot be meaningfully decomposed into morphemes. `Microsoft` does not break into `micro` + `soft` in any semantically useful way. `SolidGoldMagikarp` is a compound proper noun with no compositional meaning. These should be treated as atomic units.

**The tripartite theory:** Every word is one of:
1. **Decomposable into meaningful morphemes** — process morphologically
2. **A named entity** — treat as atomic unit with entity type label
3. **Unknown/garbage** — fall to character-level honestly

**Fork: Word-level or morpheme-level units?**

**Decision: Morpheme-level units, with named entities treated atomically.**

This gives a smaller vocabulary, closer comparison to BPE, and linguistically grounded OOV handling. Named entities as atomic units is a natural consequence of the tripartite theory.

---

## 7. What the Preprocessor Can and Cannot Do

### Division of Labor in the Transformer

The transformer has two distinct phases:

**Embedding phase:** Converts each token to a vector *independently*, with zero awareness of other tokens. Static, context-free.

**Attention layers:** Let all tokens "talk to each other." Every token looks at every other token and updates its representation. Context enters here.

**MLP layers:** After attention in each transformer block. Research suggests MLPs act as key-value memory, storing factual associations. They also apply nonlinear transformations that attention alone cannot.

**What we are targeting:** The embedding phase. We are not easing the burden of attention or MLP — those are good at what they do. We are giving the embedding phase a head start by guaranteeing certain linguistic facts are present before context arrives.

**What attention handles well (that we should not try to replicate):**
- Word sense disambiguation (which "bank"?)
- Syntactic role in context (subject vs object)
- Phrasal verb detection (give + up = quit)
- Coreference resolution (he = John)
- Named entity context (nearby words reveal entity type)

**What the embedding struggles with (our target):**
- Internal word structure (the embedding treats `unbreakable` and `breakable` as unrelated)
- Morphological identity (same root with different inflections = unrelated rows)
- Guaranteed presence of linguistic facts (model can learn to ignore or distort anything)

### Evaluating Without Generation

**Fork: Must we generate text to evaluate?**

**Decision: No.** Text generation is not required to compare models. Three evaluation strategies that avoid generation:

1. **Perplexity:** How surprised is the model by held-out text? Lower = better language model. Directly comparable between approaches. No generation needed.

2. **Probing tasks:** Train a tiny classifier on top of embeddings to predict some linguistic property (POS tag, named entity type). If the embeddings encode that information, the classifier learns easily from little data. This directly tests whether frozen dimensions are actually useful.

3. **Word similarity benchmarks:** Datasets like SimLex-999 with human-judged similarity scores. Measure correlation between embedding cosine similarity and human judgments.

**The recommended evaluation strategy for the thesis:**
- Perplexity as the headline comparison number
- Probing tasks as the mechanistic story ("why" one model is better)
- Together: "our embeddings produce lower perplexity AND probing shows the frozen dimensions are actively used"

---

## 8. The Vocabulary Design

### Early Design — Frequency-Based

The first vocabulary design used a frequency threshold: morphemes appearing at least 5 times in 10k training articles stay in the vocabulary; others are cut. Named entities required frequency ≥ 10 and length ≥ 5.

**Problem discovered:** Running the coverage check on WikiText-103 validation set showed that 34.8% of words fell to character-level fallback. Many of these were common English words like `known`, `related`, `closely` — words that clearly exist and have morphological structure, but didn't make the frequency cutoff because we only sampled 10k articles.

**Root cause:** The frequency threshold was conflating two different things:
- Low frequency because it is a rare but real morpheme (should keep)
- Low frequency because it is Morfessor noise (should discard)

Frequency cannot distinguish between these cases.

### The Better Design — Source-Based

**Fork: How do we decide what counts as a real morpheme?**

**Decision: Use MorphoLex itself as the definition of what counts as a real morpheme.**

MorphoLex is a hand-curated database of ~70k English words with their morpheme decompositions. Every morpheme that appears anywhere in MorphoLex is, by definition, a real English morpheme. No frequency threshold needed.

**The six-layer vocabulary:**

| Layer | Category | Source | Threshold | Count |
|---|---|---|---|---|
| 1 | Prefixes | MorphoLex | None | 127 |
| 2 | Suffixes | MorphoLex | None | 211 |
| 3 | Roots | MorphoLex | None | ~15,000 |
| 4 | Function words | Hardcoded list | None | 112 |
| 5 | Named entities | Dataset frequency | freq≥10, len≥5 | ~394 |
| 6 | Characters | a–z | Never removed | 26 |

**Why only layer 5 uses frequency:** Named entities only deserve an embedding slot if the model will see them enough times to learn from. A named entity that appears once cannot be meaningfully represented regardless of how many dimensions it gets. For all other layers, frequency is irrelevant — a real morpheme deserves an embedding whether it appears once or a million times.

**The fourth category — function words:** Function words (`the`, `and`, `of`, `in`, `is`, `was`, ...) are a closed class with no morphological structure and no compositional meaning. They exist in MorphoLex as single-morpheme roots, but they deserve their own frozen dimension flag. Adding a `function` category as layer 4 provides this signal for free.

**Priority ordering:** Because some tokens appear in multiple layers (e.g., `and` is both a suffix in MorphoLex and a function word), deduplication must respect priority: entity > function > prefix > suffix > root > character.

**Characters always survive:** The character layer is never deduplicated even if a character also appears as a prefix or suffix. This guarantees 100% coverage — any string can be spelled out as individual characters.

---

## 9. Tool Survey — MorphoLex, Morfessor, spaCy

### MorphoLex

**What it is:** A hand-curated Excel database mapping ~70k English words to their morpheme decompositions with a custom notation format.

**The notation format:**
- `{(beauty)}>ful>` — root `beauty`, suffix `ful`
- `<un<{(happy)}` — prefix `un`, root `happy`
- `{<re<(search)}>er>` — format B (prefix inside braces), prefix `re`, root `search`, suffix `er`

**Format B bug discovered late:** Our initial parser only handled format A (prefix before the opening brace). 4,941 out of ~70k entries use format B, meaning we were silently losing the prefix for all these words. `researchers` was being parsed as just `[('suffix', 'er')]` instead of `[('prefix', 're'), ('root', 'search'), ('suffix', 'er')]`.

**The fix:** Detect whether the prefix appears before or inside the opening brace, and extract accordingly.

**MorphoLex's lemmatization:** MorphoLex stores normalized roots, not surface forms. `beautiful` → root `beauty` (not `beauti`). `running` → root `run` (not `runn`). This solves the morphophonological normalization problem for free — for all 66k words in the database.

### Morfessor

**What it is:** An unsupervised statistical morpheme segmenter. Given a word list, it learns to segment words by minimizing description length (MDL principle).

**How it was intended to be used:** As a fallback for words not in MorphoLex, to produce some segmentation rather than falling straight to characters.

**What actually happened:** When trained on a small seed list, Morfessor produces reasonable segmentations for words with clear morpheme boundaries. However it also produces noise — Morfessor found `bi` (a real prefix) + `den` (a real root) inside `biden`, producing a confident but wrong parse.

**The normalization problem:** Morfessor produces surface-form segments, not normalized roots. `beautiful` → `beauti + ful`, not `beauty + ful`. A set of rules was written to handle the three English morphophonological alternations:
1. Doubled consonant before vowel suffix: `runn+ing` → `run+ing`
2. `i`→`y` before vowel/certain suffixes: `happi+er` → `happy+er`
3. Missing silent e: `mak+ing` → `make+ing`

**Morfessor's ultimate contribution:** Minimal. Every useful segmentation in our test set came from MorphoLex or prefix stripping. Morfessor's own segmentations either produced noise or were redundant with what prefix stripping already found.

### spaCy

**What it is:** Industrial-strength NLP library providing tokenization, POS tagging, dependency parsing, and named entity recognition.

**Our use:** Named entity recognition only. spaCy uses IOB tags (B = beginning of entity, I = inside entity, O = outside) to identify entity spans.

**Key behaviors observed:**
- Multi-word entities (`Joe Biden`, `New York Times`, `Goldman Sachs`) correctly grouped as single spans
- Possessives (`Apple's`) correctly strip the possessive and tag just the name
- Nationality adjectives (`British`, `French`, `American`) tagged as NORP — debated whether to include
- Dates (`last January`), numbers (`42`), and temporal expressions tagged but excluded from our entity vocabulary by design
- Some entities missed (`River Thames` not tagged)
- Some common words mistagged (`love`, `church`, `us`)

**The article stripping fix:** Entity spans sometimes include leading articles (`the United States` → tagged as one span including `the`). Articles are stripped before the entity surface form is stored: `the_united_states` → `united_states`.

**Entity types kept:** ORG, PERSON, GPE, LOC, FAC, NORP, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE

**Entity types excluded:** DATE, CARDINAL, ORDINAL, PERCENT, TIME, MONEY, QUANTITY — these are compositional and infinite in variety, so treating them as atomic units makes no sense.

---

## 10. Dataset Choice and First Vocabulary Build

**Dataset chosen: WikiText-103**

WikiText-103 is a standard language modeling benchmark consisting of verified Good and Featured Wikipedia articles. It contains ~103 million tokens across ~28k articles. It is clean English prose with good named entity density, freely available via HuggingFace datasets in one line.

**Why WikiText-103 over alternatives:**
- Standard benchmark → results comparable to existing literature
- Clean text → less preprocessing noise
- Large enough → meaningful statistics
- Wikipedia → rich in named entities for our entity vocabulary layer

**First vocabulary build (10k articles):**
- Unique morphemes seen: 10,462
- Unique named entities seen: 14,667
- Target vocabulary size: 10,000

**Immediate problem:** 10,462 morphemes already exceeds 10,000, leaving zero slots for named entities. The frequency threshold of 5 was the only thing controlling vocabulary size, and it wasn't working as intended.

**Entity quality issues found:** Many high-frequency "entities" were nationality adjectives being mistagged: `british` (280), `german` (262), `french` (207). Others were clearly wrong: `us` (pronoun), `white` (adjective), `love`. The frequency+length filter (`freq≥10, len≥5`) removed most of these by coincidence rather than by design.

---

## 11. The Frequency Threshold Problem

**The coverage measurement:** Running coverage on the WikiText-103 validation set showed:
- 65.2% of words covered by our vocabulary
- 34.8% falling to character fallback
- 0.0% truly uncovered (character fallback handles everything)

**The alarming content of the fallback:** Words like `known`, `closely`, `related`, `parts`, `eastern`, `becoming` — completely normal, common English words — were falling to character-level. These words exist in MorphoLex and have clear morphological structure, but they didn't appear enough times in our 10k article sample.

**Root cause:** The frequency threshold of 5 was designed to filter Morfessor noise, but it also incorrectly filtered real words that were common but underrepresented in our small sample.

**The structural problem:** Frequency cannot distinguish:
- Low frequency because it is a rare but real morpheme (should keep)
- Low frequency because it is Morfessor noise (should discard)

This led directly to the redesign in the next section.

---

## 12. Redesigning the Vocabulary Without Thresholds

**The key insight:** We already have a hand-curated linguistic database — MorphoLex. Any morpheme that appears in MorphoLex is real by definition. Any morpheme that doesn't appear in MorphoLex is either noise (from Morfessor) or a genuine OOV that should fall to characters.

**New design principle:** Source determines validity, not frequency.

- MorphoLex prefixes → always valid (78 found, expands to 141 after format B fix)
- MorphoLex suffixes → always valid (240 found)
- MorphoLex roots → always valid (~15,000 found)
- Function words → always valid (hardcoded closed class)
- Named entities → frequency-gated (only category where frequency is justified)
- Characters → always present, never removed

**Total vocabulary: ~15,739 entries** — larger than BPE's 10,000 but every entry is linguistically justified.

**The deduplication fix:** When building the vocabulary, some tokens appear in multiple layers (e.g., `and` is both a suffix in MorphoLex layer 2 and a function word in layer 4). Simple `drop_duplicates` keeps the first occurrence, which might be the wrong category. The fix: sort by priority before deduplication so the highest-priority category always wins.

**Priority order:** entity(0) > function(1) > prefix(2) > suffix(3) > root(4) > character(5)

---

## 13. BPE Baseline Build and First Comparison

**BPE implementation:** Standard GPT-2 style BPE using HuggingFace `tokenizers` library. Trained on the same 10k WikiText-103 articles for direct comparability. Target vocabulary: 10,000 tokens.

**First side-by-side comparison on test words:**

| Word | Our pipeline | BPE |
|---|---|---|
| `unbreakable` | `un + break + able` | `un \| break \| able` |
| `beautiful` | `beauty + ful` | `beautiful` (one token) |
| `quickly` | `quick + ly` | `quickly` (one token) |
| `running` | `run` (lemmatized) | `running` (one token) |
| `cats` | `cat` (lemmatized) | `c \| ats` |
| `Microsoft` | atomic entity [ORG] | `Mic \| ros \| oft` |
| `happiness` | `happy + ness` | `happ \| iness` |

**Key observations:**
- BPE wins zero comparisons, ties on a few
- `cats → c | ats` and `Microsoft → Mic | ros | oft` are BPE's most embarrassing failures
- Our pipeline correctly lemmatizes (`running → run`, `cats → cat`)
- BPE vocabulary has 739 non-alphabetic tokens (punctuation/symbols) vs zero in ours
- BPE has 1,068 tokens of length ≤ 2 vs our character layer which is intentional

**BPE frequency distribution:** Much flatter than ours. Almost all 9,745 BPE tokens appear at least 5 times. Our morpheme vocabulary had a long tail of rare items before the redesign.

---

## 14. Building the Preprocessor

**The preprocessor's job:** Take a raw sentence, return a sequence of token IDs plus metadata for the frozen dimensions.

**Initial pipeline design:**
```
sentence → spaCy NER → entity spans
         → for each token:
             if named entity → emit atomic unit
             elif in MorphoLex → emit morphemes from lookup
             elif Morfessor produces valid segments → emit those
             else → emit characters
```

**Morfessor training:** Rather than training on all 66k MorphoLex words (which caused memory crashes in our environment), a curated seed list of ~3,000 words was used. The seed list included common OOV patterns: inflected forms, prefixed words, technical terms.

**The `encode()` output format per token:**
```python
{
    'token_id':     int,    # index into vocabulary
    'token':        str,    # the morpheme/entity/character string
    'category':     str,    # prefix/suffix/root/function/entity/character
    'role':         str,    # morphological role within its word
    'entity_label': str,    # spaCy NER label if entity, else None
    'word':         str,    # original surface word this came from
}
```

**Helper methods added:**
- `encode_ids(text)` — returns just the list of integer IDs
- `pretty_print(text)` — formatted table for debugging
- `decode(ids)` — best-effort reconstruction (acknowledged as imperfect)

---

## 15. Bug Hunting — Category Priority

**Bug 1: `and` classified as suffix**

`and` appeared in the vocabulary as a suffix (it appears as a suffix in MorphoLex in some entries) AND as a function word. The `categories` dict in the preprocessor took whatever was inserted last, giving `suffix` as the category. This caused `and[suffix]` to appear in output instead of `and[function]`.

**Fix:** The `_resolve_category` method was added. Instead of looking up category from a flat dict, it queries all rows for that token in the vocabulary DataFrame and returns the minimum-priority category. This ensures entity > function > prefix > suffix > root > character always holds.

**Bug 2: `london` classified as root despite being in entity vocabulary**

`london` was in `entity_tokens` but `_segment_word` checked function words and MorphoLex before checking entities. MorphoLex also contains `london` as a root (it's a proper noun that appears in many compound entries).

**Fix:** Entity vocabulary check added as the *first* step in `_segment_word`, before function words and MorphoLex.

**Bug 3: Character fallback tokens inheriting wrong category**

Single character `a` appearing in a character fallback context was being classified as `function` (because `a` is also the article `a` in the function word layer) despite having role `char`.

**Fix:** In `_make_token`, if role is `char`, category is always forced to `character` regardless of what the vocabulary says. Character fallback always overrides vocabulary category.

---

## 16. The MorphoLex Format B Bug

**Discovery:** Running `test_sentences.py` on a larger set revealed that `researchers` produced only `er[suffix]` — the root `search` and prefix `re` were being lost entirely.

**Investigation:** Checking MorphoLex directly showed `researchers` stored as `{<re<(search)}>er>`. Our parser was looking for prefixes in the string *before* the opening `{`, but in format B entries, the prefix is *inside* the opening brace.

**Two formats in MorphoLex:**
- Format A: `<un<{(happy)}` — prefix BEFORE the opening brace
- Format B: `{<re<(search)}>er>` — prefix INSIDE the opening brace

**Scale of the problem:** 4,941 out of ~70k entries use format B. This means thousands of words were missing their prefixes silently.

**Notable examples fixed by the parser fix:**
- `researchers → re + search + er` (was: `er` only)
- `prosecution → pro + secute + ion` (was: `ion` only)
- `aline → a + line` (Latin prefix `a-`)
- `universe → uni + verse`

**Fix:** Parser updated to detect prefix position relative to parenthesized root:

```python
# format A: prefixes before the opening {
format_a_prefixes = re.findall(r'<([a-zA-Z]+)<', s[:brace_open])

# format B: prefixes inside the opening { before the root
inner = s[brace_open:brace_close]
paren_pos = inner.index('(') if '(' in inner else len(inner)
format_b_prefixes = re.findall(r'<([a-zA-Z]+)<', inner[:paren_pos])

prefixes = format_a_prefixes + format_b_prefixes
```

**Side effect:** Prefix count expanded from 78 to 141 — the 63 new prefixes are format B prefixes that were previously invisible. This includes single-letter Latin prefixes like `a-`, `e-`, `o-`.

---

## 17. Large-Scale Testing and New Problems

**Test set created:** 60 sentences across 11 categories:
- Morphologically rich
- Inflected forms
- Named entities (single and multiword)
- Function words
- OOV and rare words
- Unknown proper nouns
- Compound words
- Ambiguous words
- Morphologically simple
- Mixed realistic

**New problem found: single-character prefix/suffix noise**

The format B fix added 63 new prefixes to the vocabulary, including single-letter prefixes like `a`, `e`, `o`, `i` (real Latin prefixes). When Morfessor segments an OOV word into pieces, single-character pieces that happen to be in the prefix or suffix set were being assigned `prefix` or `suffix` roles instead of falling to characters.

Result: outputs like `e[prefix]`, `o[prefix]`, `n[suffix]` inside character soup — single characters being confidently labeled as morphemes.

**Minimum length filter applied to Morfessor validation:** A segment must be at least 2 characters long to be accepted as a valid morpheme from Morfessor output. Single characters from Morfessor always become character fallback.

**New problem: the confidence of wrongness issue**

`biden → bi[prefix] + den[root]` — Morfessor found two real English morphemes (`bi` = two, `den` = lair) inside a proper noun. The result is confident but wrong.

**Fork: what to do with unknown proper nouns?**

**Option considered: capitalisation guard** — if a word starts with a capital letter and isn't in MorphoLex or the entity vocabulary, fall straight to characters.

**Problem with capitalisation guard:** Sentence-initial words are always capitalised. Stylistic capitalisation (`God`, `Internet`). Lowercase proper nouns in informal text. The guard relies on a typographic convention, not a linguistic category. This violates our design principle of not using surface-form heuristics.

**Decision: remove the capitalisation guard, accept `bi+den` style noise.**

Reasoning: At scale, `bi+den` appearing consistently across thousands of Biden mentions will cluster with political context regardless of the wrong morpheme split. The model doesn't know morphemes are supposed to be meaningful — it just sees consistent co-occurrence patterns. Character soup (`b+i+d+e+n`) is more honestly unknown but equally uninformative.

The capitalisation guard was a band-aid that would obscure the real limitation (entity vocabulary coverage at dataset scale) with a brittle heuristic.

---

## 18. The Morfessor Noise Problem

**The growing concern:** Looking at the full test output, Morfessor's contribution was almost entirely negative:
- Useful cases: almost zero (every good segmentation came from MorphoLex or prefix stripping)
- Harmful cases: `bi+den`, `y+n+stag+ram`, single-character prefix/suffix noise throughout OOV words

**Fork: what to do about Morfessor?**

**Option 1 — Remove Morfessor entirely.** Pipeline: entity → function → MorphoLex → prefix strip + MorphoLex on remainder → characters. Fully deterministic, no statistical noise. Cleaner design.

**Option 2 — Keep Morfessor but only accept affixes.** Morfessor's root guesses are unreliable; its affix identification is more trustworthy. Only accept Morfessor segments if they are known prefixes or suffixes. Root-position segments → characters.

**Option 3 — Revert to 78 prefixes.** Discard the single-letter prefixes that came from the format B fix. This removes the noise source but also throws away 63 legitimate prefixes.

**Option 3 was rejected** as unprincipled ("just discard half the prefixes and hope the problematic ones were there").

**Option 2 was implemented** as the affix-only rule in `pipeline_morpheme.py`. The tradeoff: cleaner output for OOV words (no wrong root assignments), at the cost of slightly more character soup where a root-position segment might have been correct.

**The three-pipeline decision:** Rather than choosing between Morfessor and no-Morfessor, build both as separate pipelines and compare. This turns the uncertainty into a research finding. If the two pipelines perform similarly in training, it confirms that Morfessor adds noise without adding value. If the Morfessor pipeline performs better on specific word types, that's a useful positive finding.

---

## 19. The Three Pipelines Decision

**The comparison design:**

| Pipeline | Name | Description |
|---|---|---|
| 1 | BPE | Standard GPT-2 style BPE, 10k vocabulary, baseline |
| 2 | morph | Morpheme-based with Morfessor fallback (affix-only rule) |
| 3 | det | Morpheme-based, fully deterministic, no Morfessor |

**Why three pipelines are better than two:**
- Isolates what Morfessor specifically contributes (or doesn't)
- Makes the comparison more rigorous for the thesis
- Costs almost nothing extra since the pipelines share most code

**What the comparison reveals (from test output):**
- Pipelines 2 and 3 produce identical output on ~55/60 test sentences
- Differences only appear in character soup regions where Morfessor might find a valid affix
- This itself is a finding: Morfessor adds almost nothing over the deterministic pipeline for English with MorphoLex coverage

**Clear wins for both morpheme pipelines vs BPE:**
- Inflected forms: `cats → cat[r]`, `running → run[r]`, `happiest → happy[r]+est[s]`
- Function words: `whether → whether[f]` (1 token) vs BPE's `W|he|ther` (3 tokens)
- Simple words: consistent single tokens vs BPE splitting `b|at`, `du|ck`, `sil|ence`

**Clear losses vs BPE:**
- Unknown multi-word entities: `Goldman Sachs`, `North Korea` → character soup (BPE handles via training data statistics)
- Complex OOV: `blockchain`, `cryptocurrency`, `recontextualized` → characters (BPE produces shorter sequences)

---

## 20. Refactoring into Final Structure

**The problem:** By the end of the exploration phase, the codebase consisted of several large `*.py` files with overlapping functionality, accumulated patches, and no clear separation of concerns.

**Fork: deep folder hierarchy or flat structure?**

Initially proposed: a deep folder hierarchy with `common/`, `pipelines/bpe/`, `pipelines/morpheme/`, `pipelines/morpheme_deterministic/`, `tests/`, `data/`, `main.py` — about 15 files across multiple folders.

**Decision: flat structure with 7 files.**

Reasoning: The communication overhead of a deep hierarchy in this collaborative development context outweighs the organizational benefits. With a flat structure, every file has an obvious, single purpose and location.

**Final file structure:**

```
Morph/
├── data/
│   ├── MorphoLEX_en.xlsx
│   ├── entity_vocab.csv
│   ├── our_final_vocab.csv
│   └── bpe_tokenizer.json
├── common.py                  # MorphoLex parser, NER, vocab builder, BasePreprocessor
├── pipeline_bpe.py            # Pipeline 1 — BPE baseline
├── pipeline_morpheme.py       # Pipeline 2 — morpheme + Morfessor (affix-only)
├── pipeline_deterministic.py  # Pipeline 3 — morpheme, no Morfessor
├── build_vocabs.py            # Run once: entity extraction, vocab build, BPE training
├── test_sentences.py          # 60-sentence structured comparison runner
└── main.py                    # Entry point: `python main.py build` / `python main.py test`
```

**The `BasePreprocessor` class:** Common logic shared by pipelines 2 and 3 was extracted into a base class in `common.py`. Subclasses only implement `_segment_word()` — the one method that differs between the two pipelines. Everything else (NER handling, entity span resolution, `encode()`, `encode_ids()`, `pretty_print()`) is inherited.

**The `build_vocabs.py` workflow:**
```
python main.py build entities  → extract named entities from WikiText-103
python main.py build vocab     → build our_final_vocab.csv from MorphoLex + entities
python main.py build bpe       → train BPE tokenizer and save to bpe_tokenizer.json
python main.py build           → all three steps
```

---

## 21. Post-Refactor: Dissatisfaction with BPE Losses

After the refactor produced clean, working three-pipeline output, we reviewed the comparison honestly. Two categories were clear losses against BPE:

**Unknown multi-word entities** — `Goldman Sachs`, `North Korea`, `European Union` → character soup in our pipelines. BPE handles these via training data statistics, producing short subword sequences.

**Complex OOV compounds** — `blockchain`, `cryptocurrency`, `recontextualized` → characters. BPE produces shorter, statistically motivated sequences.

The root cause of both problems was diagnosed: the entity vocabulary was built on only 6,000 Wikipedia articles with a frequency threshold of 10. Many real entities simply didn't appear 10+ times in that small sample. `north_korea` had count=5, `european_union` count=9, `san_francisco` count=7 — all just below the threshold.

**First fix attempt:** Run entity extraction on more articles. Result: vocabulary grew from 394 to 403 entities — almost no change, because the threshold was still too high relative to the sample size, and `goldman_sachs` wasn't in the CSV at all (never appeared as a recognised entity span in those articles).

---

## 22. Entity Vocabulary — The Frequency Threshold Wars

**Attempt 1 — lower threshold to 3:** This picked up `north_korea` (5), `european_union` (9), `san_francisco` (7). But at threshold 3, common English words that spaCy occasionally misclassifies also snuck in: `light` (3), `world`, `child`, `group` appearing as `[e]` entities. The entity layer grew to 2,049 entries, many of them noise.

**Attempt 2 — exclude MorphoLex roots:** The idea: if a token is already a known MorphoLex root, it's a common English word being misclassified. Only real named entities (like `north_korea`) are not in MorphoLex.

**Problem:** MorphoLex covers proper nouns too. `london`, `france`, `germany`, `california` are all in MorphoLex as roots. The filter correctly removed `light` and `world`, but also incorrectly removed `london`, `england`, `france`, `germany`, `australia`, `florida`, `chicago` — a serious regression.

**Attempt 3 — curated ambiguity list:** Explicitly list common words that spaCy misclassifies (`light`, `world`, `child`, `group`, etc.) and exclude only those. Rejected as incomplete and unmaintainable — there are too many such words and the list would need constant curation.

**The correct insight:** The real distinguishing feature is already encoded in the token structure itself.

---

## 23. The Split Threshold Solution

**The key observation:** Multi-word entity strings contain underscores. `north_korea`, `european_union`, `san_francisco` — these are always genuine named entities. spaCy only produces them by recognising a multi-word span. There is no common English phrase `north_korea` that could be a misclassification.

Single-word entities are different. `light`, `world`, `child` are overwhelmingly common words that spaCy occasionally mislabels at low frequency.

**The principled rule: split thresholds by structure.**

- Multi-word entities (containing `_`): `freq ≥ 3` — they cannot be misclassifications
- Single-word entities: `freq ≥ 10` — needs higher confidence to distinguish genuine proper nouns from noise

This is not a heuristic — it is a structural linguistic observation. The result:

| Token | Type | Count | Threshold | Result |
|---|---|---|---|---|
| `north_korea` | multi-word | 5 | 3 | **included** ✓ |
| `european_union` | multi-word | 9 | 3 | **included** ✓ |
| `san_francisco` | multi-word | 7 | 3 | **included** ✓ |
| `london` | single-word | 97 | 10 | **included** ✓ |
| `light` | single-word | 3 | 10 | **excluded** ✓ |
| `world` | single-word | low | 10 | **excluded** ✓ |

The MorphoLex exclusion filter was reverted entirely — it was not needed with this approach. The entity vocabulary settled at 1,259 entries with clean composition.

**Remaining entity gap:** `Goldman Sachs` is still not in the entity CSV at all — it simply doesn't appear enough times as a recognised entity in our 6,000 article sample. This is a genuine corpus coverage limitation, not a design flaw, and is accepted and documented as such.

---

## 24. Compound Word Splitting

**Motivation:** After the entity vocabulary work, OOV compound words (`blockchain`, `cryptocurrency`, `livestream`, `speedrun`, `homepage`) were still producing character soup. BPE handles these correctly because it has seen them in training data. We identified compound splitting as a linguistically principled alternative.

**The safety question:** Could compound splitting introduce false positives like `therapist → the + rapist`? A quick check confirmed that all dangerous-sounding splits (`therapist`, `together`, `carpet`, `island`, `before`, `because`, `aground`, `inflate`) are in MorphoLex and therefore handled before the compound splitter is ever reached. The compound splitter only sees genuinely OOV words — words MorphoLex doesn't know — and among those, false positives are extremely rare.

**The algorithm:** Try every split point in a word from longest-left to shortest-left. Accept the first split where both halves are in MorphoLex and both parts are at least 3 characters. If no valid split found, fall to characters.

```
blockchain → try bloc|kchain — no
           → try block|chain — both in MorphoLex ✓ → block[r] + chain[r]
```

**Position in the fallback chain:**
```
entity → function → MorphoLex → prefix strip → compound split → Morfessor/characters
```

**Results — clear wins:**
- `blockchain → block + chain`
- `livestreamed → live + stream`
- `speedrun → speed + run`
- `videogame → video + game`
- `homepage → home + page`
- `spreadsheet → spread + sheet`
- `hashtags → hash + tag`
- `healthcare → health + care`
- `counterarguments → counter + argue + ment`

---

## 25. The startup Problem and the Particle Fix

**The bug:** `startup → star[r] + tup[r]` — a false positive. `tup` is a valid but obscure MorphoLex root (British English for a male sheep). The algorithm found `star + tup` (split at position 4) before trying `start + up` (split at position 5).

**Root cause analysis:** The original algorithm iterated from position `n-3` downward, guaranteeing the right part has at least 3 characters. This structurally blocked `start + up` because `up` is only 2 characters. But `up`, `in`, `on`, `out`, `off`, `by` are among the most productive compound-forming elements in English — `startup`, `backup`, `addon`, `cutoff`, `roundup` all fail silently.

**The structural insight:** These compound particles are not arbitrary short strings. They are exactly the phrasal verb particles that also form nominal compounds — and they are all already in our `FUNCTION_WORDS` set.

**The fix:** Extend the range to allow 2-character right parts, but only accept them if they are in `FUNCTION_WORDS`. This makes compound particles reachable while blocking other 2-character noise.

```python
# 2-char right part accepted only if it's a known compound particle
if len(right) == 2 and right not in function_words:
    continue
```

For `startup`:
- i=5: left=`start`, right=`up`. `up` in FUNCTION_WORDS → accepted ✓
- `star + tup` is never reached because `start + up` is found first

**Result:** `startup → start[r] + up[f]` — correct. No new false positives introduced anywhere in the test set.

**The compound splitting verdict:** After the particle fix, the feature performs cleanly across all 60 test sentences. Every target compound splits correctly. The false positive risk from obscure 3-character roots is real but negligible in practice — genuinely OOV words are unlikely to accidentally split into two plausible MorphoLex roots unless those roots are actually semantically related to the word, in which case the split may be correct anyway.

---

## 26. Current State and What Comes Next

### What Is Complete

**Vocabulary design:** Six-layer vocabulary with 16,750 entries. Split-threshold entity filtering (multi-word: freq≥3, single-word: freq≥10) with MorphoLex-grounded morpheme layers and character fallback guaranteeing 100% coverage.

**Three preprocessing pipelines:** All working and producing clean output. The compound splitter is active in all three morpheme pipelines.

**Test infrastructure:** 60 structured test sentences across 11 linguistic categories, side-by-side output for all three pipelines.

**BPE baseline:** Trained on the same dataset, vocabulary size 10k, identical evaluation interface.

### Remaining Losses vs BPE (Accepted)

**Corpus coverage gaps** — `Goldman Sachs`, `World Series`, `New York Yankees`, `South Korea` don't appear frequently enough in our 6k article sample to enter the entity vocabulary. Not a design problem — a data scale problem.

**Genuinely unknown words** — `cryptocurrency`, `Instagram`, `YouTuber`, `mRNA`. Too new, too domain-specific, or abbreviations outside MorphoLex. Character fallback is honest here.

**MorphoLex quirk** — `demonstrated → de + monster + ate`. One known etymology-based mismatch, consistent and documented.

### Wins vs BPE (Clear)

- Inflected forms: `cats → cat[r]`, `running → run[r]`, `happiest → happy[r] + est[s]`
- Function words: `whether → whether[f]` (1 token) vs BPE's `W|he|ther` (3 tokens)
- Morphologically complex: `unbreakable → un + break + able` vs BPE's noise
- Named entities: `london[e]`, `north_korea[e]`, `european_union[e]`, `san_francisco_giants[e]`
- Compounds: `blockchain → block + chain`, `speedrun → speed + run`, `startup → start + up`
- Ambiguous words: clean single root tokens for `bank`, `light`, `left`, `bat`, `duck`

### What Comes Next

**`frozen_embedding.py`:** The custom PyTorch `nn.Module` implementing the hybrid frozen/trained embedding layer — the theoretical centerpiece of the thesis. Token metadata from the preprocessor (category, role, entity_label) feeds frozen dimensions; the remaining dimensions are learned normally.

**Dataset module:** PyTorch `Dataset` streaming WikiText-103 through any pipeline, yielding `(input_ids, target_ids)` pairs.

**Training loop:** Cross-entropy language model training for all three pipelines on identical architecture.

**Evaluation:** Perplexity on held-out validation set + probing classifiers for POS and entity type.

---

## 27. Designing the Frozen Embedding Layer

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

## 28. The Dataset, Model, and the Fairness Metric

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

## 29. Calibration and Literature Grounding

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

## Appendix: Key Decisions Summary

| Decision | Chosen Option | Rejected Options | Reason |
|---|---|---|---|
| Intervention point | Embedding layer | Tokenizer only; full architecture change | Transformer unchanged; clean separation |
| Unit granularity | Morpheme-level | Word-level; phrase-level | Smaller vocab; closer to BPE; better OOV |
| Named entity handling | Atomic units | Morpheme-segment them; ignore them | Proper nouns not compositionally meaningful |
| Vocabulary threshold | Source-based (MorphoLex) | Frequency-based | Frequency conflates noise with rare-but-real |
| Morfessor role | Affix-only fallback | Full segmentation; removal | Roots unreliable; affixes more trustworthy |
| Capitalisation guard | Rejected | Applied temporarily | Typographic heuristic, not linguistic principle |
| Number of pipelines | Three (BPE + morph + det) | Two (BPE + morph) | Isolates Morfessor's contribution |
| Codebase structure | Flat (7 files) | Deep hierarchy (15+ files) | Communication overhead too high |
| Evaluation method | Perplexity + probing | Generation-based metrics | No generation needed; probing more informative |
| Entity threshold | Split threshold (multi-word≥3, single≥10) | Flat threshold; MorphoLex exclusion; length filter; curated list | Structure of token is a reliable signal of misclassification risk |
| Compound splitting | Implemented with particle exception | Not implemented; minimum length only | Concrete wins; false positive risk negligible for genuine OOV |
| Compound particle handling | FUNCTION_WORDS exception for 2-char right parts | Minimum length 4; ignore particles | Particles are linguistically productive; blocking them causes `star+tup` class errors |
| Frozen dim content | One-hot (A) and feature vectors (B) — both built | Pretrained GloVe (C) | C muddles the thesis with externally learned signal |
| Frozen/trained combination | Concatenation | Addition | Addition lets training cancel the frozen signal, defeating guaranteed presence |
| Ragged structure | Fixed schema, zero-fill, explicit 'none' slots | Variable-width per token | Frozen vector must be uniform width; 'none' is meaningful, not missing |
| Frozen signal strength | Small N default + amplification/projection/probe levers in reserve | Accept weak 4.5% signal at N=512 | Frozen block must be a meaningful fraction of the vector to matter |
| Option B root features | B1 (one-hot) now; B3 (WordNet) later | B2 (coarse hand-annotated roots) | 15k roots can't be hand-annotated; WordNet is a discrete taxonomy, hand-features infeasible |
| Embedding experiment design | Phased matrix (pipelines × frozen variants) | Single fixed embedding design | Turns design uncertainty into measurable comparison, like the 3 pipelines |
| Embedding ownership | Model owns HybridEmbedding; dataset emits IDs only | Dataset emits vectors | Embedding is the thesis contribution; its trained part has gradients |
| Cross-pipeline metric | Bits-per-byte (BPB) | Per-token perplexity; BPC | Per-token PPL not comparable across tokenizations; BPB robust to non-ASCII; Mielke→Pile→Paloma lineage |
| Within-pipeline metric | Per-token perplexity | BPB only | Same tokenization → fair and more sensitive for detecting freezing effect |
| Document boundaries | Ignore (concatenate-and-chop) | Respect boundaries | Standard GPT-2 practice; seam noise negligible at L=256 |
| Corpus caching | Tokenize once, save int stream, mmap | Re-tokenize each epoch | spaCy + morpheme segmentation far too slow to repeat |
| Model role | Fixed apparatus, small, standard | Best/largest model | Transformer measures the embedding, isn't the variable; small keeps hypothesis testable |
| d_model | 128 (= N) | 256/512 | Pinned by frozen-fraction (Lever 1): 23-dim block must be a meaningful share |
| Model depth | 4 layers (shallow on purpose) | 6+ layers | Shallow keeps reliance on input representation, where the hypothesis lives; calibration-confirmed |
| Output tying | Configurable; 'none' for Phase 1 | Always tied; always untied | Untied keeps frozen role on input side; option C (tie frozen) reserved as hook |
| Architecture selection | Calibrate against BPE-baseline curve | Choose size by argument | Measurement beats judgment; confirmed Goldilocks regime (64.9 ppl / 1.438 BPB) |
| Reported metric point | Best-val (early-stopped) | Last-val | Overfitting after the minimum must not contaminate the comparison |

---

*Document updated through the calibration of the experimental apparatus: dataset, model, training loop, and the BPE-baseline anchor result (64.9 ppl / 1.438 BPB), with literature grounding for the eventual paper.*