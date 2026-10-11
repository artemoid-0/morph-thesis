# Research journey — Part 1: The Question

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §1–§4. [index](README.md) · [Part 2 →](2_building_the_tokenizer.md)

*What we were asking, where in a language model to intervene, and what
linguistics says is knowable at each scope. This part is design only; nothing
was built yet.*

---

## 1. The Starting Point and the Intervention Point

**The thesis topic, as originally stated:** *Context-Aware Embeddings for
Tokenizer-Free Language Models.* The initial goal was vague — build some
representation of text based on morphology, syntax or semantics that replaces
tokens. No decision had been made about what the unit should be, where in the
model to intervene, or how success would be judged.

> **Retitled after the results were in** to *Context-Aware Morphological
> Embeddings for Language Models*. Two words changed, both for accuracy.
> **"Tokenizer-free" was never true**: every arm of this project uses a
> tokenizer, and §18 established that the best one is BPE. Nothing was ever
> replaced; the intervention moved to the embedding layer instead (§15).
> **"Morphological" became true**: the headline result (§23) injects root
> identity, word structure and morpheme role. The original title described an
> intention; the new one describes what was built. The original is preserved
> here because this document records what was believed at each point.

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

