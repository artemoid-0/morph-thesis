# Research journey — Part 4: The Audits

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §14–§19. [← Part 3](3_the_apparatus.md) · [index](README.md) · [Part 5 →](5_grounding_and_controls.md)

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
**communication overhead**: deep hierarchies are expensive when files are read
and passed around one at a time.

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
weakness §8 and §10 already described qualitatively — now quantified and
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

