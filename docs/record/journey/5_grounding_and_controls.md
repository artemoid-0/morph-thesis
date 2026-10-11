# Research journey — Part 5: Grounding and Controls

> **Kind:** record — append-only; corrections are annotated in place, never silently rewritten. **Covers:** §20–§22. [← Part 4](4_the_audits.md) · [index](README.md) · [Part 6 →](6_level2_features_on_bpe.md)

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

### 21.1 Matched vocabulary — the Q1 confound, finally removed

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

### 21.2 Frozen vs learned factors — the thesis claim, tested directly

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

### 21.3 The scale confound — found in our own implementation

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

### 21.4 What survives, and what it means

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

### 21.5 Convergence speed — a free finding

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

### 21.6 One claim from §11 still untested

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

`scripts/eval/probe_dep.py` was written for this: a linear probe on the hidden state
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

### 22.1 Why English BPE is so hard to beat

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

### 22.2 How much of the deficit is ours, and how much is the language

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

### 22.3 The path considered and rejected — data starvation

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

### 22.4 The reframe — stop confronting BPE

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

### 22.5 Two levels of cooperation

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

### 22.6 Where this leaves the project

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

