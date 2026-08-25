# Literature Analysis — Where This Project Sits

*Prepared from full-text reads of the primary sources. Written for someone who
has not read the papers: each is summarised in plain terms before any
comparison is drawn.*

---

## Part 1 — The papers, in plain terms

### A. MorphPiece (Jabbar, 2023) — *the closest thing to our Q1*

**What they built.** A tokenizer that looks each word up in a morpheme
dictionary and, if found, emits its morphemes; if not found, falls back to BPE.
The dictionary ("MorphTable") was built from **MorphyNet**, a database of
derivational and inflectional morphology, giving **346,340 English words**,
trimmed to **134,943 entries** after dropping items with fewer than 5
occurrences in the training corpus. The morpheme inventory is **18,304 tokens**;
a BPE vocabulary of 32,000 was trained on top, for a combined **50,006** —
deliberately matched to GPT-2's 50,257.

**How they tested it.** Trained a GPT-2 Base architecture (117M parameters) on
OpenWebText and compared against OpenAI's GPT-2.

**What they found.** MorphGPT beat GPT-2 on Penn Treebank perplexity (61.9 →
38.3), on LAMBADA, on zero-shot GLUE, and across all seven MTEB embedding
tasks — while training for roughly half the steps.

**Three details that matter enormously for us:**

1. They use **canonical (lemma) segmentation**, exactly as we do: `batting` →
   `bat` + `ing`, not `bat` + `ting`. They explicitly name this as a likely
   reason for their gains.
2. Their tokenizer produces **17% more tokens than BPE**. They call this a
   limitation and leave "fusing morphemes to reduce this" to future work.
3. They acknowledge in Section 7 that **token-level perplexity is not
   comparable across vocabularies** (citing Cotterell et al. 2018) — and then
   argue for using it anyway, on the grounds that downstream gains follow.

### B. Hou, Katinskaia, Vu & Yangarber (2023, Helsinki) — *the most useful paper for us*

**What they built.** No new tokenizer. Instead a careful controlled comparison:
BPE vs two unsupervised morphological segmenters (Morfessor and StateMorph),
across **four languages** (Finnish, Russian, English, Turkish), two
architectures (GPT and BERT), several vocabulary sizes and two model sizes.

**Their four questions:** does morphological segmentation give (1) lower
perplexity, (2) faster convergence, (3) equal or better downstream performance,
(4) equal performance at smaller model size?

**What they found — and the English numbers are the critical part:**

| language | BPE | morphological | difference |
|---|---|---|---|
| **English** GPT, vocab 151k | 19.82 | Morfessor **20.05** | **BPE wins** |
| **English** GPT, vocab 177k | 20.33 | StateMorph 18.93 | morph wins 7% |
| **English** GPT, vocab 107k | 17.92 | StateMorph-pruned 17.55 | morph wins **2%** |
| **Finnish** GPT, vocab 78k | 23.69 | StateMorph-pruned 16.65 | morph wins **30%** |
| **Turkish** GPT, vocab 60k | 12.93 | StateMorph-pruned 10.67 | morph wins **17%** |

**For English the benefit is small or negative. For morphologically rich
languages it is large.** That is the single most important fact in this
literature review, and it is discussed in Part 5.

They also found morphological segmentation **converges faster** (often several
times fewer steps) and that a *smaller* morphologically-segmented model can
match a *larger* BPE model.

**A technique we should copy directly.** From their Appendix A.1: BPE marks
word-initial tokens with a special symbol; Morfessor and StateMorph do not. So
they "pre-segment all words in lower case, then convert the tokens back to the
original case, **mark all tokens appearing word-initially** with `_` (as BPE
does)". This is precisely the fix for our whitespace overhead (see Part 4).

### C. Bostrom & Durrett (2020) — *the standard citation*

Compared BPE against Unigram LM tokenization using **identical transformer
models**, on English and Japanese. Found Unigram recovers subword units that
align more closely with morphology, avoiding artifacts of BPE's greedy merging,
and that Unigram **matches or outperforms** BPE on downstream tasks. This is the
paper everyone cites for "BPE mis-segments morphology"; it is the reference
point our Q1 should be positioned against.

### D. FLOTA (Hofmann, Schütze & Pierrehumbert, 2022) — *the cheap alternative*

"Few Longest Token Approximation". Uses the **existing BPE vocabulary** but
changes *how* words are split, recursively taking the longest vocabulary
substring so that `undesirable` becomes `un`+`desirable` rather than
`und`+`es`+`irable`. No new vocabulary, no retraining. Improves accuracy, makes
inference more efficient, and increases robustness to whitespace noise.
MorphPiece beats it, but it is a very cheap baseline and a good comparison
point.

### E. Alexandrescu & Kirchhoff (2006), *Factored Neural Language Models* — *the ancestor of our Q2*

A word is represented not as one symbol but as a **set of features** — stem,
affixes, POS, capitalisation — each with its **own embedding**, and the
per-feature embeddings are **concatenated** to form the word's input vector.
Introduced specifically to help with rare words. Extended to neural machine
translation by Sennrich & Haddow (2016) and used widely since.

Architecturally this is our Q2. **The difference: their factor embeddings are
learned; ours were frozen.** In the sources reviewed, no one appears to have
tested frozen versus learned factors, nor run a shuffled-feature control.

*Both have since been done. Journey §21 built the comparison and found that
**learned factors beat frozen ones** by 1.7–2.3% under matched capacity, so the
specific novelty claimed here is refuted; frozen factors nonetheless capture
roughly 75% of the benefit at zero parameter cost. The shuffled control is in
§21 and §23.*

---

## Part 2 — The extended comparison table

| dimension | MorphPiece (2023) | Hou et al. (2023) | Bostrom & Durrett (2020) | FLOTA (2022) | Factored NLM (2006) | **This project** |
|---|---|---|---|---|---|---|
| **Core idea** | morpheme dict + BPE fallback | compare BPE vs unsupervised morph segmenters | compare BPE vs Unigram | re-segment using existing BPE vocab | word = set of feature embeddings | **BPE segmentation + injected morphological/syntactic features** |
| **Morpheme source** | MorphyNet, 346k words | Morfessor / StateMorph (unsupervised) | — | — | any annotation | **MorphoLex, 68.6k words** |
| **OOV handling** | **BPE fallback** | segmenter always produces something | — | — | — | **character fallback** |
| **Segmentation form** | canonical (lemma) | surface | surface | surface | — | **canonical (lemma)** |
| **Whitespace** | BPE-style prefix | **word-initial marker `_`** | BPE-style | BPE-style | — | **separate `<sp>` token (32.6% of all tokens)** |
| **Vocabulary size** | 50,006 (matched to GPT-2) | **matched across arms** (60k–516k) | matched | identical by construction | — | **17,151, matched against a retrained BPE-17k baseline** |
| **Model size** | GPT-2 Base, 117M | 132M–979M | RoBERTa-scale | BERT/GPT-2/XLNet | small RNN | **3.4M–5.2M** |
| **Languages** | English | Finnish, Russian, English, Turkish | English, Japanese | English | English, Turkish | **English** |
| **Corpus** | OpenWebText | Wikipedia + news, 17M–66M instances | — | ArXiv titles | — | **WikiText-103, 536MB** |
| **Headline metric** | token perplexity + downstream | token perplexity | downstream tasks | classification accuracy | perplexity | **bits-per-byte + per-token ppl** |
| **Downstream tasks** | GLUE, MTEB, LAMBADA | topic classification, POS, acceptability, paraphrase | GLUE-style | text classification | speech recognition | **none** |
| **Convergence speed** | noted (half the steps) | **measured explicitly (RQ2)** | — | — | — | **computable, now measured: 33–39% fewer steps with frozen dep** |
| **Fertility / token count** | measured: +17% | implied by vocab matching | — | measured | — | **measured: +104% (whitespace) / 1.51 vs 1.18 tok per clean word** |
| **Information-fairness audit** | **none** (acknowledges the problem, proceeds) | **none** | **none** | n/a | **none** | **six-channel ledger; coverage + collision + restricted BPB + category bits** |
| **Losslessness guarantee** | detokenization algorithm, heuristic | not addressed | not addressed | n/a | n/a | **injectivity-verified, byte fallback, 0.0008 BPB residual** |
| **Frozen features** | — | — | — | — | **learned** | **both built and compared; learned wins on perplexity, frozen costs no parameters** |
| **Feature control (shuffle)** | — | — | — | — | **none** | **yes — shuffling removes the gain; for learned factors it is worse than baseline** |
| **Frequency-stratified analysis** | — | — | — | — | — | **yes — U-shape; rare words +12.8%** |
| **Seeds / variance** | single runs | 3 runs on downstream, single pre-train | — | — | — | **two seeds on baseline and best cell: effect 6.6× seed noise** |

---

## Part 3 — What they did that we did not

Ordered by how much it would strengthen the thesis.

**1. Matched vocabulary size between arms.** Hou et al. deliberately tuned BPE's
vocabulary to match the morphological segmenter's, stating it is "only to make
the comparison fair." Our morph vocabulary is 17,151 against BPE's 10,000 — a
**+51% parameter advantage to morph** that we documented but never removed. This
is the most obvious methodological gap and it is cheap to close.

**2. Downstream task evaluation.** Every paper here evaluates on real tasks —
GLUE, MTEB, topic classification, POS tagging, paraphrase. We have only
perplexity and BPB. MorphPiece's whole defence of using token perplexity is that
downstream gains corroborate it; we have no corroboration either way.

**3. Multiple languages.** Hou et al.'s central finding is that the benefit
depends on morphological richness. We test English only — and English is the
weakest case (Part 5).

**4. Larger models.** MorphPiece uses 117M parameters, Hou et al. 132M–979M. We
use 3.4M–5.2M. MorphPiece's final limitation reads: "MorphPiece needs to be
trained on a larger model to see if the gains scale with model size."

**5. Convergence speed as a first-class metric.** Hou et al. make it RQ2, and
report morphological segmentation converging several times faster. We had the
data and never looked. *(Now computed — see Part 6, it is a win for us.)*

**6. A real detokenization algorithm.** MorphPiece devotes a full section to
reconstructing text from morphemes: token-type classification, a heuristic
automaton for word boundaries, and a reverse lookup table. We proved
*injectivity* (information is recoverable in principle) but never built the
decoder.

**7. Segmentation quality against a gold standard.** Bostrom & Durrett and FLOTA
both measure how closely a tokenizer's splits match gold morpheme boundaries. We
never evaluated our segmentation quality directly — only its downstream effect
on bits.

---

## Part 4 — What we did that they did not

**1. Information-fairness auditing. This is the real contribution.** Not one of
these papers checks whether the token streams being compared encode the same
information. MorphPiece explicitly acknowledges the problem — token perplexity
is not comparable across vocabularies — and proceeds anyway. Our audits found
that a plausible morpheme pipeline:

- silently dropped **5.32% of corpus bytes** (punctuation and digits) that BPE encoded
- collapsed `cat`/`cats` into identical token sequences, costing **0.0440 BPB**
- appeared to beat BPE by **6.3%** and actually **lost by 9.7%** once fixed

That is a worked, quantified demonstration that the field's standard comparison
method can invert a result. Nobody else has the instrument.

**2. Frozen rather than learned linguistic factors.** Factored NLMs are 19 years
old and learn their factor embeddings. Freezing them — guaranteeing the
information is present and undistortable — appeared untested, and testing it was
the agenda this review set. **The result was negative** (§21): learned factors
win on perplexity. What survives is narrower and still unoccupied — frozen
factors reach ~75% of the benefit with *no* trainable parameters, and the
*legibility* half of the freezing argument has never been measured by anyone,
including us.

**3. The shuffled-feature control.** We showed that permuting the dependency
codes (same distribution, same width, same parameter count) makes the model
**worse than baseline**, isolating the linguistic content at **7.5%**. No paper
here runs an equivalent control, so none can rule out "any structured input
helps."

**4. Frequency-stratified decomposition.** Our U-shaped result — morph beats BPE
by **12.8% on the rarest words** and loses by 10.5% in the middle band — is a
finding none of these papers reports, because none partitions the comparison by
word frequency. It explains *when* morphological segmentation helps rather than
just *whether*.

**5. Guaranteed lossless encoding.** Byte fallback, explicit inflection tokens,
case markers, verified injectivity (0.0008 BPB residual). MorphPiece's
detokenizer is heuristic and admits it "might fail for misspelled words."

**6. Fewer parameters, not more.** Our frozen features are carved *out* of the
128 dimensions, so the improved model has **2.63% fewer** trainable parameters.
Factored NLM concatenates factors, *adding* capacity.

---

## Part 5 — The reframe: our Q1 negative is confirmed, not anomalous

This is the most important conclusion of the review.

**Hou et al. tested Morfessor against BPE for English GPT and BPE won**
(19.82 vs 20.05). Their best English result was a 2% gain (17.92 → 17.55). For
Finnish the same method gained 30%.

English has poor morphology by design — a handful of inflectional suffixes and
mostly-opaque derivation. There is simply less structure to exploit than in
Finnish, Turkish or Russian. MorphPiece's English gains came from a **346k-word
dictionary, a 50k vocabulary, a 117M model and BPE fallback** — four advantages
we do not have.

So the correct reading of our Q1 result is not "we failed." It is:

> *At small scale, with a 68.6k-word dictionary and character fallback, on
> English, morpheme segmentation loses to ByteLevel BPE by 9.7% — consistent
> with the literature showing English is the weakest case for morphological
> segmentation. Our frequency-stratified analysis localises the loss and shows
> morphological segmentation still wins by 12.8% on the rarest words.*

That is a defensible, literature-grounded negative result with a positive
sub-finding, not a failure.

---

## Part 6 — What to adopt, how, and in what order

### Adopt 1 — word-initial marking instead of `<sp>` tokens
**Source:** Hou et al., Appendix A.1.
**Problem it fixes:** `<sp>` tokens are **32.6% of all morph tokens** and cost
0.245 bits each, contributing ~38% of the Q1 gap.
**How:** instead of emitting `<sp>` before a word, mark the word's *first*
morpheme as word-initial — exactly what ByteLevel BPE does with `Ġ`. Since only
prefixes, roots, function words, entities and characters can be word-initial
(never suffixes), this adds roughly 16.5k vocabulary rows, not 17k.
**Cost:** pipeline change + vocabulary rebuild + cache rebuild + 2 runs.
**Expected:** removes ~208k tokens per validation pass; the arithmetic in §18
put morph at ~1.394 against BPE's 1.395 — parity.
**Caveat:** doubles the morph vocabulary, worsening the parameter asymmetry.
Combine with Adopt 4.

### Adopt 2 — BPE fallback instead of character fallback
**Source:** MorphPiece, Figure 1.
**Problem it fixes:** character fallback handles **14.1% of the corpus at 2.654
bits/byte**, roughly double the average, and accounts for **47.9%** of the gap.
**How:** train a small BPE vocabulary on only the words MorphoLex cannot
segment (MorphPiece removes covered words from the BPE training corpus for
exactly this reason), and route unsegmentable words there instead of to
characters.
**Cost:** new BPE training + pipeline change + vocabulary and cache rebuild.
**Expected:** the *realistic* extrapolation in §19 (fallback words merely match
BPE's rate) gives morph 1.4529 vs BPE 1.3746 — still losing, but recovering
about half the fallback penalty. Combined with Adopt 1, plausibly a win.

### Adopt 3 — MorphyNet instead of MorphoLex
**Source:** MorphPiece §3.1.
**Problem it fixes:** our dictionary has 68,608 words; theirs 346,340. **77.5%
of our character fallback is capitalised named entities**, but the rest is
dictionary coverage.
**How:** MorphyNet is a public database of derivational and inflectional
morphology. Replace the MorphoLex parser with a MorphyNet loader; the rest of
the pipeline is unchanged since both produce (word → morpheme list).
**Cost:** one new parser module, vocabulary rebuild, cache rebuild.
**Risk:** MorphyNet uses a different segmentation convention; our multi-root and
format-B fixes are MorphoLex-specific and will not transfer. Budget debugging.

### Adopt 4 — match vocabulary size between arms
**Source:** Hou et al. §3.1.
**Problem it fixes:** the **+51% parameter advantage** to morph that we have
documented but never controlled.
**How:** retrain the ByteLevel BPE tokenizer at `--bpe-vocab 17151` to match the
morpheme vocabulary exactly. One command; we already have the flag.
**Cost:** BPE retrain (~1 min) + BPE cache rebuild + 1 run.
**Value:** removes the single largest confound remaining in Q1. **Cheapest
high-value item on this list.**

### Adopt 5 — a downstream task
**Source:** all five papers.
**Problem it fixes:** we have no evidence that BPB differences mean anything
practically, and every paper here treats that as essential.
**How:** the lightest option matching our scale is **POS tagging** or **topic
classification** with a linear probe on frozen model representations. Our
`eval/probing.py` already has the extraction half written.
**Cost:** a dataset + a probe trainer. No pre-training.
**Value:** high for thesis credibility; this is the standard rebuttal to "token
perplexity isn't comparable."

### Adopt 6 — convergence speed as a reported metric
**Source:** Hou et al. RQ2.
**Status: already computed, and it is a win.**

| cell | steps to ppl ≤ 6.5 | steps to ppl ≤ 6.3 |
|---|---|---|
| det baseline | 29,000 | 41,500 |
| det + frozen dep | **19,500** | **25,500** |
| morph baseline | 30,500 | never |
| morph + frozen dep | **21,000** | **27,000** |

**Frozen dependency features reach the same perplexity in 33–39% fewer steps.**
This is an independent second axis on which Q2 wins, it costs nothing, and it
directly matches a metric the literature already values.

### Adopt 7 — FLOTA as a cheap extra baseline
**Source:** Hofmann et al. 2022.
Uses the existing BPE vocabulary with morphology-preserving segmentation. If we
implement it, we get a third arm — statistical vocabulary, linguistic
segmentation — sitting between our two. MorphPiece used it as its comparison
baseline; a reviewer may ask why we did not.

### Recommended order

| # | item | cost | value |
|---|---|---|---|
| 1 | Convergence speed | **done** | high |
| 2 | Vocabulary matching (Adopt 4) | ~1 h | **highest per hour** |
| 3 | Word-initial marking (Adopt 1) | ~6 h | high |
| 4 | BPE fallback (Adopt 2) | ~8 h | high |
| 5 | Downstream probe (Adopt 5) | ~1 day | high for credibility |
| 6 | MorphyNet (Adopt 3) | ~2 days | medium, risky |
| 7 | FLOTA baseline (Adopt 7) | ~1 day | medium |

Items 2–4 together are the honest Q1 retry: they remove the parameter confound
and both mechanisms our own measurement identified as the cause of the deficit.

---

## Part 7 — Where the thesis stands against the field

**Q1 (segmentation) is well-trodden.** MorphPiece, Hou et al., Bostrom &
Durrett and FLOTA all occupy this ground, several with far more compute. Our
contribution here is not the tokenizer — it is the **audit methodology** and the
**frequency-stratified localisation** of where morphological segmentation wins
and loses.

**Q2 (injected linguistic features) is where the contribution sits.** Factored
NLMs are the direct ancestor and are 19 years old, but they *learn* their
factors and run no control. Since this review was written, §21 tested freezing
directly and **refuted it as the mechanism** — learned wins. What remains
unoccupied, and is what the project actually demonstrates, is the combination:
a shuffled-feature control isolating linguistic content, the rare-token
mechanism, the convergence-speed result, and features injected on top of the
*better* tokenizer rather than a competing one.

**The obvious next experiment, and the one a reviewer will demand:** frozen
`dep` versus **learned** `dep`. Same 8 dimensions, same information, one
trainable and one not. That is a direct test against the 2006 baseline, it is a
single configuration change, and it needs no cache rebuild.

---

## References

- Jabbar, H. (2023). *MorphPiece: A Linguistic Tokenizer for Large Language Models.* arXiv:2307.07262
- Hou, J., Katinskaia, A., Vu, A-D., Yangarber, R. (2023). *Effects of sub-word segmentation on performance of transformer language models.* arXiv:2305.05480
- Bostrom, K., Durrett, G. (2020). *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP 2020, 4617–4624
- Hofmann, V., Schütze, H., Pierrehumbert, J. (2022). *An Embarrassingly Simple Method to Mitigate Undesirable Properties of Pretrained Language Model Tokenizers.* ACL 2022 (Short), 385–393
- Alexandrescu, A., Kirchhoff, K. (2006). *Factored Neural Language Models.* NAACL-HLT 2006
- Sennrich, R., Haddow, B. (2016). *Linguistic Input Features Improve Neural Machine Translation.* WMT 2016
- Batsuren, K., Bella, G., Giunchiglia, F. (2021). *MorphyNet: a large multilingual database of derivational and inflectional morphology.* SIGMORPHON 2021
- Cotterell, R., Mielke, S.J., Eisner, J., Roark, B. (2018). *Are all languages equally hard to language-model?* NAACL-HLT 2018
