# Related work — every finding from other work, with its reference

> **Kind:** reference — what other work found, as this project uses it, with full references.
> **Authoritative for:** findings from the literature and their bibliographic data. A finding from
> other work enters the project here first; documents that rely on it cite this file.

Each entry gives the reference, **what the work found** (numbers as the source
states them), and **where this project uses it**. Deeper comparisons live in
two older documents: the tokenization papers closest to Q1 in
[`literature_analysis.md`](literature_analysis.md), and the four-condition audit of
feature-injection work (Q2) in [`literature_audit.md`](literature_audit.md).

**Local copies.** Three cited works are kept as PDFs in `external/literature/` — gitignored,
never packed, as they are not ours to share: Borodavko (2025), Merity et al. (2016, the
WikiText-103 paper) and Gao et al. (2020, The Pile).

**How each entry was checked:** **F** — the full text was read; **A** — the
abstract or publisher page; **P** — checked against the paper for the cells used
in the audit (`literature_audit.md`); **S** — known only through another work's
account of it; **V** — still to verify before the thesis cites it (section 6).
Bibliographic data were checked against the ACL Anthology, PMLR, the
proceedings or the publisher, on 7 October 2026 unless noted.

## 1. Morphology in the tokenizer (Q1; DIRECTIONS D7)

**Borodavko (2025)** — Vitalii Borodavko, *Hybrid tokenization for Ukrainian
language: Using Morphemes and BPE*, MSc thesis, Ukrainian Catholic University,
Lviv (supervisor Artur Kiulian), 30 pp., hdl:20.500.14570/5694; code and data:
github.com/V1King30/morpheme-based-tokenization (commit f0a2cfa, May 2025; no
licence stated). **F.**
- Built morpheme vocabularies from three dictionaries, OCR'd and cleaned (with
  GPT-assisted validation): Yatsenko (1981) 117,393 words, 467,868 morphemes,
  17,843 unique, 3.98 per word; Polyga (2009) 35,444, 133,290, 7,650, 3.76; a
  private database of Taras Shevchenko National University of Kyiv 204,624,
  794,219, 29,952, 3.88 — Ukrainian words average 3.85 ± 0.1 morphemes.
- Tokenizers built on the morphemes, against BPE trained on UberText: on an
  UberText subset the morpheme tokenizers need 25.9–29.1 tokens per text, BPE
  20.0 (15k) to 25.3 (3k), Gemma's tokenizer 25.8, Llama's 30.5 — "no significant
  improvement", attributed to the dictionaries' mismatch with modern text.
- The planned language-model comparison (GPT-2 per tokenizer, top-1 and top-5
  next-token accuracy) did not converge: no language-model result.
- Its §2.4 attributes Arnett and Bergen's findings to Cotterell et al. (2018).
- *Used:* journey §33.61 — no language-model evidence on morphology in a
  Ukrainian tokenizer; per-token accuracy is not comparable across tokenizers;
  its digitized Polyga and Yatsenko dictionaries check the ending rule
  (`scripts/audits/ending_gold_check.py`, from a clone in the gitignored
  `external/`, never packed).

**Asgari, El Kheir & Sadraei Javaheri (2025)** — *MorphBPE: A Morpho-Aware
Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across
Morphologies*, arXiv:2502.00894; revised as Asgari, El Kheir, SadraeiJavaheri &
Nazari (2026), *MorphBPE: Morphology-Aware Tokenization for Efficient LLM
Training*, Findings of ACL 2026, pp. 41610–41621. **A** (both), method and data
from the arXiv full text.
- BPE whose merges never cross a morpheme boundary, the constraint applied
  while learning merges; boundaries from the SIGMORPHON 2022 segmentation data
  (English, Russian, Hungarian) and Arabic treebanks plus Farasa.
- 300M and 1B decoder models, ~6B and ~20B training tokens, vocabularies 24k
  (Hungarian) to 96k: lower cross-entropy, faster convergence, better
  morphological alignment (Hungarian F1 0.13 → 0.87). The arXiv version notes
  that cross-entropy is comparable only at equal vocabulary size and gives no
  per-byte normalization; the 2026 version adds that the gains "are not
  attributable to materially shorter tokens" and reports Belebele gains for
  Russian and Arabic.
- *Used:* the design of `bpe-endings` (journey §33.61): one boundary,
  inflection only, enforced when encoding as well as in training. *Compared*
  (journey §33.65): at 5M parameters and ~38M Ukrainian tokens, 10k steps, it
  ends 1.6% below BPE of the same vocabulary size in bits per byte, 4 of 4 seeds —
  the direction MorphBPE reports at 300M–1B parameters, here per byte.

**Arnett & Bergen (2025)** — Catherine Arnett and Benjamin K. Bergen, *Why do
language models perform worse for morphologically complex languages?*, COLING
2025, pp. 6607–6623; arXiv:2411.14198. **F.**
- MorphScore (does a token boundary fall at the morpheme boundary of interest)
  for 22 languages; "some evidence that tokenization quality explains the
  performance gap, but none for the role of morphological alignment" —
  alignment is higher for agglutinative languages, not lower.
- The gap between language types shrinks when training sets are equal in size
  scaled by the "byte premium": data size explains it.
- *Used:* journey §33.59 — a morphology-aware tokenizer is not guaranteed to
  help, which is why it is measured. *Compared* (journey §33.65): in Ukrainian,
  every inflectional boundary placed by a dictionary rule helped by 1.6%, while
  Unigram's own boundaries, which expose 43% of endings, gave nothing: partial
  alignment did not help, full alignment did.

**Schmidt et al. (2024)** — Craig W. Schmidt, Varshini Reddy, Haoran Zhang, Alec
Alameddine, Omri Uzan, Yuval Pinter, Chris Tanner, *Tokenization Is More Than
Compression*, EMNLP 2024, pp. 678–702; arXiv:2402.18376. **F.**
- 64 language models of 350M–2.4B parameters; fewer tokens do not give better
  downstream results; pre-tokenization and BPE-initialized vocabulary
  construction matter.
- "The top five tokenizers, PathPieceL-BPE, Unigram, BPE, BPE-Greedy, and
  WordPiece do not have any other row … significantly different from them";
  SaGe-BPE (rank 6) barely worse (p = 0.047). Ten tasks, 5-shot accuracy.
- *Used:* journey §33.59 — more subword algorithms would repeat D7's Unigram arm.

**Bostrom & Durrett (2020)** — Kaj Bostrom and Greg Durrett, *Byte Pair Encoding
is Suboptimal for Language Model Pretraining*, Findings of EMNLP 2020,
pp. 4617–4624, doi:10.18653/v1/2020.findings-emnlp.414. **F** (`literature_analysis.md`).
- Unigram tokenization aligns better with morphology than BPE and matches or
  beats it downstream (English, Japanese).
- *Used:* the Unigram arm of Q1 and of D7.

**Hou, Katinskaia, Vu & Yangarber (2023)** — *Effects of sub-word segmentation on
performance of transformer language models*, EMNLP 2023, pp. 7413–7425,
doi:10.18653/v1/2023.emnlp-main.459; arXiv:2305.05480. **F** (`literature_analysis.md`).
- Morfessor and StateMorph against BPE in Finnish, Russian, English and Turkish:
  large perplexity gains where morphology is rich (Finnish −30%, Turkish −17%),
  small or negative in English; faster convergence; smaller models matching
  larger BPE ones.
- *Used:* the reading of Q1's English negative (`literature_analysis.md`, Part 5);
  consistent with the Ukrainian result of journey §33.65, smaller (1.6% in bits
  per byte).

**Jabbar (2023)** — Haris Jabbar, *MorphPiece: A Linguistic Tokenizer for Large
Language Models*, arXiv:2307.07262 (v2, 2024; v1, 2023, titled *MorphPiece: Moving
away from Statistical Language Representation*); no venue — "under review".
**F** (`literature_analysis.md`).
- A MorphyNet-derived table of 134,943 words plus BPE fallback (50,006 tokens);
  GPT-2-size models beat GPT-2 on Penn Treebank token perplexity (61.9 → 38.3),
  LAMBADA, GLUE and MTEB; 17% more tokens than BPE; token perplexity across
  vocabularies acknowledged as not comparable.
- *Used:* Q1's closest comparison; the design choices adopted in `literature_analysis.md`.

**Hofmann, Schütze & Pierrehumbert (2022)** — *An Embarrassingly Simple Method to
Mitigate Undesirable Properties of Pretrained Language Model Tokenizers* (FLOTA),
ACL 2022 (short), pp. 385–393, doi:10.18653/v1/2022.acl-short.43. **F**
(`literature_analysis.md`).
- Longest-token segmentation with an existing vocabulary improves accuracy and
  robustness. *Used:* a cheap Q1 baseline, not built.

**Vania & Lopez (2017)** — Clara Vania and Adam Lopez, *From Characters to Words
to in Between: Do We Capture Morphology?*, ACL 2017, pp. 2016–2027;
arXiv:1704.08352. **F.**
- Ten languages; characters, character trigrams, BPE and Morfessor morphs
  composed by addition, bi-LSTM or CNN: trigrams with bi-LSTMs give the lowest
  perplexity in six of ten.
- "None of the character-level models match the predictive accuracy of a model
  with access to true morphological analyses, even when learned from an order of
  magnitude more data" (gold UD analyses, Czech and Russian).
- *Used:* `literature_audit.md` (oracle features read the right context: exposed
  to the leak of journey §33.3); the background to Q2.

**Toraman, Yilmaz, Şahinuç & Ozcelik (2023)** — *Impact of Tokenization on
Language Models: An Analysis for Turkish*, ACM TALLIP 22(4); arXiv:2204.08832.
**F** (arXiv version).
- RoBERTa models pretrained on Turkish (e.g. 42.7M parameters), six tasks:
  WordPiece and BPE best; a morphological tokenizer (Zemberek) close behind
  (differences of 0.01–0.02, significant); larger vocabularies help
  morphological and word-level tokenizers more; vocabulary at ~20% of the
  parameters for subword tokenizers, ~40% for the others.
- *Used:* related work for D7.

**Üstün, Kurfalı & Can (2018)** — *Characters or Morphemes: How to Represent
Words?*, Rep4NLP@ACL 2018, pp. 144–153. **F.**
- Morpheme-based word representations (segmentations weighted by attention)
  beat word-, character- and character-n-gram models for Turkish, most on
  syntactic tasks. *Used:* related work.

**Jeon, Yang, Kim & Lim (2023)** — *Improving Korean NLP Tasks with Linguistically
Informed Subword Tokenization and Sub-character Decomposition*,
arXiv:2311.03928. **F.**
- Morpheme-aware subwords with sub-character decomposition improve Korean
  tasks, notably the syntactic NIKL-CoLA. *Used:* related work.

**Liu et al. (2025)** — Alisa Liu, Jonathan Hayase, Valentin Hofmann, Sewoong Oh,
Noah A. Smith, Yejin Choi, *SuperBPE: Space Travel for Language Models*, COLM
2025; arXiv:2503.13423. **A.**
- Subwords, then superwords that bridge whitespace: at a 200k vocabulary up to
  33% fewer tokens; 8B models at fixed size, vocabulary and compute: +4.0% over
  30 tasks (+8.2% MMLU), 27% less compute at inference.
- *Used:* journey §33.59 — outside the design: its tokens span words, so a
  word's feature has no token of its own.

**Pagnoni et al. (2025)** — Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, et
al., *Byte Latent Transformer: Patches Scale Better Than Tokens*, ACL 2025,
pp. 9238–9258; arXiv:2412.09871. **A.**
- Bytes grouped into patches by the next byte's entropy; "the first FLOP
  controlled scaling study of byte-level models up to 8B parameters and 4T
  training bytes"; matches tokenized models, and scales better at fixed
  inference cost. *Used:* journey §33.59 — outside the design (its own
  architecture).

**Petrov, La Malfa, Torr & Bibi (2023)** — *Language Model Tokenizers Introduce
Unfairness Between Languages*, NeurIPS 2023; arXiv:2305.15425. **A.**
- The same text translated can differ "up to 15 times" in tokenized length;
  character- and byte-level models still differ over 4× for some pairs.
  *Used:* related work.

**Maksymenko & Turuta (2025)** — Daniil Maksymenko and Oleksii Turuta,
*Tokenization efficiency of current foundational large language models for the
Ukrainian language*, Frontiers in Artificial Intelligence 8:1538165. **F.**
- Fertility (tokens per word) in Ukrainian against English: Llama 3.1 1.88
  (1.08), Gemma 1.92 (1.07), GPT-4o 1.98 (1.07), Mistral 7B 2.48, GPT-3.5/4 3.32,
  GPT-2 6.31; most models need about one more token per word in Ukrainian;
  transliteration no longer helps.
- *Used:* context for the tokens per word of our Ukrainian tokenizers (§33.55).

## 2. Morphology as input features (Q2)

**Alexandrescu & Kirchhoff (2006)** — Andrei Alexandrescu and Katrin Kirchhoff,
*Factored Neural Language Models*, HLT-NAACL 2006, Short Papers, pp. 1–4 (N06-2001).
**F** (`literature_analysis.md`); the audit's design questions **V**.
- A word as a set of factors (stem, affixes, POS, …), each embedded and
  concatenated; aimed at rare words. *Used:* the ancestor of Q2.

**Botha & Blunsom (2014)** — Jan A. Botha and Phil Blunsom, *Compositional
Morphology for Word Representations and Language Modelling*, ICML 2014, PMLR
32(2):1899–1907. **P.**
- Word vector plus morpheme vectors in a log-bilinear LM: 2–6% perplexity on
  large data, largest in Czech and Russian; gains shrink without singletons.
  *Used:* `literature_audit.md`.

**Sennrich & Haddow (2016)** — Rico Sennrich and Barry Haddow, *Linguistic Input
Features Improve Neural Machine Translation*, WMT 2016, pp. 83–91,
doi:10.18653/v1/W16-2209. **P.**
- Linguistic features concatenated into a fixed 500-dim input (words 350):
  perplexity, BLEU and chrF3 improve. *Used:* `literature_audit.md` — the
  carved-slot form whose architecture effect §33.11 measured.

**Nzeyimana & Niyongabo Rubungo (2022)** — Antoine Nzeyimana and Andre Niyongabo
Rubungo, *KinyaBERT: a Morphology-aware Kinyarwanda Language Model*, ACL 2022,
pp. 5347–5363, doi:10.18653/v1/2022.acl-long.367. **P.**
- A two-tier encoder over morphological analyses: +2% NER F1, +4.3% on
  translated GLUE. *Used:* `literature_audit.md`.

**Ismayilzada et al. (2025)** — Mete Ismayilzada, Defne Circi, Jonne Sälevä, et
al., *Evaluating Morphological Compositional Generalization in Large Language
Models*, NAACL 2025, pp. 1270–1305; arXiv:2410.12656. **F** (arXiv v1).
- Turkish and Finnish: LLMs (Aya-23, Qwen-2.5, Gemini-1.5, GPT-4) struggle
  with novel roots, and accuracy falls "sharply … to nearly zero" as words grow
  more complex; GPT-4 best, 40–51 points below humans. *Used:* related work —
  morphology is not simply learned at scale.

## 3. Ukrainian resources and language-model work

**Chaplynskyi (2023)** — Dmytro Chaplynskyi, *Introducing UberText 2.0: A Corpus
of Modern Ukrainian at Scale*, UNLP 2023, pp. 1–10. **F.**
- 3.274B tokens before filtering, 2.49B after; 8.59M texts; news, fiction,
  court decisions, Wikipedia, social media. *Used:* Borodavko's corpus; a
  candidate second Ukrainian corpus (D9).

**Kiulian et al. (2024)** — Artur Kiulian, Anton Polishko, Mykola Khandoga, et
al., *From Bytes to Borsch: Fine-Tuning Gemma and Mistral for the Ukrainian
Language Representation*, UNLP 2024 (LREC-COLING), pp. 83–94. **F.**
- Fine-tuned Gemma and Mistral on Ukrainian; a fine-tuned Gemma 7B began
  code-switching. *Used:* related work.

**Kiulian et al. (2025)** — *From English-Centric to Effective Bilingual: LLMs with
Custom Tokenizers for Underrepresented Languages*, UNLP 2025, pp. 1–13;
arXiv:2410.18836. **F.**
- Vocabulary extension with new embeddings initialized from shorter tokens
  (Ukrainian, Arabic, Georgian): Ukrainian fertility Mistral 3.35 → 2.55, Gemma
  2.55 → 1.16; code-switching 0.515 → 0.001 of words; grammar correctness
  0.264 → 0.503. *Used:* related work.

**Batsuren et al. (2022)** — Khuyagbaatar Batsuren, Omer Goldman, Salam Khalifa,
et al., *UniMorph 4.0: Universal Morphology*, LREC 2022, pp. 840–855. **F.**
- 182 languages, 122M inflections, 769K derivations; Ukrainian paradigms in
  `unimorph/ukr` (Wiktionary and the VESUM dictionary). *Used:* not yet — a
  resource for paradigms.

**Polyga (2009); Yatsenko (1981)** — L. Polyga, *Dictionary of Ukrainian morphemes*
(Dovira, 2009); I. Yatsenko, *Morpheme analysis: dictionary* (Vyshcha Shkola,
1981). **S** (as digitized by Borodavko 2025). *Used:* the gold check of the
ending rule (journey §33.61).

**Korobov (2015)** — Mikhail Korobov, *Morphological Analyzer and Generator for
Russian and Ukrainian Languages*, AIST 2015, Communications in Computer and
Information Science 542, Springer, pp. 320–332, doi:10.1007/978-3-319-26123-2_31;
arXiv:1503.07283. **A.** pymorphy2: dictionary analysis with lexicons built from
OpenCorpora and LanguageTool data, and rules for words outside the dictionary.
pymorphy3 is its maintained fork; its Ukrainian dictionary (pymorphy3-dicts-uk
2.4.1) holds 417,330 lexemes (package metadata). *Used:* the ending rule
(`morphlm/data/endings.py`).

## 4. Measurement, training dynamics and evaluation

**Geirhos et al. (2020)** — Robert Geirhos, Jörn-Henrik Jacobsen, Claudio
Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann,
*Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence (2020);
arXiv:2004.07780. **A.** Shortcuts are "decision rules that perform well on
standard benchmarks but fail to transfer". *Used:* journey §32 (leaky features
plateau early).

**Shah et al. (2020)** — Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek
Jain, Praneeth Netrapalli, *The Pitfalls of Simplicity Bias in Neural Networks*,
NeurIPS 2020; arXiv:2006.07710. **A.** Networks can "exclusively rely on the
simplest feature and remain invariant to all predictive complex features".
*Used:* journey §32; `scripts/analysis/temporal_profile.py`.

**Yang, Gan, Dziugaite & Mirzasoleiman (2024)** — Yu Yang, Eric Gan, Gintare
Karolina Dziugaite, Baharan Mirzasoleiman, *Identifying Spurious Biases Early in
Training through the Lens of Simplicity Bias*, AISTATS 2024, PMLR 238:2953–2961;
arXiv:2305.18761. **A.** "Examples with spurious features are provably separable
based on the model's output early in training." *Used:* journey §32 (cited there
as "Yang & Dziugaite 2023").

**Jumelet, Weissweiler, Nivre & Bisazza (2026)** — *MultiBLiMP 1.0: A Massively
Multilingual Benchmark of Linguistic Minimal Pairs*, TACL 14 (2026), pp. 193–216,
doi:10.1162/tacl.a.600; arXiv:2504.02768. **A.** 101 languages, more than 128,000
minimal pairs; **Ukrainian included (2,744 pairs)** — a ready grammatical test for
the Ukrainian models. *Used:* the case for a second language (journey §32.4, G-31). Its
Ukrainian file, CC BY 4.0, is in `external/multiblimp/` (downloaded 9 October from the
Hugging Face dataset `jumelet/multiblimp`; SHA-256 7973fb33…): 2,744 pairs of subject–verb
agreement — person 1,181, gender 1,041, number 522 (journey §33.75).

**Cotterell, Mielke, Eisner & Roark (2018)** — *Are All Languages Equally Hard to
Language-Model?*, NAACL-HLT 2018, pp. 536–541, doi:10.18653/v1/N18-2085. **F.**
- 21 Europarl languages, n-gram and LSTM models, measured in bits per English
  character; "complex inflectional morphology [is] a cause of performance
  differences": difficulty correlates with inflectional complexity (ρ = 0.59),
  and not after lemmatization (ρ = −0.13). *Used:* `literature_analysis.md`;
  the motivation for testing morphology in a fusional language.

**Mielke (2019)** — Sabrina J. Mielke, *Can you compare perplexity across different
segmentations?*, blog post, 23 April 2019 (sjmielke.com/comparing-perplexities.htm).
**F.** Perplexities over different segmentations "aren't directly comparable,
but the log-likelihoods that are hidden inside them are": take the total
negative log-likelihood of the string and divide by a common unit. *Used:* the
choice of bits per byte (journey §13).

**Gao et al. (2020)** — Leo Gao, Stella Biderman, Sid Black, et al., *The Pile: An
800GB Dataset of Diverse Text for Language Modeling*, arXiv:2101.00027. **F.**
"Our preferred metric is bits per UTF-8 encoded byte (bpb)", for its "invariance to
different tokenization schemes". *Used:* journey §13.

**Magnusson et al. (2024)** — Ian Magnusson, Akshita Bhagia, Valentin Hofmann, et
al., *Paloma: A Benchmark for Evaluating Language Model Fit*, NeurIPS 2024
Datasets and Benchmarks; arXiv:2312.10523. **F.** Perplexity is its primary metric;
where vocabularies differ it "follows Gao et al. [2020] in using bits per byte".
*Used:* journey §13.

**Radford et al. (2019)** — Alec Radford, Jeffrey Wu, Rewon Child, David Luan,
Dario Amodei, Ilya Sutskever, *Language Models are Unsupervised Multitask
Learners*, OpenAI (February 2019). **F.** Zero-shot perplexity (Table 3):
WikiText-103 37.50 (117M), 26.37 (345M), 22.05 (762M), 17.48 (1542M); WikiText-2
29.41, 22.76, 19.93, 18.34. *Used:* journey §13's calibration points — corrected
there (they had given the WikiText-2 figure, ≈29, as WikiText-103's; ≈16 is the
Hugging Face perplexity guide's sliding-window WikiText-2 figure for GPT-2 Large).

**Dai et al. (2019)** — Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc
Le, Ruslan Salakhutdinov, *Transformer-XL: Attentive Language Models beyond a
Fixed-Length Context*, ACL 2019, pp. 2978–2988, doi:10.18653/v1/P19-1285. **F.**
WikiText-103 test perplexity 18.3 (257M parameters); the corpus has "103M training
tokens from 28K articles, with an average length of 3.6K tokens per article".
*Used:* journey §13.

**Baevski & Auli (2019)** — Alexei Baevski and Michael Auli, *Adaptive Input
Representations for Neural Language Modeling*, ICLR 2019; arXiv:1809.10853. **A.**
18.7 perplexity on WikiText-103 in the final version; 20.51 in arXiv v1, the
figure Transformer-XL's table lists (20.5). *Used:* journey §13 (which quotes 20.5).

**Hawthorne et al. (2022)** — Curtis Hawthorne, Andrew Jaegle, Cătălina Cangea, et
al., *General-purpose, long-context autoregressive modeling with Perceiver AR*,
ICML 2022 (PMLR 162). **F.** "Wikitext-103 is a small dataset where strong
regularization is required to prevent severe overfitting." *Used:* journey §13.

**spaCy `uk_core_news_sm`** — Explosion, model card and release notes. **F.**
Version 3.8.0: part of speech 98.13%, morphological features 94.57%, lemma not
evaluated; version 3.7.0: 98.1% and 94.7%. *Used:* the Ukrainian tagger behind
every Ukrainian feature (journey §32.4 quotes 94.7%).

## 5. Methods and data this project uses

Checked against the ACL Anthology, the proceedings or the publisher.

- **BPE** — Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with
  Subword Units*, ACL 2016, pp. 1715–1725, doi:10.18653/v1/P16-1162.
- **Unigram** — Kudo, *Subword Regularization: Improving Neural Network Translation
  Models with Multiple Subword Candidates*, ACL 2018, pp. 66–75,
  doi:10.18653/v1/P18-1007. **SentencePiece** — Kudo & Richardson, EMNLP 2018 System
  Demonstrations, pp. 66–71, doi:10.18653/v1/D18-2012.
- **Rotary positions** — Su, Lu, Pan, Murtadha, Wen & Liu, *RoFormer: Enhanced
  Transformer with Rotary Position Embedding*, Neurocomputing 568:127063 (2024),
  doi:10.1016/j.neucom.2023.127063; arXiv:2104.09864.
- **RMSNorm** — Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS
  2019. **SwiGLU** — Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202
  (2020).
- **WikiText-103** — Merity, Xiong, Bradbury & Socher, *Pointer Sentinel Mixture
  Models*, ICLR 2017; arXiv:1609.07843.
- **BLiMP** — Warstadt, Parrish, Liu, Mohananey, Peng, Wang & Bowman, *BLiMP: The
  Benchmark of Linguistic Minimal Pairs for English*, TACL 8 (2020), pp. 377–392,
  doi:10.1162/tacl_a_00321.
- **MorphoLex** — Sánchez-Gutiérrez, Mailhot, Deacon & Wilson, *MorphoLex: A
  derivational morphological database for 70,000 English words*, Behavior Research
  Methods 50(4):1568–1580 (2018), doi:10.3758/s13428-017-0981-8.
- **Morfessor 2.0** — Smit, Virpioja, Grönroos & Kurimo, EACL 2014 Demonstrations,
  pp. 21–24, doi:10.3115/v1/E14-2006.
- **MorphyNet** — Batsuren, Bella & Giunchiglia, *MorphyNet: a Large Multilingual
  Database of Derivational and Inflectional Morphology*, SIGMORPHON 2021,
  pp. 39–48, doi:10.18653/v1/2021.sigmorphon-1.5.
- **BERT** — Devlin, Chang, Lee & Toutanova, NAACL-HLT 2019, pp. 4171–4186,
  doi:10.18653/v1/N19-1423.
- **pymorphy2 / pymorphy3** — Korobov (2015), section 3.

## 6. Still to verify, and corrections made

**To verify:** *Neural Polysynthetic Language Modelling* (Schwartz, Tyers, Levin, et
al., arXiv:2005.05477, the final report of the 2019 Jelinek workshop) — journey §13
cites it for the per-character normalization recipe; its section on
character-level perplexity could not be read.

**Corrected in the record** (7 October 2026, from this register):
- GPT-2's calibration points (journey §13; `literature_analysis.md`, §8) — see
  Radford et al. above.
- KinyaBERT's second author is Niyongabo Rubungo (`literature_audit.md`).
- "Yang & Dziugaite 2023" is Yang, Gan, Dziugaite & Mirzasoleiman (AISTATS 2024)
  (journey §32; `scripts/analysis/temporal_profile.py`).
- Baevski & Auli's "20.5" is their arXiv v1 figure; the published one is 18.7.
- Borodavko (2025) §2.4 credits Cotterell et al. (2018) with Arnett and Bergen's
  findings — noted, not ours to correct.

## 7. To read (parked with D6, outside the thesis)

Xiao, Tian, Chen, Han & Lewis, *Efficient Streaming Language Models with
Attention Sinks*, ICLR 2024 (arXiv:2309.17453) — models attend strongly to
initial tokens as a "sink". Sun, Chen, Kolter & Liu, *Massive Activations in
Large Language Models*, COLM 2024 (arXiv:2402.17762) — a few activations
100,000× larger than others act as bias terms. Darcet, Oquab, Mairal &
Bojanowski, *Vision Transformers Need Registers*, ICLR 2024 (arXiv:2309.16588) —
extra input tokens absorb artifact high-norm tokens. BERT's segment embedding
(Devlin et al. 2019; section 5 above) — one learned vector added to every token of a segment.
All **A**; relevant only to the shared input vector (DIRECTIONS, Parked).
