# Literature Audit — Four Fairness Conditions

> **Kind:** reference — the four conditions a fair test of injected features must meet, and how
> the audited literature fares against them.

Every finding from other work, with its reference, is registered in
`docs/reference/related_work.md`. `docs/reference/literature_analysis.md` compares this project with the closest tokenization
work (Q1). This file asks a different question of the feature-injection work
(Q2): **could a reported gain be one of the artifacts this project found in
its own numbers?** Each condition below is an artifact we measured, not a
hypothetical one.

## The four conditions

| | condition | what goes wrong without it | where we measured it |
|---|---|---|---|
| **C1** | **Injective** — tokenizations compared are lossless | a tokenizer that silently drops or merges information is not measuring the same text | journey Part II; `verify.py` losslessness checks |
| **C2** | **Coverage-matched** — the feature exists for the same words in every arm | a feature present only for dictionary words helps exactly where coverage differs | journey §18–§22, the six-channel ledger |
| **C3** | **Causal** (language models only) — nothing the model reads may depend on text it has not yet seen | word-form features leak the rest of the word (86–97% of their gain sat on continuation fragments); tagger and parser labels depend on the right context (7–8% of tags) | §30, §33.3, §33.6 |
| **C4** | **Form-matched** — a same-shape null feature is the baseline | giving a feature its own input dims, or adding a vector to every input, improves a small transformer with NO information (+0.0097 ± 0.0015 BPB over eight random starts); in our codebase a shared seed matches initial weights only between identical shapes | §33.10–§33.16 |

C3 applies to language modelling only: a translation encoder or a classifier
sees its whole input, so context-dependent features are legitimate there.
C4 applies whenever a feature changes the input architecture; our effect was
measured in a 5M-parameter decoder-only transformer and may not transfer to
other architectures or scales — the audit flags **exposure**, not proven
artifacts.

## The audit

Verification: **P** = checked against the paper's text or an abstract quoted
in a secondary source; **V** = design detail still to verify in the paper. The
full texts of Alexandrescu & Kirchhoff, Sennrich & Haddow, Vania & Lopez and
Nzeyimana & Niyongabo Rubungo were read on 15 October 2026; no **V** remains.

| work | task | how features enter | C3 causal | C4 form | null / shuffle control | reported gain | what our findings imply |
|---|---|---|---|---|---|---|---|
| Alexandrescu & Kirchhoff (2006), factored neural LMs | LM; Egyptian Colloquial Arabic and Turkish | the factors of the previous n−1 words, each selecting its own row of size d (32–64), concatenated; plain words predicted (§2, §4–§5) (P) | **causal as built; exposed for Turkish**: its factors come from text disambiguated by Hakkani-Tür et al. (2002), which scores trigrams over the sentence, so a context word's analysis may depend on the words after it, the predicted word among them (§6); the Arabic factors come from the LDC ECA lexicon (P) | **exposed: width not fixed** — every factor adds its own row (P) | none: their "NULL" is an OOV strategy, zero vectors for unknown factors (§5), not a control (P) | perplexity gains | ancestor of our Q2; the missing null is the gap our §33.11 found |
| Botha & Blunsom (2014), compositional morphology (CLBL++) | LM, log-bilinear | **additive**: word vector plus morpheme vectors | fine: morphemes of already-seen context words | exposed: our additive null gained in a transformer (§33.14); untested for LBL | none reported; compared with the same model without morphemes | 2–6% perplexity on large data, **largest on Czech and Russian** (P) | the strongest positive result in the set, and their own rare-word test (gains shrink when singletons are excluded) is evidence of information, not form |
| Sennrich & Haddow (2016), linguistic input features for NMT | MT (source side) | **concatenated, total width fixed at 500**: words 350, small features ~10; the word embedding shrinks to make room (§4, Table 1) (P) | not applicable (encoder input) | **exactly our carved-slot form** | none: contrastive runs, each adding one feature to the baseline (Table 3) (P) | perplexity, BLEU and chrF3 improve; BLEU +1.5 German→English, +0.6 English→German, +1.0 English→Romanian (Tables 2, 4) (P) | the design our §33.11 showed can gain without information; whether an RNN encoder shows the effect is unknown — a same-shape null is the missing control |
| Vania & Lopez (2017), characters vs morphology | LM predicting whole words (the 5,000 most frequent and UNK; §3.3, §4.1) | context words represented from **gold** morphological analyses (oracle; the lemma and each feature a unit) vs characters, trigrams, BPE and Morfessor — segments change how context words are represented, not what is predicted, so they are not morphology in the tokenizer (P) | **exposed**: the gold analyses are UD v1.3's human annotations for Czech and Russian, made in sentence context (§4, §5.1), the leak §33.3 measured (7–8% of English tags); Arabic's MADAMIRA analyses are disambiguated in context too (§5.2, footnote 4) (P) | fixed: word, character and morph vectors are all 200-dimensional (§4.1) (P) | none (P) | oracle analyses beat all character models by a wide margin (P) | the oracle margin may partly be right-context information; a prefix-disambiguated oracle would bound it |
| Nzeyimana & Niyongabo Rubungo (2022), KinyaBERT | masked LM (encoder) | two-tier: a 4-layer transformer over POS, stem, affix and affix-set embeddings per word (P) | not applicable (bidirectional encoder) | **exposed at scale**: architecture and information change together | none: the baselines are a BPE BERT on the same corpus (same position encoding, batch size and steps), a morpheme-sequence BERT and XLM-R; the STEM→STEM ablation drops the morphology encoder, changing architecture and information together (§3.4, Table 6) (P) | +2% F1 NER, +4.3% average on machine-translated GLUE; in the ablation affixes help QNLI and SST-2, not NER or NEWS (P) | the gain may be the architecture; the null would feed random analyses through the same two tiers |
| MorphPiece (2023), Hou et al. (2023), Bostrom & Durrett (2020), FLOTA (2022) | tokenization (Q1) | no features: different segmentations | not applicable | not applicable | — | see `docs/reference/literature_analysis.md` | C1 and C2 are the relevant conditions; none reports an information-fairness audit (`docs/reference/literature_analysis.md`, Part 2) |
| **This project** | LM, BPE, 5M | carved slot and additive, learned and frozen | causal shift; lexical variants; right-context measured | **same-shape null in every family; initialization-matched** | yes: shuffles, nulls, zero and flat slots | **none survives the null** | |

## What the audit says

1. **No audited Q2 work reports a same-shape null.** Every positive feature
   result above compares a feature-bearing model with a baseline of a
   different form. On our evidence, that comparison can show a gain with no
   information at all. This is the methodological gap the thesis fills.
2. **Two of the audited causal LMs read analyses disambiguated in sentence context**,
   the configuration where §33.3 found a right-context leak: Vania & Lopez's gold
   analyses, the one causal-LM result with oracle morphology, and Alexandrescu &
   Kirchhoff's Turkish factors. Neither measures the leak, and its size in their
   languages is unknown. Vania & Lopez's segment arms do not test morphology in the
   tokenizer: they change how context words are represented, and the model predicts
   whole words.
3. **The positive results cluster in morphologically rich languages** —
   Czech and Russian (Botha & Blunsom), Finnish and Turkish (Hou et al., Q1),
   Kinyarwanda (KinyaBERT) — and are small or absent for English, matching our
   own English null. That is the case for Ukrainian (`docs/reference/ukrainian_pipeline.md`):
   the hypothesis has its best chance in a fusional Slavic language, and it can
   now be tested with all four conditions in place from the start.

## Read before citing (15 October 2026)

The full texts settled every **V** cell:
- total input width: not fixed in Alexandrescu & Kirchhoff, fixed in Vania & Lopez and in
  Sennrich & Haddow;
- no audited work ran a random-feature control;
- Vania & Lopez's gold analyses are UD v1.3's human annotations, made in sentence context.
