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
in a secondary source; **V** = design detail still to verify in the paper.

| work | task | how features enter | C3 causal | C4 form | null / shuffle control | reported gain | what our findings imply |
|---|---|---|---|---|---|---|---|
| Alexandrescu & Kirchhoff (2006), factored neural LMs | LM | factor embeddings (stem, morph class, …) as inputs | likely fine: factors of context words (V) | exposed (V: whether total input width was fixed) | none (per `docs/reference/literature_analysis.md`) | perplexity gains | ancestor of our Q2; the missing null is the gap our §33.11 found |
| Botha & Blunsom (2014), compositional morphology (CLBL++) | LM, log-bilinear | **additive**: word vector plus morpheme vectors | fine: morphemes of already-seen context words | exposed: our additive null gained in a transformer (§33.14); untested for LBL | none reported; compared with the same model without morphemes | 2–6% perplexity on large data, **largest on Czech and Russian** (P) | the strongest positive result in the set, and their own rare-word test (gains shrink when singletons are excluded) is evidence of information, not form |
| Sennrich & Haddow (2016), linguistic input features for NMT | MT (source side) | **concatenated, total width fixed at 500**: words 350, small features ~10 (P) | not applicable (encoder input) | **exactly our carved-slot form** | none reported (V) | perplexity, BLEU and chrF3 improve (P) | the design our §33.11 showed can gain without information; whether an RNN encoder shows the effect is unknown — a same-shape null is the missing control |
| Vania & Lopez (2017), characters vs morphology | LM | word representations composed from **gold** morphological analyses (oracle) vs characters | **exposed**: gold analyses are disambiguated with the whole sentence, the leak §33.3 measured (7–8% of English tags) — likely larger in fusional languages (V) | arms differ in architecture by design | none reported (V) | oracle analyses beat all character models by a wide margin (P) | the oracle margin may partly be right-context information; a prefix-disambiguated oracle would bound it |
| Nzeyimana & Niyongabo Rubungo (2022), KinyaBERT | masked LM (encoder) | two-tier: a 4-layer transformer over POS, stem, affix and affix-set embeddings per word (P) | not applicable (bidirectional encoder) | **exposed at scale**: architecture and information change together | none reported (V) — no uninformative analyses through the same tiers | +2% F1 NER, +4.3% average on machine-translated GLUE (P) | the gain may be the architecture; the null would feed random analyses through the same two tiers |
| MorphPiece (2023), Hou et al. (2023), Bostrom & Durrett (2020), FLOTA (2022) | tokenization (Q1) | no features: different segmentations | not applicable | not applicable | — | see `docs/reference/literature_analysis.md` | C1 and C2 are the relevant conditions; none reports an information-fairness audit (`docs/reference/literature_analysis.md`, Part 2) |
| **This project** | LM, BPE, 5M | carved slot and additive, learned and frozen | causal shift; lexical variants; right-context measured | **same-shape null in every family; initialization-matched** | yes: shuffles, nulls, zero and flat slots | **none survives the null** | |

## What the audit says

1. **No audited Q2 work reports a same-shape null.** Every positive feature
   result above compares a feature-bearing model with a baseline of a
   different form. On our evidence, that comparison can show a gain with no
   information at all. This is the methodological gap the thesis fills.
2. **The one causal-LM result with oracle morphology** (Vania & Lopez) uses
   context-disambiguated analyses, the configuration where §33.3 found a
   right-context leak. Its size in their languages is unknown.
3. **The positive results cluster in morphologically rich languages** —
   Czech and Russian (Botha & Blunsom), Finnish and Turkish (Hou et al., Q1),
   Kinyarwanda (KinyaBERT) — and are small or absent for English, matching our
   own English null. That is the case for Ukrainian (`docs/reference/ukrainian_pipeline.md`):
   the hypothesis has its best chance in a fusional Slavic language, and it can
   now be tested with all four conditions in place from the start.

## To do before citing

Read the full texts for every **V** cell: whether Alexandrescu & Kirchhoff and
Vania & Lopez fixed total input width, whether any work ran a random-feature
control, and how Vania & Lopez's gold analyses were produced.
