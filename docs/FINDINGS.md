# What We Know — 15 October 2026

> **Kind:** claims — what the thesis claims now, each with its numbers, firmness and the journey
> sections behind it. **Authoritative for:** the claims. Rewritten in place when a result changes one.

The claims this thesis can make, each with its evidence and how firmly it
stands. The narrative of how each was reached is in `docs/record/journey/README.md`
(section numbers below); the numbers are in `docs/record/results.md`. A 5M-parameter
decoder-only transformer on WikiText-103 and on Ukrainian Wikipedia; gains are in
bits per byte (BPB), paired by seed, mean ± standard error.

## The two questions

**Q1 — Does segmenting English into morphemes beat BPE for language
modelling? No.** BPE wins by 13.5% at matched parameters (two seeds), 12.9% on
the held-out test set, and Unigram by 12.6% — against the morpheme arm with an
unsupervised fallback; the deterministic arm, which the article accepted for
publication reports, loses by 13.15%, 12.0% and 12.29% (results §5.9, §5.19–§5.20).
Morphemes help only the rarest
words (+9% below frequency 10), learn more slowly, and an ideal router between
the two tokenizers would gain at most 0.6%. *Firm; unaffected by anything
below — no features are involved.* (Parts II–IV, §32.3) **What lost was replacing BPE,
not morphology.** Kept inside BPE — BPE 17k that never joins a stem to its inflectional
ending, by a rule that reads the word alone, the Ukrainian result's tokenizer below — English
morphology helps: **+0.0046 ± 0.0010 BPB (0.30%, 4/4)** at a 10k budget, every run below
every BPE 17k run. That is a fifth of the Ukrainian gain, as English inflects far less: the
rule cuts 10.2% of words and reaches half of the inflected ones. It does not fade: at twice
the budget **+0.0055 ± 0.0008 (0.38%, 4/4)**, and at full training — 50k steps — **+0.0053 on
the median seed (0.39%)**, at three seeds of four. There, in each tokenizer one run in four
drifts after 20k steps, by 0.005 to 0.009, which leaves one pair level and inflates the mean
(+0.0069 ± 0.0037). The grammar test agrees at every seed: on BLiMP the full-training models
are **+1.0 ± 0.2 points** better over all 67 paradigms (4/4; +0.3 at 10k), +2.1 ± 0.8 in
morphology (4/4), and better on average in each agreement phenomenon (3/4 each). *Plain
baselines, four seeds, three budgets; at full training three seeds of four.* (§33.85, §33.87,
§33.92)

**Q1 in Ukrainian, with morphology inside BPE: yes — and the gain lasts, to full
training.** BPE that never joins a stem to its inflectional ending — the cut read from
the word form alone by a dictionary rule — beats plain BPE of the same vocabulary size by
**+0.0174 ± 0.0014 BPB (1.6%)** at a 10k budget, 4 of 4 seeds, from identical initial
weights, although it reads 11% less text in the same steps. **At widths 128 and 256 it does not
fade as a feature does.** At twice the budget the lead is +0.0169 ± 0.0007 (4/4), and **at full
training on four times the text — 151M tokens, 50k steps — +0.0165 ± 0.0008 (1.75%,
4/4)**, flat from step 5,000 to the end while the null falls from 1.13 to 0.944: at two
corpus sizes and three budgets it holds at about 0.017 wherever the null stands between
1.21 and 0.944 BPB, the axis along which a feature's gain falls to nothing. In a model twice as wide —
width 256, four times the core's parameters — it is larger: **+0.0254 ± 0.0012 (2.6%, 4/4)** at the 10k budget, and
at twice the budget, where it settles, **+0.0218 ± 0.0029 (2.4%, 4/4)**, flat from step
7,500 to the end (width 128 at 20k: +0.0169, 1.7%). It is larger at every matched null
BPB. **It peaks there.** At width 512 — 30M parameters, four times width 128 — it is
**+0.0125 ± 0.0009 (1.4%, 4/4)** at the 10k budget, below both smaller widths. At the same point
of learning it starts above width 256's (+0.041 against +0.038 at null BPB 1.18), then keeps
falling as the model learns, as a feature's gain does: below width 256's from 1.11 on, and at
width 128's level near 0.98 (+0.018), +0.014 at 0.92, while the smaller widths' leads hold
flat. At 20k, at one seed, it levels off: about +0.011 from step 7,500 to 20,000 (+0.0115, 1.4%,
at the end), smaller than at widths 128 and 256 but lasting, where a feature's gain falls to
nothing (§33.94, §33.96). Unigram, which cuts
words differently and into even more tokens, leads early, loses the lead by step 3,000
and gains nothing by the end (−0.0005 ± 0.0004): where the cuts fall matters, not how
many there are. Every tokenizer's validation stream decodes to the same text, byte for
byte, so each one's bits per byte is a code length for that text (G-75). **And it shows
in grammar, not only in bits per byte.** On MultiBLiMP's 2,744 Ukrainian subject–verb
agreement pairs the endings-apart models prefer the grammatical sentence more often: at
full training 76.3% against 71.4% (**+4.9 ± 0.4 points**, 4/4; in number, gender and
person; +5.0 on the pairs whose two sentences have the same token length), at ~38M and 10k
69.0% against 64.8% (+4.3 ± 0.3, 4/4); +4.5 ± 0.5 at twice the budget, +5.3 ± 0.3 at width 256 and
**+6.8 ± 0.2 at width 512** (4/4; +5.6 at 20k, seed 0), where the lead in bits per byte is half width 256's: the
grammatical gain grows with the model, and is not the code length's. Exposing more endings elsewhere does not
do it: BPE 4k is 1.2 points below BPE 17k, Unigram level with it. **And on a second corpus.**
On UberText 2.0's fiction subcorpus — novels and prose, both tokenizers trained on it — keeping
endings apart beats BPE 17k at every seed, **+0.0097 ± 0.0010 BPB (0.8%, 4/4)** at a 10k
budget, half its share on Wikipedia, and on MultiBLiMP by **+5.3 ± 0.6 points (4/4)**, as much
as on Wikipedia or more. *Two corpora (fiction at one size and budget), two corpus sizes and
three budgets on Wikipedia, three widths (512 at 20k at one seed).* (§33.61–§33.68, §33.76,
§33.79, §33.86, §33.88, §33.94, §33.96)

**Q2 — Do injected linguistic facts help? No — in either way of injecting
them, and in either language.** Against a **null of the same shape** — which in this codebase sees
the feature cell's data and dropout at the same seed, and starts from its initial weights when
the feature has the null's eight classes; part of speech (19 classes) and inflection (72) share
data and dropout only (§33.15):

| feature | against its null | seeds |
|---|---|---|
| word structure, given its own input dims | +0.0004 ± 0.0005 | 4 |
| word structure, added to a full-width embedding | −0.0013 | 4 |
| structure shuffled among words of similar frequency | +0.0015 ± 0.0005 | 4 |
| real structure against its shuffled version | −0.0012 ± 0.0008 | 4 |
| frequency band of the word | +0.0007 ± 0.0003 | 2 |
| part of speech (tagger) | +0.0012 ± 0.0007 | 4 |
| part of speech (strictly causal) | −0.0007 | 2 |
| inflection (strictly causal) | **−0.0046 ± 0.0013** | 4 |
| **Ukrainian** case (tagger) | −0.0013 ± 0.0008 | 4 |
| **Ukrainian** case (strictly causal) | −0.0020 ± 0.0010 | 4 |

*Firm for structure and for Ukrainian case; the two-seed rows are consistent
with them. The Ukrainian rows are against a null constant on every word — as
first recorded, −0.0009 and −0.0015 (G-73, §33.56).* **Before the model has learned it, the answer turns: yes.** At a 10k-step
budget Ukrainian case beats its same-shape null at every seed: **+0.0076 ±
0.0016** at ~1.5M train tokens, +0.0031 at ~7.6M, +0.0032 at ~38M, **+0.0026 ±
0.0004** at 151M — and on the same 151M tokens +0.0055 ± 0.0006 at 5k steps,
nothing at 20k (−0.0003 ± 0.0004) or 50k. *Restated against a null constant on
every word (G-72, G-73, §33.55–§33.58; re-runs reproduce them exactly): as first recorded, +0.0096, +0.0052,
+0.0052, +0.0044 and +0.0015 at 20k — the null had been scored with words unseen
in its train split marked as non-words, and 20 runs on another corpus size's
lexicon.* Case read from the word form alone keeps two-fifths to two-thirds of
it (+0.0050 at ~1.5M, +0.0014 at ~7.6M) and about a third at ~38M under six
tokenizers, so it is case, not only the tagger's right context — shares that are lower
bounds, since the word form marks words unseen in train `none` and its model falls below
the null on them (G-76, §33.74). **English agrees**: part of speech at the same
10k-step budget beats its null by **+0.0130 ± 0.0004** (4 seeds) — five times
Ukrainian case on 151M tokens (+0.0026). (§33.31–§33.33, §33.38) **What reading the sentence adds over reading the word form is mostly not context —
and at full training none of it is.** At
~38M tokens and 10k steps Ukrainian case from the full sentence leads case from the word
form alone by +0.0021 to +0.0033 under all six tokenizers (4 seeds;
§33.65, §33.81). **A part is the tagger's right-context leak** (G-38): a preposition's case in
context is the next noun's, read from the word being predicted. Measured in trained
models, with prepositions reading their form's case and every other word its case in
context, it is **29% of the lead under BPE 17k (+0.0006 ± 0.0002) and 43% under endings
apart (+0.0014 ± 0.0004)**, 4/4 each. What remains (+0.0015 ± 0.0003, +0.0018 ± 0.0002)
is about half words unseen in train, where the word form gives nothing and misleads its
model below the null on them (−0.0006, −0.0007; G-76), +0.0002 to +0.0003 context
resolving another word's case, and +0.0004 to +0.0005 spread over targets that both models
see alike. **At full training — 151M tokens, 50k steps — the whole lead is the leak.**
Under endings apart case leads its word form by +0.0014 ± 0.0002 (4/4), and reading
prepositions from their form takes away +0.0015 ± 0.0001 (4/4): the control and the word
form end level (−0.0001 ± 0.0003). Along the law's axis context without the leak falls as a
feature's gain falls — +0.0011 to +0.0013 over the word form at null BPB 1.25 to 1.05, +0.0002
at 0.94 — while the leak, which carries the case of the noun being predicted, keeps +0.0012 to
+0.0014 to the end. Under BPE 17k the lead at full training is +0.0006 ± 0.0004 (3/4), the
prepositions' part still positive (+0.0008, 4/4). A context-aware feature helps before the
model has learned to read the context for itself; after that, what it adds is what the
tagger saw of the word being predicted. *The control at four seeds: under two tokenizers at
~38M, under endings apart at full training. A split of the lead by the word the feature
describes had put 55–60% after prepositions — two trained models compared target by target
are not a model and its counterfactual (§33.74, §33.76, §33.81).* Identity-like
features (inflection bundles, lemma and root hashes) do worse than the null.
BLiMP shows no grammatical gain from any feature, and MultiBLiMP none from Ukrainian case:
within 0.4 points of its null at both sizes, under both tokenizers. (§33.9–§33.19, §33.79)

## What linguistic information does do: speed early learning

Every feature that informs the next word — part of speech, inflection, case,
frequency band — beats its matched null early in training, at every seed, in
both languages: at step 2,000, English part of speech **+0.0140 ± 0.0011**
(4 seeds) and Ukrainian case **+0.0056 ± 0.0005** (4 seeds, against a null
constant on every word; first recorded +0.0060, §33.58), and every other
informative feature and design likewise. Every such advantage is gone by
the end of training. Structure, which says little about the next word, gives
none. The model uses the information, then learns it itself. *Firm as a
pattern (2–4 seeds per feature, 21 feature-null pairs, none an exception).* And where the
model has not yet learned the information — little training, or data too
scarce to learn it — the advantage survives to the end (§33.31–§33.33). In
training steps the early help is worth **1–9%**: part of speech 7–9%,
inflection 3–5%, Ukrainian case 1–3% against the constant null (as first recorded 2–4%;
§33.33, §33.91). **It is a crutch, not a
teacher**: switched off after the first quarter of a 10k budget, English part
of speech drops the model from 0.016 ahead to 0.029 behind one that never had
it, and the model ends below that one (−0.0023 ± 0.0006, 4 seeds; "Removing a
feature", below). The model learns to use the feature, not better
representations from it (§33.34, §33.38). (§33.24, §33.28)

**No benefit hides in the rare tail.** At the end of 50k runs, by token
frequency, English part of speech keeps its gain only in the most frequent
tokens (4/4 seeds in the top two buckets); English inflection is worse than its
null in rare and mid-frequency tokens (0/4 positive) — where its end deficit
lives; Ukrainian case is neutral to negative across the tail. *Firm (4 seeds);
by BPE-token frequency, not word rarity.* (§33.47)

## How much a feature helps depends on how much the model has learned

Plotted against the null's own validation BPB, a feature's gain traces one
curve whatever the training length or budget. English part of speech under 5k,
10k, 20k and 50k schedules deviates from a single curve by at most 0.0010
(average 0.0002–0.0005); four other English features — inflection, part of
speech from the word form, frequency band, structure — agree across 10k, 20k and
50k schedules within 0.0004 at null BPB 1.55 (4 seeds per schedule; 2 at 50k for
word form and frequency band). The curve predicts each feature's gain at a short
budget — at 10k steps part of speech +0.0130, part of speech from the word form
+0.0099, inflection +0.0067, frequency band +0.0046, structure ≈ 0; at 20k
+0.0075, +0.0050, +0.0022, +0.0014 and −0.0008 — and each feature has its own
point where the gain reaches zero. *Firm in English; in Ukrainian within the seed spread.*
Against the constant null, five Ukrainian conditions — 5k and 50k steps on 151M tokens, 10k on
~38M, ~7.6M and ~1.5M — agree within 0.0015 at null BPB 1.30, and closer as the null learns:
within 0.0006 at 1.20, and the three that reach 1.15 within 0.0001 (+0.0045 to +0.0046). At
1.10 the 50k run is 0.0008 below the ~38M run (+0.0029 against +0.0037), then in its last
steps. At every point the conditions differ no more than their seeds do: dealing the seeds'
values to the conditions at random gives as large a gap 24–100% of the time, each condition's
seeds spreading by 0.0004–0.0015 (sd). Against the first null, which was not constant, the 50k
run had sat 0.0015 below (G-73). (§33.35, §33.38, §33.49, §33.52, §33.58, §33.91, §33.98) Across six Ukrainian tokenizers case's gain agrees at
null BPB 1.25 — all within 0.0006 of +0.0052 (4 seeds) — and parts later, where
endings are exposed (next section). (§33.58, §33.65)

## Morphology in the tokenizer and in the embedding (Ukrainian)

**They add up.** Under BPE that keeps endings apart, case still beats its null at
every seed (**+0.0048 ± 0.0006**, 4/4), and the two together end **+0.0222 ±
0.0017 BPB (2.0%)** below plain BPE 17k's null — the tokenizer contributing four
fifths. Case beats its null under each of six tokenizers (BPE at 4k, 8k, 17k and
32k, Unigram at 17k, endings apart; +0.0029 to +0.0054, every seed), and leads
early under each (+0.0045 to +0.0090 at step 2,000).

**Exposed endings make case more useful, not redundant.** The hypothesis behind
the comparison was the reverse: case should help where the tokenizer hides
endings. Paired by seed from identical initial weights, case gains more under
Unigram 17k, which exposes 43% of endings, than under BPE 17k, which exposes 9% —
**+0.0021 ± 0.0006 (4/4)** at the same null BPB — and under endings apart, which
exposes all, +0.0016 ± 0.0010 (4/4), its null 0.017 further along. Along the law's
axis the six agree up to null BPB 1.25; after it the two that expose endings gain
again — at 1.15, +0.0031 ± 0.0006 (Unigram) and +0.0043 ± 0.0013 (endings apart)
over BPE 17k, 4/4 each — while the BPE curves fall. Case read from the word form,
which reads no context and so cannot leak, shows the same pattern at about two
thirds of the size (+0.0021 ± 0.0008 and +0.0033 ± 0.0017 at 1.15, 4/4 each).
Unigram's early lead is real:
at null BPB 1.40, +0.0094 against BPE 17k's +0.0045 (+0.0049 ± 0.0009 paired, 4/4).
*Firm at this size (4 seeds), ~38M tokens and a 10k budget; where it
sits, below.* (§33.58, §33.65) At the end of that budget about half of the extra under
endings apart is the tagger's leak: with prepositions read from their form it is +0.0008 ±
0.0012 (3/4), not +0.0016; at null BPB 1.15 it is mostly not (+0.0035 ± 0.0015 against
+0.0043; §33.76). **And the difference grows with training, to full training.** At twice
the budget case under BPE 17k has faded to +0.0010 ± 0.0008 (3/4), while under endings
apart it holds +0.0038 ± 0.0008 (4/4; paired +0.0029 ± 0.0016, 4/4). At 151M tokens and
50k steps, where under BPE 17k case ends below its null (−0.0013 ± 0.0008), **under
endings apart it still beats its null at every seed, +0.0030 ± 0.0009** (paired +0.0044 ±
0.0011, 4/4), +0.0033 at null BPB 0.94; the two together end +0.0195 ± 0.0013 below BPE
17k's null. Where endings are their own tokens, case keeps helping after it has stopped
helping under BPE 17k — **and half of it does without the tagger's leak**, which there can
reach the noun's ending, a token of its own. With prepositions read from their form case
still beats its null, +0.0015 ± 0.0010 (3/4); read from the word form alone — no context,
so no leak — +0.0016 ± 0.0009 (3/4), which is **+0.0036 ± 0.0010 (4/4)** more than the word
form gains under BPE 17k, where it ends below its null. What lasts to full training where
endings are tokens is the word's own case; the other half is the leak (+0.0015 ± 0.0001,
4/4). *Four seeds each; the gains without the leak at three seeds of four.* (§33.68,
§33.76, §33.81)

**Case helps by predicting endings; the exposure effect lies elsewhere.** Re-scored
token by token, case's gain sits on the tokens that hold an inflectional ending under
every tokenizer — all of it under BPE 17k, where it helps only endings that fall in a
word's later token, not those fused into a word's first token; about three fifths of it
under Unigram and under endings apart, on a fifth of the targets. But the words the rule
cuts gain about the same under all three (+0.0036 to +0.0041 at 10k): the exposing
tokenizers' extra comes from words it does not cut (+0.0010 ± 0.0003 under endings
apart, +0.0014 ± 0.0004 under Unigram), and at 20k under BPE 17k the feature costs
everything but the endings (uncut words −0.0008, non-word targets −0.0007) while under
endings apart it does not. Under endings apart case read from the word form barely helps
the endings (+0.0005, against +0.0028 from the sentence): the previous word's ending is in
plain view there, and what the sentence adds on the endings is largely the leak at
prepositions, whose case the tagger reads from the noun being predicted: read from their
form, case's gain on those tokens falls from +0.0028 to +0.0012 (§33.74, §33.76). *Four
seeds at 10k, three at 20k.* (§33.69–§33.70, §33.74, §33.76)

**What the model builds for itself: the form's grammar, more accurately; context, no
better.** Probed linearly at each word's last token, the full-training nulls, which carry no
morphology in their input, hold a word's grammar about as well as its form's most frequent
reading does. Case is right for 83.2% of words under BPE 17k and 84.7% under endings apart,
against the form's 84.1% and 83.8%.

Keeping endings apart makes that account more accurate where the form decides (paired, 4/4
each):
- **number** +2.5 points;
- **gender** +5.7 points;
- **case where the form's majority case is right** +2.3 points.

These are the features subject–verb agreement turns on, where those models judge better.
Where only context decides a word's case, both tokenizers' models read it weakly, 10–13 points
over that subset's floor, and endings apart is 2.5 points lower (0/4). Read one word later the
models hold less, not more: a model this size states a word's grammar at the word and does
not revise it. *Four seeds each, one corpus, linear probes, the tagger's labels.* (§33.84)

**Within one model size only.** At null BPB 1.55 a model twice as wide gains +0.0036
where the base model gains +0.0134 — its curve collapses abruptly between 1.70
and 1.60 — and at a 20k budget its part of speech ends at its null (−0.0008 ±
0.0005, 4 seeds); a model half as wide (64) gains less throughout. The collapse is not
the slot's share (tiled to the base share: +0.0024 and +0.0037 at 1.55, 2 seeds),
nor the model's linear knowledge of the current word's part of speech: a probe
finds the wider model's deeper layers ahead at every matched null BPB (+0.8 to
+1.3 points, input layers alike), yet at the same probe accuracy (0.870) the base
model gains +0.0133 and the wider one +0.0076 (one probed seed per width). **What
the feature carries — the previous word's part of speech — accounts for about half
of the difference.** The wider model reads it from context far earlier (+3.9 to
+10.4 points at matched null BPB), and on that axis the widths' gains differ by at
most 0.0035, against 0.0063 on the current word's tag and 0.0098 on the null's
BPB; at the top of the shared range they meet (+0.0075 at accuracy 0.62). Not
fully: the law is stated within one model size. The feature stops paying well
before the null knows the tag fully — the wider model's gain collapses when its
deepest layer reads it at 0.68 — and at the end of a 20k budget the base null
reads it at 0.619 (the current word's: 0.881). *The probes, one seed per width.*
(§33.40, §33.42, §33.49–§33.50, §33.52, §33.60)

**Removing a feature leaves a lasting deficit; adding one late does not.**
Switched off at step 2,500 of a 10k budget, English part of speech ends 0.0023
below a model that never had it; at 7,500, 0.0118 below; and at 5,000 of a 20k
budget — with 15,000 steps to recover — still **0.0039 ± 0.0009** below (5 of 5
seeds), the gap levelling off near 0.004 from step 12,500 while the learning rate
is still a third of its peak. **It is not lost knowledge of part of speech**: a
probe finds the switched model's deepest layer 1.2 points behind one step after
the switch, level 1,000 steps later, and level or ahead to the end (seed 4). Nor
of what the feature carried, the previous word's tag: rebuilt from context within
1,000 steps of the switch, it ends 2.0 points better read at the deepest layer
than in the model never fed it (seed 4).
Switched *on* late the feature is integrated slowly but well: with 15,000 steps
left it recovers 54–104% of the kept gain (mean 73%). Weaning — dropped at a
rising rate, fully by step 5,000 — leaves 70% of the abrupt switch's deficit
(−0.0029 ± 0.0016 against −0.0042 ± 0.0011, 4 seeds). *Firm (4–5 seeds; the
probe, one seed).* (§33.34, §33.38, §33.40, §33.42, §33.50, §33.52)

**As a target, morphology teaches — and the knowledge stops paying.** Predicted
as an auxiliary target from the final hidden state (weight 0.1), English part of
speech beats an information-free target — the same head on shuffled labels — by
+0.0094 at step 1k, fading to **+0.0009 ± 0.0008** at the end of a 20k budget (5
seeds, 3 positive): faster than the same feature as an input (+0.0150 at 1k,
+0.0075 at the end). The teacher works — a probe finds part of speech better
encoded at every layer of the real-target model, its input embeddings included:
+4.6 points at layer 2 by step 2,000 and still +1.6 at the end (seed 4) — yet the
advantage in prediction fades: knowing more part of speech than language modelling
itself acquires does not help predict text. Any auxiliary head helps a little
(the shuffled target +0.0006 ± 0.0004 over the plain baseline); the morphology in
it does not, at the end. *Firm (4–5 seeds; the probe, one seed).* (§33.50, §33.52)

## What the slot did instead: a token-independent input

For the thesis this is a confound, and the reason for the same-shape null: what
first looked like word structure helping was the feature slot itself, helping
with no information in it (method 3, below; §33.10–§33.11). How a shared vector
helps a small transformer is a question about such models in general, parked as
outside the thesis (§33.59); the measurements stand.

**One learned vector shared by every input.** Under rotary positions — nothing
positional in the input at all — a single learned vector, the same at every
position, improves the plain baseline by **+0.0064 ± 0.0004 BPB** in 8 dims of its
own (4 of 4 seeds) and **+0.0072 ± 0.0008** added across all 128 dims (3 of 3).
The same 8 dims held at exactly zero give nothing reliable (+0.0013 ± 0.0015);
from identical initial weights the vector beats the empty slot by +0.0051 ± 0.0017
(4/4), and added it beats zeros added by +0.0060 ± 0.0016 (3/3). Whether it also
marks words is irrelevant (the word-marking constant slot: +0.0057 ± 0.0006).
Added to every input, the vector is a reparameterization of the token embedding —
it lets the model reach nothing it could not reach without it — so the gain lies
in how the model learns, not in what it can represent. *Firm under rotary
positions.* It is the vector, not a mark of the window's start: the causal shift
gives each window's first word a sentinel class, which a rotary model cannot
otherwise see, but without the shift — a pure constant, the same initial weights
— the flat slot keeps 89% of its gain (+0.0068) and the vector added 82%
(+0.0059) at seed 0 (G-71; §33.51–§33.52, §33.58).

**With learned positions the same effect looked like a position channel.** There
8 dims kept from token identity, holding the null's constant, improve on eight random
starts of the plain baseline by **+0.0097 ± 0.0015 BPB (0.72%)**: every null
run beats every baseline start, and training becomes four times less sensitive to
its random start. Held at zero they gain about as much, +0.0105 ± 0.0035 over the baseline
(4 seeds), for such dims are never empty: learned position fills them — a
token-independent signal. Taking position out of the slot costs the zero slot −0.0082 ±
0.0012, and taking it out of the token dims instead −0.0062 ± 0.0022. One vector added to
every input gains about as much (+0.0085 to +0.0090 over its initialization-matched
control, 4 of 4) — and there the window start is visible to the model anyway. The
slot's width barely matters (8 to 64 dims). *Firm.* (§33.10–§33.16, §33.19, §33.24)

**It builds with training, steadies training, and survives width and language.**
With learned positions the null over the plain baseline is +0.0035 ± 0.0011 at a
10k budget, +0.0078 ± 0.0009 at 20k and +0.0112 at 50k (4 seeds each). In a model
twice as wide it is +0.0055 ± 0.0039 at 20k (5 seeds, 4 positive), and there the
plain baseline's final BPB varies 5.7 times more across seeds than the null's
(spread 0.0237 against 0.0041, no run unstable): at width its clearest effect is
consistency. In Ukrainian it replicates at every seed, smaller (+0.0028 ± 0.0010
against the constant null, 4 of 4; first recorded +0.0023 ± 0.0010 — G-73; 0.3%
against English's 0.83%). Under rotary positions the gain is even
across token frequency (+0.015 to +0.020 nats in every bucket, 4/4); with learned
positions it sat in rare tokens. Rotary positions themselves beat learned ones by
+0.0443 ± 0.0011 BPB. At larger scale it is untested. (§33.30, §33.42–§33.46,
§33.49–§33.52)

## How to measure this — the methodological contribution

Each is an artifact this project found in its own numbers or pipeline; together they are the
four fairness conditions of `docs/reference/literature_audit.md`, which no audited Q2 paper
meets in full.

1. **Word-level features leak within the word.** A subword model given a
   whole-word feature at every fragment reads the word's ending before writing
   it: 86–97% of such gains sat on continuation fragments. Fix: shift the
   feature to the previous word, with boundaries from the tokenizer. (§30)
2. **Tagger and parser labels read the right context.** Shifted or not, 7–8% of
   part-of-speech tags depend on the words that follow. Fix: lexical variants
   (each word's majority value), and measure before trusting. (§33.3, §33.6)
3. **A feature changes the architecture as well as the information.** Its
   baseline is a null of the same shape; without it, architecture passes for
   information. (§33.10–§33.11)
4. **A shared seed matches initial weights only between identical shapes.**
   The baseline sits outside every feature family, and its sensitivity to its
   random start differs fourfold — a seed spread measured on one configuration
   is not another's. (§33.15–§33.16)
5. **Minimal-pair gains can be preference shifts.** Structure's anaphor
   "agreement" gain was a learned dislike of *itself*: split items by the
   direction of change before reading a gain as grammar. (§33.8)
6. **A word is what the text says it is.** A byte-level tokenizer can split
   a character across tokens; counting every fragment as part of a letter
   glued Ukrainian quotes and dashes into words; dropping paragraph breaks
   ran headings into the next paragraph's first word. The tests that caught
   them: words must be exactly the text's runs of letters, and must agree
   with the tagger's words. (§33.21–§33.22)
7. **Analysis code needs the same scrutiny as models.** The exported comparison
   table measured every effect against a 10k-step sweep baseline and compared
   means over different seed sets. (§33.6)
8. **A control must be checked for what it carries, not assumed empty.** The
   "empty" slot carried learned position whenever positions were learned; the
   "flat" slot marks the first word of every window, because the causal shift
   gives a word with no known predecessor a sentinel class; the word-form control
   for the tagger's leak marks every word unseen in train `none`, which in training
   marked a word without case, and so misleads its model on them. Each was found by
   asking what the control's input actually holds. (§33.51–§33.52, §33.74)
9. **Run identity is part of the measurement.** A label that ignored a setting
   would have merged different runs into one cell (model size, slot tiling); a
   run id that hashes the whole configuration splits one cell's seed in two once
   a field is added; a quarantine rule written for one generation of runs misfired
   on the next; tests left out of a hand-kept list never ran. Each was caught
   before it changed a number. (§33.39, §33.43–§33.45, §33.48)

## Withdrawn along the way

The 5.1% gain of leaky word-identity features (the leak); the 0.89% then
0.85% "word structure helps" headline (the architecture); "the headline is a
floor" (the slot's own growth over training); the anaphor BLiMP gain (a
preference); "inflection behaves like structure" (it fades like identity);
"position placed only in the slot keeps half its gain" (it did at two seeds,
not at four); "case gains beside a zero slot" (one seed, not three); the slot as "a clean
position channel" (it needs a token-independent signal, which learned position
supplied); "a critical period" (given time, a late feature is integrated); "the
architecture effect needs a high learning rate" (its one seed, reproduced bit for
bit, was seed variation).
Each withdrawal was decided by a rule recorded before its result.

## Open

The research programme, [`DIRECTIONS.md`](DIRECTIONS.md), pursues:
- the width-512 lead at 20k at seeds 1–3 — its size and spread (seed 0 levels off at +0.011);
- whether the tokenizer's gain holds on fiction beyond a 10k budget;

It leaves open, as limitations: whether reliance's deficit is permanent beyond a 20k
budget; and, across model sizes, the half of
the width difference that the previous word's tag does not account for (§33.83).
