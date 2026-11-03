# What We Know — 21 October 2026

> **Kind:** claims — what the thesis claims now, each with its numbers, firmness and the journey sections behind it.
> **Authoritative for:** the claims.
> Rewritten in place when a result changes one.

The claims this thesis can make, each with its evidence and how firmly it stands.
The narrative of how each was reached is in `docs/record/journey/README.md` (section numbers below); the numbers are in `docs/record/results.md`.
A 5M-parameter decoder-only transformer on WikiText-103 and on Ukrainian Wikipedia; gains are in bits per byte (BPB), paired by seed, mean ± standard error.
Each claim has a heading of its own; its last line, in italics, says how firmly it stands.

## The two questions

### Q1 in English: morphemes against BPE

**Q1 — Does segmenting English into morphemes beat BPE for language modelling? No.**
BPE wins by 13.5% at matched parameters (two seeds), 12.9% on the held-out test set, and Unigram by 12.6% — against the morpheme arm with an unsupervised fallback; the deterministic arm, reported in the article accepted for publication, loses by 13.15%, 12.0% and 12.29% (results §5.9, §5.19–§5.20).
Morphemes help only the rarest words (+9% below frequency 10), learn more slowly, and an ideal router between the two tokenizers would gain at most 0.6%.

*Firm; unaffected by anything below — no features are involved.* (Parts II–IV, §32.3)

### Q1 in English: morphology inside BPE

**What lost was replacing BPE, not morphology: kept inside BPE, English morphology helps the 5M model.**
The tokenizer is the Ukrainian result's (below): BPE 17k that never joins a stem to its inflectional ending, by a rule that reads the word alone.
English inflects far less (the rule cuts 10.2% of words, reaching half of the inflected ones), so the gain is a fifth of the Ukrainian one; it does not fade:
- 10k: **+0.0046 ± 0.0010 BPB (0.30%, 4/4)**, every run below every BPE 17k run;
- twice the budget: **+0.0055 ± 0.0008 (0.38%, 4/4)**;
- full training, 50k: **+0.0053 on the median seed (0.39%)**, at seven seeds of eight.

At 30M parameters (width 512, on a second machine) it is gone at 10k: **+0.0002 ± 0.0004 (2/4)** (§33.142).

After 20k steps two BPE 17k runs of eight drift by about 0.009 and two endings-apart runs by about 0.005, leaving one pair level; the mean (+0.0066 ± 0.0024) tells less.
The drift is chance within training, not the seed's: a run repeated on another machine, same initial weights and batches, drifts or not (§33.107).
**Grammar agrees at every seed:** BLiMP **+1.0 ± 0.2 points** over all 67 paradigms (8/8; +0.3 at 10k), morphology +1.1 ± 0.6 (6/8); better on average in each agreement phenomenon — subject–verb +1.1 ± 0.5, anaphor +3.8 ± 1.8 (six seeds of eight), determiner–noun +1.2 ± 1.1 (only four) (§33.117).

*Plain baselines; four seeds at 10k and 20k, eight at full training (seeds 4–7 on a second machine); BLiMP at eight seeds; width 512 at 10k on a second machine.* (§33.85, §33.87, §33.92, §33.102, §33.107, §33.142)

### Q1 in Ukrainian: morphology inside BPE

**Q1 in Ukrainian, with morphology inside BPE: yes — and the gain lasts, to full training.**
BPE that never joins a stem to its inflectional ending (cut by a dictionary rule from the word form alone) beats plain BPE of the same vocabulary size by **+0.0174 ± 0.0014 BPB (1.6%)** at a 10k budget, 4/4 from identical initial weights, though it reads 11% less text in the same steps.
Unigram, cutting words elsewhere and into even more tokens, leads early, loses the lead by step 3,000 and gains nothing by the end (−0.0005 ± 0.0004): where the cuts fall matters, not how many there are.
**Where it sits** (5M, full training; §33.156): **81% of the gain is in the words the rule cuts** (+0.0134 ± 0.0003 of +0.0166 on validation; 86% on test), almost all of it in words seen 1–9,999 times in training; the most frequent words and unseen ones give nothing; the uncut words a fifth (+0.0038), punctuation none. Over articles the headline's 95% interval is +0.0142 to +0.0188 on validation, +0.0104 to +0.0147 on test; the two samples differ by 2.5 standard errors.
**What it is worth** near 5M parameters (~38M tokens): plain BPE 17k needs **1.15 ± 0.01 times the parameters** (about 5.8M) or **1.17 ± 0.01 times the training steps** to match it, each read between two runs trained to their own end (§33.155).
Widely used tokenizers do not keep the ending apart: at the rule's cut XLM-R's puts a token boundary for 22.8% of the words, mBERT's 16.6%, GPT-4o's 25.0%, and Qwen2.5's 50.7% only by cutting a Ukrainian word into 3.5 tokens (§33.154).
A second length control agrees: plain BPE 8k, of nearly endings apart's token length (3.13 characters per token against 3.21), ends **0.0829 ± 0.0007** behind it (4/4); BPE 32k, with longer tokens and 1.9M more parameters, ends 0.0276 ahead — endings apart's gain is 39% of what going from 17k to 32k gains, with no parameters added (~38M, 10k; §33.152).
**It is the morpheme boundary:** BPE 17k trained with the same words cut as often, one letter off the rule's boundary (3.18 characters per token against endings apart's 3.21; 64.9% of words split against 64.4%), ends **0.0207 ± 0.0008 behind plain BPE 17k (−1.90%, 0/4)** and 0.0381 behind endings apart (§33.148).
At width 512 Unigram's null ends **0.0056 ± 0.0009** behind BPE 17k's (0/4; on a second machine, where width 128 gives −0.0014 ± 0.0003; §33.126).
Every tokenizer's validation stream decodes to the same text, byte for byte, so each one's bits per byte is a code length for that text (G-75).

**At widths 128 and 256 it does not fade as a feature does.**
At 20k it is +0.0169 ± 0.0007 (4/4); **at full training on four times the text (151M tokens, 50k) +0.0165 ± 0.0008 (1.75%, 4/4)**, its 95% interval +0.0141 to +0.0189 (every row of the tokenizer's table excludes zero, §33.151); on the held-out test articles **+0.0126 ± 0.0005 (1.3%, 4/4)**, interval +0.0109 to +0.0142 — every model scores worse there, endings apart's 0.0039 more (§33.152), flat from step 5,000 while the null falls from 1.13 to 0.944: about 0.017 at two corpus sizes and three budgets, wherever the null stands between 1.21 and 0.944 BPB.
At width 256 (four times the core's parameters) it is larger at every matched null BPB: **+0.0254 ± 0.0012 (2.6%, 4/4)** at 10k, settling at **+0.0218 ± 0.0029 (2.4%, 4/4)** at 20k, flat from step 7,500 (width 128 at 20k: +0.0169, 1.7%).

**At a better learning rate it is smaller.**
The default rate, 3·10⁻⁴, is not the best for the 5M model at full training: BPE 17k's null ends **0.0491 ± 0.0007 BPB (5.2%)** lower at 3·10⁻³ (4/4), 0.0381 lower at 1·10⁻³ (seed 0), on a second machine; 3·10⁻³ is the best of the three for both tokenizers.
There the gain is **+0.0136 ± 0.0011 (1.52%, 4/4)**, against +0.0172 ± 0.0013 (1.82%) at the default on the same machine: 0.0036 ± 0.0010 less, at every seed (§33.145).
The larger models ran at the default rate only.

**It peaks there.**
At width 512 (30M parameters, four times width 128) it is **+0.0125 ± 0.0009 (1.4%, 4/4)** at 10k, below both smaller widths.
At the same point of learning it starts above width 256's (+0.041 against +0.038 at null BPB 1.18), then falls as a feature's gain does — below width 256's from 1.11 on, at width 128's level near 0.98 (+0.018), +0.014 at 0.92 — while the smaller widths' leads hold flat.
At 20k it levels off from about step 7,500 at every seed, at **+0.0102 ± 0.0006 (1.25%, 4/4)**: smaller than at widths 128 and 256, but lasting (§33.94, §33.96, §33.100, §33.102).
At full training (151M tokens, 50k, on a second machine) it is **+0.0072 ± 0.0003 (0.97%, 4/4)**, level from step 10,000 (+0.0078; +0.0072 at 25,000, +0.0077 at 45,000): less than half the 5M model's, and lasting (§33.142); on the test articles +0.0062 ± 0.0004 (0.83%, 4/4; §33.152).

**Beyond 30M parameters it is smaller again at 10k and 20k, and at full training as large as at 30M.**
At width 1024 (85M parameters) at full training (151M tokens, 50k, the 1,000-step warm-up, a second machine) it is **+0.0071 ± 0.0004 (1.02%, 4/4)**, level from step 10,000 (+0.0069; +0.0073 at 25,000, +0.0076 at 45,000), against width 512's 0.97%: from 30M parameters on, the gain at full training does not fall (§33.149).
At width 1024 (85M parameters, on a second machine), with the 100-step warm-up of every other run, it is **+0.0067 ± 0.0030 (0.81%, 3/4)** at 10k and **+0.0075 ± 0.0018 (0.97%, 4/4)** at 20k, against width 512's +0.0141 ± 0.0016 and +0.0099 ± 0.0006 there, and level over the last 5,000 steps (+0.0002).
Two seeds keep width 512's level (+0.0115, +0.0093 at 20k). At the other two the endings-apart null falls behind in early training, from about step 1,500, at both budgets, and keeps a smaller gain (+0.0033, +0.0058 at 20k; G-81).
The fall is no spike: at 100-step resolution that null's validation BPB falls at every evaluation, and a rerun on the same machine repeats it to 3 × 10⁻⁵, so there it is the seed's (§33.122).
**It is the warm-up's.** With 1,000 steps of warm-up every seed keeps the gain, **+0.0110 ± 0.0011 (1.34%, 4/4)** at 10k, from +0.0095 to +0.0143; the two seeds that fell gain +0.0143 and +0.0100, their endings-apart nulls ending 0.0135 and 0.0067 lower than with 100 steps.
Width 512 under the same warm-up gains +0.0147 ± 0.0022, as with 100 steps (+0.0141), so at 85M parameters the gain is smaller than at 30M under either recipe (§33.130, §33.132); width 128 gains +0.0175 ± 0.0004 under it, as with 100 steps (+0.0175 ± 0.0015; §33.145).
At the same point of learning the lead starts above width 512's (at 10k +0.059 against +0.048 at null BPB 1.30) and from 1.11 down is 0.003–0.015 below it, at both budgets.
With the 100-step warm-up, by 20k the 85M model's BPE 17k null stops improving, after 4.3 passes over its ~38M-token text, so the last 1,500 steps add about 0.0025 to the gain (+0.0049 at step 18,500) (§33.120, §33.124).
With the 1,000-step warm-up at 20k the gain is +0.0105 ± 0.0004 (1.36%, 4/4) on that text, level over the last 5,000 steps, and **+0.0082 ± 0.0005 (1.09%, 4/4)** on the 151M-token corpus, read about 1.1 times: smaller at every seed, by 0.0022 ± 0.0002, and still rising (+0.0008 over the last 5,000 steps).
Repeating the text added to the gain: BPE 17k's null gains more from the larger corpus than the endings-apart null (+0.0170 against +0.0148) (§33.137, §33.141).

**Under a modern block it holds; in a deeper model it falls, as in a wider one.**
At width 128 and 10k the LLaMA-style block (rotary positions, RMSNorm, SwiGLU) improves BPE 17k's null by 0.0627 (5.8%) and keeps the gain at **+0.0152 ± 0.0004 (1.48%, 4/4)**, against +0.0175 ± 0.0015 under the standard block on the same machine.
At width 512 and 10k, 8 layers instead of 4 improve the null by 0.0263 (3.0%) and halve the gain: **+0.0079 ± 0.0014 (0.93%, 4/4)** against +0.0147 ± 0.0022 under the same warm-up (§33.149).

These gains are between the two tokenizers' nulls.
In the 5M model at full training the gain between plain baselines, with no slot at all, is **+0.0219 ± 0.0007 (2.31%, 4/4)**, against +0.0172 ± 0.0013 between the nulls on the same machine: the null helps BPE 17k (+0.0021 ± 0.0010) and costs endings apart (0.0026 ± 0.0009), so the figure between the nulls is the conservative one (§33.145).
Over the empty slots the gain is larger in the 5M model, the same at 30M parameters and smaller at 85M:
- the 5M model's null costs endings apart about 0.002 (below), so at 10k the gain between the empty slots is +0.0203 ± 0.0013, against +0.0175 ± 0.0015 between the nulls on a second machine;
- at width 512 the two references agree (+0.0142, +0.0141), and its gain is the smaller over either (§33.118);
- at width 1024 the null costs BPE 17k and helps endings apart (below), so at 10k with the 1,000-step warm-up the gain between the empty slots is **+0.0079 ± 0.0008 (0.96%, 4/4)**, against +0.0110 ± 0.0011 between the nulls: 0.0032 ± 0.0010 less, at every seed (§33.139).

**And it shows in grammar, not only in bits per byte.**
On MultiBLiMP's 2,744 Ukrainian subject–verb agreement pairs the endings-apart models prefer the grammatical sentence more often (4/4 each):
- full training: 76.3% against 71.4%, **+4.9 ± 0.4 points**, in number, gender and person; +5.0 on pairs of equal token length;
- ~38M, 10k: 69.0% against 64.8%, +4.3 ± 0.3; 20k: +4.5 ± 0.5;
- width 256: +5.3 ± 0.3; width 512: **+6.8 ± 0.2** (20k: +5.9 ± 0.2), where the lead in bits per byte is half width 256's;
- width 512 at full training (151M tokens, 50k, a second machine): 87.2% against 82.2%, **+5.0 ± 0.6** (§33.142);
- width 1024 (85M parameters, a second machine), 10k: **+5.3 ± 0.4**, from +4.6 to +6.3 at every seed, also at the two where the lead in bits per byte is nearly gone; width 512 on that machine +5.9 ± 0.3; at 20k +4.7 ± 0.3 against +5.8 ± 0.3 (two seeds); with a 1,000-step warm-up **+5.9 ± 0.4**, as much as width 512's +6.0 ± 0.5 under it; at 20k with that warm-up +5.4 ± 0.3 on the 151M-token corpus and +5.1 ± 0.1 on ~38M tokens (§33.141); at full training 89.4% against 86.2%, **+3.1 ± 0.2 (4/4)**, smaller than at 30M (+5.0) and 5M (+4.9) (§33.149).

**Beyond subject–verb agreement** (§33.147, §33.150).
The thesis's own 3,999 pairs come from the test split: an adjective agreeing with the noun after it in case, number or gender, and a noun after a preposition in a case the preposition governs.
A native speaker's check found 98 of 100 valid.
Both tokenizers are near the ceiling, and endings apart makes fewer errors at every seed:
- 5M at full training: 99.0% against 98.2%, +0.8 ± 0.1 points, **47% fewer errors**; agreement in the noun phrase **+0.6 ± 0.1**, government **+1.5 ± 0.3** (4/4 each); an adjective's number alone +0.2 ± 0.2 (2/4);
- 30M at full training: 99.8% against 99.4%, +0.4 ± 0.1, **63% fewer errors**; agreement +0.3 ± 0.0, government +0.6 ± 0.1 (4/4 each);
- the same without the check's two kinds of error (3,520 pairs), and on the pairs of equal token length.

MultiBLiMP's gain grows with the model up to 30M parameters, not beyond (on one machine 85M's is no larger than 30M's), and it is not the code length's.
At full training it is about the same at 5M and 30M parameters: +4.9 and +5.0 points, on two machines (§33.142).
Between machines it moves more than within one: width 512's twins gain 0.9 ± 0.4 points less than the thesis machine's models, at every seed (§33.122).
Exposing more endings elsewhere does not do it: BPE 4k is 1.2 points below BPE 17k, Unigram level with it.

**And on a second corpus.**
On UberText 2.0's fiction (novels and prose; both tokenizers trained on it) endings apart beats BPE 17k at every seed: **+0.0097 ± 0.0010 BPB (0.8%, 4/4)** at 10k, half its share on Wikipedia, and **+5.3 ± 0.6 points (4/4)** on MultiBLiMP, as much as on Wikipedia or more.
At full training on four times that text (154M tokens, 50k) it is **+0.0155 ± 0.0010 (1.45%, 4/4)**, flat from step 20,000 — near its share on Wikipedia, so the smaller share belonged to the 10k budget — and +2.3 ± 0.3 points (4/4) on MultiBLiMP, from 88.1% (§33.108, §33.109).

*Two corpora (fiction at 10k, and at full training on a second machine), two corpus sizes and three budgets on Wikipedia, four widths (512 at 20k: seeds 2–3 on a second machine, and at full training on it; 1024 on a second machine); two grammar tests, MultiBLiMP's subject–verb pairs and the thesis's own pairs inside the noun phrase and after a preposition.* (§33.61–§33.68, §33.76, §33.79, §33.86, §33.88, §33.94, §33.96, §33.100, §33.102, §33.108, §33.120, §33.122, §33.124, §33.132, §33.139, §33.141, §33.142, §33.150)

### Q2: no feature beats its null at the end of training

**Q2 — Do injected linguistic facts help? No — in either way of injecting them, and in either language.**
Against a **null of the same shape** — which here sees the feature cell's data and dropout at the same seed, and starts from its initial weights when the feature has the null's eight classes; part of speech (19 classes) and inflection (72) share data and dropout only (§33.15):

| feature | against its null | seeds |
|---|---|---|
| word structure, given its own input dims | +0.0004 ± 0.0005 | 4 |
| word structure, added to a full-width embedding | −0.0013 | 4 |
| structure shuffled among words of similar frequency | +0.0015 ± 0.0005 | 4 |
| real structure against its shuffled version | −0.0012 ± 0.0008 | 4 |
| frequency band of the word | −0.0005 ± 0.0011 | 4 |
| part of speech (tagger) | +0.0012 ± 0.0007 | 4 |
| part of speech (strictly causal) | −0.0001 ± 0.0014 | 4 |
| inflection (strictly causal) | **−0.0046 ± 0.0013** | 4 |
| **Ukrainian** case (tagger) | −0.0013 ± 0.0008 | 4 |
| **Ukrainian** case (strictly causal) | −0.0020 ± 0.0010 | 4 |

Identity-like features (inflection bundles, lemma and root hashes) do worse than the null.
BLiMP shows no grammatical gain from any feature, and MultiBLiMP none from Ukrainian case: within 0.4 points of its null at both sizes, under both tokenizers.
**At 30M parameters too, by the bound:** at full training (151M tokens, 50k, on a second machine) case (tagger) under BPE 17k ends **+0.0015 ± 0.0005** over its null — within 0.002, but positive at every seed and level from step 10,000 (+0.0017), where in the 5M model it ends below its null (§33.142).

*Four seeds for every row, none beyond 0.002 on average; at the one-sided 95% bound none beyond +0.0031 (part of speech, strictly causal; four rows pass 0.002), and four seeds detect a feature's effect with 80% power only from 0.002–0.006, row by row (§33.151).
The frequency band and strictly causal part of speech (read from the word form) have seeds 2–3 from a second machine, each against the null's twin there; per seed they range from −0.0037 to +0.0023 (§33.114).
The Ukrainian rows are against a null constant on every word; first recorded −0.0009 and −0.0015 (G-73, §33.56).* (§33.9–§33.19, §33.79, §33.142, §33.151)

### Before the model has learned it, features help

**Before the model has learned it, the answer turns: yes.**
At a 10k-step budget Ukrainian case beats its same-shape null at every seed:

| train tokens | ~1.5M | ~7.6M | ~38M | 151M |
|---|---|---|---|---|
| over the null | **+0.0076 ± 0.0016** | +0.0031 | +0.0032 | **+0.0026 ± 0.0004** |
| over the empty slot, larger (second machine, §33.113) | +0.0097 | +0.0040 | +0.0038 | |
| word form alone | +0.0050 | +0.0014 | about a third (six tokenizers) | |

On the same 151M tokens: +0.0055 ± 0.0006 at 5k steps, nothing at 20k (−0.0003 ± 0.0004) or 50k.
On less data the null is the better reference — an empty slot (its 8 dims held at zero) ends 0.0032 ± 0.0013 worse at ~1.5M, 0.0015 ± 0.0005 at ~7.6M (0/4 each) — so there these are gains over the best model without the feature.
Keeping two-fifths to two-thirds, the word form shows it is case, not only the tagger's right context; the shares are lower bounds, since it marks words unseen in train `none` and its model falls below the null on them (G-76, §33.74).
**And it is case's content:** with the word form's values reassigned among word types of similar frequency, the same grouping of words costs 0.0011 ± 0.0004 at ~38M and 10k (0/4), where the word form gains +0.0011 and case in context +0.0032 (§33.148).
**English agrees:** part of speech at 10k, **+0.0130 ± 0.0004** (4 seeds), five times Ukrainian case on 151M (+0.0026).

*Against a null constant on every word (G-72, G-73, §33.55–§33.58; re-runs reproduce them exactly); first recorded +0.0096, +0.0052, +0.0052, +0.0044 and +0.0015 at 20k, when words unseen in the null's train split were scored as non-words and 20 runs read another corpus size's lexicon.* (§33.31–§33.33, §33.38)

### What the sentence adds over the word form

**What reading the sentence adds over reading the word form is mostly not context — and at full training none of it is.**
At ~38M tokens and 10k, Ukrainian case from the full sentence leads case from the word form alone by +0.0021 to +0.0033 under all six tokenizers (4 seeds; §33.65, §33.81).

**A part is the tagger's right-context leak** (G-38): a preposition's case in context is the next noun's, read from the word being predicted.
Measured in trained models, with prepositions reading their form's case and every other word its case in context, it is **29% of the lead under BPE 17k (+0.0006 ± 0.0002) and 43% under endings apart (+0.0014 ± 0.0004)**, 4/4 each.
Of the rest (+0.0015 ± 0.0003, +0.0018 ± 0.0002) about half is words unseen in train, where the word form gives nothing and misleads its model below the null (−0.0006, −0.0007; G-76); +0.0002 to +0.0003 is context resolving another word's case; +0.0004 to +0.0005 is spread over targets both models see alike.

**At full training — 151M tokens, 50k steps — the whole lead is the leak, under both tokenizers.**
Under endings apart case leads its word form by +0.0014 ± 0.0002 (4/4), the leak is +0.0015 ± 0.0001 (4/4), and the control — prepositions read from their form — ends level with the word form (−0.0001 ± 0.0003).
On the law's axis, context without the leak falls as a feature's gain does (+0.0011 to +0.0013 over the word form at null BPB 1.25 to 1.05, +0.0002 at 0.94), while the leak, carrying the case of the noun being predicted, keeps +0.0012 to +0.0014 to the end.
Under BPE 17k, on a second machine, the lead is +0.0005 ± 0.0007 (+0.0006 ± 0.0004 on the thesis machine), the leak +0.0006 ± 0.0001 (4/4), and the control level with the word form (−0.0001 ± 0.0007; §33.114).
A context-aware feature helps before the model has learned to read the context itself; after that, it adds what the tagger saw of the word being predicted.
**Not because the tagger misreads the context** (UD Ukrainian-IU's gold labels, §33.153): its case is right for **94.2%** of the words with one; where the gold case is not the form's usual one (one word in five) for **81.1%**; and where it departs from the form's usual case it is right **87.0%** of the time, the form 8.3%.
The feature carries the context where context decides; the models at full training do not need it.

**And in English.**
Part of speech from a causal tagger (each word tagged from its left context alone) leads the word form by only +0.0006 ± 0.0004 at 10k and +0.0004 ± 0.0003 at 20k (3/4 each), against the tagged feature's +0.0031 and +0.0025; the leak is +0.0025 ± 0.0004 and +0.0021 ± 0.0005 (4/4 each), four fifths of the lead (§33.99, §33.107).
**At full training** (WikiText-103, 50k, a second machine) the causal tagger keeps nothing reliable over the word form: +0.0005 ± 0.0011 (2/4), its one steady part (+0.0003, 4/4) after words unseen in training. The lead in context, +0.0022 ± 0.0011 (3/4), is the leak by its mean, +0.0017 (77%), though the seeds split: at two the causal tagger matches the tagged feature, at two it falls behind the word form (§33.134, §33.135).

**Gender and number, at 10k.**
Read in context they lead their word forms by about +0.001 (+0.0008 to +0.0010, 4/4 each, under BPE 17k and endings apart; a third to a half of case's lead), and little of that is the tagger's leak: 30% and none for gender, 10% and 21% for number.
A causal tagger keeps +0.0007 to +0.0009 over the word form (4/4 each), and that rest is not context.
Split by the word the feature describes, 61–88% of it sits after words unseen in training (+0.0006 to +0.0007, 4/4 each), and there it is mostly the word form's loss — its model ends 0.00045 to 0.00055 below the null (0/4 each; G-76) — not the tagger's gain (+0.0001, 4/4 each).
After a seen word whose gender or number the sentence changes the two models differ by −4% to +6% of the rest; 10–34% is spread over targets both models see alike (§33.125, §33.127–§33.129).
**At full training** (151M tokens, 50k, endings apart) their lead over the word form is **+0.0002** (3/4 each) and the leak has gone (−0.0001 for both); the causal tagger's +0.0002 and +0.0003 is not context either: half sits after words unseen in training, 2–4% after seen words whose value the sentence changes.
Over the null gender still gains +0.0011 ± 0.0006 (3/4), from the word form as much as from the sentence; number gains nothing (−0.0005 ± 0.0005, 1/4) (§33.131, §33.133).

*The control at four seeds: under two tokenizers at ~38M and at full training; for English part of speech at 10k and 20k; for Ukrainian gender and number at 10k and at full training, and for English part of speech at full training, on a second machine.
A split of the lead by the word the feature describes had put 55–60% after prepositions — but two trained models compared target by target are not a model and its counterfactual (§33.74, §33.76, §33.81).
Gender's and number's split is such a comparison too; the part where both models see the same input says they differ in what they learned as well (§33.129).*

## What linguistic information does do: speed early learning

### Early help, in every informative feature

Every feature that informs the next word — part of speech, inflection, case, frequency band — beats its matched null early in training, at every seed, in both languages: at step 2,000, English part of speech **+0.0140 ± 0.0011** (4 seeds), Ukrainian case **+0.0056 ± 0.0005** (4 seeds), and every other informative feature and design likewise.
Every such advantage is gone by the end of training; structure, which says little about the next word, gives none.
The model uses the information, then learns it itself; where it has not yet learned it — little training, or data too scarce to learn it — the advantage survives to the end (above; §33.31–§33.33).
In training steps the early help is worth **1–9%**: part of speech 7–9%, inflection 3–5%, Ukrainian case 1–3% (§33.33, §33.91).

*Firm as a pattern (2–4 seeds per feature, 21 feature-null pairs, none an exception).
Ukrainian case against a null constant on every word; first recorded +0.0060 at step 2,000 (§33.58) and 2–4% of steps.* (§33.24, §33.28, §33.30)

### Removing a feature, and adding it late

**Removing English part of speech leaves a deficit that closes with long training; adding it late does not.**
It is a crutch, not a teacher: the model learns to use the feature, not better representations from it.
Switched off at step 2,500 of a 10k budget, it drops the model from 0.016 ahead of one never fed it to 0.029 behind, and the model ends **0.0023 ± 0.0006** below that one (4 seeds); switched off at 7,500, 0.0118 below.
Switched off at 5,000 of a 20k budget, 15,000 steps before the end, the model still ends **0.0039 ± 0.0009** below (5/5), the gap levelling off near 0.004 from step 12,500 while the learning rate is still a third of its peak.
At full training, 45,000 steps after the switch, the deficit halves: **0.0019 ± 0.0011** below (3/4; the fourth seed recovers by step 30,000).
Over twice that budget it closes: switched off at 5,000 of 100k, the model ends **0.0005 ± 0.0010** below (2/4), within its seed spread from about step 75,000 (in the same runs −0.0048 at 10,000, −0.0020 at 30,000, −0.0012 at 50,000; on a second machine, §33.126).
The reliance is temporary: the model leans on the feature while it is there and needs about 70,000 more steps to do without it.
**Nor is it the input's sudden change:** set at step 5,000 to one class (NOUN) instead of zeros, keeping a shared vector in the slot, the model ends **0.0038 ± 0.0009** below the one never fed it (5/5), within 0.0001 of the switch to zeros at every seed; the constant's jump at the switch is larger (+0.0264 ± 0.0012) and gone within 500 steps (§33.148).
**It is not lost knowledge of part of speech**: the switched model's deepest layer probes 0.5–1.3 points behind at the switch, within 0.3 points from 1,000 steps later to the end (5/5).
Nor of the previous word's tag, which the feature carried: rebuilt from context within 1,000 steps of the switch, it ends **+2.0 ± 0.1 points** better read at the deepest layer than in the model never fed it (5/5).
**The deficit is not general**: Ukrainian case, switched off at 5,000 of a 20k budget under endings apart, ends level with or ahead of the model never fed it (0.0008 ± 0.0006 ahead, 3/4), level again within 2,500 steps of the switch.
Switched *on* late, the feature is integrated slowly but well: with 15,000 steps left it recovers 54–104% of the kept gain (mean 73%).
Weaning — dropped at a rising rate, fully by step 5,000 — leaves 70% of the abrupt switch's deficit (−0.0029 ± 0.0016 against −0.0042 ± 0.0011, 4 seeds).

*Firm (4–5 seeds; the probes at five; Ukrainian case at one size and budget; 50k and 100k on a second machine).* (§33.34, §33.38, §33.40, §33.42, §33.50, §33.52, §33.102, §33.111, §33.126)

### Morphology as a target

**As a target, morphology teaches — and the knowledge stops paying.**
Predicted as an auxiliary target from the final hidden state (weight 0.1), English part of speech beats an information-free target — the same head on shuffled labels — by +0.0094 at step 1k, fading to **+0.0009 ± 0.0008** at the end of a 20k budget (5 seeds, 3 positive): faster than the same feature as an input (+0.0150 at 1k, +0.0075 at the end).
The teacher works: the real-target model encodes part of speech better at every layer, its input embeddings included — **+4.8 ± 0.1 points** at layer 2 by step 2,000, still +1.7 ± 0.1 at the end (5/5, probed).
Yet knowing more part of speech than language modelling itself acquires does not help predict text.
Any auxiliary head helps a little (the shuffled target +0.0006 ± 0.0004 over the plain baseline); the morphology in it does not, at the end.

*Firm (4–5 seeds; the probe at five).* (§33.50, §33.52, §33.111)

### No benefit in the rare tail

**No benefit hides in the rare tail.**
At the end of 50k runs, by token frequency, English part of speech keeps its gain only in the most frequent tokens (4/4 seeds in the top two buckets); English inflection is worse than its null in rare and mid-frequency tokens (0/4 positive), where its end deficit lives; Ukrainian case is neutral to negative across the tail.

*Firm (4 seeds); by BPE-token frequency, not word rarity.* (§33.47)

## How much a feature helps depends on how much the model has learned

### One curve per feature, within one model size

Plotted against the null's own validation BPB, a feature's gain traces one curve whatever the training length or budget.
English part of speech under 5k, 10k, 20k and 50k schedules deviates from one curve by at most 0.0010 (average 0.0002–0.0005); four other English features — inflection, part of speech from the word form, frequency band, structure — agree across 10k, 20k and 50k schedules within 0.0004 at null BPB 1.55 (4 seeds per schedule; 2 at 50k for word form and frequency band, measured before their seeds 2–3 ran, §33.114).
The curve predicts each feature's short-budget gain — at 10k part of speech +0.0130, part of speech from the word form +0.0099, inflection +0.0067, frequency band +0.0046, structure ≈ 0; at 20k +0.0075, +0.0050, +0.0022, +0.0014 and −0.0008 — and each feature has its own point where the gain reaches zero.

In Ukrainian five conditions (5k and 50k steps on 151M tokens; 10k on ~38M, ~7.6M and ~1.5M) agree within 0.0015 at null BPB 1.30, and closer as the null learns: within 0.0006 at 1.20, the three that reach 1.15 within 0.0001 (+0.0045 to +0.0046).
At 1.10 the 50k run, in its last steps, is 0.0008 below the ~38M run (+0.0029 against +0.0037).
At every point the conditions differ no more than their seeds do: dealt to the conditions at random, the seeds' values give as large a gap 24–100% of the time; each condition's seeds spread by 0.0004–0.0015 (sd).
Across six Ukrainian tokenizers case's gain agrees at null BPB 1.25 — all within 0.0006 of +0.0052 (4 seeds) — and parts later, where endings are exposed (§33.58, §33.65; below).

*Firm in English; in Ukrainian within the seed spread, against a null constant on every word — against the first null, which was not constant, the 50k run had sat 0.0015 below (G-73).* (§33.35, §33.38, §33.49, §33.52, §33.58, §33.91, §33.98)

### Across model sizes

**Within one model size only.**
At null BPB 1.55 a model twice as wide gains +0.0036 where the base model gains +0.0134 — its curve collapses abruptly between 1.70 and 1.60 — and at a 20k budget its part of speech ends at its null (−0.0008 ± 0.0005, 4 seeds); a model half as wide (64) gains less throughout.
The collapse is not the slot's share (tiled to the base share: +0.0024 and +0.0037 at 1.55, 2 seeds), nor linear knowledge of the current word's part of speech: the wider model's deeper layers probe ahead at every matched null BPB (+0.4 to +1.5 points, input layers alike), yet at the same probe accuracy the widths' gains still differ by up to 0.0068 ± 0.0005.
**What the feature carries — the previous word's part of speech — accounts for about half of the difference.**
The wider model reads it from context far earlier (+2.5 to +10.4 points at matched null BPB along the 20k budgets; each seed's largest lead 8.7 to 10.4), and on that axis the widths' gains differ by at most **0.0039 ± 0.0001** (0.0035 to 0.0043 per seed), against 0.0098 on the null's BPB; at the top of the shared range they nearly meet (+0.0075 and +0.0070 at accuracy 0.63, averaged over the five seeds).
Not fully, so the law is stated within one model size.
The feature stops paying well before the null knows the tag fully: the wider model's gain collapses when its deepest layer reads it at 0.68, at every seed; at the end of a 20k budget the base null reads it at 0.629 ± 0.004 (the current word's: 0.883 ± 0.001).

*The probes at five seeds per width, four of them on a second machine; the gains at four.* (§33.40, §33.42, §33.49–§33.50, §33.52, §33.60, §33.114)

## Morphology in the tokenizer and in the embedding (Ukrainian)

### Tokenizer and case add up

**They add up.**
Under BPE that keeps endings apart, case still beats its null at every seed (**+0.0048 ± 0.0006**, 4/4), and the two together end **+0.0222 ± 0.0017 BPB (2.0%)** below plain BPE 17k's null — the tokenizer contributing four fifths.
Case beats its null under each of six tokenizers (BPE at 4k, 8k, 17k and 32k, Unigram at 17k, endings apart; +0.0029 to +0.0054, every seed), and leads early under each (+0.0045 to +0.0090 at step 2,000).

*Four seeds under each tokenizer; ~38M tokens, 10k.* (§33.65, §33.81)

### Exposed endings: case gains less for a stretch

**Where the tokenizer exposes endings, adding case gains less for a stretch of training, in models of 5M and 30M parameters; late in training, only under Unigram in the 5M model.**
Case is compared with the best model without it — in the 5M model an empty slot where endings are exposed, the same-shape null under BPE (below).
At ~38M tokens and 10k it gains **+0.0022 ± 0.0007 under endings apart (4/4) and +0.0018 ± 0.0012 under Unigram 17k (3/4)**, against **+0.0030 to +0.0049 under BPE** at 4k, 8k, 17k and 32k (4/4 each; +0.0032 ± 0.0006 at 17k): about half as much where endings are exposed, as the hypothesis had it (case helps where the tokenizer hides endings).

**It is an early-training effect.**
By 20k the difference is gone: over the empty slot, the better reference there under both tokenizers, case gains +0.0005 ± 0.0004 under BPE 17k and +0.0003 ± 0.0006 under endings apart.
At full training it adds only the tagger's leak under endings apart, and nothing under BPE 17k (below).

**Read at the same point of learning, the redundancy is a middle stretch.**
On the law's axis (each tokenizer's gain at its reference's bits per byte along training), endings apart learns faster, its models further along at a given step.
Early (reference BPB 1.37–1.57) both exposing tokenizers gain more than BPE 17k; then endings apart gains less for a stretch (+0.0017 ± 0.0010 less at 1.27, 3/4), and late in the budget as much (+0.0002 ± 0.0011 less at 1.09).
Its fixed-budget shortfall is that head start: the law has a feature's gain fall while the model learns.
Unigram learns no faster than BPE 17k, and in the 5M model its shortfall stays at the same point of learning: +0.0020 ± 0.0007 (3/4) and +0.0016 ± 0.0003 (4/4) less at 1.09 and 1.19.

**A model of 30M parameters (width 512) shows the same order.**
Endings apart gains more early (by 0.0037 ± 0.0001 at 1.35), less in the middle (+0.0033 ± 0.0003 less at 1.18, 4/4), and as much from 1.09 on (+0.0001 ± 0.0004 less at 0.88).
A 10k budget takes that model past the stretch, so at a fixed budget too the two end level: **+0.0022 ± 0.0004 under BPE 17k and +0.0024 ± 0.0003 under endings apart** (4/4 each); there the null costs nothing (below), and over it too case gains alike, +0.0020 ± 0.0002 and +0.0022 ± 0.0003 (4/4 each).
Under Unigram too: at width 512 case gains **+0.0024 ± 0.0003** over the empty slot (4/4), and the 5M model's late shortfall is gone, −0.0004 ± 0.0003 at reference BPB 0.89 and −0.0001 ± 0.0002 at the end (§33.126).

*Four seeds per tokenizer, ~38M tokens; in the 5M model a 10k budget under six tokenizers and 20k under two, at width 512 a 10k budget under three; on a second machine, each comparison within it.* (§33.110, §33.113, §33.116–§33.118, §33.126)

### Over each tokenizer's own null

**Over each tokenizer's own null the order turns in the 5M model, and the difference grows with training.**
Case's slot does what the null's does (below), so over the null case's own values are compared, and they gain more where endings are exposed.
Paired by seed from identical initial weights, case gains more under Unigram 17k (43% of endings exposed) than under BPE 17k (9%) — **+0.0021 ± 0.0006 (4/4)** at the same null BPB — and under endings apart (all), +0.0016 ± 0.0010 (4/4), its null 0.017 further along.
On the law's axis, after null BPB 1.25 (above), the two that expose endings gain again — at 1.15, +0.0031 ± 0.0006 (Unigram) and +0.0043 ± 0.0013 (endings apart) over BPE 17k, 4/4 each — while the BPE curves fall.
Case from the word form, which reads no context and cannot leak, shows the same at about two thirds of the size (+0.0021 ± 0.0008 and +0.0033 ± 0.0017 at 1.15, 4/4 each).
Unigram's early lead is real: at null BPB 1.40, +0.0094 against BPE 17k's +0.0045 (+0.0049 ± 0.0009 paired, 4/4).
The information case carries is larger where endings are exposed; the slot carrying it costs there, so adding case buys less (above).

At the end of that budget about half the extra under endings apart is the tagger's leak: with prepositions read from their form it is +0.0008 ± 0.0012 (3/4), not +0.0016; at null BPB 1.15 mostly not (+0.0035 ± 0.0015 against +0.0043; §33.76).
At 20k case under BPE 17k fades to +0.0010 ± 0.0008 (3/4), while under endings apart it holds +0.0038 ± 0.0008 (4/4; paired +0.0029 ± 0.0016, 4/4).
At 151M tokens and 50k, where under BPE 17k case ends below its null (−0.0013 ± 0.0008), **under endings apart it still beats its null at every seed, +0.0030 ± 0.0009** (paired +0.0044 ± 0.0011, 4/4), +0.0033 at null BPB 0.94; the two together end +0.0195 ± 0.0013 below BPE 17k's null.
Half that gain does without the leak, which there can reach the noun's ending, a token of its own: case with prepositions read from their form keeps +0.0015 ± 0.0010 (3/4), the word form alone +0.0016 ± 0.0009 (3/4) — **+0.0036 ± 0.0010 (4/4)** more than the word form gains under BPE 17k, where it ends below its null.
The other half is the leak (above); over the empty slot, the better reference under endings apart, case keeps only about the leak (below).
In the 30M model the order does not turn (above).

*Four seeds each; the gains without the leak at three seeds of four; ~38M tokens at 10k and 20k.* (§33.58, §33.65, §33.68, §33.76, §33.81, §33.102, §33.110, §33.118)

### The same-shape null is not neutral

**In the 5M model the same-shape null is not neutral: its slot costs where endings are exposed and helps under BPE; what case carries sits on the endings.**
At ~38M tokens and 10k an empty slot (the null switched off from step 0: its 8 dims held at zero, from identical initial weights) ends better than the null where endings are exposed: **+0.0022 ± 0.0007** under endings apart (4/4), **+0.0039 ± 0.0016** under Unigram 17k (3/4).
Under BPE the null ends better: by **0.0037 ± 0.0014** at 4k, 0.0016 ± 0.0006 at 8k and 0.0025 ± 0.0006 at 32k (0/4 each), and at 17k on less data (above); at 17k on ~38M the two are level (−0.0006 ± 0.0004 at 10k, +0.0005 ± 0.0003 at 20k), and in English too.
Under endings apart the cost holds at every budget measured: **+0.0025 ± 0.0004** at 20k (+0.0034 ± 0.0009 on a second machine), +0.0020 ± 0.0014 at full training (3/4; one seed +0.0060).
Token by token, the cost sits at units' first tokens and non-words where endings are exposed (72% of it under endings apart, 79% under Unigram), where the slot tells whether the previous unit was a word; under BPE 4k and 8k the help sits on a word's later tokens, half and three fifths of it on the ending's token, under 32k three fifths at first tokens and non-words.
Case's slot says the same as the null's and, under endings apart, pays the same cost.

**What case carries is on the endings:** +0.0024 ± 0.0002 there (4/4), where the empty slot gains +0.0001.
Net, case beats the empty slot by half its gain over the null at 10k (+0.0022 against +0.0044; above), a third at 20k (+0.0013 ± 0.0007, 3/4, against +0.0038; +0.0003 on the second machine), and at full training by +0.0010 ± 0.0007 (3/4), about the tagger's leak: without the leak, and from the word form, it ends level with the empty slot (−0.0005 ± 0.0006, −0.0004 ± 0.0008), while its gain on the endings stays (+0.0021 ± 0.0001, 4/4).

**In a model of 30M parameters the null costs nothing.**
At width 512 and 10k the empty slot ends level with it under both tokenizers: −0.0002 ± 0.0001 under endings apart (1/4), −0.0002 ± 0.0005 under BPE 17k.
Along training the slot helps there, as early in the 5M model, and the help fades: where the 5M model's slot costs under endings apart (+0.0017 at null BPB 1.15, +0.0021 at 1.10), the 30M model's still helps (−0.0015, −0.0012; 0/4 each).
Under endings apart the cost belongs to the small model, not to a stage of learning.

**In a model of 85M parameters it is not neutral again, and its sign turns with the tokenizer.**
At width 1024 and 10k, with the 1,000-step warm-up, the empty slot ends better than the null under BPE 17k, **+0.0017 ± 0.0011** (3/4), and worse under endings apart, **−0.0015 ± 0.0006** (0/4).
Along training the slot helps under both early, as at the smaller widths; under BPE 17k the help turns into a cost from about null BPB 1.0 (+0.0033 ± 0.0020 at 0.95, 4/4), while under endings apart it still helps at the end, less than before (−0.0041 ± 0.0005 at 0.95).
In the tokenizer's gain the two add up: between the nulls it is 0.0032 ± 0.0010 larger than between the empty slots, at every seed ([above](#q1-in-ukrainian-morphology-inside-bpe)).

*Six tokenizers and three corpus sizes at 10k, on a second machine; endings apart at three budgets and BPE 17k at two, on both machines; widths 512 and 1024 under both at 10k, on the second machine.* (§33.106, §33.109, §33.110, §33.113, §33.116, §33.118, §33.139)

### Case helps by predicting endings

**Case helps by predicting endings; the exposure effect lies elsewhere.**
Re-scored token by token, case's gain sits on the tokens that hold an inflectional ending under every tokenizer: all of it under BPE 17k, where it helps only endings in a word's later token, not those fused into its first; about three fifths under Unigram and under endings apart, on a fifth of the targets.
The words the rule cuts gain about the same under all three (+0.0036 to +0.0041 at 10k): the exposing tokenizers' extra comes from words it does not cut (+0.0010 ± 0.0003 under endings apart, +0.0014 ± 0.0004 under Unigram), and at 20k under BPE 17k the feature costs everything but the endings (uncut words −0.0008, non-word targets −0.0007), while under endings apart it does not.
Under endings apart case read from the word form barely helps the endings (+0.0005, against +0.0028 from the sentence), the previous word's ending being in plain view; what the sentence adds there is largely the leak at prepositions: read from their form, case's gain on those tokens falls from +0.0028 to +0.0012 (§33.74, §33.76).

*Four seeds at 10k, three at 20k; over each tokenizer's own null.* (§33.69–§33.70, §33.74, §33.76)

### What the model builds for itself

**What the model builds for itself: the form's grammar, more accurately; context, no better.**
Probed linearly at each word's last token, the full-training nulls (no morphology in their input) hold a word's grammar about as well as its form's most frequent reading: case is right for 83.2% of words under BPE 17k and 84.7% under endings apart, against the form's 84.1% and 83.8%.
Keeping endings apart makes that account more accurate where the form decides (paired, 4/4 each):

- **number** +2.5 points;
- **gender** +5.7 points;
- **case where the form's majority case is right** +2.3 points.

These are the features subject–verb agreement turns on, where those models judge better.
Where only context decides a word's case, both tokenizers' models read it weakly, 10–13 points over that subset's floor, and endings apart is 2.5 points lower (0/4).
Read one word later the models hold less, not more: a model this size states a word's grammar at the word and does not revise it.

*Four seeds each, one corpus, linear probes, the tagger's labels.* (§33.84)

## What the slot did instead: a token-independent input

For the thesis this is a confound, and the reason for the same-shape null: what first looked like word structure helping was the feature slot itself, helping with no information in it (method 3, below; §33.10–§33.11).
How a shared vector helps a small transformer is a question about such models in general, parked as outside the thesis (§33.59); the measurements stand.

### One learned vector shared by every input

Under rotary positions — nothing positional in the input at all — a single learned vector, the same at every position, improves the plain baseline by **+0.0064 ± 0.0004 BPB** in 8 dims of its own (4 of 4 seeds) and **+0.0072 ± 0.0008** added across all 128 dims (3 of 3).
The same 8 dims held at exactly zero give nothing reliable (+0.0013 ± 0.0015); from identical initial weights the vector beats the empty slot by +0.0051 ± 0.0017 (4/4), and added it beats zeros added by +0.0060 ± 0.0016 (3/3).
Whether it also marks words is irrelevant (the word-marking constant slot: +0.0057 ± 0.0006).
Added to every input, the vector reparameterizes the token embedding — the model can reach nothing it could not reach without it — so the gain lies in how the model learns, not in what it can represent.
It is the vector, not a mark of the window's start: the causal shift gives each window's first word a sentinel class, which a rotary model cannot otherwise see, but without the shift — a pure constant, the same initial weights — the flat slot keeps 89% of its gain (+0.0068) and the vector added 82% (+0.0059), at seed 0 (G-71; §33.51–§33.52, §33.58).

*Firm under rotary positions.*

### With learned positions, a position channel

**With learned positions the same effect looked like a position channel.**
There 8 dims kept from token identity, holding the null's constant, improve on eight random starts of the plain baseline by **+0.0097 ± 0.0015 BPB (0.72%)**: every null run beats every baseline start, and training becomes four times less sensitive to its random start.
Held at zero they gain about as much, +0.0105 ± 0.0035 over the baseline (4 seeds), for such dims are never empty: learned position fills them — a token-independent signal.
Taking position out of the slot costs the zero slot −0.0082 ± 0.0012; out of the token dims instead, −0.0062 ± 0.0022.
One vector added to every input gains about as much (+0.0085 to +0.0090 over its initialization-matched control, 4 of 4) — and there the window start is visible to the model anyway.
The slot's width barely matters (8 to 64 dims).

*Firm.* (§33.10–§33.16, §33.19, §33.24)

### It builds with training, steadies training, and survives width and language

With learned positions the null over the plain baseline is +0.0035 ± 0.0011 at a 10k budget, +0.0078 ± 0.0009 at 20k and +0.0111 ± 0.0025 at 50k (4 seeds each).
In a model twice as wide it is +0.0055 ± 0.0039 at 20k (5 seeds, 4 positive), and there the plain baseline's final BPB varies 5.7 times more across seeds than the null's (spread 0.0237 against 0.0041, no run unstable): at width its clearest effect is consistency.
In Ukrainian it replicates at every seed, smaller: +0.0028 ± 0.0010 (4 of 4), 0.3% against English's 0.83%.
Under rotary positions the gain is even across token frequency (+0.015 to +0.020 nats in every bucket, 4/4); with learned positions it sat in rare tokens.
Rotary positions themselves beat learned ones by +0.0443 ± 0.0011 BPB.

*Untested at larger scale.
The Ukrainian figure is against the constant null; first recorded +0.0023 ± 0.0010 (G-73).* (§33.30, §33.42–§33.46, §33.49–§33.52)

## How to measure this — the methodological contribution

Each is an artifact this project found in its own numbers or pipeline; together they are the four fairness conditions of `docs/reference/literature_audit.md`, which no audited Q2 paper meets in full.

1. **Word-level features leak within the word.**
   A subword model given a whole-word feature at every fragment reads the word's ending before writing it: 86–97% of such gains sat on continuation fragments.
   Fix: shift the feature to the previous word, with boundaries from the tokenizer. (§30)
2. **Tagger and parser labels read the right context.**
   Shifted or not, 7–8% of part-of-speech tags depend on the words that follow.
   Fix: lexical variants (each word's majority value), and measure before trusting. (§33.3, §33.6)
3. **A feature changes the architecture as well as the information.**
   Its baseline is a null of the same shape; without it, architecture passes for information. (§33.10–§33.11)
4. **A shared seed matches initial weights only between identical shapes.**
   The baseline sits outside every feature family, and its sensitivity to its random start differs fourfold — a seed spread measured on one configuration is not another's. (§33.15–§33.16)
5. **Minimal-pair gains can be preference shifts.**
   Structure's anaphor "agreement" gain was a learned dislike of *itself*: split items by the direction of change before reading a gain as grammar. (§33.8)
6. **A word is what the text says it is.**
   A byte-level tokenizer can split a character across tokens; counting every fragment as part of a letter glued Ukrainian quotes and dashes into words; dropping paragraph breaks ran headings into the next paragraph's first word.
   The tests that caught them: words must be exactly the text's runs of letters, and must agree with the tagger's words. (§33.21–§33.22)
7. **Analysis code needs the same scrutiny as models.**
   The exported comparison table measured every effect against a 10k-step sweep baseline and compared means over different seed sets. (§33.6)
8. **A control must be checked for what it carries, not assumed empty.**
   The "empty" slot carried learned position whenever positions were learned; the "flat" slot marks each window's first word, to which the causal shift gives a sentinel class; the word-form control for the tagger's leak marks every word unseen in train `none`, in training the mark of a word without case, and so misleads its model on them.
   The check is what the control's input actually holds. (§33.51–§33.52, §33.74)
9. **Run identity is part of the measurement.**
   A label that ignored a setting would have merged different runs into one cell (model size, slot tiling); a run id that hashes the whole configuration splits one cell's seed in two once a field is added; a quarantine rule written for one generation of runs misfired on the next; tests left out of a hand-kept list never ran.
   Each was caught before it changed a number. (§33.39, §33.43–§33.45, §33.48)
10. **A same-shape null is not neutral, and its sign depends on the tokenizer and the model's size.**
   In the 5M Ukrainian model the null's slot loses to the same slot held at zero where the tokenizer exposes endings (by 0.0022 to 0.0039 at 10k, 0.0025 to 0.0034 at 20k) and beats it under BPE at 4k, 8k and 32k and on less data (by 0.0015 to 0.0037); a feature with the same slot pays the cost too; at 30M parameters the two end level under endings apart and BPE 17k; at 85M the slot costs under BPE 17k and helps under endings apart, so the tokenizer's gain between the nulls is 0.003 larger than between the empty slots ([above](#the-same-shape-null-is-not-neutral)).
   At full training in the 5M model, against plain baselines with no slot, it helps BPE 17k by 0.0021 and costs endings apart 0.0026, so the tokenizer's gain between the nulls is 0.0047 below its gain between plain baselines (§33.145).
   A gain over the null then overstates or understates the gain over the best model without the feature, the tokenizer's own gain included, and gains compared across tokenizers over their own nulls can turn a conclusion round: check against an empty slot, at each model size. (§33.106, §33.113, §33.116, §33.118, §33.139)
11. **Two GPUs run the same model alike, not the same run.**
   The same weights evaluate within 2 × 10⁻⁹ on both machines, and a matrix product, a softmax and a layer norm are bitwise identical; with dropout off, two machines train the same run to within 2 × 10⁻⁸.
   But dropout's masks are drawn on the GPU, differently by kernel layout, so a run repeated on the other machine differs like a dropout draw: −0.0001 ± 0.0001 over 143 pairs, no consistent sign (71 above, 72 below), the largest difference growing with training (0.0034 at 10k, 0.0044 at 20k, 0.0063 at 50k); at width 512 about as much as one seed's run from another's.
   An effect measured within each machine agrees between them (case over the null under eight conditions at 10k, within 0.0011; at 20k +0.0038 and +0.0037; the tokenizer's gain at width 512, +0.0125 ± 0.0009 and +0.0141 ± 0.0016); a single pair at full training does not.
   Grammar agrees less closely: the tokenizer's MultiBLiMP gain at width 512 is +6.8 ± 0.2 on the thesis machine and +5.9 ± 0.3 on the second, lower at every seed.
   Every comparison is made within one machine. (§33.106, §33.107, §33.109, §33.114, §33.118, §33.122)

## Withdrawn along the way

Each withdrawal was decided by a rule recorded before its result:
- the 5.1% gain of leaky word-identity features (the leak);
- the 0.89% then 0.85% "word structure helps" headline (the architecture);
- "the headline is a floor" (the slot's own growth over training);
- the anaphor BLiMP gain (a preference);
- "inflection behaves like structure" (it fades like identity);
- "little data, as such, is what makes case useful" (it is training progress; §33.32);
- "position placed only in the slot keeps half its gain" (it did at two seeds, not at four);
- "case gains beside a zero slot" (one seed, not three);
- the slot as "a clean position channel" (it needs a token-independent signal, which learned position supplied);
- "a critical period" (given time, a late feature is integrated);
- "the architecture effect needs a high learning rate" (its one seed, reproduced bit for bit, was seed variation).

## Open

Open questions that could change a claim are in [`GAPS.md`](GAPS.md); what runs next, and why, in [`DIRECTIONS.md`](DIRECTIONS.md), "The queue".

Left open, as limitations: across model sizes, the half of the width difference that the previous word's tag does not account for (§33.83); the learning rate beyond the 5M model, where every run used the default 3·10⁻⁴ (§33.145); the modern block and depth measured at 10k only (§33.149).
