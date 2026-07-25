# Migration — flat draft to `morphlm/` package

## Why the structure changed

Journey §20 chose a flat layout deliberately, and its stated reason was
*communication overhead in a collaborative context*: deep hierarchies are
expensive when files get pasted into a chat one at a time. That premise expired
— `pack.sh` made sharing one command, and the work ahead is a matrix across
branches rather than debugging one pipeline. This reversal should be recorded
in the journey as a premise change, not drift.

The concrete trigger was two bugs the flat structure was about to produce:

1. **Results overwrite on seed.** `results/{pipeline}_{variant}.json` had no
   seed in the identity, so a second seed silently clobbered the first — the
   exact run needed for effect-size confidence.
2. **Cache staleness on features.** The cache key was
   `(pipeline, split, max_articles)`. Once the stream carries annotation codes,
   that key cannot tell which annotation produced it: change the stamping rule,
   get a cache hit on stale data, train on silently wrong features.

---

## What changed

| area | before | after |
|---|---|---|
| config | defaults scattered; `max_steps` had 3 different values | one `ExperimentConfig` tree, one default per knob |
| LR schedule | cosine parameterized by `max_steps` | `lr_horizon` separate from `max_steps` |
| seeding | none anywhere | `train.seed`, seeds torch/numpy/random/dataloader |
| run identity | `{pipeline}_{variant}.json` | `results/<config-hash>/` with `config.json` |
| cache key | pipeline+split+cap | + feature spec; training knobs excluded |
| frozen dims | fixed 23-dim schema classes | composed blocks, T derived |
| context facts | impossible (static vocab rows only) | `dep` / `pos` / `ectx` per position |
| BPE freezing | impossible | possible via char-offset alignment |

### The frozen redesign in one paragraph

The old schema was `[category(6) | role(5) | entity_label(12)]` = 23 dims.
Measured against the real vocabulary: 12 columns constant (entity_label was
never populated from vocab rows), 5 redundant (role is a deterministic function
of category), **6 distinct frozen vectors across all 16,757 tokens**. So 23
dimensions of width bought one 6-way categorical, and a constant input dimension
is just a bias term. The old validation checked per-token properties (3 nonzero
bits per row — correct) and never checked the population, which is where it
failed. `tests/test_blocks.py` now asserts the population property and includes
a regression test that reproduces the old bug.

### The axis that is now the experiment

- `source='vocab'` — fact is a function of token id. A trained embedding row
  could already represent it, so freezing is **structured parameter tying**.
- `source='context'` — fact varies across occurrences of the same id. A static
  lookup table **provably cannot** represent it: genuine information injection.

"Static freezing does little; context freezing helps" is a real, measurable
claim — and it makes the thesis title (*Context-Aware Embeddings*) honest for
the first time.

### §28 revisited

§28 ruled "dataset emits IDs only" because embeddings carry gradients. That
reasoning does not extend to frozen metadata, which by definition has none. The
dataset now emits `(ids, codes)` — still a pure int-in/int-out machine, so the
principle holds while the §5 insight (*preprocessor scope ≠ unit scope*) is
restored.

---

## Run this first

```bash
python scripts/verify.py           # steps 1-6: imports, tests, freeze guarantee
python scripts/verify.py --spacy   # + context annotation check
```

Everything except the torch and spaCy paths was executed before shipping;
those two could not be run in the authoring environment, so they are the
likely failure points.

---

## Command mapping

| old | new |
|---|---|
| `python experiment.py run bpe baseline` | `python -m morphlm.cli run --pipeline bpe` |
| `python experiment.py run morph onehot` | *(retired — see the frozen redesign)* |
| `python experiment.py report` | `python -m morphlm.cli report` |
| `python main.py build` | `python -m morphlm.cli build all` |
| — | `python -m morphlm.cli cache --pipeline morph` |
| — | `python -m morphlm.cli inspect --pipeline morph --facts dep` |

Any knob: `-o section.field=value`, e.g. `-o train.seed=1`.

---

## Suggested next runs

**Leg 1 — segmentation.** Ready now, independent of the frozen redesign, and
builds the caches everything else needs. These also become the regression test:
if a later refactor reproduces these numbers at a fixed seed, it is proven
behavior-preserving.

```bash
python -m morphlm.cli run --pipeline bpe
python -m morphlm.cli run --pipeline morph
python -m morphlm.cli run --pipeline det
```

`lr_horizon` defaults to 50000 and is now independent of `max_steps`, so cells
of different length stay comparable.

**Leg 2 — injection.** After the caches exist:

```bash
python -m morphlm.cli cache --pipeline morph --facts dep    # measure spaCy cost
python -m morphlm.cli run --pipeline morph --facts dep
python -m morphlm.cli run --pipeline bpe   --facts dep      # newly possible
```

**Note on frozen fraction.** `dep` coarse is 8 dims = 6.2% of d_model 128. The
Lever-1 argument wanted a larger share; `-o frozen.amplify=3` gives 24 dims
(18.75%) with *all* of it informative — unlike the old 23 dims where 12 were
constant. Amplification adds salience, not information; the information ceiling
is set by the facts themselves.

---

## Known-unfinished

- `eval/probing.py` — extraction implemented, classifier deliberately not
  written. What to probe should follow from Leg 1 results, not precede them.
- `tie_output='frozen'` (option C) — still a hook that refuses.
- `etype` fact requires rebuilding the entity vocabulary with labels
  (`cli build entities`); the old CSV has only `entity,count`.
- The `stamp` fork (`all` vs `root`) is wired in `BasePreprocessor` but not yet
  exposed in config — it is a genuine comparison, flagged in journey §6.
