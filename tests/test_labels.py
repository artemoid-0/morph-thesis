"""
A run's label is its cell's identity: the tables group runs by label as seeds of one cell, and the
twin guards and a chain's resume find runs by it. Two runs that differ in anything but the seed must
therefore not share a label. This suite records how label() stands today; it changes nothing.

- Every field of every config section, on a BPE feature cell and on a Ukrainian Wikipedia cell,
  either changes the label when set to another value, or is listed in LABEL_NEUTRAL with the reason
  runs that differ in it are still one cell. A field label() reads only under a setting is listed in
  NEUTRAL_UNLESS, and must enter the label under that setting. A field added to the config fails
  here until it is given a place in the label or a reason.
- Over results/: one label never holds runs whose configurations differ in anything but the seed
  and the fields that change nothing a run computes (INERT), save the documented exception; each
  reason given in LABEL_NEUTRAL holds in the records; and today's label() gives every recorded run
  the label it was stored under.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataclasses
import glob
import json

from morphlm.config import ExperimentConfig, apply_overrides

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the cells the fields are walked on, as the chains give them
CELLS = {
    'English feature cell': ['data.pipeline=bpe', 'data.bpe_path=data/bpe_17k.json', 'frozen.facts=pos',
                             'frozen.trainable=true', 'frozen.causal_identity=true'],
    'Ukrainian Wikipedia cell': ['data.pipeline=bpe', 'data.corpus=ukwiki', 'data.corpus_holdout=2000',
                                 'data.spacy_model=uk_core_news_sm', 'data.corpus_fraction=0.05',
                                 'data.bpe_path=data/bpe_uk_17k.json', 'frozen.facts=ucase',
                                 'frozen.trainable=true', 'frozen.causal_identity=true',
                                 'train.max_steps=10000', 'train.lr_horizon=10000'],
}

# Fields label() leaves out, each with why runs that differ in it are still one cell. A reason that
# names a value is checked against the records below (FIXED, FIXED_PER).
LABEL_NEUTRAL = {
    'data.seq_len': 'the window length: 256 in every run',
    'data.max_articles_train': 'a cap for smoke tests: no recorded run has one',
    'data.max_articles_val': 'a cap for smoke tests: no recorded run has one',
    'data.cache_dir': 'where the streams are kept, not what they hold',
    'data.morpholex_path': "an input file's place (MorphoLEX, the English morpheme pipelines'): one in every run",
    'data.vocab_csv': "an input file's place (the English vocabulary): one in every run",
    'data.spacy_model': "the tagger of the corpus's language, set with the corpus, which is in the label",
    'data.corpus_holdout': 'fixed per corpus (Wikipedia 2000, fiction 500), and the corpus is in the label',
    'frozen.dep_granularity': '"coarse" in every run',
    'frozen.root_width': "the morpheme root factor's width: 16 in every run",
    'model.tie_output': '"none" in every run ("frozen" is a planned variant the model refuses)',
    'train.batch_size': '32 in every run',
    'train.lr_horizon': 'the chains tie it to max_steps (steps() sets both), and max_steps is in the label',
    'train.eval_every': '500 in every run',
    'train.patience': '6 in every run',
    'train.grad_clip': '1.0 in every run',
    'train.weight_decay': '0.01 in every run',
    'train.seed': "runs of one cell are its seeds: the label groups them",
    'train.eval_buckets': 'adds a per-frequency breakdown at the end; what the run trains and scores is the same',
}

# Fields label() reads only under a setting: neutral on a cell without it. (the setting, why)
_SURFACE = ('data.pipeline=morph', "the morpheme encoder's channels: a BPE stream is set by its tokenizer file")
NEUTRAL_UNLESS = {
    'data.corpus_fraction': ('data.corpus=ukwiki', 'a share of a non-default corpus; WikiText-103 is read '
                             'whole and refuses a fraction (stream.load_corpus)'),
    'frozen.aux_weight': ('frozen.role=target', "the auxiliary target's weight; a feature fed as input has none"),
    'surface.emit_punctuation': _SURFACE,
    'surface.emit_digits': _SURFACE,
    'surface.emit_inflection': _SURFACE,
    'surface.emit_whitespace': _SURFACE,
    'surface.emit_case': _SURFACE,
}

# Of the neutral fields, those that change nothing a run computes: they may differ within one label.
INERT = ('train.seed', 'data.cache_dir', 'train.eval_buckets')

# What the reasons above state about the records: one value in every run, or one per value of another field.
FIXED = ('data.seq_len', 'data.max_articles_train', 'data.max_articles_val', 'data.morpholex_path',
         'data.vocab_csv', 'frozen.dep_granularity', 'frozen.root_width', 'model.tie_output',
         'train.batch_size', 'train.eval_every', 'train.patience', 'train.grad_clip',
         'train.weight_decay')
FIXED_PER = {'data.corpus_holdout': 'data.corpus', 'data.spacy_model': 'data.corpus',
             'train.lr_horizon': 'train.max_steps'}

# A label whose recorded runs differ beyond that, documented: (fields, why they are still one cell)
DOCUMENTED = {
    'bpe_baseline_bpe_17k': (
        {'frozen.trainable', 'frozen.causal_identity'},
        'a baseline has no feature slot, so neither flag changes what it computes; seeds 0-3 ran '
        'without them, seeds 4-7 (scripts/chains/night_2026-10-16.sh) with them'),
}

# A run whose settings depart from what the reasons above state, documented: (fields, why). The
# record checks pass over these fields for this run alone; it must stay alone in its cell.
DOCUMENTED_RUNS = {
    'e50b75e8fdd2': (
        {'train.eval_every', 'train.patience', 'train.lr_horizon'},
        "the 5k look at G-81 (journey §33.121): the first 5,000 steps of a 10k run replayed with the "
        "10k run's schedule, evaluated every 100 steps and never stopped early; alone in its cell"),
}

# Values to probe a field with where one derived from its type would mean nothing to label()
PROBE = {
    'data.pipeline': ['morph', 'bpe'],
    'data.bpe_path': ['data/bpeend_17k.json'],
    'data.spacy_model': ['uk_core_news_sm', 'en_core_web_sm'],
    'data.corpus': ['ukwiki', 'ukfiction'],
    'frozen.facts': [('dep',), ('pos',)],
    'frozen.dep_granularity': ['fine', 'coarse'],
    'frozen.struct_keep': ['morphology', 'all'],
    'frozen.inject': ['add', 'concat'],
    'frozen.role': ['target', 'input'],
    'model.tie_output': ['frozen', 'none'],
    'model.position': ['rope', 'learned'],
    'model.norm': ['rmsnorm', 'layernorm'],
    'model.mlp': ['swiglu', 'gelu'],
}


def _fields(cfg):
    """Every field of every section, as ('section.field', value)."""
    for sec in dataclasses.fields(cfg):
        sub = getattr(cfg, sec.name)
        for f in dataclasses.fields(sub):
            yield f'{sec.name}.{f.name}', getattr(sub, f.name)


def _other(name, current):
    """A value other than `current` to set the field to."""
    for v in PROBE.get(name, []):
        if v != current:
            return v
    if isinstance(current, bool):
        return not current
    if current is None:
        return 5000
    if isinstance(current, int):
        return current * 2 + 1
    if isinstance(current, float):
        return current / 2 + 0.125
    if isinstance(current, str):
        return current + '.alt'
    if isinstance(current, tuple):
        return current + ('alt',)
    raise AssertionError(f'{name}: no value to probe it with; add one to PROBE')


def _with(cfg, name, value):
    sec, key = name.split('.')
    return dataclasses.replace(cfg, **{sec: dataclasses.replace(getattr(cfg, sec), **{key: value})})


def _holds(cfg, setting):
    return apply_overrides(cfg, [setting]).to_dict() == cfg.to_dict()


def test_every_field_enters_the_label_or_is_neutral():
    unplaced, stale = [], []
    for cell, overrides in CELLS.items():
        cfg = apply_overrides(ExperimentConfig(), overrides)
        base = cfg.label()
        for name, value in _fields(cfg):
            changes = _with(cfg, name, _other(name, value)).label() != base
            neutral = name in LABEL_NEUTRAL or (name in NEUTRAL_UNLESS and not _holds(cfg, NEUTRAL_UNLESS[name][0]))
            if not changes and not neutral:
                unplaced.append(f'{name} on the {cell}')
            if changes and neutral:
                stale.append(f'{name} on the {cell}')
    assert not unplaced, ('these fields leave the label unchanged and have no reason in LABEL_NEUTRAL: '
                          f'{unplaced} — put each in label() or give it a reason')
    assert not stale, f'LABEL_NEUTRAL lists fields that label() now reads: {stale} — take them out'
    listed = set(LABEL_NEUTRAL) | set(NEUTRAL_UNLESS) | set(PROBE)
    known = {name for name, _ in _fields(ExperimentConfig())}
    assert listed <= known, f'tables name fields the config no longer has: {sorted(listed - known)}'
    assert set(INERT) <= set(LABEL_NEUTRAL), sorted(set(INERT) - set(LABEL_NEUTRAL))
    n = len(known)
    print(f'  {n} fields: {n - len(LABEL_NEUTRAL) - len(NEUTRAL_UNLESS)} in the label, '
          f'{len(NEUTRAL_UNLESS)} under a setting, {len(LABEL_NEUTRAL)} neutral with a reason OK')


def test_a_field_neutral_unless_a_setting_enters_the_label_under_it():
    for cell, overrides in CELLS.items():
        for name, (setting, _) in NEUTRAL_UNLESS.items():
            cfg = apply_overrides(apply_overrides(ExperimentConfig(), overrides), [setting])
            value = dict(_fields(cfg))[name]
            assert _with(cfg, name, _other(name, value)).label() != cfg.label(), \
                f'{name} stays out of the label even with {setting} on the {cell}'
    print('  each field read under a setting enters the label under it OK')


def _recorded(raw):
    """A run's config.json as a config: today's from_dict where the config has one; otherwise the
    same rule — a field added since takes its default, and a record without a surface section
    predates it (the legacy stream, every channel off)."""
    if hasattr(ExperimentConfig, 'from_dict'):
        return ExperimentConfig.from_dict(raw)
    from morphlm.config import DataConfig, SurfaceConfig, FrozenConfig, ModelConfig, TrainConfig
    surface = raw.get('surface', {f.name: False for f in dataclasses.fields(SurfaceConfig)})
    frozen = dict(raw.get('frozen') or {})
    if 'facts' in frozen:
        frozen['facts'] = tuple(frozen['facts'])
    return ExperimentConfig(data=DataConfig(**raw.get('data', {})), surface=SurfaceConfig(**surface),
                            frozen=FrozenConfig(**frozen), model=ModelConfig(**raw.get('model', {})),
                            train=TrainConfig(**raw.get('train', {})))


def _records():
    """(run id, stored label, config) of every run in results/."""
    out = []
    for rp in sorted(glob.glob(os.path.join(ROOT, 'results', '*', 'result.json'))):
        cp = os.path.join(os.path.dirname(rp), 'config.json')
        if not os.path.exists(cp):
            continue
        out.append((os.path.basename(os.path.dirname(rp)), json.load(open(rp))['label'],
                    _recorded(json.load(open(cp)))))
    return out


def test_one_label_one_configuration_in_results():
    runs = _records()
    if not runs:
        print('  no runs in results/ — skipped')
        return
    by = {}
    for rid, label, cfg in runs:
        by.setdefault(label, []).append((rid, dict(_fields(cfg))))
    mixed = []
    for label, members in sorted(by.items()):
        allowed = set(INERT) | DOCUMENTED.get(label, (set(), ''))[0]
        first = members[0][1]
        for rid, f in members[1:]:
            differ = sorted(k for k in f if f[k] != first[k] and k not in allowed)
            if differ:
                mixed.append(f'{label}: {members[0][0]} and {rid} differ in {differ}')
    assert not mixed, 'runs under one label differ beyond the seed:\n  ' + '\n  '.join(mixed)
    for label, (fields, _) in DOCUMENTED.items():
        assert label in by, f'the documented exception {label} is no longer in results/'
        assert any(len({f[k] for _, f in by[label]}) > 1 for k in fields), \
            f'{label} no longer mixes {sorted(fields)}: drop it from DOCUMENTED'
    print(f'  {len(runs)} runs, {len(by)} labels: none mixes configurations '
          f'(documented: {", ".join(DOCUMENTED)}) OK')


def test_the_neutral_reasons_hold_in_results():
    runs = _records()
    if not runs:
        print('  no runs in results/ — skipped')
        return
    rows = [dict(_fields(cfg)) for _, _, cfg in runs]
    ids = [rid for rid, _, _ in runs]
    labels = [label for _, label, _ in runs]
    for rid, (fields, _) in DOCUMENTED_RUNS.items():
        assert rid in ids, f'the documented run {rid} is no longer in results/: drop it from DOCUMENTED_RUNS'
        assert labels.count(labels[ids.index(rid)]) == 1, f'the documented run {rid} has company in its cell'
    skip = lambda rid, name: name in DOCUMENTED_RUNS.get(rid, (set(), ''))[0]
    for name in FIXED:
        assert name in LABEL_NEUTRAL, name
        seen = {repr(r[name]) for rid, r in zip(ids, rows) if not skip(rid, name)}
        assert len(seen) == 1, f'{name} takes {sorted(seen)} across the runs; its reason says one value'
    for name, key in FIXED_PER.items():
        assert name in LABEL_NEUTRAL, name
        per = {}
        for rid, r in zip(ids, rows):
            if not skip(rid, name):
                per.setdefault(repr(r[key]), set()).add(repr(r[name]))
        many = {k: sorted(v) for k, v in per.items() if len(v) > 1}
        assert not many, f'{name} is not one value per {key}: {many}'
    assert all(r['train.lr_horizon'] == r['train.max_steps'] for rid, r in zip(ids, rows)
               if not skip(rid, 'train.lr_horizon')), \
        'a run with lr_horizon unlike its max_steps: its label hides its schedule'
    # max_steps enters the label in thousands (`_st10k`): a budget off the thousand would share
    # the round budget's label
    assert all(r['train.max_steps'] % 1000 == 0 for r in rows), 'a run whose max_steps is not whole thousands'
    print(f'  every reason in LABEL_NEUTRAL holds over {len(rows)} runs; every budget is whole thousands OK')


def test_todays_label_gives_every_recorded_run_its_label():
    runs = _records()
    if not runs:
        print('  no runs in results/ — skipped')
        return
    wrong = [f'{rid}: stored {label}, today {cfg.label()}' for rid, label, cfg in runs if cfg.label() != label]
    assert not wrong, 'label() no longer gives these runs their labels:\n  ' + '\n  '.join(wrong)
    print(f'  label() reproduces all {len(runs)} stored labels OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_labels: PASS')
