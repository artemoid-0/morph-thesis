"""Config invariants: one default per knob, and the lr_horizon guard."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphlm.config import ExperimentConfig, apply_overrides
import dataclasses


def test_lr_horizon_guard():
    cfg = ExperimentConfig()
    cfg.validate()
    bad = apply_overrides(cfg, ['train.max_steps=60000'])
    try:
        bad.validate()
    except ValueError as e:
        assert 'lr_horizon' in str(e)
        print('  lr_horizon < max_steps is refused OK')
        return
    raise AssertionError('expected lr_horizon guard to fire')


def test_run_id_includes_seed():
    a = ExperimentConfig()
    b = apply_overrides(a, ['train.seed=1'])
    assert a.run_id() != b.run_id(), 'seed must change run identity'
    print('  seed changes run_id (no silent overwrite) OK')


def test_overrides():
    cfg = ExperimentConfig()
    cfg = apply_overrides(cfg, ['data.pipeline=morph', 'train.seed=3',
                                'train.lr=0.001', 'frozen.facts=cat,dep'])
    assert cfg.data.pipeline == 'morph'
    assert cfg.train.seed == 3
    assert abs(cfg.train.lr - 0.001) < 1e-12
    assert cfg.frozen.facts == ('cat', 'dep')
    print('  dotted overrides coerce types OK')


def test_bpe_rejects_static_facts():
    cfg = apply_overrides(ExperimentConfig(),
                          ['data.pipeline=bpe', 'frozen.facts=cat'])
    try:
        cfg.validate()
    except ValueError as e:
        assert 'BPE' in str(e)
        print('  BPE + static facts refused OK')
        return
    raise AssertionError('expected BPE static-fact guard to fire')


def test_bpe_allows_context_facts():
    cfg = apply_overrides(ExperimentConfig(),
                          ['data.pipeline=bpe', 'frozen.facts=dep'])
    cfg.validate()
    print('  BPE + context facts allowed OK (the 2x3 matrix)')


def test_provenance_functions_all_import_and_run():
    """
    Regression: `frozen_detail` used a relative import one level too high
    (`..features` from inside `morphlm/`), raising ImportError. In the training
    loop that would have crashed AFTER 50,000 steps, at save time; in the
    backfill script a bare `except: pass` hid it for all 25 runs.

    Exercises every provenance function that does not need torch.
    """
    from morphlm.provenance import (environment, frozen_detail, convergence,
                                    data_stats)
    env = environment()
    assert 'python' in env and 'timestamp_utc' in env

    for facts in ('', 'mroot', 'mroot,mstruct,mrole'):
        ov = ['data.pipeline=bpe'] + ([f'frozen.facts={facts}'] if facts else [])
        cfg = apply_overrides(ExperimentConfig(), ov)
        blocks = frozen_detail(cfg)
        assert isinstance(blocks, list)
        if facts:
            assert len(blocks) == len(facts.split(',')), \
                f'{facts}: expected {len(facts.split(","))} blocks'
            for b in blocks:
                assert {'name', 'source', 'n_classes', 'dim',
                        'is_projection'} <= set(b)

    log = [{'step': s_, 'mean_loss': 4.5 - s_ / 50000,
            'bits_per_byte': 1.5 - s_ / 100000} for s_ in range(0, 50000, 500)]
    c = convergence(log, 50000)
    assert 'best_step' in c and 'steps_to' in c
    assert data_stats(None, None) == {'train': None, 'validation': None}
    print('  provenance functions all import and run OK')


def test_environment_names_the_tagger_used():
    """
    G-74: until 7 October `environment()` named en_core_web_sm whenever it was
    installed, so every Ukrainian run and audit claimed the English tagger and
    the Ukrainian tagger's version went unrecorded. The tagger is now the one the
    caller names, with its version; with none named, nothing is claimed about it.
    The export reads an old record's tagger from the run's config.
    """
    import os, importlib.util
    from morphlm.provenance import environment, SPACY_MODELS
    try:
        import spacy
        have = [m for m in SPACY_MODELS if spacy.util.is_package(m)]
    except Exception:
        have = []
    plain = environment()
    assert 'spacy_model' not in plain and 'spacy_model_version' not in plain, plain
    for m in have:
        e = environment(m)
        assert e['spacy_model'] == m and e['spacy_model_version'], e
        assert m in (e['spacy_models_installed'] or ''), e
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'export_results', os.path.join(here, '..', 'scripts', 'analysis', 'export_results.py'))
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    old = {'environment': {'spacy_model': 'en_core_web_sm', 'spacy_model_version': '3.8.0'},
           '_config': {'data': {'spacy_model': 'uk_core_news_sm'}}}
    assert ex.run_environment(old)['spacy_model'] == 'uk_core_news_sm'
    assert 'not recorded' in ex.run_environment(old)['spacy_model_version']
    english = {'environment': {'spacy_model': 'en_core_web_sm', 'spacy_model_version': '3.8.0'},
               '_config': {'data': {'spacy_model': 'en_core_web_sm'}}}
    assert ex.run_environment(english)['spacy_model_version'] == '3.8.0'
    new = {'environment': {'spacy_model': 'uk_core_news_sm', 'spacy_model_version': '3.8.0',
                           'spacy_models_installed': 'uk_core_news_sm 3.8.0'},
           '_config': {'data': {'spacy_model': 'uk_core_news_sm'}}}
    assert ex.run_environment(new) == new['environment']
    print(f'  environment names the tagger used ({", ".join(have) or "no tagger installed here"}) OK')


def test_tokenizer_table_pairs_by_seed():
    """
    D7's table (scripts/analysis/tokenizer_table.py, journey §33.65) must pair
    every comparison by seed — a feature with its own null, a tokenizer with the
    reference — and read gains at matched null BPB from the curves. Fake records
    with known answers: under tokenizer B the null is 0.010 lower at every seed
    and case gains 0.002 more than under A.
    """
    import os, json, glob, tempfile, subprocess, sys
    here = os.path.dirname(os.path.abspath(__file__))
    tool = os.path.join(here, '..', 'scripts', 'analysis', 'tokenizer_table.py')
    with tempfile.TemporaryDirectory() as d:
        n = 0
        for tok, null_end, case_gain in (('tokA', 1.100, 0.003), ('tokB', 1.090, 0.005)):
            for seed in range(3):
                for fact, end in (('mconst', null_end + 0.001 * seed),
                                  ('ucase', null_end + 0.001 * seed - case_gain),
                                  ('ucaselex', null_end + 0.001 * seed - 0.001)):
                    # a straight-line curve from 1.6 down to the end value
                    log = [{'step': s_, 'bits_per_byte': 1.6 + (end - 1.6) * s_ / 10000}
                           for s_ in range(0, 10001, 500)]
                    n += 1
                    os.makedirs(os.path.join(d, f'run{n}'))
                    json.dump({'label': f'bpe_{fact}-LEARN-CAUSAL_{tok}_ukwiki0.05_st10k', 'seed': seed,
                               'bits_per_byte': end, 'best_step': 10000, 'log': log},
                              open(os.path.join(d, f'run{n}', 'result.json'), 'w'))
        out = subprocess.run([sys.executable, tool, '--roots', d, '--tokenizers', 'tokA', 'tokB',
                              '--reference', 'tokA', '--audit-dir', d, '--seed-levels', '1.3'],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-2000:]
        rec = json.load(open(glob.glob(os.path.join(d, 'tokenizer_table_*.json'))[0]))['results']
    a, b = rec['tokenizers']['tokA'], rec['tokenizers']['tokB']
    assert abs(a['gain']['ucase']['mean'] - 0.003) < 1e-9 and a['gain']['ucase']['positive'] == 3
    assert abs(b['gain']['ucase']['mean'] - 0.005) < 1e-9
    assert abs(b['first_over_second']['mean'] - 0.004) < 1e-9
    vs = rec['against_reference']['tokB']
    assert abs(vs['null']['mean'] - 0.010) < 1e-9 and vs['null']['positive'] == 3
    assert abs(vs['ucase']['mean'] - 0.002) < 1e-9
    # on straight lines the gain at null BPB 1.3 is the end gain scaled by how far the null has come
    lv = rec['matched_null_bpb']['ucase']['tokA']['levels']['1.3']
    assert lv is not None and 0 < lv < 0.003, lv
    print('  tokenizer table: gains paired by seed, tokenizers against the reference, matched levels OK')


def test_probe_separates_inputs_from_labels():
    """
    Regression: the probe fed the model the codes of the PROBED fact as its
    input. Every published probe asked about the fact the model was trained on,
    so it never mattered. The first probe of a different fact would have made a
    single-fact model read part-of-speech tags as lemma codes, silently.
    """
    import importlib.util
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        'probe_dep', os.path.join(here, 'scripts', 'eval', 'probe_dep.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for model_facts, probe in ([], 'pos'), (['mlemma'], 'pos'), \
            (['mlemma', 'mstruct'], 'pos'), (['dep'], 'dep'):
        request, model_rows, label_row = m.plan_codes(model_facts, probe)
        assert [request[i] for i in model_rows] == model_facts, \
            f'{model_facts}/{probe}: model would not read its own facts'
        assert request[label_row] == probe, f'{model_facts}/{probe}: bad label'
    print('  probe separates model inputs from labels OK')


def test_export_pairs_each_cell_with_its_own_baseline():
    """
    comparisons.md once measured every cell against whichever feature-free cell
    came last in its group (a 10k-step sweep run), and compared MEANS over
    different seed sets. Both are encoded here as the failures they were.
    """
    import os, importlib.util, tempfile
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'export_results', os.path.join(here, '..', 'scripts', 'analysis', 'export_results.py'))
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    assert ex.own_baseline('bpe_mstruct-LEARN-CAUSAL_bpe_17k') == 'bpe_baseline_bpe_17k'
    assert ex.own_baseline('bpe_mstruct_bpe_17k_st10k') == 'bpe_baseline_bpe_17k_st10k'
    assert ex.own_baseline('det_dep_Spdiwc') == 'det_baseline_Spdiwc'
    assert ex.own_baseline('bpe_dep') == 'bpe_baseline'
    runs = [
        # the real baseline: seed 2 is unlucky by 0.01
        dict(label='bpe_baseline_bpe_17k', facts=[], seed=0, bits_per_byte=1.34),
        dict(label='bpe_baseline_bpe_17k', facts=[], seed=1, bits_per_byte=1.34),
        dict(label='bpe_baseline_bpe_17k', facts=[], seed=2, bits_per_byte=1.35),
        # a sweep baseline in the SAME group, listed last: must never be used
        dict(label='bpe_baseline_bpe_17k_lr0.001_st10k', facts=[], seed=0, bits_per_byte=1.43),
        # a cell at seeds 0 and 1 only, gaining exactly 0.002 at each
        dict(label='bpe_x-LEARN_bpe_17k', facts=['x'], seed=0, bits_per_byte=1.338),
        dict(label='bpe_x-LEARN_bpe_17k', facts=['x'], seed=1, bits_per_byte=1.338),
    ]
    for r in runs:
        r['_config'] = {'data': {'pipeline': 'bpe', 'bpe_path': 'data/bpe_17k.json'}}
    with tempfile.TemporaryDirectory() as d:
        ex.write_comparisons(runs, d)
        text = open(os.path.join(d, 'comparisons.md')).read()
    row = [l for l in text.splitlines() if l.startswith('| bpe_x-LEARN_bpe_17k ')][0]
    assert 'baseline `bpe_baseline_bpe_17k`' in text, 'must use the cell\'s own baseline'
    # paired: +0.0020 at both seeds; the unpaired mean difference would be +0.0053
    assert '+0.0020 +/- 0.0000' in row, row
    assert '2/2' in row, row


def test_additive_injection_is_labelled_and_guarded():
    """-ADD in the label, default concat untouched, nonsensical combinations refused."""
    from morphlm.config import ExperimentConfig, apply_overrides
    base = ['data.pipeline=bpe', 'data.bpe_path=data/bpe_17k.json', 'frozen.trainable=true',
            'frozen.causal_identity=true', 'frozen.facts=mstruct']
    assert apply_overrides(ExperimentConfig(), base).label() == 'bpe_mstruct-LEARN-CAUSAL_bpe_17k'
    assert apply_overrides(ExperimentConfig(), base + ['frozen.inject=add']).label() == \
        'bpe_mstruct-LEARN-CAUSAL-ADD_bpe_17k'
    assert apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.bpe_path=data/bpe_17k.json',
                                                 'frozen.inject=add']).label() == 'bpe_baseline_bpe_17k', \
        'a baseline has no facts to add; its label must not change'
    assert apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.bpe_path=data/bpe_17k.json',
        'frozen.trainable=true', 'frozen.causal_identity=true', 'frozen.facts=mzero',
        'frozen.inject=add']).label() == 'bpe_mzero-LEARN-CAUSAL-ADD_bpe_17k'



def test_model_size_in_label():
    """A non-default model size marks the label; the default adds nothing (§33.39)."""
    from morphlm.config import ExperimentConfig, apply_overrides
    base = ['data.pipeline=bpe', 'frozen.facts=pos']
    plain = apply_overrides(ExperimentConfig(), base).label()
    assert '_w' not in plain and '_L' not in plain, plain
    for over, mark in (('model.d_model=256', '_w256'), ('model.n_layers=6', '_L6'), ('model.n_heads=8', '_h8'),
                       ('frozen.amplify=2', '-x2')):
        lab = apply_overrides(ExperimentConfig(), base + [over]).label()
        assert lab.endswith(mark) and lab != plain, (over, lab)
    print('  model size in label OK')



def test_quarantine_spares_plain_baselines():
    """A plain baseline run with the causal flags is not a causal-shift casualty (§33.44)."""
    import importlib.util, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'ops', 'quarantine.py')
    spec = importlib.util.spec_from_file_location('quarantine', path)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    rule = {name: (affected, fixed) for name, _, affected, fixed in q.RULES}['causal_partial']
    baseline = {'frozen': {'facts': [], 'causal_identity': True}}
    feature = {'frozen': {'facts': ['pos'], 'causal_identity': True}}
    assert not rule[0](baseline), 'a baseline has nothing to shift'
    assert rule[0](feature), 'a causal feature run is in scope'
    assert rule[1]({'frozen_blocks': [{'causal_identity': True, 'causal_semantics': 3}]}), 'v3 is fixed'
    assert not rule[1]({'frozen_blocks': [{'causal_identity': True, 'causal_semantics': 2}]}), 'v2 is not'
    print('  quarantine spares plain baselines OK')



def test_run_ids_done_is_cell_and_seed():
    """run_ids.py --done: finished = same label and seed in results/ (journey §33.51)."""
    import importlib.util, json, os, tempfile
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'ops', 'run_ids.py')
    spec = importlib.util.spec_from_file_location('run_ids', path)
    ri = importlib.util.module_from_spec(spec); spec.loader.exec_module(ri)
    with tempfile.TemporaryDirectory() as d:
        def put(rid, text):
            os.makedirs(os.path.join(d, rid))
            open(os.path.join(d, rid, 'result.json'), 'w').write(text)
        put('aaaaaaaaaaaa', json.dumps({'label': 'bpe_pos-LEARN_st20k', 'seed': 2}))
        put('bbbbbbbbbbbb', '{"label": "bpe_pos-LEARN_st20k", "seed": 3')   # half-written
        assert ri.finished_run(d, 'bpe_pos-LEARN_st20k', 2) == 'aaaaaaaaaaaa'
        assert ri.finished_run(d, 'bpe_pos-LEARN_st20k', 3) is None, 'a half-written result is not finished'
        assert ri.finished_run(d, 'bpe_pos-LEARN_st20k', 4) is None, 'another seed is not done'
        assert ri.finished_run(d, 'bpe_pos-LEARN_st10k', 2) is None, 'another cell is not done'
    print('  run_ids --done: cell and seed, finished results only OK')



def test_fit_check_measures_a_model():
    """scripts/ops/fit_check.py times a real training step and projects the run (§33.54)."""
    import importlib.util, os
    import morphlm.pipelines.registry as reg
    from morphlm.config import ExperimentConfig, apply_overrides
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'ops', 'fit_check.py')
    spec = importlib.util.spec_from_file_location('fit_check', path)
    fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
    cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'frozen.facts=pos', 'frozen.trainable=true',
                                               'model.d_model=32', 'model.n_layers=1', 'data.seq_len=16',
                                               'train.batch_size=2', 'train.max_steps=100', 'train.lr_horizon=100',
                                               'train.eval_every=50', 'model.norm=rmsnorm', 'model.mlp=swiglu'])
    real = reg.pipeline_vocab
    reg.pipeline_vocab = lambda c: (None, 60)
    try:
        parts, peak, t_step, eval_s, run_s = fc.measure(cfg, steps=1, warmup=1, val_windows=8, device='cpu')
    finally:
        reg.pipeline_vocab = real
    assert parts['trainable_total'] > 0 and t_step > 0 and run_s >= 100 * t_step, (parts, t_step, run_s)
    print('  fit check measures a model OK')


def _refused(overrides, expect=ValueError):
    """True when the overrides, or the config they make, are refused."""
    try:
        apply_overrides(ExperimentConfig(), overrides).validate()
    except expect:
        return True
    return False


def test_morpheme_pipelines_are_english_only():
    """
    det and morph segment with MorphoLEX, and their cache key carries no corpus:
    on another corpus they would read and write the English stream's files under
    a Ukrainian label. Only BPE takes a corpus.
    """
    for pipe in ('det', 'morph'):
        assert _refused([f'data.pipeline={pipe}', 'data.corpus=ukwiki']), pipe
        assert not _refused([f'data.pipeline={pipe}'])
    assert not _refused(['data.pipeline=bpe', 'data.corpus=ukwiki'])
    print('  det/morph refuse a non-English corpus OK')


def test_overrides_refuse_unknown_values():
    """
    A misspelt boolean used to become False (`=ture`), and a misspelt role,
    injection, struct_keep or granularity went through: a role typo silently
    joined the input cell. Each is refused now; the spellings in use still work.
    """
    def coerced(raw):
        return apply_overrides(ExperimentConfig(), [f'frozen.trainable={raw}']).frozen.trainable
    for raw, want in (('true', True), ('false', False), ('1', True), ('0', False), ('yes', True),
                      ('no', False), ('on', True), ('off', False), ('True', True), ('FALSE', False)):
        assert coerced(raw) is want, raw
    for raw in ('ture', '', 'flase', '2'):
        try:
            coerced(raw)
            raise AssertionError(f'{raw!r} must be refused')
        except ValueError:
            pass
    for bad in (['frozen.role=targets'], ['frozen.inject=sum'], ['frozen.struct_keep=morph'],
                ['frozen.dep_granularity=medium'],
                ['frozen.struct_keep=names', 'frozen.facts=pos'],     # keep_struct acts on mstruct only
                ['frozen.struct_keep=morphology']):
        assert _refused(['data.pipeline=bpe'] + bad), bad
    for good in (['frozen.role=target', 'frozen.facts=pos'], ['frozen.inject=add'],
                 ['frozen.struct_keep=names', 'frozen.facts=mstruct'], ['frozen.dep_granularity=fine']):
        assert not _refused(['data.pipeline=bpe'] + good), good
    print('  unknown booleans and enumerated values refused OK')


def _records():
    """(run id, stored config, result label or None) for every results/<run id>/config.json."""
    import json, glob
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    out = []
    for p in sorted(glob.glob(os.path.join(root, '*', 'config.json'))):
        rid = os.path.basename(os.path.dirname(p))
        label = None
        rp = os.path.join(os.path.dirname(p), 'result.json')
        if os.path.exists(rp):
            label = json.load(open(rp)).get('label')
        out.append((rid, json.load(open(p)), label))
    return out


def test_recorded_configs_rebuild():
    """
    The regression net for config identity. Every recorded config rebuilt by
    `ExperimentConfig.from_dict` names its cell exactly as its record does, keeps
    every stored value, and passes `validate()` — so no recorded run is refused
    by a newer check. Its stored dict still hashes to its directory (the run id
    rule is unchanged), and `run_id()` reproduces the directory for every config
    that holds all of today's fields; one written before a field existed hashes
    without it, so its rebuilt id differs, and that is history, not a fault.
    """
    import json, hashlib
    from dataclasses import fields
    recs = _records()
    if not recs:
        print('  (skipped: no results/*/config.json)')
        return
    current = {s: {f.name for f in fields(getattr(ExperimentConfig(), s))}
               for s in ('data', 'surface', 'frozen', 'model', 'train')}
    complete = reproduced = labelled = 0
    for rid, d, label in recs:
        cfg = ExperimentConfig.from_dict(d)
        if label is not None:
            assert cfg.label() == label, (rid, label, cfg.label())
            labelled += 1
        back = json.loads(json.dumps(cfg.to_dict(), default=str))
        for section, values in d.items():
            for k, v in values.items():
                assert back[section][k] == v, (rid, section, k)
        cfg.validate()
        stored = hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]
        assert stored == rid, (rid, stored)
        if all(set(d.get(s, {})) == names for s, names in current.items()):
            complete += 1
            assert cfg.run_id() == rid, rid
        reproduced += cfg.run_id() == rid
    assert reproduced == complete, (reproduced, complete)
    print(f'  {len(recs)} recorded configs rebuild: {labelled} labels match their records, '
          f'all validate; run_id() reproduces {reproduced} directories (every config holding '
          f'all of today\'s fields; the other {len(recs) - reproduced} predate a field) OK')


def test_from_dict_reads_old_and_new_dicts():
    """
    A dict without `surface` predates SurfaceConfig: its stream was the legacy
    one, every channel off (results.md: 7b28f2074be6 is that cell). Fields added
    since take their defaults; an unknown field is refused.
    """
    cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=det', 'surface.emit_whitespace=false',
                                               'frozen.facts=dep,pos', 'model.position=rope'])
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
    assert ExperimentConfig.from_dict(cfg.to_dict()).label() == 'det_dep+pos_Spdi-c_rope'
    old = cfg.to_dict()
    del old['surface']
    del old['model']['position']
    legacy = ExperimentConfig.from_dict(old)
    assert legacy.surface.spec() == '' and legacy.label() == 'det_dep+pos'
    assert legacy.model.position == 'learned'
    try:
        ExperimentConfig.from_dict({**cfg.to_dict(), 'train': {'seed': 0, 'no_such_field': 1}})
        raise AssertionError('an unknown field must be refused')
    except TypeError:
        pass
    print('  from_dict: legacy surface, defaults for new fields, unknown fields refused OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_config: PASS')
