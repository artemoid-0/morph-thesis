"""
Cache key invariants.

The critical one: with no context facts requested, the key must reduce EXACTLY
to the historical format, or the existing BPE cache (and any morph/det cache
built before this refactor) is silently thrown away.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morphlm.data.stream import cache_stem


def test_backward_compatible_stems():
    """These stems exist on disk today. They must keep resolving."""
    cases = [
        ('bpe', 'train', None, 'bpe_train_None_99bc1462ce'),
        ('bpe', 'validation', None, 'bpe_validation_None_e483bcd459'),
        ('bpe', 'train', 3000, 'bpe_train_3000_17a8ff3163'),
    ]
    for pipeline, split, cap, expected in cases:
        got = cache_stem(pipeline, split, cap, '')
        assert got == expected, f'{got} != {expected}'
    print('  backward-compatible cache stems OK')


def test_feature_spec_changes_key():
    """A different feature spec must produce a different cache file."""
    plain = cache_stem('morph', 'train', None, '')
    dep_c = cache_stem('morph', 'train', None, 'dep:coarse')
    dep_f = cache_stem('morph', 'train', None, 'dep:fine')
    both = cache_stem('morph', 'train', None, 'dep:coarse+pos')
    assert len({plain, dep_c, dep_f, both}) == 4, 'feature specs must not collide'
    print('  feature spec changes the key OK')


def test_training_knobs_absent():
    """lr/seed must not appear anywhere in the key derivation."""
    import inspect
    from morphlm.data import stream
    src = inspect.getsource(stream.cache_stem)
    for forbidden in ('lr', 'seed', 'batch'):
        assert forbidden not in src, f'{forbidden} leaked into the cache key'
    print('  training knobs excluded from cache key OK')


def test_ids_and_codes_keyed_separately():
    """
    Token ids do not depend on requested facts; only codes do. Keying them
    together caused one full spaCy pass per fact request.
    """
    from morphlm.data.stream import codes_spec
    ids_a = cache_stem('morph', 'train', None, '')
    ids_b = cache_stem('morph', 'train', None, '')
    assert ids_a == ids_b, 'ids key must not vary with facts'
    c1 = codes_spec(['dep'], 'coarse')
    c2 = codes_spec(['dep'], 'fine')
    c3 = codes_spec(['dep', 'pos'], 'coarse')
    assert len({c1, c2, c3}) == 3, 'codes spec must distinguish gran and facts'
    assert codes_spec([], 'coarse') == '', 'no facts -> no codes file'
    print('  ids/codes keyed separately OK')


def test_facts_to_build_is_per_pipeline():
    """
    A pipeline must only be asked for facts it can PRODUCE.

    morph/det encode() sets dep/pos/entity_label from the spaCy parse but never
    mroot/mstruct/mrole — those need the WordAnalyser, which only the BPE
    pipeline carries. A global fact list made facts_to_build('morph') request
    three impossible facts, which changed the morph codes cache key and
    silently orphaned a four-hour cache.
    """
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import facts_to_build, PIPELINE_CONTEXT_FACTS
    m = apply_overrides(ExperimentConfig(), ['data.pipeline=morph'])
    b = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe'])
    assert facts_to_build(m) == PIPELINE_CONTEXT_FACTS['morph']
    assert 'mroot' not in facts_to_build(m), \
        'morph cannot produce mroot and must not be asked for it'
    assert facts_to_build(b) == (), 'bpe must not pay for unrequested spaCy'
    bd = apply_overrides(b, ['frozen.facts=mroot'])
    assert facts_to_build(bd) == PIPELINE_CONTEXT_FACTS['bpe']
    print('  facts_to_build is per-pipeline OK')


def test_morph_codes_key_is_stable():
    """
    Regression: extending the fact inventory for Level 2 must NOT change the
    morph codes key. It did once, orphaning the cache that every Q2 cell used.
    """
    from morphlm.data.stream import codes_spec
    from morphlm.data.stream import cache_stem as st
    key = st('morph', 'train', None,
             'sfc:pdiwc|' + codes_spec(['dep', 'pos', 'ectx'], 'coarse'))
    assert key == 'morph_train_None_903ea2c955', \
        f'morph codes key drifted to {key}'
    print('  morph codes key stable OK')


def test_codes_are_not_borrowed_across_token_streams():
    """
    Codes are one value PER TOKEN of a specific ids array. Content-based reuse
    (added for G-16) matched on the fact list alone, which is correct when only
    the fact inventory changed but WRONG when the surface configuration
    changed: the ids differ, so borrowed codes describe different tokens.

    Found by scripts/data/cache_index.py --plan, which reported a whitespace-off
    cell as able to reuse punctuation-off codes.

    Asserts the two conditions the loader now requires: distinct surface specs
    produce distinct ids stems, and the reuse guard keys on token count.
    """
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem

    a = apply_overrides(ExperimentConfig(),
                        ['data.pipeline=det', 'surface.emit_whitespace=false'])
    b = apply_overrides(ExperimentConfig(),
                        ['data.pipeline=det', 'surface.emit_punctuation=false',
                         'surface.emit_digits=false'])
    assert a.surface_spec() != b.surface_spec()
    sa = cache_stem('det', 'train', None, a.surface_spec())
    sb = cache_stem('det', 'train', None, b.surface_spec())
    assert sa != sb, 'different surface configs must not share an ids stem'

    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'morphlm', 'data', 'stream.py')).read()
    # the guard lives in stream_codes_files since G-72, which the loader calls
    # with the token count of the ids stream it reads
    assert "cm.get('num_tokens') != num_tokens" in src and \
        "ids_stem, ids_meta_tokens)" in src, \
        'the codes-reuse guard must compare token counts'
    assert 'code/id length mismatch on LOAD' in src, \
        'the load path must assert code/id alignment, not just the build path'
    print('  codes are not borrowed across token streams OK')


def test_exact_key_that_does_not_satisfy_falls_through_to_reuse():
    """
    Regression: `codes_stem` is keyed on the pipeline's free-ride fact list,
    not on what a cell requested. A derived fact (`mprefix`) is never on that
    list, so the stem resolves to a file that predates it. The file EXISTS, so
    the old guard skipped reuse and the run died claiming its cache lacked a
    fact that file was never going to hold.

    Asserts the loader tries reuse whenever the exact file does not SATISFY
    the request, not merely when it is missing.
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'morphlm', 'data', 'stream.py')).read()
    assert 'exact_ok = set(requested).issubset(' in src, \
        'loader must check whether the exact file satisfies the request'
    assert 'and not exact_ok:' in src, \
        'reuse must be gated on satisfaction, not on file existence'
    assert 'not os.path.exists(codes_npy):' not in src.split(
        'reuse is only possible')[1][:300], \
        'the old existence-only guard must be gone'
    print('  exact key that does not satisfy falls through to reuse OK')


def test_cache_codes_ignore_coarsening():
    """
    Codes produced for the cache (and for BLiMP) must be full-range whatever
    the run's `root_classes`: the model folds them itself. A coarsened block
    folding at build time would poison the shared cache and, when scoring new
    text, fold twice and turn `none` into class 0.
    """
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    from morphlm.data.stream import cache_blocks
    base = ['data.pipeline=bpe', 'frozen.facts=mroot,mstruct']
    plain = apply_overrides(ExperimentConfig(), base)
    coarse = apply_overrides(ExperimentConfig(), base + ['frozen.root_classes=512'])
    folded = build_blocks(coarse)[0]
    assert folded.coarsen == 512, 'precondition: the model-side block is coarsened'
    full_blocks = cache_blocks(coarse, ['mroot', 'mstruct'])
    ref_blocks = cache_blocks(plain, ['mroot', 'mstruct'])
    for tok in ({'mroot': 10000, 'mstruct': 'suffixed'}, {'mroot': None},
                {'mroot': 3, 'mstruct': 'simple'}):
        assert [b.code(tok) for b in full_blocks] == [b.code(tok) for b in ref_blocks], tok
    assert full_blocks[0].code({'mroot': 10000}) == 10000, 'must stay unfolded'
    assert folded.code({'mroot': 10000}) != 10000, 'precondition: build_blocks folds'
    assert full_blocks[0].code({'mroot': None}) == full_blocks[0].n_classes - 1


def test_scorer_encoding_is_model_independent():
    """
    blimp_eval shares one encoding across checkpoints whose configs differ
    only in things the MODEL applies — the halves, the causal flag,
    coarsening. That is safe only if no block's code() reads them.
    """
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_blocks
    tok = {'mstruct': 'suffixed', 'mroot': 77, 'mfeat': 'Number=Plur', 'magr': 'Plur'}
    names = ['mstruct', 'mroot', 'mfeat', 'magr']
    ref = [b.code(tok) for b in cache_blocks(apply_overrides(
        ExperimentConfig(), ['data.pipeline=bpe', 'frozen.facts=' + ','.join(names)]), names)]
    for extra in (['frozen.struct_keep=names'], ['frozen.struct_keep=morphology'],
                  ['frozen.causal_identity=true'], ['frozen.root_classes=64'],
                  ['frozen.trainable=true']):
        cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe',
                              'frozen.facts=' + ','.join(names)] + extra)
        assert [b.code(tok) for b in cache_blocks(cfg, names)] == ref, extra


def test_ukrainian_path_leaves_english_untouched():
    """A non-default corpus changes keys, labels and the free ride — only for itself."""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import facts_to_build, PIPELINE_CONTEXT_FACTS, UD_FREE_RIDE
    en = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.bpe_path=data/bpe_17k.json',
                                              'frozen.facts=mstruct'])
    uk = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.bpe_path=data/bpe_uk_17k.json',
                                              'data.corpus=ukwiki', 'frozen.facts=ucase'])
    assert en.surface_spec() == 'tok:bpe_17k', 'the English key must not change'
    # ':nl' — Ukrainian paragraphs end in a newline (G-64); the glued streams
    # built before it carry the old key and are never reused
    assert uk.surface_spec() == 'tok:bpe_uk_17k|corpus:ukwiki:1:2000:nl'
    assert en.label() == 'bpe_mstruct_bpe_17k'
    assert uk.label() == 'bpe_ucase_bpe_uk_17k_ukwiki'
    assert facts_to_build(en) == PIPELINE_CONTEXT_FACTS['bpe']
    assert facts_to_build(uk) == UD_FREE_RIDE
    assert not set(UD_FREE_RIDE) & {'mstruct', 'mroot', 'mrole', 'ectx'}, 'no English-only facts'



def _old_ukwiki_rows(ds, split, frac, hold):
    """The ukwiki loader exactly as it stood before the corpus registry (journey §33.53)."""
    import hashlib
    for art in ds:
        key = hashlib.md5(art['title'].encode('utf-8')).hexdigest()
        h = int(key[:8], 16) % hold
        which = 'validation' if h == 0 else 'test' if h == 1 else 'train'
        if which != split:
            continue
        if which == 'train' and frac < 1.0 and int(key[8:16], 16) % 10000 >= frac * 10000:
            continue
        for para in art['text'].split('\n'):
            yield {'text': para, 'sep': '\n'}


def test_corpus_registry_reproduces_ukwiki():
    """corpora.rows serves ukwiki exactly as the historical loader did, at every split and size."""
    from morphlm.data.corpora import get_corpus, rows
    docs = [{'title': f'Стаття {i}', 'text': f'Перший абзац {i}.\nДругий {i}.\n\nТретій'} for i in range(400)]
    spec = get_corpus('ukwiki')
    for split in ('train', 'validation', 'test'):
        for frac in (1.0, 0.5, 0.2, 0.01):
            for hold in (3, 2000):
                new = list(rows(spec, split, frac, hold, lambda *a, **k: docs))
                old = list(_old_ukwiki_rows(docs, split, frac, hold))
                assert new == old, (split, frac, hold)
    print('  corpus registry reproduces the ukwiki loader OK')


def test_corpus_splits_are_nested_and_disjoint():
    """The hash split partitions documents; a smaller train fraction is a subset of a larger one."""
    from morphlm.data.corpora import CorpusSpec, rows
    docs = [{'title': f'doc {i}', 'text': f'text {i}'} for i in range(3000)]
    spec = CorpusSpec(name='t', language='en', dataset='x', config=None, split_rule='hash',
                      serve='rows', spacy_model='en_core_web_sm', word_rule='symbol', code='t')
    got = {s: {r['text'] for r in rows(spec, s, 1.0, 10, lambda *a, **k: docs)} for s in ('train', 'validation', 'test')}
    assert sum(len(v) for v in got.values()) == len(docs), 'every document in exactly one split'
    assert not (got['train'] & got['validation']) and not (got['train'] & got['test'])
    small = {r['text'] for r in rows(spec, 'train', 0.2, 10, lambda *a, **k: docs)}
    big = {r['text'] for r in rows(spec, 'train', 0.5, 10, lambda *a, **k: docs)}
    assert small < big < got['train'], 'nested sizes'
    assert {r['text'] for r in rows(spec, 'validation', 0.2, 10, lambda *a, **k: docs)} == got['validation'], \
        'the fraction never touches validation'
    native = CorpusSpec(name='n', language='en', dataset='x', config=None, split_rule='native', serve='paragraphs',
                        spacy_model='en_core_web_sm', word_rule='symbol', code='n', key_field=None)
    seen = {}
    def fake(name, config, split):
        seen['split'] = split
        return [{'text': 'a\nb'}]
    assert list(rows(native, 'validation', 1.0, 10, fake)) == [{'text': 'a', 'sep': '\n'}, {'text': 'b', 'sep': '\n'}]
    assert seen['split'] == 'validation', 'native splits are read from the dataset'
    print('  corpus splits disjoint, sizes nested, native and paragraph serving OK')


def test_corpus_guards():
    """Unknown corpora, a WikiText fraction and a wrong-language tagger all fail loudly."""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.corpora import check_tagger
    from morphlm.data.stream import load_corpus
    for bad in (['data.corpus=nosuch'],):
        try:
            apply_overrides(ExperimentConfig(), ['data.pipeline=bpe'] + bad).validate()
            raise AssertionError(f'{bad} must be refused')
        except ValueError:
            pass
    try:
        load_corpus(apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.corpus_fraction=0.5']), 'train')
        raise AssertionError('a WikiText fraction must be refused')
    except ValueError:
        pass
    check_tagger('ukwiki', 'uk_core_news_sm'); check_tagger('wikitext103', 'en_core_web_sm')
    try:
        check_tagger('ukwiki', 'en_core_web_sm')
        raise AssertionError('an English tagger on Ukrainian must be refused')
    except ValueError:
        pass
    print('  corpus guards: unknown name, WikiText fraction, wrong-language tagger OK')



def test_lexicon_names_per_condition():
    """G-66: lexicons are named per corpus and size outside WikiText; loaders fall back to the old name."""
    import os, tempfile
    import numpy as np
    from morphlm.features.lexical import lexicon_condition, lexicon_path, load_lexicon, save_lexicon
    assert lexicon_condition('wikitext103', 1.0) == '' and lexicon_condition(None, 0.5) == ''
    assert lexicon_condition('ukwiki', 0.2) == 'ukwiki0.2' and lexicon_condition('ukwiki', 1) == 'ukwiki1'
    assert lexicon_path('data/bpe_17k.json', 'pos', 'c') == os.path.join('c', 'lexicon_pos_bpe_17k.npz')
    assert lexicon_path('data/bpe_uk_17k.json', 'ucase', 'c', 'ukwiki0.05') == \
        os.path.join('c', 'lexicon_ucase_bpe_uk_17k_ukwiki0.05.npz')
    with tempfile.TemporaryDirectory() as d:
        lex = (np.array([1, 2]), np.array([3, 4], dtype=np.int16), np.array([5, 6]))
        save_lexicon(lexicon_path('t.json', 'ucase', d), lex, {})                     # the historical name only
        assert np.array_equal(load_lexicon('t.json', 'ucase', d, 'ukwiki0.2')[1], lex[1]), 'falls back'
        new = (lex[0], np.array([7, 8], dtype=np.int16), lex[2])
        save_lexicon(lexicon_path('t.json', 'ucase', d, 'ukwiki0.2'), new, {})
        assert np.array_equal(load_lexicon('t.json', 'ucase', d, 'ukwiki0.2')[1], new[1]), 'prefers its own'
        assert np.array_equal(load_lexicon('t.json', 'ucase', d)[1], lex[1]), 'the old name untouched'
    print('  lexicon names per condition, with fallback OK')


def test_derived_codes_stay_with_their_stream():
    """G-72: a corpus's validation split is the same tokens at every fraction;
    a run reads only the derived files of its own stream, and still shares
    build files and reuses an older build file by content."""
    import json, tempfile
    import numpy as np
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import (build_stream, codes_owner, codes_spec, facts_to_build,
                                     stream_codes_files)
    assert codes_owner('bpe_validation_None_34e4d191cf.codes.json', 'bpe_validation_None_') == ('build', None)
    assert codes_owner('bpe_validation_None_34e4d191cf_mzero.codes.json', 'bpe_validation_None_') == \
        ('derived', 'bpe_validation_None_34e4d191cf')
    assert codes_owner('bpe_validation_None_e483bcd459_prefix.codes.json', 'bpe_validation_None_') == \
        ('derived', 'bpe_validation_None_e483bcd459')
    assert codes_owner('bpe_validation_None_odd.codes.json', 'bpe_validation_None_') == (None, None)
    n = 64
    old = os.environ.get('MORPHLM_NO_BUILD')
    os.environ['MORPHLM_NO_BUILD'] = '1'
    try:
        with tempfile.TemporaryDirectory() as d:
            def put(stem, names, value):
                # derived constants as written since G-73: the rule declared
                rules = {f: 'every word' for f in names if f in ('mconst', 'mzero')}
                np.save(os.path.join(d, stem + '.codes.npy'), np.full((len(names), n), value, dtype=np.int16))
                json.dump({'fact_names': names, 'dep_granularity': 'coarse', 'num_tokens': n,
                           **({'rules': rules} if rules else {})},
                          open(os.path.join(d, stem + '.codes.json'), 'w'))

            def cfg(frac, facts, corpus='ukwiki', tok='data/bpe_uk_17k.json'):
                return apply_overrides(ExperimentConfig(), [
                    'data.pipeline=bpe', f'data.bpe_path={tok}', f'data.corpus={corpus}', f'data.cache_dir={d}',
                    f'frozen.facts={facts}'] + ([f'data.corpus_fraction={frac}'] if corpus != 'wikitext103' else []))

            stems = {}
            for frac, mark in ((0.01, 1), (0.05, 5), (0.2, 20)):
                c = cfg(frac, 'ucase')
                sfc = c.surface_spec()
                stems[frac] = ids = build_stream.__globals__['cache_stem']('bpe', 'validation', None, sfc)
                np.save(os.path.join(d, ids + '.npy'), np.zeros(n, dtype=np.int32))
                json.dump({'num_tokens': n, 'num_bytes': n, 'num_chars': n, 'num_articles': 1},
                          open(os.path.join(d, ids + '.json'), 'w'))
                ud = list(facts_to_build(c))
                put(build_stream.__globals__['cache_stem']('bpe', 'validation', None,
                                                           f'{sfc}|{codes_spec(ud, "coarse")}'), ud, 0)
                put(ids + '_mconst', ['ucase', 'mconst'], mark)
            put(stems[0.2] + '_mzero', ['ucase', 'mconst', 'mzero'], 20)       # only 0.2 has a zero slot
            for frac, mark in ((0.01, 1), (0.05, 5), (0.2, 20)):
                got = build_stream(None, cfg(frac, 'mconst'), 'validation', verbose=False)
                assert int(np.asarray(got['codes'])[0, 0]) == mark, f'{frac}: read another fraction\'s null'
                assert int(np.asarray(build_stream(None, cfg(frac, 'ucase'), 'validation',
                                                   verbose=False)['codes'])[0, 0]) == 0
            try:
                build_stream(None, cfg(0.05, 'mzero'), 'validation', verbose=False)
                raise AssertionError('0.05 has no zero slot of its own: it must not borrow 0.2\'s')
            except RuntimeError:
                pass
            own = [os.path.basename(p) for p, _, _ in
                   stream_codes_files(d, 'bpe', 'validation', None, stems[0.05], n)]
            assert stems[0.05] + '_mconst.codes.json' in own and not any(stems[0.2] in p for p in own)
            assert len(own) == 4, own                                       # three build files, its own null
            # a derivation reads its source rows from its own stream too (the
            # zero slot at 0.05 copied 0.01's null row into its file)
            import importlib.util
            spec = importlib.util.spec_from_file_location('derive_lexical_codes', os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'data',
                'derive_lexical_codes.py'))
            dlc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dlc)
            src = dlc.find_source(d, 'validation', stems[0.05], 'mconst', n, out_fact='mzero')
            assert src[2].endswith(stems[0.05] + '_mconst.codes.json'), src[2]
            assert dlc.find_source(d, 'validation', stems[0.2], 'mconst', n, out_fact='mzero')[2].endswith(
                stems[0.2] + '_mconst.codes.json'), 'never the file being replaced'
            # English: an older build file still lends a fact the exact file lacks
            e = cfg(1.0, 'mstruct', corpus='wikitext103', tok='data/bpe_17k.json')
            sfc = e.surface_spec()
            ids = build_stream.__globals__['cache_stem']('bpe', 'validation', None, sfc)
            np.save(os.path.join(d, ids + '.npy'), np.zeros(n + 1, dtype=np.int32))
            json.dump({'num_tokens': n + 1, 'num_bytes': n, 'num_chars': n, 'num_articles': 1},
                      open(os.path.join(d, ids + '.json'), 'w'))
            older = build_stream.__globals__['cache_stem']('bpe', 'validation', None, f'{sfc}|older')
            np.save(os.path.join(d, older + '.codes.npy'), np.full((2, n + 1), 7, dtype=np.int16))
            json.dump({'fact_names': ['dep', 'mstruct'], 'dep_granularity': 'coarse', 'num_tokens': n + 1},
                      open(os.path.join(d, older + '.codes.json'), 'w'))
            assert int(np.asarray(build_stream(None, e, 'validation', verbose=False)['codes'])[0, 0]) == 7
    finally:
        if old is None:
            os.environ.pop('MORPHLM_NO_BUILD', None)
        else:
            os.environ['MORPHLM_NO_BUILD'] = old
    print('  derived codes stay with their stream; build files shared and reused OK')


def test_constant_codes_must_declare_the_rule():
    """G-73: a run of the constant null refuses codes derived before the fix (no
    rule declared) and records the codes it read; a zero slot, whose codes never
    reach the model, is not held to it. The quarantine rule picks the Ukrainian
    constant runs that predate the fix and keeps the rest."""
    import json, tempfile, importlib.util
    import numpy as np
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import build_stream, cache_stem
    old = os.environ.get('MORPHLM_NO_BUILD')
    os.environ['MORPHLM_NO_BUILD'] = '1'
    n = 32
    try:
        with tempfile.TemporaryDirectory() as d:
            def cfg(facts):
                return apply_overrides(ExperimentConfig(), [
                    'data.pipeline=bpe', 'data.bpe_path=data/bpe_uk_17k.json', 'data.corpus=ukwiki',
                    'data.corpus_fraction=0.05', f'data.cache_dir={d}', f'frozen.facts={facts}'])
            ids = cache_stem('bpe', 'validation', None, cfg('mconst').surface_spec())
            np.save(os.path.join(d, ids + '.npy'), np.zeros(n, dtype=np.int32))
            json.dump({'num_tokens': n, 'num_bytes': n, 'num_chars': n, 'num_articles': 1},
                      open(os.path.join(d, ids + '.json'), 'w'))

            def put(out, names, rules=None):
                np.save(os.path.join(d, f'{ids}_{out}.codes.npy'), np.zeros((len(names), n), dtype=np.int16))
                json.dump({'fact_names': names, 'dep_granularity': 'coarse', 'num_tokens': n,
                           **({'rules': rules} if rules else {})},
                          open(os.path.join(d, f'{ids}_{out}.codes.json'), 'w'))
            put('mconst', ['ucase', 'mconst'])
            put('mzero', ['ucase', 'mzero'])
            try:
                build_stream(None, cfg('mconst'), 'validation', verbose=False)
                raise AssertionError('codes derived before G-73 must be refused')
            except RuntimeError as e:
                assert 'G-73' in str(e)
            assert build_stream(None, cfg('mzero'), 'validation', verbose=False)['codes'] is not None
            put('mconst', ['ucase', 'mconst'], {'mconst': 'every word'})
            got = build_stream(None, cfg('mconst'), 'validation', verbose=False)
            assert got['codes_rules'] == {'mconst': 'every word'} and got['codes_file'].endswith('_mconst.codes.npy')
    finally:
        if old is None:
            os.environ.pop('MORPHLM_NO_BUILD', None)
        else:
            os.environ['MORPHLM_NO_BUILD'] = old
    spec = importlib.util.spec_from_file_location('quarantine', os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'ops', 'quarantine.py'))
    q = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(q)
    uk = {'data': {'corpus': 'ukwiki'}, 'frozen': {'facts': ['mconst']}}
    assert q._ukrainian_constant(uk)
    assert not q._ukrainian_constant({'data': {'corpus': 'wikitext103'}, 'frozen': {'facts': ['mconst']}})
    assert not q._ukrainian_constant({'data': {'corpus': 'ukwiki'}, 'frozen': {'facts': ['mzero', 'ucase']}})
    assert not q._scored_on_constant({'facts': ['mconst'], 'data': {'validation': {}}})
    assert q._scored_on_constant({'facts': ['mconst'],
                                  'data': {'validation': {'codes_rules': {'mconst': 'every word'}}}})
    print('  constant codes must declare the rule; quarantine picks the old Ukrainian ones OK')


def test_condition_estimate_reads_the_slice():
    """
    journey §33.63: the pilot's estimate was 15% low on tokens — validation's
    share of the bytes is not its share of the documents — and 2–2.7 times high on
    tagging time, which follows the bytes and not the tokenizer. Another
    tokenizer's train cache of the same slice holds its bytes, and, once caches
    record it, its tagging time. The numbers are the bpeend_uk_17k pilot's
    (7 October), built at 42,600,731 tokens in 58 minutes.
    """
    import json, tempfile, importlib.util
    from morphlm.config import ExperimentConfig, apply_overrides
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        'prepare_condition', os.path.join(root, 'scripts', 'data', 'prepare_condition.py'))
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, 'cache')
        os.makedirs(cache)
        for name in ('bpe_uk_17k.json', 'bpe_uk_17k.meta.json', 'bpeend_uk_17k.json'):
            open(os.path.join(d, name), 'w').write('{}')
        over = lambda tok: ['data.pipeline=bpe', 'data.corpus=ukwiki', 'data.corpus_fraction=0.05',
                            'data.corpus_holdout=2000', f'data.bpe_path={os.path.join(d, tok + ".json")}',
                            'data.spacy_model=uk_core_news_sm', f'data.cache_dir={cache}']
        new = os.path.join(d, 'bpeend_uk_17k.json')
        val = {'num_tokens': 363158, 'num_bytes': 1989866}

        assert pc.same_text(over('bpeend_uk_17k'), cache, new) == (None, None)
        tokens, basis, hours = pc.estimate(80, val, 2000, 0.05)
        assert abs(tokens - 36.28e6) < 0.05e6 and 'at most' in hours, (tokens, hours)   # what it printed

        stem = cache_stem('bpe', 'train', None, apply_overrides(ExperimentConfig(), over('bpe_uk_17k')).surface_spec())
        meta = os.path.join(cache, stem + '.json')
        json.dump({'num_tokens': 37972206, 'num_bytes': 235673205}, open(meta, 'w'))
        sib = pc.same_text(over('bpeend_uk_17k'), cache, new)
        assert sib[0] == 'bpe_uk_17k', sib
        tokens, basis, hours = pc.estimate(80, val, 2000, 0.05, sib)
        assert abs(tokens / 42600731 - 1) < 0.02, tokens
        assert 'at most ~2.6 h' in hours, hours                 # an upper bound: it took 58 min

        json.dump({'num_tokens': 37972206, 'num_bytes': 235673205, 'build_seconds': 3479.0}, open(meta, 'w'))
        tokens, basis, hours = pc.estimate(80, val, 2000, 0.05, pc.same_text(over('bpeend_uk_17k'), cache, new))
        assert hours.startswith('~58 min'), hours
    print('  a condition\'s estimate reads the slice from another tokenizer\'s cache OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_cache_keys: PASS')
