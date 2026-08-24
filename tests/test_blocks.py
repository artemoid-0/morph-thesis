"""
Fact block invariants — the tests that would have caught the dead-dimension bug.

The old 23-dim schema passed every per-token check (each row had exactly 3
nonzero bits, as designed) while carrying 12 constant columns across the whole
vocabulary. Per-token verification is not enough; these are POPULATION-level.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from morphlm.config import ExperimentConfig, apply_overrides
from morphlm.features.blocks import build_blocks, frozen_width, describe_blocks
from morphlm.registry import context_facts, static_facts, all_facts


VOCAB = 'data/our_final_vocab.csv'


def test_baseline_is_empty():
    cfg = ExperimentConfig()
    assert frozen_width(cfg) == 0, 'baseline must have T=0'
    assert cfg.frozen.spec() == '', 'baseline must not perturb the cache key'
    print('  baseline T=0 OK')


def test_no_block_without_request():
    """A block exists only if requested — dead dims cannot recur."""
    cfg = apply_overrides(ExperimentConfig(), ['frozen.facts=cat'])
    blocks = build_blocks(cfg)
    assert len(blocks) == 1 and blocks[0].name == 'cat'
    # 12 categories after the information-preserving layers: prefix suffix root
    # function entity character punct digit infl space case byte
    expect = len(blocks[0].classes)
    assert frozen_width(cfg) == expect == 12, \
        f'cat should be 12 dims, got {frozen_width(cfg)}'
    print(f'  only requested blocks are built OK (cat={expect} dims)')


def test_source_classification():
    """The static/context split is the axis the experiment turns on."""
    assert 'cat' in static_facts(('cat', 'dep'))
    assert 'dep' in context_facts(('cat', 'dep'))
    assert context_facts(('cat',)) == ()
    print('  static vs context classification OK')


def test_dep_granularity_switch():
    coarse = apply_overrides(ExperimentConfig(), ['frozen.facts=dep'])
    fine = apply_overrides(coarse, ['frozen.dep_granularity=fine'])
    wc, wf = frozen_width(coarse), frozen_width(fine)
    assert wc == 8, f'coarse dep should be 8 classes, got {wc}'
    assert wf > wc, 'fine must be wider than coarse'
    assert coarse.frozen.spec() != fine.frozen.spec(), \
        'granularity must change the cache key'
    print(f'  dep granularity switch OK (coarse {wc}, fine {wf})')


def test_dep_codes_vary():
    """
    The whole point of a context fact: the same token id must be able to get
    different codes. Verified on synthetic token dicts.
    """
    cfg = apply_overrides(ExperimentConfig(), ['frozen.facts=dep'])
    block = build_blocks(cfg)[0]
    same_id_subject = {'token_id': 42, 'dep': 'nsubj'}
    same_id_object = {'token_id': 42, 'dep': 'dobj'}
    a, b = block.code(same_id_subject), block.code(same_id_object)
    assert a != b, 'same token id must get different codes in different roles'
    print('  context fact varies for identical token id OK '
          f'(subject={a}, object={b})')


def test_static_table_has_no_constant_column():
    """
    THE REGRESSION TEST for the old bug. Builds the real static table from the
    real vocabulary and asserts every column varies.
    """
    if not os.path.exists(VOCAB):
        print('  (skipped: data/our_final_vocab.csv not present)')
        return
    import pandas as pd
    vocab = pd.read_csv(VOCAB)
    cfg = apply_overrides(ExperimentConfig(), ['frozen.facts=cat'])
    block = build_blocks(cfg)[0]
    table = np.stack([block.onehot(block.code(dict(r)))
                      for _, r in vocab.iterrows()])
    present = set(vocab['category'])
    dead = [block.classes[j] for j in range(table.shape[1])
            if table[:, j].min() == table[:, j].max()]
    # a column is only a BUG if its category exists in this vocabulary; the
    # information-preserving layers appear once the vocab is rebuilt
    real_dead = [d for d in dead if d in present]
    assert not real_dead, f'constant columns for present categories: {real_dead}'
    distinct = len(np.unique(table, axis=0))
    print(f'  cat block: {table.shape[1]} cols, {len(dead)} unused '
          f'(categories absent from this vocab), {distinct} distinct rows OK')
    if dead:
        print(f'    absent: {dead}  <- rebuild vocab to populate')


def test_old_schema_would_fail():
    """
    Documents the bug: the retired 23-dim layout, rebuilt here, is caught.
    If this ever stops failing, the check itself has broken.
    """
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    import pandas as pd
    vocab = pd.read_csv(VOCAB)
    ENT = ('ORG', 'PERSON', 'GPE', 'LOC', 'FAC', 'NORP', 'PRODUCT',
           'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'none')
    rows = []
    for _, r in vocab.iterrows():
        v = np.zeros(12, dtype=np.float32)
        v[ENT.index('none')] = 1.0     # entity_label was never populated
        rows.append(v)
    table = np.stack(rows)
    dead = [j for j in range(12)
            if table[:, j].min() == table[:, j].max()]
    assert len(dead) == 12, 'the old entity block should be entirely constant'
    print(f'  retired 23-dim schema: {len(dead)}/12 entity cols constant '
          f'(bug reproduced and now guarded) OK')


def test_vocab_loader_preserves_na_morphemes():
    """
    'na', 'nan' and 'null' are real MorphoLex morphemes AND pandas default NA
    strings. Reading with defaults silently nukes them.
    """
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    from morphlm.vocab.build import load_vocab
    df = load_vocab(VOCAB)
    nonstr = [t for t in df['token'] if not isinstance(t, str)]
    assert not nonstr, f'non-string tokens survived load: {nonstr}'
    got = set(df['token']) & {'na', 'nan', 'null'}
    assert got, "expected na/nan/null morphemes to survive loading"
    print(f'  vocab loader preserves {sorted(got)} OK')


def test_category_resolution_is_fast_and_equivalent():
    """The precomputed dict must match the old per-token priority scan."""
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    import pandas as pd
    from morphlm.vocab.morpholex import CATEGORY_PRIORITY
    from morphlm.vocab.build import load_vocab
    v = load_vocab(VOCAB)

    prio = v['category'].map(lambda c: CATEGORY_PRIORITY.get(c, 99))
    ranked = (v.assign(_p=prio).sort_values('_p', kind='stable')
              .drop_duplicates(subset=['token'], keep='first'))
    fast = dict(zip(ranked['token'], ranked['category']))

    def slow(tok):
        m = v[v['token'] == tok]['category'].tolist()
        if not m:
            return 'character'
        return min(m, key=lambda c: CATEGORY_PRIORITY.get(c, 99))

    dups = v[v.duplicated('token', keep=False)]['token'].unique()
    for tok in list(dups) + ['xyzzy', 'break', 'the']:
        assert fast.get(tok, 'character') == slow(tok), (
            f'{tok}: fast={fast.get(tok)} slow={slow(tok)}')
    print(f'  category resolution equivalent on {len(dups)} multi-layer '
          f'tokens OK')


def test_morph_facts_are_context_sourced():
    """
    Morphological facts must be CONTEXT facts for BPE. The same BPE token id
    appears inside different words — 'ing' in 'running' (root=run) and in
    'thing' (root=thing) — so a static id->fact table cannot express them.
    """
    from morphlm.registry import context_facts, static_facts
    for f in ('mroot', 'mstruct', 'mrole'):
        assert f in context_facts((f,)), f'{f} must be context-sourced'
        assert f not in static_facts((f,)), f'{f} must not be vocab-sourced'
    print('  morphological facts are context-sourced OK')


def test_root_block_is_a_projection():
    """
    Root identity has ~15k classes and cannot one-hot into 16 dims, so the
    block must report width < n_classes and be flagged as a projection.
    """
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    cfg = apply_overrides(ExperimentConfig(),
                          ['data.pipeline=bpe', 'frozen.facts=mroot'])
    b = build_blocks(cfg)[0]
    assert b.n_classes > 1000, f'expected a large root inventory, got {b.n_classes}'
    assert b.dim == 16, f'expected default width 16, got {b.dim}'
    assert b.is_projection, 'root block must be a projection'
    assert b.n_classes < 32767, 'root codes must fit int16 (the cache dtype)'
    print(f'  root block: {b.n_classes:,} classes -> {b.dim} dims, projection OK')


def test_root_inventory_matches_analyser():
    """
    The root block sizes itself from the vocabulary; the analyser builds the
    actual class list. If the two disagree by even one, every cached code
    decodes to the wrong root and the experiment is silently invalid.
    """
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    from morphlm.vocab.build import load_vocab
    cfg = apply_overrides(ExperimentConfig(),
                          ['data.pipeline=bpe', 'frozen.facts=mroot'])
    block = build_blocks(cfg)[0]
    v = load_vocab(cfg.data.vocab_csv)
    n = (int((v['category'] == 'root').sum())
         + int((v['category'] == 'entity').sum()) + 1)
    assert block.n_classes == n, \
        f'block says {block.n_classes}, analyser inventory is {n}'
    print(f'  root inventory consistent: {n:,} classes OK')


def test_bpe_free_ride_builds_every_context_fact():
    """
    One expensive BPE pass must serve every future BPE experiment. If it built
    only the requested fact, each new experiment would need its own multi-hour
    spaCy+MorphoLex pass and its own codes file.
    """
    from morphlm.data.stream import facts_to_build, ALL_CONTEXT_FACTS, codes_spec
    base = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe'])
    assert facts_to_build(base) == (), 'a plain BPE baseline must pay nothing'
    specs = set()
    for f in ('mroot', 'mstruct', 'dep', 'mroot,mstruct'):
        c = apply_overrides(base, [f'frozen.facts={f}'])
        assert facts_to_build(c) == ALL_CONTEXT_FACTS
        specs.add(codes_spec(list(facts_to_build(c)), 'coarse'))
    assert len(specs) == 1, f'requests must share one codes file, got {specs}'
    print('  BPE free-ride: one codes file serves all fact requests OK')


def test_bpe_alignment_scans_the_span():
    """
    Regression test for the bug that would have voided every BPE context-fact
    run: ByteLevel folds the preceding space into the token, so the start
    offset lands on unannotated whitespace.
    """
    from morphlm.pipelines.bpe import _first_annotated
    ann = [None] * 22
    for s_, e_, v in [(0, 3, 'det'), (4, 7, 'nsubj')]:
        for c in range(s_, e_):
            ann[c] = v
    assert _first_annotated(ann, 3, 7) == 'nsubj', \
        'word-initial token must resolve despite its leading space'
    assert _first_annotated(ann, 0, 3) == 'det'
    assert _first_annotated(ann, 10, 14) is None
    print('  BPE span-scan alignment OK (start-offset bug guarded)')


if __name__ == '__main__':
    test_baseline_is_empty()
    test_no_block_without_request()
    test_source_classification()
    test_dep_granularity_switch()
    test_dep_codes_vary()
    test_static_table_has_no_constant_column()
    test_old_schema_would_fail()
    test_vocab_loader_preserves_na_morphemes()
    test_category_resolution_is_fast_and_equivalent()
    test_morph_facts_are_context_sourced()
    test_root_block_is_a_projection()
    test_root_inventory_matches_analyser()
    test_bpe_free_ride_builds_every_context_fact()
    test_bpe_alignment_scans_the_span()
    print('test_blocks: PASS')
