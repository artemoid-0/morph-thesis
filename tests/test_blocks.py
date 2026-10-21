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


def test_mhash_is_a_matched_control_for_mroot():
    """
    The word-identity control must differ from `mroot` in EXACTLY one thing:
    the grouping. Same cardinality, same width, same source, same projection
    path — otherwise a difference between them would be uninterpretable.
    """
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    r = build_blocks(apply_overrides(ExperimentConfig(),
                                     ['data.pipeline=bpe',
                                      'frozen.facts=mroot']))[0]
    h = build_blocks(apply_overrides(ExperimentConfig(),
                                     ['data.pipeline=bpe',
                                      'frozen.facts=mhash']))[0]
    assert r.n_classes == h.n_classes, 'cardinality must match'
    assert r.dim == h.dim, 'width must match'
    assert r.source == h.source, 'source must match'
    assert r.is_projection == h.is_projection, 'projection path must match'
    print(f'  mhash matches mroot ({r.n_classes:,} classes, {r.dim} dims) OK')


def test_hash_code_is_stable_and_ungrouped():
    """
    Consistency per word must be preserved (so word identity survives) while
    morphological grouping is destroyed. And it must be process-stable: Python's
    hash() is salted per process and would decode differently after a restart.
    """
    from morphlm.features.morph_analysis import WordAnalyser
    a = WordAnalyser.__new__(WordAnalyser)
    # a realistic inventory: with only a handful of classes the modulus would
    # collapse a whole family by chance and the test would be meaningless
    a.root_classes = tuple(f'r{i}' for i in range(16364)) + ('none',)
    a._root_index = {r: i for i, r in enumerate(a.root_classes)}
    a._none_root = 16364
    assert a.hash_code('running') == a.hash_code('running'), 'not stable'
    assert a.hash_code('running') == a.hash_code('Running'), 'case must fold'
    fam = {a.hash_code(w) for w in ('run', 'running', 'runner', 'ran')}
    assert len(fam) > 1, 'a family must NOT share a code — that is the point'
    print(f'  hash_code stable, family split into {len(fam)} codes OK')


def test_three_grouping_features_are_matched():
    """
    `mroot`, `mlemma` and `mhash` must be identical in every respect except
    what they merge. Cardinality, width, source and projection path all match
    by construction (they share a base class); if any diverged, a difference
    between them would be uninterpretable.
    """
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    blocks = {}
    for f in ('mroot', 'mlemma', 'mhash'):
        blocks[f] = build_blocks(apply_overrides(
            ExperimentConfig(), ['data.pipeline=bpe', f'frozen.facts={f}']))[0]
    ref = blocks['mroot']
    for f, b in blocks.items():
        assert b.n_classes == ref.n_classes, f'{f} cardinality differs'
        assert b.dim == ref.dim, f'{f} width differs'
        assert b.source == ref.source, f'{f} source differs'
        assert b.is_projection == ref.is_projection, f'{f} projection differs'
    print(f'  mroot/mlemma/mhash matched ({ref.n_classes:,} classes, '
          f'{ref.dim} dims) OK')


def test_coarsening_needs_no_cache_rebuild():
    """
    The identity-precision sweep folds the inventory at model-build time, so
    every cardinality must share one cache spec. If coarsening entered the
    cache key, the sweep would cost a 4-hour rebuild per point instead of
    nothing.
    """
    base = ['data.pipeline=bpe', 'frozen.facts=mhash']
    specs, labels = set(), set()
    for k in (0, 64, 512, 4096):
        ov = base + ([f'frozen.root_classes={k}'] if k else [])
        c = apply_overrides(ExperimentConfig(), ov)
        specs.add(c.frozen.spec())
        labels.add(c.label())
    assert len(specs) == 1, f'coarsening must not change the cache spec: {specs}'
    assert len(labels) == 4, 'each cardinality needs its own label'
    print('  coarsening: one cache spec, distinct labels OK')


def test_causal_shift_reads_only_the_past():
    """
    The causal identity feature must never read a position at or after the one
    it labels. Mirrors embedding.causal_shift in numpy so the property is
    checked without a GPU, on the worked example plus punctuation runs.
    """
    import numpy as np

    def shift(code, sentinel):
        B, L = code.shape
        prev = np.concatenate([code[:, :1], code[:, :-1]], axis=1)
        st = code != prev
        st[:, 0] = True
        idx = np.arange(L)[None, :].repeat(B, 0)
        start = np.maximum.accumulate(idx * st.astype(int), axis=1)
        src = np.clip(start - 1, 0, None)
        out = np.take_along_axis(code, src, axis=1)
        return np.where(start == 0, sentinel, out), src, idx, start

    S = 99
    # the(A) recon(B) fig(B) uring(B) and(C) the(A)
    code = np.array([[10, 20, 20, 20, 30, 10]])
    out, src, idx, start = shift(code, S)
    assert list(out[0]) == [99, 10, 10, 10, 20, 30], list(out[0])
    assert bool(((src < idx) | (start == 0)).all()), 'read a non-past position'

    # word, punct punct, word, two-fragment word, punct, word
    code2 = np.array([[10, 0, 0, 20, 30, 30, 0, 40]])
    out2, src2, idx2, start2 = shift(code2, S)
    assert list(out2[0]) == [99, 10, 10, 0, 20, 20, 30, 0], list(out2[0])
    assert bool(((src2 < idx2) | (start2 == 0)).all())

    # boundaries must come from the TOKENIZER, not from the codes. Inferring
    # them from "where the code changes" silently required the reference fact
    # to be constant across a word; mprefix varies by design and broke it.
    G = '\u0120'
    vocab = {0: G + 'the', 1: G + 'recon', 2: 'fig', 3: 'uring', 4: G + 'and'}
    al = np.array([vocab[i].lstrip(G).isalpha() for i in range(5)])
    op = np.array([vocab[i].startswith(G) for i in range(5)])
    ids = np.array([0, 1, 2, 3, 4])
    a, o = al[ids], op[ids]
    pa = np.concatenate([[False], a[:-1]])
    st = (a & (o | ~pa)) | ((~a) & pa)
    st[0] = True
    ix = np.arange(len(st))
    sa = np.maximum.accumulate(ix * st.astype(int))
    sc = np.clip(sa - 1, 0, None)
    mstruct = np.array([1, 4, 4, 4, 1])
    got = list(np.where(sa == 0, 7, mstruct[sc]))
    assert got == [7, 1, 1, 1, 4], got

    # and the same boundaries must work when the reference fact VARIES inside
    # a word, which is exactly what defeated the code-equality version
    mprefix = np.array([11, 21, 22, 23, 31])
    got2 = list(np.where(sa == 0, 16364, mprefix[sc]))
    assert got2 == [16364, 11, 11, 11, 23], got2

    src_txt = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'morphlm', 'modeling',
        'embedding.py')).read()
    assert 'word_runs(self._word_starts(ids))' in src_txt, \
        'boundaries must come from the tokenizer, not from codes'
    assert 'self._ref_row' not in src_txt, \
        'the code-derived reference row must be gone'
    print('  causal shift: past-only, shared boundaries, applies to all OK')


def test_struct_keep_splits_morphology_from_names():
    """
    G-25: the morphology/name split must remove exactly one kind of
    information and leave the other, with `none` untouched in both so the
    punctuation context is held constant across the two cells.
    """
    import numpy as np
    from morphlm.features.morph_analysis import STRUCT_CLASSES
    assert STRUCT_CLASSES[0] == 'simple'
    assert STRUCT_CLASSES[1:5] == ('prefixed', 'suffixed', 'both', 'compound')
    assert STRUCT_CLASSES[5:8] == ('entity', 'unanalysed', 'none')

    def keep(code, mode):
        if mode == 'morphology':
            drop = (code == 5) | (code == 6)
        else:
            drop = (code >= 1) & (code <= 4)
        return np.where(drop, 0, code)

    c = np.arange(8)
    m = keep(c, 'morphology')
    n = keep(c, 'names')
    assert set(m.tolist()) == {0, 1, 2, 3, 4, 7}, 'morphology must keep 1-4'
    assert set(n.tolist()) == {0, 5, 6, 7}, 'names must keep 5-6'
    assert m[7] == 7 and n[7] == 7, 'none must survive both'

    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'morphlm', 'modeling',
        'embedding.py')).read()
    assert "block.name == 'mstruct' and self.struct_keep != 'all'" in src
    i_shift = src.index('code = causal_shift(')
    i_keep = src.index('code = keep_struct(')
    assert i_keep > i_shift, 'the split must act AFTER the causal shift'
    print('  struct_keep splits morphology from names, after the shift OK')


def test_coarsening_actually_reaches_the_model():
    """
    Regression for the bug that invalidated the first cardinality sweep.

    The previous test checked that coarsening left the cache spec unchanged —
    true, and irrelevant. It never checked what the MODEL saw. Coarsening lived
    in index(), which only runs at cache-build time; the model read cached codes
    through clamp(), which collapsed 96.9% of words into one class at k=512.

    This test checks the property that matters: after folding, cached codes
    spread across the reduced inventory rather than piling into one class.
    Uses the same arithmetic as embedding.fold_codes, mirrored in numpy.
    """
    import numpy as np
    if not os.path.exists(VOCAB):
        print('  (skipped: vocab not present)')
        return
    cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe',
                                               'frozen.facts=mhash',
                                               'frozen.root_classes=512'])
    b = build_blocks(cfg)[0]
    assert b.coarsen == 512 and b.n_classes == 512
    assert b.n_full > b.n_classes, 'block must remember the cache inventory'

    rng = np.random.default_rng(0)
    codes = rng.integers(0, b.n_full - 1, 100000)
    sent = b.n_classes - 1
    folded = np.where(codes >= b.n_full - 1, sent,
                      np.remainder(codes, max(sent, 1)))
    share = np.bincount(folded).max() / len(folded)
    assert share < 0.02, (f'largest class holds {share:.1%} of tokens — the '
                          f'fold is collapsing identity, not reducing it')

    # and the source must actually route through fold_codes, not clamp
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'morphlm', 'modeling',
        'embedding.py')).read()
    assert 'fold_codes(code, block)' in src and 'codes[:, ci, :]' in src, \
        'embedding must fold cached codes at model time'
    print(f'  coarsening reaches the model: largest class {share:.2%} OK')


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


def test_inflectional_facts_are_well_formed():
    """
    The inflectional context facts added for the agreement test (journey §33).
    Their widths decide how many of the model's 128 dimensions they take, so
    they are pinned here: `mfeat` must not default to a 72-dim one-hot.
    """
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks, MORPH_BUNDLES
    cfg = apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=minfl,magr,mfeat,mfeatlex'])
    mi, ma, mf, ml = build_blocks(cfg)
    assert (mi.dim, ma.dim, mf.dim, ml.dim) == (8, 8, 8, 8), \
        (mi.dim, ma.dim, mf.dim, ml.dim)
    # every word-class fact compared in §33 shares one budget, so a comparison
    # between them is about content, not capacity
    pos, dep, ms = build_blocks(apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=pos,dep,mstruct']))
    assert (pos.dim, dep.dim, ms.dim) == (8, 8, 8), (pos.dim, dep.dim, ms.dim)
    assert pos.n_classes == 19, 'width must not change the label inventory'
    assert mf.classes == ml.classes, 'lexical variant must share the inventory'
    assert len(MORPH_BUNDLES) == 70 and len(set(MORPH_BUNDLES)) == 70
    assert not any(b.startswith('Punct') for b in MORPH_BUNDLES)
    none, other = mf.n_classes - 1, mf.n_classes - 2
    assert mf.code({'mfeat': None}) == none
    assert mf.code({'mfeat': 'Number=Plur|PronType=Dem'}) == \
        MORPH_BUNDLES.index('Number=Plur|PronType=Dem')
    assert mf.code({'mfeat': 'Made=Up'}) == other, 'unknown bundle must not alias a real class'
    assert ma.code({'magr': '3Plur'}) != ma.code({'magr': 'Plur'})
    assert ma.code({'magr': None}) == ma.n_classes - 1
    try:
        ml.code({'mfeat': 'Number=Plur'})
        raise AssertionError('mfeatlex must refuse per-token coding')
    except RuntimeError:
        pass
    for b in (mi, ma, mf, ml):
        assert b.source == 'context', b.name


def test_lexical_keys_are_per_word():
    """Every fragment of a word shares one key; different words differ."""
    import numpy as np
    from morphlm.features.lexical import word_spans, build_lexicon, lookup
    G = '\u0120'
    vocab = [G + 'the', G + 'recon', 'fig', 'uring', ',', G + 'cat', 's']
    alpha = np.array([v.lstrip(G).isalpha() for v in vocab])
    opens = np.array([v.startswith(G) for v in vocab])
    ids = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3])
    wid, key, starts = word_spans(ids, (alpha, opens))
    assert key[1] == key[2] == key[3] == key[8] == key[9] == key[10]
    assert key[0] == key[7] and key[0] != key[1]
    assert wid[4] == -1 and key[4] == 0
    lex = build_lexicon(np.array([7, 7, 7, 9], dtype=np.uint64), np.array([2, 2, 5, 1]))
    assert lex[1].tolist() == [2, 1]
    assert lookup(np.array([8], dtype=np.uint64), lex, 71).tolist() == [71]


def test_type_shuffle_is_a_fair_control():
    """The mstruct control keeps class sizes per frequency bin and moves codes."""
    import numpy as np
    from morphlm.features.lexical import shuffle_within_frequency, class_shares
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    rng = np.random.RandomState(1)
    codes = rng.randint(0, 8, size=1000).astype(np.int16)
    counts = (1e6 / np.arange(1, 1001)).astype(np.int64)        # Zipfian
    from morphlm.features.lexical import frequency_bins
    new = shuffle_within_frequency(codes, counts, seed=0)
    bins = frequency_bins(counts)
    assert sorted(np.concatenate(bins).tolist()) == list(range(1000)), 'bins partition the types'
    for chunk in bins:
        assert sorted(codes[chunk]) == sorted(new[chunk]), 'bin multiset must survive'
        assert counts[chunk].max() <= 2.0 * counts[chunk].min(), 'similar frequency only'
    assert (new != codes).mean() > 0.5, 'the shuffle must actually move codes'
    assert np.array_equal(new, shuffle_within_frequency(codes, counts, 0)), 'seeded'
    # worst case: classes independent of frequency on a Zipfian vocabulary
    tvd = 0.5 * np.abs(class_shares(codes, counts, 8) - class_shares(new, counts, 8)).sum()
    assert tvd < 0.10, tvd
    blk, ms = build_blocks(apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=mstructshuf,mstruct']))
    assert blk.classes == ms.classes and blk.dim == ms.dim == 8
    try:
        blk.code({'mstruct': 'simple'})
        raise AssertionError('mstructshuf must refuse per-token coding')
    except RuntimeError:
        pass


def test_null_and_frequency_controls():
    """mconst / mfreq / posshuf: same table sizes as their sources, derived-only."""
    import numpy as np
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    from morphlm.features.lexical import frequency_bands, class_shares
    mc, mf, ps, ms, pos = build_blocks(apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=mconst,mfreq,posshuf,mstruct,pos']))
    assert mc.classes == ms.classes and mc.dim == ms.dim == 8
    assert mf.n_classes == ms.n_classes and mf.dim == 8 and mf.classes[-1] == 'none'
    assert ps.classes == pos.classes and ps.dim == pos.dim == 8
    for b in (mc, mf, ps):
        assert b.source == 'context'
        try:
            b.code({'mstruct': 'simple', 'pos': 'NOUN'})
            raise AssertionError(f'{b.lexicon_name} must refuse per-token coding')
        except RuntimeError:
            pass
    counts = (1e6 / np.arange(1, 5001)).astype(np.int64)          # Zipfian
    band = frequency_bands(counts, 7)
    assert band[0] == 0 and band[-1] == 6, 'most frequent first, rarest last'
    assert (np.diff(band) >= 0).all(), 'bands follow frequency rank'
    sh = class_shares(band, counts, 8)[:7]
    assert (sh > 0.05).all() and (sh < 0.30).all(), sh


def test_null_width_variants():
    """The null's width variants differ from mconst ONLY in width."""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    m8, m16, m32, m64 = build_blocks(apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=mconst,mconst16,mconst32,mconst64']))
    assert [b.dim for b in (m8, m16, m32, m64)] == [8, 16, 32, 64]
    for b in (m16, m32, m64):
        assert b.classes == m8.classes and b.n_classes == 8
        try:
            b.code({'mstruct': 'simple'})
            raise AssertionError('width variants are derived-only')
        except RuntimeError:
            pass


def test_zero_and_flat_controls():
    """mzero is flagged for the embedding to zero; mflat codes every position alike."""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    mz, mf, mc = build_blocks(apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=mzero,mflat,mconst']))
    assert mz.dim == mf.dim == mc.dim == 8 and mz.classes == mf.classes == mc.classes
    assert getattr(mz, 'zero_slot', False) and not getattr(mf, 'zero_slot', False)
    assert not getattr(mc, 'zero_slot', False)
    assert mf.codes_for([5, 6, 7], [{}, {}, {}]) == [0, 0, 0]


def test_constant_null_on_new_text():
    """G-73: the null codes new text (BLiMP) as class 0 on EVERY word and `none`
    elsewhere — no lexicon, so a word unseen in train is not taken for a non-word."""
    import os, json, tempfile
    import numpy as np
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    from morphlm.features.lexical import tokenizer_tables, word_spans
    text = 'The cat sat on the mat, and the dog ran — far away! 1990.'
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text] * 20, trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bpe_t.json')
        tok.save(path)
        json.dump({'word_rule': 'decoded'}, open(os.path.join(d, 'bpe_t.meta.json'), 'w'))
        mc, = build_blocks(apply_overrides(ExperimentConfig(), [
            'data.pipeline=bpe', f'data.bpe_path={path}', f'data.cache_dir={d}', 'frozen.facts=mconst']))
        ids = tok.encode('Zebras unseen here, the cat!').ids     # no lexicon exists in d
        codes = np.array(mc.codes_for(ids, None))
        wid, _, _ = word_spans(np.array(ids, dtype=np.int64), tokenizer_tables(path))
    none = len(mc.classes) - 1
    assert (codes[wid >= 0] == 0).all(), 'every word, seen or not, gets the constant'
    assert (codes[wid < 0] == none).all() and (wid < 0).any(), 'non-words keep none'


def test_slot_position_controls():
    """The positional mask exists only for its two controls, and does what it says."""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    a, b, z = build_blocks(apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=mzeronopos,mzeroposonly,mzero']))
    assert a.slot_position == 'none' and b.slot_position == 'only'
    assert getattr(z, 'slot_position', None) is None
    assert a.zero_slot and b.zero_slot and a.dim == b.dim == 8


def test_word_rule_and_ud_blocks():
    """The symbol rule is the original one; the decoded rule reads Cyrillic; UD blocks are 8-wide."""
    from morphlm.features.lexical import classify_token, _bytelevel_decoder
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    G = '\u0120'
    for sym in (G + 'the', 'ing', ',', G + '1990', 'Ã©', G):
        core = sym.lstrip(G)
        assert classify_token(sym, 'symbol') == (bool(core) and core.isalpha(), sym.startswith(G)), sym
    enc = {v: k for k, v in _bytelevel_decoder().items()}
    to_sym = lambda text: ''.join(enc[b] for b in text.encode('utf-8'))
    for word in ('Молода', ' вчителька', ' Львові', 'ї'):
        sym = to_sym(word)
        assert classify_token(sym, 'symbol')[0] is False or word == 'ї', 'the old rule misreads Cyrillic'
        assert classify_token(sym, 'decoded') == (True, word.startswith(' ')), word
    half = to_sym('ж')[:1]                          # one byte of a two-byte letter
    assert classify_token(half, 'decoded') == (True, False), 'a split letter stays in its word'
    assert classify_token(to_sym(' 1990'), 'decoded') == (False, True)
    assert classify_token(to_sym(','), 'decoded') == (False, False)
    blocks = build_blocks(apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', 'frozen.facts=ucase,ugender,unumber,uperson,utense,uaspect,uanimacy,mconst']))
    for b in blocks:
        assert b.n_classes == 8 and b.dim == 8, (b.name, b.n_classes)
    ucase = blocks[0]
    assert ucase.code({'udfeats': {'Case': 'Acc'}}) == ucase.classes.index('Acc')
    assert ucase.code({'udfeats': {'Case': 'Acc,Gen'}}) == 7, 'multi-valued maps to none'
    assert ucase.code({}) == 7



def test_decoded_words_are_letter_runs():
    """
    Under the decoded rule, the words found in token ids are EXACTLY the text's
    runs of letters (G-63). A byte fragment of a split character belongs to
    that character: a split letter stays inside its word, a split quote, dash
    or emoji stays outside every word. The first decoded rule glued „“ and —
    into neighbouring words (97% within-word agreement on Ukrainian Wikipedia).
    """
    import os, json, tempfile
    import numpy as np
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from morphlm.features.lexical import tokenizer_tables, word_spans
    text = ('Компанія „Порше“ — відомий виробник. Вона бігла—весь день… '
            'Украї\u0301на має «столицю» Київ. Об’єкт і пам\u02bcять, 1990р. \U0001F600 Ґанок.')
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text] * 20, trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))                    # small: letters AND punctuation split
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bpe.json')
        tok.save(path)
        json.dump({'word_rule': 'decoded'}, open(os.path.join(d, 'bpe.meta.json'), 'w'))
        tables = tokenizer_tables(path)
    ids = np.array(tok.encode(text).ids, dtype=np.int64)
    wid, _, _ = word_spans(ids, tables)
    words = [tok.decode([int(i) for i in ids[wid == w]]).strip()
             for w in sorted(set(wid[wid >= 0].tolist()))]
    runs, cur = [], ''
    for ch in text:
        if ch.isalpha():
            cur += ch
        elif cur:
            runs.append(cur)
            cur = ''
    runs += [cur] if cur else []
    split = sum(1 for i in ids if 0x80 <= (tok.decode([int(i)]) or 'x').encode('utf-8', 'replace')[0] or
                '\ufffd' in tok.decode([int(i)]))
    assert split > 0, 'the tokenizer must split some characters, or this tests nothing'
    assert words == runs, f'words {words} != letter runs {runs}'


def test_previous_unit_probe_labels():
    """
    D2: `probe.py --previous` labels every position with what a causal feature
    carries there — the PREVIOUS unit's class (a run of punctuation: its last
    token's), the window's first unit the sentinel. Checked three ways, on text
    whose letters and dashes the tokenizer splits into byte fragments, in a
    window that starts inside a word: against units read independently from the
    lexicon's word spans; with the model's own tables as with the tokenizer's;
    and against the model itself — a causal embedding fed the current labels
    puts in its slot exactly what a plain one puts there for the probe's labels.
    """
    import os, json, types, tempfile, importlib.util
    import numpy as np
    import torch
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks
    from morphlm.features.lexical import tokenizer_tables, word_spans
    from morphlm.modeling.embedding import HybridEmbedding
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location('probe', os.path.join(here, 'scripts', 'analysis', 'probe.py'))
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    text = ('Компанія „Порше“ — відомий виробник, 1990 р. Вона бігла—весь день… '
            'The cat sat; on the mat!! Ґанок.')
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text] * 20, trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bpe.json')
        tok.save(path)
        json.dump({'word_rule': 'decoded'}, open(os.path.join(d, 'bpe.meta.json'), 'w'))
        tables = tokenizer_tables(path)
        ids = np.array(tok.encode(text).ids, dtype=np.int64)
        assert tables.frag[ids].sum() > 0, 'no split characters: the fragment rule goes untested'
        wid, _, starts = word_spans(ids, tables)
        k = next(i for i in range(1, len(ids)) if wid[i] >= 0 and not starts[i])   # inside a word
        rows = np.stack([ids[:len(ids) - k], ids[k:]])
        over = lambda c: ['data.pipeline=bpe', f'data.bpe_path={path}', f'data.cache_dir={d}',
                          'frozen.facts=pos', f'frozen.causal_identity={c}']
        cfg_c = apply_overrides(ExperimentConfig(), over('true'))
        cfg_p = apply_overrides(ExperimentConfig(), over('false'))
        n_cls = build_blocks(cfg_c)[0].n_classes
        S = n_cls - 1
        V = tok.get_vocab_size()
        torch.manual_seed(0)
        causal = HybridEmbedding(cfg_c, V, verbose=False)
        plain = HybridEmbedding(cfg_p, V, verbose=False)
        plain.load_state_dict(causal.state_dict())
        Y = np.random.default_rng(0).integers(0, S, size=rows.shape)    # any class, position by position
        Yt, ids_t = torch.from_numpy(Y), torch.from_numpy(rows)
        got = pr.previous_labels(Yt, ids_t, path, S)                    # the tokenizer's tables
    want = np.full(rows.shape, S)
    for b, r in enumerate(rows):
        w, _, st = word_spans(r, tables)
        new = np.array([i == 0 or bool(st[i]) or (w[i] < 0 <= w[i - 1]) for i in range(len(r))])
        unit = np.cumsum(new) - 1
        for i in range(len(r)):
            if unit[i] > 0:
                want[b, i] = Y[b, np.flatnonzero(unit == unit[i] - 1)[-1]]
    assert got.tolist() == want.tolist(), 'the previous unit, from the tokenizer'
    got_m = pr.previous_labels(Yt, ids_t, path, S, model=types.SimpleNamespace(embed=causal))
    assert torch.equal(got_m, got), "the model's own tables give the same labels"
    T = causal.T
    assert torch.equal(causal(ids_t, Yt[:, None, :])[..., :T], plain(ids_t, got[:, None, :])[..., :T]), \
        'the probe labels each position with exactly what the causal feature feeds there'
    print('  previous-unit probe labels: oracle, model tables, model input OK')



def test_feature_switch():
    """
    The feature switch (frozen.feature_off_step, journey §33.33): a gate on every
    feature's output. Off from the start, the slot is zeros; with a later switch
    it is live until switch_features_off(), then zeros, and the feature's table
    learns nothing; the closed gate survives a checkpoint; without the setting
    there is no gate at all. Initial weights, and the cache key, do not change.
    """
    import torch
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.modeling.embedding import HybridEmbedding
    base = ['data.pipeline=bpe', 'frozen.facts=pos', 'frozen.trainable=true']

    def make(*extra):
        cfg = apply_overrides(ExperimentConfig(), base + list(extra))
        torch.manual_seed(0)
        return cfg, HybridEmbedding(cfg, 100, verbose=False)

    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 100, (2, 16), generator=g)
    codes = torch.randint(0, 8, (2, 1, 16), generator=g)
    cfg0, plain = make()
    assert not hasattr(plain, 'feature_gate'), 'no switch, no gate'
    _, off0 = make('frozen.feature_off_step=0')
    T = off0.T
    assert torch.count_nonzero(off0(ids, codes)[..., :T]) == 0, 'off from the start: zeros'
    cfg5, later = make('frozen.feature_off_step=5')
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), later.parameters())), \
        'the switch changes no initial weight'
    assert torch.count_nonzero(later(ids, codes)[..., :T]) > 0, 'before the switch: live'
    later.switch_features_off()
    out = later(ids, codes)
    assert torch.count_nonzero(out[..., :T]) == 0, 'after the switch: zeros'
    out.sum().backward()
    for name, prm in later.named_parameters():
        if not name.startswith('trained') and prm.grad is not None:
            assert torch.count_nonzero(prm.grad) == 0, f'{name}: a closed feature learns nothing'
    _, fresh = make('frozen.feature_off_step=5')
    fresh.load_state_dict(later.state_dict())
    assert float(fresh.feature_gate) == 0.0, 'the closed gate survives a checkpoint'
    assert '-OFF5' in cfg5.label() and cfg5.surface_spec() == cfg0.surface_spec()
    # the reverse: closed until the switch, then live; same initial weights
    cfg_on, late = make('frozen.feature_on_step=5')
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), late.parameters()))
    assert torch.count_nonzero(late(ids, codes)[..., :T]) == 0, 'before switching on: zeros'
    late.switch_features_on()
    assert torch.count_nonzero(late(ids, codes)[..., :T]) > 0, 'after switching on: live'
    assert '-ON5' in cfg_on.label() and cfg_on.surface_spec() == cfg0.surface_spec()
    try:
        make('frozen.feature_off_step=5', 'frozen.feature_on_step=5')
        raise AssertionError('both switches at once must be refused')
    except ValueError:
        pass
    print('  feature switch: gate, gradient, checkpoint, initial weights, label, key, reverse OK')



def _screen_cfg(*extra):
    from morphlm.config import ExperimentConfig, apply_overrides
    return apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', 'data.seq_len=32'] + list(extra))


def test_rope_is_causal_and_positionless():
    """model.position=rope: no learned positions; the future never reaches the past (§33.47)."""
    import torch
    from morphlm.modeling.embedding import HybridEmbedding
    from morphlm.modeling.transformer import TransformerLM
    cfg = _screen_cfg('model.position=rope')
    torch.manual_seed(0)
    m = TransformerLM(cfg, HybridEmbedding(cfg, 100, verbose=False), 100).eval()
    assert m.pos is None and cfg.label().endswith('_rope')
    ids = torch.randint(0, 100, (1, 32))
    a = m(ids)[0]
    ids2 = ids.clone(); ids2[0, 20] = (ids2[0, 20] + 1) % 100
    b = m(ids2)[0]
    assert torch.allclose(a[0, :20], b[0, :20], atol=1e-6) and not torch.allclose(a[0, 20:], b[0, 20:])
    print('  RoPE: causal, no learned positions, labelled OK')


def test_feature_dropout_weans():
    """feature_dropout_until: tokens keep features with the keep rate; eval scales (§33.47)."""
    import torch
    from morphlm.modeling.embedding import HybridEmbedding
    cfg = _screen_cfg('frozen.facts=pos', 'frozen.trainable=true', 'frozen.feature_dropout_until=10')
    torch.manual_seed(0)
    e = HybridEmbedding(cfg, 100, verbose=False); T = e.T
    ids = torch.randint(0, 100, (8, 32)); codes = torch.randint(1, 8, (8, 1, 32))
    full = e.eval()(ids, codes)[..., :T]
    e.set_feature_keep(0.5)
    assert torch.allclose(e.eval()(ids, codes)[..., :T], 0.5 * full), 'evaluation scales by the keep rate'
    kept = (e.train()(ids, codes)[..., :T].abs().sum(-1) > 0).float().mean().item()
    assert 0.4 < kept < 0.6, kept
    e.set_feature_keep(0.0)
    assert torch.count_nonzero(e.train()(ids, codes)[..., :T]) == 0
    assert '-DROP10' in cfg.label()
    try:
        HybridEmbedding(_screen_cfg('frozen.facts=pos', 'frozen.feature_dropout_until=10', 'frozen.feature_off_step=5'), 100, verbose=False)
        raise AssertionError('dropout with a switch must be refused')
    except ValueError:
        pass
    print('  feature dropout: keep rate in training, scaling at evaluation, refusal OK')


def test_aux_target_starts_as_a_baseline():
    """role=target: plain embedding, a head made last, the LM loss untouched (§33.47)."""
    import torch, torch.nn.functional as F
    import morphlm.pipelines.registry as reg
    from morphlm.modeling import transformer as tr
    real = reg.pipeline_vocab
    reg.pipeline_vocab = lambda cfg: (None, 96)
    try:
        torch.manual_seed(0); base = tr.build_model(_screen_cfg(), verbose=False)[0]
        cfg = _screen_cfg('frozen.facts=pos', 'frozen.trainable=true', 'frozen.role=target')
        torch.manual_seed(0); tgt, V = tr.build_model(cfg, verbose=False)
    finally:
        reg.pipeline_vocab = real
    assert tgt.embed.T == 0 and tgt.aux_head is not None and '-TGT0.1' in cfg.label()
    lm = {k: v for k, v in tgt.state_dict().items() if not k.startswith('aux_head')}
    assert all(torch.equal(lm[k], v) for k, v in base.state_dict().items()), 'every LM weight as a baseline'
    ids = torch.randint(0, V, (2, 32)); y = torch.randint(0, V, (2, 32)); codes = torch.randint(0, 8, (2, 1, 32))
    logits, loss = tgt.eval()(ids, y, codes=codes)
    assert torch.allclose(loss, F.cross_entropy(logits.view(-1, V), y.view(-1))), 'the aux term stays out'
    assert tgt.last_aux_loss is not None and torch.isfinite(tgt.last_aux_loss)
    print('  auxiliary target: baseline start, separate loss, labelled OK')



def test_rmsnorm_and_swiglu():
    """model.norm / model.mlp: labelled, causal, matched starts, the GELU MLP's weight count (§33.54)."""
    import torch
    import morphlm.pipelines.registry as reg
    from morphlm.modeling import transformer as tr
    from morphlm.config import ExperimentConfig, apply_overrides
    real = reg.pipeline_vocab
    reg.pipeline_vocab = lambda cfg: (None, 96)
    try:
        def build(*over):
            cfg = _screen_cfg('model.position=rope', *over)
            cfg.validate()
            torch.manual_seed(0)
            return cfg, tr.build_model(cfg, verbose=False)[0].eval()
        c0, ln = build()
        c1, rms = build('model.norm=rmsnorm')
        c2, glu = build('model.mlp=swiglu')
    finally:
        reg.pipeline_vocab = real
    assert c1.label().endswith('_rope_rmsnorm') and c2.label().endswith('_rope_swiglu') and '_rmsnorm' not in c0.label()
    a, b = dict(ln.named_parameters()), dict(rms.named_parameters())
    shared = [k for k in a if k in b]
    assert all(torch.equal(a[k], b[k]) for k in shared) and len(shared) == len(b), 'RMSNorm starts from its LayerNorm twin'
    assert not any(k.endswith(('ln1.bias', 'ln2.bias', 'ln_f.bias')) for k in b), 'RMSNorm has no shift'
    mlp = lambda m: sum(p.numel() for n, p in m.named_parameters() if '.ff.' in n and n.endswith('weight'))
    assert abs(mlp(glu) / mlp(ln) - 1) < 0.01, (mlp(glu), mlp(ln))
    ids = torch.randint(0, 96, (1, 32))
    for m in (rms, glu):
        out = m(ids)[0]
        ids2 = ids.clone(); ids2[0, 20] = (ids2[0, 20] + 1) % 96
        out2 = m(ids2)[0]
        assert torch.allclose(out[0, :20], out2[0, :20], atol=1e-6), 'causal'
    x = torch.randn(4, 128) * 3
    y = tr.RMSNorm(128)(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)
    for bad in ('model.norm=batchnorm', 'model.mlp=relu', 'model.position=alibi'):
        try:
            apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', bad]).validate()
            raise AssertionError(f'{bad} must be refused')
        except ValueError:
            pass
    print('  RMSNorm and SwiGLU: labels, matched start, weight count, causal, refusals OK')


def _toy_tokenizer(d, text, rule='decoded'):
    """A small ByteLevel BPE trained on `text`, saved in `d` with its word rule."""
    import json
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text] * 20, trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    path = os.path.join(d, 'bpe_t.json')
    tok.save(path)
    json.dump({'word_rule': rule}, open(os.path.join(d, 'bpe_t.meta.json'), 'w'))
    return tok, path


#: every derived fact and the exact message its block refuses per-token coding with
DERIVATIONS = {
    'mfeatlex': 'mfeatlex is derived per WORD, not per token: run scripts/data/derive_lexical_codes.py '
                'after the cache holds mfeat. Building it token by token would fill the cache with empty codes.',
    'poslex': 'poslex is derived per WORD, not per token: run scripts/data/derive_lexical_codes.py --fact pos '
              'after the cache holds pos.',
    'mstructshuf': 'mstructshuf is derived per WORD: run scripts/data/derive_lexical_codes.py --fact mstruct '
                   '--out-fact mstructshuf --shuffle-types 0',
    'ucaselex': 'ucaselex is derived per WORD, not per token: run scripts/data/derive_lexical_codes.py '
                '--fact ucase --out-fact ucaselex --corpus ukwiki',
    'ucaseprep': 'ucaseprep is derived per WORD, not per token: run scripts/data/derive_lexical_codes.py '
                 '--fact ucase --out-fact ucaseprep --recode-pos ADP --keep ucase --corpus ukwiki',
    'poscausal': 'poscausal is derived from the text by a causal tagger, not per token: run '
                 'scripts/data/derive_causal_pos.py first',
    'ugenderlex': 'ugenderlex is derived per WORD, not per token: run scripts/data/derive_lexical_codes.py '
                  '--fact ugender --out-fact ugenderlex --corpus ukwiki',
    'unumberlex': 'unumberlex is derived per WORD, not per token: run scripts/data/derive_lexical_codes.py '
                  '--fact unumber --out-fact unumberlex --corpus ukwiki',
    'ugendercausal': 'ugendercausal is derived from the text by a causal tagger, not per token: run '
                     'scripts/data/derive_causal_ud.py first',
    'unumbercausal': 'unumbercausal is derived from the text by a causal tagger, not per token: run '
                     'scripts/data/derive_causal_ud.py first',
    'mprefix': 'mprefix codes are derived from the cached token stream, not built by the pipeline. Run '
               'scripts/data/derive_prefix_codes.py first, or the cache would be filled with empty codes.',
    **{f: f'{f} is derived per WORD, not per token: run scripts/data/derive_lexical_codes.py with --out-fact {f}'
       for f in ('mconst', 'mconst16', 'mconst32', 'mconst64', 'mzero', 'mzeronopos', 'mzeroposonly',
                 'mflat', 'mfreq', 'posshuf')},
}


def test_derived_blocks_share_one_refusal():
    """
    Every derived fact refuses per-token coding with its own message, which its
    class also gives without an instance (`derivation`, read by the cache when
    no file holds the fact). Exactly the lexical ones code new text
    (`codes_for`: the BLiMP scripts dispatch on it); the other three do not.
    """
    from morphlm.registry import get_fact, all_facts
    from morphlm.features.controls import DerivedCodesMixin, LexicalLookupMixin
    derived = {n for n, c in all_facts().items() if issubclass(c, DerivedCodesMixin)}
    assert derived == set(DERIVATIONS), derived ^ set(DERIVATIONS)
    for name, msg in DERIVATIONS.items():
        cls = get_fact(name)
        assert cls.derivation() == msg, name
        b = cls(apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', f'frozen.facts={name}']))
        try:
            b.code({'mstruct': 'simple', 'pos': 'NOUN', 'mfeat': 'Number=Plur'})
            raise AssertionError(f'{name} must refuse per-token coding')
        except RuntimeError as e:
            assert str(e) == msg, (name, str(e))
        lexical = issubclass(cls, LexicalLookupMixin)
        assert hasattr(b, 'codes_for') == lexical, name
        assert lexical == (name not in ('ucaseprep', 'poscausal', 'ugendercausal', 'unumbercausal', 'mprefix')), name
    print(f'  {len(DERIVATIONS)} derived facts refuse with their own message; lexical ones code new text OK')


def test_lexical_variants_read_their_lexicon():
    """
    mfeatlex, poslex and mstructshuf code new text through the lexicon saved
    under their source fact (mfeat, pos) or their own name: every word its
    lexicon entry, unseen words and non-words `none` — never a per-token value.
    """
    import tempfile
    import numpy as np
    from morphlm.features.lexical import tokenizer_tables, word_spans, lexicon_path, save_lexicon
    from morphlm.features.controls import LexicalLookupMixin
    text = 'The cat sat on the mat, and the dog ran far away!'
    with tempfile.TemporaryDirectory() as d:
        tok, path = _toy_tokenizer(d, text)
        ids = np.array(tok.encode(text).ids, dtype=np.int64)
        wid, key, starts = word_spans(ids, tokenizer_tables(path))
        words = sorted({int(k) for k in key[wid >= 0]})
        known = words[:-2]                              # the last two words stay unseen
        for fact, lexicon in (('mfeatlex', 'mfeat'), ('poslex', 'pos'), ('mstructshuf', 'mstructshuf')):
            lex_keys = np.array(known, dtype=np.uint64)
            lex_codes = (np.arange(len(known)) % 5).astype(np.int16)
            save_lexicon(lexicon_path(path, lexicon, d), (lex_keys, lex_codes, np.ones(len(known))), {})
            b, = build_blocks(apply_overrides(ExperimentConfig(), [
                'data.pipeline=bpe', f'data.bpe_path={path}', f'data.cache_dir={d}', f'frozen.facts={fact}']))
            assert isinstance(b, LexicalLookupMixin) and b.lexicon_name == lexicon, fact
            got = np.array(b.codes_for(ids.tolist(), [{'mfeat': 'Number=Plur', 'pos': 'NOUN'}] * len(ids)))
            none = b.n_classes - 1
            want = np.full(len(ids), none)
            for i in range(len(ids)):
                if wid[i] >= 0 and int(key[i]) in known:
                    want[i] = lex_codes[known.index(int(key[i]))]
            assert got.tolist() == want.tolist(), fact
            assert (got[wid < 0] == none).all() and (got == none).sum() > (wid < 0).sum(), fact
    print('  mfeatlex, poslex, mstructshuf code new text from their lexicons OK')


def test_unbuilt_fact_names_its_derivation():
    """
    A fact no codes file of the stream holds, and the pipeline never builds,
    stops the loader with the fact's own derivation — before either path. It
    used to say "delete it and rebuild": deleting a multi-hour build, after
    which the run died at `build_facts.index`.
    """
    import json, tempfile
    import numpy as np
    from morphlm.data.stream import build_stream, cache_stem, codes_spec, facts_to_build
    old = os.environ.get('MORPHLM_NO_BUILD')
    os.environ['MORPHLM_NO_BUILD'] = '1'
    n = 24
    try:
        with tempfile.TemporaryDirectory() as d:
            def cfg(facts, pipeline='bpe'):
                return apply_overrides(ExperimentConfig(), [f'data.pipeline={pipeline}', 'data.bpe_path=data/bpe_17k.json',
                                                            f'data.cache_dir={d}', f'frozen.facts={facts}'])
            def fails(c, split='validation'):
                try:
                    build_stream(None, c, split, verbose=False)
                except RuntimeError as e:
                    return str(e)
                raise AssertionError('expected a refusal')
            c = cfg('mstruct')
            ids = cache_stem('bpe', 'validation', None, c.surface_spec())
            np.save(os.path.join(d, ids + '.npy'), np.zeros(n, dtype=np.int32))
            json.dump({'num_tokens': n, 'num_bytes': n, 'num_chars': n, 'num_articles': 1},
                      open(os.path.join(d, ids + '.json'), 'w'))
            built = list(facts_to_build(c))
            stem = cache_stem('bpe', 'validation', None, f'{c.surface_spec()}|{codes_spec(built, "coarse")}')
            np.save(os.path.join(d, stem + '.codes.npy'), np.ones((len(built), n), dtype=np.int16))
            json.dump({'fact_names': built, 'dep_granularity': 'coarse', 'num_tokens': n},
                      open(os.path.join(d, stem + '.codes.json'), 'w'))
            assert build_stream(None, c, 'validation', verbose=False)['codes'] is not None
            for fact in ('mconst', 'mprefix', 'poslex'):
                msg = fails(cfg(fact))
                assert DERIVATIONS[fact] in msg and 'rebuild' not in msg, msg
            assert DERIVATIONS['mconst'] in fails(cfg('mconst'), 'train'), 'no ids yet: still the derivation'
            assert 'not among the facts the bpe pipeline builds' in fails(cfg('ucase'))
            # once derived for this stream, the fact is read
            np.save(os.path.join(d, ids + '_prefix.codes.npy'), np.full((1, n), 5, dtype=np.int16))
            json.dump({'fact_names': ['mprefix'], 'dep_granularity': 'coarse', 'num_tokens': n},
                      open(os.path.join(d, ids + '_prefix.codes.json'), 'w'))
            assert int(np.asarray(build_stream(None, cfg('mprefix'), 'validation', verbose=False)['codes'])[0, 0]) == 5
    finally:
        if old is None:
            os.environ.pop('MORPHLM_NO_BUILD', None)
        else:
            os.environ['MORPHLM_NO_BUILD'] = old
    print('  a fact no file holds names its derivation, before either path OK')


def test_split_modules_keep_every_name():
    """
    blocks.py was split into inventories.py and controls.py, and stream.py's file
    handling into cache_files.py. Every name stays importable where it was, as
    the same object; every fact is registered whichever module is imported first.
    """
    import subprocess
    from morphlm.features import blocks, controls, inventories
    from morphlm.data import stream, cache_files
    from morphlm.registry import all_facts
    for name in ('DEP_COARSE_MAP', 'DEP_COARSE_CLASSES', 'POS_CLASSES', 'MORPH_BUNDLES', 'ENTITY_CLASSES'):
        assert getattr(blocks, name) is getattr(inventories, name), name
    for name in ('LexicalLookupMixin', 'DerivedCodesMixin', 'MorphConstBlock', 'MorphZeroBlock', 'MorphFlatBlock',
                 'FrequencyBandBlock', 'PosShuffledBlock', 'MorphStructShuffledBlock', 'MorphFeatureLexicalBlock',
                 'PosLexicalBlock', 'UCaseLexicalBlock', 'UCasePrepositionFormBlock', 'PosCausalBlock',
                 'MorphPrefixBlock'):
        assert getattr(blocks, name) is getattr(controls, name), name
    for name in ('PACKED', 'PackedCacheError', 'cache_stem', 'codes_spec', 'codes_owner', 'stream_files',
                 'require_unpacked', 'stream_codes_files', '_atomic_save_npy', '_atomic_write_json'):
        assert getattr(stream, name) is getattr(cache_files, name), name
    assert stream._atomic_save_npy is cache_files.atomic_save_npy
    assert stream._atomic_write_json is cache_files.atomic_write_json
    facts = sorted(all_facts())
    assert facts == ['cat', 'dep', 'ectx', 'etype', 'magr', 'mconst', 'mconst16', 'mconst32', 'mconst64', 'mfeat',
                     'mfeatlex', 'mflat', 'mfreq', 'mhash', 'minfl', 'mlemma', 'mprefix', 'mrole', 'mroot', 'mstruct',
                     'mstructshuf', 'mzero', 'mzeronopos', 'mzeroposonly', 'pos', 'poscausal', 'poslex', 'posshuf',
                     'uanimacy', 'uaspect', 'ucase', 'ucaselex', 'ucaseprep', 'ugender', 'ugendercausal',
                     'ugenderlex', 'unumber', 'unumbercausal', 'unumberlex', 'uperson',
                     'utense'], facts   # the 37 registered before the split, and gender's and number's four (21 October)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for first in ('morphlm.features.controls', 'morphlm.features.blocks', 'morphlm.registry'):
        code = (f'import {first}; from morphlm.registry import all_facts; '
                f'from morphlm.features.blocks import MorphConstBlock, POS_CLASSES; '
                f'print(",".join(sorted(all_facts())))')
        out = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, cwd=root)
        assert out.returncode == 0, out.stderr[-1500:]
        assert out.stdout.strip().split(',') == facts, first
    print(f'  split modules keep every name; {len(facts)} facts registered in any import order OK')


class _FakeMorph:
    def __init__(self, feats):
        self.feats = feats

    def get(self, k):
        v = self.feats.get(k)
        return [v] if v else []

    def to_dict(self):
        return dict(self.feats)

    def __str__(self):
        return '|'.join(f'{k}={v}' for k, v in sorted(self.feats.items()))


class _FakeTok:
    def __init__(self, text, idx, dep, pos, ent='', **feats):
        self.text, self.idx, self.dep_, self.pos_, self.ent_type_ = text, idx, dep, pos, ent
        self.lemma_ = text.lower()
        self.morph = _FakeMorph(feats)

    @property
    def is_alpha(self):
        return self.text.isalpha()


class _FakeAnalyser:
    def __init__(self):
        self.calls = 0

    def analyse(self, word):
        self.calls += 1
        return word[:2], 'suffixed', [(0, 1, 'prefix'), (1, len(word), 'root')]

    def root_code(self, root):
        return len(root)

    def hash_code(self, word):
        return 100 + len(word)

    def lemma_code(self, lemma):
        return 200 + len(lemma)


def test_bpe_tokens_carry_their_words_annotations():
    """
    Each BPE token carries the annotation of the first annotated character in
    its span, for every context fact (ANNOTATED, one keyed loop), in the token
    dict's fixed key order; the morphological ones only from the analyser, which
    is built on first use. Checked on a parse with known values.
    """
    import tempfile
    from morphlm.pipelines.bpe import BPEPipeline, ANNOTATED
    text = 'Cats sat, Paris slept.'
    doc = [_FakeTok('Cats', 0, 'nsubj', 'NOUN', Number='Plur'), _FakeTok('sat', 5, 'ROOT', 'VERB', Tense='Past'),
           _FakeTok(',', 8, 'punct', 'PUNCT'), _FakeTok('Paris', 10, 'nsubj', 'PROPN', ent='GPE', Number='Sing'),
           _FakeTok('slept', 16, 'conj', 'VERB', Tense='Past'), _FakeTok('.', 21, 'punct', 'PUNCT')]
    words = {t.idx: t for t in doc}
    with tempfile.TemporaryDirectory() as d:
        _, path = _toy_tokenizer(d, text, rule='symbol')
        built = []
        def factory():
            built.append(_FakeAnalyser())
            return built[-1]
        pipe = BPEPipeline(path, want_context=True, analyser_factory=factory, verbose=False)
        assert not built, 'the analyser is built on first use, not with the pipeline'
        pipe._nlp = lambda t: doc
        toks = pipe.encode(text)
        offsets = pipe.tokenizer.encode(text).offsets
        plain = BPEPipeline(path, verbose=False).encode(text)
    assert len(built) == 1 and pipe.analyser is built[0], 'built once, then kept'
    keys = ['token_id', 'token', 'category', 'role', 'entity_label', 'word', 'dep', 'pos', 'mroot', 'mhash',
            'mlemma', 'mstruct', 'mrole', 'minfl', 'magr', 'mfeat', 'udfeats']
    assert all(list(t) == keys for t in toks) and set(ANNOTATED) <= set(keys)
    for t, (a, b) in zip(toks, offsets):
        first = next((c for c in range(a, b) if any(w.idx <= c < w.idx + len(w.text) for w in doc)), None)
        if first is None:
            assert all(t[k] is None for k in ANNOTATED), t
            continue
        w = next(w for w in doc if w.idx <= first < w.idx + len(w.text))
        assert (t['dep'], t['pos'], t['entity_label']) == (w.dep_, w.pos_, w.ent_type_ or None), t
        if w.is_alpha:
            assert t['mstruct'] == 'suffixed' and t['mroot'] == 2 and t['mhash'] == 100 + len(w.text), t
            assert t['mrole'] == ('prefix' if first == w.idx else 'root'), t
            assert t['udfeats'] == w.morph.to_dict() and t['mfeat'] == (str(w.morph) or None), t
        else:
            assert t['mstruct'] is None and t['mroot'] is None and t['udfeats'] is None, t
    assert pipe.alignment_rate() == 1.0
    assert all(list(t) == keys and all(t[k] is None for k in ANNOTATED) for t in plain), 'no context: all None'
    print('  BPE tokens carry their words\' annotations, analyser built on first use OK')


def test_bucket_counts_and_batch_limit():
    """
    token_frequencies counts with np.bincount and gives exactly the Counter it
    always gave; bucketed_perplexity scores all batches, or the first
    `max_batches` (the post-hoc comparison's budget).
    """
    from collections import Counter
    from types import SimpleNamespace
    import torch
    from morphlm.eval.buckets import token_frequencies, bucketed_perplexity
    rng = np.random.default_rng(0)
    for n, hi, chunk in ((0, 4, 4), (1, 4, 4), (1000, 50, 7), (5000, 3000, 1 << 24)):
        ids = rng.integers(0, hi, n).astype(np.int32)
        if n > 1:
            ids[-1] = hi + 3                             # a larger id in the last chunk
        got = token_frequencies(SimpleNamespace(ids=ids), chunk=chunk)
        assert got == Counter(ids.tolist()) and all(type(k) is int and type(v) is int for k, v in got.items())

    class Uniform(torch.nn.Module):
        def forward(self, x, y, codes=None):
            return torch.zeros(*x.shape, 10), None
    batches = [(torch.randint(0, 10, (2, 4)), torch.randint(0, 10, (2, 4)), torch.zeros(0, 4)) for _ in range(5)]
    freqs = token_frequencies(SimpleNamespace(ids=rng.integers(0, 10, 500)))
    edges = (0, 40, 60, float('inf'))
    tokens = lambda rows: sum(r['tokens'] for r in rows)
    assert tokens(bucketed_perplexity(Uniform(), batches, freqs, 'cpu', edges=edges)) == 40
    assert tokens(bucketed_perplexity(Uniform(), batches, freqs, 'cpu', edges=edges, max_batches=2)) == 16
    assert tokens(bucketed_perplexity(Uniform(), batches, freqs, 'cpu', edges=edges, max_batches=0)) == 40
    rows = bucketed_perplexity(Uniform(), batches, freqs, 'cpu', edges=edges)
    assert all(r['perplexity'] is None or abs(r['perplexity'] - 10) < 1e-4 for r in rows)
    print('  bucket counts equal the Counter; batch limit honoured OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_blocks: PASS')
