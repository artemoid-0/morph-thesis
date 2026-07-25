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


def test_spacy_native_free_ride():
    """morph/det annotate on the pass they already pay for; bpe does not."""
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import facts_to_build, ALL_CONTEXT_FACTS
    m = apply_overrides(ExperimentConfig(), ['data.pipeline=morph'])
    b = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe'])
    assert facts_to_build(m) == ALL_CONTEXT_FACTS, 'morph should free-ride'
    assert facts_to_build(b) == (), 'bpe must not pay for unrequested spaCy'
    print('  spaCy free-ride asymmetry OK')


if __name__ == '__main__':
    test_backward_compatible_stems()
    test_feature_spec_changes_key()
    test_training_knobs_absent()
    test_ids_and_codes_keyed_separately()
    test_spacy_native_free_ride()
    print('test_cache_keys: PASS')
