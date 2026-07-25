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


if __name__ == '__main__':
    test_backward_compatible_stems()
    test_feature_spec_changes_key()
    test_training_knobs_absent()
    print('test_cache_keys: PASS')
