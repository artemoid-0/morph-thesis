"""
A packed stream is never read and never rebuilt (scripts/data/cache_pack.py).

A machine may keep a stream packed — every array as `<name>.npy.zst` beside its sidecar — while
another machine trains from the same stream unpacked, with the same code. The code must then
behave the same on both wherever the stream is unpacked, and stop where it is packed: never read
a packed stream as missing, rebuild it for hours, or borrow another file in its place.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location('cache_pack', os.path.join(ROOT, 'scripts', 'data', 'cache_pack.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_packed_stream_is_never_read_or_rebuilt():
    import json, shutil, tempfile, argparse
    import numpy as np
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import (PACKED, PackedCacheError, build_stream, cache_stem, codes_spec,
                                     facts_to_build, require_unpacked, stream_codes_files, stream_files)
    if shutil.which('zstd') is None:
        print('  zstd not installed — skipped')
        return
    cp = _tool()
    old = os.environ.get('MORPHLM_NO_BUILD')
    os.environ['MORPHLM_NO_BUILD'] = '1'          # a regression would fail, not rebuild
    try:
        with tempfile.TemporaryDirectory() as d:
            def cfg(facts=None):
                return apply_overrides(ExperimentConfig(), [
                    'data.pipeline=bpe', 'data.bpe_path=data/bpe_uk_17k.json', 'data.corpus=ukwiki',
                    'data.corpus_fraction=0.05', f'data.cache_dir={d}'] + ([f'frozen.facts={facts}'] if facts else []))
            c = cfg('mconst')
            sfc = c.surface_spec()
            stems = {}
            for split, n in (('train', 4096), ('validation', 512)):
                ids = cache_stem('bpe', split, None, sfc)
                stems[split] = ids
                rng = np.random.default_rng(len(split))
                np.save(os.path.join(d, ids + '.npy'), rng.integers(0, 17000, n).astype(np.int32))
                json.dump({'num_tokens': n, 'num_bytes': n, 'num_chars': n, 'num_articles': 1},
                          open(os.path.join(d, ids + '.json'), 'w'))
                ud = list(facts_to_build(c))
                build = cache_stem('bpe', split, None, f'{sfc}|{codes_spec(ud, "coarse")}')
                np.save(os.path.join(d, build + '.codes.npy'), rng.integers(0, 8, (len(ud), n)).astype(np.int16))
                json.dump({'fact_names': ud, 'dep_granularity': 'coarse', 'num_tokens': n},
                          open(os.path.join(d, build + '.codes.json'), 'w'))
                np.save(os.path.join(d, ids + '_mconst.codes.npy'), np.full((2, n), 3, dtype=np.int16))
                json.dump({'fact_names': ['ucase', 'mconst'], 'dep_granularity': 'coarse', 'num_tokens': n,
                           'rules': {'mconst': 'every word'}}, open(os.path.join(d, ids + '_mconst.codes.json'), 'w'))
            tr, va = stems['train'], stems['validation']
            before = build_stream(None, c, 'train', verbose=False)
            ids_before = np.array(before['ids'])
            codes_before = np.array(before['codes'])
            assert len(stream_files(d, tr)) == 3, stream_files(d, tr)        # ids, the build file, the null's

            # pack the training stream; the originals go (here deleted: a test's own files)
            def args(**kw):
                base = dict(cache_dir=d, stems=[tr], all=False, split=None, exclude=[], level=3,
                            trash=False, dry_run=False, full=True)
                base.update(kw)
                return argparse.Namespace(**base)
            cp.cmd_pack(args())
            assert len(stream_files(d, tr, packed=True)) == 3
            man = json.load(open(os.path.join(d, cp.MANIFEST)))
            assert all(os.path.basename(f) in man for f in stream_files(d, tr))
            for f in stream_files(d, tr):
                os.remove(f)
            assert cp.state(d, tr)[0] == 'packed'

            # a packed stream stops every reader, with the command that unpacks it
            for read in (lambda: build_stream(None, c, 'train', verbose=False),
                         lambda: build_stream(None, cfg(), 'train', verbose=False),
                         lambda: stream_codes_files(d, 'bpe', 'train', None, tr, 4096),
                         lambda: require_unpacked(d, tr)):
                try:
                    read()
                    raise AssertionError('a packed stream was read')
                except PackedCacheError as e:
                    assert 'cache_pack.py unpack ' + tr in str(e), e
            # the unpacked validation stream of the same tokenizer reads as before
            assert int(np.asarray(build_stream(None, c, 'validation', verbose=False)['codes'])[-1, 0]) == 3

            # a stream packed in part stops readers too
            cp.cmd_unpack(args())
            os.rename(os.path.join(d, tr + '_mconst.codes.npy'), os.path.join(d, 'aside.npy'))
            try:
                build_stream(None, c, 'train', verbose=False)
                raise AssertionError('a stream packed in part was read')
            except PackedCacheError:
                pass
            os.rename(os.path.join(d, 'aside.npy'), os.path.join(d, tr + '_mconst.codes.npy'))

            # unpacked, it reads exactly as before; dropped, it is packed again
            after = build_stream(None, c, 'train', verbose=False)
            assert np.array_equal(np.array(after['ids']), ids_before)
            assert np.array_equal(np.array(after['codes']), codes_before)
            gone = []
            cp.trash = gone.append
            cp.cmd_drop(args())
            assert sorted(gone) == sorted(stream_files(d, tr)), gone
            cp.cmd_verify(args(stems=[tr]))
            # an unpacked copy that differs from the packed one is never dropped
            np.save(os.path.join(d, tr + '.npy'), np.zeros(4096, dtype=np.int32))
            try:
                cp.cmd_drop(args())
                raise AssertionError('a changed copy was dropped')
            except SystemExit:
                pass
    finally:
        if old is None:
            os.environ.pop('MORPHLM_NO_BUILD', None)
        else:
            os.environ['MORPHLM_NO_BUILD'] = old
    print('  a packed stream is never read or rebuilt; unpacked, it reads as before OK')


if __name__ == '__main__':
    # Every test_ function defined in this file, in order: a test cannot be
    # written and then left out of a hand-kept list (journey §33.45).
    import inspect
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and inspect.isfunction(_fn) and _fn.__module__ == __name__:
            _fn()
    print('test_cache_pack: PASS')
