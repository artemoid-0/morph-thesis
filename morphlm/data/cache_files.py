"""
cache_files.py — the token cache's files: their names, which of them describe a
stream, whether a stream is packed, and writes no reader sees half done.

`stream.py` builds and reads the streams and re-exports every name here, so
`from morphlm.data.stream import cache_stem` and the rest keep working.
"""

from __future__ import annotations

import os
import json
import glob
import hashlib

import numpy as np


# ── names ─────────────────────────────────────────────────────────────────────

def cache_stem(pipeline_name, split, max_articles, feature_spec=''):
    """
    Stable filename stem. With an empty feature_spec this reproduces the
    historical format byte for byte — a REQUIREMENT, not a nicety.
    """
    raw = f'{pipeline_name}|{split}|{max_articles}'
    if feature_spec:
        raw += f'|{feature_spec}'
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f'{pipeline_name}_{split}_{max_articles}_{h}'


def codes_spec(fact_names, dep_granularity):
    """Canonical spec string for a set of context facts."""
    if not fact_names:
        return ''
    return f'ctx:{dep_granularity}:{"+".join(sorted(fact_names))}'


def codes_owner(basename, prefix):
    """
    What a codes file's name says about where it came from (G-72).

    A cache build writes `<prefix><hash>.codes.json`, keyed on the facts it
    built; a derivation (derive_lexical_codes, derive_prefix_codes) writes
    `<ids stem>_<name>.codes.json`, named for the ids stream it was derived for.
    Returns ('build', None), ('derived', <ids stem>), or (None, None) for a name
    of neither form (no tool writes one).
    """
    name = basename[:-len('.codes.json')] if basename.endswith('.codes.json') else basename
    if not name.startswith(prefix):
        return None, None
    rest = name[len(prefix):]
    h, tail = rest[:10], rest[10:]
    if len(h) != 10 or any(c not in '0123456789abcdef' for c in h):
        return None, None
    if not tail:
        return 'build', None
    if tail.startswith('_'):
        return 'derived', prefix + h
    return None, None


# ── which codes files describe a stream ───────────────────────────────────────

def _described_codes(cache_dir, prefix, ids_stem, num_tokens):
    """
    The codes files that describe the ids stream `ids_stem`, sorted by name, as
    (json path, npy path, meta) — whether or not the array is on disk.

    A codes array is one value per token of ONE ids array, so a matching fact
    list is not enough: a build file must have the stream's token count, or it
    may describe another surface configuration's tokens (tests/test_cache_keys).
    Nor is the token count enough: a corpus's validation split is the same
    tokens at every corpus fraction, while a derived file is built from its own
    fraction's train split, so a derived file describes only the stream it was
    derived for (G-72). A name of neither form keeps the token-count rule.
    """
    for cand in sorted(glob.glob(os.path.join(cache_dir, prefix + '*.codes.json'))):
        kind, owner = codes_owner(os.path.basename(cand), prefix)
        if kind == 'derived' and owner != ids_stem:
            continue
        try:
            with open(cand) as f:
                cm = json.load(f)
        except (OSError, ValueError):
            continue
        if cm.get('num_tokens') != num_tokens:
            continue
        yield cand, cand[:-len('.json')] + '.npy', cm


#: A machine may keep a stream packed: every array of it — its ids and each codes
#: file that describes them — as `<name>.npy.zst` beside the sidecar `.json`, made
#: and checked by scripts/data/cache_pack.py, the whole stream at once. Nothing
#: reads a packed array and nothing rebuilds one: whatever needs a packed stream
#: stops and names it. A cache that holds no packed file behaves as it always did.
PACKED = '.zst'


class PackedCacheError(RuntimeError):
    """A stream that is needed is packed on this machine (scripts/data/cache_pack.py)."""


def stream_files(cache_dir, ids_stem, packed=False):
    """
    The arrays of the stream `ids_stem`: its ids and every codes file that
    describes it (`_described_codes`, at the token count of its sidecar). Those
    on disk unpacked, or with packed=True those packed; each path is given as
    its `.npy` name.
    """
    sfx = '.npy' + (PACKED if packed else '')
    ids_npy = os.path.join(cache_dir, ids_stem + '.npy')
    out = [ids_npy] if os.path.exists(ids_npy[:-4] + sfx) else []
    try:
        with open(os.path.join(cache_dir, ids_stem + '.json')) as f:
            num_tokens = json.load(f).get('num_tokens')
    except (OSError, ValueError):
        num_tokens = None
    out += [npy for _, npy, _ in _described_codes(cache_dir, ids_stem[:-10], ids_stem, num_tokens)
            if os.path.exists(npy[:-4] + sfx)]
    return out


def require_unpacked(cache_dir, ids_stem):
    """Stop when the stream `ids_stem` is packed on this machine, in whole or in
    part: an array packed and not also unpacked is never read, and never rebuilt."""
    held = sorted(set(stream_files(cache_dir, ids_stem, packed=True)) - set(stream_files(cache_dir, ids_stem)))
    if held:
        raise PackedCacheError(
            f'{ids_stem} is packed on this machine ({len(held)} of its arrays), so nothing reads or '
            f'rebuilds it. Unpack the stream first: python scripts/data/cache_pack.py unpack {ids_stem}')


def stream_codes_files(cache_dir, pipeline, split, max_articles, ids_stem, num_tokens):
    """
    Every codes file on disk that describes the ids stream `ids_stem`
    (`_described_codes`), sorted by name, as [(json path, npy path, meta)].
    Stops first if the stream is packed.
    """
    require_unpacked(cache_dir, ids_stem)
    prefix = f'{pipeline}_{split}_{max_articles}_'
    return [(cand, npy, cm) for cand, npy, cm in _described_codes(cache_dir, prefix, ids_stem, num_tokens)
            if os.path.exists(npy)]


# ── writes ────────────────────────────────────────────────────────────────────

def atomic_save_npy(path, arr):
    """
    Write an array so that no reader ever sees a partial file.

    Cached arrays are memory-mapped by running jobs (np.load mmap_mode='r').
    np.save on the live path truncates the file first, and a job reading it at
    that moment can die with SIGBUS or read zeros. Writing to a temporary name
    and renaming is atomic on one filesystem: a job that already mapped the old
    file keeps the old inode intact, and any later reader sees the new file
    complete. This is what makes it safe to build a cache while training.
    """
    tmp = path[:-4] + '.tmp.npy' if path.endswith('.npy') else path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    """A JSON sidecar, written the same way as `atomic_save_npy`."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# the names the derivation scripts import
_atomic_save_npy = atomic_save_npy
_atomic_write_json = atomic_write_json
