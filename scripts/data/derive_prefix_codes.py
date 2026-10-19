#!/usr/bin/env python3
"""
derive_prefix_codes.py — build `mprefix` codes from the cached token stream.

WHAT IT PRODUCES
----------------
For every position, a hash of the current word's fragments SO FAR:

    Ġrecon            -> hash("recon")
    Ġrecon fig        -> hash("recon fig")
    Ġrecon fig uring  -> hash("recon fig uring")   <- the whole word

Causal by construction: the value at position i is a function of tokens at
positions <= i and nothing later. At a word's final fragment the prefix IS the
word, so the model gets true word identity without ever needing to know the
word has ended — which is what `causal_shift` could not deliver.

WHY DERIVED RATHER THAN BUILT BY THE PIPELINE
---------------------------------------------
These codes depend only on the BPE token stream, which is already cached.
Re-running spaCy and MorphoLex over 120M tokens to produce them would cost
hours and change nothing. This reads the cached ids, computes the codes, and
writes a codes file containing the EXISTING facts plus `mprefix`, so any subset
request is still satisfiable through the loader's content-based reuse. Nothing
in the loader needs to change.

    python scripts/data/derive_prefix_codes.py --bpe-path data/bpe_17k.json
    python scripts/data/derive_prefix_codes.py --bpe-path data/bpe_17k.json --verify
"""

import os
import sys
import json
import glob
import hashlib
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

GDOT = '\u0120'          # ByteLevel marks a preceding space with this
P = 1000003              # rolling-hash multiplier
NONE_CODE = 16364        # matches WordAnalyser._none_root
MOD = NONE_CODE          # real codes land in [0, 16363]


def vocab_tables(tokenizer, size):
    """Per-id constants: is this token alphabetic, does it open a word, its hash."""
    is_alpha = np.zeros(size, dtype=bool)
    opens = np.zeros(size, dtype=bool)
    tokhash = np.zeros(size, dtype=np.int64)
    for i in range(size):
        t = tokenizer.id_to_token(i)
        if t is None:
            continue
        core = t.lstrip(GDOT)
        is_alpha[i] = bool(core) and core.isalpha()
        opens[i] = t.startswith(GDOT)
        tokhash[i] = int(hashlib.md5(t.encode('utf-8')).hexdigest()[:8], 16)
    return is_alpha, opens, tokhash


def prefix_codes(ids, is_alpha, opens, tokhash):
    """
    Rolling prefix hash, reset at every word start.

    A word starts at an alphabetic token that either carries the space marker
    or follows a non-alphabetic token — the same rule `lookahead_check` uses to
    classify targets, so the two agree about where words begin. Non-alphabetic
    positions get the sentinel, matching how the other identity facts treat
    punctuation and digits.
    """
    n = len(ids)
    alpha = is_alpha[ids]
    prev_alpha = np.empty(n, dtype=bool)
    prev_alpha[0] = False
    prev_alpha[1:] = alpha[:-1]

    starts = alpha & (opens[ids] | ~prev_alpha)
    idx = np.arange(n, dtype=np.int64)
    start_of = np.maximum.accumulate(np.where(starts, idx, -1))

    # positions before any word start, or non-alphabetic, carry no word
    valid = alpha & (start_of >= 0)
    offset = np.where(valid, idx - start_of, 0)

    out = np.full(n, NONE_CODE, dtype=np.int64)
    h = np.zeros(n, dtype=np.int64)
    max_off = int(offset.max()) if n else 0
    for k in range(max_off + 1):
        m = valid & (offset >= k)
        if not m.any():
            break
        src = start_of[m] + k
        h[m] = (h[m] * P + tokhash[ids[src]]) % MOD
    out[valid] = h[valid]

    # CAUSALITY: the fragments hashed at position i run from start_of[i] to i,
    # all at positions <= i. Asserted rather than assumed.
    assert bool((offset >= 0).all()), 'prefix reached before its word start'
    assert bool((start_of[valid] <= idx[valid]).all()), 'read a future position'
    return out.astype(np.int16), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--splits', nargs='+',
                    default=['train', 'validation', 'test'])
    ap.add_argument('--verify', action='store_true',
                    help='check the codes after writing, on a sample')
    args = ap.parse_args()

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem
    from morphlm.data.tokenization import load_tokenizer

    tok = load_tokenizer(args.bpe_path, vocab_only=True)
    size = tok.get_vocab_size()
    print(f'  tokenizer {os.path.basename(args.bpe_path)}: {size:,} tokens')
    is_alpha, opens, tokhash = vocab_tables(tok, size)
    print(f'  {int(is_alpha.sum()):,} alphabetic, {int(opens.sum()):,} '
          f'word-opening')

    stem = os.path.basename(args.bpe_path).replace('.json', '')
    spec = '' if stem == 'bpe_tokenizer' else f'tok:{stem}'

    for split in args.splits:
        ids_stem = cache_stem('bpe', split, None, spec)
        ids_path = os.path.join(args.cache_dir, ids_stem + '.npy')
        from morphlm.data.stream import require_unpacked
        require_unpacked(args.cache_dir, ids_stem)      # never a packed stream skipped as missing
        if not os.path.exists(ids_path):
            print(f'  {split}: no cached ids ({ids_stem}) — skipped')
            continue
        ids = np.load(ids_path, mmap_mode='r')
        ids = np.asarray(ids, dtype=np.int64)
        print(f'\n  {split}: {len(ids):,} tokens')

        codes, valid = prefix_codes(ids, is_alpha, opens, tokhash)
        print(f'    {int(valid.sum()):,} positions carry a word code '
              f'({100.0*valid.mean():.1f}%)')
        print(f'    {len(np.unique(codes[valid])):,} distinct prefix codes')

        # find the existing codes file for this split and append a row, so any
        # subset request (mprefix alone, or with mstruct) still resolves
        src_json = src_npy = None
        for cand in sorted(glob.glob(os.path.join(
                args.cache_dir, f'bpe_{split}_None_*.codes.json'))):
            try:
                cm = json.load(open(cand))
            except Exception:
                continue
            if cm.get('num_tokens') == len(ids) and \
                    'mprefix' not in (cm.get('fact_names') or []):
                npy = cand[:-len('.json')] + '.npy'
                if os.path.exists(npy) and (
                        src_json is None or
                        len(cm['fact_names']) >
                        len(json.load(open(src_json))['fact_names'])):
                    src_json, src_npy = cand, npy
        if src_json is None:
            print('    no existing codes file to extend — skipped')
            continue

        meta = json.load(open(src_json))
        old = np.load(src_npy, mmap_mode='r')
        names = list(meta['fact_names']) + ['mprefix']
        out_stem = ids_stem + '_prefix'
        out_npy = os.path.join(args.cache_dir, out_stem + '.codes.npy')
        out_json = os.path.join(args.cache_dir, out_stem + '.codes.json')

        arr = np.lib.format.open_memmap(
            out_npy, mode='w+', dtype=np.int16, shape=(len(names), len(ids)))
        for r in range(old.shape[0]):
            arr[r] = old[r]
        arr[-1] = codes
        arr.flush()
        json.dump({'fact_names': names,
                   'dep_granularity': meta.get('dep_granularity', 'coarse'),
                   'num_tokens': int(len(ids))}, open(out_json, 'w'), indent=2)
        print(f'    wrote {out_stem}.codes.npy with {len(names)} facts')

        if args.verify:
            _verify(ids, codes, valid, tok, is_alpha, opens)


def _verify(ids, codes, valid, tok, is_alpha, opens):
    """The same word must always produce the same code at its final fragment."""
    n = len(ids)
    alpha = is_alpha[ids]
    nxt_starts = np.zeros(n, dtype=bool)
    nxt_starts[:-1] = ~alpha[1:] | opens[ids[1:]]
    nxt_starts[-1] = True
    final = valid & nxt_starts                      # last fragment of a word

    idx = np.nonzero(final)[0][:200000]
    seen = {}
    bad = 0
    for i in idx:
        j = i
        while j > 0 and valid[j - 1] and not (
                opens[ids[j]] or not alpha[j - 1]):
            j -= 1
        key = tuple(ids[j:i + 1].tolist())
        c = int(codes[i])
        if key in seen and seen[key] != c:
            bad += 1
        seen[key] = c
    print(f'    verify: {len(seen):,} distinct words at final fragments, '
          f'{bad} inconsistent codes')
    if bad:
        raise SystemExit('the same word produced different codes — the rolling '
                         'hash is not deterministic')


if __name__ == '__main__':
    main()
