"""
stream.py — tokenize once, cache, memory-map.

CACHE KEY DESIGN
----------------
Two failure modes to avoid, both of which the old flat key permitted:

  1. STALENESS. The old key was (pipeline, split, max_articles). Once the
     stream carries annotation codes, that key cannot tell which annotation
     produced it — change the stamping rule, get a cache hit on the old data,
     and train on silently wrong features. No crash, no error, just an invalid
     result you might believe. The key now includes the feature spec.

  2. NEEDLESS INVALIDATION. Only DATA-affecting fields belong in the key.
     Changing lr or seed must never throw away a multi-hour spaCy pass.

BACKWARD COMPATIBILITY (deliberate and tested)
----------------------------------------------
When no context facts are requested the spec is empty and the hashed string is
exactly the historical `pipeline|split|max_articles`, so existing caches remain
valid:

    bpe|train|None       -> 99bc1462ce   (matches bpe_train_None_99bc1462ce)
    bpe|validation|None  -> e483bcd459
    bpe|train|3000       -> 17a8ff3163

Verified in tests/test_cache_keys.py against the real filenames.
"""

from __future__ import annotations

import os
import json
import hashlib

import numpy as np

CACHE_DIR = 'data/cache'


def cache_stem(pipeline_name, split, max_articles, feature_spec=''):
    """
    Stable filename stem. With an empty feature_spec this reproduces the
    historical format byte for byte — that is a REQUIREMENT, not a nicety.
    """
    raw = f'{pipeline_name}|{split}|{max_articles}'
    if feature_spec:
        raw += f'|{feature_spec}'
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f'{pipeline_name}_{split}_{max_articles}_{h}'


def build_stream(pipeline, cfg, split='train', max_articles=None,
                 verbose=True):
    """
    Tokenize a WikiText-103 split and cache it.

    Returns a dict with:
        ids        np.memmap int32  [num_tokens]
        codes      np.memmap int16  [num_context_facts, num_tokens] or None
        num_tokens, num_bytes, num_chars, num_articles
        fact_names list[str]  the context facts, in code-row order

    num_bytes counts the UTF-8 bytes of the ORIGINAL article text, which is the
    denominator for bits-per-byte and is what makes cross-pipeline comparison
    fair regardless of how many tokens a pipeline emits.
    """
    from ..registry import context_facts
    from ..features.blocks import build_blocks

    cache_dir = cfg.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    spec = cfg.frozen.spec()
    ctx_names = list(context_facts(cfg.frozen.facts))
    stem = cache_stem(cfg.data.pipeline, split, max_articles, spec)
    npy_path = os.path.join(cache_dir, stem + '.npy')
    codes_path = os.path.join(cache_dir, stem + '.codes.npy')
    meta_path = os.path.join(cache_dir, stem + '.json')

    if os.path.exists(npy_path) and os.path.exists(meta_path):
        if verbose:
            print(f'  [cache hit] {stem}')
        with open(meta_path) as f:
            meta = json.load(f)
        meta['ids'] = np.load(npy_path, mmap_mode='r')
        meta['codes'] = (np.load(codes_path, mmap_mode='r')
                         if ctx_names and os.path.exists(codes_path) else None)
        meta['fact_names'] = ctx_names
        return meta

    if verbose:
        print(f'  [cache miss] tokenizing {split} through '
              f'{cfg.data.pipeline}' + (f' (+{spec})' if spec else '') + '...')

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)

    # only context blocks produce per-position codes
    blocks = [b for b in build_blocks(cfg) if b.source == 'context']

    all_ids = []
    all_codes = [[] for _ in blocks]
    num_bytes = 0
    num_chars = 0
    processed = 0

    for row in ds:
        if max_articles is not None and processed >= max_articles:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue

        if blocks:
            tokens = pipeline.encode(text)
            if not tokens:
                continue
            all_ids.extend(t['token_id'] for t in tokens)
            for bi, block in enumerate(blocks):
                all_codes[bi].extend(block.code(t) for t in tokens)
        else:
            ids = pipeline.encode_ids(text)
            if not ids:
                continue
            all_ids.extend(ids)

        num_bytes += len(text.encode('utf-8'))
        num_chars += len(text)
        processed += 1
        if verbose and processed % 2000 == 0:
            print(f'    {processed} articles, {len(all_ids):,} tokens...')

    arr = np.asarray(all_ids, dtype=np.int32)
    np.save(npy_path, arr)

    if blocks:
        codes = np.asarray(all_codes, dtype=np.int16)
        np.save(codes_path, codes)
        assert codes.shape[1] == arr.shape[0], (
            f'code/id length mismatch: {codes.shape[1]} vs {arr.shape[0]}')

    meta = {
        'pipeline': cfg.data.pipeline,
        'split': split,
        'max_articles': max_articles,
        'feature_spec': spec,
        'fact_names': ctx_names,
        'num_tokens': int(arr.shape[0]),
        'num_bytes': int(num_bytes),
        'num_chars': int(num_chars),
        'num_articles': processed,
    }
    # alignment diagnostics for the BPE offset path
    rate = getattr(pipeline, 'alignment_rate', lambda: None)()
    if rate is not None:
        meta['alignment_rate'] = float(rate)
        if verbose:
            print(f'  context alignment rate: {rate:.3%}')

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f'  [cached] {stem}: {meta["num_tokens"]:,} tokens, '
              f'{meta["num_bytes"]:,} bytes')

    meta['ids'] = np.load(npy_path, mmap_mode='r')
    meta['codes'] = (np.load(codes_path, mmap_mode='r') if blocks else None)
    return meta
