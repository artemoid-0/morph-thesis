"""
stream.py — tokenize once, cache, memory-map.

TWO CACHE FILES, KEYED SEPARATELY
---------------------------------
The token ids for a pipeline do not depend on which frozen facts you requested;
only the annotation codes do. Keying both on one combined string meant the
baseline, `--facts dep` and `--facts pos` each triggered a full independent
spaCy pass over the corpus — three passes where one suffices.

    ids   key = pipeline|split|max_articles                  (historical format)
    codes key = pipeline|split|max_articles|ctx:<gran>:<facts>

The ids key is byte-identical to the pre-refactor scheme, which is a
REQUIREMENT: existing caches must stay valid.

    bpe|train|None       -> 99bc1462ce   (bpe_train_None_99bc1462ce)
    bpe|validation|None  -> e483bcd459

FREE-RIDE ANNOTATION
--------------------
The morpheme pipelines call spacy.load(), which enables tagger, parser AND ner
by default — they have always parsed dependencies and discarded them. So for
those pipelines we compute ALL context facts on whichever pass happens first,
even when the current config asked for none. The marginal cost is reading two
attributes off an already-parsed token; the saving is every later pass.

BPE is different: it uses no spaCy at all, so annotating it is a genuinely new
cost and happens only when the config asks for it.

STALENESS
---------
The codes key includes the dependency granularity and the fact list, so
changing either yields a different file. Changing lr or seed changes neither
key, so training knobs can never invalidate a multi-hour pass.
"""

from __future__ import annotations

import os
import json
import hashlib

import numpy as np

CACHE_DIR = 'data/cache'

#: Every context fact worth computing on a free-ride pass.
ALL_CONTEXT_FACTS = ('dep', 'pos', 'ectx')

#: Pipelines that already run spaCy for their own reasons.
SPACY_NATIVE = ('morph', 'det')


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


def facts_to_build(cfg):
    """
    Which context facts this pass should COMPUTE (not necessarily use).

    spaCy-native pipelines compute everything, because the parse is already
    paid for. BPE computes only what was asked, because for it the parse is new.
    """
    from ..registry import context_facts
    requested = context_facts(cfg.frozen.facts)
    if cfg.data.pipeline in SPACY_NATIVE:
        return ALL_CONTEXT_FACTS
    return tuple(requested)


def build_stream(pipeline, cfg, split='train', max_articles=None,
                 verbose=True):
    """
    Tokenize a WikiText-103 split and cache it.

    Returns a dict with:
        ids        np.memmap int32  [num_tokens]
        codes      np.ndarray int16 [n_requested_facts, num_tokens] or None
        fact_names list[str]        requested facts, in code-row order
        num_tokens, num_bytes, num_chars, num_articles
    """
    from ..registry import context_facts, get_fact

    cache_dir = cfg.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    requested = list(context_facts(cfg.frozen.facts))
    build_facts = list(facts_to_build(cfg))
    gran = cfg.frozen.dep_granularity

    ids_stem = cache_stem(cfg.data.pipeline, split, max_articles, '')
    ids_npy = os.path.join(cache_dir, ids_stem + '.npy')
    ids_meta = os.path.join(cache_dir, ids_stem + '.json')

    c_spec = codes_spec(build_facts, gran)
    codes_stem = cache_stem(cfg.data.pipeline, split, max_articles, c_spec)
    codes_npy = os.path.join(cache_dir, codes_stem + '.codes.npy')
    codes_meta_path = os.path.join(cache_dir, codes_stem + '.codes.json')

    have_ids = os.path.exists(ids_npy) and os.path.exists(ids_meta)
    have_codes = os.path.exists(codes_npy) and os.path.exists(codes_meta_path)
    need_codes = bool(requested)

    # ── fast path: everything needed is already on disk ───────────────────────
    if have_ids and (have_codes or not need_codes):
        if verbose:
            extra = f' + codes({",".join(requested)})' if need_codes else ''
            print(f'  [cache hit] {ids_stem}{extra}')
        with open(ids_meta) as f:
            meta = json.load(f)
        meta['ids'] = np.load(ids_npy, mmap_mode='r')
        meta['codes'] = None
        meta['fact_names'] = requested
        if need_codes:
            with open(codes_meta_path) as f:
                cmeta = json.load(f)
            stored = cmeta['fact_names']
            missing = [f for f in requested if f not in stored]
            if missing:
                raise RuntimeError(
                    f'cached codes {codes_stem} lack {missing}; delete it and '
                    f'rebuild with: cli cache --pipeline {cfg.data.pipeline} '
                    f'--facts {",".join(requested)}')
            full = np.load(codes_npy, mmap_mode='r')
            rows = [stored.index(f) for f in requested]
            meta['codes'] = np.asarray(full[rows])
        return meta

    # ── slow path: run the pipeline ───────────────────────────────────────────
    if verbose:
        what = 'ids' + (f' + codes({",".join(build_facts)})'
                        if build_facts else '')
        print(f'  [cache miss] tokenizing {split} through '
              f'{cfg.data.pipeline} -> {what}')
        if build_facts and cfg.data.pipeline in SPACY_NATIVE:
            print('    (context facts ride along free: this pipeline already '
                  'parses with spaCy)')

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)

    blocks = [get_fact(name)(cfg) for name in build_facts]

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
    np.save(ids_npy, arr)

    meta = {
        'pipeline': cfg.data.pipeline,
        'split': split,
        'max_articles': max_articles,
        'num_tokens': int(arr.shape[0]),
        'num_bytes': int(num_bytes),
        'num_chars': int(num_chars),
        'num_articles': processed,
    }
    rate = getattr(pipeline, 'alignment_rate', lambda: None)()
    if rate is not None:
        meta['alignment_rate'] = float(rate)
        if verbose:
            print(f'  context alignment rate: {rate:.3%}')
    with open(ids_meta, 'w') as f:
        json.dump(meta, f, indent=2)

    codes_arr = None
    if blocks:
        codes_arr = np.asarray(all_codes, dtype=np.int16)
        assert codes_arr.shape[1] == arr.shape[0], (
            f'code/id length mismatch: {codes_arr.shape[1]} vs {arr.shape[0]}')
        np.save(codes_npy, codes_arr)
        with open(codes_meta_path, 'w') as f:
            json.dump({'fact_names': build_facts,
                       'dep_granularity': gran,
                       'num_tokens': int(arr.shape[0])}, f, indent=2)

    if verbose:
        print(f'  [cached] {ids_stem}: {meta["num_tokens"]:,} tokens, '
              f'{meta["num_bytes"]:,} bytes')
        if blocks:
            print(f'  [cached] codes {build_facts} -> {codes_stem}')

    meta['ids'] = np.load(ids_npy, mmap_mode='r')
    meta['fact_names'] = requested
    if requested and codes_arr is not None:
        rows = [build_facts.index(f) for f in requested]
        meta['codes'] = codes_arr[rows]
    else:
        meta['codes'] = None
    return meta
