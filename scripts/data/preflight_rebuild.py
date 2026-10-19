#!/usr/bin/env python3
"""
preflight_rebuild.py — the gate to pass BEFORE the multi-hour cache rebuild.

The rebuild runs unattended for ~5 hours and produces codes for EVERY fact at
once — the existing ones and the new ones (`minfl`, `magr`, `mfeat`). Two
things can go wrong that nothing downstream would notice:

  * the annotation code for the new facts is broken, and the night produces a
    cache of empty or misaligned codes;
  * the edit that added them silently CHANGED an existing fact. After the
    rebuild, every request resolves to the new file, so later runs of old cells
    would train on codes that no longer match the ones they were recorded with.

This runs the real pipeline, with the real tokenizer and MorphoLEX, over the
first lines of the validation split — read exactly as build_stream reads them —
and exits non-zero unless ALL of these hold:

  1. IDS REPRODUCE: the token ids equal the cached validation ids.
  2. EXISTING FACTS REPRODUCE: every fact already in a cached validation codes
     file matches it position by position.
  3. NEW FACTS ARE PLAUSIBLE: among alphabetic word tokens, the share with a
     non-empty value lies in a sane range, and not every word gets the same one.
  4. NEW FACTS ARE WORD-LEVEL: a continuation fragment carries the same value as
     the fragment before it (the causal shift needs word-level codes); the rare
     exceptions are BPE words spanning two spaCy tokens (`don't`).

Chain it so the rebuild only starts if it passes:
    python scripts/data/preflight_rebuild.py && python -m morphlm.cli cache ...
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

NEW = ('minfl', 'magr', 'mfeat')
RANGE = {'minfl': (0.10, 0.60), 'magr': (0.15, 0.80), 'mfeat': (0.30, 0.95)}
GDOT = '\u0120'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--lines', type=int, default=300)
    ap.add_argument('--cache-dir', default='data/cache')
    args = ap.parse_args()

    from datasets import load_dataset
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.features.blocks import build_blocks
    from morphlm.data.stream import cache_stem, facts_to_build
    from morphlm.record import AuditRecord
    rec = AuditRecord('preflight_rebuild', args)

    cfg = apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', f'data.bpe_path={args.bpe_path}',
        'frozen.facts=' + ','.join(NEW)])
    build = list(facts_to_build(cfg))
    missing_new = [f for f in NEW if f not in build]
    if missing_new:
        raise SystemExit(f'{missing_new} not on the free-ride list — the rebuild '
                         f'would not produce them')
    full = apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', f'data.bpe_path={args.bpe_path}',
        'frozen.facts=' + ','.join(build)])
    pipe = build_pipeline(full, verbose=False)
    blocks = {b.name: b for b in build_blocks(full) if b.source == 'context'}

    stem = os.path.basename(args.bpe_path).replace('.json', '')
    spec = '' if stem == 'bpe_tokenizer' else f'tok:{stem}'
    ids_stem = cache_stem('bpe', 'validation', None, spec)
    from morphlm.data.stream import require_unpacked
    require_unpacked(args.cache_dir, ids_stem)          # a packed stream is unpacked first (cache_pack.py)
    cached_ids = np.load(os.path.join(args.cache_dir, ids_stem + '.npy'), mmap_mode='r')

    # read the corpus EXACTLY as build_stream does (including its quirk: the
    # heading test runs after strip() and never matches)
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    ids, codes, n = [], {f: [] for f in build}, 0
    for row in ds:
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue
        toks = pipe.encode(text)
        if not toks:
            continue
        ids += [t['token_id'] for t in toks]
        for f in build:
            codes[f] += [blocks[f].code(t) for t in toks]
        n += 1
        if n >= args.lines:
            break
    ids = np.array(ids)
    L = len(ids)
    fails = []

    # 1 ── ids
    ok = bool(np.array_equal(ids, np.asarray(cached_ids[:L])))
    print(f'  1. ids reproduce the cache over {L:,} tokens: {ok}')
    if not ok:
        fails.append('ids')

    # 2 ── existing facts
    best = None
    for cand in glob.glob(os.path.join(args.cache_dir, 'bpe_validation_None_*.codes.json')):
        cm = json.load(open(cand))
        if cm.get('num_tokens') != int(cached_ids.shape[0]):
            continue
        if best is None or len(cm['fact_names']) > len(best[1]['fact_names']):
            best = (cand, cm)
    if best is None:
        fails.append('no cached validation codes to compare against')
    else:
        arr = np.load(best[0][:-5] + '.npy', mmap_mode='r')
        print(f'  2. existing facts vs {os.path.basename(best[0])}:')
        for i, f in enumerate(best[1]['fact_names']):
            if f not in codes:
                continue
            same = float((np.asarray(arr[i, :L]) == np.array(codes[f])).mean())
            flag = 'OK' if same == 1.0 else 'CHANGED'
            print(f'       {f:8} {100 * same:8.4f}% identical   {flag}')
            if same != 1.0:
                fails.append(f'existing fact {f} changed')

    # 3 & 4 ── new facts
    from morphlm.data.tokenization import load_tokenizer
    tk = load_tokenizer(args.bpe_path, vocab_only=True)
    tstr = [tk.id_to_token(int(i)) or '' for i in ids]
    alpha = np.array([bool(t.lstrip(GDOT)) and t.lstrip(GDOT).isalpha() for t in tstr])
    opens = np.array([t.startswith(GDOT) for t in tstr])
    cont = np.zeros(L, bool)
    cont[1:] = alpha[1:] & alpha[:-1] & ~opens[1:]
    print('  3-4. new facts:')
    for f in NEW:
        c = np.array(codes[f])
        none = blocks[f].n_classes - 1
        share = float((c[alpha] != none).mean())
        lo, hi = RANGE[f]
        distinct = len(set(c[alpha].tolist()))
        const = float((c[1:][cont[1:]] == c[:-1][cont[1:]]).mean()) if cont.any() else 1.0
        top = Counter(c[alpha].tolist()).most_common(3)
        print(f'       {f:6} non-empty {100 * share:5.1f}% (want {100*lo:.0f}-{100*hi:.0f}%)  '
              f'distinct {distinct:3}  word-constant {100 * const:6.2f}%  '
              f'top {[blocks[f].classes[k][:18] for k, _ in top]}')
        if not lo <= share <= hi:
            fails.append(f'{f} non-empty share {share:.2%}')
        if distinct < 3:
            fails.append(f'{f} has {distinct} distinct values')
        if const < 0.97:
            fails.append(f'{f} word constancy {const:.2%}')
        rec.add(f, {'non_empty': share, 'distinct': distinct, 'word_constant': const})

    rec.add('failures', fails)
    rec.write()
    if fails:
        print(f'\n  PREFLIGHT FAILED: {fails}\n  The rebuild must not start.')
        sys.exit(1)
    print('\n  PREFLIGHT PASSED — the rebuild may start.')


if __name__ == '__main__':
    main()
