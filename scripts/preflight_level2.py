#!/usr/bin/env python3
"""
preflight_level2.py — check everything that could void a Level-2 run, BEFORE
the multi-hour cache build.

WHY THIS EXISTS
---------------
The BPE + context-fact path had never been executed. Reviewing it turned up a
bug that would have silently voided the entire experiment: ByteLevel folds the
preceding space into the following token, so a word-initial token's START
OFFSET lands on whitespace, which no spaCy token covers. The old code indexed
annotations by that start offset, so every word-initial token — 78% of the BPE
stream — would have received `none`. The cache would have been built, the runs
would have completed, and the result would have been a near-no-op that looked
like a real negative.

That class of failure is invisible in the training logs. It is only visible if
something measures the codes themselves. This script does that, on a few
hundred lines, in a couple of minutes.

WHAT IT CHECKS
--------------
    1. alignment rate         — what fraction of BPE tokens receive a fact
    2. code distribution      — do the facts actually vary, or is everything
                                falling to the 'none' sentinel
    3. root coverage          — how many distinct roots appear, and how often
                                the root is unknown
    4. within-token-id variance — the property that makes a fact CONTEXTUAL:
                                the same BPE token id must receive different
                                codes in different words, otherwise a static
                                embedding table could already represent it
    5. int16 safety           — codes are stored as int16 in the cache
    6. parameter arithmetic   — the factor must SUBSTITUTE for trained
                                capacity, not add to it

Anything failing here means the cache build would be wasted.

    python scripts/preflight_level2.py --lines 300
"""

import os
import sys
import math
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

FACTS = ('dep', 'pos', 'ectx', 'mstruct', 'mrole', 'mroot')

#: fact name -> key in the pipeline's token dict. `ectx` reads `entity_label`,
#: which is why an earlier run of this script reported ectx as 100% constant:
#: it was querying a key that does not exist. A script bug, not a pipeline bug.
TOKEN_KEY = {'ectx': 'entity_label'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lines', type=int, default=300)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--root-width', type=int, default=16)
    args = ap.parse_args()

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.features.blocks import build_blocks
    from morphlm.data.stream import facts_to_build
    from datasets import load_dataset

    cfg = apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe',
        f'data.bpe_path={args.bpe_path}',
        'frozen.facts=mroot,mstruct,mrole',
        f'frozen.root_width={args.root_width}',
    ])
    cfg.validate()

    print(f'facts that WILL be built into the cache: {facts_to_build(cfg)}')
    print('(all of them, so one pass serves every future BPE experiment)')
    print()
    print('building BPE pipeline with morphological analyser...')
    pipe = build_pipeline(cfg, verbose=True)

    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)
    from morphlm.progress import Progress
    prog = Progress(total=args.lines)

    counts = {f: Counter() for f in FACTS}
    per_id = {f: defaultdict(set) for f in FACTS}
    n_tok = 0
    processed = 0
    roots_seen = set()

    for row in ds:
        if processed >= args.lines:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue
        for t in pipe.encode(text):
            n_tok += 1
            for f in FACTS:
                v = t.get(TOKEN_KEY.get(f, f))
                counts[f][v] += 1
                per_id[f][t['token_id']].add(v)
            if t.get('mroot') is not None:
                roots_seen.add(t['mroot'])
        processed += 1
        if processed % 100 == 0:
            print(f'  {processed} lines, {n_tok:,} tokens | {prog.status(processed)}')

    ok = True
    print()
    print('=' * 74)
    print(f'PREFLIGHT — {processed} {args.split} lines, {n_tok:,} BPE tokens')
    print('=' * 74)

    rate = pipe.alignment_rate()
    print(f'\n1. ALIGNMENT RATE: {rate:.2%}' if rate is not None
          else '\n1. ALIGNMENT RATE: n/a')
    if rate is not None and rate < 0.90:
        print('   FAIL — expected >90%. The span-scan fix is not working.')
        ok = False
    else:
        print('   ok (the start-offset bug would have shown ~20% here)')

    print('\n2. CODE DISTRIBUTION (share falling to the none sentinel)')
    print(f'   {"fact":10} {"distinct":>9} {"none share":>11}')
    for f in FACTS:
        c = counts[f]
        nones = c.get(None, 0)
        share = nones / max(n_tok, 1)
        distinct = len([k for k in c if k is not None])
        flag = ''
        if distinct <= 1:
            flag = '  FAIL: constant'
            ok = False
        elif share > 0.60:
            flag = '  WARN: mostly none'
        print(f'   {f:10} {distinct:9,} {share:10.1%}{flag}')

    print('\n3. ROOT COVERAGE')
    n_roots = pipe.analyser.n_roots if pipe.analyser else 0
    none_root = n_roots - 1
    unknown = counts['mroot'].get(none_root, 0)
    not_a_word = counts['mroot'].get(None, 0)
    real = n_tok - unknown - not_a_word
    print(f'   inventory        : {n_roots:,} classes (incl. none)')
    print(f'   distinct observed: {len(roots_seen):,}')
    print(f'   not a word       : {not_a_word / max(n_tok,1):.1%} '
          f'(punctuation, digits — no root exists)')
    print(f'   word, no root    : {unknown / max(n_tok,1):.1%} '
          f'(unanalysable; entities now carry their own root)')
    print(f'   CARRIES A ROOT   : {real / max(n_tok,1):.1%} of tokens')
    if real / max(n_tok, 1) < 0.30:
        print('   WARN — the root feature is informative for under a third of')
        print('   tokens, which caps the effect it can have.')
    if len(roots_seen) < 50:
        print('   FAIL — too few distinct roots; the analyser is not firing.')
        ok = False
    else:
        print('   ok')

    print('\n4. WITHIN-TOKEN-ID VARIANCE (what makes a fact CONTEXTUAL)')
    print(f'   {"fact":10} {"ids with >1 code":>18} {"share of ids":>13}')
    for f in FACTS:
        varying = sum(1 for v in per_id[f].values() if len(v) > 1)
        tot = len(per_id[f])
        share = varying / max(tot, 1)
        flag = ''
        if f in ('mroot', 'mstruct', 'mrole') and varying == 0:
            flag = '  FAIL: static, a lookup table could do this'
            ok = False
        print(f'   {f:10} {varying:18,} {share:12.1%}{flag}')

    print('\n5. INT16 SAFETY (cache code dtype)')
    # codes are class INDICES, so the bound is the largest inventory, not the
    # largest observed label. (Earlier this compared raw labels and crashed on
    # str vs int.)
    mx = 0
    for b in build_blocks(cfg):
        mx = max(mx, b.n_classes)
    for f in ('dep', 'pos', 'ectx'):
        mx = max(mx, len([k for k in counts[f] if k is not None]) + 1)
    print(f'   largest inventory: {mx:,} classes (int16 limit 32,767)')
    if mx >= 32767:
        print('   FAIL — codes would overflow the cache dtype.')
        ok = False
    else:
        print('   ok')

    print('\n6. PARAMETER ARITHMETIC (the factor must SUBSTITUTE, not add)')
    d, L, ff, seq = 128, 4, 512, 256
    core = L * ((d*3*d + 3*d) + (d*d + d) + (d*ff + ff) + (ff*d + d) + 4*d) \
        + seq*d + 2*d
    V = 17155
    base = core + V*d + d*V
    blocks = build_blocks(cfg)
    T = sum(b.dim for b in blocks)
    frozen_tot = core + V*(d-T) + d*V
    learned_tot = frozen_tot + sum(b.n_classes * b.dim for b in blocks)
    print(f'   requested facts occupy T = {T} of {d} dims')
    for b in blocks:
        print(f'     {b.name:8} classes={b.n_classes:>7,} dim={b.dim:>3} '
              f'{"projection" if b.is_projection else "one-hot"}')
    print(f'   baseline            {base:>12,}')
    print(f'   frozen  factors     {frozen_tot:>12,}  ({frozen_tot-base:+,})')
    print(f'   learned factors     {learned_tot:>12,}  ({learned_tot-base:+,})')
    if learned_tot > base * 1.05:
        print('   WARN — learned factors add >5% parameters; the comparison to')
        print('   baseline is no longer a clean substitution. Lower root_width.')

    print()
    print('=' * 74)
    print('PREFLIGHT PASSED — safe to build the cache' if ok
          else 'PREFLIGHT FAILED — do NOT build the cache yet')
    print('=' * 74)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
