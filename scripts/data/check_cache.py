#!/usr/bin/env python3
"""
check_cache.py — the health of a built cache, in any language.

For one split: its size, and for every fact in its codes files (this stream's own, G-72), how
often the fact fires on word tokens, which classes it uses, and whether it is
constant within words under the tokenizer's own word rule (the causal shift
needs word-level codes). Run it on a pilot validation build before committing
hours to a train build (docs/reference/ukrainian_pipeline.md).

    python scripts/data/check_cache.py --bpe-path data/bpe_uk_17k.json --corpus ukwiki
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--corpus', default='wikitext103')
    ap.add_argument('--corpus-fraction', type=float, default=1.0)
    ap.add_argument('--corpus-holdout', type=int, default=2000)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--require-facts', nargs='+', default=None, metavar='FACT',
                    help='as a gate: exit 2 unless every one of these facts has codes')
    ap.add_argument('--require-constant', type=float, default=None, metavar='PCT',
                    help='as a gate: exit 1 unless the required facts (all, without '
                         '--require-facts) are at least PCT%% constant within words')
    args = ap.parse_args()
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem
    from morphlm.features.lexical import tokenizer_tables, word_spans, word_rule_for
    from morphlm.features.blocks import build_blocks
    from morphlm.record import AuditRecord
    rec = AuditRecord('check_cache', args)

    cfg = apply_overrides(ExperimentConfig(), [
        'data.pipeline=bpe', f'data.bpe_path={args.bpe_path}', f'data.corpus={args.corpus}',
        f'data.corpus_fraction={args.corpus_fraction}', f'data.corpus_holdout={args.corpus_holdout}'])
    stem = cache_stem('bpe', args.split, None, cfg.surface_spec())
    ids_path = os.path.join(args.cache_dir, stem + '.npy')
    if not os.path.exists(ids_path):
        raise SystemExit(f'no cache at {ids_path}')
    ids = np.load(ids_path, mmap_mode='r')
    # every codes file for THIS stream (G-72: not another fraction's derived
    # files, whose validation tokens are the same): the fullest first, each fact
    # shown once — a fact's own derivation file before any copy of it, so a row
    # carried into a later file cannot stand in for the original
    from morphlm.data.stream import stream_codes_files, codes_owner
    files = [(p, m) for p, _, m in stream_codes_files(args.cache_dir, 'bpe', args.split, None,
                                                       stem, int(ids.shape[0]))]
    if not files:
        raise SystemExit('no codes file for this stream')
    files.sort(key=lambda pm: -len(pm[1]['fact_names']))
    arrays = {p: np.load(p[:-5] + '.npy', mmap_mode='r') for p, _ in files}
    own = {m['fact_names'][-1]: (p, len(m['fact_names']) - 1) for p, m in files
           if m['fact_names'] and p.endswith(f'_{m["fact_names"][-1]}.codes.json')}
    rows = {}
    for p, m in files:
        for i, f in enumerate(m['fact_names']):
            if f not in rows:
                q, j = own.get(f, (p, i))
                rows[f] = (arrays[q], j, q)
    rule = word_rule_for(args.bpe_path)
    tables = tokenizer_tables(args.bpe_path)
    wid, key, starts = word_spans(np.asarray(ids, dtype=np.int64), tables)
    words = wid >= 0
    cont = np.flatnonzero(words & ~starts)
    frag = getattr(tables, 'frag', None)
    n_frag = int(frag[np.asarray(ids, dtype=np.int64)].sum()) if frag is not None else 0
    print(f'  {args.split}: {ids.shape[0]:,} tokens; {int(starts.sum()):,} words '
          f'({int(words.sum()):,} word tokens, {len(cont):,} continuation fragments) '
          f'under the "{rule}" word rule; {n_frag:,} tokens are byte fragments of a split character')
    print(f'  {"fact":10} {"fires on":>9} {"constant in words":>18}  {"read from":12} classes used')
    out = {}
    for f, (arr, i, path) in rows.items():
        kind, _ = codes_owner(os.path.basename(path), f'bpe_{args.split}_None_')
        src = ('build' if kind == 'build' else
               os.path.basename(path)[len(stem) + 1:-len('.codes.json')] if kind == 'derived' else
               os.path.basename(path))
        r = np.asarray(arr[i])
        try:
            blk = build_blocks(apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', f'frozen.facts={f}']))[0]
            none = blk.n_classes - 1
            names = blk.classes
        except Exception:
            none, names = int(r.max()), None
        fires = float((r[words] != none).mean()) if words.any() else float('nan')
        const = float((r[cont] == r[cont - 1]).mean()) if len(cont) else float('nan')
        used = sorted(set(r[words].tolist()))
        shown = [names[k] for k in used] if names else used
        print(f'  {f:10} {100 * fires:8.1f}% {100 * const:17.2f}%  {src[:12]:12} {shown[:9]}{" …" if len(shown) > 9 else ""}')
        out[f] = {'fires': fires, 'word_constant': const, 'classes_used': len(used),
                  'file': os.path.basename(path)}
    rec.add('split', {'tokens': int(ids.shape[0]), 'words': int(starts.sum()), 'rule': rule})
    rec.add('facts', out)
    required = args.require_facts or list(out)
    missing = [f for f in required if f not in out]
    low = [f for f in required if f in out and args.require_constant is not None
           and not out[f]['word_constant'] * 100 >= args.require_constant]
    rec.add('gate', {'required': required, 'missing': missing, 'below': low,
                     'threshold': args.require_constant})
    rec.write()
    if missing:
        print(f'  GATE: no codes for {missing}')
        sys.exit(2)
    if low:
        print(f'  GATE: below {args.require_constant}% within words: {low}')
        sys.exit(1)
    if args.require_facts or args.require_constant is not None:
        print(f'  GATE passed: {required}' + (f' at >= {args.require_constant}%' if args.require_constant is not None else ''))


if __name__ == '__main__':
    main()
