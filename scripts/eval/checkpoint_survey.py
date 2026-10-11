#!/usr/bin/env python3
"""
checkpoint_survey.py — what is in results/, and which checkpoints still load?

WHY THIS EXISTS
---------------
The frozen-vs-learned rewrite changed the embedding's buffer layout, so
checkpoints saved before it use a different schema:

    embed.static_table   -> embed.static_codes_{i}   (float one-hot -> int code)
    embed.ctx_eye_{i}    -> embed.factor_eye_{i}     (rename)

A checkpoint that cannot load is a cell that cannot be re-analysed — the probe,
the bucket comparison and the category bit accounting all need one. Since a run
costs ~1 hour, knowing which are recoverable BEFORE planning analysis is worth
the thirty seconds this takes.

Loads every checkpoint through the real code path, so a "loads: yes" here means
the analysis scripts will work.

    python scripts/eval/checkpoint_survey.py
    python scripts/eval/checkpoint_survey.py --results-dir results
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--no-load', action='store_true',
                    help='inspect metadata only, do not build models')
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f'no such directory: {args.results_dir}')
        sys.exit(1)

    rows = []
    for entry in sorted(os.listdir(args.results_dir)):
        d = os.path.join(args.results_dir, entry)
        res = os.path.join(d, 'result.json')
        ckpt = os.path.join(d, 'model.pt')
        if not os.path.exists(res):
            continue
        with open(res) as f:
            r = json.load(f)
        row = {
            'run_id': entry,
            'label': r.get('label', '?'),
            'seed': r.get('seed', '?'),
            'bpb': r.get('bits_per_byte'),
            'ppl': r.get('perplexity'),
            'ckpt': os.path.exists(ckpt),
            'size_mb': (os.path.getsize(ckpt) / 1e6
                        if os.path.exists(ckpt) else 0.0),
            'format': None,
            'loads': None,
            'note': '',
        }
        if row['ckpt'] and not args.no_load:
            try:
                import torch
                blob = torch.load(ckpt, map_location='cpu', weights_only=False)
                row['format'] = blob.get('format', 1)
                from morphlm.training.metrics import load_checkpoint
                load_checkpoint(entry, args.results_dir)
                row['loads'] = True
            except Exception as e:
                row['loads'] = False
                row['note'] = f'{type(e).__name__}: {str(e)[:60]}'
        rows.append(row)

    if not rows:
        print(f'no runs found under {args.results_dir}/')
        sys.exit(0)

    print()
    print('=' * 100)
    print(f'CHECKPOINT SURVEY — {args.results_dir}/')
    print('=' * 100)
    print(f'  {"run_id":14} {"label":38} {"seed":>4} {"BPB":>8} '
          f'{"ckpt":>6} {"fmt":>4} {"loads":>6}')
    print('  ' + '-' * 96)
    for r in sorted(rows, key=lambda x: (x['label'], x['seed'])):
        bpb = f'{r["bpb"]:.4f}' if r['bpb'] is not None else '-'
        ck = f'{r["size_mb"]:.0f}MB' if r['ckpt'] else 'NONE'
        fmt = str(r['format']) if r['format'] else '-'
        ld = {True: 'yes', False: 'NO', None: '-'}[r['loads']]
        print(f'  {r["run_id"]:14} {r["label"]:38} {str(r["seed"]):>4} '
              f'{bpb:>8} {ck:>6} {fmt:>4} {ld:>6}')
        if r['note']:
            print(f'    -> {r["note"]}')

    n = len(rows)
    have = sum(1 for r in rows if r['ckpt'])
    ok = sum(1 for r in rows if r['loads'] is True)
    bad = sum(1 for r in rows if r['loads'] is False)
    legacy = sum(1 for r in rows if r['format'] == 1)

    print()
    print(f'  {n} runs · {have} with checkpoints · {ok} load · {bad} fail'
          + (f' · {legacy} legacy format (auto-migrated)' if legacy else ''))
    if bad:
        print()
        print('  Failing checkpoints cannot be re-analysed. If a cell is quoted')
        print('  in the write-up and its checkpoint will not load, either')
        print('  retrain it or drop any claim that needs post-hoc analysis.')
    total = sum(r['size_mb'] for r in rows)
    print(f'  total checkpoint size: {total:.0f} MB')
    print()


if __name__ == '__main__':
    main()
