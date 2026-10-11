#!/usr/bin/env python3
"""
evaluate_split.py — score saved checkpoints on a HELD-OUT split.

WHY THIS EXISTS
---------------
Every number in this project was measured on WikiText-103 **validation**, and
the training loop selects the reported checkpoint by *validation* loss
("best-val, never last-val"). Selection and reporting therefore happen on the
same split.

With 101 evaluations per run, the reported figure is the minimum of 101 draws on
the split it is reported on, which is optimistically biased. The bias is
probably similar across arms, so DIFFERENCES are likely safe — but every
absolute number is a development number presented as a result, and that is a
standard and entirely avoidable objection.

Standard practice: select on validation, report on test. WikiText-103 has a test
split that this project has never touched.

This scores existing checkpoints on any split WITHOUT retraining. Validation
cache construction took 32 seconds for BPE with all context facts, so the test
cache is comparable, and re-scoring is about a minute per checkpoint.

    python scripts/eval/evaluate_split.py --all --split test
    python scripts/eval/evaluate_split.py --runs <id> <id> --split test
"""

import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


def score(run_id, results_dir, split, batches=0):
    """Score one checkpoint on `split`. Returns (cfg, loss, ppl, bpb, tokens)."""
    import torch
    import dataclasses
    from torch.utils.data import DataLoader

    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import get_device
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.stream import build_stream
    from morphlm.data.windows import LMWindowDataset

    model, cfg = load_checkpoint(run_id, results_dir)
    device = get_device()
    model = model.to(device).eval()

    pipeline = build_pipeline(cfg, verbose=False)
    meta = build_stream(pipeline, cfg, split=split,
                        max_articles=None, verbose=False)
    ds = LMWindowDataset(meta, seq_len=cfg.data.seq_len,
                         shuffle_codes=getattr(cfg.frozen, 'shuffle_codes',
                                               False),
                         seed=cfg.train.seed)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size)

    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for bi, (x, y, c) in enumerate(loader):
            if batches and bi >= batches:
                break
            x, y = x.to(device), y.to(device)
            cc = c.to(device) if c.numel() else None
            _, loss = model(x, y, codes=cc)
            n = y.numel()
            total_loss += loss.item() * n
            total_tok += n

    mean = total_loss / max(total_tok, 1)
    return (cfg, mean, math.exp(mean),
            ds.bits_per_byte(total_loss, total_tok), total_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--split', default='test',
                    choices=['test', 'validation'])
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default=None,
                    help='write a JSON summary here')
    args = ap.parse_args()

    if args.all:
        ids = [d for d in sorted(os.listdir(args.results_dir))
               if os.path.exists(os.path.join(args.results_dir, d, 'model.pt'))]
    elif args.runs:
        ids = args.runs
    else:
        ap.error('pass --all or --runs')

    print(f'scoring {len(ids)} checkpoint(s) on `{args.split}`')
    print('(the first will build the split cache; the rest reuse it)')
    print()

    rows = []
    for i, rid in enumerate(ids, 1):
        try:
            cfg, loss, ppl, bpb, tok = score(rid, args.results_dir, args.split)
        except Exception as e:
            print(f'  [{i}/{len(ids)}] {rid}: FAILED — '
                  f'{type(e).__name__}: {str(e)[:70]}')
            continue
        val = None
        rp = os.path.join(args.results_dir, rid, 'result.json')
        if os.path.exists(rp):
            with open(rp) as f:
                val = json.load(f).get('bits_per_byte')
        rows.append({'run_id': rid, 'label': cfg.label(),
                     'seed': cfg.train.seed, 'split': args.split,
                     'loss': loss, 'ppl': ppl, 'bpb': bpb, 'tokens': tok,
                     'val_bpb': val})
        print(f'  [{i}/{len(ids)}] {cfg.label():40} '
              f'{args.split} BPB {bpb:.4f}' +
              (f'  (val {val:.4f}, {bpb - val:+.4f})' if val else ''))

    if not rows:
        print('nothing scored')
        return

    print()
    print('=' * 92)
    print(f'HELD-OUT EVALUATION — {args.split}')
    print('=' * 92)
    print(f'  {"label":42} {"seed":>4} {"val BPB":>9} '
          f'{"{} BPB".format(args.split):>10} {"delta":>8}')
    print('  ' + '-' * 88)
    for r in sorted(rows, key=lambda x: (x['label'], x['seed'])):
        v = f'{r["val_bpb"]:.4f}' if r['val_bpb'] else '-'
        d = (f'{r["bpb"] - r["val_bpb"]:+.4f}' if r['val_bpb'] else '-')
        print(f'  {r["label"]:42} {r["seed"]:>4} {v:>9} {r["bpb"]:10.4f} {d:>8}')

    deltas = [r['bpb'] - r['val_bpb'] for r in rows if r['val_bpb']]
    if deltas:
        print()
        print(f'  val -> {args.split} shift: mean {np.mean(deltas):+.4f}, '
              f'sd {np.std(deltas):.4f}, '
              f'range {min(deltas):+.4f} to {max(deltas):+.4f}')
        print()
        print('  READING THIS')
        print('    A roughly CONSTANT shift across arms means selection bias is')
        print('    uniform and every reported DIFFERENCE survives; only the')
        print('    absolute numbers move. A shift that varies by arm would mean')
        print('    some arms overfit the validation split more than others, and')
        print('    the comparisons themselves would need restating.')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'\n  wrote {args.out}')
    print()


if __name__ == '__main__':
    main()
