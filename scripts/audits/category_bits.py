#!/usr/bin/env python3
"""
category_bits.py — where does each pipeline SPEND its bits?

THE QUESTION
------------
The information-fair comparison says BPE wins: 1.3946 vs morph 1.5292 BPB, a
gap of 0.1346. But an aggregate number does not say WHY, and the two pipelines
differ in a way that has nothing to do with segmentation quality:

    morph   emits a separate <sp> token at every word boundary
    BPE     folds the space into the following token as a Ġ prefix

Both encode the same information; BPE just uses a cheaper code for it. From the
token counts, whitespace is ~32.6% of ALL morph tokens, and the arithmetic says
that if a <sp> costs ~0.72 bits then whitespace ALONE accounts for the entire
gap.

That is a hypothesis, not a measurement. This script measures it: it loads a
saved checkpoint and reports total bits by token category, so the gap can be
attributed rather than guessed.

WHY THE DISTINCTION MATTERS FOR THE THESIS
------------------------------------------
Two very different conclusions are on the table:

  (a) morph segments words worse than BPE
        -> the thesis premise is wrong

  (b) morph segments words as well or better, but encodes word BOUNDARIES with
      a wasteful code
        -> the premise survives and the fix is an encoding change, not a
           redesign

Only a per-category breakdown separates them.

    python scripts/audits/category_bits.py --run-id <id> [--lines 400]
    python scripts/audits/category_bits.py --all
"""

import os
import sys
import math
import argparse
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def category_of_ids(cfg, vocab_size):
    """id -> category label, for grouping. Works for both pipeline families."""
    labels = ['<special>'] * vocab_size
    if cfg.data.pipeline == 'bpe':
        from morphlm.data.tokenization import load_tokenizer
        tk = load_tokenizer(cfg.data.bpe_path, vocab_only=True)
        vocab = tk.get_vocab()
        for tokstr, tid in vocab.items():
            if tid >= vocab_size:
                continue
            # ByteLevel marks a preceding space with U+0120
            if tokstr.startswith('\u0120'):
                labels[tid] = 'word-initial (carries space)'
            else:
                labels[tid] = 'word-internal'
        return labels

    from morphlm.vocab.build import load_vocab
    vocab = load_vocab(cfg.data.vocab_csv)
    for idx, row in vocab.iterrows():
        if idx < vocab_size:
            labels[idx] = str(row['category'])
    return labels


def analyse(run_id, results_dir=os.path.join(ROOT, 'results'), lines=None, batches=40):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import get_device
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.windows import make_datasets

    print(f'loading checkpoint {run_id}...')
    model, cfg = load_checkpoint(run_id, results_dir)
    device = get_device()
    model = model.to(device).eval()

    pipeline = build_pipeline(cfg, verbose=False)
    _, val_ds = make_datasets(pipeline, cfg, verbose=False)
    loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    labels = category_of_ids(cfg, model.vocab_size)

    bits = defaultdict(float)
    count = defaultdict(int)
    total_bits = 0.0
    total_tokens = 0

    with torch.no_grad():
        for bi, (x, y, c) in enumerate(loader):
            if batches and bi >= batches:
                break
            x, y = x.to(device), y.to(device)
            cc = c.to(device) if c.numel() else None
            logits, _ = model(x, y, codes=cc)
            per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      y.view(-1), reduction='none')
            per_tok = (per_tok / math.log(2)).cpu().numpy()   # nats -> bits
            flat_y = y.view(-1).cpu().numpy()
            for tid, b in zip(flat_y, per_tok):
                lab = labels[tid] if tid < len(labels) else '<oob>'
                bits[lab] += float(b)
                count[lab] += 1
                total_bits += float(b)
                total_tokens += 1

    return cfg, bits, count, total_bits, total_tokens, val_ds


def print_report(cfg, bits, count, total_bits, total_tokens, val_ds):
    print()
    print('=' * 74)
    print(f'CATEGORY BIT ACCOUNTING — {cfg.label()}')
    print('=' * 74)
    print(f'  tokens scored : {total_tokens:,}')
    print(f'  total bits    : {total_bits:,.0f}')
    print()
    print(f'  {"category":32} {"tokens":>10} {"% tok":>7} '
          f'{"bits":>12} {"% bits":>7} {"bits/tok":>9}')
    print('  ' + '-' * 72)
    for lab in sorted(bits, key=lambda k: -bits[k]):
        n = count[lab]
        b = bits[lab]
        print(f'  {lab:32} {n:10,} {100.0*n/total_tokens:6.1f}% '
              f'{b:12,.0f} {100.0*b/total_bits:6.1f}% {b/max(n,1):9.3f}')

    # scale the sampled bits to a full-corpus BPB, so the numbers are on the
    # same scale as the headline result
    frac = total_tokens / max(val_ds.num_tokens, 1)
    scored_bytes = val_ds.num_bytes * frac
    print()
    print(f'  sampled BPB   : {total_bits / max(scored_bytes,1):.4f}')

    ws = sum(b for k, b in bits.items() if k == 'space')
    if ws:
        without = (total_bits - ws) / max(scored_bytes, 1)
        print()
        print(f'  bits spent on whitespace tokens : {ws:,.0f} '
              f'({100.0*ws/total_bits:.1f}% of all bits)')
        print(f'  BPB EXCLUDING whitespace tokens : {without:.4f}')
        print()
        print('  That second number is the segmentation-only comparison: what')
        print('  morph would cost if word boundaries were encoded as cheaply')
        print('  as BPE encodes them (folded into the next token, not spent as')
        print('  a token of their own).')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id')
    ap.add_argument('--all', action='store_true',
                    help='every run under results/')
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--batches', type=int, default=40,
                    help='validation batches to score (0 = all)')
    args = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('category_bits', args)

    if args.all:
        ids = [d for d in sorted(os.listdir(args.results_dir))
               if os.path.exists(os.path.join(args.results_dir, d, 'model.pt'))]
        if not ids:
            print(f'no checkpoints (model.pt) found under {args.results_dir}/')
            print('runs saved before checkpointing was added cannot be '
                  'analysed without retraining.')
            sys.exit(1)
    elif args.run_id:
        ids = [args.run_id]
    else:
        ap.error('pass --run-id or --all')

    for rid in ids:
        try:
            out = analyse(rid, args.results_dir, batches=args.batches)
            print_report(*out)
            cfg, bits, toks, total_bits, total_toks, val_ds = out
            rec.add(rid, {
                'label': cfg.label(),
                'bits_by_category': {str(k): float(v) for k, v in bits.items()},
                'tokens_by_category': {str(k): int(v) for k, v in toks.items()},
                'total_bits': float(total_bits) if total_bits else None,
                'total_tokens': int(total_toks) if total_toks else None,
                'bytes': int(val_ds.num_bytes),     # the validation split's, of which the tokens scored are a part
            })
        except FileNotFoundError as e:
            print(f'  {rid}: no checkpoint ({e})')
            rec.add(rid, {'error': f'no checkpoint: {e}'})

    rec.write()


if __name__ == '__main__':
    main()
