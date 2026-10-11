#!/usr/bin/env python3
"""
ukwiki_stats.py — size the Ukrainian Wikipedia dump and choose the train
fraction (docs/reference/ukrainian_pipeline.md, block U0).

Applies EXACTLY the split `load_corpus` applies (a hash of each title; 1 in
`--holdout` articles to validation, 1 to test), counts characters per split,
and suggests `data.corpus_fraction` so the train split holds about as much text
as WikiText-103's train split — the same number of epochs in the same 50k
steps, so the Ukrainian runs sit in the English runs' regime.

    python scripts/data/ukwiki_stats.py                  # downloads the dump once (~1.5 GB)
    python scripts/data/ukwiki_stats.py --target-chars 540e6
"""
import os
import sys
import hashlib
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', type=int, default=2000)
    ap.add_argument('--target-chars', type=float, default=539e6,
                    help='WikiText-103 train is ~539M characters of raw text')
    args = ap.parse_args()
    from datasets import load_dataset
    from morphlm.record import AuditRecord
    rec = AuditRecord('ukwiki_stats', args)
    ds = load_dataset('wikimedia/wikipedia', '20231101.uk', split='train')
    n = {'train': 0, 'validation': 0, 'test': 0}
    chars = {'train': 0, 'validation': 0, 'test': 0}
    train_sub = []                                   # (sub-hash, chars) per train article
    for art in ds:
        key = hashlib.md5(art['title'].encode('utf-8')).hexdigest()
        h = int(key[:8], 16) % args.holdout
        which = 'validation' if h == 0 else 'test' if h == 1 else 'train'
        c = len(art['text'])
        n[which] += 1
        chars[which] += c
        if which == 'train':
            train_sub.append((int(key[8:16], 16) % 10000, c))
    for s in n:
        print(f'  {s:10} {n[s]:9,} articles  {chars[s] / 1e6:9.1f}M characters')
    # the fraction whose train split reaches the target (sub-hash < frac*10000)
    per_bucket = [0] * 10000
    for b, c in train_sub:
        per_bucket[b] += c
    acc, frac = 0, 1.0
    for b in range(10000):
        acc += per_bucket[b]
        if acc >= args.target_chars:
            frac = (b + 1) / 10000
            break
    got = sum(per_bucket[:int(round(frac * 10000))])
    print(f'\n  suggested data.corpus_fraction = {frac:g}  -> train {got / 1e6:.1f}M characters '
          f'(target {args.target_chars / 1e6:.1f}M)')
    print(f'  validation {chars["validation"] / 1e6:.2f}M characters '
          f'(WikiText-103 validation is ~1.1M)')
    rec.add('splits', {'articles': n, 'chars': chars})
    rec.add('suggested_fraction', {'fraction': frac, 'train_chars': got})
    rec.write()


if __name__ == '__main__':
    main()
