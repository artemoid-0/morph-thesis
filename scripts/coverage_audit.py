#!/usr/bin/env python3
"""
coverage_audit.py — what fraction of the corpus does each pipeline ACTUALLY encode?

THE ASYMMETRY THIS MEASURES
---------------------------
BasePreprocessor.encode() contains:

    if not spacy_tok.is_alpha:
        continue

so the morpheme pipelines emit NOTHING for punctuation, digits or symbols.
BPE's Whitespace pre-tokenizer splits on `\\w+|[^\\w\\s]+`, so BPE DOES emit
punctuation tokens and must predict them.

Bits-per-byte divides both models by the same full byte count. If one pipeline
silently declines to encode 8% of the corpus, it is being charged for bytes it
never attempts — a straight discount on its BPB that has nothing to do with
segmentation quality.

Unlike the collision problem, this is not a shared limitation: it is asymmetric
between the two arms of the Q1 comparison, so it biases the result directly.

Whitespace is dropped by BOTH pipelines (HuggingFace Whitespace() does not
preserve it either), so that channel is symmetric and reported separately.

WHAT IT REPORTS
---------------
Per pipeline, the share of corpus bytes falling into:
    covered      appears in the emitted token stream
    whitespace   dropped by both (symmetric, not a bias)
    punctuation  dropped by morph, kept by BPE  <-- the asymmetry
    digits       same
    other        anything else unaccounted

    python scripts/coverage_audit.py --lines 2000
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def classify(ch):
    if ch.isspace():
        return 'whitespace'
    if ch.isdigit():
        return 'digits'
    if ch.isalpha():
        return 'alpha'
    return 'punctuation'


def audit_pipeline(name, cfg_overrides, ds, lines):
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline

    cfg = apply_overrides(ExperimentConfig(), cfg_overrides)
    pipe = build_pipeline(cfg, verbose=False)

    byte_class = Counter()      # every byte of the corpus, by character class
    covered = Counter()         # bytes that appear in emitted tokens, by class
    total_bytes = 0
    processed = 0

    for row in ds:
        if processed >= lines:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue

        for ch in text:
            byte_class[classify(ch)] += len(ch.encode('utf-8'))
        total_bytes += len(text.encode('utf-8'))

        toks = pipe.encode(text)
        if not toks:
            continue

        # multiset of characters the stream actually contains
        emitted = Counter()
        for t in toks:
            for ch in str(t['token']):
                emitted[ch] += 1

        original = Counter(text.lower())
        for ch, n in original.items():
            got = min(n, emitted.get(ch, 0))
            covered[classify(ch)] += got * len(ch.encode('utf-8'))

        processed += 1
        if processed % 500 == 0:
            print(f'    {name}: {processed} lines...')

    return name, total_bytes, byte_class, covered, processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lines', type=int, default=1000)
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)

    results = []
    for name, ov in [('bpe', ['data.pipeline=bpe']),
                     ('det', ['data.pipeline=det'])]:
        print(f'building {name}...')
        results.append(audit_pipeline(name, ov, ds, args.lines))

    print()
    print('=' * 72)
    print(f'COVERAGE AUDIT — {args.lines} {args.split} lines')
    print('=' * 72)

    for name, total, byte_class, covered, processed in results:
        print()
        print(f'  {name}:  {total:,} bytes over {processed} lines')
        print(f'    {"class":12} {"in corpus":>12} {"covered":>12} {"share":>8}')
        for cls in ('alpha', 'digits', 'punctuation', 'whitespace'):
            inc = byte_class.get(cls, 0)
            cov = covered.get(cls, 0)
            pct = 100.0 * cov / inc if inc else 0.0
            print(f'    {cls:12} {inc:12,} {cov:12,} {pct:7.1f}%')
        tot_cov = sum(covered.values())
        print(f'    {"TOTAL":12} {total:12,} {tot_cov:12,} '
              f'{100.0*tot_cov/max(total,1):7.1f}%')

    # ── the asymmetry ─────────────────────────────────────────────────────────
    if len(results) == 2:
        (_, tb, bc_b, cv_b, _), (_, td, bc_d, cv_d, _) = results
        print()
        print('  ASYMMETRY (bytes one pipeline encodes and the other does not):')
        for cls in ('digits', 'punctuation'):
            b = cv_b.get(cls, 0)
            d = cv_d.get(cls, 0)
            inc = bc_b.get(cls, 0)
            print(f'    {cls:12} bpe covers {b:9,}  det covers {d:9,}  '
                  f'gap {b-d:+9,} of {inc:,} in corpus')
        gap_bytes = (cv_b.get('digits', 0) - cv_d.get('digits', 0)) \
            + (cv_b.get('punctuation', 0) - cv_d.get('punctuation', 0))
        print()
        print(f'    det declines to encode {gap_bytes:,} bytes that bpe encodes')
        print(f'    = {100.0*gap_bytes/max(tb,1):.2f}% of the corpus')
        print()
        print('    Those bytes still sit in the BPB denominator for BOTH models,')
        print('    so det gets a free discount proportional to this share. At an')
        print('    assumed 2-4 bits per dropped character the implied bias is:')
        for bits in (2.0, 3.0, 4.0):
            # dropped CHARACTERS, approximated by bytes for ASCII punctuation
            implied = gap_bytes * bits / max(tb, 1)
            print(f'      {bits:.0f} bits/char -> {implied:.4f} BPB '
                  f'({100*implied/0.0894:.0f}% of the 0.0894 gap)')
        print()
        print('    Combined with the measured collision correction of 0.0440,')
        print('    this determines whether the Q1 advantage survives at all.')
    print()


if __name__ == '__main__':
    main()
