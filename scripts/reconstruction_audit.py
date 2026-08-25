#!/usr/bin/env python3
"""
reconstruction_audit.py — is the morpheme token stream information-preserving?

WHY THIS DECIDES WHETHER Q1 IS VALID
------------------------------------
Bits-per-byte divides the model's total surprise by the byte count of the
ORIGINAL text. That normalization is only fair if both token streams encode the
same information. BPE is lossless by construction: concatenating its tokens
reproduces the input exactly.

The morpheme pipelines may not be. From research_journey.md's own tables:

    cats       -> cat            the plural -s is gone
    running    -> run            the progressive -ing is gone
    beautiful  -> beauty + ful   concatenates to "beautyful", not "beautiful"

MorphoLex stores NORMALIZED roots, and section 7 celebrates this as solving
morphophonological normalization "for free". For a tagger that is a feature.
For a LANGUAGE MODEL scored in bits-per-byte it is a problem: if the stream
cannot reconstruct the bytes, the model is predicting an easier, lower-entropy
sequence while being credited with the full original byte count.

That would make the observed -6.3% BPB advantage partly — possibly entirely —
an artifact of information destruction rather than better segmentation.

WHAT THIS SCRIPT MEASURES
-------------------------
    1. word-level exact reconstruction rate
    2. byte-level recoverable fraction
    3. a lower bound on the correction term that morph's BPB would need

It does NOT assume the answer. If reconstruction is ~100%, the Q1 result stands
as measured and this concern is closed. Run it before writing any Q1 claim.

    python scripts/reconstruction_audit.py                 # det, 300 lines
    python scripts/reconstruction_audit.py --pipeline morph --lines 1000
"""

import os
import sys
import math
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def naive_reconstruct(tokens):
    """
    Rebuild surface words from the emitted token stream, the most generous way
    a decoder could: concatenate all morphemes belonging to one source word.

    This is an UPPER bound on reconstruction quality — a real decoder has no
    word-boundary oracle. If even this fails, the stream is definitively lossy.
    """
    words = []
    current = []
    current_src = None
    for t in tokens:
        if t.get('category') == 'space':
            if current:
                words.append((current_src, ''.join(current)))
                current, current_src = [], None
            continue
        src = t.get('word')
        if src != current_src and current:
            words.append((current_src, ''.join(current)))
            current = []
        current_src = src
        current.append(t['token'])
    if current:
        words.append((current_src, ''.join(current)))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', default='det', choices=['det', 'morph'])
    ap.add_argument('--lines', type=int, default=300)
    args = ap.parse_args()

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from datasets import load_dataset

    cfg = apply_overrides(ExperimentConfig(),
                          [f'data.pipeline={args.pipeline}'])
    print(f'building {args.pipeline} pipeline...')
    pipe = build_pipeline(cfg, verbose=False)

    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')

    n_words = 0
    n_exact = 0
    orig_bytes = 0
    recov_bytes = 0
    failures = Counter()
    processed = 0

    for row in ds:
        if processed >= args.lines:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue
        tokens = pipe.encode(text)
        if not tokens:
            continue
        for src, rebuilt in naive_reconstruct(tokens):
            if src is None:
                continue
            src_l = src.lower()
            n_words += 1
            orig_bytes += len(src_l.encode('utf-8'))
            if rebuilt == src_l:
                n_exact += 1
                recov_bytes += len(src_l.encode('utf-8'))
            else:
                failures[(src_l, rebuilt)] += 1
        processed += 1

    print()
    print('=' * 66)
    print(f'RECONSTRUCTION AUDIT — {args.pipeline} on {processed} val lines')
    print('=' * 66)
    print(f'  words examined        : {n_words:,}')
    print(f'  exactly reconstructed : {n_exact:,} ({100.0*n_exact/max(n_words,1):.2f}%)')
    print(f'  NOT reconstructable   : {n_words-n_exact:,} '
          f'({100.0*(n_words-n_exact)/max(n_words,1):.2f}%)')
    print(f'  byte recovery         : {100.0*recov_bytes/max(orig_bytes,1):.2f}%')

    print()
    print('  most common reconstruction failures (original -> rebuilt):')
    for (src, got), c in failures.most_common(25):
        print(f'    {c:6,}  {src:22} -> {got}')

    loss_rate = (n_words - n_exact) / max(n_words, 1)
    print()
    print('  INTERPRETATION')
    if loss_rate < 0.005:
        print('    Stream is effectively lossless. The Q1 BPB comparison is')
        print('    measuring what it claims. This concern is CLOSED.')
    else:
        print(f'    {100*loss_rate:.1f}% of words cannot be recovered from the token')
        print('    stream, so the morpheme model predicts a REDUCED-INFORMATION')
        print('    sequence while BPB credits it with the full original bytes.')
        print('    The -6.3% BPB advantage is confounded by this and must be')
        print('    corrected or the comparison restated before it can be claimed.')
        print()
        print('    A crude lower bound on the missing information: recovering')
        print('    each failed word needs at least log2(#candidate surface forms)')
        print('    bits. Even at 1 bit per failed word:')
        extra_bits = (n_words - n_exact) * 1.0
        print(f'      {extra_bits:,.0f} bits over {orig_bytes:,} bytes '
              f'= {extra_bits/max(orig_bytes,1):.4f} BPB unaccounted')
        print(f'      (compare: the entire morph-vs-BPE gap is 0.0894 BPB)')
    print()


if __name__ == '__main__':
    main()
