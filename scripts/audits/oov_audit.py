#!/usr/bin/env python3
"""
oov_audit.py — what falls through to character fallback, and why?

WHY THIS IS THE QUESTION NOW
----------------------------
The category bit accounting located the Q1 loss precisely:

    character   24.5% of tokens   24.8% of ALL BITS   2.654 bits/byte

Character fallback handles ~14% of the corpus and costs ~2.65 bits per byte on
that segment — roughly double both morph's own average (1.51) and BPE's (1.39).
Depending on what BPE actually pays for the same rare text, that excess accounts
for somewhere between half and all of the 0.11 BPB gap.

So the Q1 result is NOT 'morphological segmentation is worse'. Where morph can
segment, it does well: inflection costs 0.807 bits/token and suffixes 1.622,
both far cheaper than any BPE token — the model is demonstrably exploiting the
morphological structure. The loss is concentrated entirely in the words the
pipeline CANNOT analyse, which it spells out one character at a time.

That is the weakness journey sections 19, 21 and 26 already identified
qualitatively ('complex OOV -> characters, BPE produces shorter sequences').
This script quantifies it and, more usefully, classifies it — because the fix
depends on WHY a word fell through:

    unknown entity      -> the entity vocabulary is corpus-limited (freq gate)
    genuine OOV         -> not in MorphoLex at all
    number-adjacent     -> alphanumeric mixtures
    foreign / accented  -> outside the Latin morphology database

Each has a different remedy, and some are much cheaper than others.

    python scripts/audits/oov_audit.py --pipeline det --lines 1000
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def classify_failure(word, pipe):
    """Best-effort reason a word reached character fallback."""
    w = word.lower()
    if any(c.isdigit() for c in word):
        return 'alphanumeric'
    if not all(ord(c) < 128 for c in word):
        return 'non-ascii'
    if word[:1].isupper():
        return 'capitalised (likely entity)'
    if len(w) <= 3:
        return 'very short'
    if len(w) >= 13:
        return 'very long / compound'
    return 'plain lowercase OOV'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', default='det', choices=['det', 'morph'])
    ap.add_argument('--lines', type=int, default=1000,
                    help='0 = the entire split. The paper quotes these sample sizes, so running the full split removes any question of sampling error.')
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('oov_audit', args)

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from datasets import load_dataset

    cfg = apply_overrides(ExperimentConfig(),
                          [f'data.pipeline={args.pipeline}'])
    print(f'building {args.pipeline}...')
    pipe = build_pipeline(cfg, verbose=False)

    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)
    from morphlm.progress import Progress
    prog = Progress(total=args.lines or None)

    total_words = 0
    char_words = 0
    total_alpha_bytes = 0
    char_bytes = 0
    reasons = Counter()
    examples = Counter()
    processed = 0

    for row in ds:
        if args.lines and processed >= args.lines:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue
        toks = pipe.encode(text)
        if not toks:
            continue

        # group tokens by source word, ignoring standalone surface tokens
        groups = {}
        order = []
        for t in toks:
            if t.get('category') in ('space', 'punct', 'digit'):
                continue
            w = t.get('word')
            if w is None:
                continue
            if w not in groups:
                groups[w] = []
                order.append(w)
            groups[w].append(t)

        for w in order:
            ts = groups[w]
            if not any(c.isalpha() for c in str(w)):
                continue
            total_words += 1
            nb = len(str(w).encode('utf-8'))
            total_alpha_bytes += nb
            cats = [t['category'] for t in ts]
            # a word "fell through" if the majority of its tokens are chars/bytes
            n_char = sum(1 for c in cats if c in ('character', 'byte'))
            if n_char and n_char >= len(cats) / 2:
                char_words += 1
                char_bytes += nb
                reasons[classify_failure(str(w), pipe)] += 1
                examples[str(w)] += 1

        processed += 1
        if processed % 500 == 0:
            print(f'  {processed} lines, {total_words:,} words | {prog.status(processed)}')

    print()
    print('=' * 70)
    print(f'OOV AUDIT — {args.pipeline}, {processed} {args.split} lines')
    print('=' * 70)
    print(f'  alphabetic words          : {total_words:,}')
    print(f'  fell to character fallback: {char_words:,} '
          f'({100.0*char_words/max(total_words,1):.2f}%)')
    print(f'  bytes in those words      : {char_bytes:,} '
          f'({100.0*char_bytes/max(total_alpha_bytes,1):.2f}% of word bytes)')

    rec.add('summary', {
        'alphabetic_words': total_words,
        'character_fallback_words': char_words,
        'fallback_share': char_words / max(total_words, 1),
        'fallback_bytes': char_bytes,
        'fallback_byte_share_of_word_bytes':
            char_bytes / max(total_alpha_bytes, 1),
        'lines': processed, 'split': args.split, 'pipeline': args.pipeline,
    })
    rec.add('reasons', dict(reasons))

    print()
    print('  reasons:')
    for r, c in reasons.most_common():
        print(f'    {r:30} {c:7,} ({100.0*c/max(char_words,1):5.1f}%)')

    print()
    print('  most frequent unsegmentable words:')
    for w, c in examples.most_common(30):
        print(f'    {c:6,}  {w}')

    print()
    print('  READING THIS')
    cap = reasons.get('capitalised (likely entity)', 0)
    if cap > char_words * 0.3:
        print(f'    {100.0*cap/max(char_words,1):.0f}% are capitalised, i.e. named')
        print('    entities the frequency-gated entity vocabulary never')
        print('    admitted. That is a CORPUS COVERAGE limit, not a')
        print('    morphological one, and it is the cheapest thing to fix:')
        print('    rebuild the entity vocabulary over more articles, or lower')
        print('    the single-word threshold.')
    print('    Whatever the mix, remember these words cost ~2.65 bits/byte')
    print('    against a corpus average of ~1.5. Halving this population is')
    print('    worth more than any change to the morpheme analysis itself.')
    print()

    rec.write()


if __name__ == '__main__':
    main()
