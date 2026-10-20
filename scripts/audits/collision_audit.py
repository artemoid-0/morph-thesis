#!/usr/bin/env python3
"""
collision_audit.py — the RIGOROUS bits-per-byte correction.

WHY THIS SUPERSEDES reconstruction_audit.py
-------------------------------------------
The reconstruction audit asks "can I rebuild the surface string by
concatenating morphemes?" That is a useful smoke test but the wrong question
for BPB validity, and it OVERSTATES the problem.

What actually invalidates BPB is COLLISION: two different texts mapping to the
same token sequence. Then the model provably cannot distinguish them, and the
missing bits are real.

    cat  -> [cat]        COLLISION with cats -> [cat]
                         genuinely lossy, BPB inflated

    beautiful -> [beauty, ful]
                         concatenation gives "beautyful", so reconstruction
                         fails — but no OTHER word maps to [beauty, ful], so a
                         decode table recovers it. NOT a BPB problem.

BPE's merge table is side information too and nobody counts it; a
morpheme->surface table is the same kind of object. What cannot be excused is
two distinct inputs producing one output.

WHAT THIS COMPUTES
------------------
The exact context-free correction:

    correction_BPB = sum over word occurrences of -log2 P(surface | token_seq)
                     / total bytes

grouping surface forms by their token-ID sequence and estimating
P(surface | seq) from corpus frequencies.

This is an UPPER BOUND on the true correction, because a real decoder can use
sentence context to disambiguate (after "three ___" the plural is near-certain)
while this estimate assumes none. The true correction lies between 0 and this
number, and reporting the bound is the honest move.

    python scripts/audits/collision_audit.py --pipeline det --lines 2000
"""

import os
import sys
import math
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def group_words(tokens):
    """Yield (surface_word, token_id_tuple) for each source word in a stream."""
    # Skip ONLY whitespace: it sits between words and would otherwise merge
    # neighbouring groups. Everything else — punctuation, digits, inflection,
    # case — belongs to the word whose `word` field it carries. Skipping digits
    # previously made 29a collide with a, and 20th with 19th, which inflated the
    # measured correction by roughly 70%.
    # Whitespace SEPARATES groups rather than being skipped: skipping it made
    # two adjacent identical words merge, so '1 1' was recorded as '1' with the
    # sequence (1,1) and collided with '11'.
    cur_src, cur_ids = None, []
    for t in tokens:
        if t.get('category') == 'space':
            if cur_ids:
                yield cur_src, tuple(cur_ids)
                cur_ids, cur_src = [], None
            continue
        src = t.get('word')
        if src != cur_src and cur_ids:
            yield cur_src, tuple(cur_ids)
            cur_ids = []
        cur_src = src
        cur_ids.append(t['token_id'])
    if cur_ids:
        yield cur_src, tuple(cur_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', default='det', choices=['det', 'morph'])
    ap.add_argument('--lines', type=int, default=2000,
                    help='0 = the entire split. The paper quotes these sample sizes, so running the full split removes any question of sampling error.')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--gap', type=float, default=None,
                    help='observed BPB gap to compare against. NO DEFAULT on '
                         'purpose: the old 0.0894 came from a lossy pipeline '
                         'and a Whitespace BPE, so quoting it now would be '
                         'comparing against a number that no longer exists.')
    args = ap.parse_args()
    from morphlm.record import AuditRecord
    rec = AuditRecord('collision_audit', args)

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from datasets import load_dataset

    cfg = apply_overrides(ExperimentConfig(),
                          [f'data.pipeline={args.pipeline}'])
    print(f'building {args.pipeline} pipeline...')
    pipe = build_pipeline(cfg, verbose=False)

    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)
    from morphlm.progress import Progress
    prog = Progress(total=args.lines or None)

    # seq -> Counter(surface):  how many distinct words share a token sequence
    seq_to_surface = defaultdict(Counter)
    total_bytes = 0
    total_words = 0
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
        for surface, seq in group_words(toks):
            if surface is None:
                continue
            s = surface.lower()
            seq_to_surface[seq][s] += 1
            total_words += 1
        total_bytes += len(text.encode('utf-8'))
        processed += 1
        if processed % 500 == 0:
            print(f'  {processed} lines, {total_words:,} words | {prog.status(processed)}')

    # ── the correction ────────────────────────────────────────────────────────
    total_bits = 0.0
    colliding_occurrences = 0
    colliding_groups = 0
    worst = []

    for seq, counter in seq_to_surface.items():
        n = sum(counter.values())
        if len(counter) > 1:
            colliding_groups += 1
            colliding_occurrences += n
            for surface, c in counter.items():
                total_bits += c * -math.log2(c / n)
            worst.append((n, dict(counter)))

    worst.sort(reverse=True, key=lambda x: x[0])
    correction = total_bits / max(total_bytes, 1)

    print()
    print('=' * 70)
    print(f'COLLISION AUDIT — {args.pipeline}, {processed} {args.split} lines')
    print('=' * 70)
    print(f'  word occurrences        : {total_words:,}')
    print(f'  distinct token sequences: {len(seq_to_surface):,}')
    print(f'  COLLIDING groups        : {colliding_groups:,}')
    print(f'  occurrences in collision: {colliding_occurrences:,} '
          f'({100.0*colliding_occurrences/max(total_words,1):.2f}%)')
    print(f'  bytes                   : {total_bytes:,}')
    print()
    print(f'  MISSING INFORMATION     : {total_bits:,.0f} bits')
    print(f'  BPB CORRECTION (upper bd): {correction:.4f}')
    rec.add('summary', {'occurrences': total_words,
                        'distinct_sequences': len(seq_to_surface),
                        'colliding_groups': colliding_groups,
                        'occurrences_in_collision': colliding_occurrences,
                        'bytes': total_bytes,
                        'missing_information_bits': total_bits,
                        'bpb_correction_upper_bound': correction})
    rec.add('largest_groups', [
        {'occurrences': n, 'surfaces': dict(c)}
        for n, c in worst[:20]])
    rec.note('correction is an UPPER BOUND: expression is context-free, '
             'while a running model disambiguates from context. Measured '
             'looseness on the inflection channel: 30% (journey C-02).')
    print()
    if args.gap is None:
        print('  No reference gap supplied, and that is deliberate: pass')
        print('  --gap X once you have a CURRENT morph-vs-BPE difference.')
        print()
        if correction < 0.005:
            print('  This correction is negligible in absolute terms. The')
            print('  token stream is information-preserving, so BPB measured')
            print('  on it means what it claims.')
        else:
            print('  This correction is large enough to matter. Any BPB')
            print('  comparison using this pipeline needs it subtracted.')
    else:
        gap = args.gap
        print(f'  supplied gap            : {gap:.4f} BPB')
        print(f'  correction is {100*correction/gap:.0f}% of it')
        if correction < gap:
            print(f'  corrected advantage     : >= {gap-correction:.4f} BPB')
            print('  => survives the worst-case, context-free correction.')
        else:
            print('  => the correction EXCEEDS the gap; the advantage cannot')
            print('     be claimed from BPB.')

    print()
    print('  largest collision groups (surface forms sharing one token seq):')
    for n, members in worst[:20]:
        items = sorted(members.items(), key=lambda kv: -kv[1])
        shown = ', '.join(f'{k}:{v}' for k, v in items[:6])
        print(f'    {n:6,}  {{{shown}}}')
    print()

    rec.write()


if __name__ == '__main__':
    main()
