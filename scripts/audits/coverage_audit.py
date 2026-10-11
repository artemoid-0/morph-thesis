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

    python scripts/audits/coverage_audit.py --lines 2000
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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
        if lines and processed >= lines:
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

        # Multiset of characters the stream actually contains.
        #
        # CASE FOLDING IS APPLIED TO BOTH SIDES. An earlier version lowercased
        # only the source text while leaving the emitted tokens as-is, so every
        # capital letter in the original failed to match a BPE token that had
        # faithfully preserved it. That understated BPE's alpha coverage as
        # 95.0% when it is in fact near-total, and the figure needed a footnote
        # explaining that its own headline row was an artifact.
        #
        # Case is a SEPARATE information channel, measured by the `case` layer
        # and reported in journey section 17; conflating it with byte coverage
        # measures neither cleanly.
        # BYTE-FALLBACK TOKENS MUST BE DECODED, NOT READ LITERALLY.
        #
        # The byte layer writes `<b00>` .. `<bff>`. Counting the characters
        # inside that string contributes '<', 'b', '0', '0', '>' and NEVER the
        # byte it actually encodes, so every character routed to byte fallback
        # was invisible to this audit and counted as UNCOVERED.
        #
        # On 1,000 validation lines the effect was small enough to miss: det
        # read 100.0% punctuation coverage. On the full split it read 94.3%,
        # implying det "declines to encode" 2,153 punctuation bytes it does in
        # fact encode losslessly. Non-ASCII punctuation — en/em dashes, curly
        # quotes — is exactly what the 32-entry ASCII punct layer sends to byte
        # fallback, and exactly what the audit could not see.
        #
        # This is the third measurement artifact of the same family in this
        # codebase (journey sections 17, 21 and here): the instrument disagreed
        # with the thing it measured about what counts as encoded.
        emitted = Counter()
        pending = bytearray()

        def _flush():
            if not pending:
                return
            try:
                for ch in pending.decode('utf-8').lower():
                    emitted[ch] += 1
            except UnicodeDecodeError:
                pass                      # partial sequence at a chunk edge
            pending.clear()

        for t in toks:
            tok = str(t['token'])
            if t.get('category') == 'byte' and len(tok) == 5 and \
                    tok.startswith('<b') and tok.endswith('>'):
                try:
                    pending.append(int(tok[2:4], 16))
                    continue
                except ValueError:
                    pass
            _flush()
            for ch in tok.lower():
                emitted[ch] += 1
        _flush()

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
    ap.add_argument('--lines', type=int, default=1000,
                    help='0 = the entire split. The paper quotes these sample sizes, so running the full split removes any question of sampling error.')
    ap.add_argument('--split', default='validation')
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)

    from morphlm.record import AuditRecord
    rec = AuditRecord('coverage_audit', args)
    results = []
    for name, ov in [('bpe', ['data.pipeline=bpe']),
                     ('det', ['data.pipeline=det'])]:
        print(f'building {name}...')
        r = audit_pipeline(name, ov, ds, args.lines)
        results.append(r)
        rec.add(name, r)

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
        print('  WHY THE ALPHA ROW IS EXCLUDED FROM THE ASYMMETRY BELOW')
        print('    The morpheme pipelines store NORMALISED roots, so the')
        print('    character multiset legitimately changes: beautiful ->')
        print('    beauty+ful loses an "i" and gains a "y"; cities -> city+s')
        print('    loses an "e" and gains a "y". A character-multiset audit')
        print('    therefore measures LEMMA NORMALISATION on the alpha row, not')
        print('    coverage, and comparing it across pipelines would compare')
        print('    two different things.')
        print()
        print('    The information is not lost: collision_audit.py measures the')
        print('    residual ambiguity of the same stream at 0.0008 BPB over')
        print('    0.74% of word occurrences. A pipeline genuinely discarding')
        print('    ~2% of alpha bytes could not collide that rarely. Two')
        print('    independent instruments, and the second settles it.')
        print()
        print('    Digits and punctuation carry no normalisation, so their rows')
        print('    ARE comparable and are what the asymmetry uses.')
        print()
        rec.note('alpha excluded from the asymmetry: morpheme pipelines store '
                 'normalised roots, so the character multiset legitimately '
                 'changes (beautiful -> beauty+ful). collision_audit measures '
                 'the residual ambiguity of the same stream at 0.0008 BPB, '
                 'which settles that the information is not lost.')
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

    rec.write()


if __name__ == '__main__':
    main()
