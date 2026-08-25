#!/usr/bin/env python3
"""
multiroot_audit.py — how many MorphoLex entries did the multi-root bug affect?

The parser collected roots with re.search (first match only), while MorphoLex
writes compounds with several parenthesized roots: {(high)(way)},
{(touch)(down)}>s>, {(book)(keep)}>er>. Every root after the first was
discarded, so those words could not be rebuilt from their own token streams —
which is how the reconstruction audit surfaced it.

This reports the scale, the same way section 7 reported 4,941 format-B entries.
Run it after applying the fix; it compares old and new parses directly.

    python scripts/multiroot_audit.py
"""

import os
import sys
import re
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_old(s):
    """The buggy parser, kept here only to measure what it lost."""
    root_match = re.search(r'\(([a-zA-Z]+)\)', s)
    root = root_match.group(1) if root_match else None
    brace_open = s.index('{') if '{' in s else len(s)
    brace_close = s.rindex('}') + 1 if '}' in s else 0
    fa = re.findall(r'<([a-zA-Z]+)<', s[:brace_open])
    inner = s[brace_open:brace_close]
    pp = inner.index('(') if '(' in inner else len(inner)
    fb = re.findall(r'<([a-zA-Z]+)<', inner[:pp])
    prefixes = fa + fb
    after = re.sub(r'^[^)]*\)', '', inner).rstrip('}')
    suf = re.findall(r'>([a-zA-Z]+)>', after) \
        + re.findall(r'>([a-zA-Z]+)>', s[brace_close:])
    return prefixes + ([root] if root else []) + suf


def main():
    from morphlm.vocab.morpholex import _read_sheets, parse_morpholex_segm

    path = 'data/MorphoLEX_en.xlsx'
    if not os.path.exists(path):
        print(f'not found: {path}')
        sys.exit(1)

    df = _read_sheets(path, verbose=True)
    df = df[['Word', 'MorphoLexSegm']].dropna()

    total = 0
    affected = 0
    root_counts = Counter()
    examples = []
    recovered_morphemes = Counter()

    for _, row in df.iterrows():
        segm = str(row['MorphoLexSegm'])
        word = str(row['Word']).lower().strip()
        total += 1
        n_roots = len(re.findall(r'\(([a-zA-Z]+)\)', segm))
        root_counts[n_roots] += 1
        if n_roots > 1:
            affected += 1
            old = parse_old(segm)
            new, _ = parse_morpholex_segm(segm)
            gained = [m for m in new if m not in old]
            for g in gained:
                recovered_morphemes[g] += 1
            if len(examples) < 25:
                examples.append((word, segm, old, new))

    print()
    print('=' * 70)
    print('MULTI-ROOT AUDIT')
    print('=' * 70)
    print(f'  entries parsed          : {total:,}')
    print(f'  entries with >1 root    : {affected:,} '
          f'({100.0 * affected / max(total, 1):.2f}%)')
    print()
    print('  roots per entry:')
    for n in sorted(root_counts):
        print(f'    {n} root(s): {root_counts[n]:7,}')

    print()
    print('  examples of what the old parser dropped:')
    print(f'    {"word":20} {"old":28} new')
    for word, segm, old, new in examples:
        print(f'    {word:20} {str(old):28} {new}')

    print()
    print('  most frequently recovered morphemes:')
    for m, c in recovered_morphemes.most_common(20):
        print(f'    {c:6,}  {m}')

    print()
    print(f'  compare: the format-B fix of section 7 recovered prefixes for')
    print(f'  4,941 entries. This one affects {affected:,}.')
    print()
    print('  NOTE: fixing this changes the vocabulary (new roots become')
    print('  reachable) and therefore the token stream. The morph/det caches')
    print('  MUST be rebuilt, and any morph/det result predating the fix is')
    print('  not comparable to results after it.')


if __name__ == '__main__':
    main()
