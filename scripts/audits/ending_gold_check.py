#!/usr/bin/env python3
"""
ending_gold_check.py — does the ending rule cut where a morpheme dictionary does?
(DIRECTIONS D7; morphlm/data/endings.py; journey §33.61)

Two Ukrainian morpheme dictionaries segment their entries — citation forms —
into morphemes: Polyga (2009) and Yatsenko (1981), as digitized by Borodavko
(2025) in his thesis repository (github.com/V1King30/morpheme-based-tokenization;
no licence is stated, so it is cloned into `external/`, which git ignores and no
pack or archive carries, and read there). For every entry this reports whether the
rule cuts it, and if so whether the cut falls on a dictionary morpheme boundary
and whether it isolates exactly the final morpheme — the ending, where the word
has one. Citation forms test the rule's stem (the lemma minus its citation
ending); inflected forms reuse that stem, so a wrong stem here is wrong there.

    git clone https://github.com/V1King30/morpheme-based-tokenization external/morpheme-based-tokenization
    python scripts/audits/ending_gold_check.py

Writes results/audits/ending_gold_<UTC timestamp>.json.
"""
import os
import re
import sys
import json
import argparse
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

STRESS = '́'
SEGMENTED = re.compile(r"^[а-щьюяєіїґ'’ʼ]+(?:/[а-щьюяєіїґ'’ʼ]+)+$")


def polyga_entries(path):
    """'абетк/а (~тц/і, ~ток)' -> 'абетк/а': the citation form's segmentation."""
    for line in open(path, encoding='utf-8'):
        main = line.split('(')[0].strip().replace(STRESS, '').lower()
        if main:
            yield main


def yatsenko_entries(path):
    for w in json.load(open(path, encoding='utf-8')).get('words', []):
        main = str(w).split('(')[0].strip().replace(STRESS, '').lower()
        if main:
            yield main


def check(entries, rule):
    c = collections.Counter()
    by_pos = collections.defaultdict(collections.Counter)
    ends = collections.Counter()
    seen = set()
    for seg in entries:
        if not SEGMENTED.match(seg):
            c['skipped (not a clean segmentation)'] += 1
            continue
        parts = seg.split('/')
        word = ''.join(parts)
        if word in seen:
            continue
        seen.add(word)
        bounds, at = [], 0
        for p in parts[:-1]:
            at += len(p)
            bounds.append(at)
        k = rule.cut(word)
        pos = str(rule.morph.parse(word.replace('’', "'").replace('ʼ', "'"))[0].tag.POS)
        c['words'] += 1
        by_pos[pos]['words'] += 1
        if not k:
            continue
        c['cut'] += 1
        by_pos[pos]['cut'] += 1
        if k in bounds:
            c['on a morpheme boundary'] += 1
            by_pos[pos]['on a morpheme boundary'] += 1
        if k == bounds[-1]:
            c['isolates the final morpheme'] += 1
            by_pos[pos]['isolates the final morpheme'] += 1
        else:
            ends[f'{word[:k]}|{word[k:]}  (dictionary {seg})'] += 1
    return c, by_pos, ends


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    clone = os.path.join(ROOT, 'external', 'morpheme-based-tokenization', 'data', 'processed')
    ap.add_argument('--polyga', default=os.path.join(clone, 'polyga_clean_word_data.txt'))
    ap.add_argument('--yatsenko', default=os.path.join(clone, 'yatsenko_morph_words_processed.json'))
    ap.add_argument('--examples', type=int, default=12, help='disagreements to print per dictionary')
    ap.add_argument('--audit-dir', default=os.path.join(ROOT, 'results', 'audits'))
    a = ap.parse_args()
    present = [p for p in (a.polyga, a.yatsenko) if p and os.path.exists(p)]
    if not present:
        sys.exit('no dictionary found — clone them into external/ (see the docstring)')
    from morphlm.data.endings import EndingRule, RULE
    from morphlm.record import AuditRecord
    rule = EndingRule()
    record = {'rule': RULE, 'fingerprint': rule.fingerprint(), 'dictionaries': {}}
    for name, path, reader in (('Polyga 2009', a.polyga, polyga_entries), ('Yatsenko 1981', a.yatsenko, yatsenko_entries)):
        if not path or not os.path.exists(path):
            print(f'\n  {name}: {path} not found — skipped')
            continue
        c, by_pos, ends = check(reader(path), rule)
        n, cut = c['words'], c['cut']
        print(f'\n  {name}: {n:,} distinct segmented words ({c["skipped (not a clean segmentation)"]:,} entries skipped)')
        print(f'    cut by the rule        {cut:7,}  {100 * cut / max(n, 1):5.1f}% of words')
        print(f'    on a morpheme boundary {c["on a morpheme boundary"]:7,}  {100 * c["on a morpheme boundary"] / max(cut, 1):5.1f}% of cuts')
        print(f'    the final morpheme     {c["isolates the final morpheme"]:7,}  {100 * c["isolates the final morpheme"] / max(cut, 1):5.1f}% of cuts')
        print(f'    {"part of speech":10} {"words":>7} {"cut":>6} {"on boundary":>12} {"final":>7}')
        for pos, pc in sorted(by_pos.items(), key=lambda kv: -kv[1]['words'])[:8]:
            pcut = max(pc['cut'], 1)
            print(f'    {pos:10} {pc["words"]:7,} {100 * pc["cut"] / pc["words"]:5.1f}% '
                  f'{100 * pc["on a morpheme boundary"] / pcut:11.1f}% {100 * pc["isolates the final morpheme"] / pcut:6.1f}%')
        if a.examples:
            print('    cuts that are not the final morpheme, e.g.:')
            for e, _ in list(ends.items())[:a.examples]:
                print(f'      {e}')
        record['dictionaries'][name] = {'path': os.path.relpath(path, ROOT), **c,
                                        'by_pos': {k: dict(v) for k, v in by_pos.items()}}
    rec = AuditRecord('ending_gold', a, out_dir=a.audit_dir)
    for k, v in record.items():
        rec.add(k, v)
    print(f'\n  -> {rec.write(verbose=False)}')


if __name__ == '__main__':
    main()
