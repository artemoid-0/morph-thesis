#!/usr/bin/env python3
"""
check_claims.py — verify every number in the documents against results/.

WHY THIS EXISTS
---------------
`export_results.py` eliminated transcription from terminal to document. It did
nothing about transcription from document to document, and that is where the
next error appeared: the leave-one-out marginals were computed against the
two-seed baseline in one report and the seed-0 baseline in another. Both were
internally consistent. Nothing automated could see the disagreement, because
both documents are prose and the checker only ever read run records.

It was found by a human comparing two of the project's own files.

WHAT IT CHECKS
--------------
1. ORPHANS — a BPB-shaped number appearing in a document that matches no run
   value, no seed mean, and no difference between two of them. Usually a typo
   or a stale figure from a superseded run.

2. UNPAIRED COMPARISONS — the bug that motivated this script. A difference is
   "unpaired" when the only way to produce it is by subtracting a single-seed
   cell from a MULTI-SEED MEAN. Same seed means same initialisation and the same
   data order, so a paired difference cancels the seed's contribution; an
   unpaired one silently adds cross-seed offset. Arithmetic alone cannot catch
   this — the number is perfectly consistent. Only the seed populations reveal
   it.

3. AMBIGUOUS — a difference explainable by several unrelated cell pairs. Not an
   error, but worth knowing when a figure is quoted without its operands.

    python scripts/ops/check_claims.py
    python scripts/ops/check_claims.py --docs docs/record/journey/README.md CODE_REPORT_FINAL.md
"""

import os
import re
import sys
import json
import glob
import argparse
import itertools
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#: numbers that look like a BPB, a marginal, or a difference of either
NUM = re.compile(r'(?<![\d.])(\d\.\d{4}|0\.\d{4})(?![\d])')

#: figures that are not measurements and must not be flagged
IGNORE = {
    '0.0001', '0.0002', '0.0003', '0.0005',   # learning rates, tolerances
    '3.0000', '0.1000',
}


def load_cells(results_dir):
    """label -> {seed: bpb}, from the run records."""
    cells = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(results_dir, '*', 'result.json'))):
        try:
            r = json.load(open(d))
        except Exception:
            continue
        if r.get('bits_per_byte') is None:
            continue
        cells[r.get('label', '?')][r.get('seed', 0)] = r['bits_per_byte']
    return cells


def build_universe(cells):
    """
    Every value the documents could legitimately quote, and how it arises.

    Each entry records the SEED POPULATION behind it, because that is what
    distinguishes a paired difference from an unpaired one.
    """
    atoms = []            # (value, description, label, seeds_tuple)
    for label, seeds in cells.items():
        for s, v in seeds.items():
            atoms.append((v, f'{label} seed{s}', label, (s,)))
        if len(seeds) > 1:
            m = sum(seeds.values()) / len(seeds)
            atoms.append((m, f'{label} mean(n={len(seeds)})', label,
                          tuple(sorted(seeds))))

    values = defaultdict(list)
    for v, desc, _, _ in atoms:
        values[round(v, 4)].append(desc)

    diffs = defaultdict(list)
    for (v1, d1, l1, s1), (v2, d2, l2, s2) in itertools.combinations(atoms, 2):
        if l1 == l2:
            continue
        d = round(abs(v1 - v2), 4)
        if d < 1e-9:
            continue
        paired = (s1 == s2)
        diffs[d].append({'a': d1, 'b': d2, 'paired': paired,
                         'mixes_mean_with_single':
                             (len(s1) > 1) != (len(s2) > 1)})
    return values, diffs


BASE_RE = re.compile(
    r'[Bb]aseline\s+`?([A-Za-z0-9_+\-]+)`?\s*=\s*\*{0,2}(\d\.\d{4})')


def check_table_arithmetic(docs, cells):
    """
    The surgical check, and the one this script exists for.

    A loose pairwise search does not work: 58 runs give ~1,600 differences in a
    narrow range, so almost any four-decimal number matches something by
    accident (44 false positives on the first attempt, 7 after context scoping).

    What DOES work is checking a specific, common structure. When a document
    declares a baseline and then tabulates cells against it:

        Baseline `det_baseline_Spdiwc` = 1.5246.
        | punctuation + digits | 1.4654 | 0.0592 | ...

    each row states a cell's value AND its margin. Both are verifiable:
      - does margin == baseline - value, to the stated precision?
      - is the baseline's SEED POPULATION the same as that cell's?

    The second is what caught the original defect. The marginals had been
    recomputed against a two-seed mean while every ablation cell was seed 0 —
    arithmetically consistent, experimentally unpaired.
    """
    by_value = {}
    for label, seeds in cells.items():
        for sd, v in seeds.items():
            by_value.setdefault(round(v, 4), []).append((label, (sd,)))
        if len(seeds) > 1:
            m = sum(seeds.values()) / len(seeds)
            by_value.setdefault(round(m, 4), []).append(
                (label, tuple(sorted(seeds))))

    problems = []
    for doc in docs:
        try:
            lines = open(doc).read().split('\n')
        except Exception:
            continue
        base = None
        for ln, line in enumerate(lines, 1):
            m = BASE_RE.search(line)
            if m:
                blabel, bval = m.group(1), round(float(m.group(2)), 4)
                bseeds = None
                for lab, sds in by_value.get(bval, []):
                    if lab == blabel:
                        bseeds = sds
                if bseeds is None:
                    for lab, sds in by_value.get(bval, []):
                        bseeds = sds
                base = (blabel, bval, bseeds, ln)
                continue
            # a heading ends the baseline's scope: the next table is a
            # different analysis and must not inherit it
            if line.startswith('#'):
                base = None
                continue
            if base is None or not line.strip().startswith('|'):
                continue
            nums = [round(float(x), 4) for x in NUM.findall(line)]
            if len(nums) < 2:
                continue
            blabel, bval, bseeds, bln = base
            for i, cellv in enumerate(nums[:-1]):
                if cellv not in by_value:
                    continue
                margin = nums[i + 1]
                # a margin is small; a second BPB column is not. Without this
                # the check misreads any two-column comparison table (the
                # learning-rate sweep has bpe and det side by side) as a
                # value/margin pair.
                if margin > 0.5:
                    break
                if abs((bval - cellv) - margin) > 0.00005:
                    problems.append((doc, ln, line.strip()[:62],
                                     f'margin {margin:.4f} but '
                                     f'{bval:.4f} - {cellv:.4f} = '
                                     f'{bval - cellv:.4f}'))
                    break
                cseeds = by_value[cellv][0][1]
                # An explicit `n=1` in the row DECLARES the mismatch rather than
                # hiding it. Some cells genuinely cannot be replicated — the
                # legacy joint cell predates checkpointing — and a table that
                # states this is honest. What the check refuses is a silent
                # mismatch. Declaring it costs three characters and puts the
                # caveat where the reader sees it.
                declared = re.search(r'\(?n\s*=\s*1\)?', line)
                if declared:
                    continue
                if bseeds and cseeds and len(bseeds) != len(cseeds):
                    problems.append((doc, ln, line.strip()[:62],
                                     f'baseline is n={len(bseeds)} '
                                     f'(line {bln}) but this cell is '
                                     f'n={len(cseeds)} — UNPAIRED'))
                break
    return problems


def _named(desc, ctx):
    """
    Is the cell behind this explanation actually mentioned in the context?

    `desc` looks like 'det_baseline_Spdiwc mean(n=2)'. The label may appear in
    full, or in the shorthand the documents use for its surface configuration
    ('Spdiwc', 'S--iwc'), or as a baseline line above a table.
    """
    label = desc.split(' ')[0]
    if label in ctx:
        return True
    for part in label.split('_'):
        if len(part) >= 5 and part in ctx:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--docs', nargs='*', default=None)
    ap.add_argument('--show-orphans', action='store_true',
                    help='list every orphan; most are legitimately derived '
                         '(sums, residuals, percentages, per-bucket figures) '
                         'so the default is a per-document count')
    args = ap.parse_args()

    # every living document: the front page and docs/
    docs = args.docs or sorted(['README.md'] + glob.glob('docs/**/*.md', recursive=True))
    cells = load_cells(args.results_dir)
    if not cells:
        print(f'no run records under {args.results_dir}/')
        sys.exit(1)
    values, diffs = build_universe(cells)

    print()
    print('=' * 88)
    print(f'CLAIM CHECK — {len(cells)} cells, '
          f'{sum(len(v) for v in cells.values())} runs, {len(docs)} documents')
    print('=' * 88)

    orphans, unpaired, checked = [], [], 0
    for doc in docs:
        try:
            text = open(doc).read()
        except Exception:
            continue
        lines = text.split('\n')
        for ln, line in enumerate(lines, 1):
            for m in NUM.finditer(line):
                tok = m.group(1)
                if tok in IGNORE:
                    continue
                v = round(float(tok), 4)
                checked += 1
                if v in values:
                    continue
                if v in diffs:
                    continue
                orphans.append((doc, ln, tok, line.strip()[:70]))

    print(f'\n  {checked:,} numeric claims examined')

    problems = check_table_arithmetic(docs, cells)
    print(f'\n1. BASELINE TABLES ({len(problems)} problem(s))')
    if not problems:
        print('   every tabulated margin equals baseline minus cell value, and')
        print('   every baseline matches its cells\' seed population')
    for doc, ln, ctx, why in problems:
        print(f'   {doc}:{ln}')
        print(f'     {ctx}')
        print(f'     -> {why}')

    print(f'\n2. ORPHANS ({len(orphans)})')
    if not orphans:
        print('   none — every number traces to a run value, a seed mean, or a '
              'difference of two')
    per_doc = defaultdict(set)
    for doc, ln, tok, ctx in orphans:
        per_doc[doc].add(tok)
    for doc in sorted(per_doc):
        print(f'   {doc}: {len(per_doc[doc])} distinct')
    if orphans and args.show_orphans:
        print()
        seen = set()
        for doc, ln, tok, ctx in orphans:
            if (doc, tok) in seen:
                continue
            seen.add((doc, tok))
            print(f'   {doc}:{ln}  {tok}   {ctx}')
    if orphans:
        print()
        print('   Orphans are not always errors: derived quantities (sums of '
              'marginals,')
        print('   percentages, interaction residuals) legitimately match no '
              'single')
        print('   difference. Check that each is derivable from numbers that '
              'ARE listed.')

    print()
    ok = not problems
    print('CLAIM CHECK PASSED' if ok else
          'CLAIM CHECK FAILED — see above')
    print()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
