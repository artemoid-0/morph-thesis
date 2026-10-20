#!/usr/bin/env python3
"""
exposure_axis.py — does case gain less where the tokenizer exposes endings, at the same point of
learning? (journey §33.113, §33.116–§33.118)

Against the best model without the feature, case at ~38M tokens and 10k steps gains less under
endings apart than under BPE 17k (journey §33.113), and by 20k the two are level (§33.116). A wider
model learns faster per step, so a step is not the same point of learning at two
widths. Here each tokenizer's gain is read along training against its reference's bits per byte —
the law's axis — and the two tokenizers are compared at common levels, per width, within one
machine (FINDINGS, method point 11).

For each width and tokenizer, case's gain over the reference (the empty slot, or with --reference
null the same-shape null) at every evaluation, paired by seed, each pair to the earlier of its two
best steps; per seed it is interpolated at common reference levels, and the redundancy at a level
is BPE 17k's gain minus endings apart's, seed by seed. The end of the budget is reported too. With
--feature empty the gain read is the empty slot's over the null instead (method point 10 along
training; journey §33.118).

    python scripts/analysis/exposure_axis.py --widths 128 512
    python scripts/analysis/exposure_axis.py --widths 128 --reference null --levels 1.15 1.10 1.05
    python scripts/analysis/exposure_axis.py --widths 128 512 --feature empty --reference null

Records: the second machine's by default, its twin copies included (results/audits/device_*/),
as `morphlm.runs.load(machine=...)` reads them; --machine thesis for the thesis machine's.
Written to results/audits/exposure_axis_<time>.json.
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs  # noqa: E402

NAMES = {'bpe_uk_17k': 'BPE 17k', 'bpeend_uk_17k': 'endings apart', 'unigram_uk_17k': 'Unigram 17k',
         'bpe_uk_4k': 'BPE 4k', 'bpe_uk_8k': 'BPE 8k', 'bpe_uk_32k': 'BPE 32k'}


def label(fact, tok, budget, width, off=False):
    return (f'bpe_{fact}-LEARN-CAUSAL' + ('-OFF0' if off else '') + f'_{tok}_ukwiki0.05_st{budget // 1000}k'
            + ('' if width == 128 else f'_w{width}'))


def per_seed_curves(R, ref_label, case_label):
    """{seed: [(reference BPB, gain)]} along training, each pair to the earlier of its best steps,
    the points sorted by the reference's BPB (R: one machine's records)."""
    out = {}
    pairs, _ = runs.pair(R, ref_label, case_label)
    for s, (ra, rb) in pairs.items():
        pts, _, _ = runs.gain_curve({(ref_label, s): ra, (case_label, s): rb}, ref_label, case_label)
        pts = sorted(((b, g) for _, b, g, _ in pts), reverse=True)
        if len(pts) >= 2:
            out[s] = {'points': pts, 'end': ra['bits_per_byte'] - rb['bits_per_byte'],
                      'runs': [ra['run_id'], rb['run_id']]}
    return out


def mse(xs):
    """(mean, se — None at one seed, seeds positive, seeds), as this script's records hold it."""
    s = runs.summary(xs)
    return (s['mean'], s['se'], s['positive'], s['n']) if s else (None, None, 0, 0)


def fmt(m, se, pos, n):
    return runs.fmt(None if m is None else {'mean': m, 'se': se, 'positive': pos, 'n': n})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--widths', nargs='+', type=int, default=[128, 512])
    ap.add_argument('--budget', type=int, default=10000)
    ap.add_argument('--reference', choices=('empty', 'null'), default='empty')
    ap.add_argument('--feature', choices=('case', 'empty'), default='case',
                    help="whose gain is read: case's, or the empty slot's (over the null only)")
    ap.add_argument('--machine', choices=('second', 'thesis'), default='second')
    ap.add_argument('--levels', nargs='*', type=float, default=None,
                    help="reference BPB levels (default: six across each width's common range)")
    ap.add_argument('--tokenizers', nargs='+', default=['bpeend_uk_17k'],
                    help='each compared with --against at common levels')
    ap.add_argument('--against', default='bpe_uk_17k', help='the tokenizer that hides endings')
    runs.add_results_flag(ap, root=True)
    a = ap.parse_args()
    if a.feature == 'empty' and a.reference != 'null':
        ap.error('--feature empty is read over the null: add --reference null')
    from morphlm.record import AuditRecord
    R = runs.load(runs.results_of(a), machine=a.machine)
    rec = AuditRecord('exposure_axis', a, out_dir=os.path.join(a.root, 'results', 'audits') if a.root else runs.AUDITS)
    out = {}
    for w in a.widths:
        for tok in a.tokenizers:
            pair = [a.against, tok]
            cur = {}
            for t in pair:
                ref = label('mconst', t, a.budget, w, off=(a.reference == 'empty'))
                feat = label('mconst', t, a.budget, w, off=True) if a.feature == 'empty' else label('ucase', t, a.budget, w)
                cur[t] = per_seed_curves(R, ref, feat)
            seeds = sorted(set(cur[a.against]) & set(cur[tok]))
            what = 'the empty slot' if a.feature == 'empty' else 'case'
            print(f'\n  width {w}, {a.budget // 1000}k, {what} over the {"empty slot" if a.reference == "empty" else "null"}: '
                  f'{NAMES.get(a.against, a.against)} against {NAMES.get(tok, tok)}, the {a.machine} machine, seeds {seeds}')
            if not seeds:
                print('    no seed has both tokenizers — skipped')
                continue
            lo = max(min(b for b, _ in cur[t][s]['points']) for t in pair for s in seeds)
            hi = min(max(b for b, _ in cur[t][s]['points']) for t in pair for s in seeds)
            levels = a.levels or ([hi - (hi - lo) * k / 5 for k in range(6)] if hi > lo else [])
            rows = []
            print(f'    {"reference BPB":>13}  {NAMES.get(a.against, a.against):>24}  {NAMES.get(tok, tok):>24}  {"difference (redundancy)":>28}')
            for v in levels:
                gb = {s: runs.at(cur[a.against][s]['points'], v) for s in seeds}
                ge = {s: runs.at(cur[tok][s]['points'], v) for s in seeds}
                d = [gb[s] - ge[s] for s in seeds if gb[s] is not None and ge[s] is not None]
                r = {'level': v, 'against': mse(list(gb.values())), 'tokenizer': mse(list(ge.values())), 'redundancy': mse(d)}
                rows.append(r)
                print(f'    {v:13.4f}  {fmt(*r["against"]):>24}  {fmt(*r["tokenizer"]):>24}  {fmt(*r["redundancy"]):>28}')
            eb = [cur[a.against][s]['end'] for s in seeds]
            ee = [cur[tok][s]['end'] for s in seeds]
            end = {'against': mse(eb), 'tokenizer': mse(ee), 'redundancy': mse([x - y for x, y in zip(eb, ee)])}
            print(f'    {"end of budget":>13}  {fmt(*end["against"]):>24}  {fmt(*end["tokenizer"]):>24}  {fmt(*end["redundancy"]):>28}')
            out[f'{w}:{tok}'] = {'width': w, 'tokenizer': tok, 'against': a.against, 'seeds': seeds, 'levels': rows, 'end': end,
                                 'runs': {t: {str(s): cur[t][s]['runs'] for s in seeds} for t in pair},
                                 'curves': {t: {str(s): cur[t][s]['points'] for s in seeds} for t in pair}}
    rec.add('comparisons', out)
    rec.note(f'gain = the reference\'s bits per byte minus {"the empty slot" if a.feature == "empty" else "case"}\'s at the same step, paired by seed; '
             'each tuple (mean, se, seeds positive, seeds); redundancy = --against\'s gain minus the tokenizer\'s, per seed')
    rec.write()


if __name__ == '__main__':
    main()
