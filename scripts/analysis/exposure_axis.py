#!/usr/bin/env python3
"""
exposure_axis.py — does case gain less where the tokenizer exposes endings, at the same point of
learning? (journey §33.113, §33.116–§33.118)

Against the best model without the feature, case at ~38M tokens and 10k steps gains less under
endings apart than under BPE 17k (+0.0022 against +0.0038 at width 128), and by 20k the two are
level. A wider model learns faster per step, so a step is not the same point of learning at two
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

Records: the second machine's by default (its twins are read from results/audits/device_*/), as
device_twins.py reads them. Written to results/audits/exposure_axis_<time>.json.
"""
import os
import sys
import math
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))

NAMES = {'bpe_uk_17k': 'BPE 17k', 'bpeend_uk_17k': 'endings apart', 'unigram_uk_17k': 'Unigram 17k',
         'bpe_uk_4k': 'BPE 4k', 'bpe_uk_8k': 'BPE 8k', 'bpe_uk_32k': 'BPE 32k'}


def label(fact, tok, budget, width, off=False):
    return (f'bpe_{fact}-LEARN-CAUSAL' + ('-OFF0' if off else '') + f'_{tok}_ukwiki0.05_st{budget // 1000}k'
            + ('' if width == 128 else f'_w{width}'))


def curve(r):
    return {e['step']: e['bits_per_byte'] for e in r.get('log', []) if e.get('step', 0) > 0}


def per_seed_curves(recs, machine, ref_label, case_label, budget):
    """{seed: [(reference BPB, gain)]} along training, each pair to the earlier of its best steps."""
    from device_twins import one
    out = {}
    for s in range(16):
        a = recs.get((ref_label, s, budget), {}).get(machine)
        b = recs.get((case_label, s, budget), {}).get(machine)
        if not a or not b:
            continue
        ra, rb = one(a)[0], one(b)[0]
        ca, cb = curve(ra), curve(rb)
        stop = min(ra.get('best_step') or budget, rb.get('best_step') or budget)
        pts = sorted(((ca[x], ca[x] - cb[x]) for x in ca if x in cb and x <= stop), reverse=True)
        if len(pts) >= 2:
            out[s] = {'points': pts, 'end': ra['bits_per_byte'] - rb['bits_per_byte'],
                      'runs': [ra['run_id'], rb['run_id']]}
    return out


def at(points, level):
    """The gain where the reference's BPB falls to `level`, interpolated; None outside the curve."""
    for (b0, g0), (b1, g1) in zip(points, points[1:]):
        if b0 >= level >= b1:
            t = (b0 - level) / (b0 - b1) if b0 > b1 else 0.0
            return g0 + t * (g1 - g0)
    return None


def mse(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0, 0
    m = st.mean(xs)
    se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
    return m, se, sum(x > 0 for x in xs), len(xs)


def fmt(m, se, pos, n):
    if m is None:
        return '—'
    return f'{m:+.4f}' + (f' ± {se:.4f}' if se is not None else '') + f' ({pos}/{n})'


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
    ap.add_argument('--root', default=ROOT)
    a = ap.parse_args()
    if a.feature == 'empty' and a.reference != 'null':
        ap.error('--feature empty is read over the null: add --reference null')
    from device_twins import load
    from morphlm.record import AuditRecord
    recs = load(a.root)
    rec = AuditRecord('exposure_axis', a, out_dir=os.path.join(a.root, 'results', 'audits'))
    out = {}
    for w in a.widths:
        for tok in a.tokenizers:
            pair = [a.against, tok]
            cur = {}
            for t in pair:
                ref = label('mconst', t, a.budget, w, off=(a.reference == 'empty'))
                feat = label('mconst', t, a.budget, w, off=True) if a.feature == 'empty' else label('ucase', t, a.budget, w)
                cur[t] = per_seed_curves(recs, a.machine, ref, feat, a.budget)
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
                gb = {s: at(cur[a.against][s]['points'], v) for s in seeds}
                ge = {s: at(cur[tok][s]['points'], v) for s in seeds}
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
