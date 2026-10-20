#!/usr/bin/env python3
"""
tokenizer_table.py — D7's table: a feature's gain under each tokenizer, and the
tokenizers against each other (DIRECTIONS D7; journey §33.58, §33.65).

For every tokenizer of one condition (corpus, size, budget), from the run
records:

  * the null's final BPB per seed, and each feature's gain over its own null at
    the end of the budget, paired by seed (mean ± standard error, how many
    positive), and the first feature over the second (case over its word form);
  * each tokenizer against a reference one, paired by seed: null against null —
    at the end and along training (`--null-steps`, with the reference null's BPB
    there, the law's axis) — and the difference between
    their features' gains. Seeds pair initial weights only between identical
    shapes, which tokenizers of one vocabulary size have;
  * each feature's gain at a fixed step (2,000 by default);
  * each feature's gain at matched null BPB — the law's axis — from the curves
    averaged over the seeds both cells share, kept to each run's best step
    (`learning_law.py`), and per seed at the levels asked for.

Every comparison pairs runs of one machine (FINDINGS, method point 11): a seed whose
runs ran on two machines is named and left out.

Writes results/audits/tokenizer_table_<UTC timestamp>.json.

    python scripts/analysis/tokenizer_table.py
    python scripts/analysis/tokenizer_table.py --results results unpacked/results_74 --levels 1.40 1.25 1.15
"""
import os
import sys
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs  # noqa: E402

TOKENIZERS = ['bpe_uk_4k', 'bpe_uk_8k', 'bpe_uk_17k', 'bpe_uk_32k', 'unigram_uk_17k', 'bpeend_uk_17k']


def summary(xs):
    """runs.summary, with a standard error of 0.0 at one seed."""
    return runs.summary(xs, se_one=0.0)


show = runs.fmt


def bpb(r, step=None):
    """A run's bits per byte: at its end (the best step), or at `step` (None if never evaluated there)."""
    if step is None:
        return r['bits_per_byte']
    v = None
    for e in r['log']:
        if e['step'] == step:
            v = e['bits_per_byte']
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs.add_results_flag(ap)
    ap.add_argument('--corpus', default='ukwiki')
    ap.add_argument('--fraction', default='0.05')
    ap.add_argument('--budget', default='_st10k', help="the label's budget suffix ('' for 50k)")
    ap.add_argument('--tokenizers', nargs='+', default=TOKENIZERS)
    ap.add_argument('--reference', default='bpe_uk_17k')
    ap.add_argument('--null', default='mconst')
    ap.add_argument('--facts', nargs='+', default=['ucase', 'ucaselex'],
                    help='features over the null; the first over the second is reported too')
    ap.add_argument('--step', type=int, default=2000)
    ap.add_argument('--null-steps', nargs='+', type=int, default=[500, 1000, 2000, 3000, 5000, 7500],
                    help='steps at which each null is compared with the reference null as well')
    ap.add_argument('--levels', nargs='+', type=float, default=[1.60, 1.50, 1.40, 1.30, 1.25, 1.20, 1.15, 1.10, 1.08])
    ap.add_argument('--seed-levels', nargs='+', type=float, default=[1.40, 1.25, 1.15, 1.10])
    ap.add_argument('--audit-dir', default=os.path.join(ROOT, 'results', 'audits'))
    a = ap.parse_args()

    R = runs.load(runs.results_of(a))
    corpus = a.corpus + (a.fraction if a.corpus != 'wikitext103' else '')
    lab = lambda f, t: f'bpe_{f}-LEARN-CAUSAL_{t}_{corpus}{a.budget}'
    final = lambda f, t: {s: R[(l, s)]['bits_per_byte'] for (l, s) in R if l == lab(f, t)}
    left_out = {}

    def paired(cells, step=None):
        """{seed: [each cell's BPB]} — at the end, or at `step` — over the seeds at which every cell
        has a run, on one machine (and, at a step, an evaluation there); the others are noted."""
        pairs, left = runs.pair(R, *cells)
        if left:
            left_out.setdefault(' / '.join(cells), {}).update({str(s): list(m) for s, m in left.items()})
        out = {}
        for s, rs in pairs.items():
            v = [bpb(r, step) for r in rs]
            if None not in v:
                out[s] = v
        return out

    def gains(t, f, step=None):
        return {s: n - x for s, (n, x) in paired([lab(a.null, t), lab(f, t)], step).items()}

    def named(since):
        """Print the seeds left out since `since` (the size of left_out before a section)."""
        for cells in list(left_out)[since:]:
            print(f'  {"":18}   {cells}: ' + runs.left_out({int(s): m for s, m in left_out[cells].items()}))

    from morphlm.record import AuditRecord
    rec = AuditRecord('tokenizer_table', a, out_dir=a.audit_dir)
    rows = {}
    f1, f2 = (a.facts + [None])[:2]
    print(f'\n  {corpus}, budget {a.budget or "50k"} — gains over each tokenizer\'s own null, paired by seed, '
          f'at the end (mean ± se, positive/seeds)')
    print(f'  {"tokenizer":18} {"null (mean)":>11} ' + ' '.join(f'{f:>24}' for f in a.facts)
          + (f' {f1 + " over " + f2:>26}' if f2 else '') + f' {f1 + " at step " + str(a.step):>26}')
    for t in a.tokenizers:
        n = final(a.null, t)
        if not n:
            continue
        row = {'null': n, 'gain': {f: summary(gains(t, f).values()) for f in a.facts},
               'step': {f: summary(gains(t, f, a.step).values()) for f in a.facts}}
        if f2:
            row['first_over_second'] = summary([(v[0] - v[1]) - (v[0] - v[2]) for v in
                                                paired([lab(a.null, t), lab(f1, t), lab(f2, t)]).values()])
        rows[t] = row
        print(f'  {t:18} {st.mean(n.values()):11.4f} ' + ' '.join(f'{show(row["gain"][f]):>24}' for f in a.facts)
              + (f' {show(row["first_over_second"]):>26}' if f2 else '') + f' {show(row["step"][f1]):>26}')

    named(0)
    print(f'\n  against {a.reference}, paired by seed (same initial weights where the vocabularies match)')
    vs = {}
    since = len(left_out)
    for t in a.tokenizers:
        if t == a.reference or t not in rows:
            continue
        rn_, tn_ = lab(a.null, a.reference), lab(a.null, t)
        d = {'null': summary([r - n for r, n in paired([rn_, tn_]).values()])}
        for f in a.facts:
            # this tokenizer's gain minus the reference's: four runs a seed
            d[f] = summary([(v[0] - v[1]) - (v[2] - v[3]) for v in
                            paired([tn_, lab(f, t), rn_, lab(f, a.reference)]).values()])
            # the tokenizer and the feature together, against the reference null
            d[f + '_together'] = summary([r - x for r, x in paired([rn_, lab(f, t)]).values()])
        d['null_at_step'], d['reference_null_at_step'] = {}, {}
        for k in a.null_steps:
            rn = {s: bpb(r, k) for (l, s), r in R.items() if l == rn_ and bpb(r, k) is not None}
            d['null_at_step'][str(k)] = summary([r - n for r, n in paired([rn_, tn_], k).values()])
            d['reference_null_at_step'][str(k)] = st.mean(rn.values()) if rn else None   # where it sits on the law's axis
        vs[t] = d
        print(f'  {t:18} reference null minus this null {show(d["null"]):>24}   '
              + '   '.join(f'{f} gain, this minus reference {show(d[f])}' for f in a.facts))
        print(f'  {"":18}   reference null minus this tokenizer with ' + '   '.join(
            f'{f}: {show(d[f + "_together"])}' for f in a.facts))
        print(f'  {"":18}   the same at step ' + '   '.join(f'{k}: {show(d["null_at_step"][str(k)])}'
                                                        for k in a.null_steps))
        print(f'  {"":18}   the reference null there ' + '   '.join(
            f'{k}: {v:.4f}' if v is not None else f'{k}: —' for k, v in d['reference_null_at_step'].items()))

    named(since)
    print(f'\n  at matched null BPB (seeds both cells share, to each best step)')
    print(f'  {"":18} {"":10} ' + ' '.join(f'{lv:>8.2f}' for lv in a.levels))
    law = {}
    since = len(left_out)

    def curve(R_, null, feat):
        pts, seeds, left = runs.gain_curve(R_, null, feat)
        if left and R_ is R:
            left_out.setdefault(f'{null} / {feat}', {}).update({str(s): list(m) for s, m in left.items()})
        return [(b, g) for _, b, g, _ in pts], len(seeds)
    for f in a.facts:
        for t in a.tokenizers:
            pts, ns = curve(R, lab(a.null, t), lab(f, t))
            if not pts:
                continue
            vals = [runs.at(pts, lv) for lv in a.levels]
            per_seed = {}
            for lv in a.seed_levels:
                per_seed[str(lv)] = {}
                for s in sorted(final(a.null, t)):
                    sub = {k: v for k, v in R.items() if k[1] == s}
                    p, _ = curve(sub, lab(a.null, t), lab(f, t))
                    per_seed[str(lv)][s] = runs.at(p, lv) if p else None
            law.setdefault(f, {})[t] = {'levels': dict(zip(map(str, a.levels), vals)), 'seeds': ns,
                                        'per_seed': per_seed}
            print(f'  {t:18} {f[:10]:10} ' + ' '.join(f'{v:+8.4f}' if v is not None else f'{"—":>8}' for v in vals)
                  + f'   (n={ns})')
    named(since)
    print(f'\n  {f1} gain at matched null BPB against {a.reference}, paired by seed (this minus reference)')
    since = len(left_out)
    for t in a.tokenizers:
        if t == a.reference or t not in law.get(f1, {}) or a.reference not in law.get(f1, {}):
            continue
        cells = []
        one = paired([lab(a.null, t), lab(f1, t), lab(a.null, a.reference), lab(f1, a.reference)])
        for lv in map(str, a.seed_levels):
            mine, ref = law[f1][t]['per_seed'][lv], law[f1][a.reference]['per_seed'][lv]
            d = summary([mine[s] - ref[s] for s in mine if s in ref and s in one and mine[s] is not None and ref[s] is not None])
            vs.setdefault(t, {}).setdefault('at_null_bpb', {})[lv] = d
            cells.append(f'{lv}: {show(d)}')
        print(f'  {t:18} ' + '   '.join(cells))
    named(since)
    rec.add('tokenizers', rows)
    rec.add('against_reference', vs)
    rec.add('matched_null_bpb', law)
    if left_out:
        rec.add('left_out', left_out)
    rec.write()


if __name__ == '__main__':
    main()
