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

Writes results/audits/tokenizer_table_<UTC timestamp>.json.

    python scripts/analysis/tokenizer_table.py
    python scripts/analysis/tokenizer_table.py --roots results unpacked/results_74 --levels 1.40 1.25 1.15
"""
import os
import sys
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOKENIZERS = ['bpe_uk_4k', 'bpe_uk_8k', 'bpe_uk_17k', 'bpe_uk_32k', 'unigram_uk_17k', 'bpeend_uk_17k']


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = st.mean(xs)
    return {'mean': m, 'se': (st.stdev(xs) / len(xs) ** 0.5) if len(xs) > 1 else 0.0,
            'n': len(xs), 'positive': sum(x > 0 for x in xs), 'values': xs}


def show(s):
    return '—' if s is None else f"{s['mean']:+.4f} ± {s['se']:.4f} ({s['positive']}/{s['n']})"


def main():
    import learning_law as LL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='+', default=[os.path.join(ROOT, 'results')])
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

    R = LL.load(a.roots)
    corpus = a.corpus + (a.fraction if a.corpus != 'wikitext103' else '')
    lab = lambda f, t: f'bpe_{f}-LEARN-CAUSAL_{t}_{corpus}{a.budget}'
    final = lambda f, t: {s: R[(l, s)]['bits_per_byte'] for (l, s) in R if l == lab(f, t)}

    def at_step(f, t, step=None):
        out = {}
        for (l, s), r in R.items():
            if l == lab(f, t):
                for e in r['log']:
                    if e['step'] == (a.step if step is None else step):
                        out[s] = e['bits_per_byte']
        return out

    def gains(t, f, table=final):
        n, x = table(a.null, t), table(f, t)
        return {s: n[s] - x[s] for s in sorted(n) if s in x}

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
               'step': {f: summary(gains(t, f, at_step).values()) for f in a.facts}}
        if f2:
            g1, g2 = gains(t, f1), gains(t, f2)
            row['first_over_second'] = summary([g1[s] - g2[s] for s in g1 if s in g2])
        rows[t] = row
        print(f'  {t:18} {st.mean(n.values()):11.4f} ' + ' '.join(f'{show(row["gain"][f]):>24}' for f in a.facts)
              + (f' {show(row["first_over_second"]):>26}' if f2 else '') + f' {show(row["step"][f1]):>26}')

    print(f'\n  against {a.reference}, paired by seed (same initial weights where the vocabularies match)')
    vs = {}
    ref_null = final(a.null, a.reference)
    for t in a.tokenizers:
        if t == a.reference or t not in rows:
            continue
        n = final(a.null, t)
        d = {'null': summary([ref_null[s] - n[s] for s in n if s in ref_null])}
        for f in a.facts:
            gr, gt = gains(a.reference, f), gains(t, f)
            d[f] = summary([gt[s] - gr[s] for s in gt if s in gr])
            x = final(f, t)                  # the tokenizer and the feature together, against the reference null
            d[f + '_together'] = summary([ref_null[s] - x[s] for s in x if s in ref_null])
        d['null_at_step'], d['reference_null_at_step'] = {}, {}
        for k in a.null_steps:
            rn, tn = at_step(a.null, a.reference, k), at_step(a.null, t, k)
            d['null_at_step'][str(k)] = summary([rn[s] - tn[s] for s in tn if s in rn])
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

    print(f'\n  at matched null BPB (seeds both cells share, to each best step)')
    print(f'  {"":18} {"":10} ' + ' '.join(f'{lv:>8.2f}' for lv in a.levels))
    law = {}
    for f in a.facts:
        for t in a.tokenizers:
            pts, ns = LL.curve(R, lab(a.null, t), lab(f, t))
            if not pts:
                continue
            vals = [LL.at(pts, lv) for lv in a.levels]
            per_seed = {}
            for lv in a.seed_levels:
                per_seed[str(lv)] = {}
                for s in sorted(final(a.null, t)):
                    sub = {k: v for k, v in R.items() if k[1] == s}
                    p, _ = LL.curve(sub, lab(a.null, t), lab(f, t))
                    per_seed[str(lv)][s] = LL.at(p, lv) if p else None
            law.setdefault(f, {})[t] = {'levels': dict(zip(map(str, a.levels), vals)), 'seeds': ns,
                                        'per_seed': per_seed}
            print(f'  {t:18} {f[:10]:10} ' + ' '.join(f'{v:+8.4f}' if v is not None else f'{"—":>8}' for v in vals)
                  + f'   (n={ns})')
    print(f'\n  {f1} gain at matched null BPB against {a.reference}, paired by seed (this minus reference)')
    for t in a.tokenizers:
        if t == a.reference or t not in law.get(f1, {}) or a.reference not in law.get(f1, {}):
            continue
        cells = []
        for lv in map(str, a.seed_levels):
            mine, ref = law[f1][t]['per_seed'][lv], law[f1][a.reference]['per_seed'][lv]
            d = summary([mine[s] - ref[s] for s in mine if s in ref and mine[s] is not None and ref[s] is not None])
            vs.setdefault(t, {}).setdefault('at_null_bpb', {})[lv] = d
            cells.append(f'{lv}: {show(d)}')
        print(f'  {t:18} ' + '   '.join(cells))
    rec.add('tokenizers', rows)
    rec.add('against_reference', vs)
    rec.add('matched_null_bpb', law)
    rec.write()


if __name__ == '__main__':
    main()
