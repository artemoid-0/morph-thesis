#!/usr/bin/env python3
"""
learning_law.py — is a feature's gain a function of how much the model has learned?

For each matched pair (a feature and its same-shape null, same seeds), every
evaluation step gives a point: the null's validation BPB (how far the model has
got) and the feature's gain over the null. Points are averaged over seeds and
kept only up to each run's best step (before any over-fitting). If the gain
depends only on how much the model has learned, the curves of different
training lengths, budgets and corpus sizes lie on one curve (DIRECTIONS D1).

Reports per condition: the peak gain and where it falls, the zero crossing (if
reached), and the gain at a common reference point of learning — the number
that says whether the curves collapse. Draws one panel per feature.

    python scripts/analysis/learning_law.py --out exports/figures
    python scripts/analysis/learning_law.py --roots results unpacked_packs --out exports/figures
    python scripts/analysis/learning_law.py --refs 1.30 1.25 1.20 1.15 1.10 --record

--refs prints each condition's gain at more points of learning than the panel's own;
--spread adds, at each of those points, each condition's gain per seed and a permutation
test: how often the gap between the conditions' means is as large when the seeds' values are
dealt to the conditions at random — whether the conditions differ by more than their seeds do;
--record writes every curve's numbers to results/audits/learning_law_<UTC timestamp>.json.

    python scripts/analysis/learning_law.py --refs 1.30 1.25 1.20 1.15 1.10 --spread --record
"""
import os
import glob
import json
import random
import argparse
import datetime
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

U = lambda f, fr='0.2', x='': f'bpe_{f}-LEARN-CAUSAL_bpe_uk_17k_ukwiki{fr}{x}'
E = lambda f, x='': f'bpe_{f}-LEARN-CAUSAL_bpe_17k{x}'
PANELS = [  # title, reference null BPB for comparing conditions, conditions
    ('Ukrainian case', 1.10, [
        ('151M, 50k steps', U('mconst'), U('ucase')), ('151M, 20k', U('mconst', x='_st20k'), U('ucase', x='_st20k')),
        ('151M, 10k', U('mconst', x='_st10k'), U('ucase', x='_st10k')), ('151M, 5k', U('mconst', x='_st5k'), U('ucase', x='_st5k')),
        ('38M, 10k', U('mconst', '0.05', '_st10k'), U('ucase', '0.05', '_st10k')),
        ('7.6M, 10k', U('mconst', '0.01', '_st10k'), U('ucase', '0.01', '_st10k')), ('1.5M, 10k', U('mconst', '0.002', '_st10k'), U('ucase', '0.002', '_st10k'))]),
    ('English part of speech', 1.55, [
        ('50k steps', E('mconst'), E('pos')), ('20k', E('mconst', '_st20k'), E('pos', '_st20k')),
        ('10k', E('mconst', '_st10k'), E('pos', '_st10k')), ('5k', E('mconst', '_st5k'), E('pos', '_st5k'))]),
    ('English part of speech, by model width', 1.55, [
        ('128 wide, 50k steps', E('mconst'), E('pos')), ('128 wide, 10k', E('mconst', '_st10k'), E('pos', '_st10k')),
        ('256 wide, 10k', E('mconst', '_st10k_w256'), E('pos', '_st10k_w256')), ('256 wide, 20k', E('mconst', '_st20k_w256'), E('pos', '_st20k_w256')),
        ('64 wide, 10k', E('mconst', '_st10k_w64'), E('pos', '_st10k_w64'))]),
    ('English, other features (solid 50k, dashed 10k, dotted 20k)', 1.55, [
        ('inflection', E('mconst'), E('mfeatlex')), ('inflection, 10k', E('mconst', '_st10k'), E('mfeatlex', '_st10k')), ('inflection, 20k', E('mconst', '_st20k'), E('mfeatlex', '_st20k')),
        ('part of speech, word form only', E('mconst'), E('poslex')), ('word form only, 10k', E('mconst', '_st10k'), E('poslex', '_st10k')), ('word form only, 20k', E('mconst', '_st20k'), E('poslex', '_st20k')),
        ('frequency band', E('mconst'), E('mfreq')), ('frequency band, 10k', E('mconst', '_st10k'), E('mfreq', '_st10k')), ('frequency band, 20k', E('mconst', '_st20k'), E('mfreq', '_st20k')),
        ('structure', E('mconst'), E('mstruct')), ('structure, 10k', E('mconst', '_st10k'), E('mstruct', '_st10k')), ('structure, 20k', E('mconst', '_st20k'), E('mstruct', '_st20k'))]),
]
T = lambda f, t: f'bpe_{f}-LEARN-CAUSAL_{t}_ukwiki0.05_st10k'
PANELS.append(  # D7: one condition (~38M tokens, 10k steps) under six tokenizers (journey §33.58, §33.65)
    ('Ukrainian case by tokenizer (38M, 10k)', 1.25, [
        (name, T('mconst', t), T('ucase', t)) for name, t in (
            ('BPE 4k', 'bpe_uk_4k'), ('BPE 8k', 'bpe_uk_8k'), ('BPE 17k', 'bpe_uk_17k'), ('BPE 32k', 'bpe_uk_32k'),
            ('Unigram 17k', 'unigram_uk_17k'), ('BPE 17k, endings apart', 'bpeend_uk_17k'))]))
COLOURS = ['#c53030', '#dd6b20', '#d69e2e', '#38a169', '#3182ce', '#805ad5', '#2c7a7b']


def load(roots):
    R = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, '**', 'result.json'), recursive=True):
            try:
                r = json.load(open(p))
            except (OSError, ValueError):
                continue
            R.setdefault((r['label'], r['seed']), r)
    return R


def curve(R, null, feat):
    """[(null BPB, gain)] averaged over the seeds both cells share, to each run's best step."""
    seeds = sorted(s for (l, s) in R if l == feat and (null, s) in R)
    pts = {}
    for s in seeds:
        ln = {e['step']: e['bits_per_byte'] for e in R[(null, s)]['log'] if e['step'] > 0}
        lf = {e['step']: e['bits_per_byte'] for e in R[(feat, s)]['log'] if e['step'] > 0}
        stop = min(R[(null, s)]['best_step'], R[(feat, s)]['best_step'])
        for x in ln:
            if x in lf and x <= stop:
                pts.setdefault(x, []).append((ln[x], ln[x] - lf[x]))
    out = [(st.mean(a for a, _ in v), st.mean(b for _, b in v)) for x, v in sorted(pts.items()) if len(v) == len(seeds)]
    return out, len(seeds)


def per_seed(R, null, feat, ref):
    """{seed: gain at `ref`} from each seed's own curve, to its best step."""
    out = {}
    for s in sorted(s for (l, s) in R if l == feat and (null, s) in R):
        pts, _ = curve({k: v for k, v in R.items() if k in ((null, s), (feat, s))}, null, feat)
        g = at(pts, ref)
        if g is not None:
            out[s] = g
    return out


def spread_test(groups, draws=20000, seed=0):
    """The gap between the groups' means, and the share of random deals of the pooled values
    to groups of the same sizes whose gap is as large or larger."""
    gap = lambda gs: max(st.mean(g) for g in gs) - min(st.mean(g) for g in gs)
    observed = gap(groups)
    pool = [x for g in groups for x in g]
    sizes = [len(g) for g in groups]
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        rng.shuffle(pool)
        dealt, i = [], 0
        for n in sizes:
            dealt.append(pool[i:i + n]); i += n
        hits += gap(dealt) >= observed - 1e-12
    return observed, hits / draws


def at(points, ref):
    """The gain where the null passes `ref`, interpolated; None if never reached."""
    for (a0, g0), (a1, g1) in zip(points, points[1:]):
        if a0 >= ref >= a1 and a0 != a1:
            return g0 + (g1 - g0) * (a0 - ref) / (a0 - a1)
    return None


def zero(points):
    for (a0, g0), (a1, g1) in zip(points, points[1:]):
        if g0 > 0 >= g1:
            return a0 + (a1 - a0) * g0 / (g0 - g1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', default=['results'])
    ap.add_argument('--out', default='exports/figures')
    ap.add_argument('--refs', nargs='*', type=float, default=[], help='more null BPB points to read each gain at')
    ap.add_argument('--record', action='store_true', help='write the numbers to results/audits/')
    ap.add_argument('--spread', action='store_true', help='per-seed gains at --refs, and the permutation test')
    args = ap.parse_args()
    record = []
    spread = []
    R = load(args.roots)
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 4.2))
    for ax, (title, ref, conds) in zip(axes, PANELS):
        print(f'\n  {title} — gain at null BPB {ref} is the collapse test')
        print(f'    {"condition":32} {"n":>2} {"peak gain":>10} {"at null":>8} {"gain at ref":>12} {"zero at":>8} {"end gain":>9}')
        paired = title.startswith('English, other')       # a feature's 10k curve shares its colour, dashed
        for k, (name, n, f) in enumerate(conds):
            c = COLOURS[k // 3] if paired else COLOURS[k % len(COLOURS)]
            pts, ns = curve(R, n, f)
            if not pts:
                continue
            peak = max(pts, key=lambda p: p[1]); g = at(pts, ref); z = zero(pts)
            print(f'    {name:32} {ns:>2} {peak[1]:+10.4f} {peak[0]:8.3f} '
                  f'{(f"{g:+.4f}" if g is not None else "—"):>12} {(f"{z:.3f}" if z else "—"):>8} {pts[-1][1]:+9.4f}')
            more = {r: at(pts, r) for r in args.refs}
            if more:
                print(f'    {"":32}    at ' + '  '.join(f'{r:.2f} {v:+.4f}' if v is not None else f'{r:.2f} —' for r, v in more.items()))
            record.append({'panel': title, 'condition': name, 'null': n, 'feature': f, 'seeds': ns,
                           'peak': {'gain': peak[1], 'null_bpb': peak[0]}, 'reference': ref, 'gain_at_reference': g,
                           'zero_at': z, 'end': {'gain': pts[-1][1], 'null_bpb': pts[-1][0]},
                           'gain_at': {str(r): v for r, v in more.items()}})
            ax.plot([a for a, _ in pts], [b for _, b in pts], ('-o', '--o', ':o')[k % 3] if paired else '-o', ms=2.2, lw=1.2, color=c, label=f'{name} (n={ns})')
        ax.axhline(0, color='black', lw=0.7)
        ax.axvline(ref, color='grey', lw=0.6, ls=':')
        ax.invert_xaxis()
        ax.set_xlabel("the null's validation BPB  (→ more learned)", fontsize=9)
        ax.set_ylabel('feature over its null (BPB)', fontsize=9)
        ax.set_title(title, fontsize=10, loc='left')
        ax.legend(fontsize=7, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    if args.spread:
        for title, ref, conds in PANELS:
            for r in args.refs:
                rows = [(name, per_seed(R, n, f, r)) for name, n, f in conds]
                rows = [(name, ps) for name, ps in rows if len(ps) >= 2]
                if len(rows) < 2:
                    continue
                gap, p = spread_test([list(ps.values()) for _, ps in rows])
                print(f'\n  {title} at null BPB {r:.2f}: gap between conditions {gap:.4f}; '
                      f'as large or larger in {100 * p:.0f}% of random deals of the seeds')
                for name, ps in rows:
                    v = list(ps.values())
                    print(f'    {name:32} {st.mean(v):+.4f}  sd {st.stdev(v):.4f}  n={len(v)}  '
                          + ' '.join(f'{x:+.4f}' for x in v))
                spread.append({'panel': title, 'null_bpb': r, 'gap': gap, 'p_random_deal': p,
                               'conditions': [{'condition': name, 'per_seed': {str(k): x for k, x in ps.items()}}
                                              for name, ps in rows]})
    fig.tight_layout()
    path = os.path.join(args.out, 'law_gain_vs_learned.png')
    fig.savefig(path, dpi=170)
    print(f'\n  figure -> {path}')
    if args.record:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '+0000'
        rec = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           'results', 'audits', f'learning_law_{ts}.json')
        json.dump({'script': 'scripts/analysis/learning_law.py', 'timestamp_utc': ts, 'roots': args.roots,
                   'refs': args.refs, 'curves': record, 'spread': spread}, open(rec, 'w'), indent=1)
        print(f'  [record] {rec}')


if __name__ == '__main__':
    main()
