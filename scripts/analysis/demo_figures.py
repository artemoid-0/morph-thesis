#!/usr/bin/env python3
"""
demo_figures.py — the figures of docs/README.md, each computed from raw run records.
(The thesis's own figures are thesis_figures.py's, journey §33.80.)

Reads every result.json under the given results directories (default: results/),
keeps one record per (label, seed), and draws:

  fig1_q2_end.png        each feature against its same-shape null, end of training
  fig2_early_help.png    feature minus null over training: the early help and its fade
  fig3_budget.png        case's gain against train tokens and training length
  fig4_architecture.png  the slot over the plain baseline, and the position test

Every comparison is paired by seed within one machine (FINDINGS, method point 11;
a seed whose runs ran on two machines is named and left out); bars are mean ±
standard error over seeds. Positive = the feature (or the slot) is better, in bits
per byte.

    python scripts/analysis/demo_figures.py --out exports/figures
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs  # noqa: E402

EN = lambda f: f'bpe_{f}-LEARN-CAUSAL_bpe_17k'
UK = lambda f, fr='0.2', steps='': f'bpe_{f}-LEARN-CAUSAL_bpe_uk_17k_ukwiki{fr}{steps}'
EN_BASE = 'bpe_baseline_bpe_17k'
UK_BASE = lambda fr='0.2', steps='': f'bpe_baseline_bpe_uk_17k_ukwiki{fr}{steps}'
TOKENS = {'0.002': 1.51e6, '0.01': 7.66e6, '0.05': 37.8e6, '0.2': 151.1e6}
BLUE, ORANGE, GREY, GREEN, RED = '#2b6cb0', '#dd6b20', '#718096', '#2f855a', '#c53030'


def paired(R, base, feat, what=None):
    """Per-seed (base − feature) in bits per byte at the end of training, the seeds paired within
    one machine; the seeds left out are named under `what`."""
    pairs, left = runs.pair(R, base, feat)
    if left and what:
        print(f'    {what}: {runs.left_out(left)}')
    return [rb['bits_per_byte'] - rf['bits_per_byte'] for rb, rf in pairs.values()]


def ms(d):
    """(mean, standard error) over seeds — 0.0 at one seed; (None, None) for none."""
    s = runs.summary(d, se_one=0.0)
    return (s['mean'], s['se']) if s else (None, None)


# Every run a figure is planned to show. A figure with any of them missing is NOT
# drawn: a figure that silently drops points reads as a different result (the
# Ukrainian nulls quarantined under G-73 and not yet re-run took four of fig3's six
# points). The script names what is missing instead.
NEEDS = {
    'fig3_budget.png': [UK('mconst', fr, '_st10k') for fr in TOKENS] + [UK('mconst', '0.2', '_st20k'), UK('mconst', '0.2')],
    'fig4_architecture.png': [UK('mconst'), UK('mconst', '0.01', '_st10k'), UK('mconst', '0.002', '_st10k'),
                              UK_BASE('0.01', '_st10k'), UK_BASE('0.002', '_st10k')],
}


def missing(R, name):
    have = {lab for lab, _ in R}
    return [lab for lab in NEEDS.get(name, []) if lab not in have]


def style(ax, title, ylabel='gain over the null (bits per byte)'):
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)


def fig1(R, out):
    rows = [('English: word structure', EN('mconst'), EN('mstruct'), BLUE),
            ('control: structure shuffled across words', EN('mconst'), EN('mstructshuf'), GREY),
            ('English: frequency band', EN('mconst'), EN('mfreq'), BLUE),
            ('English: part of speech', EN('mconst'), EN('pos'), BLUE),
            ('English: inflection', EN('mconst'), EN('mfeatlex'), BLUE),
            ('Ukrainian: case (tagger)', UK('mconst'), UK('ucase'), ORANGE),
            ('Ukrainian: case (word form only)', UK('mconst'), UK('ucaselex'), ORANGE)]
    vals = []
    for name, b, f, c in rows:
        d = paired(R, b, f, f'fig1, {name}')
        vals.append((name, *ms(d), len(d), c))
    vals = [v for v in vals if v[1] is not None]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = range(len(vals))
    ax.barh(list(y), [v[1] for v in vals], xerr=[v[2] for v in vals], color=[v[4] for v in vals], alpha=0.85, capsize=3)
    ax.set_yticks(list(y)); ax.set_yticklabels([f'{v[0]}  (n={v[3]})' for v in vals], fontsize=8)
    ax.invert_yaxis(); ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('gain over the same-shape null at the end of training (bits per byte)', fontsize=9)
    ax.set_title('End of training: every feature lies within 0.002 bits per byte\n'
                 'of its same-shape null — inflection falls below it', fontsize=11, loc='left')
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig1_q2_end.png'), dpi=200); plt.close(fig)


def curve(R, base, feat):
    """Along training: the steps, and the mean gain (base − feature) with its standard error over
    the seeds paired within one machine, every evaluation (not cut at the best step)."""
    pts, _, _ = runs.gain_curve(R, base, feat, to_best=False)
    m, e = [], []
    for _, _, _, v in pts:
        mm, ee = ms([g for _, g in v]); m.append(mm); e.append(ee)
    return [p[0] for p in pts], m, e


def fig2(R, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, b, f, c in (('English part of speech', EN('mconst'), EN('pos'), BLUE),
                          ('English inflection', EN('mconst'), EN('mfeatlex'), '#4299e1'),
                          ('Ukrainian case', UK('mconst'), UK('ucase'), ORANGE)):
        x, m, e = curve(R, b, f)
        if not x:
            continue
        n = len(paired(R, b, f, f'fig2, {name}'))
        ax.plot(x, m, color=c, lw=1.6, label=f'{name} (n={n})')
        ax.fill_between(x, [a - s for a, s in zip(m, e)], [a + s for a, s in zip(m, e)], color=c, alpha=0.18, lw=0)
    ax.set_xscale('log'); ax.set_xlabel('training step', fontsize=9)
    style(ax, 'Informative features speed early learning; the advantage fades as the model learns')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig2_early_help.png'), dpi=200); plt.close(fig)


def fig3(R, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for kind, f, c, mk in (('case (tagger), 10k steps', 'ucase', ORANGE, 'o'),
                           ('case (word form only), 10k steps', 'ucaselex', '#f6ad55', 's')):
        xs, ys, es = [], [], []
        for fr, tok in TOKENS.items():
            m, e = ms(paired(R, UK('mconst', fr, '_st10k'), UK(f, fr, '_st10k'), f'fig3, {kind}, {fr}'))
            if m is not None:
                xs.append(tok); ys.append(m); es.append(e)
        if xs:
            ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, lw=1.4, capsize=3, label=kind)
    # longer training on the SAME 151M tokens: drawn at 151M, not offset —
    # an offset on a log axis would read as a larger corpus
    for steps, mk, txt in (('_st20k', 'D', '20k steps'), ('', '*', '50k steps')):
        m, e = ms(paired(R, UK('mconst', '0.2', steps), UK('ucase', '0.2', steps), f'fig3, {txt}'))
        if m is not None:
            ax.errorbar([TOKENS['0.2']], [m], yerr=[e], color=RED, marker=mk, ms=10 if mk == '*' else 6,
                        capsize=3, ls='none', label=f'case (tagger), the same 151M tokens, {txt}')
            if mk == '*':
                ax.annotate('same data,\ntrained 5× longer', xy=(TOKENS['0.2'], m), xytext=(TOKENS['0.2'] / 9, m - 0.0004),
                            fontsize=8, color=RED, arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
    ax.set_xscale('log'); ax.set_xlabel('Ukrainian train tokens', fontsize=9)
    style(ax, 'Case helps until the model has learned it: little data, or little training')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig3_budget.png'), dpi=200); plt.close(fig)


def fig4(R, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.0), gridspec_kw={'width_ratios': [1.25, 1]})
    rows = [('English\n(full, 50k)', EN_BASE, EN('mconst'), BLUE),
            ('Ukrainian\n(full, 50k)', UK_BASE(), UK('mconst'), ORANGE),
            ('Ukrainian\n7.6M, 10k', UK_BASE('0.01', '_st10k'), UK('mconst', '0.01', '_st10k'), ORANGE),
            ('Ukrainian\n1.5M, 10k', UK_BASE('0.002', '_st10k'), UK('mconst', '0.002', '_st10k'), ORANGE)]
    v = []
    for n, x, y, c in rows:
        d = paired(R, x, y, 'fig4, ' + n.replace('\n', ' '))
        v.append((n, *ms(d), len(d), c))
    v = [r for r in v if r[1] is not None]
    a.bar(range(len(v)), [r[1] for r in v], yerr=[r[2] for r in v], color=[r[4] for r in v], alpha=0.85, capsize=3)
    a.set_xticks(range(len(v))); a.set_xticklabels([f'{r[0]}\n(n={r[3]})' for r in v], fontsize=7)
    style(a, 'Reserving 8 input dims (no information)\nbeats the plain baseline', 'gain over the plain baseline (bits per byte)')
    rows = [('no position\nin the slot', EN('mzero'), EN('mzeronopos')),
            ('position only\nin the slot', EN('mzero'), EN('mzeroposonly'))]
    v = []
    for n, x, y in rows:
        d = paired(R, x, y, 'fig4, ' + n.replace('\n', ' '))
        v.append((n, *ms(d), len(d)))
    v = [r for r in v if r[1] is not None]
    b.bar(range(len(v)), [r[1] for r in v], yerr=[r[2] for r in v], color=GREY, alpha=0.85, capsize=3)
    b.set_xticks(range(len(v))); b.set_xticklabels([f'{r[0]}\n(n={r[3]})' for r in v], fontsize=7)
    style(b, 'English: against the zero slot,\nposition is needed in both places', 'change against the zero slot (bits per byte)')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig4_architecture.png'), dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    runs.add_results_flag(ap)
    ap.add_argument('--out', default=os.path.join(ROOT, 'exports', 'figures'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = runs.load(runs.results_of(args))
    print(f'  {len(R)} run records')
    incomplete = 0
    for f, name in ((fig1, 'fig1_q2_end.png'), (fig2, 'fig2_early_help.png'),
                    (fig3, 'fig3_budget.png'), (fig4, 'fig4_architecture.png')):
        gone = missing(R, name)
        if gone:
            incomplete += 1
            print(f'  NOT DRAWN {name}: no runs of ' + ', '.join(gone))
            if os.path.exists(os.path.join(args.out, name)):
                print(f'    an older {name} is still in {args.out} — it is stale')
            continue
        f(R, args.out)
    print(f'  figures -> {args.out}' + (f'; {incomplete} not drawn' if incomplete else ''))


if __name__ == '__main__':
    main()
