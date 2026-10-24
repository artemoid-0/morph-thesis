#!/usr/bin/env python3
"""
thesis_figures.py — the thesis's figures, in Ukrainian and in English, from the run
records (journey §33.80).

The thesis captions every figure below it, so a figure carries no title; panels
carry letters (а, б, в, г). Sized for a 16 cm column at 300 dpi, labels 10 pt or
more. Colours: the reference categorical palette of the data-visualisation
guidance, in its fixed order (validated there for lines up to eight series);
controls and contrasts in grey.

  fig_q2_end        each feature against its same-shape null, end of training
  fig_early_help    a feature's gain over its null along training
  fig_law_ukrainian Ukrainian case's gain against the null's validation BPB: by data
                    and budget, and by tokenizer
  fig_law_english   English part of speech's: by schedule, and by model width
  fig_tokenizer     BPE 17k with endings apart against BPE 17k, null against null,
                    along training and against the reference null's BPB, with
                    Unigram 17k for contrast; width 1024's runs are all the second
                    machine's, and its legend key says so

A figure is drawn only if every run it is planned to show exists (as in
demo_figures.py): one that silently dropped points would read as another result.
Runs are paired by seed within one machine (FINDINGS, method point 11); a seed whose
runs ran on two machines is named and left out.

    python scripts/analysis/thesis_figures.py --out exports/figures
      -> exports/figures/uk/*.png, exports/figures/en/*.png
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

CM = 1 / 2.54
WIDTH = 16 * CM
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GREY, INK, MUTED = '#7a7974', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                     'legend.fontsize': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})

from matplotlib.ticker import ScalarFormatter, FixedLocator, FuncFormatter, NullLocator  # noqa: E402


class CommaFormatter(ScalarFormatter):
    """Ukrainian decimals: a comma, not a point."""
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',')


def numbers(ax, lang, axes='xy'):
    for which in axes:
        f = CommaFormatter(useOffset=False) if lang == 'uk' else ScalarFormatter(useOffset=False)
        (ax.xaxis if which == 'x' else ax.yaxis).set_major_formatter(f)


def step_axis(ax, lang, top):
    """A log axis of training steps, in thousands."""
    ticks = [t for t in (500, 1000, 2000, 5000, 10000, 20000, 50000) if t <= top * 1.05]
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, p: (f'{v / 1000:g}'.replace('.', ',') if lang == 'uk' else f'{v / 1000:g}')))


TOK = lambda f, t, fr='0.05', x='_st10k': f'bpe_{f}-LEARN-CAUSAL_{t}_ukwiki{fr}{x}'
EN = lambda f: f'bpe_{f}-LEARN-CAUSAL_bpe_17k'
UK = lambda f, fr='0.2', x='': TOK(f, 'bpe_uk_17k', fr, x)
T = lambda f, t: TOK(f, t)
U5 = lambda f, x='': UK(f, '0.05', x)
LONG_DASH = (0, (7, 2.5))      # width 1024: a fourth line style after '-', '-.', ':' (Unigram's grey '--' is shorter)

TXT = {
    'uk': {
        'gain': 'виграш над контролем,\nбіти на байт',
        'gain_end': 'виграш над контролем тієї самої\nархітектури наприкінці навчання,\nбіти на байт',
        'step': 'крок навчання, тис.',
        'null_bpb': 'BPB контролю на валідації\n(→ модель навчилася більше)',
        'ref_bpb': 'BPB контролю BPE 17k на валідації\n(→ модель навчилася більше)',
        'lead': 'перевага над BPE 17k,\nбіти на байт',
        'tok_title': 'BPE 17k з відокремленими закінченнями, n = 4:',
        'letters': 'абвг',
        'q2': [('англійська: структура слова', EN('mconst'), EN('mstruct'), 0),
               ('контроль: перемішана структура', EN('mconst'), EN('mstructshuf'), None),
               ('англійська: частотна смуга', EN('mconst'), EN('mfreq'), 0),
               ('англійська: частина мови', EN('mconst'), EN('pos'), 0),
               ('англійська: словозміна', EN('mconst'), EN('mfeatlex'), 0),
               ('українська: відмінок (розпізнавач)', UK('mconst'), UK('ucase'), 1),
               ('українська: відмінок з форми слова', UK('mconst'), UK('ucaselex'), 1)],
        'early': [('англійська: частина мови', EN('mconst'), EN('pos')),
                  ('українська: відмінок', UK('mconst'), UK('ucase')),
                  ('англійська: словозміна', EN('mconst'), EN('mfeatlex'))],
        'law': [  # panel: reference null BPB, conditions
            (1.10, [('151 млн токенів, 50 тис. кроків', UK('mconst'), UK('ucase')),
                    ('38 млн токенів, 10 тис. кроків', U5('mconst', '_st10k'), U5('ucase', '_st10k'))]),
            (1.55, [('50 тис. кроків', EN('mconst'), EN('pos')),
                    ('20 тис. кроків', EN('mconst') + '_st20k', EN('pos') + '_st20k'),
                    ('10 тис. кроків', EN('mconst') + '_st10k', EN('pos') + '_st10k'),
                    ('5 тис. кроків', EN('mconst') + '_st5k', EN('pos') + '_st5k')]),
            (1.55, [('ширина 128, 50 тис. кроків', EN('mconst'), EN('pos')),
                    ('ширина 128, 10 тис. кроків', EN('mconst') + '_st10k', EN('pos') + '_st10k'),
                    ('ширина 256, 10 тис. кроків', EN('mconst') + '_st10k_w256', EN('pos') + '_st10k_w256'),
                    ('ширина 256, 20 тис. кроків', EN('mconst') + '_st20k_w256', EN('pos') + '_st20k_w256'),
                    ('ширина 64, 10 тис. кроків', EN('mconst') + '_st10k_w64', EN('pos') + '_st10k_w64')]),
            (1.25, [(n, T('mconst', t), T('ucase', t)) for n, t in (
                ('BPE 4k', 'bpe_uk_4k'), ('BPE 8k', 'bpe_uk_8k'), ('BPE 17k', 'bpe_uk_17k'), ('BPE 32k', 'bpe_uk_32k'),
                ('Unigram 17k', 'unigram_uk_17k'), ('BPE 17k з відокремленими\nзакінченнями', 'bpeend_uk_17k'))])],
        'tok': [('151 млн токенів, 50 тис. кроків', '0.2', '', 0, '-'),
                ('38 млн токенів, 20 тис. кроків', '0.05', '_st20k', 2, '-'),
                ('38 млн токенів, 10 тис. кроків', '0.05', '_st10k', 1, '-'),
                ('38 млн токенів, 10 тис. кроків, ширина 256', '0.05', '_st10k_w256', 1, '-.'),
                ('38 млн токенів, 20 тис. кроків, ширина 256', '0.05', '_st20k_w256', 2, '-.'),
                ('38 млн токенів, 10 тис. кроків, ширина 512', '0.05', '_st10k_w512', 1, ':'),
                ('38 млн токенів, 20 тис. кроків, ширина 512', '0.05', '_st20k_w512', 2, ':'),
                ('38 млн токенів, 10 тис. кроків, ширина 1024', '0.05', '_st10k_w1024', 1, LONG_DASH),
                ('38 млн токенів, 20 тис. кроків, ширина 1024', '0.05', '_st20k_w1024', 2, LONG_DASH)],
        'tok_keys': [('151 млн токенів, 50 тис. кроків', 0), ('38 млн токенів, 20 тис. кроків', 2),
                     ('38 млн токенів, 10 тис. кроків', 1)],
        'tok_widths': [('ширина 128', '-'), ('ширина 256', '-.'), ('ширина 512', ':'),
                       ('ширина 1024 (друга машина)', LONG_DASH)],
        'unigram': 'Unigram 17k',
    },
    'en': {
        'gain': 'gain over the null,\nbits per byte',
        'gain_end': 'gain over the same-shape null\nat the end of training, bits per byte',
        'step': 'training step, thousands',
        'null_bpb': "the null's validation BPB\n(→ more learned)",
        'ref_bpb': "BPE 17k null's validation BPB\n(→ more learned)",
        'lead': 'lead over BPE 17k,\nbits per byte',
        'tok_title': 'BPE 17k with endings apart, n = 4:',
        'letters': 'abcd',
        'q2': [('English: word structure', EN('mconst'), EN('mstruct'), 0),
               ('control: structure shuffled', EN('mconst'), EN('mstructshuf'), None),
               ('English: frequency band', EN('mconst'), EN('mfreq'), 0),
               ('English: part of speech', EN('mconst'), EN('pos'), 0),
               ('English: inflection', EN('mconst'), EN('mfeatlex'), 0),
               ('Ukrainian: case (tagger)', UK('mconst'), UK('ucase'), 1),
               ('Ukrainian: case from the word form', UK('mconst'), UK('ucaselex'), 1)],
        'early': [('English: part of speech', EN('mconst'), EN('pos')),
                  ('Ukrainian: case', UK('mconst'), UK('ucase')),
                  ('English: inflection', EN('mconst'), EN('mfeatlex'))],
        'law': None,   # filled below from the Ukrainian panels, with English names
        'tok': [('151M tokens, 50k steps', '0.2', '', 0, '-'),
                ('38M tokens, 20k steps', '0.05', '_st20k', 2, '-'),
                ('38M tokens, 10k steps', '0.05', '_st10k', 1, '-'),
                ('38M tokens, 10k steps, width 256', '0.05', '_st10k_w256', 1, '-.'),
                ('38M tokens, 20k steps, width 256', '0.05', '_st20k_w256', 2, '-.'),
                ('38M tokens, 10k steps, width 512', '0.05', '_st10k_w512', 1, ':'),
                ('38M tokens, 20k steps, width 512', '0.05', '_st20k_w512', 2, ':'),
                ('38M tokens, 10k steps, width 1024', '0.05', '_st10k_w1024', 1, LONG_DASH),
                ('38M tokens, 20k steps, width 1024', '0.05', '_st20k_w1024', 2, LONG_DASH)],
        'tok_keys': [('151M tokens, 50k steps', 0), ('38M tokens, 20k steps', 2), ('38M tokens, 10k steps', 1)],
        'tok_widths': [('width 128', '-'), ('width 256', '-.'), ('width 512', ':'),
                       ('width 1024 (second machine)', LONG_DASH)],
        'unigram': 'Unigram 17k',
    },
}
_EN_LAW_NAMES = [['151M tokens, 50k steps', '38M tokens, 10k steps'],
                 ['50k steps', '20k steps', '10k steps', '5k steps'],
                 ['width 128, 50k steps', 'width 128, 10k steps', 'width 256, 10k steps', 'width 256, 20k steps',
                  'width 64, 10k steps'],
                 ['BPE 4k', 'BPE 8k', 'BPE 17k', 'BPE 32k', 'Unigram 17k', 'BPE 17k, endings apart']]
TXT['en']['law'] = [(ref, [(nm, n, f) for nm, (_, n, f) in zip(names, conds)])
                    for names, (ref, conds) in zip(_EN_LAW_NAMES, TXT['uk']['law'])]


def needs(lang):
    """Every run each figure is planned to show."""
    t = TXT[lang]
    law = lambda panels: [x for _, conds in panels for _, n, f in conds for x in (n, f)]
    out = {'fig_q2_end': [x for _, b, f, _ in t['q2'] for x in (b, f)],
           'fig_early_help': [x for _, b, f in t['early'] for x in (b, f)],
           'fig_law_ukrainian': law([t['law'][0], t['law'][3]]),
           'fig_law_english': law([t['law'][1], t['law'][2]])}
    tok = []
    for _, fr, x, _, _ in t['tok']:
        tok += [TOK('mconst', 'bpe_uk_17k', fr, x), TOK('mconst', 'bpeend_uk_17k', fr, x)]
    out['fig_tokenizer'] = tok + [TOK('mconst', 'unigram_uk_17k')]
    return out


_NAMED = set()


def named(what, left):
    """Name the seeds left out of a pairing (runs.pair), once per run of the script."""
    if left and what not in _NAMED:
        _NAMED.add(what)
        print(f'    {what}: {runs.left_out(left)}')


def ms(d):
    """(mean, standard error) over seeds — 0.0 at one seed; (None, None) for none."""
    s = runs.summary(d, se_one=0.0)
    return (s['mean'], s['se']) if s else (None, None)


def paired(R, base, feat):
    """Per-seed (base − feature) at the end of training, paired within one machine."""
    pairs, left = runs.pair(R, base, feat)
    named(f'{feat} against {base}', left)
    return [rb['bits_per_byte'] - rf['bits_per_byte'] for rb, rf in pairs.values()]


def early(R, base, feat):
    """Along training: the steps, and the mean gain (base − feature) with its standard error,
    every evaluation."""
    pts, _, left = runs.gain_curve(R, base, feat, to_best=False)
    named(f'{feat} against {base}', left)
    m, e = [], []
    for _, _, _, v in pts:
        mm, ee = ms([g for _, g in v]); m.append(mm); e.append(ee)
    return [p[0] for p in pts], m, e


def law(R, null, feat):
    """[(null BPB, gain)] averaged over the seeds, to each run's best step (the law's axis); the seeds."""
    pts, seeds, left = runs.gain_curve(R, null, feat)
    named(f'{feat} against {null}', left)
    return [(b, g) for _, b, g, _ in pts], len(seeds)


def lead(R, ref, other):
    """Per step: the step, the reference null's BPB, and ref − other (mean, se), over shared seeds."""
    pts, seeds, left = runs.gain_curve(R, ref, other, to_best=False)
    named(f'{other} against {ref}', left)
    out = []
    for x, _, _, v in pts:
        m, e = ms([g for _, g in v])
        out.append((x, sum(b for b, _ in v) / len(seeds), m, e, len(seeds)))
    return out


def letter(ax, ch):
    ax.text(-0.02, 1.02, ch, transform=ax.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')


def fig_q2_end(R, t, path):
    rows = []
    for n, b, f, c in t['q2']:
        d = paired(R, b, f)
        rows.append((n, *ms(d), len(d), c))
    rows = [r for r in rows if r[1] is not None]
    fig, ax = plt.subplots(figsize=(WIDTH, 8.2 * CM))
    y = list(range(len(rows)))
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], height=0.62,
            color=[GREY if r[4] is None else SERIES[r[4]] for r in rows], capsize=2.5,
            error_kw={'elinewidth': 1, 'ecolor': INK})
    ax.set_yticks(y)
    ax.set_yticklabels([f'{r[0]}  (n = {r[3]})' for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel(t['gain_end'])
    numbers(ax, t['lang'], 'x')
    ax.grid(axis='x', color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_early_help(R, t, path):
    fig, ax = plt.subplots(figsize=(WIDTH, 8.5 * CM))
    for k, (name, b, f) in enumerate(t['early']):
        x, m, e = early(R, b, f)
        if not x:
            continue
        n = len(paired(R, b, f))
        ax.plot(x, m, color=SERIES[k], lw=1.6, label=f'{name}  (n = {n})')
        ax.fill_between(x, [a - s for a, s in zip(m, e)], [a + s for a, s in zip(m, e)], color=SERIES[k], alpha=0.16, lw=0)
    step_axis(ax, t['lang'], 50000)
    numbers(ax, t['lang'], 'y')
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel(t['step'])
    ax.set_ylabel(t['gain'])
    ax.grid(color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _law(R, t, path, panels, anchors):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 13.5 * CM))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.45, wspace=0.34)
    for ax, ch, (ref, conds), x0 in zip(axes, t['letters'], panels, anchors):
        for k, (name, n, f) in enumerate(conds):
            pts, ns = law(R, n, f)
            if not pts:
                continue
            ax.plot([a for a, _ in pts], [b for _, b in pts], '-', lw=1.4, color=SERIES[k], label=f'{name}  (n = {ns})')
        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(ref, color=GREY, lw=0.7, ls=':')
        ax.invert_xaxis()
        numbers(ax, t['lang'])
        ax.grid(color='#e6e5e0', lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(t['null_bpb'], fontsize=9.5)
        ax.legend(frameon=False, fontsize=10, loc='upper left', bbox_to_anchor=(x0, -0.36), ncol=1,
                  handlelength=1.4, handletextpad=0.5, borderaxespad=0)
        letter(ax, ch)
    axes[0].set_ylabel(t['gain'])
    fig.savefig(path)
    plt.close(fig)


def fig_law_ukrainian(R, t, path):
    _law(R, t, path, [t['law'][0], t['law'][3]], anchors=(-0.32, 0.0))


def fig_law_english(R, t, path):
    _law(R, t, path, [t['law'][1], t['law'][2]], anchors=(-0.12, -0.36))


def fig_tokenizer(R, t, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 11.0 * CM), sharey=True)
    for name, fr, x, k, ls in t['tok']:
        pts = lead(R, TOK('mconst', 'bpe_uk_17k', fr, x), TOK('mconst', 'bpeend_uk_17k', fr, x))
        st_, rb, m, e = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
        for ax, xs in ((a, st_), (b, rb)):
            ax.plot(xs, m, linestyle=ls, color=SERIES[k], lw=1.5)
            ax.fill_between(xs, [u - v for u, v in zip(m, e)], [u + v for u, v in zip(m, e)], color=SERIES[k], alpha=0.14, lw=0)
    pts = lead(R, TOK('mconst', 'bpe_uk_17k'), TOK('mconst', 'unigram_uk_17k'))
    st_, rb, m, e = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], [p[3] for p in pts]
    for ax, xs in ((a, st_), (b, rb)):
        ax.plot(xs, m, '--', color=GREY, lw=1.4)
        ax.fill_between(xs, [u - v for u, v in zip(m, e)], [u + v for u, v in zip(m, e)], color=GREY, alpha=0.14, lw=0)
    # the contrast is labelled on its line, not in the legend (n = 4 as well): past its end and under
    # the zero line, where nothing else runs (beside its end run the width-1024 lines, at step 2,000
    # the width-512 ones)
    a.annotate(t['unigram'], xy=(st_[-1], m[-1]), xytext=(4, -4), textcoords='offset points', color=MUTED,
               fontsize=9.5, va='top', ha='left')
    step_axis(a, t['lang'], 50000)
    b.invert_xaxis()
    numbers(b, t['lang'], 'x')
    numbers(a, t['lang'], 'y')
    a.set_xlabel(t['step'])
    b.set_xlabel(t['ref_bpb'], fontsize=9.5)
    a.set_ylabel(t['lead'])
    for ax, ch in ((a, t['letters'][0]), (b, t['letters'][1])):
        ax.axhline(0, color=INK, lw=0.8)
        ax.grid(color='#e6e5e0', lw=0.6)
        ax.set_axisbelow(True)
        letter(ax, ch)
    # two keys, not one entry per line: colour is the corpus and budget, the line style the width
    # (nine lines' full names overflowed the figure in Ukrainian); blank entries keep each key in its
    # own column, which the legend fills top to bottom
    from matplotlib.lines import Line2D
    colours = [Line2D([], [], color=SERIES[k], lw=1.5) for _, k in t['tok_keys']]
    widths = [Line2D([], [], color=INK, lw=1.5, ls=ls) for _, ls in t['tok_widths']]
    blank = [Line2D([], [], lw=0)] * (len(widths) - len(colours))
    fig.legend(colours + blank + widths, [n for n, _ in t['tok_keys']] + [''] * len(blank) + [n for n, _ in t['tok_widths']],
               title=t['tok_title'], title_fontsize=9, loc='lower center', ncol=2, frameon=False, fontsize=8.8,
               bbox_to_anchor=(0.5, 0.0), alignment='left')
    fig.tight_layout(rect=(0, 0.25, 1, 1), w_pad=1.2)
    fig.savefig(path)
    plt.close(fig)


FIGS = {'fig_q2_end': fig_q2_end, 'fig_early_help': fig_early_help, 'fig_law_ukrainian': fig_law_ukrainian,
        'fig_law_english': fig_law_english, 'fig_tokenizer': fig_tokenizer}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    runs.add_results_flag(ap)
    ap.add_argument('--out', default=os.path.join(ROOT, 'exports', 'figures'))
    ap.add_argument('--langs', nargs='+', default=['uk', 'en'])
    a = ap.parse_args()
    R = runs.load(runs.results_of(a))
    have = {lab for lab, _ in R}
    for lang in a.langs:
        out = os.path.join(a.out, lang)
        os.makedirs(out, exist_ok=True)
        for name, need in needs(lang).items():
            path = os.path.join(out, name + '.png')
            gone = sorted(set(l for l in need if l not in have))
            if gone:
                print(f'  NOT DRAWN {lang}/{name}.png: no runs of ' + ', '.join(gone))
                if os.path.exists(path):
                    print(f'    an older {lang}/{name}.png is still there — it is stale')
                continue
            FIGS[name](R, dict(TXT[lang], lang=lang), path)
            print(f'  {lang}/{name}.png')
    print(f'  thesis figures -> {a.out}/{{{",".join(a.langs)}}}')


if __name__ == '__main__':
    main()
