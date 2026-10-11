#!/usr/bin/env python3
"""
probe_axis.py — does a probe's accuracy line the model widths up on the law?
(DIRECTIONS D2)

The law (D1) holds within one model size: a feature's gain over its null is a
function of the null's BPB. A wider model's curve collapses earlier, so BPB is
not the axis across sizes. This tool tries a probe's accuracy instead: for every
probed null (a record of scripts/analysis/probe.py in <root>/audits), each
checkpoint's accuracy at one layer is paired with the feature's gain over the
null at the same step — averaged over the seeds that both cells share, each run
to its best step — and the conditions are compared at common accuracy levels.
If the gains agree there, the gain is a function of that knowledge.

    python scripts/analysis/probe_axis.py                       # the current word's part of speech
    python scripts/analysis/probe_axis.py --previous            # the previous unit's (probe.py --previous)
    python scripts/analysis/probe_axis.py --previous --layer 2 --levels 0.60 0.65 0.70
    python scripts/analysis/probe_axis.py --previous --out exports/figures

A condition is the null's label; its feature is the same label with the null's
fact replaced (`--feature`, default the probed fact). The latest record per run
is used. The verdict compares the widths within each budget: the largest gap
between their gains at a common level, against `--tolerance`.
"""
import os
import re
import sys
import glob
import json
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))


def load_results(roots):
    from learning_law import load
    return load(roots)


def load_probes(roots, fact, previous):
    """{run id: record} — the latest probe record per run for this fact and kind."""
    found = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, '**', 'audits', 'probe_*.json'), recursive=True):
            m = re.search(r'_(\d{8}T\d{6})\+0000\.json$', p)
            try:
                d = json.load(open(p))
            except (OSError, ValueError):
                continue
            if not isinstance(d, dict) or 'rows' not in d or d.get('fact') != fact:
                continue
            if bool(d.get('previous', False)) != previous:
                continue
            ts = m.group(1) if m else ''
            if d['run'] not in found or ts > found[d['run']][0]:
                found[d['run']] = (ts, d)
    return {r: d for r, (_, d) in found.items()}


def accuracy_by_step(record, layer):
    """{step: accuracy} at one layer; layer None means the deepest, 'max' the best layer."""
    rows = record['rows']
    layers = sorted({r['layer'] for r in rows})
    out = {}
    for r in rows:
        if layer == 'max' or r['layer'] == (layers[-1] if layer is None else int(layer)):
            out[r['step']] = max(out.get(r['step'], 0.0), r['accuracy'])
    return out, layers


def gain_by_step(R, null, feat):
    """{step: (null BPB, gain)} averaged over the seeds both cells share, each to its best step."""
    seeds = sorted(s for (l, s) in R if l == feat and (null, s) in R)
    pts = {}
    for s in seeds:
        ln = {e['step']: e['bits_per_byte'] for e in R[(null, s)]['log'] if e['step'] > 0}
        lf = {e['step']: e['bits_per_byte'] for e in R[(feat, s)]['log'] if e['step'] > 0}
        stop = min(R[(null, s)]['best_step'], R[(feat, s)]['best_step'])
        for x in ln:
            if x in lf and x <= stop:
                pts.setdefault(x, []).append((ln[x], ln[x] - lf[x]))
    return ({x: (st.mean(a for a, _ in v), st.mean(g for _, g in v))
             for x, v in pts.items() if len(v) == len(seeds)}, seeds)


def at_level(points, level):
    """(gain, null BPB) where accuracy first reaches `level`, interpolated; None if never."""
    for (a0, b0, g0), (a1, b1, g1) in zip(points, points[1:]):
        if a0 <= level <= a1:
            t = (level - a0) / (a1 - a0) if a1 > a0 else 0.0
            return g0 + t * (g1 - g0), b0 + t * (b1 - b0)
    return None


def at_bpb(points, level):
    """Accuracy where the null's BPB first falls to `level`, interpolated; None if never."""
    for (a0, b0, _), (a1, b1, _) in zip(points, points[1:]):
        if b0 > level >= b1:
            return a0 + (a1 - a0) * (b0 - level) / (b0 - b1)
    return None


def width_of(label):
    m = re.search(r'_w(\d+)', label)
    return int(m.group(1)) if m else 128


def budget_of(label):
    m = re.search(r'_st(\d+)k', label)
    return f'{m.group(1)}k' if m else '50k'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roots', nargs='+', default=[os.path.join(ROOT, 'results')])
    ap.add_argument('--fact', default='pos', help='the probed fact')
    ap.add_argument('--feature', default=None, help="the feature whose gain is paired (default: the probed fact)")
    ap.add_argument('--null', default='mconst', help="the null's fact")
    ap.add_argument('--previous', action='store_true', help="records of probe.py --previous")
    ap.add_argument('--layer', default=None, help="a layer, or 'max' (default: the deepest)")
    ap.add_argument('--levels', nargs='*', type=float, default=None,
                    help='accuracy levels to compare at (default: six across the common range)')
    ap.add_argument('--bpb-levels', nargs='*', type=float,
                    default=[1.75, 1.70, 1.65, 1.60, 1.55, 1.50, 1.45],
                    help="null BPB levels at which each condition's accuracy is read")
    ap.add_argument('--tolerance', type=float, default=0.0015,
                    help='widths line up if their gains differ by at most this at every common level')
    ap.add_argument('--out', default=None, help='a folder for the figure')
    args = ap.parse_args()
    feature = args.feature or args.fact

    R = load_results(args.roots)
    probes = load_probes(args.roots, args.fact, args.previous)
    by_id = {r['run_id']: r for r in R.values()}
    kind = f"the previous unit's {args.fact}" if args.previous else f"the current word's {args.fact}"
    conds = []
    for run, rec in sorted(probes.items()):
        res = by_id.get(run)
        if res is None or list(res.get('facts', [])) != [args.null]:
            continue
        null = res['label']
        feat = null.replace(f'_{args.null}-', f'_{feature}-', 1)
        acc, layers = accuracy_by_step(rec, args.layer)
        gains, seeds = gain_by_step(R, null, feat)
        pts = sorted((s, acc[s], *gains[s]) for s in acc if s in gains)
        if len(pts) < 2:
            print(f'  {run} ({null}): fewer than two steps with both a probe and a gain — skipped')
            continue
        conds.append(dict(run=run, null=null, width=width_of(null), budget=budget_of(null),
                          seeds=seeds, points=pts, layers=layers))
    if not conds:
        sys.exit(f'no probed {args.null} null with {feature} runs beside it — run probe.py'
                 + (' --previous' if args.previous else '') + ' on a checkpointed null first')
    layer_name = 'best layer' if args.layer == 'max' else f'layer {args.layer if args.layer is not None else conds[0]["layers"][-1]}'
    print(f'\n  probe axis: {kind}, {layer_name}; gain of {feature} over its null')
    for c in conds:
        print(f'\n  {c["null"]}  (probe of {c["run"]}; gain over seeds {",".join(map(str, c["seeds"]))})')
        print(f'    {"step":>6} {"accuracy":>9} {"null BPB":>9} {"gain":>8}')
        for s, a, b, g in c['points']:
            print(f'    {s:>6} {a:9.3f} {b:9.4f} {g:+8.4f}')

    curves = {id(c): [(a, b, g) for _, a, b, g in c['points']] for c in conds}
    name = lambda c: f'{c["width"]} wide, {c["budget"]}'
    print('\n  accuracy at a common null BPB — who knows more at the same overall progress')
    print('    ' + f'{"null BPB":>8}  ' + '  '.join(f'{name(c):>16}' for c in conds))
    for v in args.bpb_levels:
        cells = [at_bpb(curves[id(c)], v) for c in conds]
        print('    ' + f'{v:8.3f}  ' + '  '.join(f'{(f"{x:.3f}" if x is not None else "—"):>16}' for x in cells))

    # the widths compared within each budget, at levels across the range they share
    verdicts = []
    for budget in sorted({c['budget'] for c in conds}, key=lambda b: int(b[:-1])):
        group = sorted((c for c in conds if c['budget'] == budget), key=lambda c: c['width'])
        if len({c['width'] for c in group}) < 2:
            continue
        lo = max(min(a for a, _, _ in curves[id(c)]) for c in group)
        hi = min(max(a for a, _, _ in curves[id(c)]) for c in group)
        levels = args.levels or ([lo + (hi - lo) * k / 5 for k in range(6)] if hi > lo else [])
        print(f'\n  {budget} budget — gain at a common accuracy (the null BPB there)')
        print('    ' + f'{"accuracy":>8}  ' + '  '.join(f'{name(c):>22}' for c in group))
        gaps = []
        for v in levels:
            r = [at_level(curves[id(c)], v) for c in group]
            print('    ' + f'{v:8.3f}  ' + '  '.join(f'{(f"{x[0]:+.4f} ({x[1]:.3f})" if x else "—"):>22}' for x in r))
            if all(r):
                g = [x[0] for x in r]
                gaps.append((v, max(g) - min(g), g[0] - g[-1]))
        if not gaps:
            verdicts.append(f'    {budget}: no common level reached by every width — nothing to line up')
            continue
        worst = max(gaps, key=lambda t: t[1])
        verdict = 'LINE UP' if worst[1] <= args.tolerance else 'do not line up'
        verdicts.append(f'    {budget}: widths {", ".join(str(c["width"]) for c in group)} — at {len(gaps)} common '
                        f'levels the largest gap is {worst[1]:.4f} (at {worst[0]:.3f}); narrow minus wide, mean '
                        f'{st.mean(t[2] for t in gaps):+.4f} → {verdict}')
    print('\n  verdict, the widths within each budget (tolerance ±%.4f):' % args.tolerance)
    print('\n'.join(verdicts) if verdicts else '    no budget has two widths probed')

    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(args.out, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        for c in conds:
            pts = curves[id(c)]
            ax.plot([a for a, _, _ in pts], [g for _, _, g in pts], '-o' if c['budget'] == '20k' else '--o',
                    ms=2.4, lw=1.2, color='#c53030' if c['width'] > 128 else '#3182ce', label=name(c))
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xlabel(f"the null's probe accuracy: {kind}, {layer_name}", fontsize=8.5)
        ax.set_ylabel(f'{feature} over its null (BPB)', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        path = os.path.join(args.out, f'probe_axis_{args.fact}{"prev" if args.previous else ""}.png')
        fig.savefig(path, dpi=170)
        print(f'\n  figure -> {path}')


if __name__ == '__main__':
    main()
