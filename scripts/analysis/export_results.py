#!/usr/bin/env python3
"""
export_results.py — turn results/ into thesis material.

WHAT IT PRODUCES
----------------
    runs.csv                one row per run, every headline number
    runs.md                 the same as a markdown table
    runs.tex                the same as a LaTeX booktabs table
    curves.csv              learning curves, long form, ready to plot
    buckets.csv             per-frequency-bucket perplexity per run
    parameters.md           parameter breakdown per configuration
    environment.md          reproducibility appendix
    comparisons.md          every effect, paired by seed against its baseline

WHY A SCRIPT RATHER THAN COPYING NUMBERS
----------------------------------------
Every table written by hand so far was reconstructed from config plus
arithmetic, and one of those reconstructions was wrong (the control-asymmetry
paragraph). Numbers that appear in the thesis should be produced by a program
reading the run records, so that a re-run regenerates them and a typo cannot
survive.

Handles old runs that predate the provenance fields: missing values are shown
as `-` rather than guessed at.

    python scripts/analysis/export_results.py --out exports/
"""

import os
import sys
import csv
import glob
import json
import math
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_runs(results_dir):
    runs = []
    for entry in sorted(os.listdir(results_dir)):
        rp = os.path.join(results_dir, entry, 'result.json')
        cp = os.path.join(results_dir, entry, 'config.json')
        if not os.path.exists(rp):
            continue
        with open(rp) as f:
            r = json.load(f)
        r['_run_id'] = entry
        r['_config'] = json.load(open(cp)) if os.path.exists(cp) else {}
        r['_has_checkpoint'] = os.path.exists(
            os.path.join(results_dir, entry, 'model.pt'))
        runs.append(r)
    return runs


def g(d, *path, default=None):
    """Nested get that tolerates records written before provenance existed."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fmt(v, nd=4):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    if isinstance(v, int):
        return f'{v:,}'
    return str(v)


# ── runs table ────────────────────────────────────────────────────────────────

COLUMNS = [
    ('run_id', lambda r: r['_run_id']),
    ('label', lambda r: r.get('label')),
    ('pipeline', lambda r: r.get('pipeline')),
    ('facts', lambda r: '+'.join(r.get('facts') or []) or 'baseline'),
    ('seed', lambda r: r.get('seed')),
    ('bpb', lambda r: r.get('bits_per_byte')),
    ('ppl', lambda r: r.get('perplexity')),
    ('val_loss', lambda r: r.get('val_loss')),
    ('params', lambda r: g(r, 'parameters', 'trainable_total')),
    ('T', lambda r: g(r, 'parameters', 'frozen_width_T')),
    ('vocab', lambda r: g(r, 'parameters', 'vocab_size')),
    ('val_tokens', lambda r: r.get('val_tokens')),
    ('train_tokens', lambda r: g(r, 'data', 'train', 'tokens')),
    ('bytes_per_token', lambda r: g(r, 'data', 'validation', 'bytes_per_token')),
    ('best_step', lambda r: r.get('best_step')),
    ('still_improving', lambda r: g(r, 'convergence',
                                    'still_improving_at_ceiling')),
    ('wall_s', lambda r: g(r, 'timing', 'wall_seconds')),
    ('steps_per_s', lambda r: g(r, 'timing', 'steps_per_second')),
    ('gpu', lambda r: g(r, 'environment', 'gpu')),
    ('has_ckpt', lambda r: r['_has_checkpoint']),
]


def write_runs(runs, out):
    with open(os.path.join(out, 'runs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([c for c, _ in COLUMNS])
        for r in sorted(runs, key=lambda x: (x.get('label') or '',
                                             x.get('seed', 0))):
            w.writerow([fn(r) for _, fn in COLUMNS])

    show = ['label', 'seed', 'bpb', 'ppl', 'params', 'T', 'val_tokens',
            'best_step']
    idx = {c: i for i, (c, _) in enumerate(COLUMNS)}
    lines = ['| ' + ' | '.join(show) + ' |',
             '|' + '|'.join('---' for _ in show) + '|']
    tex = ['\\begin{tabular}{lrrrrrrr}', '\\toprule',
           ' & '.join(show) + ' \\\\', '\\midrule']
    for r in sorted(runs, key=lambda x: (x.get('label') or '',
                                         x.get('seed', 0))):
        vals = [fn(r) for _, fn in COLUMNS]
        row = [fmt(vals[idx[c]]) for c in show]
        lines.append('| ' + ' | '.join(row) + ' |')
        tex.append(' & '.join(s.replace('_', r'\_') for s in row) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(out, 'runs.md'), 'w').write('\n'.join(lines) + '\n')
    open(os.path.join(out, 'runs.tex'), 'w').write('\n'.join(tex) + '\n')


# ── curves, buckets, parameters, environment ─────────────────────────────────

def write_curves(runs, out):
    with open(os.path.join(out, 'curves.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'label', 'seed', 'step', 'train_loss',
                    'val_loss', 'perplexity', 'bits_per_byte', 'lr',
                    'elapsed_s'])
        for r in runs:
            for e in r.get('log', []):
                w.writerow([r['_run_id'], r.get('label'), r.get('seed'),
                            e.get('step'), e.get('train_loss'),
                            e.get('mean_loss'), e.get('perplexity'),
                            e.get('bits_per_byte'), e.get('lr'),
                            e.get('elapsed')])


def write_buckets(runs, out):
    rows = []
    for r in runs:
        for b in r.get('buckets', []) or []:
            rows.append([r['_run_id'], r.get('label'), r.get('seed'),
                         b.get('range'), b.get('tokens'), b.get('mean_loss'),
                         b.get('perplexity')])
    with open(os.path.join(out, 'buckets.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'label', 'seed', 'freq_range', 'val_tokens',
                    'mean_loss', 'perplexity'])
        w.writerows(rows)
    return len(rows)


def write_parameters(runs, out):
    seen = {}
    for r in runs:
        p = r.get('parameters')
        if not p:
            continue
        key = (r.get('label') or '').replace('_seed', '')
        seen.setdefault(key, p)
    lines = ['# Parameter breakdown', '',
             '| configuration | total | trained emb | factor | head | core | T |',
             '|---|---|---|---|---|---|---|']
    for k, p in sorted(seen.items()):
        lines.append(f"| {k} | {p['trainable_total']:,} | "
                     f"{p['embedding_trained']:,} | {p['embedding_factor']:,} | "
                     f"{p['output_head']:,} | {p['transformer_core']:,} | "
                     f"{p['frozen_width_T']} |")
    if not seen:
        lines.append('| *(no run carries a parameter breakdown yet)* | | | | | | |')
    open(os.path.join(out, 'parameters.md'), 'w').write('\n'.join(lines) + '\n')


def run_environment(r):
    """
    A run's environment block, with the tagger taken from its config where the
    record predates G-74: until 7 October `spacy_model` named en_core_web_sm
    whenever it was installed, so every Ukrainian run claimed the English tagger.
    The config names the tagger the run used; its version went unrecorded.
    """
    e = dict(r.get('environment') or {})
    used = g(r.get('_config') or {}, 'data', 'spacy_model')
    if e and used and 'spacy_models_installed' not in e and e.get('spacy_model') != used:
        e['spacy_model'] = used
        e['spacy_model_version'] = 'not recorded (G-74)'
    return e


def write_environment(runs, out):
    envs = [run_environment(r) for r in runs if r.get('environment')]
    lines = ['# Reproducibility appendix', '']
    if not envs:
        lines += ['*No run carries environment metadata. Runs predating '
                  'provenance capture cannot report it retrospectively.*']
    else:
        keys = sorted({k for e in envs for k in e})
        lines += ['| field | value(s) observed |', '|---|---|']
        for k in keys:
            vals = sorted({str(e.get(k)) for e in envs if e.get(k) is not None})
            lines.append(f'| {k} | {", ".join(vals[:4])}'
                         f'{" ..." if len(vals) > 4 else ""} |')
        lines += ['', f'*Captured from {len(envs)} of {len(runs)} runs.*']
    open(os.path.join(out, 'environment.md'), 'w').write('\n'.join(lines) + '\n')


def _baseline_group(label, cfg):
    """
    The comparison group a cell belongs to.

    Q2 claims are WITHIN-pipeline (per-token perplexity against that pipeline's
    own baseline); Q1 claims are CROSS-pipeline (bits-per-byte against BPE).
    Comparing a morph cell to the BPE baseline mixes the two, which is exactly
    the metric-mixing flagged as R-08 — and a generated table must not
    reproduce it.

    For BPE the tokenizer file also defines the group: bpe_17k and
    bpe_bytelevel are different vocabularies and different baselines.
    """
    pipe = g(cfg, 'data', 'pipeline', default='?')
    if pipe == 'bpe':
        path = str(g(cfg, 'data', 'bpe_path', default=''))
        stem = os.path.basename(path).replace('.json', '')
        return f'bpe/{stem}'
    # mirror SurfaceConfig.spec()'s pdiwc convention rather than inventing one:
    # every key starts with `emit_`, so first letters would all be 'e'
    sfc = g(cfg, 'surface', default={}) or {}
    order = [('emit_punctuation', 'p'), ('emit_digits', 'd'),
             ('emit_inflection', 'i'), ('emit_whitespace', 'w'),
             ('emit_case', 'c')]
    tag = ''.join(ch if sfc.get(k) else '-' for k, ch in order)
    if tag == '-----':
        tag = 'legacy'
    return f'{pipe}/{tag}'


def own_baseline(label):
    """
    The baseline a cell must be compared with: its own label with the feature
    part replaced by 'baseline'. Non-default training settings are part of
    every label by construction (config.label), so this is the run identical
    to the cell except for its features:
        bpe_mstruct-LEARN-CAUSAL_bpe_17k      -> bpe_baseline_bpe_17k
        bpe_mstruct_bpe_17k_st10k             -> bpe_baseline_bpe_17k_st10k
    """
    parts = label.split('_', 2)
    return '_'.join([parts[0], 'baseline'] + parts[2:])


def write_comparisons(runs, out):
    """
    Every effect against its OWN baseline, PAIRED BY SEED.

    gain = baseline BPB - cell BPB at the same seed (positive: the cell helps),
    over the seeds the cell shares with its baseline: per-seed gains, mean,
    standard error, and how many seeds agree in sign — how the project's reporting rule says
    small effects are reported.

    A pair is counted only when both runs come from the same GPU (FINDINGS, method point 11;
    journey §33.113): a seed whose cell and baseline ran on two machines is named and left out.

    Two defects of the earlier version (journey §33.6, GAPS G-27):
      * the baseline was "the" feature-free cell of a group — whichever came
        last — and once the learning-rate sweep existed that was a 10k-step
        sweep run, 0.08 BPB worse than the real baseline;
      * effects were cell MEAN minus baseline MEAN over different seed sets,
        judged against the widest seed spread of any cell. Seeds differ by more
        than most effects, so a cell lacking the baseline's unlucky seed 2
        looked better by exactly that luck.
    """
    import statistics as st
    by_label = defaultdict(list)
    for r in runs:
        if r.get('bits_per_byte') is not None:
            by_label[r.get('label')].append(r)
    groups = defaultdict(dict)
    for label, v in by_label.items():
        groups[_baseline_group(label, v[0].get('_config', {}))][label] = v

    notes = []

    def by_seed(label, v):
        """{seed: (bits per byte, machine)} — the first run at each seed."""
        seen = defaultdict(list)
        for x in v:
            seen[x.get('seed', 0)].append(x)
        for s_, xs in seen.items():
            b_ = [x['bits_per_byte'] for x in xs]
            if max(b_) - min(b_) > 1e-9:
                notes.append(f'`{label}` has {len(xs)} runs at seed {s_} that DIFFER '
                             f'by {max(b_) - min(b_):.4f} BPB; the first is used.')
        return {s_: (xs[0]['bits_per_byte'], machine(xs[0])) for s_, xs in seen.items()}

    def machine(x):      # not g(): this function binds g to a list below
        return 'second' if '5090' in ((x.get('environment') or {}).get('gpu') or '') else 'thesis'

    lines = ['# Effects, paired by seed', '',
             'Each effect is **gain = baseline BPB - cell BPB at the same seed** '
             '(positive: the cell helps), over the seeds the cell shares with its '
             'OWN baseline (the same label with the features replaced by '
             '`baseline`): per-seed gains, mean +/- standard error, and how many '
             'seeds agree in sign. Cells are never compared by their means: seeds '
             'differ by more than most effects, and a cell lacking an unlucky '
             'baseline seed would look better by exactly that luck.', '',
             'Within-pipeline comparisons (Q2) and cross-pipeline ones (Q1) are '
             'never mixed in a single table.', '']
    n_paired = 0
    for grp in sorted(groups):
        cells = groups[grp]
        sections = defaultdict(list)
        for label, v in cells.items():
            if v[0].get('facts') or []:
                sections[own_baseline(label)].append(label)
        if not sections:
            continue
        lines += [f'## {grp}', '']
        for base_label in sorted(sections, key=lambda b_: (b_ not in cells, b_)):
            members = sorted(sections[base_label])
            if base_label not in cells:
                lines += [f'*No run of `{base_label}` exists, so these are not '
                          f'comparable:* ' + ', '.join(f'`{l}`' for l in members), '']
                continue
            base = by_seed(base_label, cells[base_label])
            lines += [f'baseline `{base_label}`, seeds {sorted(base)}: '
                      + ', '.join(f'{base[k][0]:.4f}' for k in sorted(base)), '',
                      '| cell | paired seeds | gain per seed | mean gain +/- se | % | agree |',
                      '|---|---|---|---|---|---|']
            for label in members:
                c = by_seed(label, cells[label])
                both = sorted(set(c) & set(base))
                shared = [k for k in both if c[k][1] == base[k][1]]
                apart = [k for k in both if c[k][1] != base[k][1]]
                if apart:
                    notes.append(f'`{label}` against `{base_label}`: seed(s) {apart} left out, '
                                 f'the cell and its baseline ran on two machines.')
                if not shared:
                    why = (f'its baseline ran on the other machine at seed(s) {apart}' if apart
                           else 'no seed shared with the baseline')
                    lines.append(f'| {label} | none of {sorted(c)} | - | not comparable: {why} | | |')
                    continue
                g = [base[k][0] - c[k][0] for k in shared]
                m = st.mean(g)
                pct = 100.0 * m / st.mean(base[k][0] for k in shared)
                per = ' '.join(f'{x:+.4f}' for x in g)
                seeds = ','.join(str(k) for k in shared)
                if len(g) >= 2:
                    se = st.stdev(g) / len(g) ** 0.5
                    agree = max(sum(x > 0 for x in g), sum(x < 0 for x in g))
                    lines.append(f'| {label} | {seeds} | {per} | {m:+.4f} +/- {se:.4f} | '
                                 f'{pct:+.2f}% | {agree}/{len(g)} |')
                else:
                    lines.append(f'| {label} | {seeds} | {per} | {m:+.4f} (n=1: no claim) | '
                                 f'{pct:+.2f}% | - |')
                n_paired += 1
            lines.append('')
    if notes:
        lines += ['## Duplicate runs that disagree, and seeds left out', ''] + [f'- {n}' for n in notes] + ['']
    lines += ['---', '',
              '*One paired seed carries no claim. Two agreeing seeds are not a '
              'noise estimate: read the standard error and the '
              'sign count together.*']
    open(os.path.join(out, 'comparisons.md'), 'w').write('\n'.join(lines) + '\n')
    return n_paired


def write_audits(out, audit_dir=os.path.join('results', 'audits')):
    """
    Index the audit records so their numbers reach the paper the same way run
    metrics do — by being read from a file rather than transcribed from a
    terminal.
    """
    recs = sorted(glob.glob(os.path.join(audit_dir, '*.json')))
    lines = ['# Audit records', '']
    if not recs:
        lines += ['*No audit records yet. Re-run the analysis scripts; each now '
                  'writes to `results/audits/`.*']
        open(os.path.join(out, 'audits.md'), 'w').write('\n'.join(lines) + '\n')
        return 0
    lines += ['| script | when | arguments | result blocks |', '|---|---|---|---|']
    for r in recs:
        try:
            d = json.load(open(r))
        except Exception:
            continue
        a = ' '.join(f'{k}={v}' for k, v in (d.get('args') or {}).items()
                     if v not in (None, '', False, []))
        res = d.get('results') or {}
        # a record's results are a dict of sections, or a list of rows (tokenizer_widths)
        what = ', '.join(sorted(res.keys())) if isinstance(res, dict) else f'{len(res)} rows'
        lines.append(f"| `{d.get('script','?')}` | "
                     f"{d.get('timestamp_utc','?')} | {a or '-'} | "
                     f"{what or '-'} |")
    notes = []
    for r in recs:
        try:
            d = json.load(open(r))
        except Exception:
            continue
        for n in d.get('notes', []):
            notes.append(f"- **{d.get('script')}**: {n}")
    if notes:
        lines += ['', '## Caveats recorded with the numbers', ''] + notes
    open(os.path.join(out, 'audits.md'), 'w').write('\n'.join(lines) + '\n')
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--out', default='exports')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.results_dir)
    if not runs:
        print(f'no runs under {args.results_dir}/')
        sys.exit(1)

    write_runs(runs, args.out)
    write_curves(runs, args.out)
    nb = write_buckets(runs, args.out)
    write_parameters(runs, args.out)
    write_environment(runs, args.out)
    n_paired = write_comparisons(runs, args.out)
    n_audits = write_audits(args.out)

    withp = sum(1 for r in runs if r.get('parameters'))
    withe = sum(1 for r in runs if r.get('environment'))
    withb = sum(1 for r in runs if r.get('buckets'))
    withf = sum(1 for r in runs if r.get('frozen_blocks') is not None)

    print(f'exported {len(runs)} runs -> {args.out}/')
    for f in ('runs.csv', 'runs.md', 'runs.tex', 'curves.csv', 'buckets.csv',
              'parameters.md', 'environment.md', 'comparisons.md',
              'audits.md'):
        p = os.path.join(args.out, f)
        print(f'  {f:20} {os.path.getsize(p):>9,} bytes')
    print()
    print(f'  provenance coverage: parameters {withp}/{len(runs)}, '
          f'frozen_blocks {withf}/{len(runs)}, '
          f'environment {withe}/{len(runs)}, buckets {withb}/{len(runs)}')
    print(f'  bucket rows: {nb}')
    print(f'  paired comparisons: {n_paired} (comparisons.md, paired by seed)')
    print(f'  audit records indexed: {n_audits}')
    if withp < len(runs):
        print()
        print('  Runs predating provenance capture lack these fields. '
              'scripts/ops/backfill_results.py')
        print('  can recover the derivable ones (parameters, frozen blocks) '
              'from config alone.')


if __name__ == '__main__':
    main()
