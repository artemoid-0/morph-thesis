"""
metrics.py — result persistence and the comparison report.

Results land in results/<run_id>/ with config.json beside result.json, so the
seed and every other knob are part of the identity. The old scheme wrote
results/<pipeline>_<variant>.json, which meant a second seed silently
overwrote the first — exactly the run you need for effect-size confidence.
"""

from __future__ import annotations

import os
import json

RESULTS_DIR = 'results'


def save_result(cfg, log, best, results_dir=RESULTS_DIR):
    run_dir = os.path.join(results_dir, cfg.run_id())
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    record = {
        'run_id': cfg.run_id(),
        'label': cfg.label(),
        'pipeline': cfg.data.pipeline,
        'facts': list(cfg.frozen.facts),
        'seed': cfg.train.seed,
        'best_step': best['step'],
        'val_loss': best['mean_loss'],
        'perplexity': best['perplexity'],
        'bits_per_byte': best['bits_per_byte'],
        'val_tokens': best['tokens'],
        'lr_horizon': cfg.train.lr_horizon,
        'max_steps': cfg.train.max_steps,
        'log': log,
    }
    path = os.path.join(run_dir, 'result.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    return path


def load_results(results_dir=RESULTS_DIR):
    out = []
    if not os.path.isdir(results_dir):
        return out
    for entry in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, entry, 'result.json')
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def _agg(records):
    """Group by (label) and average across seeds, reporting spread."""
    groups = {}
    for r in records:
        groups.setdefault(r['label'], []).append(r)
    rows = []
    for label, rs in groups.items():
        ppl = [r['perplexity'] for r in rs]
        bpb = [r['bits_per_byte'] for r in rs]
        rows.append({
            'label': label,
            'pipeline': rs[0]['pipeline'],
            'facts': rs[0]['facts'],
            'n': len(rs),
            'ppl': sum(ppl) / len(ppl),
            'ppl_min': min(ppl), 'ppl_max': max(ppl),
            'bpb': sum(bpb) / len(bpb),
            'bpb_min': min(bpb), 'bpb_max': max(bpb),
        })
    return rows


def report(results_dir=RESULTS_DIR):
    records = load_results(results_dir)
    if not records:
        print('No results yet. Run some cells first.')
        return

    rows = _agg(records)

    print('\n=== results (best-val, averaged over seeds) ===\n')
    hdr = f'{"cell":22} {"n":>2} {"ppl":>9} {"ppl range":>17} {"bpb":>8}'
    print(hdr)
    print('-' * len(hdr))
    for r in sorted(rows, key=lambda x: (x['pipeline'], x['label'])):
        rng = f'{r["ppl_min"]:.2f}-{r["ppl_max"]:.2f}' if r['n'] > 1 else '-'
        print(f'{r["label"]:22} {r["n"]:2d} {r["ppl"]:9.2f} {rng:>17} '
              f'{r["bpb"]:8.4f}')

    by_label = {r['label']: r for r in rows}

    # ── Q1: segmentation, cross-pipeline -> BPB ──────────────────────────────
    print('\n--- Q1: segmentation (cross-pipeline -> BPB) ---')
    anchor = by_label.get('bpe_baseline')
    if anchor:
        print(f'  bpe_baseline BPB {anchor["bpb"]:.4f}  (anchor)')
        for name in ('morph_baseline', 'det_baseline'):
            r = by_label.get(name)
            if r:
                d = r['bpb'] - anchor['bpb']
                print(f'  {name:22} BPB {r["bpb"]:.4f}  ({d:+.4f}, '
                      f'{"better" if d < 0 else "worse"} than BPE)')
    else:
        print('  (run bpe_baseline to populate the anchor)')

    # ── Q2: freezing, within-pipeline -> perplexity ──────────────────────────
    print('\n--- Q2: freezing (within-pipeline -> per-token perplexity) ---')
    print('  static facts = parameter tying; context facts = information the')
    print('  trained table provably cannot represent. Read them separately.')
    for pipe in ('bpe', 'morph', 'det'):
        base = by_label.get(f'{pipe}_baseline')
        if not base:
            continue
        for r in rows:
            if r['pipeline'] != pipe or not r['facts']:
                continue
            d = r['ppl'] - base['ppl']
            pct = 100.0 * d / base['ppl']
            spread = ''
            if base['n'] > 1 or r['n'] > 1:
                noise = max(base['ppl_max'] - base['ppl_min'],
                            r['ppl_max'] - r['ppl_min'])
                flag = ' (WITHIN SEED NOISE)' if abs(d) <= noise else ''
                spread = f' | seed spread {noise:.2f}{flag}'
            print(f'  {r["label"]:22} {base["ppl"]:8.2f} -> {r["ppl"]:8.2f}  '
                  f'({d:+.2f}, {pct:+.1f}%){spread}')
    print()
