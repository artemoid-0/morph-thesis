"""
experiment.py

Run the pipeline × embedding-variant matrix ONE CELL AT A TIME.

Each cell is an independent command that trains a single (pipeline, variant)
combination and writes its best-val result to results/<cell>.json. Cells can be
run in any order, on any schedule; nothing is lost if you stop between cells.
A separate `report` command reads every saved result and prints the comparison
table — the two questions Phase 1 answers:

    1. Does morpheme segmentation beat BPE?   → baseline column, in BPB
    2. Does freezing flags help?              → baseline vs onehot, per-pipeline,
                                                 in per-token perplexity

Usage:
    # Phase-1 cells (run each when you have time; ~30-40 min each):
    python experiment.py run bpe   baseline      # the anchor (already calibrated)
    python experiment.py run morph baseline
    python experiment.py run det   baseline
    python experiment.py run morph onehot
    python experiment.py run det   onehot

    # once cells are done, assemble the comparison:
    python experiment.py report

    # options mirror the calibration run:
    python experiment.py run morph onehot --steps 50000 --eval-every 500 --patience 6
"""

import os
import json
import argparse

import pandas as pd

from train import train_cell

MORPHOLEX_PATH = 'data/MorphoLEX_en.xlsx'
VOCAB_CSV      = 'data/our_final_vocab.csv'
BPE_PATH       = 'data/bpe_tokenizer.json'
RESULTS_DIR    = 'results'

# the Phase-1 matrix. BPE only supports baseline (no morpheme metadata to freeze).
VALID_CELLS = {
    'bpe':   ['baseline'],
    'morph': ['baseline', 'onehot', 'features', 'wordnet'],
    'det':   ['baseline', 'onehot', 'features', 'wordnet'],
}


def _build_pipeline(pipeline_name):
    """Instantiate a pipeline and return (pipeline, vocab_df_or_None, vocab_size_or_None)."""
    if pipeline_name == 'bpe':
        from pipeline_bpe import BPEPipeline
        bpe = BPEPipeline(tokenizer_path=BPE_PATH)
        return bpe, None, bpe.vocab_size_actual()

    vocab_df = pd.read_csv(VOCAB_CSV)
    if pipeline_name == 'morph':
        from pipeline_morpheme import MorphemePipeline
        return MorphemePipeline(vocab_df, MORPHOLEX_PATH), vocab_df, None
    if pipeline_name == 'det':
        from pipeline_deterministic import DeterministicPipeline
        return DeterministicPipeline(vocab_df, MORPHOLEX_PATH), vocab_df, None

    raise ValueError(f'unknown pipeline {pipeline_name!r}')


def run_cell(pipeline_name, variant, **train_kw):
    """Train one matrix cell and persist its best-val result to results/<cell>.json."""
    if pipeline_name not in VALID_CELLS:
        raise ValueError(f'unknown pipeline {pipeline_name!r}; '
                         f'choose from {list(VALID_CELLS)}')
    if variant not in VALID_CELLS[pipeline_name]:
        raise ValueError(
            f'pipeline {pipeline_name!r} does not support variant {variant!r}; '
            f'valid: {VALID_CELLS[pipeline_name]}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cell = f'{pipeline_name}_{variant}'
    out_path = os.path.join(RESULTS_DIR, cell + '.json')

    print(f'\n########## CELL: {cell} ##########')

    pipeline, vocab_df, vocab_size = _build_pipeline(pipeline_name)

    model, log, best = train_cell(
        pipeline=pipeline, pipeline_name=pipeline_name,
        vocab_df=vocab_df, vocab_size=vocab_size, variant=variant,
        **train_kw)

    # persist everything needed for the report (and for re-reading later)
    record = {
        'cell':          cell,
        'pipeline':      pipeline_name,
        'variant':       variant,
        'best_step':     best['step'],
        'val_loss':      best['mean_loss'],
        'perplexity':    best['perplexity'],
        'bits_per_byte': best['bits_per_byte'],
        'val_tokens':    best['tokens'],
        # keep the full curve so we can plot / inspect later
        'log': [{k: v for k, v in e.items()} for e in log],
        # record the training config for reproducibility
        'config': {k: v for k, v in train_kw.items()},
    }
    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2)

    print(f'\n[saved] {out_path}')
    print(f'  {cell}: ppl {best["perplexity"]:.2f} | '
          f'bpb {best["bits_per_byte"]:.4f} | best step {best["step"]}')
    return record


def report():
    """Read every results/*.json and print the Phase-1 comparison table."""
    if not os.path.isdir(RESULTS_DIR):
        print('No results yet. Run some cells first.')
        return

    records = {}
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                r = json.load(f)
            records[r['cell']] = r

    if not records:
        print('No results yet. Run some cells first.')
        return

    # ── raw table ──────────────────────────────────────────────────────────────
    print('\n=== Phase-1 results (best-val) ===\n')
    header = f'{"cell":18} {"pipeline":9} {"variant":9} {"ppl":>10} {"bpb":>9} {"step":>7}'
    print(header)
    print('-' * len(header))
    order = ['bpe_baseline', 'morph_baseline', 'det_baseline',
             'morph_onehot', 'det_onehot',
             'morph_features', 'det_features',
             'morph_wordnet', 'det_wordnet']
    for cell in order:
        if cell in records:
            r = records[cell]
            print(f'{r["cell"]:18} {r["pipeline"]:9} {r["variant"]:9} '
                  f'{r["perplexity"]:10.2f} {r["bits_per_byte"]:9.4f} '
                  f'{r["best_step"]:7d}')
    # any cells not in the canonical order
    for cell, r in records.items():
        if cell not in order:
            print(f'{r["cell"]:18} {r["pipeline"]:9} {r["variant"]:9} '
                  f'{r["perplexity"]:10.2f} {r["bits_per_byte"]:9.4f} '
                  f'{r["best_step"]:7d}')

    # ── Q1: does morpheme segmentation beat BPE? (cross-pipeline → BPB) ─────────
    print('\n--- Q1: morpheme segmentation vs BPE (cross-pipeline → BPB) ---')
    if 'bpe_baseline' in records:
        anchor = records['bpe_baseline']['bits_per_byte']
        print(f'  BPE baseline BPB: {anchor:.4f}  (the anchor)')
        for cell in ['morph_baseline', 'det_baseline']:
            if cell in records:
                b = records[cell]['bits_per_byte']
                delta = b - anchor
                verdict = 'better' if delta < 0 else 'worse'
                print(f'  {cell:16} BPB: {b:.4f}  '
                      f'({delta:+.4f}, {verdict} than BPE)')
    else:
        print('  (run bpe baseline first to populate the anchor)')

    # ── Q2: does freezing flags help? (within-pipeline → perplexity) ───────────
    print('\n--- Q2: freezing flags (within-pipeline → per-token perplexity) ---')
    for pipe in ['morph', 'det']:
        base = records.get(f'{pipe}_baseline')
        oneh = records.get(f'{pipe}_onehot')
        if base and oneh:
            bp, op = base['perplexity'], oneh['perplexity']
            delta = op - bp
            verdict = 'helps' if delta < 0 else 'hurts'
            pct = 100.0 * delta / bp
            print(f'  {pipe}: baseline ppl {bp:.2f} → onehot ppl {op:.2f}  '
                  f'({delta:+.2f}, {pct:+.1f}%, freezing {verdict})')
        elif base or oneh:
            have = 'baseline' if base else 'onehot'
            print(f'  {pipe}: only {have} done — run the other to compare')
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Run the experiment matrix one cell at a time.')
    sub = p.add_subparsers(dest='command', required=True)

    pr = sub.add_parser('run', help='train one (pipeline, variant) cell')
    pr.add_argument('pipeline', choices=list(VALID_CELLS))
    pr.add_argument('variant')
    pr.add_argument('--steps', type=int, default=50000,
                    help='hard step ceiling (early stop usually ends sooner)')
    pr.add_argument('--eval-every', type=int, default=500)
    pr.add_argument('--patience', type=int, default=6)
    pr.add_argument('--batch', type=int, default=32)
    pr.add_argument('--lr', type=float, default=3e-4)
    pr.add_argument('--articles', type=int, default=0,
                    help='0 = full corpus (default)')

    sub.add_parser('report', help='print the comparison table from saved results')

    args = p.parse_args()

    if args.command == 'run':
        articles = None if args.articles == 0 else args.articles
        run_cell(args.pipeline, args.variant,
                 max_steps=args.steps, eval_every=args.eval_every,
                 patience=args.patience, batch_size=args.batch, lr=args.lr,
                 max_articles_train=articles)
    elif args.command == 'report':
        report()


if __name__ == '__main__':
    main()
