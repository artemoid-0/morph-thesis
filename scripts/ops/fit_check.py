#!/usr/bin/env python3
"""
fit_check.py — will a model fit, and how long will its run take? (journey §33.54)

Builds the model a `morphlm.cli run` command line would build, and times the
training step exactly as the loop runs it — AdamW, gradient clipping, the same
batch shape, random tokens and codes — and one validation pass as the loop runs
it. Reports the parameters by part, peak GPU memory against the card's, steps per
second, and the projected run time for the configured budget. With --widths and
--layers it sizes a grid in one call, to plan a scale study before chaining it.

    python scripts/ops/fit_check.py --facts mconst -o data.bpe_path=data/bpe_17k.json -o frozen.trainable=true -o frozen.causal_identity=true -o train.max_steps=20000 -o train.lr_horizon=20000
    python scripts/ops/fit_check.py --facts mconst ... --widths 128 256 512 --layers 4 8

Nothing is written; no data or cache is read (only the tokenizer, for the
vocabulary and the word boundaries of the causal shift).
"""
import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def measure(cfg, steps, warmup, val_windows, device):
    import torch
    from morphlm.modeling.transformer import build_model
    from morphlm.provenance import parameter_breakdown
    torch.manual_seed(0)
    model, V = build_model(cfg, verbose=False)
    model.to(device).train()
    parts = parameter_breakdown(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    B, L = cfg.train.batch_size, cfg.data.seq_len
    n_rows = len(cfg.frozen.facts)
    g = torch.Generator(device='cpu').manual_seed(0)
    # ids as real data has them: below the TOKENIZER's size — the model's
    # vocabulary adds special slots that no text produces, and the causal
    # shift's word tables are sized to the tokenizer
    tables = getattr(model.embed, 'tok_alpha', None)
    n_ids = min(V, tables.shape[0]) if tables is not None else V
    x = torch.randint(0, n_ids, (B, L), generator=g).to(device)
    y = torch.randint(0, n_ids, (B, L), generator=g).to(device)
    c = torch.randint(0, 8, (B, n_rows, L), generator=g).to(device) if n_rows else None
    aux = getattr(cfg.frozen, 'aux_weight', 0.0) if getattr(cfg.frozen, 'role', 'input') == 'target' else 0.0
    cuda = device.startswith('cuda')
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def step():
        _, loss = model(x, y, codes=c)
        total = loss + aux * model.last_aux_loss if aux and model.last_aux_loss is not None else loss
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

    for _ in range(warmup):
        step()
    sync(); t0 = time.time()
    for _ in range(steps):
        step()
    sync(); t_step = (time.time() - t0) / steps
    # one validation batch, exactly as training/loop.evaluate runs it
    model.eval()
    sync(); t0 = time.time()
    for _ in range(3):
        _, loss = model(x, y, codes=c)
        loss.item()
    sync(); t_eval_batch = (time.time() - t0) / 3
    model.train()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else float('nan')
    n_evals = cfg.train.max_steps // cfg.train.eval_every + 1
    eval_s = n_evals * -(-val_windows // B) * t_eval_batch
    run_s = cfg.train.max_steps * t_step + eval_s
    del model, opt
    if cuda:
        torch.cuda.empty_cache()
    return parts, peak, t_step, eval_s, run_s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pipeline', default='bpe')
    ap.add_argument('--facts', default='')
    ap.add_argument('-o', '--override', action='append', default=[], dest='over')
    ap.add_argument('--widths', nargs='*', type=int, default=None)
    ap.add_argument('--layers', nargs='*', type=int, default=None)
    ap.add_argument('--steps', type=int, default=20, help='timed training steps per model')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--val-windows', type=int, default=985, help='validation windows (WikiText: 985)')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()

    import torch
    from morphlm.config import ExperimentConfig, apply_overrides
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    base = [f'data.pipeline={a.pipeline}'] + ([f'frozen.facts={a.facts}'] if a.facts else []) + a.over
    cfg0 = apply_overrides(ExperimentConfig(), base)
    widths = a.widths or [cfg0.model.d_model]
    layers = a.layers or [cfg0.model.n_layers]
    if device.startswith('cuda'):
        p = torch.cuda.get_device_properties(0)
        card = f'{p.name}, {p.total_memory / 2 ** 30:.1f} GB'
    else:
        card = 'CPU — times are not the GPU\'s; use for checking only'
    print(f'fit check on {device} ({card}); {cfg0.train.max_steps:,} steps, eval every {cfg0.train.eval_every}, '
          f'batch {cfg0.train.batch_size} × {cfg0.data.seq_len}')
    print(f'  {"label":56} {"params":>8} {"emb+head":>9} {"core":>8} {"peak GB":>8} {"steps/s":>8} {"run":>8}')
    for w in widths:
        for n in layers:
            cfg = apply_overrides(ExperimentConfig(), base + [f'model.d_model={w}', f'model.n_layers={n}'])
            cfg.validate()
            try:
                parts, peak, t_step, eval_s, run_s = measure(cfg, a.steps, a.warmup, a.val_windows, device)
            except torch.cuda.OutOfMemoryError:
                print(f'  {cfg.label()[:56]:56} OUT OF MEMORY at batch {cfg.train.batch_size}')
                torch.cuda.empty_cache()
                continue
            emb = parts['embedding_trained'] + parts['embedding_factor'] + parts['output_head'] + parts['positional']
            core = parts['trainable_total'] - emb - parts.get('auxiliary_head', 0)
            print(f'  {cfg.label()[:56]:56} {parts["trainable_total"] / 1e6:7.2f}M {emb / 1e6:8.2f}M {core / 1e6:7.2f}M '
                  f'{peak:8.2f} {1 / t_step:8.1f} {run_s / 60:6.1f} min')
    print('  run = the training steps plus every validation pass, as the loop makes them; '
          'add ~20 s per run for loading')


if __name__ == '__main__':
    main()
