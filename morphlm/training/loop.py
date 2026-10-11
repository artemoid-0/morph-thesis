"""
loop.py — train one cell.

TWO CORRECTNESS FIXES OVER THE PREVIOUS VERSION
-----------------------------------------------
1. SEEDING. There was none anywhere. Runs were not reproducible, and — worse —
   the expected effect size for the freezing comparison is small, so a single
   unseeded run per cell cannot distinguish a real 1-3% difference from
   initialization noise. Seeding is step one; multiple seeds per cell is the
   actual answer for effect-size confidence, and `train.seed` makes that a
   one-flag sweep.

   Note that seeding every cell identically does NOT give matched
   initialization across cells: different vocabulary sizes draw different
   numbers of random values. Same seed buys reproducibility, not variance
   control.

2. LR HORIZON. The cosine schedule used to be parameterized by max_steps, so
   two cells run to different lengths had different learning rates at the same
   step and were not comparable — the 40k calibration run and a 50k matrix cell
   were silently different experiments. Now the schedule is pinned by
   lr_horizon while max_steps only decides when to stop, so cells of different
   length remain comparable by construction.

Reported result is always BEST-val, never last-val: WikiText-103 is a
documented overfitter, and the minimum must not be contaminated by what happens
after it.
"""

from __future__ import annotations

import math
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def set_seed(seed: int):
    """Seed every source of randomness the training loop touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, val_ds, device):
    """Full pass over validation. Returns loss / perplexity / bits-per-byte."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y, c in loader:
        x, y = x.to(device), y.to(device)
        c = c.to(device) if c.numel() else None
        _, loss = model(x, y, codes=c)
        n = y.numel()
        total_loss += loss.item() * n
        total_tokens += n

    mean_loss = total_loss / max(total_tokens, 1)
    result = {
        'mean_loss': mean_loss,
        'perplexity': math.exp(mean_loss),
        'bits_per_byte': val_ds.bits_per_byte(total_loss, total_tokens),
        'tokens': total_tokens,
    }
    model.train()
    return result


def train_cell(cfg, verbose=True):
    """
    Train one experiment cell. Returns (model, log, best).

    `best` is the eval dict at minimum validation loss — the number reported
    for this cell.
    """
    cfg.validate()
    set_seed(cfg.train.seed)
    device = get_device()

    if verbose:
        from ..features.blocks import describe_blocks
        print(f'\n=== cell: {cfg.describe()} on {device} ===')
        print(f'  frozen: {describe_blocks(cfg)}')

    from ..pipelines.registry import build_pipeline
    from ..data.windows import make_datasets
    from ..modeling.transformer import build_model

    pipeline = build_pipeline(cfg, verbose=verbose)
    train_ds, val_ds = make_datasets(pipeline, cfg, verbose=verbose)

    generator = torch.Generator()
    generator.manual_seed(cfg.train.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, drop_last=True,
                              generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            shuffle=False, drop_last=False)

    model, vocab_size = build_model(cfg, verbose=verbose)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    warmup = cfg.train.warmup
    horizon = cfg.train.lr_horizon
    peak = cfg.train.lr

    def lr_at(step):
        if step < warmup:
            return peak * step / max(warmup, 1)
        prog = (step - warmup) / max(horizon - warmup, 1)
        prog = min(prog, 1.0)
        return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    from ..progress import Progress, fmt_duration
    log = []
    best = None
    best_state = None
    since_improve = 0
    step = 0
    off_step = getattr(cfg.frozen, 'feature_off_step', None)
    on_step = getattr(cfg.frozen, 'feature_on_step', None)
    drop_until = getattr(cfg.frozen, 'feature_dropout_until', None)
    import os as _os
    ckpt_every = int(_os.environ.get('MORPHLM_CHECKPOINT_EVERY', '0') or 0)   # D2 (§33.48)
    aux_w = cfg.frozen.aux_weight if getattr(cfg.frozen, 'role', 'input') == 'target' else 0.0
    # ETA is to max_steps, so it is an UPPER BOUND: early stopping can end the
    # run sooner, and the printed estimate does not try to predict that.
    prog = Progress(total=cfg.train.max_steps)
    t0 = time.time()
    model.train()
    done = False

    while not done:
        for x, y, c in train_loader:
            x, y = x.to(device), y.to(device)
            c = c.to(device) if c.numel() else None

            for g in opt.param_groups:
                g['lr'] = lr_at(step)

            if off_step is not None and off_step > 0 and step == off_step:
                # the feature switch: close the gate, and let only evaluations
                # from here on count — the result describes the model WITHOUT
                # the feature, and the jump after the switch cannot exhaust
                # the patience counter (journey §33.33)
                model.embed.switch_features_off()
                best, best_state, since_improve = None, None, 0
                if verbose:
                    print(f'  features switched off at step {step}: best and patience reset')
            if on_step is not None and on_step > 0 and step == on_step:
                # the reverse switch: the feature arrives mid-training; only
                # evaluations with it count toward the result (journey §33.34)
                model.embed.switch_features_on()
                best, best_state, since_improve = None, None, 0
                if verbose:
                    print(f'  features switched on at step {step}: best and patience reset')

            if drop_until:
                # weaning: the keep rate falls linearly to 0 at drop_until;
                # from there only feature-free evaluations count (§33.47)
                model.embed.set_feature_keep(max(0.0, 1.0 - step / drop_until))
                if step == drop_until:
                    best, best_state, since_improve = None, None, 0
                    if verbose:
                        print(f'  features fully dropped at step {step}: best and patience reset')

            _, loss = model(x, y, codes=c)
            total = loss + aux_w * model.last_aux_loss if aux_w and model.last_aux_loss is not None else loss
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.train.grad_clip)
            opt.step()
            if ckpt_every and (step % ckpt_every == 0 or step == cfg.train.max_steps - 1):
                from .metrics import save_step_checkpoint
                save_step_checkpoint(cfg, model, step)

            if step % cfg.train.eval_every == 0 or step == cfg.train.max_steps - 1:
                ev = evaluate(model, val_loader, val_ds, device)
                ev['step'] = step
                ev['train_loss'] = loss.item()
                ev['lr'] = lr_at(step)
                ev['elapsed'] = time.time() - t0
                log.append(ev)

                improved = best is None or ev['mean_loss'] < best['mean_loss']
                marker = ''
                if improved:
                    best = ev
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    since_improve = 0
                    marker = '  *best'
                else:
                    since_improve += 1

                if verbose:
                    print(f'  step {step:6d} | train {loss.item():.3f} | '
                          f'val {ev["mean_loss"]:.3f} | '
                          f'ppl {ev["perplexity"]:8.2f} | '
                          f'bpb {ev["bits_per_byte"]:.4f} | '
                          f'{prog.status(step)}{marker}')

                if (cfg.train.patience is not None
                        and since_improve >= cfg.train.patience):
                    if verbose:
                        print(f'  early stop: no improvement for '
                              f'{cfg.train.patience} evals '
                              f'(best at step {best["step"]})')
                    done = True
                    break

            step += 1
            if step >= cfg.train.max_steps:
                done = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f'  finished {step:,} steps in {fmt_duration(prog.elapsed)} '
              f'({prog.rate(step):.1f} steps/s)')

    # ── provenance: everything a thesis table will need, captured NOW ─────────
    #
    # Wrapped defensively AS A WHOLE. Provenance is valuable but it is not the
    # result: a metadata bug must never cost an hour of GPU time. An earlier
    # version had an ImportError in frozen_detail that would have crashed here,
    # after 50,000 steps, at save time. Each field is captured independently so
    # one failure does not take the others with it, and failures are RECORDED
    # in the output rather than swallowed.
    meta = {}
    from ..provenance import (environment, parameter_breakdown, frozen_detail,
                              data_stats, convergence, tokenizer_record)
    for key, fn in (
            ('environment', lambda: environment(getattr(cfg.data, 'spacy_model', None))),
            ('parameters', lambda: parameter_breakdown(model, cfg)),
            ('frozen_blocks', lambda: frozen_detail(cfg)),
            ('data', lambda: data_stats(train_ds, val_ds)),
            ('tokenizer', lambda: tokenizer_record(cfg)),
            ('convergence', lambda: convergence(log, cfg.train.max_steps)),
            ('timing', lambda: {'wall_seconds': prog.elapsed,
                                'steps_completed': step,
                                'steps_per_second': prog.rate(step),
                                'device': device}),
    ):
        try:
            meta[key] = fn()
        except Exception as e:
            meta.setdefault('provenance_errors', {})[key] = \
                f'{type(e).__name__}: {e}'
            if verbose:
                print(f'  (provenance: {key} failed — {type(e).__name__}: {e})')

    # per-frequency-bucket perplexity on the BEST model, computed here so the
    # breakdown never depends on a checkpoint surviving
    if getattr(cfg.train, 'eval_buckets', True):
        try:
            from ..eval.buckets import (token_frequencies, bucketed_perplexity,
                                        adaptive_edges)
            import numpy as _np
            freqs = token_frequencies(train_ds)
            # equal-mass edges: fixed thresholds put 64-70% of BPE tokens in a
            # single bucket, which made the rare-token question unanswerable
            edges = adaptive_edges(freqs, _np.asarray(val_ds.ids))
            meta['bucket_edges'] = [float(e) for e in edges]
            meta['buckets'] = bucketed_perplexity(model, val_loader, freqs,
                                                  device, edges=edges)
            if verbose:
                print('  per-frequency buckets computed')
        except Exception as e:
            meta.setdefault('provenance_errors', {})['buckets'] = \
                f'{type(e).__name__}: {e}'
            if verbose:
                print(f'  (bucket eval skipped: {e})')

    return model, log, best, meta
