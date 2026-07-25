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

    log = []
    best = None
    best_state = None
    since_improve = 0
    step = 0
    t0 = time.time()
    model.train()
    done = False

    while not done:
        for x, y, c in train_loader:
            x, y = x.to(device), y.to(device)
            c = c.to(device) if c.numel() else None

            for g in opt.param_groups:
                g['lr'] = lr_at(step)

            _, loss = model(x, y, codes=c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           cfg.train.grad_clip)
            opt.step()

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
                          f'{ev["elapsed"]:.0f}s{marker}')

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

    return model, log, best
