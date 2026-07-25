"""
train.py

Training + evaluation for one cell of the experiment matrix
(a pipeline × embedding-variant combination).

Phase-1 first job is CALIBRATION: train the BPE-baseline cell and watch the
learning curve to confirm the apparatus sits in the right regime — learns the
text well, but with visible headroom for an embedding advantage to surface.
Too-low final loss → model/corpus too easy; too-high plateau → too weak.

The eval pass reports BOTH metrics:
  - per-token perplexity (comparable WITHIN a pipeline)
  - bits-per-byte        (comparable ACROSS pipelines)
"""

import time
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import make_datasets
from model import build_model


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


@torch.no_grad()
def evaluate(model, loader, val_ds, device):
    """
    Full pass over the validation loader.
    Returns dict with mean_loss (nats), perplexity, bits_per_byte.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n      # un-mean to sum
        total_tokens += n

    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss)
    bpb = val_ds.bits_per_byte(total_loss, total_tokens)
    model.train()
    return {'mean_loss': mean_loss, 'perplexity': ppl,
            'bits_per_byte': bpb, 'tokens': total_tokens}


def train_cell(pipeline, pipeline_name, vocab_df=None, vocab_size=None,
               variant='baseline',
               # model
               d_model=128, n_layers=4, n_heads=4, seq_len=256, dropout=0.1,
               tie_output='none', freeze_amplify=1,
               # optimization
               batch_size=32, lr=3e-4, max_steps=2000, warmup=100,
               eval_every=250, grad_clip=1.0,
               # early stopping: stop if val loss hasn't improved for this many
               # consecutive evals (None disables). The reported result is always
               # the BEST val point, never the last — so overfitting after the
               # minimum can't contaminate the comparison.
               patience=5,
               # data
               max_articles_train=3000, max_articles_val=None,
               device=None, verbose=True):
    """
    Train one matrix cell. Returns (model, log, best) where `best` is the eval
    dict at the minimum validation loss — the number we report for this cell.
    """
    device = device or get_device()
    if verbose:
        print(f'\n=== training cell: pipeline={pipeline_name} '
              f'variant={variant} on {device} ===')

    train_ds, val_ds = make_datasets(
        pipeline, pipeline_name, seq_len=seq_len,
        max_articles_train=max_articles_train,
        max_articles_val=max_articles_val, verbose=verbose)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              drop_last=False)

    model = build_model(
        vocab_df=vocab_df, vocab_size=vocab_size, variant=variant,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        seq_len=seq_len, dropout=dropout, tie_output=tie_output,
        freeze_amplify=freeze_amplify).to(device)

    if verbose:
        print(f'  model: {model.cfg}')
        print(f'  trainable params: {model.num_params():,}')
        print(f'  embedding: {model.embed.extra_repr()}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_at(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        # cosine decay to 10% over the remaining steps
        prog = (step - warmup) / max(max_steps - warmup, 1)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    log = []
    best = None
    best_state = None
    since_improve = 0
    step = 0
    t0 = time.time()
    model.train()
    done = False
    while not done:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)

            _, loss = model(x, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            if step % eval_every == 0 or step == max_steps - 1:
                ev = evaluate(model, val_loader, val_ds, device)
                ev['step'] = step
                ev['train_loss'] = loss.item()
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
                    print(f'  step {step:5d} | train {loss.item():.3f} | '
                          f'val {ev["mean_loss"]:.3f} | '
                          f'ppl {ev["perplexity"]:8.2f} | '
                          f'bpb {ev["bits_per_byte"]:.4f} | '
                          f'{ev["elapsed"]:.0f}s{marker}')

                # early stop: val hasn't improved for `patience` consecutive evals
                if patience is not None and since_improve >= patience:
                    if verbose:
                        print(f'  early stop: no val improvement for '
                              f'{patience} evals (best at step {best["step"]})')
                    done = True
                    break

            step += 1
            if step >= max_steps:
                done = True
                break

    # restore best checkpoint so the returned model IS the best-val model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, log, best


def calibrate_bpe(bpe_path='data/bpe_tokenizer.json',
                  max_steps=40000, max_articles_train=None,
                  eval_every=500, patience=6, **kw):
    """
    THE CALIBRATION RUN. Train BPE-baseline and print the learning curve so we
    can judge whether the apparatus has headroom for an embedding advantage.

    Defaults now target the FULL corpus (max_articles_train=None) with early
    stopping on validation loss — the run ends itself when val stops improving,
    and the reported numbers are the BEST val point, not the last.
    """
    from pipeline_bpe import BPEPipeline
    bpe = BPEPipeline(tokenizer_path=bpe_path)
    vocab_size = bpe.vocab_size_actual()

    model, log, best = train_cell(
        pipeline=bpe, pipeline_name='bpe', vocab_size=vocab_size,
        variant='baseline', max_steps=max_steps,
        max_articles_train=max_articles_train,
        eval_every=eval_every, patience=patience, **kw)

    print('\n=== calibration summary ===')
    print(f'  best val @ step {best["step"]}:')
    print(f'    loss {best["mean_loss"]:.3f} | ppl {best["perplexity"]:.1f} '
          f'| bpb {best["bits_per_byte"]:.4f}')
    print(f'  start ppl {log[0]["perplexity"]:.1f} → best ppl '
          f'{best["perplexity"]:.1f}')
    print('  Regime read: we want best-val ppl well below random '
          f'(≈{vocab_size}) with the train/val curves tracking together —')
    print('  if val turned up long before early-stop, corpus may still be thin.')
    return model, log, best


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['calibrate'], default='calibrate',
                   nargs='?')
    p.add_argument('--steps', type=int, default=40000)
    p.add_argument('--articles', type=int, default=0,
                   help='0 = full corpus (default); otherwise cap')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--patience', type=int, default=6)
    args = p.parse_args()

    if args.command == 'calibrate':
        articles = None if args.articles == 0 else args.articles
        calibrate_bpe(max_steps=args.steps,
                      max_articles_train=articles,
                      batch_size=args.batch, lr=args.lr,
                      eval_every=args.eval_every, patience=args.patience)