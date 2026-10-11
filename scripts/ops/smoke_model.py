#!/usr/bin/env python3
"""
smoke_model.py — build each cell's REAL model and push one batch through it.

`inspect` builds the feature blocks but not the model. This goes one step
further, with no data and no cache: it constructs the model exactly as the
trainer does (`build_model`), feeds a random batch with random VALID codes for
every context fact, runs forward and backward, and checks that

  * the loss is finite;
  * every context fact's learned table receives gradient — a fact that never
    reaches the output would train nothing and show no effect, silently;
  * under `causal_identity`, the first word of a sequence cannot depend on
    its own code (the shift reads only earlier words).

Seconds per cell. Run it before a night that introduces new facts:

    python scripts/ops/smoke_model.py --cells pos mfeatlex mstruct,mfeatlex
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', required=True,
                    help="fact lists, e.g. mstruct mfeatlex mstruct,mfeatlex")
    ap.add_argument('--bpe-path', default='data/bpe_17k.json')
    ap.add_argument('--no-causal', action='store_true')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='extra config overrides for every cell, e.g. frozen.inject=add')
    args = ap.parse_args()

    import torch
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.modeling.transformer import build_model
    from morphlm.data.stream import cache_blocks
    from morphlm.training.loop import get_device
    device = get_device()
    torch.manual_seed(0)
    failures = []

    for facts in args.cells:
        ov = ['data.pipeline=bpe', f'data.bpe_path={args.bpe_path}',
              'frozen.trainable=true', f'frozen.facts={facts}']
        if not args.no_causal:
            ov.append('frozen.causal_identity=true')
        ov += list(args.set)
        cfg = apply_overrides(ExperimentConfig(), ov)
        try:
            model, V = build_model(cfg, verbose=False)
            model = model.to(device).train()
            names = [n.strip() for n in facts.split(',')]
            blocks = cache_blocks(cfg, names)
            B, L = 4, 64
            x = torch.randint(0, V - 4, (B, L), device=device)
            y = torch.randint(0, V - 4, (B, L), device=device)
            c = torch.stack([torch.randint(0, b.n_classes, (B, L), device=device)
                             for b in blocks], dim=1)
            logits, loss = model(x, y, codes=c)
            ok_loss = bool(torch.isfinite(loss))
            loss.backward()
            tables = getattr(model.embed, 'factor_tables', None)
            grads = []
            if tables is not None:
                for i, t in enumerate(tables):
                    g = t.weight.grad
                    grads.append(float(g.abs().sum()) if g is not None else 0.0)
            zero = [bool(getattr(b, 'zero_slot', False)) for b in blocks]
            # a zero slot's table must get NO gradient; every other table must
            ok_grad = bool(grads) and all((g > 0) != z for g, z in zip(grads, zero))

            ok_causal = True
            if not args.no_causal:
                model.eval()
                with torch.no_grad():
                    c2 = c.clone()
                    # change every code in the batch EXCEPT nothing before the
                    # first word: position 0's output must not move
                    c2[:, :, 0] = (c2[:, :, 0] + 1) % torch.tensor(
                        [b.n_classes for b in blocks], device=device).view(1, -1)
                    a, _ = model(x, None, codes=c)
                    b2, _ = model(x, None, codes=c2)
                    ok_causal = bool(torch.allclose(a[:, 0], b2[:, 0], atol=1e-5))
            if all(zero):
                # the strongest check a zero slot allows: the codes are inert
                model.eval()
                with torch.no_grad():
                    c4 = torch.stack([torch.randint(0, b.n_classes, (B, L), device=device)
                                      for b in blocks], dim=1)
                    o1, _ = model(x, None, codes=c)
                    o2, _ = model(x, None, codes=c4)
                    ok_causal = ok_causal and bool(torch.equal(o1, o2))
            T = sum(b.dim for b in blocks)
            status = 'OK' if (ok_loss and ok_grad and ok_causal) else 'FAIL'
            print(f'  {facts:22} T={T:3}  loss {float(loss):7.3f} finite={ok_loss}  '
                  f'gradient as expected={ok_grad} {["%.2g" % g for g in grads]}  '
                  f'first word ignores its own code={ok_causal}   {status}')
            if status == 'FAIL':
                failures.append(facts)
        except Exception as e:                      # report every cell, then fail
            print(f'  {facts:22} ERROR {type(e).__name__}: {e}')
            failures.append(facts)

    if failures:
        raise SystemExit(f'\n  smoke test FAILED for {failures}')
    print('\n  smoke test passed for every cell')


if __name__ == '__main__':
    main()
