#!/usr/bin/env python3
"""
probe.py — when does a model come to know what a feature encodes? (DIRECTIONS D2)

For every checkpoint of a run (ckpt/step_*.pt, written when the run trained
with MORPHLM_CHECKPOINT_EVERY=N), a linear probe reads a feature's class from
the model's hidden states at each layer — trained on half the validation
windows, tested on the other half (alternating windows; tokens labelled with the
'none' sentinel left out). Layer 0 is the input itself: its accuracy is what
token identity alone says; what deeper layers add is learned from context.

The model is fed its OWN features; the labels come from the probed feature —
two validation datasets over the same windows, their tokens asserted identical.

--previous probes for what a causal feature actually carries: at each token,
the probed fact of the PREVIOUS unit (word or punctuation run), shifted exactly
as the model shifts its features (embedding.word_runs and causal_shift, word
starts from the tokenizer's tables), the window's first unit left out. The
feature becomes redundant when the model can infer that from context
(DIRECTIONS D2) — so this, not the current word's tag, is the knowledge the
law's axis should measure.

    python scripts/analysis/probe.py --run RUN_ID --fact pos
    python scripts/analysis/probe.py --run RUN_ID --fact pos --previous
    python scripts/analysis/probe.py --run RUN_ID --fact pos --layers 0 4 --tokens 80000

Writes results/audits/probe_<run>_<fact>[prev]_<UTC timestamp>.json — the
packer collects audits by that timestamp, even for a run packed before — and
prints accuracy by step and layer.
"""
import os
import re
import sys
import glob
import json
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def fit_probe(h, y, train_w, sentinel, n_cls, epochs, dev):
    """Multinomial logistic regression; held-out accuracy and the majority baseline."""
    import torch
    W, L, d = h.shape
    keep = y != sentinel
    tr = keep & train_w[:, None].expand(W, L)
    te = keep & ~train_w[:, None].expand(W, L)
    Xtr, ytr, Xte, yte = h[tr], y[tr], h[te], y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev)
    ytr, yte = ytr.to(dev), yte.to(dev)
    torch.manual_seed(0)
    lin = torch.nn.Linear(d, n_cls).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        acc = (lin(Xte).argmax(-1) == yte).float().mean().item()
    majority = torch.bincount(ytr, minlength=n_cls).argmax()
    return acc, (yte == majority).float().mean().item(), int(te.sum())


def previous_labels(y, ids, bpe_path, sentinel, model=None):
    """Each position's label replaced by the PREVIOUS unit's, as the model's
    causal shift replaces its features; the window's first unit gets the
    sentinel. Word starts come from the model's own tables when it has them,
    otherwise from the tokenizer's — the same rule either way."""
    import types
    import torch
    from morphlm.modeling.embedding import HybridEmbedding, word_runs, causal_shift
    emb = getattr(model, 'embed', None) if model is not None else None
    if emb is None or getattr(emb, 'tok_alpha', None) is None:
        from morphlm.features.lexical import tokenizer_tables
        tb = tokenizer_tables(bpe_path)
        frag = getattr(tb, 'frag', None)
        emb = types.SimpleNamespace(
            tok_alpha=torch.from_numpy(tb[0]), tok_opens=torch.from_numpy(tb[1]),
            tok_frag=torch.from_numpy(frag) if frag is not None and frag.any() else None)
    ids = ids.to(emb.tok_alpha.device)
    src, start = word_runs(HybridEmbedding._word_starts(emb, ids))
    return causal_shift(y.to(ids.device), src, start, sentinel).cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='run id (folder under results/)')
    ap.add_argument('--fact', required=True, help='the feature to probe for, e.g. pos or ucase')
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--layers', nargs='*', type=int, default=None, help='default: every layer')
    ap.add_argument('--tokens', type=int, default=60000, help='validation tokens to use, both halves')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--previous', action='store_true',
                    help="probe for the previous unit's class, shifted as the model shifts features")
    args = ap.parse_args()

    import torch
    import dataclasses
    from morphlm.training.metrics import load_checkpoint
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.windows import make_val_dataset
    from morphlm.features.blocks import build_blocks

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = os.path.join(args.results_dir, args.run)
    files = sorted(glob.glob(os.path.join(run_dir, 'ckpt', 'step_*.pt')))
    if not files:
        sys.exit(f'no checkpoints in {run_dir}/ckpt — train with MORPHLM_CHECKPOINT_EVERY=N')
    rel = lambda f: os.path.relpath(f, run_dir)
    _, cfg = load_checkpoint(args.run, args.results_dir, filename=rel(files[0]))
    cfg_lab = dataclasses.replace(cfg, frozen=dataclasses.replace(
        cfg.frozen, facts=(args.fact,), shuffle_codes=False, role='input'))
    ds_in = make_val_dataset(build_pipeline(cfg, verbose=False), cfg, verbose=False)
    ds_lab = make_val_dataset(build_pipeline(cfg_lab, verbose=False), cfg_lab, verbose=False)
    n_cls = build_blocks(cfg_lab)[0].n_classes
    n_win = min(len(ds_in), max(2, args.tokens // cfg.data.seq_len))
    X, C, Y = [], [], []
    for i in range(n_win):
        x, _, c = ds_in[i]
        x2, _, c2 = ds_lab[i]
        assert torch.equal(x, x2), f'window {i}: model inputs and probe labels disagree on tokens'
        X.append(x); C.append(c); Y.append(c2[0])
    X, C, Y = torch.stack(X), torch.stack(C), torch.stack(Y)
    if args.previous:
        first, _ = load_checkpoint(args.run, args.results_dir, filename=rel(files[0]))
        Y = previous_labels(Y, X, cfg.data.bpe_path, n_cls - 1, model=first)
        del first
    train_w = torch.arange(n_win) % 2 == 0

    what = f"the previous unit's {args.fact}" if args.previous else args.fact
    print(f'  {args.run}: probing {what} ({n_cls} classes) over {n_win} windows, {len(files)} checkpoints')
    out = {'run': args.run, 'fact': args.fact, 'previous': bool(args.previous), 'windows': n_win, 'rows': []}
    for f in files:
        step = int(re.search(r'step_(\d+)\.pt$', f).group(1))
        model, _ = load_checkpoint(args.run, args.results_dir, filename=rel(f))
        model.to(dev).eval()
        per_layer = None
        with torch.no_grad():
            for b in range(0, n_win, 16):
                _, _, hidden = model(X[b:b + 16].to(dev), codes=C[b:b + 16].to(dev), return_hidden=True)
                hs = [h.float().cpu() for h in hidden]
                per_layer = [[h] for h in hs] if per_layer is None else [p + [h] for p, h in zip(per_layer, hs)]
        layers = range(len(per_layer)) if args.layers is None else args.layers
        line = []
        for l in layers:
            acc, maj, n = fit_probe(torch.cat(per_layer[l]), Y, train_w, n_cls - 1, n_cls, args.epochs, dev)
            out['rows'].append({'step': step, 'layer': l, 'accuracy': acc, 'majority': maj, 'test_tokens': n})
            line.append(f'L{l} {acc:.3f}')
        print(f'    step {step:>6}: ' + '  '.join(line) + f'   (majority {maj:.3f}, {n} test tokens)')
    os.makedirs(os.path.join(args.results_dir, 'audits'), exist_ok=True)
    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S') + '+0000'
    tag = args.fact + ('prev' if args.previous else '')
    path = os.path.join(args.results_dir, 'audits', f'probe_{args.run}_{tag}_{ts}.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'  -> {path}')


if __name__ == '__main__':
    main()
