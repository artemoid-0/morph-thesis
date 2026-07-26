"""
cli.py — entry point.

    python -m morphlm.cli run --pipeline morph --facts dep
    python -m morphlm.cli run --pipeline bpe -o train.seed=1
    python -m morphlm.cli cache --pipeline morph        # build cache only
    python -m morphlm.cli report
    python -m morphlm.cli inspect --pipeline morph --facts dep
    python -m morphlm.cli build entities
    python -m morphlm.cli build vocab

Every knob is reachable as `-o section.field=value`, so there is exactly one
name per setting and no duplicated argparse defaults drifting out of sync.
"""

from __future__ import annotations

import argparse
import dataclasses

from .config import ExperimentConfig, DataConfig, FrozenConfig, apply_overrides


def _base_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if getattr(args, 'pipeline', None):
        cfg = dataclasses.replace(
            cfg, data=dataclasses.replace(cfg.data, pipeline=args.pipeline))
    if getattr(args, 'facts', None) is not None:
        facts = tuple(f.strip() for f in args.facts.split(',') if f.strip())
        cfg = dataclasses.replace(
            cfg, frozen=dataclasses.replace(cfg.frozen, facts=facts))
    cfg = apply_overrides(cfg, getattr(args, 'override', None))
    cfg.validate()
    return cfg


def cmd_run(args):
    from .training.loop import train_cell
    from .training.metrics import save_result, save_checkpoint

    cfg = _base_config(args)
    model, log, best = train_cell(cfg, verbose=True)
    path = save_result(cfg, log, best)
    if not args.no_checkpoint:
        ck = save_checkpoint(cfg, model)
        print(f'[saved] {ck}')

    print(f'\n[saved] {path}')
    print(f'  {cfg.describe()}: ppl {best["perplexity"]:.2f} | '
          f'bpb {best["bits_per_byte"]:.4f} | best step {best["step"]}')

    if args.buckets:
        from .eval.buckets import (token_frequencies, bucketed_perplexity,
                                   print_buckets)
        from .data.windows import make_datasets
        from .pipelines.registry import build_pipeline
        from .training.loop import get_device
        from torch.utils.data import DataLoader
        pipeline = build_pipeline(cfg, verbose=False)
        train_ds, val_ds = make_datasets(pipeline, cfg, verbose=False)
        loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)
        freqs = token_frequencies(train_ds)
        rows = bucketed_perplexity(model, loader, freqs, get_device())
        print_buckets(rows, cfg.label())


def cmd_cache(args):
    """Build the token cache without training — measure the spaCy cost first."""
    from .pipelines.registry import build_pipeline
    from .data.windows import make_datasets

    cfg = _base_config(args)
    print(f'building cache for {cfg.describe()}')
    pipeline = build_pipeline(cfg, verbose=True)
    make_datasets(pipeline, cfg, verbose=True)
    print('cache ready.')


def cmd_report(args):
    from .training.metrics import report
    report()


def cmd_inspect(args):
    """Print what the frozen block WOULD be, without training anything."""
    from .features.blocks import build_blocks, describe_blocks

    cfg = _base_config(args)
    print(f'\ncell: {cfg.describe()}')
    print(f'frozen: {describe_blocks(cfg)}')
    print(f'cache spec: {cfg.frozen.spec()!r}')

    blocks = build_blocks(cfg)
    if not blocks:
        print('\n(baseline: T=0, plain trained embedding)')
        return
    print(f'\n{"block":10} {"source":9} {"dim":>4}  classes')
    for b in blocks:
        cls = ', '.join(b.classes[:6]) + ('...' if len(b.classes) > 6 else '')
        print(f'{b.name:10} {b.source:9} {b.dim:4d}  {cls}')

    d = cfg.model.d_model
    from .features.blocks import total_frozen_width
    T = total_frozen_width(cfg)
    print(f'\nT = {T} of d_model {d}  ({100.0 * T / d:.1f}% frozen, '
          f'{d - T} trained dims)')
    ctx = [b for b in blocks if b.source == 'context']
    if ctx:
        print(f'context facts present ({", ".join(b.name for b in ctx)}): the '
              f'frozen vector varies per OCCURRENCE, so it encodes information '
              f'a static id->vector table cannot represent.')
    else:
        print('all facts static: this is structured parameter tying, not '
              'information injection.')


def cmd_build(args):
    cfg = _base_config(args)
    if args.what in ('entities', 'all'):
        from .vocab.entities import extract_entities
        print('building entity vocabulary (with labels)...')
        extract_entities(max_articles=args.articles,
                         spacy_model=cfg.data.spacy_model)
    if args.what in ('vocab', 'all'):
        from .vocab.build import build_vocabulary
        print('building vocabulary...')
        df = build_vocabulary(cfg.data.morpholex_path, 'data/entity_vocab.csv')
        df.to_csv(cfg.data.vocab_csv, index=False)
        print(f'  wrote {cfg.data.vocab_csv}')


def main():
    p = argparse.ArgumentParser(prog='morphlm')
    sub = p.add_subparsers(dest='command', required=True)

    def add_common(sp):
        sp.add_argument('--pipeline', choices=['bpe', 'morph', 'det'])
        sp.add_argument('--facts', help="comma list, e.g. 'dep' or 'cat,dep'")
        sp.add_argument('-o', '--override', action='append', metavar='k=v',
                        help='e.g. -o train.seed=1 -o train.max_steps=50000')

    r = sub.add_parser('run', help='train one cell')
    add_common(r)
    r.add_argument('--buckets', action='store_true',
                   help='also report per-frequency-bucket perplexity')
    r.add_argument('--no-checkpoint', action='store_true',
                   help='skip saving model weights (~14MB per run)')
    r.set_defaults(func=cmd_run)

    c = sub.add_parser('cache', help='build the token cache only')
    add_common(c)
    c.set_defaults(func=cmd_cache)

    i = sub.add_parser('inspect', help='show the frozen layout, train nothing')
    add_common(i)
    i.set_defaults(func=cmd_inspect)

    rep = sub.add_parser('report', help='comparison table from saved results')
    rep.set_defaults(func=cmd_report)

    b = sub.add_parser('build', help='build data assets')
    b.add_argument('what', choices=['entities', 'vocab', 'all'])
    b.add_argument('--articles', type=int, default=6000)
    add_common(b)
    b.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
