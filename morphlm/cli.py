"""
cli.py — entry point.

    python -m morphlm.cli run --pipeline morph --facts dep
    python -m morphlm.cli run --pipeline bpe -o train.seed=1
    python -m morphlm.cli cache --pipeline morph        # build cache only
    python -m morphlm.cli inspect --pipeline morph --facts dep
    python -m morphlm.cli build entities
    python -m morphlm.cli build vocab

Every knob is reachable as `-o section.field=value`, so there is exactly one
name per setting and no duplicated argparse defaults drifting out of sync.
"""

from __future__ import annotations

import argparse
import dataclasses

from .config import ExperimentConfig, apply_overrides


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
    model, log, best, meta = train_cell(cfg, verbose=True)
    path = save_result(cfg, log, best, meta=meta)
    if not args.no_checkpoint:
        ck = save_checkpoint(cfg, model)
        print(f'[saved] {ck}')

    print(f'\n[saved] {path}')
    print(f'  {cfg.describe()}: ppl {best["perplexity"]:.2f} | '
          f'bpb {best["bits_per_byte"]:.4f} | best step {best["step"]}')


def cmd_cache(args):
    """Build the token cache without training — measure the spaCy cost first."""
    from .pipelines.registry import build_pipeline
    from .data.windows import make_datasets

    cfg = _base_config(args)
    print(f'building cache for {cfg.describe()}')
    pipeline = build_pipeline(cfg, verbose=True)
    splits = list(getattr(args, 'splits', None) or ['train', 'validation'])
    if sorted(splits) == ['train', 'validation']:
        make_datasets(pipeline, cfg, verbose=True)          # the historical path
    else:
        # e.g. validation alone: a pilot build in minutes, to check a new
        # corpus before committing hours to its train split
        from .data.stream import build_stream
        for split in splits:
            build_stream(pipeline, cfg, split=split, verbose=True)
    print('cache ready.')


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
    if args.what in ('bpe', 'all'):
        # one training path for every tokenizer (data/tokenization.py); the
        # legacy sidecar is exactly what this command always wrote
        from .data.tokenization import train_tokenizer
        print(f'training BPE (vocab={args.bpe_vocab}, '
              f'pre_tokenizer={args.pre_tokenizer}, '
              f'algorithm={getattr(args, "algorithm", "bpe")})...')
        train_tokenizer(cfg, args.bpe_vocab, args.bpe_out,
                        algorithm=getattr(args, 'algorithm', 'bpe'),
                        pre_tokenizer=args.pre_tokenizer, max_lines=args.bpe_lines,
                        sidecar='legacy')
        print(f'  saved -> {args.bpe_out}')
        print('  NOTE: point runs at it with -o data.bpe_path=' + args.bpe_out)

    if args.what in ('vocab', 'all'):
        from .vocab.build import build_vocabulary
        print('building vocabulary...')
        df = build_vocabulary(cfg.data.morpholex_path, 'data/entity_vocab.csv')
        df.to_csv(cfg.data.vocab_csv, index=False)
        print(f'  wrote {cfg.data.vocab_csv}')


def build_parser():
    """
    The CLI's argument parser, exposed so other tools can parse a command line
    exactly as `morphlm.cli` would. `scripts/ops/run_batch.py` validates every job
    in a batch this way before any GPU time is spent; a second hand-written
    parser would drift from this one, and a batch that validates against the
    wrong parser is worse than one that is not validated at all.
    """
    p = argparse.ArgumentParser(prog='morphlm')
    sub = p.add_subparsers(dest='command', required=True)

    def add_common(sp):
        sp.add_argument('--pipeline', choices=['bpe', 'morph', 'det'])
        sp.add_argument('--facts', help="comma list, e.g. 'dep' or 'cat,dep'")
        sp.add_argument('-o', '--override', action='append', metavar='k=v',
                        help='e.g. -o train.seed=1 -o train.max_steps=50000')

    r = sub.add_parser('run', help='train one cell')
    add_common(r)
    r.add_argument('--no-checkpoint', action='store_true',
                   help='skip saving model weights (~14MB per run)')
    r.set_defaults(func=cmd_run)

    c = sub.add_parser('cache', help='build the token cache only')
    add_common(c)
    c.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'],
                   help='which splits (default: train and validation)')
    c.set_defaults(func=cmd_cache)

    i = sub.add_parser('inspect', help='show the frozen layout, train nothing')
    add_common(i)
    i.set_defaults(func=cmd_inspect)

    b = sub.add_parser('build', help='build data assets')
    b.add_argument('what', choices=['entities', 'vocab', 'bpe', 'all'])
    b.add_argument('--articles', type=int, default=6000)
    b.add_argument('--bpe-vocab', type=int, default=10000)
    b.add_argument('--bpe-lines', type=int, default=0,
                   help='0 = full corpus')
    b.add_argument('--bpe-out', default='data/bpe_bytelevel.json')
    b.add_argument('--algorithm', default='bpe', choices=['bpe', 'unigram'],
                   help='unigram is the Bostrom & Durrett (2020) '
                        'comparison arm; both are trained losslessly '
                        'with the ByteLevel pre-tokenizer')
    b.add_argument('--pre-tokenizer', default='bytelevel',
                   choices=['bytelevel', 'whitespace'])
    add_common(b)
    b.set_defaults(func=cmd_build)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
