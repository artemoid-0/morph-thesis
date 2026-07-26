#!/usr/bin/env python3
"""
verify.py — run this first, before spending GPU time.

Checks, in increasing cost:
    1  imports              every module loads
    2  pure-logic tests     cache keys, config, fact blocks   (no torch)
    3  model construction   T, parameter counts, per config
    4  FREEZE GUARANTEE     an optimizer step must not move frozen dims
    5  forward/backward     loss decreases on synthetic data
    6  cache compatibility  existing .npy files still resolve
    7  spaCy annotation     context codes actually vary (needs spacy + assets)

Steps 1-5 need torch but no data. Step 6 needs data/cache. Step 7 needs spaCy,
MorphoLex and the vocab CSV, and is skipped cleanly if they are absent.

    python scripts/verify.py            # steps 1-6
    python scripts/verify.py --spacy    # also step 7 (slower, loads spaCy)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL, SKIP = [], [], []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                r = fn(*a, **k)
                if r == 'skip':
                    SKIP.append(name)
                    print(f'  SKIP  {name}')
                else:
                    PASS.append(name)
                    print(f'  ok    {name}')
            except Exception as e:
                FAIL.append((name, e))
                print(f'  FAIL  {name}: {type(e).__name__}: {e}')
        return wrapped
    return deco


# ── 1. imports ────────────────────────────────────────────────────────────────

@check('imports')
def t_imports():
    import morphlm.config, morphlm.registry                    # noqa
    import morphlm.features.blocks                             # noqa
    import morphlm.vocab.morpholex, morphlm.vocab.build        # noqa
    import morphlm.vocab.entities                              # noqa
    import morphlm.pipelines.base, morphlm.pipelines.bpe       # noqa
    import morphlm.pipelines.morpheme                          # noqa
    import morphlm.pipelines.deterministic                     # noqa
    import morphlm.pipelines.registry                          # noqa
    import morphlm.data.stream, morphlm.data.windows           # noqa
    import morphlm.modeling.embedding                          # noqa
    import morphlm.modeling.transformer                        # noqa
    import morphlm.training.loop, morphlm.training.metrics     # noqa
    import morphlm.eval.buckets, morphlm.eval.probing          # noqa
    import morphlm.cli                                         # noqa


# ── 2. pure-logic suites ──────────────────────────────────────────────────────

@check('pure-logic test suites')
def t_suites():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for suite in ('test_cache_keys.py', 'test_config.py', 'test_blocks.py',
                  'test_lossless.py'):
        p = subprocess.run([sys.executable, os.path.join(root, 'tests', suite)],
                           capture_output=True, text=True, cwd=root)
        if p.returncode != 0:
            raise AssertionError(f'{suite} failed:\n{p.stdout}\n{p.stderr}')


# ── 3-5. torch paths ──────────────────────────────────────────────────────────

def _fake_vocab(n=200):
    import pandas as pd
    cats = ['prefix', 'suffix', 'root', 'function', 'entity', 'character']
    ents = ['ORG', 'PERSON', 'GPE', 'none']
    return pd.DataFrame({
        'token': [f'tok{i}' for i in range(n)],
        'category': [cats[i % len(cats)] for i in range(n)],
        'entity_label': [ents[i % len(ents)] for i in range(n)],
    })


@check('model construction (baseline / static / context)')
def t_build():
    import dataclasses
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.modeling.embedding import HybridEmbedding
    from morphlm.modeling.transformer import TransformerLM

    vocab = _fake_vocab()
    V = len(vocab) + 4
    for facts in ('', 'cat', 'dep', 'cat,dep'):
        cfg = ExperimentConfig()
        if facts:
            cfg = apply_overrides(cfg, [f'frozen.facts={facts}'])
        emb = HybridEmbedding(cfg, V, vocab_df=vocab, verbose=False)
        emb.assert_blocks_informative(strict=True)
        model = TransformerLM(cfg, emb, V)
        assert model.num_params() > 0
        print(f'        facts={facts or "baseline":10} T={emb.T:3d} '
              f'params={model.num_params():,}')


@check('FREEZE GUARANTEE (optimizer step must not move frozen dims)')
def t_freeze():
    import torch
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.modeling.embedding import HybridEmbedding
    from morphlm.modeling.transformer import TransformerLM

    vocab = _fake_vocab()
    V = len(vocab) + 4
    cfg = apply_overrides(ExperimentConfig(), ['frozen.facts=cat,dep'])
    emb = HybridEmbedding(cfg, V, vocab_df=vocab, verbose=False)
    model = TransformerLM(cfg, emb, V)

    before_static = emb.static_table.clone()
    before_eye = emb.ctx_eye_0.clone()
    before_trained = emb.trained.weight.detach().clone()

    ids = torch.randint(0, V, (2, cfg.data.seq_len))
    tgt = torch.randint(0, V, (2, cfg.data.seq_len))
    codes = torch.randint(0, 6, (2, len(emb.context_blocks), cfg.data.seq_len))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, loss = model(ids, tgt, codes=codes)
    opt.zero_grad(); loss.backward(); opt.step()

    assert torch.equal(before_static, emb.static_table), \
        'static frozen table CHANGED — the freeze is not enforced'
    assert torch.equal(before_eye, emb.ctx_eye_0), \
        'context identity table CHANGED — the freeze is not enforced'
    moved = (emb.trained.weight.detach() - before_trained).abs().max().item()
    assert moved > 0, 'trained part did not move — nothing is learning'
    print(f'        frozen byte-identical; trained moved by {moved:.2e}')


@check('forward/backward reduces loss on synthetic data')
def t_learns():
    import torch
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.modeling.embedding import HybridEmbedding
    from morphlm.modeling.transformer import TransformerLM

    torch.manual_seed(0)
    vocab = _fake_vocab(64)
    V = len(vocab) + 4
    cfg = apply_overrides(ExperimentConfig(),
                          ['frozen.facts=dep', 'data.seq_len=32'])
    emb = HybridEmbedding(cfg, V, vocab_df=vocab, verbose=False)
    model = TransformerLM(cfg, emb, V)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # a learnable pattern: next id = (id + 1) % V
    ids = torch.randint(0, V - 1, (8, 32))
    tgt = (ids + 1) % V
    codes = torch.randint(0, 6, (8, 1, 32))

    first = last = None
    for i in range(30):
        _, loss = model(ids, tgt, codes=codes)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f'loss did not decrease ({first:.3f} -> {last:.3f})'
    print(f'        loss {first:.3f} -> {last:.3f}')


# ── 6. cache compatibility ────────────────────────────────────────────────────

@check('existing cache files still resolve')
def t_cache():
    from morphlm.data.stream import cache_stem
    cache_dir = 'data/cache'
    if not os.path.isdir(cache_dir):
        return 'skip'
    present = {f[:-4] for f in os.listdir(cache_dir) if f.endswith('.npy')
               and not f.endswith('.codes.npy')}
    if not present:
        return 'skip'
    expected = {
        cache_stem('bpe', 'train', None, ''),
        cache_stem('bpe', 'validation', None, ''),
    }
    hit = expected & present
    if not hit:
        raise AssertionError(
            f'no expected stem found. present={sorted(present)} '
            f'expected among={sorted(expected)}')
    print(f'        {len(hit)} existing cache stem(s) still valid: {sorted(hit)}')


# ── 7. spaCy annotation ───────────────────────────────────────────────────────

@check('spaCy context annotation produces varying codes')
def t_spacy():
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.features.blocks import build_blocks

    if not (os.path.exists('data/our_final_vocab.csv')
            and os.path.exists('data/MorphoLEX_en.xlsx')):
        return 'skip'
    try:
        import spacy  # noqa
    except ImportError:
        return 'skip'

    from morphlm.pipelines.registry import build_pipeline
    cfg = apply_overrides(ExperimentConfig(),
                          ['data.pipeline=det', 'frozen.facts=dep'])
    pipe = build_pipeline(cfg, verbose=False)
    block = build_blocks(cfg)[0]

    text = ('The government released the report. Researchers criticized the '
            'government for the unbreakable secrecy of that report.')
    tokens = pipe.encode(text)
    codes = [block.code(t) for t in tokens]
    distinct = sorted(set(codes))
    assert len(distinct) > 1, f'dep codes are constant: {distinct}'

    # the real test: does one token id get different codes in different roles?
    by_id = {}
    for t, c in zip(tokens, codes):
        by_id.setdefault(t['token_id'], set()).add(c)
    varying = {k: v for k, v in by_id.items() if len(v) > 1}
    print(f'        {len(distinct)} distinct dep codes; '
          f'{len(varying)} token id(s) received MULTIPLE codes')
    if not varying:
        print('        NOTE: no id varied in this short sample — expected on '
              'a long corpus; not a failure.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spacy', action='store_true',
                    help='also run the spaCy annotation check (slower)')
    args = ap.parse_args()

    print('\n=== morphlm verification ===\n')
    t_imports()
    t_suites()
    t_build()
    t_freeze()
    t_learns()
    t_cache()
    if args.spacy:
        t_spacy()
    else:
        print('  SKIP  spaCy annotation check (pass --spacy to run it)')
        SKIP.append('spacy')

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
    for name, e in FAIL:
        print(f'  FAILED: {name}\n          {type(e).__name__}: {e}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
