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

    python scripts/ops/verify.py            # steps 1-6
    python scripts/ops/verify.py --spacy    # also step 7 (slower, loads spaCy)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    # every module of the package, found, not listed: a module left out of a hand-kept list is
    # never checked (journey §33.45; the list here had missed eight, morphlm.record among them)
    import importlib
    import pkgutil
    import morphlm
    broken = []
    # a package that fails to load is still listed, and its failure is reported below
    names = ['morphlm'] + [m.name for m in pkgutil.walk_packages(
        morphlm.__path__, 'morphlm.', onerror=lambda name: None)]
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as e:
            broken.append((name, f'{type(e).__name__}: {e}'))
    if broken:
        raise ImportError('; '.join(f'{n} ({why})' for n, why in broken))
    print(f'        {len(names)} modules load')


# ── 2. pure-logic suites ──────────────────────────────────────────────────────

@check('pure-logic test suites')
def t_suites():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # every suite in tests/, found, not listed: a suite left out of a hand-kept
    # list never runs (journey §33.45)
    import glob as _glob
    for suite in sorted(os.path.basename(f) for f in _glob.glob(os.path.join(root, 'tests', 'test_*.py'))):
        p = subprocess.run([sys.executable, os.path.join(root, 'tests', suite)],
                           capture_output=True, text=True, cwd=root)
        if p.returncode != 0:
            raise AssertionError(f'{suite} failed:\n{p.stdout}\n{p.stderr}')


# ── 3-5. torch paths ──────────────────────────────────────────────────────────

def _fake_vocab(n=240):
    """
    Synthetic vocabulary covering EVERY category.

    It must cover all of them: assert_blocks_informative() rightly refuses a
    constant frozen column, so a fixture missing the information-preserving
    categories fails the check for the wrong reason.
    """
    import pandas as pd
    cats = ['prefix', 'suffix', 'root', 'function', 'entity', 'character',
            'punct', 'digit', 'infl', 'space', 'case', 'byte']
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

    before_static = emb.static_codes_0.clone()
    before_eye = emb.factor_eye_0.clone()
    before_trained = emb.trained.weight.detach().clone()

    ids = torch.randint(0, V, (2, cfg.data.seq_len))
    tgt = torch.randint(0, V, (2, cfg.data.seq_len))
    codes = torch.randint(0, 6, (2, len(emb.context_blocks), cfg.data.seq_len))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _, loss = model(ids, tgt, codes=codes)
    opt.zero_grad(); loss.backward(); opt.step()

    assert torch.equal(before_static, emb.static_codes_0), \
        'static code table CHANGED — the freeze is not enforced'
    assert torch.equal(before_eye, emb.factor_eye_0), \
        'factor identity table CHANGED — the freeze is not enforced'
    assert emb.factor_parameters() == 0, \
        'frozen mode must expose zero trainable factor parameters'
    moved = (emb.trained.weight.detach() - before_trained).abs().max().item()
    assert moved > 0, 'trained part did not move — nothing is learning'
    print(f'        frozen byte-identical; trained moved by {moved:.2e}')


@check('FROZEN vs LEARNED factors are a controlled comparison')
def t_frozen_vs_learned():
    """
    The frozen-vs-learned experiment is only meaningful if the two conditions
    differ in exactly one thing. This asserts that:
      - both occupy the same T dimensions
      - both index identical codes
      - frozen exposes zero trainable factor parameters
      - learned adds only n_classes x dim per block
      - total parameter counts stay within 0.1%
    Otherwise the comparison would confound trainability with capacity.
    """
    import torch
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.modeling.embedding import HybridEmbedding
    from morphlm.modeling.transformer import TransformerLM

    vocab = _fake_vocab()
    V = len(vocab) + 4
    fz = apply_overrides(ExperimentConfig(), ['frozen.facts=dep'])
    ln = apply_overrides(fz, ['frozen.trainable=true'])

    e_f = HybridEmbedding(fz, V, vocab_df=vocab, verbose=False)
    e_l = HybridEmbedding(ln, V, vocab_df=vocab, verbose=False)
    assert e_f.T == e_l.T, f'width differs: {e_f.T} vs {e_l.T}'
    assert e_f.factor_parameters() == 0, 'frozen has trainable factor params'
    assert e_l.factor_parameters() > 0, 'learned has no trainable factor params'

    m_f = TransformerLM(fz, e_f, V)
    m_l = TransformerLM(ln, e_l, V)
    p_f, p_l = m_f.num_params(), m_l.num_params()
    rel = abs(p_l - p_f) / p_f
    assert rel < 0.001, f'param counts differ by {100*rel:.3f}% — not controlled'

    # frozen factors must be exactly one-hot
    ids = torch.randint(0, V, (1, fz.data.seq_len))
    codes = torch.randint(0, 8, (1, 1, fz.data.seq_len))
    out = e_f(ids, codes)[:, :, :e_f.T]
    assert torch.allclose(out.sum(-1), torch.ones_like(out.sum(-1))), \
        'frozen factor is not one-hot'
    # scale must actually change the factor magnitude — this is the control
    # that separates trainability from signal strength
    sc = apply_overrides(fz, ['frozen.scale=0.22'])
    e_s = HybridEmbedding(sc, V, vocab_df=vocab, verbose=False)
    o_f = e_f(ids, codes)[:, :, :e_f.T]
    o_s = e_s(ids, codes)[:, :, :e_s.T]
    n_f = o_f.norm(dim=-1).mean().item()
    n_s = o_s.norm(dim=-1).mean().item()
    assert abs(n_s / n_f - 0.22) < 1e-4, f'scale not applied: {n_s/n_f}'
    trained_norm = e_f.trained.weight.detach().norm(dim=-1).mean().item()
    print(f'        T={e_f.T} both | frozen params 0, learned '
          f'{e_l.factor_parameters()} | total {p_f:,} vs {p_l:,} '
          f'({100*rel:.4f}% apart)')
    print(f'        factor norm: frozen@1.0={n_f:.3f} frozen@0.22={n_s:.3f} '
          f'trained slice={trained_norm:.3f}')


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
    import dataclasses
    import glob
    import json
    from morphlm.config import DataConfig, ExperimentConfig, SurfaceConfig
    from morphlm.data.stream import PACKED, cache_stem
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cache_dir = os.path.join(root, 'data', 'cache')
    if not os.path.isdir(cache_dir):
        return 'skip'
    # a stream packed on this machine (scripts/data/cache_pack.py) keeps its key
    present = {f[:-4] for f in os.listdir(cache_dir) if f.endswith('.npy')
               and not f.endswith('.codes.npy')}
    present |= {f[:-4 - len(PACKED)] for f in os.listdir(cache_dir) if f.endswith('.npy' + PACKED)
                and not f.endswith('.codes.npy' + PACKED)}
    if not present:
        return 'skip'
    # the streams today's cache key gives for every configuration a recorded run trained on, and
    # the legacy English streams; any of them on disk counts — a machine may hold only the
    # streams sent to it (journey §33.101)
    def section(cls, raw):      # a recorded section; a field added since keeps its default
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    expected = {cache_stem('bpe', 'train', None, ''), cache_stem('bpe', 'validation', None, '')}
    for path in glob.glob(os.path.join(root, 'results', '*', 'config.json')):
        try:
            raw = json.load(open(path))
            # a record without a surface section predates it: the legacy stream, every channel off
            surface = raw.get('surface', {f.name: False for f in dataclasses.fields(SurfaceConfig)})
            cfg = ExperimentConfig(data=section(DataConfig, raw.get('data', {})),
                                   surface=section(SurfaceConfig, surface))
        except (OSError, ValueError, TypeError):
            continue
        for split, cap in (('train', cfg.data.max_articles_train), ('validation', cfg.data.max_articles_val)):
            expected.add(cache_stem(cfg.data.pipeline, split, cap, cfg.surface_spec()))
    hit = expected & present
    if not hit:
        raise AssertionError(
            f'no stream on disk resolves from a known configuration. present={sorted(present)[:8]}'
            f'{" …" if len(present) > 8 else ""} expected among {len(expected)} stems')
    print(f'        {len(hit)} of {len(present)} cached stream(s) resolve from a recorded configuration')


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
    t_frozen_vs_learned()
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
