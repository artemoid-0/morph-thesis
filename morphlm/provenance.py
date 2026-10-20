"""
provenance.py — everything about a run that cannot be recovered later.

The rule: if a number will appear in the thesis, the run that produced it
writes it down — parameter splits, corpus statistics, convergence — rather
than a table rebuilding it by hand, as tables once did (and once got wrong).
Three are unrecoverable once the run is over: library versions, the GPU
model, and wall-clock timing.
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
from datetime import datetime, timezone


#: the taggers the project uses; every one installed is recorded with its version
SPACY_MODELS = ('en_core_web_sm', 'uk_core_news_sm')


def environment(spacy_model=None):
    """
    Library and hardware versions. UNRECOVERABLE after the fact.

    `spacy_model` is the tagger the run or audit used — a run's
    `data.spacy_model` — recorded with its installed version; every project
    tagger installed is listed too (`spacy_models_installed`), so an audit that
    names none still says what could have tagged its data. Records written before
    7 October named `en_core_web_sm` whenever it was installed, whatever the run
    used: every Ukrainian run claims the English tagger (G-74; their config names
    the real one, `scripts/analysis/export_results.py` reads it from there).
    """
    env = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    try:
        import torch
        env['torch'] = torch.__version__
        env['cuda_available'] = bool(torch.cuda.is_available())
        env['cuda'] = getattr(torch.version, 'cuda', None)
        if torch.cuda.is_available():
            env['gpu'] = torch.cuda.get_device_name(0)
            env['gpu_count'] = torch.cuda.device_count()
    except Exception:
        pass
    for mod in ('numpy', 'pandas', 'spacy', 'tokenizers', 'datasets', 'pymorphy3'):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import spacy
        installed = {m: spacy.util.get_package_version(m) for m in SPACY_MODELS
                     if spacy.util.is_package(m)}
        if spacy_model and spacy_model not in installed and spacy.util.is_package(spacy_model):
            installed[spacy_model] = spacy.util.get_package_version(spacy_model)
        env['spacy_models_installed'] = ', '.join(f'{m} {v}' for m, v in installed.items()) or None
        if spacy_model:
            env['spacy_model'] = spacy_model
            env['spacy_model_version'] = installed.get(spacy_model)
    except Exception:
        pass
    env['git_commit'] = _git('rev-parse', 'HEAD')
    env['git_dirty'] = bool(_git('status', '--porcelain'))
    return env


def _git(*args):
    try:
        out = subprocess.run(['git', *args], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def parameter_breakdown(model, cfg):
    """
    Where the parameters actually are.

    Every "N% fewer parameters" claim in the project depends on this split, and
    it was recomputed by hand each time. Recording it makes the parameter tables
    a lookup rather than an arithmetic exercise.
    """
    emb = model.embed
    total = model.num_params()
    trained_emb = emb.trained.weight.numel()
    factor = emb.factor_parameters()
    head = sum(p.numel() for p in model.head.parameters())
    pos = model.pos.weight.numel() if model.pos is not None else 0   # none under RoPE
    aux = (sum(p.numel() for p in model.aux_head.parameters())
           if getattr(model, 'aux_head', None) is not None else 0)    # role=target
    core = total - trained_emb - factor - head - pos - aux
    return {
        'trainable_total': total,
        'embedding_trained': trained_emb,
        'embedding_factor': factor,
        'output_head': head,
        'positional': pos,
        'auxiliary_head': aux,
        # Both, under unambiguous names: the blocks and final norm alone
        # (793,344 at the standard configuration), and with the 256x128 learned
        # position table (826,112) — two documents once quoted one figure each
        # for the same quantity (docs/reference/apparatus.md)
        'transformer_core': core,
        'transformer_core_with_positional': core + pos,
        'vocab_size': model.vocab_size,
        'd_model': cfg.model.d_model,
        'frozen_width_T': emb.T,
        'trained_width': cfg.model.d_model - emb.T,
        'factors_trainable': emb.trainable_factors,
        'factor_scale': getattr(emb, 'factor_scale', 1.0),
    }


#: Bumped when the MEANING of a coarsened run changes without its config (so its
#: run_id): 1 folded at cache build and the model clamped, collapsing up to
#: 99.8% of words into one class; 2 folds at model time (journey §29.4).
COARSEN_SEMANTICS = 2

#: Bumped when the MEANING of a causal run changes without its config changing
#: (so with its run_id): the version is the only thing telling a valid causal run
#: from an invalid one. 1: only the identity blocks were shifted, `mstruct` left
#: leaking (journey §30.5). 2: every per-word fact, boundaries inferred from where
#: the codes change. 3: boundaries from the TOKENIZER — a v2 checkpoint re-scored
#: under v3 sees boundaries it never trained with (journey §30.8).
CAUSAL_SEMANTICS = 3


def frozen_detail(cfg):
    """Resolved fact blocks: cardinality is only known after the vocab loads."""
    from .features.blocks import build_blocks
    out = []
    for b in build_blocks(cfg):
        entry = {'name': b.name, 'source': b.source,
                 'n_classes': b.n_classes, 'dim': b.dim,
                 'is_projection': b.is_projection}
        if bool(getattr(getattr(cfg, 'frozen', None),
                        'causal_identity', False)):
            entry['causal_identity'] = True
            entry['causal_semantics'] = CAUSAL_SEMANTICS
            entry['boundary_source'] = 'tokenizer'
        coarsen = getattr(b, 'coarsen', 0)
        if coarsen:
            entry['coarsen'] = coarsen
            entry['n_full'] = getattr(b, 'n_full', None)
            entry['coarsen_semantics'] = COARSEN_SEMANTICS
        out.append(entry)
    return out


def data_stats(train_ds, val_ds):
    """Corpus statistics. Needed for every fertility / tokens-per-byte table."""
    def one(ds):
        if ds is None:
            return None
        return {
            'tokens': int(ds.num_tokens),
            'bytes': int(ds.num_bytes),
            'chars': int(ds.num_chars),
            'windows': int(len(ds)),
            'bytes_per_token': (ds.num_bytes / ds.num_tokens
                                if ds.num_tokens else None),
            # which codes file the split was read from, and the rules it
            # declares (G-72, G-73): what an audit otherwise reconstructs
            **({'codes_file': ds.codes_file} if getattr(ds, 'codes_file', None) else {}),
            **({'codes_rules': ds.codes_rules} if getattr(ds, 'codes_rules', None) else {}),
        }
    return {'train': one(train_ds), 'validation': one(val_ds)}


def convergence(log, max_steps, thresholds=(2.0, 1.8, 1.6, 1.5, 1.45, 1.40)):
    """
    Did it converge, and how fast did it get there?

    `steps_to` is the convergence-speed metric the literature reports (Hou et
    al. RQ2) and which this project computed ad hoc, after the fact, from raw
    logs. `still_improving` records whether the comparison is at convergence or
    at fixed compute — a limitation that must be stated and is easy to forget.
    """
    if not log:
        return {}
    by_step = sorted(log, key=lambda e: e['step'])
    bpb = [e['bits_per_byte'] for e in by_step]
    best_i = min(range(len(bpb)), key=lambda i: by_step[i]['mean_loss'])
    tail = [e for e in by_step if e['step'] >= max(0, max_steps - 5000)]
    delta_tail = (tail[-1]['bits_per_byte'] - tail[0]['bits_per_byte']
                  if len(tail) > 1 else None)
    return {
        'best_eval_index': best_i,
        'n_evals': len(by_step),
        'best_step': by_step[best_i]['step'],
        'final_step': by_step[-1]['step'],
        'best_bpb': min(bpb),
        'final_bpb': bpb[-1],
        'best_is_final': best_i == len(by_step) - 1,
        'early_stopped': by_step[-1]['step'] < max_steps - 1,
        'bpb_change_last_5k': delta_tail,
        'still_improving_at_ceiling': (delta_tail is not None
                                       and delta_tail < -1e-4),
        'steps_to': {str(t): next((e['step'] for e in by_step
                                   if e['bits_per_byte'] <= t), None)
                     for t in thresholds},
    }


def tokenizer_record(cfg):
    """
    The tokenizer a run read, from its sidecar: file, algorithm, sha256 and —
    for one that cuts endings (data/endings.py) — the cut rule and its
    fingerprint, which the stream's tokens depend on.
    """
    import os
    path = getattr(cfg.data, 'bpe_path', None)
    if getattr(cfg.data, 'pipeline', 'bpe') != 'bpe' or not path:
        return None
    from .data.tokenization import tokenizer_meta
    meta = tokenizer_meta(path)
    return {'file': os.path.basename(path),
            **{k: meta[k] for k in ('algorithm', 'vocab_size', 'sha256', 'presplit', 'presplit_fingerprint')
               if k in meta}}
