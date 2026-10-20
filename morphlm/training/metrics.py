"""
metrics.py — results and checkpoints.

Results land in results/<run_id>/ with config.json beside result.json, so the
seed and every other knob are part of the identity: keyed without the seed, a
second seed overwrote the first (bugs.md #6).
"""

from __future__ import annotations

import os
import json

RESULTS_DIR = 'results'


def save_result(cfg, log, best, results_dir=RESULTS_DIR, meta=None):
    run_dir = os.path.join(results_dir, cfg.run_id())
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    record = {
        'run_id': cfg.run_id(),
        'label': cfg.label(),
        'pipeline': cfg.data.pipeline,
        'facts': list(cfg.frozen.facts),
        'seed': cfg.train.seed,
        'best_step': best['step'],
        'val_loss': best['mean_loss'],
        'perplexity': best['perplexity'],
        'bits_per_byte': best['bits_per_byte'],
        'val_tokens': best['tokens'],
        'lr_horizon': cfg.train.lr_horizon,
        'max_steps': cfg.train.max_steps,
        'log': log,
    }
    # provenance: parameter split, corpus statistics, environment, timing,
    # convergence and the per-frequency breakdown. Three of these cannot be
    # recovered once the run is over.
    if meta:
        record.update(meta)
    path = os.path.join(run_dir, 'result.json')
    with open(path, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    return path


def load_results(results_dir=RESULTS_DIR):
    out = []
    if not os.path.isdir(results_dir):
        return out
    for entry in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, entry, 'result.json')
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


#: Bumped whenever the module/buffer layout of the model changes in a way that
#: makes older checkpoints unloadable. `migrate_state_dict` must gain a branch
#: for every bump, or previously-trained cells become unanalysable.
CHECKPOINT_FORMAT = 2


def migrate_state_dict(sd, verbose=True):
    """
    Bring a legacy checkpoint's state_dict up to the current layout.

    FORMAT 1 -> 2, introduced by the frozen-vs-learned rewrite:

      `embed.static_table`   a [vocab_size, dim] one-hot FLOAT table, registered
                             UNCONDITIONALLY — including as `zeros(vocab, 0)`
                             when no static fact was requested. Replaced by
                             per-block `embed.static_codes_{i}` INT buffers so
                             frozen and learned modes could share one indexing
                             path. Every run in this project used CONTEXT facts
                             only, so the stored table is always zero-width and
                             carries no information: dropped.

      `embed.ctx_eye_{i}`    renamed to `embed.factor_eye_{i}`. Pure rename; the
                             tensors are identity matrices either way.

    Anything not covered here is left alone so that `load_state_dict(strict=True)`
    still fails loudly. A migration that silently swallows unknown keys would
    turn a schema error into a wrong result.
    """
    import re
    out = {}
    dropped, renamed = [], []
    for k, v in sd.items():
        if k == 'embed.static_table':
            if getattr(v, 'ndim', 0) == 2 and v.shape[1] == 0:
                dropped.append(k)
                continue
            raise RuntimeError(
                f'checkpoint has a NON-EMPTY {k} (shape {tuple(v.shape)}). '
                f'That means it used a STATIC fact, whose one-hot table cannot '
                f'be mapped onto the current integer-code layout. Retrain this '
                f'cell rather than migrating it.')
        m = re.match(r'^embed\.ctx_eye_(\d+)$', k)
        if m:
            nk = f'embed.factor_eye_{m.group(1)}'
            renamed.append((k, nk))
            out[nk] = v
            continue
        out[k] = v
    if verbose and (dropped or renamed):
        print(f'  [checkpoint migrated] dropped {dropped}, '
              f'renamed {[a for a, _ in renamed]}')
    return out


def save_checkpoint(cfg, model, results_dir=RESULTS_DIR):
    """
    Persist model weights alongside the result.

    Without this, every post-hoc analysis — per-frequency-bucket perplexity,
    probing classifiers, anything that needs the trained embedding — requires
    retraining the cell from scratch. The first three baseline runs were saved
    without weights, so those 2.5 GPU-hours cannot be re-analysed.
    """
    import torch
    run_dir = os.path.join(results_dir, cfg.run_id())
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, 'model.pt')
    torch.save({'state_dict': model.state_dict(),
                'config': cfg.to_dict(),
                'run_id': cfg.run_id(),
                'format': CHECKPOINT_FORMAT}, path)
    return path


def save_step_checkpoint(cfg, model, step, results_dir=RESULTS_DIR):
    """
    A checkpoint DURING training, beside model.pt: ckpt/step_NNNNNN.pt (journey
    §33.48). Taken only when MORPHLM_CHECKPOINT_EVERY is set — an environment
    switch, not a config field, so a checkpointed run keeps its run id and label
    and joins its cell as an ordinary seed. The probe reads these.
    """
    import torch
    ckpt_dir = os.path.join(results_dir, cfg.run_id(), 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'step_{step:06d}.pt')
    torch.save({'state_dict': model.state_dict(), 'config': cfg.to_dict(),
                'run_id': cfg.run_id(), 'step': step,
                'format': CHECKPOINT_FORMAT}, path)
    return path


def load_checkpoint(run_id, results_dir=RESULTS_DIR, filename='model.pt'):
    """Rebuild a model from a saved checkpoint (model.pt, or ckpt/step_*.pt)."""
    import torch
    from ..config import ExperimentConfig
    from ..modeling.transformer import build_model
    path = os.path.join(results_dir, run_id, filename)
    blob = torch.load(path, map_location='cpu')
    # all five sections: rebuilt without `surface`, a det surface ablation came
    # back as the all-on cell, with its label and its stream
    cfg = ExperimentConfig.from_dict(blob['config'])
    model, _ = build_model(cfg, verbose=False)
    fmt = blob.get('format', 1)
    sd = blob['state_dict']
    if fmt < CHECKPOINT_FORMAT:
        sd = migrate_state_dict(sd)
    model.load_state_dict(sd)          # strict: unknown keys still raise
    return model, cfg
