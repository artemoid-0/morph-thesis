"""
runs.py — the run records, read one way.

Every analysis script reads `results/<run>/result.json` and compares runs paired by
seed. This module is the one place that does it:

    records, load        every run record under the results directories, keyed by
                         (label, seed); `machine='second'` reads the second machine's
                         records, its twin copies included (results/audits/device_*/)
    load_by_key          keyed by (label, seed, steps), each machine's records apart:
                         a key with a record on each machine is a twin pair
    select, by_id        runs by label pattern (valid runs only), or by run id
    machine, pair        the machine a run ran on, and runs paired by seed within one
                         machine (FINDINGS, method point 11): across the two machines a
                         pair differs by dropout's draw as well as by the cells
                         (journey §33.109, §33.113), so such seeds are left out and named
    curve, gain_curve    a run's validation BPB along training; one cell's gain over
                         another along training, paired by seed
    at                   a value interpolated where a curve passes a level
    summary, fmt, agree  mean ± standard error over seeds, as the project reports it
    own_baseline         the feature-free cell a cell is compared with (journey §33.6)
    rescore              a saved model's loss on each validation target, again

Callers keep their own conventions — the standard error at one seed (None, 0.0 or NaN),
"positive" or "agree", a curve cut at the best step or not — and pass them in.

    from morphlm import runs
    R = runs.load()
    pairs, left_out = runs.pair(R, 'bpe_mconst-LEARN-CAUSAL_bpe_17k', 'bpe_pos-LEARN-CAUSAL_bpe_17k')
"""

from __future__ import annotations

import os
import glob
import json
import math
import fnmatch
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')
AUDITS = os.path.join(RESULTS, 'audits')
TWINS = os.path.join('audits', 'device_*', '*.json')      # under a results directory
TWIN_MODELS = 'device_twins_models'                         # beside the twin copies


# ── where records are ─────────────────────────────────────────────────────────

def add_results_flag(ap, root=False):
    """The one flag that says where run records are: --results DIR [DIR ...], default the
    project's results/. The older spellings --roots and --results-dir stay; with root=True,
    --root PROJECT too, read as PROJECT/results. Read back with `results_of`."""
    ap.add_argument('--results', '--roots', '--results-dir', dest='results', nargs='+', default=[RESULTS],
                    metavar='DIR', help='results directories to read run records from (default: the '
                                        "project's results/; --roots and --results-dir are older spellings)")
    if root:
        ap.add_argument('--root', default=None,
                        help='a project root: its results/ is read (an older spelling of --results ROOT/results)')
    return ap


def results_of(a):
    """The results directories a parsed command line names."""
    if getattr(a, 'root', None):
        return [os.path.join(a.root, 'results')]
    return list(a.results)


def _roots(roots):
    if roots is None:
        return [RESULTS]
    return [roots] if isinstance(roots, str) else list(roots)


def _read(path):
    try:
        with open(path) as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None
    return r if isinstance(r, dict) and 'label' in r else None


def records(roots=None, twins=False):
    """(path, record) for every run record under `roots` (default: the project's results/):
    each root's `**/result.json` in sorted order, roots in the order given; with twins=True,
    then each root's twin copies, `audits/device_*/*.json` — the second machine's repeats of
    thesis runs and its calibration runs, kept apart from results/ (journey §33.101). Unreadable
    files and JSON that is not a run record are skipped. Each record carries its path as
    '_path'."""
    roots = _roots(roots)
    groups = [(os.path.join('**', 'result.json'), True)] + ([(TWINS, False)] if twins else [])
    for pattern, recursive in groups:
        for root in roots:
            for p in sorted(glob.glob(os.path.join(root, pattern), recursive=recursive)):
                r = _read(p)
                if r is not None:
                    r['_path'] = p
                    yield p, r


def load(roots=None, twins=False, machine=None, keep=None):
    """{(label, seed): record}. The first record of a key wins: roots in the order given, each
    root's paths sorted. With twins=True the twin copies fill only keys no result.json holds.
    machine='thesis' or 'second' keeps that machine's records only; 'second' reads the twin
    copies as well — the second machine's records, wherever they are. `keep`, a predicate,
    drops records before they are keyed."""
    twins = twins or machine == 'second'
    R = {}
    for p, r in records(roots, twins=twins):
        if machine is not None and _machine(r) != machine:
            continue
        if keep is not None and not keep(r):
            continue
        R.setdefault((r['label'], r['seed']), r)
    return R


def load_by_key(roots=None):
    """{(label, seed, steps): {machine: [(path, record), ...]}} — every record with a log, the
    twin copies included, in sorted path order, each machine's apart; a record whose GPU is
    neither machine's is left out. A key with a record on each machine is a twin pair
    (scripts/analysis/device_twins.py)."""
    out = defaultdict(lambda: defaultdict(list))
    for p, r in sorted(records(roots, twins=True), key=lambda pr: pr[0]):
        if 'log' not in r or 'bits_per_byte' not in r:
            continue
        m = _machine(r)
        if m is None:
            continue
        out[(r['label'], r['seed'], r.get('max_steps'))][m].append((p, r))
    return out


def model_dir(r):
    """The directory holding a run's config.json, model.pt and checkpoints: its record's own,
    or for a twin copy results/audits/device_twins_models/<run id>/."""
    p = r['_path']
    if os.path.basename(p) == 'result.json':
        return os.path.dirname(p)
    return os.path.join(os.path.dirname(os.path.dirname(p)), TWIN_MODELS, r['run_id'])


def valid(r):
    """A causal run must carry causal semantics >= 3 (the rule in scripts/ops/quarantine.py): an
    older causal checkpoint evaluated under current code sees boundaries it never trained with.
    result.json holds no config; causality shows in the label and in the per-block record, and
    either is enough to demand it."""
    lab = r.get('label', '')
    blocks = r.get('frozen_blocks') or []
    causal = 'CAUSAL' in lab or any(b.get('causal_identity') for b in blocks)
    return not causal or any(b.get('causal_identity') and (b.get('causal_semantics') or 0) >= 3 for b in blocks)


def select(patterns, roots=None, machine=None):
    """The valid runs whose label matches one of the glob `patterns`, as records sorted by label,
    seed and run id; an invalid one is named and left out. machine='second' selects among the
    second machine's records, its twin copies included; a run id is taken once."""
    out, seen = [], set()
    for p, r in records(roots, twins=machine == 'second'):
        if machine is not None and _machine(r) != machine:
            continue
        lab = r.get('label', '')
        if not any(fnmatch.fnmatch(lab, pat) for pat in patterns):
            continue
        if not valid(r):
            print(f'  excluded (causal semantics < 3): {r.get("run_id")} {lab}')
            continue
        if r.get('run_id') in seen:
            continue
        seen.add(r.get('run_id'))
        out.append(r)
    return sorted(out, key=lambda r: (r['label'], r.get('seed', 0), r.get('run_id', '')))


def by_id(roots=None, twins=False, machine=None):
    """{run id: record}, the first record of an id winning as in `load`. A twin shares its id with
    the run it repeats, so the second machine's record of an id is asked for by machine='second'."""
    twins = twins or machine == 'second'
    out = {}
    for p, r in records(roots, twins=twins):
        if 'run_id' in r and (machine is None or _machine(r) == machine):
            out.setdefault(r['run_id'], r)
    return out


# ── machines and pairs ────────────────────────────────────────────────────────

def gpu(r):
    return (r.get('environment') or {}).get('gpu') or ''


def machine(r):
    """'second' (RTX 5090 Laptop), 'thesis' (RTX 4060 Laptop, or no GPU named: the thesis
    machine's records from before records named it), or None for any other GPU."""
    g = gpu(r)
    if '5090' in g:
        return 'second'
    if '4060' in g or not g:
        return 'thesis'
    return None


_machine = machine          # for functions whose argument is called `machine`


def _where(r):
    """The machine, or for a GPU neither machine has, that GPU."""
    return machine(r) or gpu(r)


def pair(R, a, b, *more, same_machine=True):
    """The seeds at which cells a, b (and more) all have a run in R = {(label, seed): record},
    paired within one machine: ({seed: (a's record, b's, ...)}, {seed: (a's machine, b's, ...)}),
    the second the seeds left out because their runs ran on two machines."""
    cells = (a, b) + more
    have = None
    for c in cells:
        s = {s for (l, s) in R if l == c}
        have = s if have is None else have & s
    pairs, left = {}, {}
    for s in sorted(have):
        rs = tuple(R[(c, s)] for c in cells)
        if same_machine and len({_where(r) for r in rs}) > 1:
            left[s] = tuple(_where(r) for r in rs)
        else:
            pairs[s] = rs
    return pairs, left


def left_out(left):
    """A line naming the seeds `pair` left out; '' when it left none."""
    if not left:
        return ''
    return 'left out, runs on two machines: ' + ', '.join(
        f'seed {s} ({" / ".join(map(str, m))})' for s, m in sorted(left.items()))


# ── curves ────────────────────────────────────────────────────────────────────

def curve(r):
    """{step: validation bits per byte} along training, the step-0 evaluation left out."""
    return {e['step']: e['bits_per_byte'] for e in r.get('log', []) if e.get('step', 0) > 0}


def _best(r):
    return r.get('best_step') or math.inf


def gain_curve(R, a, b, to_best=True):
    """b's gain over a along training, paired by seed within one machine: (points, seeds, left out),
    the points [(step, a's BPB, the gain, per_seed)] with a's BPB and the gain (a's BPB − b's)
    averaged over the seeds, and per_seed [(a's BPB, gain)] in seed order. A step is kept when
    every paired seed has it in both runs and, with to_best, when it is no later than either run's
    best step (before either over-fits)."""
    pairs, left = pair(R, a, b)
    seeds = sorted(pairs)
    pts = {}
    for s in seeds:
        ra, rb = pairs[s]
        ca, cb = curve(ra), curve(rb)
        stop = min(_best(ra), _best(rb)) if to_best else math.inf
        for x in ca:
            if x in cb and x <= stop:
                pts.setdefault(x, []).append((ca[x], ca[x] - cb[x]))
    return [(x, st.mean(p for p, _ in v), st.mean(g for _, g in v), v)
            for x, v in sorted(pts.items()) if len(v) == len(seeds)], seeds, left


def at(points, level):
    """The second value of `points` [(x, y, ...)] where x passes `level` between two consecutive
    points, interpolated linearly: the first such crossing, or None if x never passes it."""
    for p0, p1 in zip(points, points[1:]):
        x0, y0, x1, y1 = p0[0], p0[1], p1[0], p1[1]
        if x0 >= level >= x1 and x0 != x1:
            return y0 + (y1 - y0) * (x0 - level) / (x0 - x1)
    return None


# ── over seeds ────────────────────────────────────────────────────────────────

def summary(xs, se_one=None):
    """{'mean', 'se', 'n', 'positive', 'values'} over the values that are not None (None if no
    value is): the standard error is the seeds' standard deviation over √n, and `se_one` at a
    single value — the callers' conventions differ (None, 0.0 or NaN)."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {'mean': st.mean(xs), 'se': (st.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else se_one,
            'n': len(xs), 'positive': sum(x > 0 for x in xs), 'values': xs}


def agree(xs):
    """How many values share the commoner sign (zeros count for neither)."""
    return max(sum(x > 0 for x in xs), sum(x < 0 for x in xs))


def fmt(s, nd=4, scale=1.0):
    """'mean ± se (positive/n)' for a summary, '—' for None; without '± se' when se is None."""
    if s is None:
        return '—'
    out = f"{s['mean'] * scale:+.{nd}f}"
    if s['se'] is not None:
        out += f" ± {s['se'] * scale:.{nd}f}"
    return out + f" ({s['positive']}/{s['n']})"


def own_baseline(label):
    """The baseline a cell is compared with: its label with the feature part replaced by
    'baseline'. Non-default training settings are part of every label, so this is the run
    identical to the cell except for its features (journey §33.6, G-27):
        bpe_mstruct-LEARN-CAUSAL_bpe_17k      -> bpe_baseline_bpe_17k
        bpe_mstruct_bpe_17k_st10k             -> bpe_baseline_bpe_17k_st10k
    """
    parts = label.split('_', 2)
    return '_'.join([parts[0], 'baseline'] + parts[2:])


# ── a saved model, scored again ───────────────────────────────────────────────

def rescore(run_id, results_dir=RESULTS, batches=0):
    """A saved model's loss on each validation target, in nats (float32), as training evaluated
    it: its own context facts, the same windows, `batches` batches of its batch size (0: all).
    Returns (config, pipeline, dataset, losses, inputs, targets), the last three flat."""
    import dataclasses
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from .training.metrics import load_checkpoint
    from .training.loop import get_device
    from .pipelines.registry import build_pipeline
    from .data.windows import make_val_dataset
    from .registry import context_facts

    model, cfg = load_checkpoint(run_id, results_dir)
    device = get_device()
    model = model.to(device).eval()
    facts = list(context_facts(cfg.frozen.facts))
    data_cfg = dataclasses.replace(cfg, frozen=dataclasses.replace(cfg.frozen, facts=tuple(facts),
                                                                   shuffle_codes=False))
    pipe = build_pipeline(data_cfg, verbose=False)
    ds = make_val_dataset(pipe, data_cfg, verbose=False)
    loaded = list(getattr(ds, 'fact_names', []) or [])
    if facts and loaded != facts:
        raise RuntimeError(f'{run_id}: the model reads {facts}, the loader gave {loaded} — refusing to score')
    nll, xs, ys = [], [], []
    with torch.no_grad():
        for bi, (x, y, c) in enumerate(DataLoader(ds, batch_size=cfg.train.batch_size)):
            if batches and bi >= batches:
                break
            cc = c.to(device) if c.numel() else None
            logits, _ = model(x.to(device), None, codes=cc)
            nll.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1),
                                       reduction='none').cpu().numpy())
            xs.append(x.reshape(-1).numpy())
            ys.append(y.reshape(-1).numpy())
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return cfg, pipe, ds, np.concatenate(nll), np.concatenate(xs), np.concatenate(ys)
