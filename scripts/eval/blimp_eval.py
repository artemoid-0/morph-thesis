#!/usr/bin/env python3
"""
blimp_eval.py — BLiMP minimal pairs on existing checkpoints, paired by seed.

WHY
---
BLiMP (Warstadt et al. 2020) is the standard grammatical evaluation for small
language models — the BabyLM challenge uses it at this scale — and it tests
what morphology governs: agreement, irregular forms. Each item is two
sentences differing minimally; a model is right when it gives the grammatical
one the higher probability. No training: checkpoints are scored as they are.

WHAT IT REPORTS
---------------
For every checkpoint and paradigm: ACCURACY, and the mean log-probability
MARGIN (nats by which the model prefers the good sentence). The margin is
continuous and far less noisy than accuracy on 1,000 binary items.

Checkpoints are grouped by cell (label) and PAIRED BY SEED against a reference
cell (the first run's label by default): for each seed, cell minus reference;
then the mean over seeds, its standard error, and how many seeds agree in
sign — the reporting the project uses for every small effect.
A single seed is shown but never summarised as an effect.

Results are aggregated by BLiMP's own annotation, `field` (morphology, syntax,
semantics, syntax-semantics) and `linguistics_term`. Per-item margins are saved
beside the audit record, so later analyses need no re-scoring.

HOW IT SCORES
-------------
Each sentence is preceded by a '.' token — what the model saw before every
paragraph in training, where lines were concatenated — so the FIRST word is
scored too. (An earlier version dropped it, losing the difference in pairs
that differ in their first word.)

Sentences are encoded ONCE per feature configuration and shared by every seed
of it, then scored in length-sorted batches. Batching is checked at startup:
the first sentences are scored alone and together, and the run stops unless
the two agree within 1e-3 nats.

Context features come from the model's own pipeline and are shifted by the
model's own forward pass. Derived features (`mfeatlex`) come through the
block's lexicon, which reproduces the derived cache exactly.

    python scripts/eval/blimp_eval.py --runs RID [RID ...]            # 18 morphology paradigms
    python scripts/eval/blimp_eval.py --runs RID [RID ...] --all      # all 67
    python scripts/eval/blimp_eval.py --labels 'bpe_baseline_bpe_17k' '*mfeatlex*' --all
"""

import os
import sys
import json
import time
import argparse
import dataclasses
import statistics as st
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MORPHOLOGY = (
    'anaphor_gender_agreement', 'anaphor_number_agreement',
    'determiner_noun_agreement_1', 'determiner_noun_agreement_2',
    'determiner_noun_agreement_irregular_1',
    'determiner_noun_agreement_irregular_2',
    'determiner_noun_agreement_with_adjective_1',   # sic: the release is inconsistent
    'determiner_noun_agreement_with_adj_2',
    'determiner_noun_agreement_with_adj_irregular_1',
    'determiner_noun_agreement_with_adj_irregular_2',
    'irregular_past_participle_adjectives',
    'irregular_past_participle_verbs',
    'distractor_agreement_relational_noun',
    'distractor_agreement_relative_clause',
    'irregular_plural_subject_verb_agreement_1',
    'irregular_plural_subject_verb_agreement_2',
    'regular_plural_subject_verb_agreement_1',
    'regular_plural_subject_verb_agreement_2',
)


def load_pairs(paradigms, local):
    out = {}
    if local:
        if paradigms == 'all':
            paradigms = sorted(f[:-len('.jsonl')] for f in os.listdir(local)
                               if f.endswith('.jsonl'))
        for p in paradigms:
            path = os.path.join(local, p + '.jsonl')
            out[p] = [json.loads(l) for l in open(path) if l.strip()]
        return out
    from datasets import load_dataset, get_dataset_config_names
    available = get_dataset_config_names('nyu-mll/blimp')
    if paradigms == 'all':
        paradigms = available
    missing = [p for p in paradigms if p not in available]
    if missing:
        import difflib
        raise SystemExit('unknown BLiMP paradigm(s): ' + ', '.join(
            f'{m} (did you mean {(difflib.get_close_matches(m, available, 1) or ["?"])[0]}?)'
            for m in missing))
    for p in paradigms:
        out[p] = [dict(r) for r in load_dataset('nyu-mll/blimp', p, split='train')]
    return out


def annotation(paradigm, rows):
    r = rows[0] if rows else {}
    field = r.get('field') or ('morphology' if any(
        k in paradigm for k in ('agreement', 'irregular')) else 'other')
    term = r.get('linguistics_term') or paradigm.rsplit('_', 1)[0]
    return str(field).replace('_', '/'), str(term)


class Encoder:
    """Encodes sentences once per feature configuration."""

    def __init__(self):
        self.cache = {}

    def encode_all(self, cfg, facts, sentences):
        # Everything a checkpoint's config changes beyond this key acts inside
        # the model (tested in test_cache_keys), so checkpoints differing only
        # in seed, halves, coarsening or the causal flag share one encoding.
        k = (cfg.data.pipeline, cfg.data.bpe_path, tuple(facts),
             getattr(cfg.frozen, 'dep_granularity', 'coarse'))
        if k in self.cache:
            return self.cache[k]
        from morphlm.pipelines.registry import build_pipeline
        from morphlm.data.stream import cache_blocks
        data_cfg = dataclasses.replace(
            cfg, frozen=dataclasses.replace(cfg.frozen, facts=tuple(facts),
                                            shuffle_codes=False))
        pipe = build_pipeline(data_cfg, verbose=False)
        # codes exactly as the training cache stores them: the model applies
        # its own transformations (coarsening, halves, the causal shift) on top
        blocks = cache_blocks(data_cfg, facts)
        if any(b.source != 'context' for b in blocks):
            raise RuntimeError(f'non-context fact among {facts}')
        dot = pipe.tokenizer.token_to_id('.')
        if dot is None:
            raise RuntimeError("the tokenizer has no '.' token to prefix with")
        t0 = time.time()
        seqs, codes = [], []
        for s in sentences:
            if blocks:
                toks = pipe.encode(s)
                ids = [t['token_id'] for t in toks]
                cs = []
                for b in blocks:
                    c = list(b.codes_for(ids, toks)) if hasattr(b, 'codes_for') \
                        else [b.code(t) for t in toks]
                    cs.append([b.n_classes - 1] + c)       # the prefix is no word
                codes.append(cs)
            else:
                ids = pipe.encode_ids(s)
                codes.append(None)
            seqs.append([dot] + list(ids))
        nones = [b.n_classes - 1 for b in blocks]
        print(f'    encoded {len(sentences):,} sentences for '
              f'{list(facts) or "no facts"} in {time.time() - t0:.0f}s')
        self.cache[k] = (seqs, codes, nones)
        return self.cache[k]


def score(model, device, seqs, codes, nones, idx, batch=256):
    """Total NLL (nats) of each sequence in `idx`, excluding the prefix token."""
    import torch
    import torch.nn.functional as F
    out = {}
    order = sorted(idx, key=lambda i: len(seqs[i]))
    nb = len(nones)
    with torch.no_grad():
        for a in range(0, len(order), batch):
            chunk = order[a:a + batch]
            L = max(len(seqs[i]) for i in chunk) - 1
            x = torch.zeros((len(chunk), L), dtype=torch.long)
            y = torch.full((len(chunk), L), -100, dtype=torch.long)
            c = None
            if nb:
                c = torch.tensor(nones, dtype=torch.long).view(1, nb, 1).repeat(len(chunk), 1, L)
            for j, i in enumerate(chunk):
                s = seqs[i]
                n = len(s) - 1
                x[j, :n] = torch.tensor(s[:-1])
                y[j, :n] = torch.tensor(s[1:])
                if nb:
                    c[j, :, :n] = torch.tensor([cc[:-1] for cc in codes[i]])
            logits, _ = model(x.to(device), None,
                              codes=c.to(device) if nb else None)
            lp = F.log_softmax(logits.float(), dim=-1)
            yy = y.to(device)
            tok = -lp.gather(-1, yy.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            tok = tok * (yy != -100)
            sums = tok.sum(-1).tolist()
            for j, i in enumerate(chunk):
                out[i] = sums[j]
    return out


def self_check(model, device, seqs, codes, nones, k=24):
    """Batched scores must equal one-at-a-time scores, or nothing is written."""
    idx = list(range(min(k, len(seqs))))
    alone = {}
    for i in idx:
        alone.update(score(model, device, seqs, codes, nones, [i], batch=1))
    together = score(model, device, seqs, codes, nones, idx, batch=len(idx))
    worst = max(abs(alone[i] - together[i]) for i in idx)
    if worst > 1e-3:
        raise SystemExit(f'batched scoring disagrees with one-at-a-time by '
                         f'{worst:.2e} nats — padding or masking is wrong; '
                         f'no results written')
    return worst


def select(patterns, results_dir):
    """
    Run ids whose label matches a glob pattern, VALID runs only: a causal run
    must carry causal semantics >= 3 (the rule in quarantine.py), because an
    older causal checkpoint evaluated under current code sees boundaries it
    never trained with. Sorted by label then seed, for a stable reading order.
    """
    import fnmatch
    import glob as _glob
    out = []
    for p in _glob.glob(os.path.join(results_dir, '*', 'result.json')):
        try:
            r = json.load(open(p))
        except (OSError, ValueError):
            continue
        lab = r.get('label', '')
        if not any(fnmatch.fnmatch(lab, pat) for pat in patterns):
            continue
        blocks = r.get('frozen_blocks') or []
        # result.json holds no config; causality is visible in the label and in
        # the per-block record, and either is enough to demand semantics >= 3
        causal = 'CAUSAL' in lab or any(b.get('causal_identity') for b in blocks)
        if causal and not any(b.get('causal_identity') and (b.get('causal_semantics') or 0) >= 3
                              for b in blocks):
            print(f'  excluded (causal semantics < 3): {os.path.basename(os.path.dirname(p))} {lab}')
            continue
        out.append((lab, r.get('seed', 0), os.path.basename(os.path.dirname(p))))
    return [rid for _, _, rid in sorted(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--labels', nargs='*', default=[],
                    help='select every valid run whose label matches one of these '
                         'glob patterns (for checkpoints trained in the same chain)')
    ap.add_argument('--reference', help='label to pair against (default: the '
                    'first run\'s label)')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--all', action='store_true', help='all 67 paradigms')
    ap.add_argument('--paradigms', nargs='*')
    ap.add_argument('--local', help='directory of BLiMP jsonl files')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--skip-missing', action='store_true',
                    help='skip checkpoints without model.pt instead of stopping '
                         '(for unattended chains); skips are printed and recorded')
    args = ap.parse_args()

    from morphlm.record import AuditRecord
    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import get_device
    from morphlm.registry import context_facts
    rec = AuditRecord('blimp_eval', args)

    args.runs = list(args.runs) + [r for r in select(args.labels, args.results_dir)
                                   if r not in args.runs]
    if not args.runs:
        raise SystemExit('no runs given or matched')
    missing = [r for r in args.runs if not os.path.exists(
        os.path.join(args.results_dir, r, 'model.pt'))]
    if missing and not args.skip_missing:
        raise SystemExit(f'no checkpoint for {missing}; '
                         f'pass --skip-missing to continue without them')
    runs = [r for r in args.runs if r not in missing]
    if missing:
        print(f'  SKIPPED (no model.pt): {missing}')
    rec.add('skipped', missing)
    if not runs:
        raise SystemExit('no checkpoints to score')

    paradigms = 'all' if args.all else (args.paradigms or list(MORPHOLOGY))
    pairs = load_pairs(paradigms, args.local)
    if args.limit:
        pairs = {k: v[:args.limit] for k, v in pairs.items()}
    names = list(pairs)
    ann = {p: annotation(p, pairs[p]) for p in names}
    good, bad, owner = [], [], []
    for p in names:
        for it in pairs[p]:
            good.append(it['sentence_good'])
            bad.append(it['sentence_bad'])
            owner.append(p)
    owner = np.array(owner)
    n_items = len(good)
    print(f'  {len(names)} paradigm(s), {n_items:,} minimal pairs, '
          f'{len(runs)} checkpoint(s)')

    device = get_device()
    enc = Encoder()
    per_run, items = {}, {}
    for rid in runs:
        t0 = time.time()
        model, cfg = load_checkpoint(rid, args.results_dir)
        model = model.to(device).eval()
        facts = list(context_facts(cfg.frozen.facts))
        print(f'\n  {rid}  {cfg.label()}  seed {cfg.train.seed}')
        seqs, codes, nones = enc.encode_all(cfg, facts, good + bad)
        worst = self_check(model, device, seqs, codes, nones)
        nll = score(model, device, seqs, codes, nones, range(len(seqs)), args.batch)
        margin = np.array([nll[n_items + i] - nll[i] for i in range(n_items)])
        acc = {p: float((margin[owner == p] > 0).mean()) for p in names}
        mar = {p: float(margin[owner == p].mean()) for p in names}
        per_run[rid] = {'label': cfg.label(), 'seed': cfg.train.seed,
                        'accuracy': acc, 'margin': mar,
                        'batch_check_max_diff': worst}
        items[rid] = margin.astype(np.float32)
        print(f'    scored in {time.time() - t0:.0f}s (batch check {worst:.1e} nats); '
              f'accuracy {100 * st.mean(acc.values()):.1f}% (mean over paradigms, '
              f'as in the table)')
        del model

    by = defaultdict(dict)
    for rid, r in per_run.items():
        if r['seed'] in by[r['label']]:
            print(f'  note: {r["label"]} seed {r["seed"]} given twice; using the first')
            continue
        by[r['label']][r['seed']] = rid
    ref = args.reference or per_run[runs[0]]['label']
    if ref not in by:
        raise SystemExit(f'reference {ref!r} was not scored')
    labels = [ref] + [l for l in by if l != ref]

    def agg(rid, sel, metric):
        return st.mean(per_run[rid][metric][p] for p in sel)

    groups = [('ALL', names)]
    for f in sorted({ann[p][0] for p in names}):
        groups.append((f'field: {f}', [p for p in names if ann[p][0] == f]))
    for t in sorted({ann[p][1] for p in names}):
        groups.append((f'  {t}', [p for p in names if ann[p][1] == t]))

    print()
    print('=' * 112)
    print(f'BLiMP — accuracy per cell, mean over seeds; paired effect against {ref}')
    print('  effect = cell minus reference at the SAME seed, in accuracy points: '
          'mean ± se, seeds agreeing in sign')
    print('=' * 112)
    short = {l: l.replace('bpe_', '').replace('_bpe_17k', '').replace('-LEARN', '')[:25]
             for l in labels}
    print(f'  {"":32}' + ''.join(f'{short[l]:>27}' for l in labels))
    print(f'  {"":32}' + ''.join(f'{"seeds " + ",".join(map(str, sorted(by[l]))):>27}'
                                 for l in labels))
    summary = {}
    for gname, sel in groups:
        row = f'  {gname[:32]:32}'
        summary[gname] = {}
        for l in labels:
            seeds = sorted(by[l])
            a = st.mean(agg(by[l][s], sel, 'accuracy') for s in seeds)
            cell = f'{100 * a:5.1f}%'
            entry = {'accuracy': a, 'seeds': seeds}
            if l != ref:
                shared = [s for s in seeds if s in by[ref]]
                da = [100 * (agg(by[l][s], sel, 'accuracy') - agg(by[ref][s], sel, 'accuracy'))
                      for s in shared]
                dm = [agg(by[l][s], sel, 'margin') - agg(by[ref][s], sel, 'margin')
                      for s in shared]
                if len(da) >= 2:
                    se = st.stdev(da) / len(da) ** 0.5
                    agree = max(sum(x > 0 for x in da), sum(x < 0 for x in da))
                    cell += f' {st.mean(da):+5.1f}±{se:3.1f} {agree}/{len(da)}'
                elif len(da) == 1:
                    cell += f' {da[0]:+5.1f} n=1'
                entry.update(paired_seeds=shared, paired_acc_diff=da, paired_margin_diff=dm)
            summary[gname][l] = entry
            row += f'{cell:>27}'
        print(row)
    print()
    print('  READING THIS')
    print('    Chance is 50%. A cell differs from the reference only if its paired')
    print('    difference agrees in sign across seeds and is larger than its se.')
    print('    Per-item margins are saved for item-level pairing.')

    rec.add('runs', per_run)
    rec.add('reference', ref)
    rec.add('summary', summary)
    rec.add('annotation', {p: {'field': ann[p][0], 'term': ann[p][1]} for p in names})
    path = rec.write()
    base = path[:-5] if isinstance(path, str) and path.endswith('.json') else \
        os.path.join(args.results_dir, 'audits', f'blimp_{int(time.time())}')
    np.savez_compressed(base + '_items.npz', owner=owner, **items)
    print(f'  [items] {base}_items.npz')


if __name__ == '__main__':
    main()
