#!/usr/bin/env python3
"""
multiblimp_eval.py — MultiBLiMP 1.0's Ukrainian minimal pairs on saved checkpoints,
paired by seed (DIRECTIONS, "The queue"; journey §33.79).

WHY
---
Bits per byte measure prediction. MultiBLiMP 1.0 (Jumelet et al. 2026; data in
`external/multiblimp/ukr/data.tsv`, CC BY 4.0) gives 2,744 Ukrainian pairs of a
sentence and the same sentence with one verb changed to break subject–verb
agreement — in number (`SV-#`), gender (`SV-G`) or person (`SV-P`), what a verb's
ending carries. A model is right on a pair when it gives the grammatical sentence
the higher probability. It asks whether the tokenizer's gain, and case's, show in
grammatical judgements as well as in prediction.

HOW IT SCORES
-------------
As the English BLiMP evaluator does (`blimp_eval.py`: its batched scorer and its
batch check are reused), with what a Ukrainian paragraph is in training: each
sentence is encoded alone by the checkpoint's own pipeline — its tokenizer, ending
marks included, and the tagger reading the sentence — and preceded by a newline,
the token before every paragraph in the Ukrainian streams; the newline itself is not
scored. Features are coded as the caches code them (`code`, or a block's
`codes_for`: the null gives every word one class, G-73; the word form reads its
lexicon); the model shifts them itself.

A check that could fail, before anything is scored: for each feature configuration
the first paragraphs of the checkpoint's own validation stream are re-encoded by
this path, and must reproduce the cached token ids and every cached code exactly.

Accuracy and the mean log-probability margin (nats by which the grammatical sentence
is preferred) are reported over all pairs and per phenomenon; each cell is paired by
seed with a reference cell, within one machine (FINDINGS, method point 11: a seed whose
runs ran on two machines is named and left out). Per-item margins are saved beside the
audit record.

    python scripts/eval/multiblimp_eval.py --labels 'bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2' 'bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2'
    python scripts/eval/multiblimp_eval.py --labels '*_ukwiki0.05_st10k' --device cpu --threads 4
    python scripts/eval/multiblimp_eval.py --labels '*_ukwiki0.05_st10k' --machine second

--machine second scores the second machine's runs, its twins' models included
(results/audits/device_twins_models/; morphlm/runs.py).
"""
import os
import sys
import csv
import time
import argparse
import dataclasses
import statistics as st
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('MORPHLM_NO_BUILD', '1')
from morphlm import runs as R  # noqa: E402   (`runs` names the run ids scored here)
from blimp_eval import encode, locate, score, self_check  # noqa: E402   (beside this script)

DATA = os.path.join(ROOT, 'external', 'multiblimp', 'ukr', 'data.tsv')
PHENOMENA = {'SV-#': 'number', 'SV-G': 'gender', 'SV-P': 'person'}


def load_pairs(path):
    csv.field_size_limit(10 ** 8)
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8'), delimiter='\t',
                               quoting=csv.QUOTE_NONE))
    bad = [i for i, r in enumerate(rows) if not r.get('sen') or not r.get('wrong_sen')
           or r['sen'] == r['wrong_sen'] or r.get('phenomenon') not in PHENOMENA]
    if bad:
        raise SystemExit(f'{len(bad)} malformed rows in {path} (first: {bad[:5]})')
    return rows


class Encoder:
    """Each sentence encoded once per feature configuration, as a paragraph start."""

    def __init__(self):
        self.cache = {}

    def _setup(self, cfg, facts):
        from morphlm.pipelines.registry import build_pipeline
        from morphlm.data.stream import cache_blocks
        data_cfg = dataclasses.replace(cfg, frozen=dataclasses.replace(cfg.frozen, facts=tuple(facts),
                                                                       shuffle_codes=False))
        pipe = build_pipeline(data_cfg, verbose=False)
        blocks = cache_blocks(data_cfg, facts)
        if any(b.source != 'context' for b in blocks):
            raise RuntimeError(f'non-context fact among {facts}')
        nl = pipe.encode_ids('\n')
        if len(nl) != 1:
            raise RuntimeError(f'a newline encodes to {len(nl)} tokens; the scorer skips exactly one')
        return data_cfg, pipe, blocks, nl[0]

    def check(self, cfg, facts, k):
        """The first k paragraphs of the validation stream, re-encoded by this path, must
        reproduce the cached ids and codes exactly."""
        from morphlm.data.windows import make_val_dataset
        from morphlm.data.tokenization import load_tokenizer
        from morphlm.features.lexical import _bytelevel_decoder
        data_cfg, pipe, blocks, nl = self._setup(cfg, facts)
        ds = make_val_dataset(pipe, data_cfg, verbose=False)
        if list(getattr(ds, 'fact_names', []) or []) != list(facts):
            raise RuntimeError(f'the loader gave {ds.fact_names}, not {facts}')
        tok = load_tokenizer(cfg.data.bpe_path, vocab_only=True)
        dec = _bytelevel_decoder()
        raw = {}
        ids = np.asarray(ds.ids[:200_000], dtype=np.int64)
        ends = np.flatnonzero(ids == nl)
        start, done, mism = 0, 0, []
        for e in ends[:k]:
            span = ids[start:e + 1]
            for i in set(span.tolist()) - raw.keys():
                raw[i] = bytes(dec[ch] for ch in (tok.id_to_token(i) or '') if ch in dec)
            text = b''.join(raw[i] for i in span.tolist()).decode('utf-8')
            got_ids, got_codes = encode(pipe, blocks, text)
            if got_ids != span.tolist():
                mism.append((done, 'ids'))
            else:
                for r, f in enumerate(facts):
                    want = np.asarray(ds.codes[r, start:e + 1], dtype=np.int64).tolist()
                    if got_codes[r] != want:
                        mism.append((done, f))
            start, done = e + 1, done + 1
        if done < k or mism:
            raise SystemExit(f'{cfg.label()}: {len(mism)} of {done} validation paragraphs not reproduced '
                             f'by the scoring path (first: {mism[:4]}) — nothing scored')
        return done

    def encode_all(self, cfg, facts, sentences):
        k = (cfg.data.bpe_path, tuple(facts), cfg.data.corpus, cfg.data.corpus_fraction)
        if k in self.cache:
            return self.cache[k]
        _, pipe, blocks, nl = self._setup(cfg, facts)
        t0 = time.time()
        seqs, codes = [], []
        for s in sentences:
            ids, cs = encode(pipe, blocks, s)
            seqs.append([nl] + ids)
            codes.append([[b.n_classes - 1] + c for b, c in zip(blocks, cs)] if blocks else None)
        nones = [b.n_classes - 1 for b in blocks]
        print(f'    encoded {len(sentences):,} sentences for {list(facts) or "no facts"} under '
              f'{os.path.basename(cfg.data.bpe_path)} in {time.time() - t0:.0f}s', flush=True)
        self.cache[k] = (seqs, codes, nones)
        return self.cache[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', nargs='+', required=True, help='glob patterns of run labels (valid runs only)')
    ap.add_argument('--reference', help="label to pair against (default: the first label's runs)")
    R.add_results_flag(ap)
    ap.add_argument('--machine', choices=('thesis', 'second'), default=None,
                    help="one machine's runs, found among its records (the second's twins too)")
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--check', type=int, default=40, help='validation paragraphs re-encoded per configuration')
    ap.add_argument('--device', default=None, help='cpu while a chain holds the GPU')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--by-length', action='store_true',
                    help='also report the pairs whose two sentences have the same number of tokens under '
                         'every tokenizer scored together — a model may prefer a sentence for its length')
    ap.add_argument('--check-only', action='store_true',
                    help='run the reproduction check for every configuration and score nothing')
    a = ap.parse_args()

    import torch
    from morphlm.record import AuditRecord
    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import get_device
    from morphlm.registry import context_facts
    if a.threads:
        torch.set_num_threads(a.threads)
    device = a.device or get_device()
    rec = AuditRecord('multiblimp_eval', a, out_dir=R.AUDITS)

    rows = load_pairs(a.data)
    if a.limit:
        rows = rows[:a.limit]
    good, bad = [r['sen'] for r in rows], [r['wrong_sen'] for r in rows]
    phen = np.array([PHENOMENA[r['phenomenon']] for r in rows])
    n = len(rows)
    where = {}
    for pat in a.labels:                 # in the order of the patterns
        where.update({k: v for k, v in locate([], [pat], R.results_of(a), a.machine).items() if k not in where})
    runs = [r for r in where if os.path.exists(os.path.join(where[r][0], r, 'model.pt'))]
    if not runs:
        raise SystemExit('no checkpoints matched')
    print(f'  {n:,} pairs ({", ".join(f"{k} {int((phen == k).sum())}" for k in PHENOMENA.values())}); '
          f'{len(runs)} checkpoint(s); device {device}', flush=True)

    enc, checked = Encoder(), {}
    per_run, items = {}, {}
    for rid in runs:
        t0 = time.time()
        model, cfg = load_checkpoint(rid, where[rid][0])
        model = model.to(device).eval()
        facts = list(context_facts(cfg.frozen.facts))
        ck = (cfg.data.bpe_path, tuple(facts), cfg.data.corpus, cfg.data.corpus_fraction)
        if ck not in checked:
            checked[ck] = enc.check(cfg, facts, a.check)
            print(f'    {cfg.label()}: {checked[ck]} validation paragraphs reproduced, ids and codes', flush=True)
        if a.check_only:
            del model
            continue
        seqs, codes, nones = enc.encode_all(cfg, facts, good + bad)
        worst = self_check(model, device, seqs, codes, nones)
        nll = score(model, device, seqs, codes, nones, range(len(seqs)), a.batch)
        margin = np.array([nll[n + i] - nll[i] for i in range(n)])
        acc = {p: float((margin[phen == p] > 0).mean()) for p in PHENOMENA.values()}
        acc['all'] = float((margin > 0).mean())
        mar = {p: float(margin[phen == p].mean()) for p in PHENOMENA.values()}
        mar['all'] = float(margin.mean())
        per_run[rid] = {'label': cfg.label(), 'seed': cfg.train.seed,
                        'machine': R.machine(where[rid][1]) or R.gpu(where[rid][1]),
                        'accuracy': acc, 'margin': mar, 'batch_check_max_diff': worst}
        items[rid] = margin.astype(np.float32)
        print(f'  {rid}  {cfg.label()}  seed {cfg.train.seed}: accuracy {100 * acc["all"]:.1f}% '
              f'(number {100 * acc["number"]:.1f}, gender {100 * acc["gender"]:.1f}, '
              f'person {100 * acc["person"]:.1f}); {time.time() - t0:.0f}s', flush=True)
        del model

    if a.check_only:
        print(f'  check only: {len(checked)} configuration(s) reproduced; nothing scored')
        return
    if a.by_length:
        # the pairs of equal token length under EVERY configuration scored (each encoder's sequences)
        same = np.ones(n, dtype=bool)
        for seqs, _, _ in enc.cache.values():
            same &= np.array([len(seqs[i]) == len(seqs[n + i]) for i in range(n)])
        for rid in per_run:
            m = items[rid]
            per_run[rid]['accuracy']['same length'] = float((m[same] > 0).mean())
            per_run[rid]['margin']['same length'] = float(m[same].mean())
        rec.add('same_length_pairs', {'count': int(same.sum()),
                                      'by_phenomenon': {p: int((same & (phen == p)).sum()) for p in PHENOMENA.values()}})
        print(f'  pairs of equal token length under every tokenizer scored: {int(same.sum()):,} of {n:,}')
    by = defaultdict(dict)
    for rid, r in per_run.items():
        by[r['label']].setdefault(r['seed'], rid)
    ref = a.reference or per_run[runs[0]]['label']
    labels = [ref] + sorted(l for l in by if l != ref)
    # pairs within one machine (morphlm.runs.pair); the seeds left out are named once
    scored = {(l, s): where[rid][1] for l, d in by.items() for s, rid in d.items()}
    paired, left_out = {}, {}
    for l in labels[1:]:
        ab, left = R.pair(scored, ref, l)
        paired[l] = set(ab)
        if left:
            left_out[l] = {str(s): list(m) for s, m in left.items()}
            print(f'  {l} against {ref}: {R.left_out(left)}')
    keys = ['all'] + list(PHENOMENA.values()) + (['same length'] if a.by_length else [])
    summary = {}
    print(f'\n  MultiBLiMP, Ukrainian — accuracy, mean over seeds; paired by seed against {ref}: '
          f'points, mean ± se (seeds agreeing in sign)')
    print(f'  {"cell":62} ' + ''.join(f'{k:>24}' for k in keys))
    for l in labels:
        seeds = sorted(by[l])
        row, summary[l] = f'  {l[:62]:62} ', {}
        for k in keys:
            m = st.mean(per_run[by[l][s]]['accuracy'][k] for s in seeds)
            cell, entry = f'{100 * m:5.1f}%', {'accuracy': m, 'seeds': seeds}
            if l != ref:
                shared = [s for s in seeds if s in by[ref] and s in paired[l]]
                d = [100 * (per_run[by[l][s]]['accuracy'][k] - per_run[by[ref][s]]['accuracy'][k]) for s in shared]
                dm = [per_run[by[l][s]]['margin'][k] - per_run[by[ref][s]]['margin'][k] for s in shared]
                if len(d) >= 2:
                    sm = R.summary(d)
                    cell += f' {sm["mean"]:+.1f}±{sm["se"]:.1f} {R.agree(d)}/{len(d)}'
                entry.update(paired_seeds=shared, paired_acc_diff=d, paired_margin_diff=dm)
                if l in left_out:
                    entry['left_out'] = left_out[l]
            summary[l][k] = entry
            row += f'{cell:>24}'
        print(row)
    rec.add('runs', per_run)
    rec.add('reference', ref)
    rec.add('summary', summary)
    rec.add('checked_paragraphs', {f'{os.path.basename(k[0])} {list(k[1])} {k[2]}{k[3]}': v for k, v in checked.items()})
    base = rec.write()[:-len('.json')]
    np.savez_compressed(base + '_items.npz', phenomenon=phen, **items)
    print(f'  [items] {base}_items.npz')


if __name__ == '__main__':
    main()
