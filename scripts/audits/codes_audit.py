#!/usr/bin/env python3
"""
codes_audit.py — which validation codes did each run read? (G-72, journey §33.55)

A corpus's validation split is the same tokens at every corpus fraction, so
ukwiki's validation streams at 0.002, 0.01, 0.05 and 0.2 have the same ids
under four cache stems. Until G-72 a run looked a derived fact up — the null
`mconst`, the zero slot `mzero`, a word-form lexicon `ucaselex` — by token count
and file name over every stem's files, so a run at one fraction could be scored
on another fraction's codes; a derivation could copy its kept rows from one too.
Built codes (spaCy's tags of the same tokens) are the same at every fraction;
derived codes are not: a lexicon is built from its fraction's own train split,
and every word unseen there gets the class `none`, which in training marks only
non-words. Train streams differ in length by fraction, so only validation (and
test) could be mixed.

  1  FILES   every row a derived file copied, against the file that derived it;
             and every constant null's held-out rows: one class on every word,
             the rule declared (G-73)
  2  RUNS    for every recorded run with context codes: the validation file it
             read — the old rule, over the files that existed when it started —
             against its own stream's file today: the same, identical content,
             or how many tokens differ
  3  SCORE   (--score) each run that read different codes: its saved model —
             the best step's weights — on the codes it was scored on (which must
             reproduce its recorded BPB, or the reconstruction is wrong), on its
             own fraction's codes, and, for a constant null, on the constant
             itself: every word the same class, seen in train or not, exactly
             its training codes. With --nulls, every constant null that read its
             own codes too: what marking unseen words costs a null that never saw
             the mark in training (G-73)

    python scripts/audits/codes_audit.py                          # files and runs: CPU, a minute
    python scripts/audits/codes_audit.py --score                  # the affected models: GPU minutes
    python scripts/audits/codes_audit.py --score --nulls          # and every constant null
    python scripts/audits/codes_audit.py --score --nulls --match ukwiki

Nothing is changed. One record in results/audits/.
"""
import os
import sys
import glob
import json
import argparse
import datetime
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs as RUNS  # noqa: E402   (`runs` names the run rows here)

import numpy as np

#: a creation and a run start this close are reported as a close call
CLOSE_SECONDS = 600
#: the time from loading the data to the training clock's start, about
LOAD_SECONDS = 30


def iso_epoch(s):
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(str(s).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def when(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%m-%d %H:%M UTC') if t else '?'


def config_from_dict(d):
    """An ExperimentConfig from a recorded config.json, all five sections (fields added since are
    defaulted; a config without `surface` is the legacy stream): ExperimentConfig.from_dict."""
    from morphlm.config import ExperimentConfig
    return ExperimentConfig.from_dict(d)


class Cache:
    """The codes files on disk, with when each first existed."""

    def __init__(self, cache_dir, audit_dir):
        from morphlm.data.stream import cache_stem
        self.dir = cache_dir
        self.meta, self.arrays = {}, {}
        for p in glob.glob(os.path.join(cache_dir, '*.codes.json')):
            try:
                self.meta[os.path.basename(p)] = json.load(open(p))
            except (OSError, ValueError):
                continue
        self.first = {b: os.path.getmtime(os.path.join(cache_dir, b)) for b in self.meta}
        self.mtime = dict(self.first)
        self.recorded = set()
        self.const_files = {}          # validation/test file of a constant -> (fact, bpe path, ids stem)
        # a derivation's record is stamped when it starts, a few minutes before
        # its files are written; a file's mtime is its LAST writing
        self.derive_runs = []
        for p in sorted(glob.glob(os.path.join(audit_dir, 'derive_lexical_codes_*.json'))):
            try:
                r = json.load(open(p))
            except (OSError, ValueError):
                continue
            a, t = r.get('args') or {}, iso_epoch(r.get('timestamp_utc'))
            if not a.get('bpe_path') or t is None:
                continue
            out = a.get('out_fact') or (a.get('fact', '') + 'lex')
            self.derive_runs.append((t, a, out))
            spec = self.surface(a.get('bpe_path'), a.get('corpus') or 'wikitext103',
                                a.get('corpus_fraction') or 1.0, a.get('corpus_holdout') or 2000)
            for split in a.get('splits') or ['train', 'validation', 'test']:
                b = cache_stem('bpe', split, None, spec) + f'_{out}.codes.json'
                if b in self.first:
                    self.first[b] = min(self.first[b], t)
                    self.recorded.add(b)
                    if a.get('transform') == 'const' and split != 'train':
                        self.const_files[b] = (out, a.get('bpe_path'), cache_stem('bpe', split, None, spec))

    @staticmethod
    def surface(bpe_path, corpus, fraction, holdout):
        from morphlm.config import ExperimentConfig, apply_overrides
        o = ['data.pipeline=bpe', f'data.bpe_path={bpe_path}', f'data.corpus={corpus}']
        if corpus != 'wikitext103':
            o += [f'data.corpus_fraction={fraction}', f'data.corpus_holdout={holdout}']
        return apply_overrides(ExperimentConfig(), o).surface_spec()

    def rows(self, base, facts):
        if base not in self.arrays:
            self.arrays[base] = np.load(os.path.join(self.dir, base[:-5] + '.npy'), mmap_mode='r')
        names = self.meta[base]['fact_names']
        return np.stack([np.asarray(self.arrays[base][names.index(f)]) for f in facts])

    def const_facts(self):
        return {out for _, a, out in self.derive_runs if a.get('transform') == 'const'}


def stream_of(cfg, split='validation'):
    from morphlm.data.stream import cache_stem, codes_spec, facts_to_build
    from morphlm.registry import context_facts
    sfc = cfg.surface_spec()
    cap = cfg.data.max_articles_val if split == 'validation' else cfg.data.max_articles_train
    ids_stem = cache_stem(cfg.data.pipeline, split, cap, sfc)
    gran = cfg.frozen.dep_granularity
    c_spec = codes_spec(list(facts_to_build(cfg)), gran)
    c_spec = f'{sfc}|{c_spec}' if (sfc and c_spec) else (c_spec or sfc)
    exact = cache_stem(cfg.data.pipeline, split, cap, c_spec) + '.codes.json'
    return ids_stem, cap, exact, list(context_facts(cfg.frozen.facts)), gran


def resolve(cache, cfg, before=None, old_rule=False, split='validation'):
    """The codes file a run reads: today's rule, or the rule before G-72 over the
    files that existed at `before`. Returns (ids stem, file basename or None,
    tokens, the candidates that hold the requested facts)."""
    from morphlm.data.stream import stream_codes_files
    ids_stem, cap, exact, requested, gran = stream_of(cfg, split)
    ip = os.path.join(cache.dir, ids_stem + '.json')
    if not os.path.exists(ip):
        return ids_stem, None, None, []
    n = json.load(open(ip)).get('num_tokens')
    if exact in cache.meta and set(requested) <= set(cache.meta[exact].get('fact_names', [])):
        return ids_stem, exact, n, [exact]
    prefix = f'{cfg.data.pipeline}_{split}_{cap}_'
    if old_rule:
        names = sorted(b for b in cache.meta if b.startswith(prefix))
    else:
        names = [os.path.basename(p) for p, _, _ in
                 stream_codes_files(cache.dir, cfg.data.pipeline, split, cap, ids_stem, n)]
    cands = [b for b in names
             if cache.meta[b].get('num_tokens') == n and cache.meta[b].get('dep_granularity') == gran
             and set(requested) <= set(cache.meta[b].get('fact_names', []))
             and os.path.exists(os.path.join(cache.dir, b[:-5] + '.npy'))]
    live = [b for b in cands if before is None or cache.first[b] <= before]
    return ids_stem, (live[0] if live else None), n, cands


def audit_constants(cache):
    """Every constant null's held-out rows: does any word carry another class
    than the constant (G-73)? And does the file declare the rule?"""
    from morphlm.data.stream import ZERO_SLOT_FACTS
    from morphlm.features.lexical import tokenizer_tables, word_spans
    tables, out = {}, []
    for b, (fact, bpe, ids_stem) in sorted(cache.const_files.items()):
        ip = os.path.join(cache.dir, ids_stem + '.npy')
        bp = bpe if os.path.isabs(bpe) else os.path.join(ROOT, bpe)
        from morphlm.data.stream import require_unpacked
        require_unpacked(cache.dir, ids_stem)           # never a packed stream skipped as missing
        if not (os.path.exists(ip) and os.path.exists(bp)):
            continue
        if bp not in tables:
            tables[bp] = tokenizer_tables(bp)
        ids = np.asarray(np.load(ip, mmap_mode='r'), dtype=np.int64)
        wid, _, _ = word_spans(ids, tables[bp])
        row = cache.rows(b, [fact])[0]
        words = wid >= 0
        marked = float((row[words] != 0).mean()) if words.any() else 0.0
        rule = (cache.meta[b].get('rules') or {}).get(fact)
        out.append({'file': b, 'fact': fact, 'marked': marked, 'rule': rule,
                    'zero_slot': fact in ZERO_SLOT_FACTS})
    return out


def audit_files(cache, splits):
    """Every row a derived file holds besides its own, against its original."""
    from morphlm.data.stream import codes_owner
    out = []
    for split in splits:
        for b in sorted(cache.meta):
            for pipe in ('bpe',):
                prefix = f'{pipe}_{split}_None_'
                kind, owner = codes_owner(b, prefix)
                if kind != 'derived':
                    continue
                m = cache.meta[b]
                n = m.get('num_tokens')
                own = b[:-len('.codes.json')][len(owner) + 1:]
                for f in m.get('fact_names', [])[:-1]:
                    # the original: the stream's own derivation file of f, else a build file
                    orig = owner + f'_{f}.codes.json'
                    if orig not in cache.meta or f not in cache.meta[orig].get('fact_names', []):
                        orig = next((c for c in sorted(cache.meta) if codes_owner(c, prefix)[0] == 'build'
                                     and cache.meta[c].get('num_tokens') == n
                                     and f in cache.meta[c].get('fact_names', [])), None)
                    if orig is None:
                        out.append({'file': b, 'fact': f, 'status': 'no original'})
                        continue
                    a, o = cache.rows(b, [f])[0], cache.rows(orig, [f])[0]
                    diff = float((a != o).mean())
                    out.append({'file': b, 'fact': f, 'original': orig, 'differs': diff,
                                'status': 'same' if diff == 0 else 'DIFFERS'})
    return out


def score(run_id, results_dir, cache, ids_stem, n, facts, variants, device):
    """BPB of a saved model on the validation stream under each codes variant."""
    import torch
    from torch.utils.data import DataLoader
    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import evaluate
    from morphlm.data.windows import LMWindowDataset
    model, cfg = load_checkpoint(run_id, results_dir)
    model = model.to(device).eval()
    meta = json.load(open(os.path.join(cache.dir, ids_stem + '.json')))
    from morphlm.data.stream import require_unpacked
    require_unpacked(cache.dir, ids_stem)               # a packed stream is unpacked first (cache_pack.py)
    meta['ids'] = np.load(os.path.join(cache.dir, ids_stem + '.npy'), mmap_mode='r')
    meta['fact_names'] = facts
    out = {}
    for name, codes in variants.items():
        meta['codes'] = codes
        ds = LMWindowDataset(meta, seq_len=cfg.data.seq_len,
                             shuffle_codes=getattr(cfg.frozen, 'shuffle_codes', False), seed=cfg.train.seed)
        loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, drop_last=False)
        with torch.no_grad():
            out[name] = evaluate(model, loader, ds, device)['bits_per_byte']
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return out


def constant_row(cache, own_row, bpe_path, ids_stem, none_code):
    """The constant itself: class 0 on every word token, `none` elsewhere — the
    null's training codes exactly, with no word left out as unseen."""
    from morphlm.features.lexical import tokenizer_tables, word_spans
    from morphlm.data.stream import require_unpacked
    require_unpacked(cache.dir, ids_stem)               # a packed stream is unpacked first (cache_pack.py)
    ids = np.asarray(np.load(os.path.join(cache.dir, ids_stem + '.npy'), mmap_mode='r'), dtype=np.int64)
    wid, _, _ = word_spans(ids, tokenizer_tables(bpe_path))
    row = np.where(wid >= 0, 0, none_code).astype(np.int16)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    ap.add_argument('--audit-dir', default=None, help='default: <results-dir>/audits')
    ap.add_argument('--runs', nargs='*', default=None, help='only these run ids')
    ap.add_argument('--score', action='store_true', help='re-score the affected runs\' saved models')
    ap.add_argument('--nulls', action='store_true',
                    help='with --score: also every constant null that read its own codes, on the constant '
                         'itself — what marking unseen words as non-words costs it (G-73)')
    ap.add_argument('--match', default=None, help='with --nulls: only labels containing this')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    audit_dir = a.audit_dir or os.path.join(a.results_dir, 'audits')

    from morphlm.record import AuditRecord
    rec = AuditRecord('codes_audit', a, out_dir=audit_dir)
    cache = Cache(a.cache_dir, audit_dir)
    print(f'{len(cache.meta)} codes files in {a.cache_dir}; {len(cache.derive_runs)} derivation records')

    # the conditions behind each stem, for reading
    from morphlm.data.stream import cache_stem
    names = {}

    def name_stem(cfg):
        ids_stem = stream_of(cfg)[0]
        d = cfg.data
        names.setdefault(ids_stem, f'{os.path.basename(d.bpe_path)[:-5]}'
                         + (f' {d.corpus} {d.corpus_fraction:g}' if d.corpus != 'wikitext103' else ' wikitext'))

    # ── 1. files ──────────────────────────────────────────────────────────────
    files = audit_files(cache, ['validation', 'test'])
    bad = [f for f in files if f['status'] == 'DIFFERS']
    orphans = [f for f in files if f['status'] == 'no original']
    print(f'\n1  FILES — rows a derived file copied, against the file that derived them '
          f'({len(files)} copies checked' + (f'; {len(orphans)} with no original on disk' if orphans else '') + ')')
    for f in bad:
        print(f'   {f["file"][:-11]:52} row {f["fact"]:9} differs from {f["original"][:-11]} '
              f'on {100 * f["differs"]:.1f}% of tokens')
    if not bad:
        print('   every copied row equals its original')
    rec.add('files', files)

    consts = audit_constants(cache)
    stale = [c for c in consts if c['marked'] > 0 or c['rule'] != 'every word']
    print(f'\n   CONSTANTS — the constant nulls\' held-out rows ({len(consts)} files): one class on every word?')
    for c in stale:
        print(f'   {c["file"][:-11]:52} marks {100 * c["marked"]:.1f}% of word tokens unseen'
              + ('' if c['rule'] == 'every word' else ', no rule declared')
              + (' (a zero slot: its codes never reach the model)' if c['zero_slot'] else ''))
    if consts and not stale:
        print('   every one: the constant on every word, the rule declared')
    if stale:
        print('   derive them again: python scripts/data/rederive_constants.py')
    rec.add('constants', consts)

    # ── 2. runs ───────────────────────────────────────────────────────────────
    runs = []
    for entry in sorted(os.listdir(a.results_dir)):
        if a.runs and entry not in a.runs:
            continue
        rp, cp = (os.path.join(a.results_dir, entry, x) for x in ('result.json', 'config.json'))
        if not (os.path.exists(rp) and os.path.exists(cp)):
            continue
        try:
            r, c = json.load(open(rp)), json.load(open(cp))
            cfg = config_from_dict(c)
        except Exception as e:
            print(f'   {entry}: unreadable ({type(e).__name__}: {e})')
            continue
        if not stream_of(cfg)[3]:
            continue                                    # no context codes: nothing to read
        end = iso_epoch((r.get('environment') or {}).get('timestamp_utc'))
        wall = (r.get('timing') or {}).get('wall_seconds')
        start = end - wall - LOAD_SECONDS if (end and wall) else None
        name_stem(cfg)
        ids_stem, now, n, _ = resolve(cache, cfg)
        _, then, _, cands = resolve(cache, cfg, before=start, old_rule=True)
        timed = True
        if then is None and cands:
            # nothing had been written by the estimated start: a file older
            # than its records (rewritten since), so read the rule without times
            then, timed = cands[0], False
        facts = stream_of(cfg)[3]
        row = {'run_id': entry, 'label': r.get('label'), 'seed': r.get('seed'), 'bpb': r.get('bits_per_byte'),
               'facts': facts, 'ids_stem': ids_stem, 'start': start, 'read': then, 'own': now,
               'timed': timed and start is not None,
               'has_model': os.path.exists(os.path.join(a.results_dir, entry, 'model.pt')),
               'bpe_path': cfg.data.bpe_path}
        if n is None:
            row['status'] = 'no cache on disk'
        elif now is None:
            row['status'] = 'MISS today'
        elif then == now:
            row['status'] = 'own'
        else:
            d = float((cache.rows(then, facts) != cache.rows(now, facts)).mean())
            row['differs'] = d
            row['status'] = 'same content' if d == 0 else 'DIFFERENT'
        if start is not None and len(cands) > 1:
            near = [b for b in cands if abs(cache.first[b] - start) < CLOSE_SECONDS]
            if near:
                row['close_call'] = near
        if then and end and cache.mtime.get(then, 0) > end:
            # written again after the run (a re-derivation): what the run read is
            # no longer on disk, so it can be neither compared nor re-scored
            row['rewritten'] = True
        runs.append(row)

    by = defaultdict(list)
    for x in runs:
        by[(x['label'], x['status'], x.get('read'))].append(x)
    counts = defaultdict(int)
    for x in runs:
        counts[x['status']] += 1
    print(f'\n2  RUNS — the validation codes each run read, against its own stream\'s '
          f'({len(runs)} runs with context codes: '
          + ', '.join(f'{v} {k}' for k, v in sorted(counts.items())) + ')')
    stem_of = {}
    for x in runs:
        stem_of[x['ids_stem']] = names.get(x['ids_stem'], x['ids_stem'])

    def who(base):
        if not base:
            return '-'
        for s, nm in stem_of.items():
            if base.startswith(s):
                return f'{nm}: {base[len(s) + 1:-11] or "build"}'
        return base[:-11]

    shown = 0
    for (label, status, read), xs in sorted(by.items(), key=lambda kv: (kv[0][1] != 'DIFFERENT', kv[0][0])):
        if status in ('own', 'same content', 'no cache on disk'):
            continue
        seeds = ','.join(str(x['seed']) for x in sorted(xs, key=lambda x: x['seed'] or 0))
        diff = np.mean([x.get('differs', 0) for x in xs])
        close = (' (close call: a candidate file appeared within 10 min of a start)'
                 if any('close_call' in x for x in xs) else '')
        close += '' if all(x['timed'] for x in xs) else ' (without times)'
        print(f'   {status:13} {label[:58]:58} seeds {seeds:8} read [{who(read)}]'
              + (f', {100 * diff:.1f}% of tokens differ' if status == 'DIFFERENT' else '') + close)
        shown += 1
    absent = counts.get('no cache on disk', 0)
    if not shown:
        print('   every run whose caches are on disk read its own stream\'s codes, or identical ones'
              if len(runs) > absent else '   no run\'s caches are on disk: nothing to check')
    if absent and len(runs) > absent:
        print(f'   ({absent} runs not checked: their caches are not on disk)')
    same = [x for x in runs if x['status'] == 'same content']
    if same:
        print(f'   ({len(same)} more runs read another stream\'s file with identical rows — unaffected)')
    rewritten = [x for x in runs if x.get('rewritten')]
    if rewritten:
        print(f'   ({len(rewritten)} runs read a file written again since they ran — a re-derivation; '
              f'their reading is not reconstructed or re-scored)')
    rec.add('runs', runs)

    # ── 3. score ──────────────────────────────────────────────────────────────
    hit = [x for x in runs if x['status'] == 'DIFFERENT' and not x.get('rewritten')]
    consts = cache.const_facts()
    nulls = [x for x in runs if a.nulls and x['status'] != 'DIFFERENT' and x['own'] and not x.get('rewritten')
             and all(f in consts for f in x['facts']) and (not a.match or a.match in (x['label'] or ''))]
    todo = hit + nulls if a.score else []
    if todo:
        import torch
        from morphlm.features.blocks import build_blocks
        from morphlm.config import ExperimentConfig, apply_overrides
        device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n3  SCORE — saved models (the best step\'s weights) on {device}: {len(hit)} that read '
              f'different codes' + (f', {len(nulls)} constant nulls that read their own' if a.nulls else ''))
        print(f'   {"label":56} {"seed":>4} {"recorded":>9} {"as read":>9} {"own codes":>10} {"constant":>9}')
        scored = []
        for x in todo:
            if not x['has_model']:
                print(f'   {x["label"][:56]:56} {x["seed"]:>4}  no model.pt')
                continue
            facts, ids_stem = x['facts'], x['ids_stem']
            n = cache.meta[x['own']]['num_tokens']
            variants = {'as_read': cache.rows(x['read'], facts)}
            if x['read'] != x['own']:
                variants['own'] = cache.rows(x['own'], facts)
            if all(f in consts for f in facts):
                own_rows = variants.get('own', variants['as_read'])
                rows = []
                for i, f in enumerate(facts):
                    blk = build_blocks(apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', f'frozen.facts={f}']))[0]
                    rows.append(constant_row(cache, own_rows[i], x['bpe_path'], ids_stem, blk.n_classes - 1))
                variants['constant'] = np.stack(rows)
            try:
                s = score(x['run_id'], a.results_dir, cache, ids_stem, n, facts, variants, device)
            except Exception as e:
                print(f'   {x["label"][:56]:56} {x["seed"]:>4}  FAILED: {type(e).__name__}: {str(e)[:60]}')
                continue
            s.setdefault('own', s['as_read'])
            x['scores'] = s
            ok = x['bpb'] is not None and abs(s['as_read'] - x['bpb']) < 2e-4
            print(f'   {x["label"][:56]:56} {x["seed"]:>4} {x["bpb"]:9.4f} {s["as_read"]:9.4f} {s["own"]:10.4f} '
                  + (f'{s["constant"]:9.4f}' if 'constant' in s else f'{"-":>9}')
                  + ('' if ok else '   <- does NOT reproduce the record: the reconstruction is wrong'))
            scored.append(x)
        cells = defaultdict(list)
        for x in scored:
            cells[x['label']].append(x)
        print(f'\n   per cell, mean ± s.e. over seeds (positive = the run scores worse):')
        print(f'   {"label":56} {"own − as read":>16} {"constant − own":>16}')

        def ms(d):
            s = RUNS.summary(d)
            if s is None:
                return f'{"-":>16}'
            if s['n'] == 1:
                return f'{d[0]:+.4f} (1 seed)'.rjust(16)
            return f'{s["mean"]:+.4f} ± {s["se"]:.4f}'.rjust(16)
        for label, xs in sorted(cells.items()):
            d1 = [x['scores']['own'] - x['scores']['as_read'] for x in xs if x['read'] != x['own']]
            d2 = [x['scores']['constant'] - x['scores']['own'] for x in xs if 'constant' in x['scores']]
            print(f'   {label[:56]:56} {ms(d1)} {ms(d2)}  (n={len(xs)})')
        rec.add('scores', [{k: x[k] for k in ('run_id', 'label', 'seed', 'bpb', 'scores')} for x in scored])
    elif hit:
        print(f'\n3  SCORE — {len(hit)} runs read different codes; --score re-scores their saved models')
    if bad:
        print('\n   copied rows that differ: delete those files (both splits\' .codes.npy/.json) and '
              'derive again — prepare_condition.py skips what is done and re-derives what is missing')
    rec.write()


if __name__ == '__main__':
    main()
