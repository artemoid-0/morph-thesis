#!/usr/bin/env python3
"""
tagger_record.py — which spaCy tagger, at which version, tagged each corpus's
caches? (G-74; journey §33.63)

Until 7 October a run's environment record named `en_core_web_sm` whenever it
was installed, whatever the run used, so no record says which version of
`uk_core_news_sm` tagged the Ukrainian caches. That is recoverable only on the
machine that tagged them, and only while the version is still installed.

For each corpus whose runs are in results/ (or results_invalid/), this finds its
tagger (the runs' `data.spacy_model`), the installed version, and when that
version was installed — the modification time of its package's RECORD file,
which pip writes at install — and the earliest evidence of the corpus's caches:
the oldest of its streams' metadata files, of the cache checks and derivations
recorded for it (results/audits/), and of its runs. File times alone could
mislead — a copied cache folder takes the copy's time — so the earliest of all
three is used. A tagger installed before that, and not reinstalled since (a
reinstall rewrites RECORD), tagged every cache of the corpus at the version
recorded here.

    python scripts/ops/tagger_record.py

Writes results/audits/tagger_record_<UTC timestamp>.json. Changes nothing else.
"""
import os
import sys
import json
import glob
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def iso(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat(timespec='seconds') if t else None


def installed(model):
    """(version, install time as epoch seconds) of an installed tagger, or (None, None)."""
    from importlib import metadata
    try:
        dist = metadata.distribution(model)
    except metadata.PackageNotFoundError:
        return None, None
    record = next((dist.locate_file(f) for f in (dist.files or [])
                   if f.name == 'RECORD' and str(f.parent).endswith('.dist-info')), None)
    when = os.path.getmtime(record) if record is not None and os.path.exists(record) else None
    return dist.version, when


def _epoch(ts):
    try:
        t = datetime.datetime.fromisoformat(ts)
        return (t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)).timestamp()
    except Exception:
        return None


def corpora(cache_dir):
    """
    Per corpus: the taggers its runs name, how many runs, and the earliest
    evidence of its caches — (epoch seconds, where it comes from).
    """
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem
    out = {}

    def seen(corpus, t, what):
        c = out.setdefault(corpus, {'taggers': set(), 'runs': 0, 'first': None, 'first_from': None})
        if t is not None and (c['first'] is None or t < c['first']):
            c['first'], c['first_from'] = t, what

    for cp in glob.glob(os.path.join(ROOT, 'results', '*', 'config.json')) + \
            glob.glob(os.path.join(ROOT, 'results_invalid', '**', 'config.json'), recursive=True):
        try:
            data = json.load(open(cp)).get('data', {})
        except Exception:
            continue
        if data.get('pipeline', 'bpe') != 'bpe' or not data.get('bpe_path'):
            continue
        corpus = data.get('corpus', 'wikitext103')
        seen(corpus, None, None)
        out[corpus]['runs'] += 1
        if data.get('spacy_model'):
            out[corpus]['taggers'].add(data['spacy_model'])
        rp = os.path.join(os.path.dirname(cp), 'result.json')
        if os.path.exists(rp):
            try:
                env = json.load(open(rp)).get('environment') or {}
                seen(corpus, _epoch(env.get('timestamp_utc', '')),
                     'run ' + os.path.relpath(os.path.dirname(cp), ROOT))
            except Exception:
                pass
        over = ['data.pipeline=bpe', f'data.bpe_path={data["bpe_path"]}', f'data.corpus={corpus}']
        if corpus != 'wikitext103':
            over += [f'data.corpus_fraction={data.get("corpus_fraction", 1.0)}',
                     f'data.corpus_holdout={data.get("corpus_holdout", 2000)}']
        try:
            spec = apply_overrides(ExperimentConfig(), over).surface_spec()
        except Exception:
            continue
        for split in ('train', 'validation'):
            p = os.path.join(cache_dir, cache_stem('bpe', split, None, spec) + '.json')
            if os.path.exists(p):
                seen(corpus, os.path.getmtime(p), 'cache ' + os.path.basename(p))
    for script in ('check_cache', 'derive_lexical_codes'):
        for p in glob.glob(os.path.join(ROOT, 'results', 'audits', f'{script}_*.json')):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            corpus = (d.get('args') or {}).get('corpus') or 'wikitext103'
            if corpus in out:
                seen(corpus, _epoch(d.get('timestamp_utc', '')), 'audit ' + os.path.basename(p))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', default=os.path.join(ROOT, 'data', 'cache'))
    a = ap.parse_args()
    from morphlm.record import AuditRecord
    from morphlm.provenance import SPACY_MODELS
    rec = AuditRecord('tagger_record', a, out_dir=os.path.join(ROOT, 'results', 'audits'))
    found = corpora(a.cache_dir)
    taggers = {}
    for m in sorted(set(SPACY_MODELS) | {t for c in found.values() for t in c['taggers']}):
        v, t = installed(m)
        taggers[m] = {'version': v, 'installed_utc': iso(t), '_t': t}
        print(f'  {m:18} {v or "not installed":14} installed {iso(t) or "—"}')
    rows = {}
    for corpus, c in sorted(found.items()):
        for m in sorted(c['taggers']):
            tg = taggers.get(m, {})
            before = (tg.get('_t') is not None and c['first'] is not None and tg['_t'] < c['first'])
            verdict = (f'{m} {tg["version"]} tagged every {corpus} cache: installed before the earliest '
                       f'evidence of one, not reinstalled since' if before else
                       f'not established: {m} was installed after the earliest {corpus} cache, or a time is missing')
            rows[f'{corpus}:{m}'] = {'tagger': m, 'version': tg.get('version'),
                                     'installed_utc': tg.get('installed_utc'),
                                     'earliest_cache_evidence_utc': iso(c['first']),
                                     'earliest_from': c['first_from'],
                                     'runs': c['runs'], 'every_cache': before, 'verdict': verdict}
            print(f'  {corpus:12} {c["runs"]:4} runs; earliest cache evidence {iso(c["first"]) or "—"} '
                  f'({c["first_from"] or "none"})\n               {verdict}')
    rec.add('taggers', {m: {k: v for k, v in t.items() if not k.startswith('_')} for m, t in taggers.items()})
    rec.add('corpora', rows)
    rec.write()


if __name__ == '__main__':
    main()
