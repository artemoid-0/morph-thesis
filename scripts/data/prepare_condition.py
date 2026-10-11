#!/usr/bin/env python3
"""
prepare_condition.py — the data for one condition: corpus × size × tokenizer
(journey §33.53; docs/reference/conditions.md).

The recipe that built the Ukrainian runs (docs/reference/ukrainian_pipeline.md),
as one command that skips what is done, stops at the first failure, and can be
run inside a chain:

  1  validation cache — the pilot: minutes, timed, so the train build can be
     estimated before it starts (from the slice's bytes and tagging time, where
     another tokenizer's cache of the same slice records them)
  2  check it — every source fact must fire (scripts/data/check_cache.py)
  3  train cache — the tagging pass: hours on a full corpus
  4  derived facts — the null (`mconst`), the zero slot (`mzero`) and each
     word-form variant (`<fact>lex`, checked against known words)
  5  check that every fact a run will ask for is in the caches, both splits

It calls the commands the project already uses — `morphlm.cli cache`,
`check_cache.py`, `derive_lexical_codes.py` — so a condition built here is built
exactly as one built by hand. The tokenizer must exist (train_tokenizer.py).

    python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction 0.05 --bpe-path data/bpe_uk_8k.json --dry-run
    python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction 0.05 --bpe-path data/bpe_uk_8k.json --pilot
    python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction 0.05 --bpe-path data/bpe_uk_8k.json
"""
import os
import sys
import json
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

#: What a run in each language compares (DIRECTIONS D7–D9): the source fact whose
#: cache is built — requesting `ucase` builds every Universal Dependencies fact in
#: one pass, `pos` among them — and the word-form variants derived from it.
DEFAULT_LEXICAL = {'uk': ('ucase',), 'en': ('pos',)}


def stamp(msg):
    print(f'\n=== {time.strftime("%F %T")}  {msg}', flush=True)


def run(cmd):
    print('  $ ' + ' '.join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f'  FAILED (exit {r.returncode}) — stopping; fix it and run this command again '
                 f'(finished steps are skipped)')
    return time.time() - t0


def same_text(over, cache, bpe_path):
    """
    This slice's train split as cached under another tokenizer, as (name, its ids
    meta) — or (None, None). It is the same text, so its byte count is this
    slice's, and, for caches built since 7 October, which record it, tagging it
    took about as long as it will now: the latest such build is preferred.
    `over`: the condition's overrides.
    """
    import glob
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import cache_stem
    here, folder = os.path.basename(bpe_path), os.path.dirname(bpe_path)
    found = []
    for tok in sorted(glob.glob(os.path.join(ROOT, folder, '*.json'))):
        name = os.path.basename(tok)
        if name == here or name.endswith('.meta.json'):
            continue
        o = [f'data.bpe_path={os.path.join(folder, name)}' if x.startswith('data.bpe_path=') else x
             for x in over]
        p = os.path.join(cache, cache_stem('bpe', 'train', None,
                                           apply_overrides(ExperimentConfig(), o).surface_spec()) + '.json')
        if os.path.exists(p):
            meta = json.load(open(p))
            found.append((bool(meta.get('build_seconds')), os.path.getmtime(p), name[:-5], meta))
    if not found:
        return None, None
    best = max(found, key=lambda f: f[:2])
    return best[2], best[3]


def _duration(seconds):
    return f'{seconds / 60:.0f} min' if seconds < 5400 else f'{seconds / 3600:.1f} h'


def estimate(secs, val, holdout, fraction, sibling=(None, None)):
    """
    The train build's size in tokens and its tagging time, from the pilot: `secs`
    it took, `val` the validation ids meta, `sibling` what `same_text` found
    (journey §33.63). Returns (tokens, what the size rests on, the time as text).

    The size is the slice's bytes at the pilot's tokens per byte. Without another
    tokenizer's cache of the slice, its bytes are validation's times its expected
    share of the documents — 15% low on ukwiki 0.05, whose validation drew a
    smaller share of the bytes. Tagging time follows the bytes, not the tokenizer:
    a sibling that recorded its time gives it; otherwise the pilot's rate, which
    includes reading through the corpus, bounds it from above.
    """
    sib, sm = sibling
    per_byte = val['num_tokens'] / val['num_bytes']
    if sm and sm.get('num_bytes'):
        train_bytes = sm['num_bytes']
        basis = f"the slice's {train_bytes:,} bytes (cached under {sib}) at {per_byte:.4f} tokens per byte"
    else:
        train_bytes = val['num_bytes'] * (holdout - 2) * min(fraction, 1.0)
        basis = "validation's expected share of the documents (on ukwiki 0.05 this came out 15% low)"
    if sm and sm.get('build_seconds'):
        hours = f"~{_duration(sm['build_seconds'])}: the same text took that under {sib}"
    else:
        hours = (f"at most ~{_duration(secs * train_bytes / val['num_bytes'])} at the pilot's rate, which "
                 f"includes reading through the corpus (ukwiki 0.05's five measured train passes took 37–50% of it)")
    return train_bytes * per_byte, basis, hours


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--corpus-fraction', type=float, default=1.0)
    ap.add_argument('--corpus-holdout', type=int, default=2000)
    ap.add_argument('--bpe-path', required=True)
    ap.add_argument('--source', default='ucase', help='the fact whose cache is built (default ucase: every UD fact)')
    ap.add_argument('--lexical', default=None,
                    help='facts to derive word-form variants of, comma-separated (default: ucase for '
                         'Ukrainian, pos for English); "none" for none')
    ap.add_argument('--lexicon-fraction', type=float, default=None,
                    help='take the word-form lexicons from this corpus size (little data: a larger one)')
    ap.add_argument('--spacy-model', default=None, help="default: the corpus's tagger")
    ap.add_argument('--cache-dir', default='data/cache')
    ap.add_argument('--pilot', action='store_true', help='stop after the validation cache, its check and the estimate')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and what is already done')
    a = ap.parse_args()

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.corpora import get_corpus, check_tagger
    from morphlm.data.stream import cache_stem
    spec = get_corpus(a.corpus)
    if a.corpus == 'wikitext103':
        sys.exit('WikiText-103 is the historical condition; its caches are built by the existing recipes')
    model = a.spacy_model or spec.spacy_model
    check_tagger(a.corpus, model)
    if not os.path.exists(os.path.join(ROOT, a.bpe_path) if not os.path.isabs(a.bpe_path) else a.bpe_path):
        sys.exit(f'{a.bpe_path} not found — train it first: scripts/data/train_tokenizer.py')
    lexical = (DEFAULT_LEXICAL.get(spec.language, ()) if a.lexical is None else
               () if a.lexical == 'none' else tuple(f.strip() for f in a.lexical.split(',') if f.strip()))

    over = ['data.pipeline=bpe', f'data.corpus={a.corpus}', f'data.corpus_fraction={a.corpus_fraction}',
            f'data.corpus_holdout={a.corpus_holdout}', f'data.bpe_path={a.bpe_path}',
            f'data.spacy_model={model}', f'data.cache_dir={a.cache_dir}']
    cfg = apply_overrides(ExperimentConfig(), over)
    spec_str = cfg.surface_spec()
    cache = os.path.join(ROOT, a.cache_dir) if not os.path.isabs(a.cache_dir) else a.cache_dir

    def ids_meta(split):
        p = os.path.join(cache, cache_stem('bpe', split, None, spec_str) + '.json')
        return json.load(open(p)) if os.path.exists(p) else None

    def derived(out_fact):
        return all(os.path.exists(os.path.join(cache, cache_stem('bpe', s, None, spec_str) + f'_{out_fact}.codes.npy'))
                   for s in ('train', 'validation'))


    derive = [('mconst', ['--fact', a.source, '--out-fact', 'mconst', '--transform', 'const', '--keep', a.source]),
              ('mzero', ['--fact', 'mconst', '--out-fact', 'mzero', '--transform', 'const'])]
    derive += [(f'{f}lex', ['--fact', f]) for f in lexical]
    want = [a.source, 'mconst', 'mzero'] + [f'{f}lex' for f in lexical]
    run_over = ' '.join(f'-o {o}' for o in over if not o.startswith(('data.pipeline', 'data.cache_dir')))

    print(f'condition: {a.corpus} at fraction {a.corpus_fraction:g} (holdout {a.corpus_holdout}), '
          f'tokenizer {os.path.basename(a.bpe_path)}, tagger {model}')
    print(f'  facts a run can ask for once prepared: {", ".join(want)} (and every UD fact)')
    val, tr = ids_meta('validation'), ids_meta('train')
    print(f'  validation cache: {"done, %s tokens" % format(val["num_tokens"], ",") if val else "to build"}'
          f' | train cache: {"done, %s tokens" % format(tr["num_tokens"], ",") if tr else "to build"}'
          f' | derived: ' + ', '.join(f'{o} {"done" if derived(o) else "to do"}' for o, _ in derive))
    if a.dry_run:
        print(f'\n  runs will use:  {run_over}')
        return

    py = sys.executable
    cli = [py, '-m', 'morphlm.cli', 'cache', '--pipeline', 'bpe', '--facts', a.source] + \
          sum((['-o', o] for o in over if not o.startswith('data.pipeline')), [])
    check = [py, 'scripts/data/check_cache.py', '--bpe-path', a.bpe_path, '--cache-dir', a.cache_dir,
             '--corpus', a.corpus, '--corpus-fraction', str(a.corpus_fraction),
             '--corpus-holdout', str(a.corpus_holdout)]

    stamp('1  validation cache (the pilot)')
    secs = run(cli + ['--splits', 'validation'])
    val = ids_meta('validation')
    stamp('2  check the validation cache')
    run(check + ['--split', 'validation', '--require-facts', a.source])
    if spec.split_rule == 'hash' and val and secs > 5:
        est_tokens, basis, hours = estimate(secs, val, a.corpus_holdout, a.corpus_fraction,
                                            same_text(over, cache, a.bpe_path))
        print(f'\n  estimate: train ~{est_tokens / 1e6:.1f}M tokens, from {basis}\n'
              f'            tagging {hours}\n'
              f'            (the pilot took {secs:.0f} s for {val["num_tokens"]:,} tokens)')
    if a.pilot:
        print('\n  pilot done — run again without --pilot to build the train split')
        return

    stamp('3  train cache (the tagging pass)')
    run(cli)
    for out_fact, args in derive:
        if derived(out_fact):
            stamp(f'4  {out_fact}: already derived')
            continue
        stamp(f'4  derive {out_fact}')
        extra = ['--lexicon-fraction', str(a.lexicon_fraction)] if (a.lexicon_fraction is not None and out_fact.endswith('lex')) else []
        run([py, 'scripts/data/derive_lexical_codes.py'] + args +
            ['--bpe-path', a.bpe_path, '--cache-dir', a.cache_dir, '--corpus', a.corpus,
             '--corpus-fraction', str(a.corpus_fraction), '--corpus-holdout', str(a.corpus_holdout),
             '--verify'] + extra)
    stamp('5  every fact a run asks for, in both splits')
    for split in ('validation', 'train'):
        run(check + ['--split', split, '--require-facts'] + want)
    stamp('CONDITION READY')
    print(f'  runs use:  {run_over}')
    print(f'  a matched family:  for f in mconst {a.source} {" ".join(f"{x}lex" for x in lexical)}; do '
          f'python -m morphlm.cli run --pipeline bpe --facts $f {run_over} '
          f'-o frozen.trainable=true -o frozen.causal_identity=true -o train.seed=0; done')


if __name__ == '__main__':
    main()
