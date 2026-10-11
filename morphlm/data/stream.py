"""
stream.py — tokenize once, cache, memory-map.

TWO CACHE FILES, KEYED SEPARATELY
---------------------------------
The token ids for a pipeline do not depend on which frozen facts you requested;
only the annotation codes do. Keying both on one combined string meant the
baseline, `--facts dep` and `--facts pos` each triggered a full independent
spaCy pass over the corpus — three passes where one suffices.

    ids   key = pipeline|split|max_articles                  (historical format)
    codes key = pipeline|split|max_articles|ctx:<gran>:<facts>

The ids key is byte-identical to the pre-refactor scheme, which is a
REQUIREMENT: existing caches must stay valid.

    bpe|train|None       -> 99bc1462ce   (bpe_train_None_99bc1462ce)
    bpe|validation|None  -> e483bcd459

FREE-RIDE ANNOTATION
--------------------
The morpheme pipelines call spacy.load(), which enables tagger, parser AND ner
by default — they have always parsed dependencies and discarded them. So for
those pipelines we compute ALL context facts on whichever pass happens first,
even when the current config asked for none. The marginal cost is reading two
attributes off an already-parsed token; the saving is every later pass.

BPE is different: it uses no spaCy at all, so annotating it is a genuinely new
cost and happens only when the config asks for it.

STALENESS
---------
The codes key includes the dependency granularity and the fact list, so
changing either yields a different file. Changing lr or seed changes neither
key, so training knobs can never invalidate a multi-hour pass.
"""

from __future__ import annotations

import os
import json
import hashlib

import numpy as np

CACHE_DIR = 'data/cache'

#: Context facts a pipeline can actually produce, BY PIPELINE.
#:
#: This must be per-pipeline, not global. The morpheme pipelines' encode() sets
#: dep/pos/entity_label from the spaCy parse but never sets mroot/mstruct/mrole
#: — those come from the WordAnalyser, which only the BPE pipeline carries.
#: A global list meant `facts_to_build('morph')` asked for three facts morph
#: cannot produce, which (a) changed the morph codes cache key, silently
#: invalidating a 4-hour cache, and (b) would have spent another 4 hours
#: producing three all-`none` rows.
PIPELINE_CONTEXT_FACTS = {
    'morph': ('dep', 'pos', 'ectx'),
    'det':   ('dep', 'pos', 'ectx'),
    'bpe':   ('dep', 'pos', 'ectx', 'mstruct', 'mrole', 'mroot', 'mhash', 'mlemma',
              'minfl', 'magr', 'mfeat'),
}

#: The free ride on a NON-English corpus (docs/reference/ukrainian_pipeline.md): the facts
#: spaCy's morphologizer and parser give in any language. No MorphoLEX facts —
#: its analyser is English — and no entity context, whose vocabulary is too.
UD_FREE_RIDE = ('dep', 'pos', 'ucase', 'ugender', 'unumber', 'uperson',
                'utense', 'uaspect', 'uanimacy')
#: Retained for callers that want the union (English — the UD facts are
#: UD_FREE_RIDE, built only for a non-English corpus).
ALL_CONTEXT_FACTS = ('dep', 'pos', 'ectx', 'mstruct', 'mrole', 'mroot',
                     'mhash', 'mlemma', 'minfl', 'magr', 'mfeat')

#: The constant nulls: one class for every word (`derive_lexical_codes.py
#: --transform const`). Their codes must say so — `rules` in the codes file —
#: before a run reads them: derived before G-73, a constant gave every word
#: unseen in train the class `none`, a mark it met only on non-words in
#: training (journey §33.56).
CONST_FACTS = ('mconst', 'mconst16', 'mconst32', 'mconst64', 'mzero', 'mzeronopos', 'mzeroposonly')
#: ...of which the zero slots never read their codes (the slot is held at zero,
#: embedding._factor), so only the others are guarded
ZERO_SLOT_FACTS = ('mzero', 'mzeronopos', 'mzeroposonly')


def load_corpus(cfg, split):
    """
    Rows with a 'text' field, for any corpus in `corpora.CORPORA`.

    wikitext103 — exactly the historical call, so every English stream is
    unchanged; it refuses a fraction, since its cache key carries none. Every
    other corpus is served by its spec (`corpora.rows`): ukwiki — Ukrainian
    Wikipedia (the `wikimedia/wikipedia` dump, which has only a train split) —
    has its articles assigned to a split by a hash of their title (1 in
    `corpus_holdout` to validation, 1 to test, the rest to train — 2000 puts
    validation near WikiText-103's size), train sub-sampled by a second hash to
    `corpus_fraction`, and each article served as its paragraphs, like
    WikiText's lines.
    """
    from .corpora import get_corpus, rows
    corpus = getattr(cfg.data, 'corpus', 'wikitext103')
    spec = get_corpus(corpus)
    frac = float(getattr(cfg.data, 'corpus_fraction', 1.0))
    if corpus == 'wikitext103' and frac != 1.0:
        raise ValueError('wikitext103 takes no corpus_fraction: its cache key carries '
                         'none, so a sub-sample would share the full stream\'s files. '
                         'Sub-sampled English needs a corpus of its own (corpora.py)')
    from datasets import load_dataset
    if corpus == 'wikitext103':
        return load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    hold = int(getattr(cfg.data, 'corpus_holdout', 2000))
    return rows(spec, split, frac, hold, load_dataset)

#: Pipelines that already run spaCy for their own reasons.
SPACY_NATIVE = ('morph', 'det')


def cache_stem(pipeline_name, split, max_articles, feature_spec=''):
    """
    Stable filename stem. With an empty feature_spec this reproduces the
    historical format byte for byte — a REQUIREMENT, not a nicety.
    """
    raw = f'{pipeline_name}|{split}|{max_articles}'
    if feature_spec:
        raw += f'|{feature_spec}'
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f'{pipeline_name}_{split}_{max_articles}_{h}'


def codes_spec(fact_names, dep_granularity):
    """Canonical spec string for a set of context facts."""
    if not fact_names:
        return ''
    return f'ctx:{dep_granularity}:{"+".join(sorted(fact_names))}'


def codes_owner(basename, prefix):
    """
    What a codes file's name says about where it came from (G-72).

    A cache build writes `<prefix><hash>.codes.json`, keyed on the facts it
    built; a derivation (derive_lexical_codes, derive_prefix_codes) writes
    `<ids stem>_<name>.codes.json`, named for the ids stream it was derived for.
    Returns ('build', None), ('derived', <ids stem>), or (None, None) for a name
    of neither form (no tool writes one).
    """
    name = basename[:-len('.codes.json')] if basename.endswith('.codes.json') else basename
    if not name.startswith(prefix):
        return None, None
    rest = name[len(prefix):]
    h, tail = rest[:10], rest[10:]
    if len(h) != 10 or any(c not in '0123456789abcdef' for c in h):
        return None, None
    if not tail:
        return 'build', None
    if tail.startswith('_'):
        return 'derived', prefix + h
    return None, None


def stream_codes_files(cache_dir, pipeline, split, max_articles, ids_stem, num_tokens):
    """
    Every codes file that describes the ids stream `ids_stem`, sorted by name:
    the build files with its token count, and the files derived FOR IT.
    Returns [(json path, npy path, meta)].

    CODES ARE TIED TO A TOKEN STREAM. A codes array is one value per token of a
    specific ids array, so a matching fact list is not enough: codes built for
    a different surface configuration describe different tokens entirely. The
    token count is the guard against that (found by scripts/data/cache_index.py
    --plan, which reported a whitespace-off cell as able to reuse the
    punctuation-off codes).

    THE TOKEN COUNT IS NOT ENOUGH EITHER (G-72). A corpus's validation split is
    the same text at every corpus fraction, so ukwiki's 0.01 and 0.2 validation
    streams are the same tokens under two stems. Built codes — spaCy's tags of
    those tokens — are then the same too, and may be shared. Derived codes are
    not: a word-form lexicon comes from the fraction's own train split, so the
    null and the lexical facts differ in which words are unseen, and a kept row
    is whatever its source file held. Matched by token count and read in name
    order, every ukwiki run of the null from 1 October, at any fraction, was
    scored on the 0.01 fraction's validation codes. A derived file is read only
    by the stream it was derived for. A name of neither form keeps the
    token-count rule.
    """
    import glob as _glob
    prefix = f'{pipeline}_{split}_{max_articles}_'
    out = []
    for cand in sorted(_glob.glob(os.path.join(cache_dir, prefix + '*.codes.json'))):
        kind, owner = codes_owner(os.path.basename(cand), prefix)
        if kind == 'derived' and owner != ids_stem:
            continue
        try:
            with open(cand) as f:
                cm = json.load(f)
        except Exception:
            continue
        if cm.get('num_tokens') != num_tokens:
            continue
        npy = cand[:-len('.json')] + '.npy'
        if os.path.exists(npy):
            out.append((cand, npy, cm))
    return out


def facts_to_build(cfg):
    """
    Which context facts this pass should COMPUTE (not necessarily use).

    spaCy-native pipelines compute everything, because the parse is already
    paid for.

    BPE now does the same AS SOON AS ANY context fact is requested. Its pass is
    genuinely new — spaCy plus, for morphological facts, MorphoLex — so it is
    expensive, but it is expensive ONCE. Computing only the requested fact would
    mean a fresh multi-hour pass for every new BPE experiment, and each pass
    would write a different codes file. Building all of them together is the
    single decision that keeps the Level-2 experiment programme to one cache
    build.
    """
    from ..registry import context_facts
    requested = context_facts(cfg.frozen.facts)
    available = PIPELINE_CONTEXT_FACTS.get(cfg.data.pipeline, ())
    if cfg.data.pipeline == 'bpe' and getattr(cfg.data, 'corpus', 'wikitext103') != 'wikitext103':
        available = UD_FREE_RIDE
    if cfg.data.pipeline in SPACY_NATIVE:
        return available
    return available if requested else ()


def cache_blocks(cfg, names):
    """
    Fact blocks that produce codes AS THE CACHE STORES THEM.

    The cache is built once, at the full inventory, and every run reads it;
    coarsening (`frozen.root_classes`) is applied at MODEL time, by
    `embedding.fold_codes`, from that full range. But a coarsened block's own
    `index()` also folds. Built from a coarsened run's config, a cache would
    store FOLDED codes under a key that does not mention coarsening — poisoning
    it for every later run — and new text scored that way (BLiMP) would be
    folded twice, turning `none` into class 0. `root_classes` is the one config
    field that changes code values without entering the cache key
    (`dep_granularity` does too, but it is recorded and checked on reuse).
    Everything that produces codes for a model therefore builds its blocks
    here, with coarsening off.
    """
    import dataclasses
    from ..registry import get_fact
    full = dataclasses.replace(
        cfg, frozen=dataclasses.replace(cfg.frozen, root_classes=0))
    return [get_fact(name)(full) for name in names]


def _atomic_save_npy(path, arr):
    """
    Write an array so that no reader ever sees a partial file.

    Cached arrays are memory-mapped by running jobs (np.load mmap_mode='r').
    np.save on the live path truncates the file first, and a job reading it at
    that moment can die with SIGBUS or read zeros. Writing to a temporary name
    and renaming is atomic on one filesystem: a job that already mapped the old
    file keeps the old inode intact, and any later reader sees the new file
    complete. This is what makes it safe to build a cache while training.
    """
    tmp = path[:-4] + '.tmp.npy' if path.endswith('.npy') else path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def build_stream(pipeline, cfg, split='train', max_articles=None,
                 verbose=True):
    """
    Tokenize a WikiText-103 split and cache it.

    Returns a dict with:
        ids        np.memmap int32  [num_tokens]
        codes      np.ndarray int16 [n_requested_facts, num_tokens] or None
        fact_names list[str]        requested facts, in code-row order
        num_tokens, num_bytes, num_chars, num_articles
    """
    from ..registry import context_facts, get_fact

    cache_dir = cfg.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    requested = list(context_facts(cfg.frozen.facts))
    build_facts = list(facts_to_build(cfg))
    gran = cfg.frozen.dep_granularity

    # surface options change the ID STREAM ITSELF, so they belong in the ids
    # key, not just the codes key. Omitting them would mean a stale hit and
    # training on silently wrong data.
    sfc = cfg.surface_spec()
    ids_stem = cache_stem(cfg.data.pipeline, split, max_articles, sfc)
    ids_npy = os.path.join(cache_dir, ids_stem + '.npy')
    ids_meta = os.path.join(cache_dir, ids_stem + '.json')

    c_spec = codes_spec(build_facts, gran)
    if sfc:
        c_spec = f'{sfc}|{c_spec}' if c_spec else sfc
    codes_stem = cache_stem(cfg.data.pipeline, split, max_articles, c_spec)
    codes_npy = os.path.join(cache_dir, codes_stem + '.codes.npy')
    codes_meta_path = os.path.join(cache_dir, codes_stem + '.codes.json')

    # If the exact key misses, look for ANY codes file for this
    # (pipeline, split, cap) that already CONTAINS the requested facts at the
    # right granularity. The stored fact list is what matters, not the
    # filename: without this, extending the fact inventory silently orphans
    # every cache built before the change, and a 4-hour rebuild starts with no
    # warning. Filename-keyed lookup is brittle; content-keyed lookup is not.
    ids_meta_tokens = None
    if os.path.exists(ids_meta):
        try:
            with open(ids_meta) as _f:
                ids_meta_tokens = json.load(_f).get('num_tokens')
        except Exception:
            pass

    # Does the file at the exact key actually SATISFY this request?
    #
    # `codes_stem` is keyed on facts_to_build — the free-ride list a pipeline
    # produces in one pass — not on what the cell asked for. So a cell can ask
    # for a fact that is NOT on that list (`mprefix` is derived from the token
    # stream afterwards, never built by the pipeline) and the stem still
    # resolves to an existing file that predates it. The file exists, so reuse
    # was skipped, and the run died reporting that its own cache lacked a fact
    # it was never going to contain.
    #
    # Reuse must therefore be attempted whenever the exact file does not
    # satisfy the request, not merely when it is absent.
    exact_ok = False
    if os.path.exists(codes_npy) and os.path.exists(codes_meta_path):
        try:
            with open(codes_meta_path) as f:
                _cm = json.load(f)
            exact_ok = set(requested).issubset(set(_cm.get('fact_names', [])))
        except Exception:
            exact_ok = False

    # reuse is only possible when the ids stream this cell needs already
    # exists; otherwise there is nothing for borrowed codes to align to
    if requested and ids_meta_tokens is not None and not exact_ok:
        # only files that describe THIS stream: the same token count, and a
        # derived file only if it was derived for this ids stem (G-72)
        for cand, npy, cm in stream_codes_files(cache_dir, cfg.data.pipeline, split,
                                                max_articles, ids_stem, ids_meta_tokens):
            if cm.get('dep_granularity') != gran:
                continue
            if not set(requested).issubset(set(cm.get('fact_names', []))):
                continue
            if verbose:
                print(f'  [codes reuse] {os.path.basename(npy)} already holds '
                      f'{requested}')
            codes_npy, codes_meta_path = npy, cand
            break

    have_ids = os.path.exists(ids_npy) and os.path.exists(ids_meta)
    have_codes = os.path.exists(codes_npy) and os.path.exists(codes_meta_path)
    need_codes = bool(requested)

    # ── fast path: everything needed is already on disk ───────────────────────
    if have_ids and (have_codes or not need_codes):
        if verbose:
            extra = f' + codes({",".join(requested)})' if need_codes else ''
            print(f'  [cache hit] {ids_stem}{extra}')
        with open(ids_meta) as f:
            meta = json.load(f)
        meta['ids'] = np.load(ids_npy, mmap_mode='r')
        meta['codes'] = None
        meta['fact_names'] = requested
        if need_codes:
            with open(codes_meta_path) as f:
                cmeta = json.load(f)
            stored = cmeta['fact_names']
            missing = [f for f in requested if f not in stored]
            if missing:
                raise RuntimeError(
                    f'cached codes {codes_stem} lack {missing}; delete it and '
                    f'rebuild with: cli cache --pipeline {cfg.data.pipeline} '
                    f'--facts {",".join(requested)}')
            full = np.load(codes_npy, mmap_mode='r')
            if full.shape[1] != meta['ids'].shape[0]:
                raise RuntimeError(
                    f'code/id length mismatch on LOAD: codes {codes_stem} has '
                    f'{full.shape[1]:,} positions, ids {ids_stem} has '
                    f'{meta["ids"].shape[0]:,}. These describe different token '
                    f'streams and must not be paired. The build path has always '
                    f'asserted this; the load path did not.')
            rules = cmeta.get('rules') or {}
            stale = [f for f in requested if f in CONST_FACTS and f not in ZERO_SLOT_FACTS
                     and rules.get(f) != 'every word']
            if stale:
                raise RuntimeError(
                    f'the codes of {stale} in {os.path.basename(codes_npy)} were derived before '
                    f'G-73: a word unseen in train is marked `none` there, not given the '
                    f'constant class. Derive them again — '
                    f'python scripts/data/rederive_constants.py (journey §33.56)')
            rows = [stored.index(f) for f in requested]
            meta['codes'] = np.asarray(full[rows])
            meta['codes_file'] = os.path.basename(codes_npy)
            meta['codes_rules'] = {f: rules[f] for f in requested if f in rules}
        return meta

    # ── slow path: run the pipeline ───────────────────────────────────────────
    if os.environ.get('MORPHLM_NO_BUILD'):
        raise RuntimeError(
            f'cache MISS for {ids_stem} (+codes {requested}) and '
            f'MORPHLM_NO_BUILD is set. Analysis scripts set this so that a '
            f'missing cache fails immediately instead of silently starting a '
            f'multi-hour spaCy rebuild. Build the cache deliberately with '
            f'`cli cache` if that is what you want.')
    if verbose:
        what = 'ids' + (f' + codes({",".join(build_facts)})'
                        if build_facts else '')
        print(f'  [cache miss] tokenizing {split} through '
              f'{cfg.data.pipeline} -> {what}')
        if build_facts and cfg.data.pipeline in SPACY_NATIVE:
            print('    (context facts ride along free: this pipeline already '
                  'parses with spaCy)')

    from ..progress import Progress, fmt_duration
    if build_facts:
        # tagging a corpus with another language's model gives silent garbage
        from .corpora import check_tagger
        check_tagger(getattr(cfg.data, 'corpus', 'wikitext103'), cfg.data.spacy_model)
    ds = load_corpus(cfg, split)

    blocks = cache_blocks(cfg, build_facts)

    # Compact accumulators: int32 ids and int16 codes, 4 and 2 bytes per
    # entry. Python lists cost 8 bytes per entry for the pointer alone — about
    # 10.6 GB for eleven facts over 120M tokens, before the final array — and
    # the build is meant to be able to run beside a training job.
    from array import array
    all_ids = array('i')
    all_codes = [array('h') for _ in blocks]
    num_bytes = 0
    num_chars = 0
    processed = 0
    seen_rows = 0

    # Progress is tracked over ROWS CONSUMED, not articles kept: blank lines and
    # headers are skipped, so `processed` advances unevenly and would give a
    # misleading ETA. When max_articles caps the run, that cap is the total.
    try:
        total_rows = len(ds)
    except TypeError:
        total_rows = None
    prog = Progress(total=(max_articles if max_articles is not None
                           else total_rows))

    for row in ds:
        seen_rows += 1
        if max_articles is not None and processed >= max_articles:
            break
        text = row['text'].strip()
        if text:
            text += row.get('sep', '')          # a Ukrainian paragraph's newline
        # `startswith(' =')` never fires — the text is already stripped — so
        # WikiText's headings (' = Gameplay = ') stay in the stream. Every
        # English stream depends on this as it stands (G-37).
        if not text or text.startswith(' ='):
            continue

        if blocks:
            tokens = pipeline.encode(text)
            if not tokens:
                continue
            all_ids.extend(t['token_id'] for t in tokens)
            for bi, block in enumerate(blocks):
                all_codes[bi].extend(block.code(t) for t in tokens)
        else:
            ids = pipeline.encode_ids(text)
            if not ids:
                continue
            all_ids.extend(ids)

        num_bytes += len(text.encode('utf-8'))
        num_chars += len(text)
        processed += 1
        if verbose and processed % 2000 == 0:
            done = processed if max_articles is not None else seen_rows
            print(f'    {processed:,} articles, {len(all_ids):,} tokens '
                  f'| {prog.status(done)}')

    arr = np.frombuffer(all_ids, dtype=np.int32)

    # Never rewrite an ids file that already exists. The ids depend only on the
    # tokenizer and the text, not on which facts are built, so a rebuild for
    # new facts reproduces them exactly — and rewriting would pull the file out
    # from under any job that has it mapped. Verify instead: a mismatch means
    # the tokenizer or corpus changed under an unchanged cache key, which must
    # stop everything.
    if os.path.exists(ids_npy):
        existing = np.load(ids_npy, mmap_mode='r')
        if existing.shape != arr.shape or not np.array_equal(existing, arr):
            raise RuntimeError(
                f'{ids_stem}: rebuilt token ids differ from the cached ones '
                f'({existing.shape[0]:,} vs {arr.shape[0]:,} tokens). The cache '
                f'key did not change but the stream did — refusing to continue.')
        if verbose:
            print(f'  ids unchanged ({arr.shape[0]:,} tokens) — not rewritten')
    else:
        _atomic_save_npy(ids_npy, arr)

    meta = {
        'pipeline': cfg.data.pipeline,
        'split': split,
        'max_articles': max_articles,
        'num_tokens': int(arr.shape[0]),
        'num_bytes': int(num_bytes),
        'num_chars': int(num_chars),
        'num_articles': processed,
        # the tagging pass's wall clock: the next tokenizer on this text costs
        # about the same, and prepare_condition.py estimates from it
        'build_seconds': round(float(prog.elapsed), 1),
    }
    rate = getattr(pipeline, 'alignment_rate', lambda: None)()
    if rate is not None:
        meta['alignment_rate'] = float(rate)
        if verbose:
            print(f'  context alignment rate: {rate:.3%}')
    if not os.path.exists(ids_meta):
        _atomic_write_json(ids_meta, meta)

    codes_arr = None
    if blocks:
        codes_arr = np.empty((len(all_codes), arr.shape[0]), dtype=np.int16)
        for bi, col in enumerate(all_codes):
            if len(col) != arr.shape[0]:
                raise RuntimeError(f'code/id length mismatch for '
                                   f'{blocks[bi].name}: {len(col)} vs '
                                   f'{arr.shape[0]}')
            codes_arr[bi] = np.frombuffer(col, dtype=np.int16)
        # data first, metadata last: a concurrent run's content-based reuse
        # only considers a codes file once its json exists
        _atomic_save_npy(codes_npy, codes_arr)
        _atomic_write_json(codes_meta_path,
                           {'fact_names': build_facts,
                            'dep_granularity': gran,
                            'num_tokens': int(arr.shape[0])})

    if verbose:
        print(f'  [cached] {ids_stem}: {meta["num_tokens"]:,} tokens, '
              f'{meta["num_bytes"]:,} bytes '
              f'| built in {fmt_duration(prog.elapsed)}')
        if blocks:
            print(f'  [cached] codes {build_facts} -> {codes_stem}')

    meta['ids'] = np.load(ids_npy, mmap_mode='r')
    meta['fact_names'] = requested
    if requested and codes_arr is not None:
        rows = [build_facts.index(f) for f in requested]
        meta['codes'] = codes_arr[rows]
        meta['codes_file'] = os.path.basename(codes_npy)
    else:
        meta['codes'] = None
    return meta
