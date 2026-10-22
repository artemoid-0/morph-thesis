#!/usr/bin/env python3
"""
loss_by_kind.py — where a feature's gain sits, by what each predicted token is
(FINDINGS, "Case helps by predicting endings"; journey §33.69).

For every tokenizer of one condition, each saved model of the null and of each
feature is scored again on the validation stream, token by token, exactly as
training evaluated it: its own context facts, the same windows. Each target is
classified by the word rule the model's causal shift reads (`word_spans`) and by
the ending rule's cuts in the stream's own text (`mark_text` — the cuts the
endings-apart tokenizer makes):

  initial-uncut    the first token of a word the rule does not cut
  initial-ending   the first token of a cut word, holding the ending's start
                   (the whole word, or a first token that spans the cut)
  initial-stem     the first token of a cut word whose ending comes later
  cont-ending      a later token of a cut word holding the ending's start (at
                   or across the cut) — under endings apart, the ending itself
  cont-cut-other   any other later token of a cut word (mostly the stem's)
  cont-uncut       a later token of a word the rule does not cut
  non-word         punctuation, digits, line breaks

A feature's gain over its null is counted per kind in bits per byte — the
kinds add up to the whole gain — paired by seed within one machine (mean ±
standard error, how many seeds positive; FINDINGS, method point 11: a seed whose
runs ran on two machines is named and left out). Then per word class (cut, uncut, non-word: the same words under
every tokenizer, as the text is the same) and, against a reference tokenizer, the
difference between the tokenizers' gains placed in the words it comes from.

A check that could fail: every model's re-scored bits per byte must equal its
record's within --tol, or the tool stops before reporting anything.

Models are found beside their records. --machine second reads the second
machine's records and models wherever they are — its own runs in results/, its
twins' copies in results/audits/device_*/ with their models in
results/audits/device_twins_models/ (morphlm/runs.py); --machine thesis the
thesis machine's alone.

--previous (G-38; journey §33.73): what the sentence's case adds over the word
form's, placed by the word the feature describes. At the position that predicts a
target the model reads the previous word's case (the causal shift); the two
features differ only where that word's case in context is not its form's majority
case. Each target is put in one class by that described word — read through the
model's own boundary rule and shift, so the classes are exactly what each model
was given — using the stream's part-of-speech, case and word-form case codes:

  preposition, values differ  a preposition (ADP) whose case in context is not
                              its form's: the case the tagger read from the next
                              word — where its right-context leak sits (§33.21)
  preposition, values agree
  other word, values differ   a word seen in train whose case in context is not
                              its form's majority case (syncretism resolved from
                              either side)
  other word, form unseen     a word unseen in train: the form gives `none`
  inputs agree                every other target: the same input in both models

The first fact's gain minus the second's is reported per class, over all targets
and over the tokens holding an ending's start, paired by seed.

Any feature with a word-form variant goes the same way: --previous-facts names the
stream's part of speech, the feature in context (or from a causal tagger) and the
feature from the word form. With --unseen-lexicon the training lexicon decides
what is unseen (G-82, gender and number): `form unseen` is then a word the lexicon
lacks, and a seen word whose form's majority is `none` joins `values differ`;
without it, every word whose form gives `none` counts as unseen (§33.74). A check
that could fail: the form's codes must be that lexicon's at every position of the
validation stream.

    python scripts/analysis/loss_by_kind.py
    python scripts/analysis/loss_by_kind.py --budget _st20k --tokenizers bpe_uk_17k bpeend_uk_17k --facts ucase
    python scripts/analysis/loss_by_kind.py --previous
    python scripts/analysis/loss_by_kind.py --previous --fraction 0.2 --budget '' --tokenizers bpe_uk_17k
    python scripts/analysis/loss_by_kind.py --machine second --tokenizers bpe_uk_4k bpe_uk_17k --facts mconst@OFF0
    python scripts/analysis/loss_by_kind.py --machine second --tokenizers bpe_uk_17k bpeend_uk_17k \\
        --facts ugendercausal ugenderlex --previous --previous-facts pos ugendercausal ugenderlex --unseen-lexicon ugender
"""
import os
import sys
import time
import argparse
import dataclasses

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault('MORPHLM_NO_BUILD', '1')
from morphlm import runs as RUNS  # noqa: E402   (`runs` names a tokenizer's run ids here)

KINDS = ('initial-uncut', 'initial-ending', 'initial-stem', 'cont-ending', 'cont-cut-other',
         'cont-uncut', 'non-word')
WORD_CLASS = {'initial-uncut': 'uncut', 'cont-uncut': 'uncut', 'non-word': 'non-word',
              'initial-ending': 'cut', 'initial-stem': 'cut', 'cont-ending': 'cut', 'cont-cut-other': 'cut'}
CLASSES = ('cut', 'uncut', 'non-word')
ENDING = ('initial-ending', 'cont-ending')        # tokens holding the ending's start
STEM = ('initial-stem', 'cont-cut-other')         # the rest of a cut word
PREVIOUS = ('preposition, values differ', 'preposition, values agree', 'other word, values differ',
            'other word, form unseen', 'inputs agree')


def summary(xs):
    """morphlm.runs.summary, with a standard error of 0.0 at one seed."""
    return RUNS.summary(xs, se_one=0.0)


def show(s, scale=1.0):
    return RUNS.fmt(s, scale=scale)


def checkpoint(r):
    """(run id, results directory) for load_checkpoint: the model beside the run's record."""
    d = RUNS.model_dir(r)
    return os.path.basename(d), os.path.dirname(d)


def score(r):
    """Per-target loss (nats, float64) of one saved model on its validation stream, as training
    evaluated it (morphlm.runs.rescore)."""
    cfg, _, ds, nll, _, _ = RUNS.rescore(*checkpoint(r))
    return cfg, ds, nll.astype(np.float64)


def classify(ds, bpe_path, rule):
    """The kind of every scored target (index i is stream position i + 1), and counts."""
    from morphlm.features.lexical import tokenizer_tables, word_spans, _bytelevel_decoder
    from morphlm.data.tokenization import load_tokenizer
    from morphlm.data.endings import mark_text

    ids = np.asarray(ds.ids, dtype=np.int64)
    wid, _, starts = word_spans(ids, tokenizer_tables(bpe_path))
    tok = load_tokenizer(bpe_path, vocab_only=True)
    dec = _bytelevel_decoder()
    raw = [bytes(dec[ch] for ch in (tok.id_to_token(i) or '') if ch in dec) for i in range(tok.get_vocab_size())]
    off = np.zeros(len(ids) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(raw[i]) for i in ids])
    text = b''.join(raw[i] for i in ids).decode('utf-8')          # strict: the stream is a text (G-75)
    _, cuts = mark_text(text, rule)
    char_byte = np.zeros(len(text) + 1, dtype=np.int64)
    char_byte[1:] = np.cumsum([len(ch.encode('utf-8')) for ch in text])
    cut_b = np.asarray(sorted(int(char_byte[c]) for c in cuts), dtype=np.int64)
    s, e = off[:-1], off[1:]
    k = np.searchsorted(cut_b, s, side='left')                   # the first cut at or after a token's start
    holds = (k < len(cut_b)) & (cut_b[np.minimum(k, max(len(cut_b) - 1, 0))] < e) if len(cut_b) else np.zeros(len(ids), bool)
    if bool((holds & (wid < 0)).any()):
        raise RuntimeError(f'{int((holds & (wid < 0)).sum())} cuts fall outside words — the two rules disagree')
    n_words = int(wid.max()) + 1
    word_cut = np.zeros(n_words, dtype=bool)
    word_cut[wid[holds]] = True
    n_t = len(ds) * ds.seq_len
    p = np.arange(1, n_t + 1)                                    # the stream position of each target
    w, first, h = wid[p], starts[p], holds[p]
    cut = np.where(w >= 0, word_cut[np.maximum(w, 0)], False)
    kind = np.full(n_t, KINDS.index('non-word'), dtype=np.int8)
    kind[(w >= 0) & first & ~cut] = KINDS.index('initial-uncut')
    kind[(w >= 0) & first & cut & h] = KINDS.index('initial-ending')
    kind[(w >= 0) & first & cut & ~h] = KINDS.index('initial-stem')
    kind[(w >= 0) & ~first & cut & h] = KINDS.index('cont-ending')
    kind[(w >= 0) & ~first & cut & ~h] = KINDS.index('cont-cut-other')
    kind[(w >= 0) & ~first & ~cut] = KINDS.index('cont-uncut')
    counts = {'tokens': len(ids), 'targets': n_t, 'words': n_words, 'cut_words': int(word_cut.sum()),
              'cuts_in_text': len(cut_b), 'tokens_holding_a_cut': int(holds.sum()),
              # a check on the mapping: the share of cuts where a token starts — 100% by construction
              # under endings apart, ~9% under BPE 17k (journey §33.62)
              'cuts_at_a_token_start': int(np.isin(cut_b, s).sum())}
    return kind, counts


def previous_class(prep, ctx, lex, none_code, unseen=None):
    """
    The PREVIOUS class of each target from its described word's shifted inputs:
    `prep` marks a preposition, `ctx` and `lex` are the feature in context (or from a
    causal tagger) and from the word form. `unseen` marks a word the training lexicon
    lacks; without it, a word whose form gives `none` counts as unseen (§33.74).
    """
    differ = ctx != lex
    if unseen is None:
        unseen = lex == none_code
    cls = np.full(len(ctx), PREVIOUS.index('inputs agree'), dtype=np.int8)
    cls[prep & differ] = PREVIOUS.index('preposition, values differ')
    cls[prep & ~differ] = PREVIOUS.index('preposition, values agree')
    cls[~prep & differ & ~unseen] = PREVIOUS.index('other word, values differ')
    cls[~prep & differ & unseen] = PREVIOUS.index('other word, form unseen')
    # a check that could fail: the classes must be what the shifted inputs say —
    # the two inputs equal wherever the class says they agree
    agree = np.isin(cls, [PREVIOUS.index('preposition, values agree'), PREVIOUS.index('inputs agree')])
    if bool((ctx[agree] != lex[agree]).any()):
        raise RuntimeError('a target classed as agreeing has different inputs')
    return cls


def unseen_words(ids, form, cfg, lexicon_name, none_code):
    """
    Per stream position: 0 inside a word the training lexicon `lexicon_name` lacks,
    1 inside a word it holds, 2 outside words — by the lexicon's own word rule, the
    one the form's codes were made by. A check that could fail: `form`, the stream's
    codes of the feature from the word form, are this lexicon's at every position.
    """
    from morphlm.features.lexical import (load_lexicon, lexicon_condition, tokenizer_tables,
                                          word_spans, lexical_codes)
    d = cfg.data
    lex = load_lexicon(d.bpe_path, lexicon_name, d.cache_dir,
                       condition=lexicon_condition(d.corpus, d.corpus_fraction))
    tables = tokenizer_tables(d.bpe_path)
    form = np.asarray(form)
    mine = lexical_codes(ids, lex, tables, none_code)
    if not np.array_equal(mine, form):
        raise RuntimeError(f'the word-form codes are not lexicon {lexicon_name!r}\'s at '
                           f'{int((mine != form).sum()):,} of {len(ids):,} positions')
    wid, key, _ = word_spans(ids, tables)
    keys = lex[0]
    i = np.minimum(np.searchsorted(keys, key), len(keys) - 1)
    seen = keys[i] == key
    return np.where(wid < 0, 2, np.where(seen, 1, 0)).astype(np.int64)


def previous_classes(null_run, stream_sig, facts, lexicon_name=None):
    """
    The PREVIOUS class of every scored target: the word the feature describes at
    the position that predicts it, read through the model's own boundary rule and
    causal shift (`word_runs`, `causal_shift`), so each class is exactly what the
    models were given. `facts` are (part of speech, the feature in context or from
    a causal tagger, the feature from the word form), read from the same
    validation stream; `lexicon_name`, the training lexicon that decides what is
    unseen (--unseen-lexicon).
    """
    import torch
    from morphlm.training.metrics import load_checkpoint
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.windows import make_val_dataset
    from morphlm.modeling.embedding import word_runs, causal_shift
    from morphlm.features.blocks import build_blocks, POS_CLASSES

    model, cfg = load_checkpoint(*checkpoint(null_run))
    emb = model.embed.cpu()
    if not getattr(emb, 'causal_identity', False):
        raise RuntimeError(f'{null_run}: not a causal model — no shift to read')
    with_facts = lambda fs: dataclasses.replace(cfg, frozen=dataclasses.replace(cfg.frozen, facts=tuple(fs),
                                                                               shuffle_codes=False))

    def load(f):
        # one fact at a time: a derived fact sits in a codes file of its own beside the stream's
        # (`ugendercausal` with the causal tagger's, `ugenderlex` with its form's), never all three in one
        c = with_facts([f])
        d = make_val_dataset(build_pipeline(c, verbose=False), c, verbose=False)
        if list(d.fact_names) != [f]:
            raise RuntimeError(f'asked for {f}, the loader gave {list(d.fact_names)}')
        sig = (len(d.ids), int(np.asarray(d.ids[:4096], dtype=np.int64).sum()),
               int(np.asarray(d.ids[-4096:], dtype=np.int64).sum()))
        if sig != stream_sig:
            raise RuntimeError(f'the codes of {f} were read from another validation stream than the one scored')
        return d
    loaded = [load(f) for f in facts]
    ds = loaded[0]
    codes = [np.asarray(d.codes[0]) for d in loaded]
    data_cfg = with_facts(facts)
    L, n_w = ds.seq_len, len(ds)
    ids = torch.from_numpy(np.asarray(ds.ids[:n_w * L], dtype=np.int64).reshape(n_w, L))
    with torch.no_grad():
        src, start = word_runs(emb._word_starts(ids))
    blocks = build_blocks(data_cfg)
    shifted = {}
    for f, blk, c in zip(facts, blocks, codes):
        code = torch.from_numpy(c[:n_w * L].astype(np.int64).reshape(n_w, L))
        shifted[f] = causal_shift(code, src, start, blk.n_classes - 1).reshape(-1).numpy()
    pos, ctx, lex = (shifted[f] for f in facts)
    none_code = blocks[2].n_classes - 1
    prep = pos == POS_CLASSES.index('ADP')
    unseen = None
    if lexicon_name:
        word = unseen_words(np.asarray(ds.ids, dtype=np.int64), codes[2], data_cfg, lexicon_name, none_code)
        word = torch.from_numpy(word[:n_w * L].reshape(n_w, L))
        unseen = causal_shift(word, src, start, 2).reshape(-1).numpy() == 0
        if bool((unseen & (lex != none_code)).any()):
            raise RuntimeError('a word the lexicon lacks has a value from its form: the shifts disagree')
    cls = previous_class(prep, ctx, lex, none_code, unseen)
    del model
    differ = ctx != lex
    counts = {'targets': int(len(cls)), 'differ': int(differ.sum()), 'preposition': int(prep.sum()),
              'share': {c: float(np.mean(cls == i)) for i, c in enumerate(PREVIOUS)}}
    if unseen is not None:
        # against §33.74's rule: the targets after a seen word whose form's majority is `none`
        counts.update({'unseen_lexicon': lexicon_name, 'described_word_unseen': int(unseen.sum()),
                       'form_none_but_seen': int((~prep & differ & ~unseen & (lex == none_code)).sum())})
    return cls, counts


def main():
    from morphlm.data.endings import EndingRule
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    RUNS.add_results_flag(ap)
    ap.add_argument('--machine', choices=('thesis', 'second'), default=None,
                    help="one machine's records and models (default: every record under --results)")
    ap.add_argument('--corpus', default='ukwiki')
    ap.add_argument('--fraction', default='0.05')
    ap.add_argument('--budget', default='_st10k', help="the label's budget suffix")
    ap.add_argument('--tokenizers', nargs='+', default=['bpe_uk_17k', 'unigram_uk_17k', 'bpeend_uk_17k'])
    ap.add_argument('--reference', default='bpe_uk_17k')
    ap.add_argument('--null', default='mconst')
    ap.add_argument('--facts', nargs='+', default=['ucase', 'ucaselex'],
                    help="features to place against the null; 'mconst@OFF0' names the null switched off from step 0")
    ap.add_argument('--tol', type=float, default=1e-5, help='largest allowed |re-scored − recorded| bits per byte')
    ap.add_argument('--previous', action='store_true',
                    help='G-38: the first fact\'s gain minus the second\'s (or minus nothing, with one fact), by the word the feature describes')
    ap.add_argument('--previous-facts', nargs=3, default=['pos', 'ucase', 'ucaselex'],
                    metavar=('POS', 'CONTEXT', 'FORM'),
                    help='the stream\'s part of speech, the feature in context (or from a causal tagger) '
                         'and from the word form')
    ap.add_argument('--unseen-lexicon', default=None, metavar='NAME',
                    help="with --previous: 'form unseen' is a word absent from this training lexicon (e.g. ugender), "
                         "and a seen word whose form gives none joins 'values differ' (G-82)")
    ap.add_argument('--audit-dir', default=os.path.join(ROOT, 'results', 'audits'))
    a = ap.parse_args()
    t0 = time.time()

    R = RUNS.load(RUNS.results_of(a), machine=a.machine)
    corpus = a.corpus + a.fraction
    def lab(f, t):
        # 'mconst@OFF0' names a switched cell, the switch's suffix after CAUSAL in its label: the null
        # switched off from step 0 is the empty slot (G-78, journey §33.105)
        base, _, switch = f.partition('@')
        return f'bpe_{base}-LEARN-CAUSAL' + (f'-{switch}' if switch else '') + f'_{t}_{corpus}{a.budget}'
    rule = EndingRule()
    from morphlm.record import AuditRecord
    rec = AuditRecord('loss_by_kind', a, out_dir=a.audit_dir)
    rec.add('ending_rule', rule.fingerprint())

    out, checks, left_out = {}, [], {}

    def one_machine(*cells):
        """The seeds at which the cells' runs are on one machine; the others are named and noted."""
        pairs, left = RUNS.pair(R, *cells)
        if left and ' / '.join(cells) not in left_out:
            left_out[' / '.join(cells)] = {str(s): list(m) for s, m in left.items()}
            print(f'  {" / ".join(cells)}: {RUNS.left_out(left)}')
        return set(pairs)

    for t in a.tokenizers:
        runs = {f: {s: R[(l, s)] for (l, s) in R if l == lab(f, t)} for f in [a.null] + a.facts}
        seeds = sorted(runs[a.null])
        if not seeds:
            print(f'  {t}: no null runs under {lab(a.null, t)} — skipped')
            continue
        bpe_path = os.path.join(ROOT, 'data', f'{t}.json')
        kind, nll, counts, stream = None, {}, None, None
        for f, by_seed in runs.items():
            for s, r in sorted(by_seed.items()):
                rid = r['run_id']
                if not os.path.exists(os.path.join(RUNS.model_dir(r), 'model.pt')):
                    print(f'  {t} {f} seed {s}: no saved model ({rid}) — left out')
                    continue
                cfg, ds, x = score(r)
                ids_sig = (len(ds.ids), int(np.asarray(ds.ids[:4096], dtype=np.int64).sum()),
                           int(np.asarray(ds.ids[-4096:], dtype=np.int64).sum()))
                if stream is None:
                    stream = ids_sig
                    kind, counts = classify(ds, bpe_path, rule)
                elif ids_sig != stream:
                    raise RuntimeError(f'{rid}: another validation stream than its tokenizer\'s other models')
                scored = ds.num_bytes * len(x) / max(ds.num_tokens, 1)
                bpb = float(x.sum() / np.log(2) / scored)
                recd = R[(lab(f, t), s)]['bits_per_byte']
                checks.append({'run': rid, 'label': lab(f, t), 'seed': s, 'rescored': bpb, 'recorded': recd,
                               'diff': bpb - recd})
                if abs(bpb - recd) > a.tol:
                    raise SystemExit(f'  {rid} ({lab(f, t)}, seed {s}): re-scored {bpb:.6f} against recorded '
                                     f'{recd:.6f} — the scoring is not training\'s; nothing reported')
                nll.setdefault(f, {})[s] = (x, scored)
        print(f'  {t}: {counts["targets"]:,} targets, {counts["words"]:,} words, {counts["cut_words"]:,} cut; '
              f'cuts at a token start {100 * counts["cuts_at_a_token_start"] / max(counts["cuts_in_text"], 1):.1f}% '
              f'of {counts["cuts_in_text"]:,}; {sum(len(v) for v in nll.values())} models re-scored', flush=True)
        share = {k: float(np.mean(kind == i)) for i, k in enumerate(KINDS)}
        x0, scored0 = next(iter(nll[a.null].values()))
        null_bits = {k: float(x0[kind == i].mean() / np.log(2)) if (kind == i).any() else None
                     for i, k in enumerate(KINDS)}
        res = {'counts': counts, 'target_share': share, 'null_bits_per_target': null_bits, 'facts': {}}
        for f in a.facts:
            per_seed = {}
            one = one_machine(lab(a.null, t), lab(f, t))
            for s in seeds:
                if s not in nll.get(f, {}) or s not in nll[a.null] or s not in one:
                    continue
                (xn, sc), (xf, _) = nll[a.null][s], nll[f][s]
                g = (xn - xf) / np.log(2) / sc                       # bits per byte, per target
                per_seed[s] = {k: float(g[kind == i].sum()) for i, k in enumerate(KINDS)}
                per_seed[s]['total'] = float(g.sum())
            if not per_seed:
                continue
            ks = list(KINDS) + ['total']
            by_kind = {k: summary([v[k] for v in per_seed.values()]) for k in ks}
            by_class = {c: summary([sum(v[k] for k in KINDS if WORD_CLASS[k] == c) for v in per_seed.values()])
                        for c in CLASSES}
            ending = summary([sum(v[k] for k in ENDING) for v in per_seed.values()])
            stem = summary([sum(v[k] for k in STEM) for v in per_seed.values()])
            tok_end = sum(share[k] for k in ENDING)
            res['facts'][f] = {'per_seed': per_seed, 'by_kind': by_kind, 'by_class': by_class,
                               'ending_tokens': ending, 'stem_tokens': stem,
                               'ending_gain_share': (ending['mean'] / by_kind['total']['mean']) if by_kind['total']['mean'] else None,
                               'ending_target_share': tok_end}
        out[t] = res

        print(f'\n  {t} — {corpus}{a.budget}; gain over the null in bits per byte, paired by seed')
        print(f'  {"kind":16} {"targets":>8} {"null bits":>10} ' + ' '.join(f'{f:>26}' for f in a.facts if f in res['facts']))
        for i, k in enumerate(KINDS):
            print(f'  {k:16} {100 * share[k]:7.2f}% {null_bits[k] if null_bits[k] is not None else float("nan"):10.3f} '
                  + ' '.join(f'{show(res["facts"][f]["by_kind"][k]):>26}' for f in a.facts if f in res['facts']))
        print(f'  {"total":16} {"":8} {"":10} ' + ' '.join(f'{show(res["facts"][f]["by_kind"]["total"]):>26}'
                                                       for f in a.facts if f in res['facts']))
        for f in a.facts:
            if f in res['facts']:
                r = res['facts'][f]
                print(f'  {f}: ending tokens {show(r["ending_tokens"])} — {100 * r["ending_gain_share"]:.1f}% of the gain '
                      f'on {100 * r["ending_target_share"]:.1f}% of the targets; the rest of cut words {show(r["stem_tokens"])}; '
                      + '; '.join(f'{c} words {show(r["by_class"][c])}' for c in CLASSES))

        if a.previous and a.facts and a.facts[0] in nll:
            # one feature alone: its gain over the null, by the word it describes
            f1, f2 = (a.facts[0], a.facts[1]) if len(a.facts) > 1 and a.facts[1] in nll else (a.facts[0], a.null)
            pcls, pcounts = previous_classes(runs[a.null][seeds[0]], stream, a.previous_facts, a.unseen_lexicon)
            if len(pcls) != len(kind):
                raise RuntimeError('the previous-word classes and the target kinds differ in length')
            end_mask = np.isin(kind, [KINDS.index(k) for k in ENDING])
            per_seed = {}
            one = one_machine(*dict.fromkeys((lab(a.null, t), lab(f1, t), lab(f2, t))))
            for s in seeds:
                if any(s not in nll.get(f, {}) for f in (a.null, f1, f2)) or s not in one:
                    continue
                (xn, sc), (x1, _), (x2, _) = nll[a.null][s], nll[f1][s], nll[f2][s]
                g1, g2 = (xn - x1) / np.log(2) / sc, (xn - x2) / np.log(2) / sc
                d = g1 - g2
                cell = lambda m: {'first': float(g1[m].sum()), 'second': float(g2[m].sum()),
                                  'difference': float(d[m].sum()), 'difference_on_endings': float(d[m & end_mask].sum())}
                per_seed[s] = {c: cell(pcls == i) for i, c in enumerate(PREVIOUS)}
                per_seed[s]['total'] = cell(np.ones(len(d), dtype=bool))
            rows = list(PREVIOUS) + ['total']
            keys = ('first', 'second', 'difference', 'difference_on_endings')
            summ = {c: {k: summary([v[c][k] for v in per_seed.values()]) for k in keys} for c in rows}
            res['previous'] = {'facts': [f1, f2], 'codes': list(a.previous_facts), 'unseen_lexicon': a.unseen_lexicon,
                               'counts': pcounts,
                               'ending_target_share': {c: (float(end_mask[pcls == i].mean()) if (pcls == i).any() else None)
                                                       for i, c in enumerate(PREVIOUS)},
                               'per_seed': per_seed, 'summary': summ}
            print(f'\n  {t} — what {f1} adds over {f2}, by the word the feature describes; bits per byte, paired by seed')
            print(f'  {"described word":28} {"targets":>8} {f1 + " gain":>24} {f2 + " gain":>24} '
                  f'{f1 + " − " + f2:>24} {"of it on ending tokens":>24}')
            for i, c in enumerate(rows):
                share = 100 * pcounts['share'][c] if c != 'total' else 100.0
                z = summ[c]
                print(f'  {c:28} {share:7.2f}% {show(z["first"]):>24} {show(z["second"]):>24} '
                      f'{show(z["difference"]):>24} {show(z["difference_on_endings"]):>24}')
            if a.unseen_lexicon:
                n = pcounts['targets']
                print(f'  by lexicon {a.unseen_lexicon}: the described word unseen at '
                      f'{100 * pcounts["described_word_unseen"] / n:.2f}% of targets; a seen word whose form gives '
                      f'none, the inputs differing, at {100 * pcounts["form_none_but_seen"] / n:.2f}%')

    if a.reference in out:
        print(f'\n  against {a.reference}: this tokenizer\'s gain minus the reference\'s, by word class, paired by seed')
        vs = {}
        for t, res in out.items():
            if t == a.reference:
                continue
            vs[t] = {}
            for f in a.facts:
                if f not in res['facts'] or f not in out[a.reference]['facts']:
                    continue
                mine, ref = res['facts'][f]['per_seed'], out[a.reference]['facts'][f]['per_seed']
                one = one_machine(lab(a.null, t), lab(f, t), lab(a.null, a.reference), lab(f, a.reference))
                common = sorted(set(mine) & set(ref) & one)
                cls = lambda v, c: sum(v[k] for k in KINDS if WORD_CLASS[k] == c)
                d = {c: summary([cls(mine[s], c) - cls(ref[s], c) for s in common]) for c in CLASSES}
                d['total'] = summary([mine[s]['total'] - ref[s]['total'] for s in common])
                d['cut_share'] = (d['cut']['mean'] / d['total']['mean']) if d['total'] and d['total']['mean'] else None
                vs[t][f] = d
                print(f'  {t:16} {f:10} total {show(d["total"])}   '
                      + '   '.join(f'{c} {show(d[c])}' for c in CLASSES)
                      + (f'   — cut words {100 * d["cut_share"]:.0f}% of it' if d['cut_share'] is not None else ''))
        rec.add('against_reference', vs)
    worst = max(abs(c['diff']) for c in checks) if checks else None
    print(f'\n  {len(checks)} models re-scored; largest |re-scored − recorded| {worst:.2e} bits per byte; '
          f'{time.time() - t0:.0f} s')
    rec.add('tokenizers', out)
    rec.add('rescore_checks', checks)
    if left_out:
        rec.add('left_out', left_out)
    rec.write()


if __name__ == '__main__':
    main()
