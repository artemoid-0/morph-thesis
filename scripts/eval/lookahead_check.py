#!/usr/bin/env python3
"""
lookahead_check.py — how much of the word-identity gain is within-word lookahead?

THE CONCERN
-----------
`mhash`, `mlemma` and `mroot` give every BPE fragment of a word the code of the
WHOLE word. For `reconfiguring` split as recon / fig / uring, the input at
`recon` already carries hash("reconfiguring") — which identifies the word and so
reveals that `fig` and `uring` come next. During generation that code cannot
exist, because the word has not been written yet.

That is lookahead: within a word rather than across the sentence, but the same
kind of oracle the `dep` feature was flagged for. Journey §26.5 and §29
described these features as having "no lookahead" because the root depends only
on the word. That was asserted by reasoning and never checked against what "the
word" means once BPE splits it.

WHERE THE LEAK CAN AND CANNOT ACT
---------------------------------
The loss at each position is for predicting the NEXT token. Classify that target:

    WORD_INITIAL   a new word begins. It is predicted from a word that is
                   already complete, so its code is legitimate context.
    CONTINUATION   the next fragment of the SAME word. It is predicted from an
                   input carrying the whole word's code — the leak can act here.
    OTHER          punctuation, digits, line breaks.

The leak can only help CONTINUATION targets. So if the gain is real word
identity, it should appear on WORD_INITIAL targets too; if it is mostly leak, it
will pile up on CONTINUATION targets out of proportion to how many there are.

HOW TO READ THE RESULT — stated before running
----------------------------------------------
The decisive figure is each class's SHARE OF THE TOTAL GAIN against its SHARE
OF THE TOKENS.

    continuation gain share  ~  its token share   -> the leak is not the driver
    continuation gain share  >> its token share   -> the leak carries the gain

Continuation targets are also intrinsically easier (the word prefix constrains
them), so a moderate imbalance alone is not proof. A large one is. The
definitive test is a causal version of the feature; this check says whether that
test is urgent.

    python scripts/eval/lookahead_check.py --baseline a7175857fdf6 \\
        --treatment f97f597841fa 67346ed11499 56239ccfaf84
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('MORPHLM_NO_BUILD', '1')

import numpy as np

WORD_INITIAL, CONTINUATION, OTHER = 0, 1, 2
NAMES = {WORD_INITIAL: 'word-initial', CONTINUATION: 'continuation',
         OTHER: 'other'}


def _is_alpha(s):
    t = s.lstrip('\u0120')                  # ByteLevel marks a space as U+0120
    return bool(t) and t.isalpha()


def _starts_word(s):
    return s.startswith('\u0120')


def classify(prev_ids, tgt_ids, id_to_str):
    """
    Label each target. CONTINUATION requires BOTH the target and the token
    before it to be alphabetic with no word boundary between them, so that
    `(hello` — where `hello` lacks the space marker only because of the
    bracket — is not mistaken for the inside of a word.
    """
    out = np.full(len(tgt_ids), OTHER, dtype=np.int8)
    for i, (p, t) in enumerate(zip(prev_ids, tgt_ids)):
        ts, ps = id_to_str(int(t)), id_to_str(int(p))
        if not _is_alpha(ts):
            continue                                   # OTHER
        if not _starts_word(ts) and _is_alpha(ps):
            out[i] = CONTINUATION                      # inside a word: leak can act
        else:
            # a new word: either marked by a space, or following punctuation /
            # the start of a line. Either way the previous input carries no code
            # for THIS word, so the leak cannot reach it.
            out[i] = WORD_INITIAL
    return out


def per_token(run_id, results_dir, batches):
    """Per-token NLL, with the model fed ONLY its own context facts."""
    import torch
    import dataclasses
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import get_device
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.windows import make_val_dataset
    from morphlm.registry import context_facts

    model, cfg = load_checkpoint(run_id, results_dir)
    device = get_device()
    model = model.to(device).eval()
    facts = list(context_facts(cfg.frozen.facts))

    data_cfg = dataclasses.replace(
        cfg, frozen=dataclasses.replace(cfg.frozen, facts=tuple(facts),
                                        shuffle_codes=False))
    pipe = build_pipeline(data_cfg, verbose=False)
    ds = make_val_dataset(pipe, data_cfg, verbose=False)
    loaded = list(getattr(ds, 'fact_names', []) or [])
    if facts and loaded != facts:
        raise RuntimeError(f'{run_id}: model reads {facts}, loader gave '
                           f'{loaded} — refusing to score')
    loader = DataLoader(ds, batch_size=cfg.train.batch_size)

    nll, xs, ys = [], [], []
    with torch.no_grad():
        for bi, (x, y, c) in enumerate(loader):
            if batches and bi >= batches:
                break
            cc = c.to(device) if facts else None
            logits, _ = model(x.to(device), None, codes=cc)
            v = logits.size(-1)
            l = F.cross_entropy(logits.reshape(-1, v), y.to(device).reshape(-1),
                                reduction='none')
            nll.append(l.cpu().numpy())
            xs.append(x.reshape(-1).numpy())
            ys.append(y.reshape(-1).numpy())
    return (cfg, pipe, np.concatenate(nll), np.concatenate(xs),
            np.concatenate(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--treatment', nargs='+', required=True)
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--batches', type=int, default=0,
                    help='validation batches (0 = all)')
    args = ap.parse_args()

    from morphlm.record import AuditRecord
    rec = AuditRecord('lookahead_check', args)

    print(f'scoring baseline {args.baseline}...')
    bcfg, pipe, bnll, bx, by = per_token(args.baseline, args.results_dir,
                                         args.batches)
    tok = getattr(pipe, 'tokenizer', None)
    if tok is None:
        raise RuntimeError('pipeline exposes no tokenizer; cannot classify')
    cache = {}

    def id_to_str(i):
        if i not in cache:
            cache[i] = tok.id_to_token(i) or ''
        return cache[i]

    cls = classify(bx, by, id_to_str)
    n = len(cls)
    share_tok = {k: float(np.mean(cls == k)) for k in NAMES}
    rec.add('token_shares', {NAMES[k]: share_tok[k] for k in NAMES})

    print()
    print('=' * 84)
    print('WITHIN-WORD LOOKAHEAD CHECK')
    print('=' * 84)
    print(f'  {n:,} validation targets')
    for k in NAMES:
        print(f'    {NAMES[k]:14} {100*share_tok[k]:6.2f}% of targets')

    for rid in args.treatment:
        print(f'\nscoring {rid}...')
        tcfg, _, tnll, tx, ty = per_token(rid, args.results_dir, args.batches)
        if not (np.array_equal(tx, bx) and np.array_equal(ty, by)):
            raise RuntimeError(f'{rid}: token stream differs from the baseline '
                               f'— the per-token comparison would be invalid')
        gain = bnll - tnll                        # nats saved per target
        total = float(gain.sum())
        print()
        print(f'  {tcfg.label()}')
        print(f'    overall gain {gain.mean():.4f} nats/token')
        print(f'    {"class":14} {"tokens":>8} {"base NLL":>9} '
              f'{"mean gain":>10} {"share of gain":>14} '
              f'{"gain/token ratio":>17}')
        block = {}
        for k in NAMES:
            m = cls == k
            g = float(gain[m].mean()) if m.any() else 0.0
            sg = float(gain[m].sum()) / total if total else 0.0
            ratio = sg / share_tok[k] if share_tok[k] else float('nan')
            # BASELINE DIFFICULTY. A ratio above 1 does not by itself mean a
            # leak: continuation targets exist only for words BPE split into
            # several fragments, and BPE splits RARE words. They are therefore
            # a harder, rarer sample, and any feature supplying context should
            # help them more. This column shows how much harder, so the ratio
            # can be read against it instead of against an assumed 1.0.
            bd = float(bnll[m].mean()) if m.any() else 0.0
            print(f'    {NAMES[k]:14} {100*share_tok[k]:7.2f}% {bd:9.4f} '
                  f'{g:10.4f} {100*sg:13.1f}% {ratio:17.2f}')
            block[NAMES[k]] = {'mean_gain': g, 'gain_share': sg,
                               'token_share': share_tok[k],
                               'baseline_nll': bd, 'ratio': ratio}
        rec.add(rid, {'label': tcfg.label(), 'overall_gain': float(gain.mean()),
                      'by_class': block})

    print()
    print('  READING THIS')
    print('    Compare the ratio against the BASE NLL column, not against 1.0.')
    print('    Continuation targets exist only where BPE split a word, and BPE')
    print('    splits rare words, so they are a harder sample and any context')
    print('    feature should help them more. A ratio near 6-7 alongside the')
    print('    whole-word codes is leakage; a smaller ratio in proportion to')
    print('    the extra difficulty is not.')
    print('    A feature whose codes depend only on EARLIER tokens cannot leak')
    print('    at all, whatever its ratio - that is a property of the')
    print('    construction, not of this table.')
    rec.note('Continuation targets are intrinsically easier, so a moderate '
             'imbalance is not proof of leakage. The definitive test is a '
             'causal feature carrying only information available at generation '
             'time.')
    rec.write()


if __name__ == '__main__':
    main()
