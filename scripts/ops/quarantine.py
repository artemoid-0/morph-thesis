#!/usr/bin/env python3
"""
quarantine.py — move invalid runs out of results/, keeping the record.

WHY THIS EXISTS, AND WHY A GLOB CANNOT DO IT
--------------------------------------------
Result directories are named by run_id — an md5 of the config — so a label such
as `bpe_mhash-LEARN-k512_bpe_17k` appears only INSIDE `result.json`. A shell
pattern on directory names matches nothing. Invalid runs have to be identified
by reading their records.

There is a second trap. A code fix that changes behaviour without changing the
config leaves the run_id unchanged, so re-running a fixed cell OVERWRITES the
invalid one of the same seed — but leaves invalid runs of OTHER seeds in place.
`export_results.py` would then average a valid run with an invalid one and call
the result two seeds of one cell. That is worse than having no result, because
it looks like replication.

So invalid runs must be removed before anything is re-run, and fixed runs must
carry a marker proving which semantics produced them.

RULES
-----
Each rule names a defect, the config that exposes it, and the marker a run must
carry to prove it predates the defect's fix. A run is quarantined when it
matches a rule and lacks the marker.

Nothing is deleted. Quarantined runs move to `results_invalid/<rule>/` beside a
manifest recording why, so the history stays auditable.

    python scripts/ops/quarantine.py            # dry run: show what would move
    python scripts/ops/quarantine.py --apply    # move them
"""

import os
import sys
import json
import shutil
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _causal(cfg):
    # Only a run WITH features can have a causal-shift defect: chains pass the
    # causal flags to every run, plain baselines included, and a baseline has
    # no block to shift — nor, therefore, the marker (journey §33.44).
    f = cfg.get('frozen') or {}
    return bool(f.get('causal_identity')) and bool(f.get('facts'))


def _has_causal_v2(result):
    for b in result.get('frozen_blocks') or []:
        if b.get('causal_identity') and (b.get('causal_semantics') or 0) >= 3:
            return True
    return False


def _coarsened(cfg):
    return int((cfg.get('frozen') or {}).get('root_classes') or 0) > 0


def _has_coarsen_v2(result):
    for b in result.get('frozen_blocks') or []:
        if b.get('coarsen') and (b.get('coarsen_semantics') or 0) >= 2:
            return True
    return False


def _sweep_duplicate(cfg):
    """The September sweep's default-rate English baseline at a 10k budget.

    Its configuration predates frozen.causal_identity, so its run id differs from
    a current-code run of the same cell — while its label and seed are the same.
    """
    d, t, f = cfg.get('data') or {}, cfg.get('train') or {}, cfg.get('frozen') or {}
    return (d.get('pipeline') == 'bpe' and str(d.get('bpe_path', '')).endswith('bpe_17k.json')
            and not f.get('facts') and t.get('max_steps') == 10000
            and abs(float(t.get('lr') or 0) - 3e-4) < 1e-12 and 'causal_identity' not in f)


#: the constant nulls whose codes reach the model (morphlm.data.stream.CONST_FACTS
#: without the zero slots, which hold their dims at zero whatever the code)
CONST_FACTS = ('mconst', 'mconst16', 'mconst32', 'mconst64')


def _ukrainian_constant(cfg):
    """A Ukrainian run with a constant null among its facts (G-72, G-73)."""
    d, f = cfg.get('data') or {}, cfg.get('frozen') or {}
    return d.get('corpus') == 'ukwiki' and any(x in CONST_FACTS for x in (f.get('facts') or []))


def _scored_on_constant(result):
    """Its validation codes declare every constant fact one class for every word."""
    rules = ((result.get('data') or {}).get('validation') or {}).get('codes_rules') or {}
    facts = [x for x in (result.get('facts') or []) if x in CONST_FACTS]
    return bool(facts) and all(rules.get(x) == 'every word' for x in facts)


#: name, human reason, predicate(config) -> affected, predicate(result) -> fixed
RULES = [
    ('coarsen_clamp',
     'Coarsening folded codes at cache-build time only; the model clamped '
     'instead, collapsing up to 99.8% of words into a single class. Fixed by '
     'folding at model time (embedding.fold_codes, COARSEN_SEMANTICS = 2).',
     _coarsened,
     _has_coarsen_v2),
    ('causal_partial',
     'Causal runs whose word boundaries were INFERRED from the codes rather '
     'than read from the tokenizer. Two defects: v1 shifted only root-family '
     'blocks, leaving mstruct leaking; v2 shifted everything but inferred '
     'boundaries, which collapse when the reference fact varies within a word '
     '(mprefix). A v2 checkpoint is also unreproducible — the boundary rule '
     'lives in the forward pass, so re-scoring it under v3 code evaluates it '
     'under boundaries it never trained with. Requires CAUSAL_SEMANTICS >= 3.',
     _causal,
     _has_causal_v2),
    ('const_unseen',
     'Ukrainian runs with a constant null, scored on validation codes that gave '
     'every word unseen in train the class `none` — a mark the null met only on '
     'non-words in training (G-73), and, for 20 of them, on another corpus '
     'size\'s lexicon (G-72). Their saved models re-scored on the constant: '
     'better by 0.0005 (151M tokens) to 0.0039 (1.5M); English nulls by at most '
     '0.0002, so English runs are kept; a zero slot never reads its codes, so a '
     'zero slot beside case is kept. Requires validation codes that declare the '
     'constant (journey §33.55-§33.56).',
     _ukrainian_constant,
     _scored_on_constant),
    ('sweep_duplicate',
     'The September learning-rate sweep ran the English 10k-budget baseline at '
     'the default rate (cfbb4ce93538); the current-code run of the same cell '
     '(12e3d62efc84) reproduced it bit for bit at all 21 evaluations. Same label, '
     'same seed: kept once, so the cell counts seed 0 once (journey §33.43, G-69).',
     _sweep_duplicate,
     lambda result: False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--dest', default='results_invalid')
    ap.add_argument('--apply', action='store_true',
                    help='actually move runs (default is a dry run)')
    args = ap.parse_args()

    found = []
    for entry in sorted(os.listdir(args.results_dir)):
        d = os.path.join(args.results_dir, entry)
        rp, cp = os.path.join(d, 'result.json'), os.path.join(d, 'config.json')
        if not (os.path.isdir(d) and os.path.exists(rp) and os.path.exists(cp)):
            continue
        try:
            r, c = json.load(open(rp)), json.load(open(cp))
        except Exception as e:
            print(f'  {entry}: unreadable ({e}) — left in place')
            continue
        for name, reason, affected, fixed in RULES:
            if affected(c) and not fixed(r):
                found.append((entry, r.get('label', '?'), r.get('seed'),
                              r.get('bits_per_byte'), name, reason))
                break

    print()
    print('=' * 84)
    print(f'QUARANTINE — {args.results_dir}/  ({"APPLY" if args.apply else "dry run"})')
    print('=' * 84)
    if not found:
        print('  nothing to quarantine')
        return

    by_rule = {}
    for f in found:
        by_rule.setdefault(f[4], []).append(f)
    for name, rows in by_rule.items():
        reason = rows[0][5]
        print(f'\n  rule: {name}  ({len(rows)} run(s))')
        print(f'  {reason}')
        print()
        for rid, label, seed, bpb, _, _ in sorted(rows, key=lambda x: x[1]):
            b = f'{bpb:.4f}' if bpb is not None else '-'
            print(f'    {rid}  {label:36} seed{seed}  {b}')

    if not args.apply:
        print('\n  DRY RUN — pass --apply to move these')
        return

    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for name, rows in by_rule.items():
        dest = os.path.join(args.dest, name)
        os.makedirs(dest, exist_ok=True)
        manifest = os.path.join(dest, 'MANIFEST.json')
        prior = json.load(open(manifest)) if os.path.exists(manifest) else []
        for rid, label, seed, bpb, _, reason in rows:
            shutil.move(os.path.join(args.results_dir, rid),
                        os.path.join(dest, rid))
            prior.append({'run_id': rid, 'label': label, 'seed': seed,
                          'bits_per_byte': bpb, 'rule': name,
                          'reason': reason, 'quarantined_utc': stamp})
        json.dump(prior, open(manifest, 'w'), indent=2)
        print(f'\n  moved {len(rows)} run(s) -> {dest}/  (manifest updated)')


if __name__ == '__main__':
    main()
