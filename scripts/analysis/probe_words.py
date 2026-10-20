#!/usr/bin/env python3
"""
probe_words.py — what a language model builds for itself: a word's grammar in
context, read from its hidden states (journey §33.83, §33.84).

WHY
---
The other side of the title (journey §33.84): does a model build context-aware
morphological representations itself, and does a tokenizer that keeps endings
apart make them better? It reads models whose input carries no morphology (the
same-shape nulls), so whatever a probe finds, the model built.

HOW
---
For every run of each cell (a label; the final checkpoint, model.pt), over the
whole validation split:

  * each layer's hidden state at the LAST token of every word — the whole word
    seen, nothing after it. Word ends come from the model's own tables, as its
    causal shift finds them; a window's last token is left out (its word may
    go on in the next window);
  * a linear probe (morphlm.eval.probing.fit_linear, as probe.py fits it)
    trained on the words of even windows and tested on those of odd windows;
  * labels: the tagger's UD features in context (ucase, unumber, ugender, ...),
    as the caches hold them; words a feature does not apply to (its 'none'
    class) are left out of that feature's probe.

Case is also scored on three kinds of word, by the most frequent case of the
word's form in training (`ucaselex`, from the run's own condition):

  agrees   the form's majority case is its case in context;
  context  it is another case — the form alone is wrong here, only context
           can be right;
  unseen   the form was not seen in train (`ucaselex` marks it none, G-76).

The form's own accuracy — its majority case taken as the answer — is printed
beside the probe's, and each subset's majority class (from the training words)
as its floor.

Cells are paired by seed with a reference cell: seeds share initial weights
between tokenizers of one vocabulary size, and both read the same text. A pair is
counted only when both runs ran on one machine (FINDINGS, method point 11); a seed
whose runs ran on two is named and left out. --machine reads one machine's runs.

`--after N` reads each word's labels from the state at the last token of the
unit N later (a word or punctuation run, in the same window). The tagger's
labels saw the words after each word; the model's state at the word has not,
and at N = 1 it has seen one unit more (journey §33.84).

    python scripts/analysis/probe_words.py --labels 'bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2' 'bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2'
    python scripts/analysis/probe_words.py --labels ... --facts ucase unumber ugender --threads 8 --windows 200
    python scripts/analysis/probe_words.py --labels ... --after 1

Writes results/audits/probe_words_<UTC timestamp>.json.
"""
import os
import sys
import argparse
import dataclasses

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from morphlm import runs                         # noqa: E402
from morphlm.record import AuditRecord           # noqa: E402


def word_ends(model, ids):
    """The last token of every unit (word or punctuation run), from the model's
    own word-start tables — the boundaries its causal shift uses."""
    import torch
    from morphlm.modeling.embedding import HybridEmbedding
    emb = model.embed
    if getattr(emb, 'tok_alpha', None) is None:
        sys.exit('the model has no word tables (not a causal-feature model)')
    st_ = HybridEmbedding._word_starts(emb, ids.to(emb.tok_alpha.device)).cpu()
    last = torch.zeros_like(st_)
    last[:, :-1] = st_[:, 1:]
    return last          # the window's last position stays False


def load_runs(patterns, roots, machine=None):
    """[(label, seed, run id, record)] — the valid runs matching the patterns (runs.select) whose
    model is saved, sorted."""
    out = []
    for r in runs.select(patterns, roots, machine=machine):
        if os.path.exists(os.path.join(runs.model_dir(r), 'model.pt')):
            out.append((r['label'], r['seed'], r['run_id'], r))
    return sorted(out, key=lambda x: x[:3])


def probe_run(r, facts, split_fact, max_windows, epochs, dev, after=0):
    import torch
    from morphlm.eval.probing import fit_linear
    from morphlm.training.metrics import load_checkpoint
    from morphlm.pipelines.registry import build_pipeline
    from morphlm.data.windows import make_val_dataset
    from morphlm.features.blocks import build_blocks

    rid, d = r['run_id'], runs.model_dir(r)
    model, cfg = load_checkpoint(os.path.basename(d), os.path.dirname(d))
    model.eval()
    ds_in = make_val_dataset(build_pipeline(cfg, verbose=False), cfg, verbose=False)
    lab = {}
    for f in list(facts) + ([split_fact] if split_fact else []):
        cf = dataclasses.replace(cfg, frozen=dataclasses.replace(
            cfg.frozen, facts=(f,), shuffle_codes=False, role='input'))
        ds = make_val_dataset(build_pipeline(cf, verbose=False), cf, verbose=False)
        lab[f] = (ds, build_blocks(cf)[0].n_classes)
    n_win = len(ds_in) if not max_windows else min(len(ds_in), max_windows)
    X, C = [], []
    Y = {f: [] for f in lab}
    for i in range(n_win):
        x, _, c = ds_in[i]
        X.append(x); C.append(c)
        for f, (ds, _) in lab.items():
            x2, _, c2 = ds[i]
            assert torch.equal(x, x2), f'{rid} window {i}: inputs and {f} labels disagree on tokens'
            Y[f].append(c2[0])
    X, C = torch.stack(X), torch.stack(C)
    Y = {f: torch.stack(v).long() for f, v in Y.items()}
    last = word_ends(model, X)
    sent = {f: n - 1 for f, (_, n) in lab.items()}
    # every unit's last token, in window then position order; a word's labels are
    # read from the state `after` units later in the same window (0: at the word)
    wi, pi = last.nonzero(as_tuple=True)
    src = torch.arange(len(wi))
    if after:
        src = src[src + after < len(wi)]
        src = src[wi[src] == wi[src + after]]
    tgt = src + after
    valid = torch.stack([Y[f][wi[src], pi[src]] != sent[f] for f in facts]).any(0)
    src, tgt = src[valid], tgt[valid]
    hid = None
    with torch.no_grad():
        for b in range(0, n_win, 16):
            _, _, hidden = model(X[b:b + 16].to(dev), codes=C[b:b + 16].to(dev), return_hidden=True)
            m = last[b:b + 16]
            hs = [h.float().cpu()[m] for h in hidden]
            hid = [[h] for h in hs] if hid is None else [p + [h] for p, h in zip(hid, hs)]
    hid = [torch.cat(h)[tgt] for h in hid]
    ys = {f: Y[f][wi[src], pi[src]] for f in Y}
    is_train = (wi[src] % 2 == 0)
    rows = []
    for f in facts:
        n_cls, s = lab[f][1], sent[f]
        ok = ys[f] != s
        tr, te = ok & is_train, ok & ~is_train
        ytr, yte = ys[f][tr], ys[f][te]
        groups = {'all': torch.ones_like(yte, dtype=torch.bool)}
        form = None
        if split_fact and f == 'ucase':
            lx = ys[split_fact]
            lx_te, lx_tr = lx[te], lx[tr]
            ls = sent[split_fact]
            groups['agrees'] = lx_te == yte
            groups['context'] = (lx_te != yte) & (lx_te != ls)
            groups['unseen'] = lx_te == ls
            form = (lx_te == yte).float().mean().item()
            tr_groups = {'all': torch.ones_like(ytr, dtype=torch.bool), 'agrees': lx_tr == ytr,
                         'context': (lx_tr != ytr) & (lx_tr != ls), 'unseen': lx_tr == ls}
        else:
            tr_groups = {'all': torch.ones_like(ytr, dtype=torch.bool)}
        floor = {}
        for g, m in groups.items():
            yt = ytr[tr_groups[g]]
            maj = torch.bincount(yt, minlength=n_cls).argmax() if len(yt) else torch.tensor(0)
            floor[g] = (yte[m] == maj).float().mean().item() if m.any() else None
        for layer, h in enumerate(hid):
            pred = fit_linear(h[tr], ytr, n_cls, epochs, dev)(h[te])
            acc = {g: ((pred[m] == yte[m]).float().mean().item() if m.any() else None) for g, m in groups.items()}
            rows.append({'fact': f, 'layer': layer, 'accuracy': acc, 'n': {g: int(m.sum()) for g, m in groups.items()},
                         'floor': floor, 'form': form, 'train': int(tr.sum())})
    return n_win, rows


def ms(v):
    """(mean, se — 0.0 at one seed, seeds positive, seeds) over the values that are not None."""
    s = runs.summary(v, se_one=0.0)
    return (s['mean'], s['se'], s['positive'], s['n']) if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', nargs='+', required=True, help='cells (glob patterns over labels)')
    ap.add_argument('--reference', default=None, help='the cell the others are paired with (default: the first)')
    ap.add_argument('--facts', nargs='+', default=['ucase', 'unumber', 'ugender'])
    ap.add_argument('--split-fact', default='ucaselex', help="the form's fact that splits case's words ('' for none)")
    runs.add_results_flag(ap)
    ap.add_argument('--machine', choices=('thesis', 'second'), default=None, help="one machine's runs only")
    ap.add_argument('--windows', type=int, default=0, help='validation windows per run (0: all)')
    ap.add_argument('--after', type=int, default=0,
                    help="read a word's labels from the state this many units later (0: at the word's last token)")
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--audit-dir', default=os.path.join(ROOT, 'results', 'audits'))
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    found = load_runs(args.labels, runs.results_of(args), args.machine)
    if not found:
        sys.exit('no runs match ' + ' '.join(args.labels))
    cells = sorted({r[0] for r in found}, key=lambda c: [i for i, p in enumerate(args.labels)
                                                         if __import__('fnmatch').fnmatch(c, p)][0])
    ref = args.reference or cells[0]
    print(f'  {len(found)} run(s) in {len(cells)} cell(s); facts {args.facts}; device {args.device}')
    out = {'runs': []}
    for label, seed, rid, r in found:
        n_win, rows = probe_run(r, args.facts, args.split_fact or None,
                                args.windows, args.epochs, args.device, args.after)
        out['runs'].append({'run': rid, 'label': label, 'seed': seed, 'machine': runs.machine(r) or runs.gpu(r),
                            'windows': n_win, 'rows': rows})
        line = []
        for f in args.facts:
            r = [x for x in rows if x['fact'] == f]
            b = max(r, key=lambda x: x['accuracy']['all'])
            extra = (f" (context {b['accuracy']['context']:.3f}, form {b['form']:.3f})"
                     if b.get('form') is not None else '')
            line.append(f"{f} L{b['layer']} {b['accuracy']['all']:.3f}{extra}")
        print(f'  {rid}  {label}  seed {seed}: ' + ';  '.join(line))

    # summary: per cell and fact, the best layer per run (by accuracy on all words) and its subsets
    def best(run, f):
        r = [x for x in run['rows'] if x['fact'] == f]
        return max(r, key=lambda x: x['accuracy']['all'])
    by = {}
    for run in out['runs']:
        by.setdefault(run['label'], {})[run['seed']] = run
    summ = {}
    print()
    record = {rid: r for _, _, rid, r in found}
    R = {(cell, s): record[run['run']] for cell, seeds in by.items() for s, run in seeds.items()}
    paired = {}
    for cell in cells:
        if cell != ref and ref in by:
            pairs, left = runs.pair(R, ref, cell)
            paired[cell] = set(pairs)
            if left:
                print(f'  {cell} against {ref}: {runs.left_out(left)}')
    for f in args.facts:
        groups = ['all', 'agrees', 'context', 'unseen'] if (f == 'ucase' and args.split_fact) else ['all']
        print(f'  {f} — accuracy at each run\'s best layer, mean over seeds; paired by seed with {ref} (points, mean ± se, positive/seeds)')
        layers = sorted({x['layer'] for x in out['runs'][0]['rows'] if x['fact'] == f})
        for cell in cells:
            seeds = by.get(cell, {})
            line = f'    {cell[:60]:60}'
            cs = {}
            for g in groups:
                vals = [best(seeds[s], f)['accuracy'][g] for s in seeds]
                m = ms(vals)
                cs[g] = {'mean': m[0] if m else None}
                d = ''
                if cell != ref and ref in by:
                    pv = [100 * (best(seeds[s], f)['accuracy'][g] - best(by[ref][s], f)['accuracy'][g])
                          for s in seeds if s in by[ref] and s in paired[cell]
                          and best(seeds[s], f)['accuracy'][g] is not None]
                    pm = ms(pv)
                    if pm:
                        cs[g]['paired'] = {'mean': pm[0], 'se': pm[1], 'positive': pm[2], 'n': pm[3], 'values': pv}
                        d = f' {pm[0]:+.1f}±{pm[1]:.1f} {pm[2]}/{pm[3]}'
                line += f'  {g} {100 * m[0]:.1f}%{d}' if m else f'  {g} —'
            b0 = best(next(iter(seeds.values())), f)
            if b0.get('form') is not None:
                fm = ms([best(seeds[s], f)['form'] for s in seeds])
                cfl = ms([best(seeds[s], f)['floor']['context'] for s in seeds])
                cs['form'] = fm[0]; cs['context_floor'] = cfl[0]
                line += f'   [form {100 * fm[0]:.1f}%, context floor {100 * cfl[0]:.1f}%]'
            per_layer = {l: ms([next(x for x in seeds[s]['rows'] if x['fact'] == f and x['layer'] == l)['accuracy']['all']
                                for s in seeds])[0] for l in layers}
            cs['by_layer'] = per_layer
            print(line)
            print('        by layer: ' + '  '.join(f'L{l} {100 * v:.1f}' for l, v in per_layer.items()))
            summ.setdefault(f, {})[cell] = cs
        print()
    rec = AuditRecord('probe_words', args, out_dir=args.audit_dir)
    rec.add('runs', out['runs']).add('summary', summ)
    print(f'  [record] {rec.write(verbose=False)}')


if __name__ == '__main__':
    main()
