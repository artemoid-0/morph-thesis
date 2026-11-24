#!/usr/bin/env python3
"""
rule_cost.py — what the endings rule costs in computation (the review's remark; journey §33.162).

WHY
---
The tokenizer with endings apart marks each word's ending before BPE encodes the text. The mark comes from
pymorphy3's readings of the word form, parsed once per distinct form and kept (`EndingRule`'s cache). So the
rule costs twice: once when the training text is marked, from a cold cache, before the tokenizer is trained
and its stream built; and in every encoding after, in training and in use. Encoding by plain BPE is the
`tokenizers` library's compiled code; marking is Python.

HOW
---
On the CPU of this machine, one process, nothing else running:
  1. the training text (the 0.2 fraction's training paragraphs, as the streams read them): a fresh rule;
     `mark_text` timed over every paragraph, the reading of the corpus not timed; the words it saw, the
     distinct forms it parsed (its cache) and the time per distinct form;
  2. the validation text: each paragraph encoded alone, by plain BPE 17k and by endings apart (the marked
     tokenizer, `load_tokenizer`) — the latter cold, with a fresh rule, then warm, its cache filled — and
     the two parts of endings apart's encoding apart: marking, and BPE on the marked text;
  3. both set against the 5M model's training run of 50k steps on this machine (its records'
     `run_times.py` median) and the endings-apart stream's recorded build.
--limit takes the first N training paragraphs, to time a sample first.

    python scripts/ops/rule_cost.py --limit 100000
    python scripts/ops/rule_cost.py
"""
import os
import sys
import time
import json
import glob
import argparse
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def paragraphs(split, fraction):
    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.data.stream import load_corpus
    cfg = apply_overrides(ExperimentConfig(), ['data.corpus=ukwiki', f'data.corpus_fraction={fraction}',
                                               'data.corpus_holdout=2000'])
    for row in load_corpus(cfg, split):
        t = row['text'].strip()
        if t:
            yield t


def timed_encode(enc, texts):
    t0 = time.perf_counter()
    n = 0
    for t in texts:
        n += len(enc(t).ids)
    return time.perf_counter() - t0, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=0, help='the first N training paragraphs only (0: all)')
    ap.add_argument('--fraction', default='0.2')
    a = ap.parse_args()
    from morphlm.data.endings import EndingRule, mark_text, ANY_WORD
    from morphlm.data.tokenization import load_tokenizer
    from morphlm.record import AuditRecord
    rec = AuditRecord('rule_cost', a, out_dir=os.path.join(ROOT, 'results', 'audits'))
    rec.add('cpu', {'cpus': os.cpu_count(), 'model': next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo')
                                                          if l.startswith('model name')), None)})

    # 1. the training text, from a cold cache
    rule = EndingRule()
    t_mark, n_par, n_bytes, n_words, n_cuts = 0.0, 0, 0, 0, 0
    t_all = time.perf_counter()
    for text in paragraphs('train', a.fraction):
        t0 = time.perf_counter()
        _, cuts = mark_text(text, rule)
        t_mark += time.perf_counter() - t0
        n_par += 1
        n_bytes += len(text.encode('utf-8'))
        n_words += sum(1 for _ in ANY_WORD.finditer(text))
        n_cuts += len(cuts)
        if a.limit and n_par >= a.limit:
            break
        if n_par % 500000 == 0:
            print(f'    {n_par:,} paragraphs, {n_bytes / 1e6:,.0f} MB, marking {t_mark:,.0f} s', flush=True)
    t_all = time.perf_counter() - t_all
    forms = len(rule._cache)
    train = {'paragraphs': n_par, 'bytes': n_bytes, 'words': n_words, 'cuts': n_cuts, 'distinct_forms': forms,
             'marking_seconds': t_mark, 'reading_and_marking_seconds': t_all,
             'marking_mb_per_s': n_bytes / 1e6 / t_mark, 'seconds_per_distinct_form': t_mark / max(forms, 1),
             'microseconds_per_word': 1e6 * t_mark / max(n_words, 1)}
    print(f'  1. training text: {n_par:,} paragraphs, {n_bytes / 1e6:,.1f} MB, {n_words:,} words, {forms:,} distinct '
          f'forms; marking {t_mark:,.1f} s ({train["marking_mb_per_s"]:.2f} MB/s), {1e3 * t_mark / max(forms, 1):.3f} ms '
          f'per distinct form, {train["microseconds_per_word"]:.2f} µs per word', flush=True)

    # 2. the validation text, encoded by each tokenizer
    val = list(paragraphs('validation', a.fraction))
    vb = sum(len(t.encode('utf-8')) for t in val)
    plain = load_tokenizer(os.path.join(ROOT, 'data', 'bpe_uk_17k.json'))
    t_plain, n_plain = timed_encode(plain.encode, val)
    ends = load_tokenizer(os.path.join(ROOT, 'data', 'bpeend_uk_17k.json'))          # a fresh rule: cold
    t_cold, n_ends = timed_encode(ends.encode, val)
    t_warm, _ = timed_encode(ends.encode, val)
    r2 = EndingRule()
    t0 = time.perf_counter()
    marked = [mark_text(t, r2)[0] for t in val]
    t_mark_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t in val:
        mark_text(t, r2)
    t_mark_warm = time.perf_counter() - t0
    t_inner, _ = timed_encode(ends.inner.encode, marked)
    enc = {'paragraphs': len(val), 'bytes': vb, 'tokens_plain': n_plain, 'tokens_endings': n_ends,
           'plain_seconds': t_plain, 'endings_cold_seconds': t_cold, 'endings_warm_seconds': t_warm,
           'marking_cold_seconds': t_mark_cold, 'marking_warm_seconds': t_mark_warm, 'bpe_on_marked_seconds': t_inner,
           'distinct_forms': len(r2._cache),
           'mb_per_s': {k: vb / 1e6 / v for k, v in (('plain', t_plain), ('endings_cold', t_cold), ('endings_warm', t_warm))},
           'slower_cold': t_cold / t_plain, 'slower_warm': t_warm / t_plain,
           'parse_ms_per_form': 1e3 * (t_mark_cold - t_mark_warm) / max(len(r2._cache), 1)}
    print(f'  2. validation text, {vb / 1e6:.2f} MB: plain BPE {enc["mb_per_s"]["plain"]:.2f} MB/s; endings apart cold '
          f'{enc["mb_per_s"]["endings_cold"]:.2f} MB/s ({enc["slower_cold"]:.1f}× slower), warm '
          f'{enc["mb_per_s"]["endings_warm"]:.2f} MB/s ({enc["slower_warm"]:.1f}× slower); of it marking cold '
          f'{t_mark_cold:.2f} s, warm {t_mark_warm:.2f} s, BPE on the marked text {t_inner:.2f} s; '
          f'{enc["parse_ms_per_form"]:.3f} ms to parse a form', flush=True)

    # 3. against one training run and the stream's build
    from morphlm import runs as R
    times = [r['timing']['wall_seconds'] / 60 for r in R.load(machine='thesis').values()
             if r.get('label') == 'bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2' and (r.get('timing') or {}).get('wall_seconds')]
    run_min = st.median(times) if times else None
    build = json.load(open(os.path.join(ROOT, 'data', 'cache', 'bpe_train_None_25577fdac5.json'))).get('build_seconds')
    against = {'training_run_minutes': run_min, 'training_runs_timed': len(times), 'stream_build_seconds': build,
               'marking_share_of_run': (t_mark / 60 / run_min) if run_min else None,
               'marking_share_of_build': (t_mark / build) if build else None}
    print(f'  3. one 5M training run of 50k here: {run_min:.1f} min (median of {len(times)}); marking the training text '
          f'is {100 * against["marking_share_of_run"]:.1f}% of it' + (f'; the stream\'s build took {build / 3600:.1f} h, '
          f'marking {100 * against["marking_share_of_build"]:.1f}% of it' if build else ''), flush=True)
    rec.add('training_text', train)
    rec.add('validation_encoding', enc)
    rec.add('against', against)
    rec.write()


if __name__ == '__main__':
    main()
