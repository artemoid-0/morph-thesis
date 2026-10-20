#!/usr/bin/env python3
"""
coverage_economics.py — is the Q1 deficit COVERAGE or SEGMENTATION?

THE HYPOTHESIS
--------------
Category bit accounting showed character fallback consumes 24.8% of morph's
bits for 14.1% of the corpus, at 2.654 bits/byte against a corpus average of
1.51. The OOV audit showed 13.07% of words fall through, 77.5% of them
capitalised names the frequency-gated entity vocabulary never admitted.

That suggests morph loses to BPE not because morpheme segmentation is worse,
but because the pipeline cannot ANALYSE ~13% of words and spells them out one
character at a time. If true, a better vocabulary would close the gap.

WHAT THIS SCRIPT MEASURES
-------------------------
Part 1 (no model, fast) — TOKENIZATION ECONOMICS
    For every word, how many tokens does each pipeline spend, split by whether
    morph could analyse the word. BPE has no 'fallback' as such — it always has
    a subword decomposition — so the comparable quantity is tokens-per-word.

Part 2 (needs checkpoints) — RESTRICTED BITS-PER-BYTE
    Score both models on the same text, but accumulate bits separately for
    bytes belonging to words morph handled cleanly vs words it fell back on.
    Both models see the full text as context; only the SCORED positions are
    partitioned. This is a conditional comparison, not a headline number.

READING IT HONESTLY
-------------------
Restricting to covered words is SELECTION ON THE OUTCOME: it removes exactly
the cases where morph loses. It cannot support 'our tokenizer is better'. It
CAN support 'on words within its vocabulary morph is competitive, and the
deficit is attributable to coverage' — which is a diagnosis, and which predicts
that expanding the vocabulary closes the gap.

The prediction is falsifiable: if morph still loses on covered words, coverage
is NOT the explanation and a bigger vocabulary will not help.

    python scripts/audits/coverage_economics.py --lines 400
    python scripts/audits/coverage_economics.py --lines 400 \
        --morph-run <id> --bpe-run <id>
"""

import os
import sys
import math
import argparse
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np

FALLBACK_CATS = ('character', 'byte')
STANDALONE = ('space', 'punct', 'digit')


def group_words(tokens):
    """[(word, [tokens])] in text order, excluding standalone surface tokens."""
    groups = []
    cur_w, cur = None, []
    for t in tokens:
        if t.get('category') in STANDALONE:
            continue
        w = t.get('word')
        if w is None:
            continue
        if w != cur_w and cur:
            groups.append((cur_w, cur))
            cur = []
        cur_w = w
        cur.append(t)
    if cur:
        groups.append((cur_w, cur))
    return groups


def word_spans(text, groups):
    """
    Locate each word in the original text with a moving cursor.
    Words are emitted in text order, so a forward search is sufficient and
    avoids re-running spaCy just to recover offsets.
    """
    spans = []
    cursor = 0
    low = text.lower()
    for w, toks in groups:
        s = str(w)
        i = low.find(s.lower(), cursor)
        if i < 0:
            i = low.find(s.lower())
        if i < 0:
            spans.append(None)
            continue
        spans.append((i, i + len(s)))
        cursor = i + len(s)
    return spans


def build_masks(text, groups, spans):
    """
    Partition the ORIGINAL text into three regions:
        0 = punctuation, digits, and whitespace not adjacent to a word
        1 = word morph analysed cleanly  (+ its leading whitespace)
        2 = word morph fell back on      (+ its leading whitespace)

    THE LEADING-WHITESPACE RULE IS LOAD-BEARING.
    ByteLevel BPE folds the space INTO the following token (`Ġcat` spans
    " cat"), while morph emits a separate <sp> before the word. If the space
    belonged to region 0, BPE's word tokens would be charged to region 0 while
    morph's were charged to region 1, and the two models would be measured on
    different partitions. Attaching the space to the word it precedes makes the
    regions symmetric for both encoding schemes.
    """
    mask = np.zeros(len(text), dtype=np.int8)
    for (w, toks), sp in zip(groups, spans):
        if sp is None:
            continue
        cls = 2 if any(t['category'] in FALLBACK_CATS for t in toks) else 1
        a = sp[0]
        # extend backward over whitespace not already claimed
        while a > 0 and text[a - 1].isspace() and mask[a - 1] == 0:
            a -= 1
        mask[a:sp[1]] = cls
    return mask


def region_bytes(text, mask):
    """UTF-8 bytes per region — the denominator, shared by both models."""
    out = Counter()
    for ch, m in zip(text, mask):
        out[int(m)] += len(ch.encode('utf-8'))
    return out


def span_weights(mask, s, e):
    """
    How a token's bits should be split across regions.

    A token's span can straddle regions (rare once whitespace is attached to
    its word, but possible), so bits are distributed in proportion to the
    token's characters in each region rather than assigned by a single offset.
    Assigning by start offset is what produced BPE at 0.14 BPB on words: every
    Ġ-prefixed token starts on a space.
    """
    if e <= s or s >= len(mask):
        return {0: 1.0}
    seg = mask[s:min(e, len(mask))]
    if len(seg) == 0:
        return {0: 1.0}
    counts = Counter(int(v) for v in seg)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


FREQ_EDGES = (1, 10, 100, 1_000, 10_000, 100_000, float('inf'))


def freq_bucket(n):
    # a word unseen in the frequency sample is RAREST, not most frequent —
    # falling through to the last bucket would have put every unseen word in
    # the 100k+ band and inverted the whole analysis
    if n < FREQ_EDGES[0]:
        return 0
    for i in range(len(FREQ_EDGES) - 1):
        if FREQ_EDGES[i] <= n < FREQ_EDGES[i + 1]:
            return i
    return len(FREQ_EDGES) - 2


def build_word_freq(lines):
    """
    Word frequencies over the training split.

    Plain regex over raw text — no spaCy — so this costs seconds, not hours.
    It only has to be good enough to STRATIFY words into frequency bands.
    """
    import re
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    pat = re.compile(r"[A-Za-z']+")
    freq = Counter()
    n = 0
    for row in ds:
        if n >= lines:
            break
        t = row['text'].strip()
        if not t or t.startswith(' ='):
            continue
        freq.update(w.lower() for w in pat.findall(t))
        n += 1
    return freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', default='det', choices=['det', 'morph'])
    ap.add_argument('--lines', type=int, default=400,
                    help='0 = the entire split. The rare-word finding this '
                         'routing estimate rests on came from 400 lines and '
                         'only 5,475 bytes in the rarest band, so confirm it '
                         'at full scale before trusting the estimate.')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--bpe-path', default='data/bpe_bytelevel.json')
    ap.add_argument('--morph-run', help='checkpoint id for part 2')
    ap.add_argument('--bpe-run', help='checkpoint id for part 2')
    ap.add_argument('--results-dir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--freq-edges', default='',
                    help='comma-separated frequency band edges, e.g. '
                         '1,3,10,20,30,50,100,inf. The default bands jump '
                         'from 10 to 100, and the crossover where morphology '
                         'stops winning lies inside that gap.')
    ap.add_argument('--freq-lines', type=int, default=40000,
                    help='training lines scanned to build the word-frequency '
                         'table for Part 3 (0 = skip Part 3)')
    args = ap.parse_args()

    if args.freq_edges:
        global FREQ_EDGES
        FREQ_EDGES = tuple(float('inf') if e.strip() == 'inf' else int(e)
                           for e in args.freq_edges.split(','))
        if len(FREQ_EDGES) < 3 or FREQ_EDGES[-1] != float('inf'):
            raise SystemExit('--freq-edges needs >=3 values ending in inf')
        if list(FREQ_EDGES) != sorted(FREQ_EDGES):
            raise SystemExit('--freq-edges must be ascending')

    from morphlm.config import ExperimentConfig, apply_overrides
    from morphlm.pipelines.registry import build_pipeline
    from datasets import load_dataset

    cfg = apply_overrides(ExperimentConfig(),
                          [f'data.pipeline={args.pipeline}'])
    print(f'building {args.pipeline}...')
    pipe = build_pipeline(cfg, verbose=False)
    from morphlm.data.tokenization import load_tokenizer
    tk = load_tokenizer(args.bpe_path)

    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=args.split)
    from morphlm.progress import Progress
    prog = Progress(total=args.lines or None)

    # ── part 1: tokenization economics ────────────────────────────────────────
    stat = {
        1: dict(words=0, bytes=0, morph_tok=0, bpe_tok=0),
        2: dict(words=0, bytes=0, morph_tok=0, bpe_tok=0),
    }
    region_byte_total = Counter()
    texts = []
    processed = 0

    for row in ds:
        if args.lines and processed >= args.lines:
            break
        text = row['text'].strip()
        if not text or text.startswith(' ='):
            continue

        toks = pipe.encode(text)
        if not toks:
            continue
        groups = group_words(toks)
        spans = word_spans(text, groups)
        mask = build_masks(text, groups, spans)

        enc = tk.encode(text)

        for (w, ts), sp in zip(groups, spans):
            if sp is None:
                continue
            cls = 2 if any(t['category'] in FALLBACK_CATS for t in ts) else 1
            stat[cls]['words'] += 1
            stat[cls]['bytes'] += len(str(w).encode('utf-8'))
            stat[cls]['morph_tok'] += len(ts)
        # a BPE token may straddle regions; count it fractionally
        for (a, b) in enc.offsets:
            for c, frac in span_weights(mask, a, b).items():
                if c in (1, 2):
                    stat[c]['bpe_tok'] += frac

        rb = region_bytes(text, mask)
        for k, v in rb.items():
            region_byte_total[k] += v
        texts.append((text, toks, mask, enc))
        processed += 1
        if processed % 200 == 0:
            print(f'  {processed} lines | {prog.status(processed)}')

    tot_w = stat[1]['words'] + stat[2]['words']
    tot_b = stat[1]['bytes'] + stat[2]['bytes']

    print()
    print('=' * 74)
    print(f'PART 1 — TOKENIZATION ECONOMICS ({processed} {args.split} lines)')
    print('=' * 74)
    print(f'  {"word class":22} {"words":>9} {"% w":>6} {"bytes":>10} '
          f'{"morph tok/w":>12} {"bpe tok/w":>10}')
    for cls, name in ((1, 'analysed cleanly'), (2, 'character fallback')):
        s = stat[cls]
        if not s['words']:
            continue
        print(f'  {name:22} {s["words"]:9,} {100.0*s["words"]/tot_w:5.1f}% '
              f'{s["bytes"]:10,} {s["morph_tok"]/s["words"]:12.2f} '
              f'{s["bpe_tok"]/s["words"]:10.2f}')
    print()
    print(f'  fallback words are {100.0*stat[2]["words"]/max(tot_w,1):.1f}% of '
          f'words but {100.0*stat[2]["bytes"]/max(tot_b,1):.1f}% of word bytes')
    if stat[2]['words']:
        ratio = ((stat[2]['morph_tok'] / stat[2]['words'])
                 / max(stat[2]['bpe_tok'] / stat[2]['words'], 1e-9))
        print(f'  on those words morph spends {ratio:.2f}x the tokens BPE does')

    if not (args.morph_run and args.bpe_run):
        print()
        print('  (pass --morph-run and --bpe-run for the restricted BPB)')
        return

    # ── part 2: restricted BPB ────────────────────────────────────────────────
    import torch
    import torch.nn.functional as F
    from morphlm.training.metrics import load_checkpoint
    from morphlm.training.loop import get_device

    device = get_device()

    def score(run_id, stream_ids, stream_w, seq_len):
        per_tok = np.zeros(len(stream_ids) + 1, dtype=np.float64)
        model, mcfg = load_checkpoint(run_id, args.results_dir)
        if mcfg.frozen.facts:
            raise SystemExit(
                f'{run_id} is a frozen-facts cell ({list(mcfg.frozen.facts)}). '
                f'This is a TOKENIZER comparison, so use the baseline '
                f'checkpoints — otherwise the frozen dims confound it.')
        model = model.to(device).eval()
        ids = np.asarray(stream_ids, dtype=np.int64)
        n = (len(ids) - 1) // seq_len
        bits = Counter()
        toks = Counter()
        with torch.no_grad():
            for w in range(n):
                a = w * seq_len
                x = torch.from_numpy(ids[a:a + seq_len]).unsqueeze(0).to(device)
                y = torch.from_numpy(
                    ids[a + 1:a + seq_len + 1]).unsqueeze(0).to(device)
                logits, _ = model(x, y)
                per = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      y.view(-1),
                                      reduction='none').cpu().numpy()
                per = per / math.log(2)
                for j in range(seq_len):
                    per_tok[a + 1 + j] = per[j]
                    for k, frac in stream_w[a + 1 + j].items():
                        bits[k] += per[j] * frac
                        toks[k] += frac
        return bits, toks, per_tok

    # ── build aligned streams with per-token REGION WEIGHTS ──────────────────
    # morph: every token of a word lies entirely inside that word's region, so
    # the assignment is exact. <sp> tokens inherit the region of the word they
    # precede, mirroring how ByteLevel folds the space into the next token.
    m_ids, m_w = [], []
    b_ids, b_w = [], []
    for text, toks, mask, enc in texts:
        groups = group_words(toks)
        spans = word_spans(text, groups)
        cls_of_group = {}
        for (w, ts), sp in zip(groups, spans):
            cls_of_group[id(ts)] = (
                2 if any(t['category'] in FALLBACK_CATS for t in ts) else 1)
        tok_cls = {}
        for (w, ts), sp in zip(groups, spans):
            for t in ts:
                tok_cls[id(t)] = cls_of_group[id(ts)]

        pending = []
        for t in toks:
            c = tok_cls.get(id(t))
            if c is None:
                if t.get('category') == 'space':
                    pending.append(t)      # resolve once the next word appears
                    continue
                c = 0
            if pending:
                for pt in pending:
                    m_ids.append(pt['token_id'])
                    m_w.append({c: 1.0})
                pending = []
            m_ids.append(t['token_id'])
            m_w.append({c: 1.0})
        for pt in pending:
            m_ids.append(pt['token_id'])
            m_w.append({0: 1.0})

        for tid, (a, b) in zip(enc.ids, enc.offsets):
            b_ids.append(tid)
            b_w.append(span_weights(mask, a, b))

    bytes_by_cls = region_byte_total

    _cp = os.path.join(args.results_dir, args.bpe_run, 'config.json')
    if os.path.exists(_cp):
        import json as _json
        _want = _json.load(open(_cp))['data']['bpe_path']
        if os.path.basename(_want) != os.path.basename(args.bpe_path):
            raise SystemExit(
                f'--bpe-path is {args.bpe_path} but run {args.bpe_run} was '
                f'trained with {_want}. Scoring one tokenizer\'s stream '
                f'with another model is silently meaningless; pass '
                f'--bpe-path {_want}')

    print()
    print(f'scoring {args.pipeline} ({args.morph_run})...')
    mb, mt, m_per = score(args.morph_run, m_ids, m_w, cfg.data.seq_len)
    print(f'scoring bpe ({args.bpe_run})...')
    bb, bt, b_per = score(args.bpe_run, b_ids, b_w, cfg.data.seq_len)

    print()
    print('=' * 74)
    print('PART 2 — RESTRICTED BITS-PER-BYTE')
    print('=' * 74)
    # sanity: every token's weight must be fully attributed, and every region
    # with bits must have bytes. Silent misattribution is what produced the
    # first (wrong) run of this script.
    print(f'  attribution check: morph tokens {sum(mt.values()):,.0f} '
          f'of {len(m_ids):,} | bpe tokens {sum(bt.values()):,.0f} '
          f'of {len(b_ids):,}')
    for c in (0, 1, 2):
        if (mb[c] or bb[c]) and not bytes_by_cls[c]:
            print(f'  WARNING: region {c} has bits but no bytes')
    print()
    names = {0: 'punct / digits / space', 1: 'words analysed cleanly',
             2: 'words in char fallback'}
    print(f'  {"region":24} {"bytes":>10} {"morph BPB":>11} '
          f'{"bpe BPB":>10} {"delta":>9}')
    print('  ' + '-' * 70)
    for c in (1, 2, 0):
        nb = bytes_by_cls[c]
        if not nb or not mt[c] or not bt[c]:
            continue
        mbpb = mb[c] / nb
        bbpb = bb[c] / nb
        print(f'  {names[c]:24} {nb:10,} {mbpb:11.4f} {bbpb:10.4f} '
              f'{mbpb - bbpb:+9.4f}')
    tot_bytes = sum(bytes_by_cls.values())
    m_all = sum(mb.values()) / tot_bytes
    b_all = sum(bb.values()) / tot_bytes
    print('  ' + '-' * 70)
    print(f'  {"ALL":24} {tot_bytes:10,} {m_all:11.4f} {b_all:10.4f} '
          f'{m_all - b_all:+9.4f}')

    # extrapolation: what if fallback words cost what clean words cost?
    if bytes_by_cls[1] and bytes_by_cls[2]:
        clean_rate = mb[1] / bytes_by_cls[1]
        hypo = (mb[0] + mb[1] + clean_rate * bytes_by_cls[2]) / tot_bytes
        print()
        print('  EXTRAPOLATION — if fallback words were analysed as well as')
        print('  covered ones (i.e. perfect vocabulary coverage):')
        print(f'    morph BPB {m_all:.4f} -> {hypo:.4f}   vs bpe {b_all:.4f} '
              f'({hypo - b_all:+.4f})')
        print()
        print('  That is an UPPER BOUND on what vocabulary expansion can buy:')
        print('  newly covered words would be rare ones, harder than the')
        print('  average covered word, so the real gain is smaller.')
    print()
    print('  CAVEAT: restricting to covered words selects on the outcome. This')
    print('  supports a claim about WHERE the deficit comes from, not a claim')
    print('  that the tokenizer is better overall.')
    print()

    if not args.freq_lines:
        return

    # ── part 3: does the deficit depend on word frequency? ───────────────────
    print(f'building word-frequency table over {args.freq_lines:,} '
          f'training lines...')
    wfreq = build_word_freq(args.freq_lines)
    print(f'  {len(wfreq):,} distinct word forms')

    nb_ = len(FREQ_EDGES) - 1
    fb_bytes = Counter()
    m_fbits = Counter()
    b_fbits = Counter()

    # rebuild per-character frequency-bucket masks, CLEAN WORDS ONLY
    idx_m = 0
    idx_b = 0
    for text, toks, mask, enc in texts:
        groups = group_words(toks)
        spans = word_spans(text, groups)
        fmask = np.full(len(text), -1, dtype=np.int8)
        for (w, ts), sp in zip(groups, spans):
            if sp is None:
                continue
            if any(t['category'] in FALLBACK_CATS for t in ts):
                continue          # clean words only: fallback is Part 2's story
            fb = freq_bucket(wfreq.get(str(w).lower(), 0))
            fmask[sp[0]:sp[1]] = fb
        for ch, f in zip(text, fmask):
            if f >= 0:
                fb_bytes[int(f)] += len(ch.encode('utf-8'))

        # morph: token -> word -> bucket (exact)
        tokfb = {}
        for (w, ts), sp in zip(groups, spans):
            if sp is None or any(t['category'] in FALLBACK_CATS for t in ts):
                continue
            fb = freq_bucket(wfreq.get(str(w).lower(), 0))
            for t in ts:
                tokfb[id(t)] = fb
        for t in toks:
            if idx_m < len(m_per):
                fb = tokfb.get(id(t))
                if fb is not None:
                    m_fbits[fb] += m_per[idx_m]
            idx_m += 1

        # bpe: proportional over the span's frequency buckets
        for (a, bnd) in enc.offsets:
            if idx_b < len(b_per):
                seg = fmask[a:min(bnd, len(fmask))]
                seg = [int(v) for v in seg if v >= 0]
                if seg:
                    cc = Counter(seg)
                    tt = sum(cc.values())
                    for k, v in cc.items():
                        b_fbits[k] += b_per[idx_b] * (v / tt)
            idx_b += 1

    print()
    print('=' * 74)
    print('PART 3 — CLEAN WORDS ONLY, STRATIFIED BY TRAINING FREQUENCY')
    print('=' * 74)
    print(f'  {"word freq":>18} {"bytes":>10} {"morph BPB":>11} '
          f'{"bpe BPB":>10} {"delta":>9}')
    print('  ' + '-' * 62)
    for k in range(nb_):
        if not fb_bytes[k]:
            continue
        lo, hi = FREQ_EDGES[k], FREQ_EDGES[k + 1]
        rng = f'{int(lo):,}-{"inf" if hi == float("inf") else format(int(hi), ",")}'
        mm = m_fbits[k] / fb_bytes[k]
        bbv = b_fbits[k] / fb_bytes[k]
        print(f'  {rng:>18} {fb_bytes[k]:10,} {mm:11.4f} {bbv:10.4f} '
              f'{mm - bbv:+9.4f}')
    print()
    print('  READING THIS')
    print('    delta negative for RARE words -> morphological decomposition')
    print('      shares statistical strength exactly where it should, and the')
    print('      deficit on frequent words is a design choice (we decompose')
    print('      everything) rather than a flaw in the idea.')
    print('    delta positive everywhere     -> decomposition costs more than')
    print('      it returns at this model scale, independent of coverage.')
    print()

    # ── part 4: what would a FREQUENCY-ROUTED tokenizer score? ───────────────
    #
    # Q1 found morphology wins on rare words and loses on frequent ones. The
    # obvious response is to use each where it wins: morphemes below a
    # frequency threshold, BPE above it. This estimates that BEFORE any
    # tokenizer is built, from per-word costs already measured.
    #
    # Routing is by FREQUENCY BAND ONLY, using the pre-existing FREQ_EDGES.
    # The whole curve is reported, not the best point: choosing a threshold by
    # looking at the losses would be selecting on the outcome, which is the
    # error this script was rewritten to avoid.
    #
    # WHAT THIS IS NOT. It assumes a routed model would cost, on each word,
    # what the specialist model costs on that word. A real routed model sees a
    # MIXED stream: a different vocabulary, different context, different
    # statistics. It takes the best of each arm without paying for combining
    # them, so it is OPTIMISTIC — an upper bound on what routing can buy, and a
    # go / no-go signal rather than a prediction.
    print('=' * 74)
    print('PART 4 — FREQUENCY-ROUTED TOKENIZER (idealised upper bound)')
    print('=' * 74)

    # non-clean regions always go to BPE: morph cannot analyse fallback words,
    # and punctuation/digits are where it is weakest (Part 2)
    fixed_bpe = bb[0] + bb[2]
    clean_bytes = sum(fb_bytes.values())

    # DENOMINATOR. Every BPB here divides by tot_bytes, the WHOLE corpus.
    #
    # Part 3 redistributes each token's bits onto the word CHARACTERS it
    # covers, dropping the whitespace bytes from its denominator but keeping
    # all the bits. Its per-bucket rates are therefore on a word-characters
    # basis, fine for comparing morph against bpe within a bucket, but not
    # comparable with Part 2.
    #
    # An earlier version of this section summed Part 3's bytes with Part 2's
    # region bytes, losing 146,750 bytes of whitespace from the denominator
    # while keeping every bit. It inflated all of Part 4 by 1.148x and printed
    # pure morph BEATING pure bpe — the exact opposite of Part 2, computed from
    # the same two checkpoints. That impossibility is what exposed it.
    pure_bpe = (fixed_bpe + sum(b_fbits.values())) / tot_bytes
    pure_morph = (mb[0] + mb[1] + mb[2]) / tot_bytes
    if abs(pure_bpe - sum(bb.values()) / tot_bytes) > 5e-4:
        raise SystemExit(
            f'Part 4 pure-BPE {pure_bpe:.4f} disagrees with Part 2 '
            f'{sum(bb.values()) / tot_bytes:.4f}: the bit accounting is '
            f'inconsistent, so no routed figure below can be trusted.')

    print(f'  routed over {clean_bytes:,} clean-word bytes of '
          f'{tot_bytes:,} total; fallback words and punctuation always BPE')
    print()
    print(f'  {"threshold":>22} {"to morph":>10} {"routed BPB":>12} '
          f'{"vs pure BPE":>12}')
    print('  ' + '-' * 62)
    best = None
    for k in range(-1, nb_):
        # route buckets 0..k to morph, the rest to BPE
        bits = fixed_bpe
        morph_bytes = 0
        for j in range(nb_):
            if j <= k:
                bits += m_fbits[j]
                morph_bytes += fb_bytes[j]
            else:
                bits += b_fbits[j]
        bpb = bits / tot_bytes
        if k < 0:
            label = 'none (pure BPE)'
        else:
            hi = FREQ_EDGES[k + 1]
            label = f'freq < {"inf" if hi == float("inf") else format(int(hi), ",")}'
        share = 100.0 * morph_bytes / clean_bytes if clean_bytes else 0.0
        if best is None or bpb < best[1]:
            best = (label, bpb, k)
        print(f'  {label:>22} {share:9.1f}% {bpb:12.4f} '
              f'{bpb - pure_bpe:+12.4f}')
    print('  ' + '-' * 62)
    print(f'  {"pure morph":>22} {100.0:9.1f}% {pure_morph:12.4f} '
          f'{pure_morph - pure_bpe:+12.4f}')
    print()
    if best and best[2] >= 0 and best[1] < pure_bpe:
        gain = pure_bpe - best[1]
        print(f'  BEST ROUTE: {best[0]} -> {best[1]:.4f} BPB, '
              f'{gain:.4f} below pure BPE ({100*gain/pure_bpe:.2f}%)')
        print('  Routing beats BOTH pure tokenizers on this idealised estimate.')
    else:
        print('  NO ROUTE BEATS PURE BPE, even idealised. Since this bound is')
        print('  optimistic, a real routed tokenizer would do worse. That is a')
        print('  stop signal, and a cheap one.')
    print()
    print('  READING THIS')
    print('    The estimate is an UPPER BOUND: it gives the router each')
    print('    specialist\'s cost without the cost of combining them. Treat a')
    print('    positive result as permission to build, not as the result.')
    print('    The threshold is a frequency band fixed in advance; the whole')
    print('    curve is shown so no point is chosen by looking at the answer.')
    print()


if __name__ == '__main__':
    main()
