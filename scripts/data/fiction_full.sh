#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# fiction_full.sh — UberText fiction's two conditions at fraction 0.4, built from the public
# source on a machine that has none of it (journey §33.101). CPU only; a chain's fiction piece
# runs only once this has left `external/ubertext/FULL_VERIFIED`.
#
#   1  download the cleansed fiction export (lang.org.ua) — its size and SHA-256 must be the
#      file the thesis machine converted on 11 October
#   2  convert it (prepare_ubertext_fiction.py) — fiction.jsonl's SHA-256 must be that machine's
#   3  load the corpus once, before any build: two builds that both load it first hung for five
#      hours (§33.87)
#   4  both conditions at 0.4, side by side (prepare_condition.py: tagging, the null, checks)
#   5  the validation split is the same text at every fraction, so each condition's validation
#      ids and null codes must equal the thesis machine's at 0.1, byte for byte
# Every step that is done is skipped on a second start. Reference hashes: REFS (a file of
# `sha256  path` lines, the thesis machine's).
#
#   REFS=scripts/data/fiction_refs.sha256 nohup bash scripts/data/fiction_full.sh > logs/fiction_full.log 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PYTHONUNBUFFERED=1
REFS=${REFS:-scripts/data/fiction_refs.sha256}
URL=https://lang.org.ua/static/downloads/ubertext2.0/fiction/cleansed/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2
SRC=external/ubertext/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2
OUT=external/ubertext/fiction.jsonl
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }
ref() { grep -F "  $1" "$REFS" | cut -d' ' -f1; }
same() {        # same PATH REF_PATH — PATH's SHA-256 is the reference recorded for REF_PATH
    local got want; got=$(sha256sum "$1" | cut -d' ' -f1); want=$(ref "$2")
    if [ -n "$want" ] && [ "$got" = "$want" ]; then echo "  [check] $1: the thesis machine's file"; return 0; fi
    echo "  [check] $1: $got, NOT the reference ${want:-(none)}"; return 1
}
fail() { stamp "STOPPED: $*"; exit 1; }
mkdir -p external/ubertext logs

stamp "1  the fiction export"
if [ ! -f "$SRC" ]; then
    curl -fL --retry 5 --retry-delay 30 -o "$SRC.part" "$URL" && mv "$SRC.part" "$SRC" || fail "download"
fi
[ "$(stat -c %s "$SRC")" = 433084760 ] || fail "size $(stat -c %s "$SRC"), not 433,084,760"
same "$SRC" "$SRC" || fail "the download is not the file converted on 11 October"

stamp "2  conversion"
[ -f "$OUT" ] || python scripts/data/prepare_ubertext_fiction.py || fail "conversion"
same "$OUT" "$OUT" || fail "fiction.jsonl differs from the thesis machine's"

stamp "3  the corpus loaded once, before the builds"
python - <<'EOF' || fail "loading the corpus"
import sys; sys.path.insert(0, '.')
from morphlm.config import ExperimentConfig, apply_overrides
from morphlm.data.stream import load_corpus
cfg = apply_overrides(ExperimentConfig(), ['data.corpus=ukfiction', 'data.corpus_holdout=500', 'data.corpus_fraction=0.4'])
n = sum(1 for _ in load_corpus(cfg, 'validation'))
print(f'  loaded: {n:,} validation rows')
EOF

stamp "4  both conditions at 0.4, side by side"
for t in bpe_ukf_17k bpeend_ukf_17k; do
    nice -n 15 python scripts/data/prepare_condition.py --corpus ukfiction --corpus-fraction 0.4 \
        --corpus-holdout 500 --bpe-path "data/$t.json" > "logs/fiction_condition_$t.log" 2>&1 &
    sleep 120
done
wait
for t in bpe_ukf_17k bpeend_ukf_17k; do
    tail -3 "logs/fiction_condition_$t.log"
    out=$(python scripts/data/prepare_condition.py --corpus ukfiction --corpus-fraction 0.4 --corpus-holdout 500 \
          --bpe-path "data/$t.json" --dry-run 2>/dev/null)
    echo "$out" | grep -q "train cache: done" && ! echo "$out" | grep -qE "mconst to do|to build" \
        || fail "$t's condition at 0.4 is not complete (logs/fiction_condition_$t.log)"
done

stamp "5  the validation split against the thesis machine's at 0.1"
python - "$REFS" <<'EOF' || fail "a validation stream differs"
import sys, os, hashlib, subprocess
sys.path.insert(0, '.')
from morphlm.config import ExperimentConfig, apply_overrides
from morphlm.data.stream import cache_stem
refs = {l.split()[1]: l.split()[0] for l in open(sys.argv[1]) if l.strip()}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
stem = lambda t, fr: cache_stem('bpe', 'validation', None, apply_overrides(ExperimentConfig(), [
    'data.pipeline=bpe', f'data.bpe_path=data/{t}.json', 'data.corpus=ukfiction', f'data.corpus_fraction={fr}',
    'data.corpus_holdout=500', 'data.spacy_model=uk_core_news_sm']).surface_spec())
bad = 0
for t in ('bpe_ukf_17k', 'bpeend_ukf_17k'):
    here, there = stem(t, '0.4'), stem(t, '0.1')
    for suffix in ('.npy', '_mconst.codes.npy'):
        got, want = sha(f'data/cache/{here}{suffix}'), refs.get(f'data/cache/{there}{suffix}')
        ok = got == want
        bad += not ok
        print(f'  {t} validation{suffix}: {"the same as at 0.1 on the thesis machine" if ok else "DIFFERS"}')
sys.exit(1 if bad else 0)
EOF
date -u +%FT%TZ > external/ubertext/FULL_VERIFIED
stamp "FICTION AT 0.4 READY AND VERIFIED"
