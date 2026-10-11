#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Long chain, 7 October 2026 (journey §33.64). ~9.1 h of GPU in 5 pieces, in
# priority order; END_BY keeps it from starting a run it could not finish.
#
# D7 at ~38M tokens and a 10k budget: the null, case and case's word form
# (`mconst`, `ucase`, `ucaselex`), each over its own tokenizer's null.
#
#   1  BPE 17k with endings apart, seed 0 — its first runs (smoke test first) ~35 min  P126 P127
#   2  the same, seeds 1–3                                                    ~1.7 h  P128 P129
#   3  seed 1 under Unigram 17k and BPE 4k, 8k, 32k; BPE 17k's word form      ~2.3 h
#   4  seed 2, the same                                                       ~2.3 h
#   5  seed 3, the same                                                       ~2.3 h  P130 P131
#   then tables and the learning figure (seconds).
#
# Every run is a RESUMABLE PIECE: a run whose cell and seed already have a
# finished result in results/ is skipped (scripts/ops/run_ids.py --done). Stop
# the chain at any time; start it again and it carries on where it stopped.
# A tokenizer whose condition is not ready, or whose smoke test fails, has its
# runs skipped; the rest go on.
#
# END_BY — a time `date -d` reads ("21:15", "2026-10-07 21:15"): a tokenizer's
# unfinished runs for a seed start only if they would all end before then, by
# the minutes a 10k-step run took on 6 October (BPE 4k 7, 8k 8, 17k and Unigram
# 11, 32k 16; endings apart as BPE 17k), each with a minute to spare — so a null
# never runs without its case. What does not fit is left for the next chain,
# which skips what is done. Without END_BY, everything runs.
#
#   END_BY=$(date -d '+8 hours 30 minutes' '+%F %T') nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-07.sh >/dev/null 2>&1 &
#   PIECES="1 2" nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-07.sh >/dev/null 2>&1 &
# Knobs: PIECES, END_BY, ENDTOK, TOKS, UK17, D7FRAC; UKEXTRA, PEXTRA, MINUTES (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
LOG=logs/long_2026-10-07.log
exec > >(tee -a "$LOG") 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-1 2 3 4 5}
ENDTOK=${ENDTOK:-bpeend_uk_17k}
TOKS=${TOKS-unigram_uk_17k bpe_uk_4k bpe_uk_8k bpe_uk_32k}
UK17=${UK17:-bpe_uk_17k}
D7FRAC=${D7FRAC:-0.05}
UK="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S10="-o train.max_steps=10000 -o train.lr_horizon=10000"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

minutes() {  # a 10k-step run of tokenizer $1, in minutes, with one to spare
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in *_4k) echo 8 ;; *_8k) echo 9 ;; *_32k) echo 17 ;; *) echo 12 ;; esac
}
DEADLINE=""
if [ -n "${END_BY:-}" ]; then
    DEADLINE=$(date -d "$END_BY" +%s) || { echo "END_BY not understood: $END_BY — not starting"; exit 1; }
fi
LEFT=0
on() { case " $PIECES " in *" $1 "*) return 0 ;; esac; return 1; }

declare -A STATE
usable() {   # tokenizer $1's condition is ready and its smoke test passes — checked once
    local t=$1 out
    if [ -z "${STATE[$t]:-}" ]; then
        if ! out=$(python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction "$D7FRAC" \
                   --bpe-path "data/$t.json" ${PEXTRA:-} --dry-run 2>/dev/null) \
           || ! echo "$out" | grep -q "train cache: done" || echo "$out" | grep -qE "to do|to build"; then
            echo "  [not ready] $t — its condition is not prepared; its runs are skipped"; STATE[$t]=no
        elif ! python scripts/ops/smoke_model.py --cells mconst ucase ucaselex --bpe-path "data/$t.json" \
                --set data.corpus=ukwiki data.corpus_fraction="$D7FRAC"; then
            echo "  SMOKE TEST FAILED for $t — its runs are skipped"; STATE[$t]=no
        else
            STATE[$t]=yes
        fi
    fi
    [ "${STATE[$t]}" = yes ]
}
args_for() {  # args_for TOKENIZER FACT SEED — a D7 run's arguments, as on 6 October
    echo "--facts $2 $UK -o data.corpus_fraction=$D7FRAC -o data.bpe_path=data/$1.json $C $S10 -o train.seed=$3 ${UKEXTRA:-}"
}
cells() {    # cells TOKENIZER SEED FACT... — the unfinished ones, all or none: a set that
             # would end after END_BY is left whole for the next chain, never a null alone
    local t=$1 s=$2 f pending=()
    shift 2
    for f in "$@"; do
        python scripts/ops/run_ids.py --done $(args_for "$t" "$f" "$s") || pending+=("$f")
    done
    [ ${#pending[@]} -eq 0 ] && return 0
    usable "$t" || return 0
    if [ -n "$DEADLINE" ] && [ $(( $(date +%s) + ${#pending[@]} * $(minutes "$t") * 60 )) -gt "$DEADLINE" ]; then
        echo "  [left for the next chain] ${pending[*]} under $t, seed $s — would end after $END_BY"
        LEFT=$((LEFT + ${#pending[@]})); return 0
    fi
    for f in "${pending[@]}"; do $RUN $(args_for "$t" "$f" "$s"); done
}
family() {   # family TOKENIZER SEED — the null, case and case's word form
    cells "$1" "$2" mconst ucase ucaselex
}
d7seed() {   # d7seed SEED — the four other tokenizers, then BPE 17k's word form
    for t in $TOKS; do family "$t" "$1"; done
    cells "$UK17" "$1" ucaselex
}

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "long chain of 7 October starts — pieces: $PIECES${END_BY:+; runs must end by $END_BY}"

if on 1; then
    stamp "1  D7: BPE 17k with endings apart — the null, case and word form, seed 0"
    family "$ENDTOK" 0
fi
if on 2; then
    stamp "2  D7: BPE 17k with endings apart, seeds 1–3"
    for s in 1 2 3; do family "$ENDTOK" "$s"; done
fi
for p in 3 4 5; do
    if on $p; then
        s=$((p - 2))
        stamp "$p  D7: seed $s under $TOKS, and BPE 17k's word form"
        d7seed "$s"
    fi
done

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
if [ "$LEFT" -gt 0 ]; then
    stamp "CHAIN ENDED BY $END_BY — $LEFT run(s) left for the next chain; python scripts/ops/pack_results.py, then send the zip"
else
    stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
fi
