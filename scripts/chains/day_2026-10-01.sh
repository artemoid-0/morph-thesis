#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Last chain before the demo, 1 October 2026 (journey §33.31). ~6.5 h.
# Starts itself when night_2026-10-01.sh has finished.
#
#   CPU  A  lexical case at both little-data sizes from the FULL corpus's
#           lexicon (--lexicon-fraction: the same context-free knowledge as
#           the full-data runs, not a lexicon starved by a small corpus), gate
#   CPU  B  the ~38M-token cache (corpus fraction 0.05), the null's codes, gate
#   GPU  1  all the data, 10k steps: null and case, seeds 0-3         P61
#   GPU  2  after A: lexical case at ~1.5M and ~7.6M, seeds 0-3         P60
#   GPU  3  after B: null and case at ~38M tokens, seeds 0-3            P62
#   GPU  4  the baseline at ~1.5M and ~7.6M, seeds 0-3                  P63
#   then profiles and tables.
#
#   nohup bash scripts/chains/day_2026-10-01.sh >/dev/null 2>&1 &
#   status any time: bash scripts/ops/chain_status.sh
# Rehearsal knobs (defaults are the real day): WAIT_FOR, SEEDS, SMALL, MID,
#   FULL, UKTOK, UKEXTRA, DEXTRA, CEXTRA, LSTEPS_N, G1, G2, G3, G4
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
LOG=logs/day_2026-10-01.log
exec > >(tee -a "$LOG") 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

WAIT_FOR=${WAIT_FOR-'^bash scripts/chains/night_2026-10-01.sh'}
SEEDS=${SEEDS-0 1 2 3}
SMALL=${SMALL-0.002 0.01}
MID=${MID-0.05}
FULL=${FULL-0.2}
G1=${G1-1}; G2=${G2-1}; G3=${G3-1}; G4=${G4-1}
UKTOK=${UKTOK:-data/bpe_uk_17k.json}
N=${LSTEPS_N:-10000}
UKBASE="-o data.corpus=ukwiki -o data.bpe_path=$UKTOK -o data.spacy_model=uk_core_news_sm ${UKEXTRA:-}"
STEPS="-o train.max_steps=$N -o train.lr_horizon=$N"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"
D() { echo "--corpus ukwiki --corpus-fraction $1 --bpe-path $UKTOK ${DEXTRA:-}"; }
gate() { python scripts/data/check_cache.py --split "$1" --bpe-path "$UKTOK" --corpus ukwiki --corpus-fraction "$2" ${CEXTRA:-} \
             --require-facts $3 --require-constant 99.9; }

if [ -n "$WAIT_FOR" ]; then
    stamp "waiting for $WAIT_FOR to finish"
    while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 60; done
fi
stamp "day chain starts"

# ── CPU A: lexical case at the little-data sizes ─────────────────────────────
(
    for F in $SMALL; do
        stamp "A  lexical case, corpus fraction $F"
        python scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaselex $(D $F) --lexicon-fraction $FULL --verify || exit 1
        for sp in train validation; do gate $sp $F ucaselex || exit 1; done
    done
    stamp "A PASSED"
) > logs/day_2026-10-01_A.log 2>&1 &
A_PID=$!
# ── CPU B: the middle size ───────────────────────────────────────────────────
(
    stamp "B  cache, corpus fraction $MID"
    python -m morphlm.cli cache --pipeline bpe --facts ucase $UKBASE -o data.corpus_fraction=$MID || exit 1
    python scripts/data/derive_lexical_codes.py --fact ucase --out-fact mconst --transform const --keep ucase $(D $MID) --verify || exit 1
    for sp in train validation; do gate $sp $MID "ucase mconst" || exit 1; done
    stamp "B PASSED"
) > logs/day_2026-10-01_B.log 2>&1 &
B_PID=$!

# ── GPU ───────────────────────────────────────────────────────────────────────
if [ "$G1" = 1 ]; then
    stamp "1  all the data, $N steps: null and case"
    for s in $SEEDS; do for f in mconst ucase; do
        $RUN --facts $f $UKBASE -o data.corpus_fraction=$FULL $C $STEPS -o train.seed=$s ;
    done; done
fi
wait "$A_PID"; A_OK=$?
echo "CPU A exit status $A_OK:"; grep -E "^=== |lexicon:|unseen|GATE|Traceback|Error" logs/day_2026-10-01_A.log | tail -n 14
if [ "$G2" = 1 ]; then
    if [ "$A_OK" -eq 0 ]; then
        for F in $SMALL; do
            stamp "2  lexical case, corpus fraction $F"
            for s in $SEEDS; do $RUN --facts ucaselex $UKBASE -o data.corpus_fraction=$F $C $STEPS -o train.seed=$s ; done
        done
    else stamp "2 SKIPPED — CPU A failed (logs/day_2026-10-01_A.log)"; fi
fi
wait "$B_PID"; B_OK=$?
echo "CPU B exit status $B_OK:"; grep -E "^=== |cached\]|GATE|Traceback|Error" logs/day_2026-10-01_B.log | tail -n 10
if [ "$G3" = 1 ]; then
    if [ "$B_OK" -eq 0 ]; then
        stamp "3  corpus fraction $MID: null and case"
        for s in $SEEDS; do for f in mconst ucase; do
            $RUN --facts $f $UKBASE -o data.corpus_fraction=$MID $C $STEPS -o train.seed=$s ;
        done; done
    else stamp "3 SKIPPED — CPU B failed (logs/day_2026-10-01_B.log)"; fi
fi
if [ "$G4" = 1 ]; then
    for F in $SMALL; do
        stamp "4  the baseline, corpus fraction $F"
        for s in $SEEDS; do $RUN $UKBASE -o data.corpus_fraction=$F $STEPS -o train.seed=$s ; done
    done
fi
stamp "profiles, tables"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
