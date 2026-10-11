#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 6 October 2026 (journey §33.57). ~7.7 h of GPU, 6 pieces.
#
# Every run is a RESUMABLE PIECE: a run whose cell and seed already have a
# finished result in results/ is skipped (scripts/ops/run_ids.py --done). Stop
# the chain at any time; start it again and it carries on where it stopped.
# PIECES="1 4" runs just those pieces. Priority order:
#
#   1  G-71  flat slot and flat vector added WITHOUT the causal shift,
#            rotary positions, 20k, seed 0                              ~48 min  P114 P115
#   2  G-73  the ~38M-token Ukrainian null again, seeds 0–3 — the
#            17k reference of D7                                         ~44 min  P116
#   3  G-73  the 151M-token Ukrainian null at 50k again, seeds 0–1      ~1.9 h   P116 P117
#   4  D7    each new tokenizer whose condition is ready — the null,
#            case and lexical case at ~38M tokens, 10k, seed 0; and
#            lexical case under BPE 17k                                  ~2.3 h   P118–P120
#   5  G-73  the 151M-token Ukrainian null at 50k again, seeds 2–3      ~1.9 h
#   6  D7    the tokenizers not ready in piece 4, waiting (at most
#            WAIT_D7 minutes) while their conditions are prepared
#   then tables and the learning figure.
#
# The Ukrainian nulls re-run deterministically: the same models as the
# quarantined runs, every evaluation now on the constant (G-73). Before piece 2
# the chain checks that those runs are out of results/ — otherwise --done would
# take them for finished — and moves them if the dry run is exactly the one read
# on 6 October (rule const_unseen, 34 runs).
#
# CPU track: if no prepare_condition.py is running when the chain starts, the
# chain prepares the D7 conditions itself, in the background (its log:
# logs/night_2026-10-06_cpu.log); if one is running (started by hand), it is left
# to finish.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/night_2026-10-06.sh >/dev/null 2>&1 &
#   PIECES="2 4" nohup systemd-inhibit --what=sleep:idle bash scripts/chains/night_2026-10-06.sh >/dev/null 2>&1 &
# Knobs: PIECES, TOKS, UK17, D7FRAC, WAIT_D7 (minutes), WAIT_RUNS (0 skips the
# wait), CPU_TRACK (0: never start one); ENTOK, ENEXTRA, UKEXTRA, PEXTRA (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
LOG=logs/night_2026-10-06.log
CPULOG=logs/night_2026-10-06_cpu.log
exec > >(tee -a "$LOG") 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-1 2 3 4 5 6}
TOKS=${TOKS-unigram_uk_17k bpe_uk_4k bpe_uk_32k bpe_uk_8k}
UK17=${UK17:-bpe_uk_17k}
D7FRAC=${D7FRAC:-0.05}
ENTOK=${ENTOK:-data/bpe_17k.json}
EN="-o data.bpe_path=$ENTOK"
UK="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S10="-o train.max_steps=10000 -o train.lr_horizon=10000"
S20="-o train.max_steps=20000 -o train.lr_horizon=20000"
ROPE="-o model.position=rope"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

on() { case " $PIECES " in *" $1 "*) return 0 ;; esac; return 1; }
run() {     # one run — skipped if results/ already holds its cell and seed
    if python scripts/ops/run_ids.py --done "$@"; then return 0; fi
    $RUN "$@" ;
}
prep_running() { pgrep -f "scripts/data/prepare_condition.py.*${1:-}" >/dev/null; }
ready() {   # tokenizer $1's condition holds every fact D7 asks for, and nothing still prepares it
    local out
    prep_running "data/$1.json" && return 1
    out=$(python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction "$D7FRAC" \
          --bpe-path "data/$1.json" ${PEXTRA:-} --dry-run 2>/dev/null) || return 1
    echo "$out" | grep -q "train cache: done" && ! echo "$out" | grep -qE "to do|to build"
}
d7() {      # the null, case and lexical case under tokenizer $1, if its condition is ready
    local t=$1 U="$UK -o data.corpus_fraction=$D7FRAC -o data.bpe_path=data/$1.json"
    if ! ready "$t"; then echo "  [not ready] $t — left for later"; return 1; fi
    if ! python scripts/ops/smoke_model.py --cells mconst ucase ucaselex --bpe-path "data/$t.json" \
            --set data.corpus=ukwiki data.corpus_fraction="$D7FRAC" ; then
        echo "  SMOKE TEST FAILED for $t — its runs skipped"; return 1
    fi
    for f in mconst ucase ucaselex; do run --facts $f $U $C $S10 -o train.seed=0 ${UKEXTRA:-}; done
}

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "night chain of 6 October starts — pieces: $PIECES"

# ── CPU: the D7 conditions, unless a preparation started by hand is running ─────
if [ "${CPU_TRACK:-1}" = 1 ] && ! prep_running; then
    stamp "CPU track starts: it prepares any D7 condition not yet ready (log: $CPULOG)"
    (
        for t in $TOKS; do
            ready "$t" && continue
            echo "=== $(date '+%F %T')  prepare $t"
            python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction "$D7FRAC" \
                --bpe-path "data/$t.json" ${PEXTRA:-} || echo "=== $t FAILED"
        done
        echo "=== $(date '+%F %T')  CPU TRACK DONE"
    ) >> "$CPULOG" 2>&1 &
else
    stamp "CPU track not started: a preparation is already running, or CPU_TRACK=0"
fi

# ── the quarantine the Ukrainian null pieces depend on ───────────────────────
if on 2 || on 3 || on 5; then
    Q=$(python scripts/ops/quarantine.py 2>&1)
    if echo "$Q" | grep -q "^  rule:"; then
        if [ "$(echo "$Q" | grep -c '^  rule:')" = 1 ] && echo "$Q" | grep -q "rule: const_unseen  (34 run(s))"; then
            stamp "the 34 Ukrainian runs on the old null are still in results/ — moving them (the dry run read on 6 October)"
            python scripts/ops/quarantine.py --apply
        else
            stamp "the quarantine finds something unexpected — pieces 2, 3 and 5 are skipped:"
            echo "$Q"
            PIECES=$(echo " $PIECES " | sed 's/ 2 / /; s/ 3 / /; s/ 5 / /')
        fi
    fi
fi

if on 1; then
    stamp "1  G-71: flat slot, then flat vector added, WITHOUT the causal shift — rotary positions, seed 0"
    if python scripts/ops/smoke_model.py --cells mflat --no-causal --bpe-path "$ENTOK" --set model.position=rope \
       && python scripts/ops/smoke_model.py --cells mflat --no-causal --bpe-path "$ENTOK" --set model.position=rope frozen.inject=add ; then
        run --facts mflat $EN -o frozen.trainable=true $S20 $ROPE -o train.seed=0 ${ENEXTRA:-}
        run --facts mflat $EN -o frozen.trainable=true $S20 $ROPE -o frozen.inject=add -o train.seed=0 ${ENEXTRA:-}
    else
        stamp "1  SMOKE TEST FAILED — piece 1 skipped"
    fi
fi

if on 2; then
    stamp "2  G-73: the ~38M-token Ukrainian null again, seeds 0–3"
    for s in 0 1 2 3; do
        run --facts mconst $UK -o data.corpus_fraction=0.05 -o data.bpe_path=data/$UK17.json $C $S10 -o train.seed=$s ${UKEXTRA:-}
    done
fi

if on 3; then
    stamp "3  G-73: the 151M-token Ukrainian null at 50k again, seeds 0–1"
    for s in 0 1; do
        run --facts mconst $UK -o data.corpus_fraction=0.2 -o data.bpe_path=data/$UK17.json $C -o train.seed=$s ${UKEXTRA:-}
    done
fi

if on 4; then
    stamp "4  D7: each new tokenizer whose condition is ready — null, case, lexical case, seed 0"
    python scripts/data/rederive_constants.py
    for t in $TOKS; do d7 "$t"; done
    stamp "4  D7: lexical case under BPE 17k, seed 0"
    run --facts ucaselex $UK -o data.corpus_fraction=$D7FRAC -o data.bpe_path=data/$UK17.json $C $S10 -o train.seed=0 ${UKEXTRA:-}
fi

if on 5; then
    stamp "5  G-73: the 151M-token Ukrainian null at 50k again, seeds 2–3"
    for s in 2 3; do
        run --facts mconst $UK -o data.corpus_fraction=0.2 -o data.bpe_path=data/$UK17.json $C -o train.seed=$s ${UKEXTRA:-}
    done
fi

if on 6; then
    stamp "6  D7: the tokenizers not ready before — waiting while they are prepared (at most ${WAIT_D7:-150} min)"
    deadline=$(( $(date +%s) + ${WAIT_D7:-150} * 60 ))
    python scripts/data/rederive_constants.py
    for t in $TOKS; do
        while ! ready "$t" && prep_running && [ "$(date +%s)" -lt "$deadline" ]; do sleep 120; done
        d7 "$t"
    done
fi

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
