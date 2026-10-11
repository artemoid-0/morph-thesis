#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Long chain, 8 October 2026 (journey §33.67). Does morphology in the tokenizer
# last where morphology in the embedding fades? ~9.7 h of GPU in 4 pieces, in
# priority order, time-boxed by END_BY; a CPU track beside them.
#
# Ukrainian at ~38M tokens (fraction 0.05), as in D7; every comparison paired
# by seed. The null (`mconst`) under BPE 17k and under BPE 17k that keeps
# endings apart, along the two axes where features fade:
#
#   1  the tokenizer at twice the budget: both nulls, 20k steps, seeds 0–3   ~3.1 h  P132 P133
#   2  the tokenizer in a 4× wider model: both nulls, width 256, 10k,
#      seeds 0–3                                                           ~2.6 h  P134 P135
#   3  case on top at 20k: case under both tokenizers, seeds 0–3           ~3.1 h  P136–P139
#   4  D7: BPE 32k at seed 3 — the null, case and word form, 10k           ~50 min P140
#   CPU track: BPE 17k with endings apart at fraction 0.2 (~151M-token
#      text), prepared for the next chain — tagging ~3.5–5.5 h             P141 P142
#   then tables and the learning figure (seconds).
#
# Guards, recorded before the chain ran; each skips only what depends on it:
#   20k sanity    after seed 0 of piece 1, each 20k null must end below its own
#                 10k null at seed 0 (BPE 17k 1.0891, endings apart 1.0714);
#                 otherwise the rest of pieces 1 and 3 is skipped
#   width sanity  after seed 0 of piece 2, each 256-wide null must end below
#                 the 128-wide null of its tokenizer at seed 0; otherwise the
#                 rest of piece 2 is skipped
#   disk          no run starts with less than MIN_FREE_GB free; the CPU track
#                 starts only with CPU_FREE_GB free (the condition takes ~8.8 GB)
#   smoke test    per tokenizer and width, before its first run
#   END_BY        a set — a seed's runs that are compared with each other —
#                 starts only if all of it would end by then, by recorded
#                 minutes; what does not fit is left whole for the next chain
#
# Every run is a RESUMABLE PIECE: a run whose cell and seed already have a
# finished result in results/ is skipped (scripts/ops/run_ids.py --done). Stop
# the chain at any time; start it again and it carries on where it stopped.
#
#   END_BY=$(date -d '+8 hours' '+%F %T') nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-08.sh >/dev/null 2>&1 &
#   PIECES="3 4" nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-08.sh >/dev/null 2>&1 &
# Knobs: PIECES, END_BY, MIN_FREE_GB, CPU_FREE_GB, CPU_TRACK (0: never), WAIT_RUNS
#        (0: no wait), FRAC, CPUFRAC; UKEXTRA, PEXTRA, MINUTES, FREE_GB (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
LOG=logs/long_2026-10-08.log
CPULOG=logs/long_2026-10-08_cpu.log
exec > >(tee -a "$LOG") 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-1 2 3 4}
UK17=${UK17:-bpe_uk_17k}
ENDTOK=${ENDTOK:-bpeend_uk_17k}
TOK32=${TOK32:-bpe_uk_32k}
FRAC=${FRAC:-0.05}
CPUFRAC=${CPUFRAC:-0.2}
MIN_FREE_GB=${MIN_FREE_GB:-1}
CPU_FREE_GB=${CPU_FREE_GB:-12}
UK="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S10="-o train.max_steps=10000 -o train.lr_horizon=10000"
S20="-o train.max_steps=20000 -o train.lr_horizon=20000"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

shape_args() {  # 10k: the D7 budget; 20k: twice it; w256: 4× wider, 10k
    case "$1" in 10k) echo "$S10" ;; 20k) echo "$S20" ;; w256) echo "$S10 -o model.d_model=256" ;; esac
}
args_for() {    # args_for SHAPE TOKENIZER FACT SEED
    echo "--facts $3 $UK -o data.corpus_fraction=$FRAC -o data.bpe_path=data/$2.json $C $(shape_args "$1") -o train.seed=$4 ${UKEXTRA:-}"
}
minutes() {     # a run of SHAPE under TOKENIZER, in recorded minutes, with one or two to spare
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1:$2" in
        20k:*)     echo 25 ;;      # twice 10.8–11.8 (0.05, 10k); 21.7–22.9 at 0.2, 20k
        w256:*)    echo 21 ;;      # 18.5–20.2: English, 256 wide, 10k
        10k:*_32k) echo 17 ;;      # 15.7–16.1
        *)         echo 12 ;;
    esac
}
free_gb() { [ -n "${FREE_GB:-}" ] && { echo "$FREE_GB"; return; }; df -BG --output=avail . | tail -n 1 | tr -dc '0-9'; }
DEADLINE=""
if [ -n "${END_BY:-}" ]; then
    DEADLINE=$(date -d "$END_BY" +%s) || { echo "END_BY not understood: $END_BY — not starting"; exit 1; }
fi
LEFT=0
STOP=""
on() { case " $PIECES " in *" $1 "*) return 0 ;; esac; return 1; }

declare -A STATE
usable() {      # usable SHAPE TOKENIZER — its condition is ready and its smoke test passes; checked once
    local key="$1:$2" t=$2 out set=""
    [ "$1" = w256 ] && set="model.d_model=256"
    if [ -z "${STATE[$key]:-}" ]; then
        if ! out=$(python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction "$FRAC" \
                   --bpe-path "data/$t.json" ${PEXTRA:-} --dry-run 2>/dev/null) \
           || ! echo "$out" | grep -q "train cache: done" || echo "$out" | grep -qE "to do|to build"; then
            echo "  [not ready] $t — its condition is not prepared; its sets are skipped"; STATE[$key]=no
        elif ! python scripts/ops/smoke_model.py --cells mconst ucase ucaselex --bpe-path "data/$t.json" \
                --set data.corpus=ukwiki data.corpus_fraction="$FRAC" $set; then
            echo "  SMOKE TEST FAILED for $t ($1) — its sets are skipped"; STATE[$key]=no
        else
            STATE[$key]=yes
        fi
    fi
    [ "${STATE[$key]}" = yes ]
}
runset() {      # runset SHAPE SEED TOKENIZER:FACT... — the set's unfinished runs, all or none
    local shape=$1 s=$2 item pending=() mins=0
    shift 2
    [ -n "$STOP" ] && return 0
    for item in "$@"; do
        python scripts/ops/run_ids.py --done $(args_for "$shape" "${item%%:*}" "${item#*:}" "$s") || pending+=("$item")
    done
    [ ${#pending[@]} -eq 0 ] && return 0
    for item in "${pending[@]}"; do
        usable "$shape" "${item%%:*}" || { echo "  [skipped] seed $s, $shape: ${item%%:*} is not usable"; return 0; }
        mins=$((mins + $(minutes "$shape" "${item%%:*}")))
    done
    if [ -n "$DEADLINE" ] && [ $(( $(date +%s) + mins * 60 )) -gt "$DEADLINE" ]; then
        echo "  [left for the next chain] ${pending[*]} — $shape, seed $s — would end after $END_BY"
        LEFT=$((LEFT + ${#pending[@]})); return 0
    fi
    for item in "${pending[@]}"; do
        if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
            stamp "DISK: $(free_gb) GB free, below $MIN_FREE_GB GB — no further run starts"; STOP=disk; return 0
        fi
        $RUN $(args_for "$shape" "${item%%:*}" "${item#*:}" "$s")
    done
}
bpb_of() { python scripts/ops/run_ids.py --bpb $(args_for "$@") 2>/dev/null; }   # bpb_of SHAPE TOKENIZER FACT SEED
sane() {        # sane SHAPE — at seed 0, each tokenizer's null under SHAPE ends below its 10k null
    local t new ref
    for t in $UK17 $ENDTOK; do
        new=$(bpb_of "$1" "$t" mconst 0); ref=$(bpb_of 10k "$t" mconst 0)
        if [ -z "$new" ] || [ -z "$ref" ]; then
            echo "  [guard] $t: no finished null to compare ($1: ${new:-none}; 10k: ${ref:-none})"; return 1
        fi
        if awk -v a="$new" -v b="$ref" 'BEGIN { exit !(a < b) }'; then
            echo "  [guard] $t: $1 null $new, below the 10k null $ref — passes"
        else
            echo "  [guard] $t: $1 null $new, NOT below the 10k null $ref — fails"; return 1
        fi
    done
}

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "long chain of 8 October starts — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"

# ── CPU: the endings-apart condition at fraction 0.2, for the next chain ─────
CPU_PID=""
if [ "${CPU_TRACK:-1}" != 1 ]; then
    stamp "CPU track not started: CPU_TRACK=${CPU_TRACK}"
elif pgrep -f "scripts/data/prepare_condition.py" >/dev/null; then
    stamp "CPU track not started: a preparation is already running"
elif [ "$(free_gb)" -lt "$CPU_FREE_GB" ]; then
    stamp "CPU track not started: $(free_gb) GB free, below the $CPU_FREE_GB GB it needs"
else
    stamp "CPU track starts: $ENDTOK at fraction $CPUFRAC (log: $CPULOG)"
    (
        echo "=== $(date '+%F %T')  prepare $ENDTOK at $CPUFRAC"
        nice -n 10 python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction "$CPUFRAC" \
            --bpe-path "data/$ENDTOK.json" ${PEXTRA:-} || echo "=== $(date '+%F %T')  $ENDTOK at $CPUFRAC FAILED"
        echo "=== $(date '+%F %T')  CPU TRACK DONE"
    ) >> "$CPULOG" 2>&1 &
    CPU_PID=$!
fi

if on 1; then
    stamp "1  the tokenizer at twice the budget: the null under $UK17 and under $ENDTOK, 20k, seed 0"
    runset 20k 0 "$UK17:mconst" "$ENDTOK:mconst"
    if sane 20k; then
        stamp "1  20k sanity passed — seeds 1–3"
        for s in 1 2 3; do runset 20k "$s" "$UK17:mconst" "$ENDTOK:mconst"; done
    else
        stamp "1  20k SANITY FAILED — the rest of pieces 1 and 3 is skipped"
    fi
fi
if on 2; then
    stamp "2  the tokenizer in a 4× wider model: both nulls at width 256, 10k, seed 0"
    runset w256 0 "$UK17:mconst" "$ENDTOK:mconst"
    if sane w256; then
        stamp "2  width sanity passed — seeds 1–3"
        for s in 1 2 3; do runset w256 "$s" "$UK17:mconst" "$ENDTOK:mconst"; done
    else
        stamp "2  WIDTH SANITY FAILED — the rest of piece 2 is skipped"
    fi
fi
if on 3; then
    stamp "3  case on top at 20k: case under $UK17 and under $ENDTOK, seeds 0–3"
    if sane 20k; then
        for s in 0 1 2 3; do runset 20k "$s" "$UK17:ucase" "$ENDTOK:ucase"; done
    else
        stamp "3  skipped — the 20k sanity guard does not pass"
    fi
fi
if on 4; then
    stamp "4  D7: $TOK32 at seed 3 — the null, case and word form, 10k"
    runset 10k 3 "$TOK32:mconst" "$TOK32:ucase" "$TOK32:ucaselex"
fi

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
bash scripts/ops/refresh_exports.sh ;
if [ -n "$CPU_PID" ] && kill -0 "$CPU_PID" 2>/dev/null; then
    stamp "GPU work over — waiting for the CPU track (log: $CPULOG)"
    wait "$CPU_PID"
fi
[ -n "$CPU_PID" ] && stamp "CPU track: $(grep -E 'CONDITION READY|FAILED' "$CPULOG" | tail -n 1 | cut -c1-100)"
if [ "$STOP" = disk ]; then
    stamp "CHAIN STOPPED — too little disk; free space and start it again: it resumes"
elif [ "$LEFT" -gt 0 ]; then
    stamp "CHAIN ENDED BY $END_BY — $LEFT run(s) left for the next chain"
else
    stamp "CHAIN COMPLETE"
fi
