#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Long chain, prepared 9 October 2026 (journey §33.73). The quiet part of the
# queue (docs/DIRECTIONS.md, "The queue"), in the order of what each settles for
# the thesis, time-boxed by END_BY: give it the hours and it runs the top of the
# queue; what does not fit is left whole for the next start, which resumes.
#
#   0  reproduction check: BPE 17k's null at 151M tokens, 5k steps, seed 0 —
#      a quarantined run (G-73) made again, which must reproduce the audit's
#      re-score of its checkpoint, 1.196103, within 0.00001; otherwise nothing
#      else starts: runs made now would not pair with the records they meet  ~6 min  P160
#   1  full training, more data: the endings-apart null at 151M tokens,
#      50k steps, seeds 0–3 (against BPE 17k's, §33.58)                   ~3.8 h  P152–P154
#   2  full training: case under endings apart, 151M, 50k, seeds 0–3      ~3.9 h  P155–P157
#   3  the five runs left on 8 October: case at 20k for seed 3 under both
#      tokenizers; BPE 32k at seed 3 — null, case, word form (~38M)      ~1.7 h  P136–P140
#   4  where the wider model's lead settles: both nulls at width 256,
#      20k steps, seeds 0–3 (~38M)                                        ~5.5 h  P158 P159
#   5  the Ukrainian re-runs: BPE 17k's nulls at ~1.5M and ~7.6M tokens
#      (10k), and at 151M with 10k, 20k and 5k budgets, seeds 0–3 —
#      fig4, then fig3, then the law's points                             ~4.2 h  P160
#   6  case in the wider model: case under both tokenizers at width 256,
#      10k, seeds 0–3 (~38M)                                              ~2.8 h  P161 P162
#   7  D3: English part of speech switched off from the start and at
#      5,000, 50k budget, seeds 0–1                                       ~3.9 h  P163
#   then the tables and figures (seconds).
#
# Guards, recorded before the chain runs; each skips only what depends on it:
#   reproduction  piece 0 (above) — stops the chain
#   151M sanity   after seed 0 of piece 1, the endings-apart null at 151M must
#                 end below its own 20k null at ~38M at seed 0 (0.9974);
#                 otherwise the rest of pieces 1 and 2 is skipped
#   width sanity  after seed 0 of piece 4, each 256-wide 20k null must end
#                 below its 256-wide 10k null at seed 0; otherwise the rest of
#                 piece 4 is skipped
#   disk          no run starts with less than MIN_FREE_GB free
#   smoke test    per condition and width, before its first run
#   END_BY        a set — a seed's runs compared with each other — starts only
#                 if all of it would end by then, by recorded minutes
# None branches on an effect's size.
#
# Every run is a RESUMABLE PIECE: a run whose cell and seed already have a
# finished result in results/ is skipped (scripts/ops/run_ids.py --done).
#
#   PLAN=1 END_BY=$(date -d '+8 hours' '+%F %T') bash scripts/chains/long_2026-10-09.sh
#       — runs nothing: what would run by then, set by set, from the records
#   END_BY=$(date -d '+8 hours' '+%F %T') nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-09.sh >/dev/null 2>&1 &
# Knobs: PIECES, PIECE3 (both | case | 32k), END_BY, PLAN, MIN_FREE_GB, WAIT_RUNS (0: no wait);
#        UKEXTRA, ENEXTRA, PEXTRA, MINUTES, FREE_GB, REPRO_BPB (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"      # PLAN too (guard.sh does it for a run)
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/long_2026-10-09.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-0 1 2 3 4 5 6 7}
UK17=${UK17:-bpe_uk_17k}
ENDTOK=${ENDTOK:-bpeend_uk_17k}
TOK32=${TOK32:-bpe_uk_32k}
MIN_FREE_GB=${MIN_FREE_GB:-1}
REPRO_BPB=${REPRO_BPB:-1.196103}     # codes_audit_20261006T151153: 82561f48b89b on the constant
UK="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
shape_args() {  # SHAPE: 50k (the default schedule), 20k, 10k, 5k; w256 and w256_20k are 4× wider
    case "$1" in
        50k) echo "" ;;
        20k) steps 20000 ;; 10k) steps 10000 ;; 5k) steps 5000 ;;
        w256) echo "$(steps 10000) -o model.d_model=256" ;;
        w256_20k) echo "$(steps 20000) -o model.d_model=256" ;;
    esac
}
args_for() {    # args_for FRACTION SHAPE TOKENIZER FACT SEED — a Ukrainian run
    echo "--facts $4 $UK -o data.corpus_fraction=$1 -o data.bpe_path=data/$3.json $C $(shape_args "$2") -o train.seed=$5 ${UKEXTRA:-}"
}
en_args() {     # en_args OFF_STEP SEED — D3: English part of speech switched off, 50k
    echo "--facts pos $EN $C -o frozen.feature_off_step=$1 -o train.seed=$2"
}
minutes() {     # minutes FRACTION SHAPE TOKENIZER FACT — recorded minutes, one or two to spare
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1:$2:$3:$4" in
        0.2:50k:*:mconst) echo 57 ;;   # 53.9–55.5 (BPE 17k's nulls)
        0.2:50k:*)        echo 59 ;;   # 54.4–58.2 (case), 54.9–64.9 (word form; one slow run)
        0.2:20k:*)        echo 24 ;;   # 21.7–23.0
        0.2:5k:*)         echo 7 ;;    # 5.4–5.5
        *:20k:*:mconst)   echo 23 ;;   # 21.9–22.1 (~38M)
        *:20k:*)          echo 24 ;;   # 22.9–23.1
        *:w256_20k:*)     echo 41 ;;   # 2 × 19.0–19.7 (Ukrainian, 10k); 36.8–37.6 (English, 20k)
        *:w256:*)         echo 21 ;;   # 19.0–19.7
        *:10k:*_32k:*)    echo 17 ;;   # 15.7–16.1
        0.002:10k:*)      echo 10 ;;   # 8.7–9.3
        *)                echo 12 ;;   # 10.8–11.8
    esac
}
EN_MINUTES=59                          # 55.2–57.1 (English part of speech, 50k)
free_gb() { [ -n "${FREE_GB:-}" ] && { echo "$FREE_GB"; return; }; df -BG --output=avail . | tail -n 1 | tr -dc '0-9'; }
DEADLINE=""
if [ -n "${END_BY:-}" ]; then
    DEADLINE=$(date -d "$END_BY" +%s) || { echo "END_BY not understood: $END_BY — not starting"; exit 1; }
fi
CLOCK=$(date +%s)                      # PLAN: the time the plan has reached
now() { if [ -n "${PLAN:-}" ]; then echo "$CLOCK"; else date +%s; fi; }
LEFT=0; PLANNED=0; PLANNED_MIN=0
STOP=""
on() { case " $PIECES " in *" $1 "*) return 0 ;; esac; return 1; }

declare -A STATE PLANNED_RUN
usable() {      # usable FRACTION SHAPE TOKENIZER — its condition is ready and its smoke test passes; checked once
    local key="$1:$2:$3" out set=""
    [ -n "${PLAN:-}" ] && return 0
    case "$2" in w256*) set="model.d_model=256" ;; esac
    if [ -z "${STATE[$key]:-}" ]; then
        if ! out=$(python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction "$1" \
                   --bpe-path "data/$3.json" ${PEXTRA:-} --dry-run 2>/dev/null) \
           || ! echo "$out" | grep -q "train cache: done" || echo "$out" | grep -qE "mconst to do|ucaselex to do|to build"; then
            echo "  [not ready] $3 at $1 — its condition is not prepared; its sets are skipped"; STATE[$key]=no
        elif ! python scripts/ops/smoke_model.py --cells mconst ucase ucaselex --bpe-path "data/$3.json" \
                --set data.corpus=ukwiki data.corpus_fraction="$1" $set; then
            echo "  SMOKE TEST FAILED for $3 at $1 ($2) — its sets are skipped"; STATE[$key]=no
        else
            STATE[$key]=yes
        fi
    fi
    [ "${STATE[$key]}" = yes ]
}
fits() {        # fits MINUTES WHAT — END_BY lets a set of MINUTES start now
    if [ -n "$DEADLINE" ] && [ $(( $(now) + $1 * 60 )) -gt "$DEADLINE" ]; then
        echo "  [left for the next chain] $2 — would end after $END_BY"; return 1
    fi
}
disk_ok() {
    [ -n "${PLAN:-}" ] && return 0
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
        stamp "DISK: $(free_gb) GB free, below $MIN_FREE_GB GB — no further run starts"; STOP=disk; return 1
    fi
}
planned() {     # planned MINUTES N WHAT — PLAN: count the set and move the plan's clock
    PLANNED=$((PLANNED + $2)); PLANNED_MIN=$((PLANNED_MIN + $1)); CLOCK=$((CLOCK + $1 * 60))
    echo "  [would run] $3 — $2 run(s), ~$1 min; by $(date -d "@$CLOCK" '+%a %H:%M')"
}
runset() {      # runset FRACTION SHAPE SEED TOKENIZER:FACT... — the set's unfinished runs, all or none
    local frac=$1 shape=$2 s=$3 item pending=() mins=0
    shift 3
    [ -n "$STOP" ] && return 0
    for item in "$@"; do
        [ -n "${PLANNED_RUN["$frac $shape $s $item"]:-}" ] && continue          # PLAN: planned earlier
        python scripts/ops/run_ids.py --done $(args_for "$frac" "$shape" "${item%%:*}" "${item#*:}" "$s") || pending+=("$item")
    done
    [ ${#pending[@]} -eq 0 ] && return 0
    for item in "${pending[@]}"; do
        usable "$frac" "$shape" "${item%%:*}" || { echo "  [skipped] seed $s, $shape at $frac: ${item%%:*} is not usable"; return 0; }
        mins=$((mins + $(minutes "$frac" "$shape" "${item%%:*}" "${item#*:}")))
    done
    fits "$mins" "${pending[*]} — $shape at $frac, seed $s" || { LEFT=$((LEFT + ${#pending[@]})); return 0; }
    if [ -n "${PLAN:-}" ]; then
        for item in "${pending[@]}"; do PLANNED_RUN["$frac $shape $s $item"]=1; done
        planned "$mins" ${#pending[@]} "${pending[*]} — $shape at $frac, seed $s"; return 0
    fi
    for item in "${pending[@]}"; do
        disk_ok || return 0
        $RUN $(args_for "$frac" "$shape" "${item%%:*}" "${item#*:}" "$s")
    done
}
en_runset() {   # en_runset SEED OFF_STEP... — D3's runs at one seed, all or none
    local s=$1 k pending=() mins=0
    shift
    [ -n "$STOP" ] && return 0
    for k in "$@"; do python scripts/ops/run_ids.py --done $(en_args "$k" "$s") || pending+=("$k"); done
    [ ${#pending[@]} -eq 0 ] && return 0
    if [ -z "${PLAN:-}" ] && [ -z "${STATE[en]:-}" ]; then
        if python scripts/ops/smoke_model.py --cells pos --bpe-path data/bpe_17k.json; then STATE[en]=yes
        else echo "  SMOKE TEST FAILED for English part of speech — D3 is skipped"; STATE[en]=no; fi
    fi
    [ -z "${PLAN:-}" ] && [ "${STATE[en]}" != yes ] && return 0
    mins=$(( ${#pending[@]} * ${MINUTES:-$EN_MINUTES} ))
    fits "$mins" "pos off at ${pending[*]} — English, 50k, seed $s" || { LEFT=$((LEFT + ${#pending[@]})); return 0; }
    if [ -n "${PLAN:-}" ]; then planned "$mins" ${#pending[@]} "pos off at ${pending[*]} — English, 50k, seed $s"; return 0; fi
    for k in "${pending[@]}"; do disk_ok || return 0; $RUN $(en_args "$k" "$s"); done
}
bpb_of() { python scripts/ops/run_ids.py --bpb $(args_for "$@") 2>/dev/null; }   # bpb_of FRACTION SHAPE TOKENIZER FACT SEED
below() {       # below WHAT NEW REF — NEW ends below REF
    if [ -z "$2" ] || [ -z "$3" ]; then echo "  [guard] $1: no finished run to compare (${2:-none} against ${3:-none})"; return 1; fi
    if awk -v a="$2" -v b="$3" 'BEGIN { exit !(a < b) }'; then echo "  [guard] $1: $2, below $3 — passes"
    else echo "  [guard] $1: $2, NOT below $3 — fails"; return 1; fi
}
sane_151() {    # the endings-apart 151M null at seed 0 ends below its own 20k null at ~38M
    [ -n "${PLAN:-}" ] && return 0
    below "$ENDTOK, 151M null at 50k" "$(bpb_of 0.2 50k "$ENDTOK" mconst 0)" "$(bpb_of 0.05 20k "$ENDTOK" mconst 0)"
}
sane_w256() {   # each 256-wide 20k null at seed 0 ends below its 256-wide 10k null
    [ -n "${PLAN:-}" ] && return 0
    local t
    for t in $UK17 $ENDTOK; do
        below "$t, 256 wide, 20k null" "$(bpb_of 0.05 w256_20k "$t" mconst 0)" "$(bpb_of 0.05 w256 "$t" mconst 0)" || return 1
    done
}
reproduces() {  # the re-run of piece 0 reproduces the audit's re-score
    [ -n "${PLAN:-}" ] && return 0
    local got
    got=$(bpb_of 0.2 5k "$UK17" mconst 0)
    if [ -z "$got" ]; then echo "  [guard] no finished re-run to compare with $REPRO_BPB"; return 1; fi
    if awk -v a="$got" -v b="$REPRO_BPB" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= 0.00001) }'; then
        echo "  [guard] the re-run ends at $got; the audit's re-score is $REPRO_BPB — reproduces"
    else
        echo "  [guard] the re-run ends at $got; the audit's re-score is $REPRO_BPB — DOES NOT REPRODUCE"; return 1
    fi
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }long chain of 9 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"

if on 0; then
    stamp "0  reproduction check: $UK17's null at 151M tokens, 5k steps, seed 0"
    runset 0.2 5k 0 "$UK17:mconst"
    if [ -z "$STOP" ] && ! reproduces; then
        if [ -n "$DEADLINE" ] && [ -z "$(bpb_of 0.2 5k "$UK17" mconst 0)" ] && [ "$LEFT" -gt 0 ]; then
            stamp "0  the check did not fit before $END_BY — nothing else starts"
        else
            stamp "0  REPRODUCTION CHECK FAILED — nothing else starts"
        fi
        STOP=reproduction
    fi
fi
if on 1; then
    stamp "1  full training, more data: the $ENDTOK null at 151M tokens, 50k, seed 0"
    runset 0.2 50k 0 "$ENDTOK:mconst"
    if sane_151; then
        stamp "1  151M sanity passed — seeds 1–3"
        for s in 1 2 3; do runset 0.2 50k "$s" "$ENDTOK:mconst"; done
    else
        stamp "1  151M SANITY FAILED — the rest of pieces 1 and 2 is skipped"
    fi
fi
if on 2; then
    stamp "2  full training: case under $ENDTOK at 151M tokens, 50k, seeds 0–3"
    if sane_151; then
        for s in 0 1 2 3; do runset 0.2 50k "$s" "$ENDTOK:ucase"; done
    else
        stamp "2  skipped — the 151M sanity guard does not pass"
    fi
fi
if on 3; then
    stamp "3  left on 8 October: case at 20k for seed 3; $TOK32 at seed 3"
    case "${PIECE3:-both}" in both|case) runset 0.05 20k 3 "$UK17:ucase" "$ENDTOK:ucase" ;; esac
    case "${PIECE3:-both}" in both|32k)  runset 0.05 10k 3 "$TOK32:mconst" "$TOK32:ucase" "$TOK32:ucaselex" ;; esac
fi
if on 4; then
    stamp "4  where the wider model's lead settles: both nulls at width 256, 20k, seed 0"
    runset 0.05 w256_20k 0 "$UK17:mconst" "$ENDTOK:mconst"
    if sane_w256; then
        stamp "4  width sanity passed — seeds 1–3"
        for s in 1 2 3; do runset 0.05 w256_20k "$s" "$UK17:mconst" "$ENDTOK:mconst"; done
    else
        stamp "4  WIDTH SANITY FAILED — the rest of piece 4 is skipped"
    fi
fi
if on 5; then
    stamp "5  the Ukrainian re-runs: $UK17's nulls at ~1.5M and ~7.6M tokens, then at 151M (10k, 20k, 5k)"
    for s in 0 1 2 3; do runset 0.002 10k "$s" "$UK17:mconst"; runset 0.01 10k "$s" "$UK17:mconst"; done
    for s in 0 1 2 3; do runset 0.2 10k "$s" "$UK17:mconst"; runset 0.2 20k "$s" "$UK17:mconst"; done
    for s in 0 1 2 3; do runset 0.2 5k "$s" "$UK17:mconst"; done
fi
if on 6; then
    stamp "6  case in the wider model: case under both tokenizers at width 256, 10k, seeds 0–3"
    for s in 0 1 2 3; do runset 0.05 w256 "$s" "$UK17:ucase" "$ENDTOK:ucase"; done
fi
if on 7; then
    stamp "7  D3: English part of speech off from the start and at 5,000, 50k, seeds 0–1"
    for s in 0 1; do en_runset "$s" 0 5000; done
fi

if [ -n "${PLAN:-}" ]; then
    stamp "PLAN: $PLANNED run(s), ~$PLANNED_MIN min of GPU, ending ~$(date -d "@$CLOCK" '+%a %F %H:%M')${END_BY:+; $LEFT run(s) left for a later start}"
    exit 0
fi
stamp "tables and figures"
python scripts/analysis/temporal_profile.py ;
bash scripts/ops/refresh_exports.sh ;
if [ "$STOP" = disk ]; then
    stamp "CHAIN STOPPED — too little disk; free space and start it again: it resumes"
elif [ "$STOP" = reproduction ]; then
    stamp "CHAIN STOPPED at the reproduction check"
elif [ "$LEFT" -gt 0 ]; then
    stamp "CHAIN ENDED BY $END_BY — $LEFT run(s) left for the next start"
else
    stamp "CHAIN COMPLETE"
fi
