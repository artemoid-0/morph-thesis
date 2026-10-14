#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Scale chain, prepared 14 October 2026 (journey §33.93): does the tokenizer's gain keep
# growing with model size? The queue's candidate 3 (docs/DIRECTIONS.md, "The queue"; D8),
# time-boxed by END_BY.
#
#   1  both nulls (`bpe_uk_17k`, `bpeend_uk_17k`) at width 512 — twice width 256, four
#      times width 128; 30.0M parameters, 12.6M in the core — on ~38M tokens (fraction
#      0.05), 10k steps, seeds 0–3, a seed's pair at a time, paired by seed from identical
#      initial weights. Beside the 10k rows at widths 128 (+0.0174) and 256 (+0.0254)
#                                                                        ~5.9 h  P201–P204
#   then the tables and figures (seconds).
#
# Guards, recorded before the chain runs; each skips only what depends on it:
#   ready         each condition's caches exist (prepare_condition.py --dry-run) and its
#                 smoke test passes at width 512
#   width sanity  after seed 0, each 512-wide null must end below its own 256-wide 10k null
#                 at seed 0 — a wider model that ends worse has diverged — and the pair
#                 within 0.05 of each other; otherwise seeds 1–3 are skipped
#   disk          no run starts with less than MIN_FREE_GB free
#   END_BY        a seed's pair starts only if both runs would end by then, by recorded minutes;
#                 with FILL=1, once in a chain, a pair that does not fit runs its first run
#                 alone if that one fits — its pair is finished at the next start
# None branches on an effect's size.
#
# Every run is a RESUMABLE PIECE (scripts/ops/run_ids.py --done): started again, the chain
# runs only what is unfinished.
#
#   PLAN=1 END_BY=$(date -d '+6 hours' '+%F %T') bash scripts/chains/scale_2026-10-14.sh
#   END_BY=$(date -d '+6 hours' '+%F %T') nohup systemd-inhibit --what=sleep:idle bash scripts/chains/scale_2026-10-14.sh >/dev/null 2>&1 &
# Knobs: PIECES, END_BY, FILL, PLAN, MIN_FREE_GB, WAIT_RUNS (0: no wait); MINUTES, FREE_GB (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/scale_2026-10-14.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-1}
UK17=${UK17:-bpe_uk_17k}
ENDTOK=${ENDTOK:-bpeend_uk_17k}
WIDTH=${WIDTH:-512}
MIN_FREE_GB=${MIN_FREE_GB:-2}
UK="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
args_for() {    # args_for TOKENIZER WIDTH SEED — a Ukrainian null at ~38M tokens, 10k
    local w=""
    [ "$2" != 128 ] && w="-o model.d_model=$2"
    echo "--facts mconst $UK -o data.corpus_fraction=0.05 -o data.bpe_path=data/$1.json $C $(steps 10000) $w -o train.seed=$3"
}
minutes() {     # recorded or measured minutes, one or two to spare
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    echo 44     # fit_check.py, 14 October: 39.8 projected; runs have taken 1.07 × their projection
}               # at widths 128 and 256 (11.3–11.8 against 10.8; 19.0–19.7 against 18.2)
free_gb() { [ -n "${FREE_GB:-}" ] && { echo "$FREE_GB"; return; }; df -BG --output=avail . | tail -n 1 | tr -dc '0-9'; }
DEADLINE=""
if [ -n "${END_BY:-}" ]; then
    DEADLINE=$(date -d "$END_BY" +%s) || { echo "END_BY not understood: $END_BY — not starting"; exit 1; }
fi
CLOCK=$(date +%s)
now() { if [ -n "${PLAN:-}" ]; then echo "$CLOCK"; else date +%s; fi; }
LEFT=0; PLANNED=0; PLANNED_MIN=0
STOP=""
FILLED=""
on() { case " $PIECES " in *" $1 "*) return 0 ;; esac; return 1; }
declare -A STATE

ready() {       # ready TOKENIZER — its condition at 0.05 is prepared and its smoke test passes at WIDTH; checked once
    local t=$1 out
    [ -n "${PLAN:-}" ] && return 0
    if [ -z "${STATE[$t]:-}" ]; then
        if ! out=$(python scripts/data/prepare_condition.py --corpus ukwiki --corpus-fraction 0.05 \
                   --bpe-path "data/$t.json" --dry-run 2>/dev/null) \
           || ! echo "$out" | grep -q "train cache: done" || echo "$out" | grep -qE "mconst to do|to build"; then
            echo "  [not ready] $t at 0.05 — its condition is not prepared; its sets are skipped"; STATE[$t]=no
        elif ! python scripts/ops/smoke_model.py --cells mconst --bpe-path "data/$t.json" \
                --set data.corpus=ukwiki data.corpus_fraction=0.05 model.d_model="$WIDTH"; then
            echo "  SMOKE TEST FAILED for $t at width $WIDTH — its sets are skipped"; STATE[$t]=no
        else STATE[$t]=yes; fi
    fi
    [ "${STATE[$t]}" = yes ]
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
planned() {     # planned MINUTES N WHAT
    PLANNED=$((PLANNED + $2)); PLANNED_MIN=$((PLANNED_MIN + $1)); CLOCK=$((CLOCK + $1 * 60))
    echo "  [would run] $3 — $2 run(s), ~$1 min; by $(date -d "@$CLOCK" '+%a %H:%M')"
}
pairset() {     # pairset SEED — both tokenizers' nulls at WIDTH and one seed, all or none
    local s=$1 t pending=() mins=0
    [ -n "$STOP" ] && return 0
    for t in $UK17 $ENDTOK; do python scripts/ops/run_ids.py --done $(args_for "$t" "$WIDTH" "$s") || pending+=("$t"); done
    [ ${#pending[@]} -eq 0 ] && return 0
    for t in "${pending[@]}"; do
        ready "$t" || { echo "  [skipped] seed $s: $t is not usable"; return 0; }
        mins=$((mins + $(minutes)))
    done
    if ! fits "$mins" "${pending[*]} — nulls at width $WIDTH, 10k, seed $s"; then
        if [ "${FILL:-}" = 1 ] && [ -z "$FILLED" ] && [ ${#pending[@]} -gt 1 ] \
                && fits "$(minutes)" "${pending[0]} alone — width $WIDTH, 10k, seed $s"; then
            FILLED=1; LEFT=$((LEFT + ${#pending[@]} - 1))
            echo "  [fill] ${pending[0]} alone, seed $s — its pair is finished at the next start"
            if [ -n "${PLAN:-}" ]; then planned "$(minutes)" 1 "${pending[0]} — null at width $WIDTH, 10k, seed $s (fill)"; return 0; fi
            disk_ok && $RUN $(args_for "${pending[0]}" "$WIDTH" "$s")
            return 0
        fi
        LEFT=$((LEFT + ${#pending[@]})); return 0
    fi
    if [ -n "${PLAN:-}" ]; then planned "$mins" ${#pending[@]} "${pending[*]} — nulls at width $WIDTH, 10k, seed $s"; return 0; fi
    for t in "${pending[@]}"; do disk_ok || return 0; $RUN $(args_for "$t" "$WIDTH" "$s"); done
}
bpb_of() { python scripts/ops/run_ids.py --bpb $(args_for "$@") 2>/dev/null; }   # bpb_of TOKENIZER WIDTH SEED
below() {       # below WHAT NEW REF — NEW ends below REF
    if [ -z "$2" ] || [ -z "$3" ]; then echo "  [guard] $1: no finished run to compare (${2:-none} against ${3:-none})"; return 1; fi
    if awk -v a="$2" -v b="$3" 'BEGIN { exit !(a < b) }'; then echo "  [guard] $1: $2, below $3 — passes"
    else echo "  [guard] $1: $2, NOT below $3 — fails"; return 1; fi
}
near() {        # near WHAT A B — A within 0.05 of B
    if [ -z "$2" ] || [ -z "$3" ]; then echo "  [guard] $1: no finished run to compare (${2:-none} against ${3:-none})"; return 1; fi
    if awk -v a="$2" -v b="$3" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= 0.05) }'; then
        echo "  [guard] $1: $2 against $3 — within 0.05, passes"
    else echo "  [guard] $1: $2 against $3 — NOT within 0.05, fails"; return 1; fi
}
sane_width() {  # seed 0: each WIDTH null below its 256-wide 10k null, and the pair near each other
    [ -n "${PLAN:-}" ] && return 0
    local t
    for t in $UK17 $ENDTOK; do
        below "$t, $WIDTH wide against 256 wide, seed 0" "$(bpb_of "$t" "$WIDTH" 0)" "$(bpb_of "$t" 256 0)" || return 1
    done
    near "seed 0, $ENDTOK against $UK17 at width $WIDTH" "$(bpb_of "$ENDTOK" "$WIDTH" 0)" "$(bpb_of "$UK17" "$WIDTH" 0)"
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }scale chain prepared 14 October — pieces: $PIECES; width $WIDTH${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"

if on 1; then
    stamp "1  both nulls at width $WIDTH, ~38M tokens, 10k, seed 0"
    pairset 0
    if sane_width; then
        stamp "1  width sanity passed — seeds 1–3"
        for s in 1 2 3; do pairset "$s"; done
    else
        stamp "1  WIDTH SANITY FAILED — seeds 1–3 are skipped"
    fi
fi

if [ -n "${PLAN:-}" ]; then
    stamp "PLAN: $PLANNED run(s), ~$PLANNED_MIN min of GPU, ending ~$(date -d "@$CLOCK" '+%a %F %H:%M')${END_BY:+; $LEFT run(s) left for a later start}"
    exit 0
fi
stamp "tables and figures"
bash scripts/ops/refresh_exports.sh ;
if [ "$STOP" = disk ]; then
    stamp "CHAIN STOPPED — too little disk; free space and start it again: it resumes"
elif [ "$LEFT" -gt 0 ]; then
    stamp "CHAIN ENDED BY $END_BY — $LEFT run(s) left for the next start"
else
    stamp "CHAIN COMPLETE"
fi
