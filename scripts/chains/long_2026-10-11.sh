#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Long chain, prepared 11 October 2026 (journey §33.85): how far the tokenizer's
# gain travels — to English, and to a second Ukrainian corpus. The queue's
# candidates 2 and 3 (docs/DIRECTIONS.md, "The queue"), time-boxed by END_BY.
#
#   1  English with endings apart: plain baselines under `bpeend_17k` (WikiText-103,
#      en_endings_v1) at 10k, seeds 0–3, paired with BPE 17k's plain baselines at
#      10k (bpe_baseline_bpe_17k_st10k, seeds 0–3, 3 October)          ~52 min  P189–P191
#   2  UberText fiction: both tokenizers' nulls (`bpe_ukf_17k`, `bpeend_ukf_17k`,
#      trained on fiction) at fraction 0.1 (~38M tokens), 10k, holdout 500,
#      seeds 0–3 — a seed's pair at a time                              ~2.0 h  P192–P194
#   3  English with endings apart at 20k, seeds 0–3, paired with BPE 17k's
#      plain baselines at 20k (3 October) — a follow-up, run only when named   ~1.6 h
#   4  English with endings apart at full training (added 12 October, §33.90): 50k,
#      seeds 0–3, paired with BPE 17k's plain baselines at 50k (seeds 0–3) ~3.8 h  P198–P200
#   then the tables and figures (seconds).
#
# Guards, recorded before the chain runs; each skips only what depends on it:
#   ready         a condition's caches exist (English: the ids streams; fiction:
#                 prepare_condition.py --dry-run, and its smoke test passes)
#   sanity        after seed 0 of piece 1, the endings-apart baseline must end within
#                 0.05 of BPE 17k's at seed 0 (1.5266) — a broken stream would not;
#                 otherwise the rest of piece 1 is skipped. Piece 2's seed 0 pair
#                 likewise within 0.05 of each other; piece 4's seed 0 within 0.05 of
#                 BPE 17k's at 50k (1.3477), or seeds 1–3 are skipped
#   disk          no run starts with less than MIN_FREE_GB free
#   END_BY        a set starts only if all of it would end by then, by recorded minutes
# None branches on an effect's size.
#
# Every run is a RESUMABLE PIECE (scripts/ops/run_ids.py --done).
#
#   PLAN=1 END_BY=$(date -d '+4 hours' '+%F %T') bash scripts/chains/long_2026-10-11.sh
#   PIECES="1 2" END_BY=$(date -d '+4 hours' '+%F %T') nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-11.sh >/dev/null 2>&1 &
# Knobs: PIECES, END_BY, PLAN, MIN_FREE_GB, WAIT_RUNS (0: no wait); FIC_ONLY (piece 2, one tokenizer's
#        cell — its seed-0 sanity check waits for the other's); MINUTES, FREE_GB (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/long_2026-10-11.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-1 2}
EN17=${EN17:-bpe_17k}
ENEND=${ENEND:-bpeend_17k}
F17=${F17:-bpe_ukf_17k}
FEND=${FEND:-bpeend_ukf_17k}
FFRAC=${FFRAC:-0.1}
MIN_FREE_GB=${MIN_FREE_GB:-1}
FIC="-o data.corpus=ukfiction -o data.corpus_holdout=500 -o data.spacy_model=uk_core_news_sm"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
en_args() {     # en_args TOKENIZER STEPS SEED — an English plain baseline
    echo "-o data.bpe_path=data/$1.json $C $(steps "$2") -o train.seed=$3"
}
fic_args() {    # fic_args TOKENIZER SEED — a fiction null at FFRAC, 10k
    echo "--facts mconst $FIC -o data.corpus_fraction=$FFRAC -o data.bpe_path=data/$1.json $C $(steps 10000) -o train.seed=$2"
}
minutes() {     # recorded minutes, one or two to spare
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        en10k) echo 13 ;;    # 10.9–12.1 (BPE 17k's plain baselines, 10k)
        en20k) echo 24 ;;    # 21.7–21.8 (20k)
        en50k) echo 58 ;;    # 55.0–59.6, median 57.0 (BPE 17k's plain baselines, 50k)
        fic)   echo 15 ;;    # 10.8–11.8 at ~38M on ukwiki; fiction's validation split is 2.4× larger
    esac
}
free_gb() { [ -n "${FREE_GB:-}" ] && { echo "$FREE_GB"; return; }; df -BG --output=avail . | tail -n 1 | tr -dc '0-9'; }
DEADLINE=""
if [ -n "${END_BY:-}" ]; then
    DEADLINE=$(date -d "$END_BY" +%s) || { echo "END_BY not understood: $END_BY — not starting"; exit 1; }
fi
CLOCK=$(date +%s)
now() { if [ -n "${PLAN:-}" ]; then echo "$CLOCK"; else date +%s; fi; }
LEFT=0; PLANNED=0; PLANNED_MIN=0
STOP=""
on() { case " $PIECES " in *" $1 "*) return 0 ;; esac; return 1; }
declare -A STATE

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
en_ready() {    # the English condition's ids streams exist; checked once
    [ -n "${PLAN:-}" ] && return 0
    if [ -z "${STATE[en]:-}" ]; then
        if ! python - "$ENEND" <<'EOF'
import os, sys
sys.path.insert(0, '.')
from morphlm.config import ExperimentConfig, apply_overrides
from morphlm.data.stream import cache_stem
cfg = apply_overrides(ExperimentConfig(), ['data.pipeline=bpe', f'data.bpe_path=data/{sys.argv[1]}.json'])
missing = [s for s in ('train', 'validation')
           if not os.path.exists(os.path.join('data', 'cache', cache_stem('bpe', s, None, cfg.surface_spec()) + '.npy'))]
sys.exit(f'  not ready: no {missing} stream for {sys.argv[1]}' if missing else 0)
EOF
        then echo "  [not ready] English $ENEND — piece 1 and 3 are skipped"; STATE[en]=no
        else STATE[en]=yes; fi       # a plain baseline has no feature block to smoke-test
    fi
    [ "${STATE[en]}" = yes ]
}
fic_ready() {   # a fiction condition is prepared and its smoke test passes; checked once per tokenizer
    local t=$1 out
    [ -n "${PLAN:-}" ] && return 0
    if [ -z "${STATE[$t]:-}" ]; then
        if ! out=$(python scripts/data/prepare_condition.py --corpus ukfiction --corpus-fraction "$FFRAC" \
                   --corpus-holdout 500 --bpe-path "data/$t.json" --dry-run 2>/dev/null) \
           || ! echo "$out" | grep -q "train cache: done" || echo "$out" | grep -qE "mconst to do|to build"; then
            echo "  [not ready] $t at $FFRAC — its condition is not prepared; piece 2 is skipped"; STATE[$t]=no
        elif ! python scripts/ops/smoke_model.py --cells mconst --bpe-path "data/$t.json" \
                --set data.corpus=ukfiction data.corpus_fraction="$FFRAC" data.corpus_holdout=500; then
            echo "  SMOKE TEST FAILED for $t at $FFRAC — piece 2 is skipped"; STATE[$t]=no
        else STATE[$t]=yes; fi
    fi
    [ "${STATE[$t]}" = yes ]
}
en_run() {      # en_run STEPS SEED KIND — one English baseline, if unfinished
    local st=$1 s=$2 kind=$3
    [ -n "$STOP" ] && return 0
    python scripts/ops/run_ids.py --done $(en_args "$ENEND" "$st" "$s") && return 0
    en_ready || return 0
    fits "$(minutes "$kind")" "$ENEND plain baseline, $st steps, seed $s" || { LEFT=$((LEFT + 1)); return 0; }
    if [ -n "${PLAN:-}" ]; then planned "$(minutes "$kind")" 1 "$ENEND plain baseline, $st steps, seed $s"; return 0; fi
    disk_ok || return 0
    $RUN $(en_args "$ENEND" "$st" "$s")
}
fic_set() {     # fic_set SEED — the fiction tokenizers' nulls at one seed (FIC_ONLY: one tokenizer), all or none
    local s=$1 t pending=() mins=0
    [ -n "$STOP" ] && return 0
    for t in ${FIC_ONLY:-$F17 $FEND}; do python scripts/ops/run_ids.py --done $(fic_args "$t" "$s") || pending+=("$t"); done
    [ ${#pending[@]} -eq 0 ] && return 0
    for t in "${pending[@]}"; do
        fic_ready "$t" || return 0
        mins=$((mins + $(minutes fic)))
    done
    fits "$mins" "${pending[*]} — fiction nulls at $FFRAC, 10k, seed $s" || { LEFT=$((LEFT + ${#pending[@]})); return 0; }
    if [ -n "${PLAN:-}" ]; then planned "$mins" ${#pending[@]} "${pending[*]} — fiction nulls at $FFRAC, 10k, seed $s"; return 0; fi
    for t in "${pending[@]}"; do disk_ok || return 0; $RUN $(fic_args "$t" "$s"); done
}
near() {        # near WHAT A B — A within 0.05 of B
    if [ -z "$2" ] || [ -z "$3" ]; then echo "  [guard] $1: no finished run to compare (${2:-none} against ${3:-none})"; return 1; fi
    if awk -v a="$2" -v b="$3" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= 0.05) }'; then
        echo "  [guard] $1: $2 against $3 — within 0.05, passes"
    else echo "  [guard] $1: $2 against $3 — NOT within 0.05, fails"; return 1; fi
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }long chain prepared 11 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"

if on 1; then
    stamp "1  English with endings apart: $ENEND plain baselines at 10k, seed 0"
    en_run 10000 0 en10k
    if [ -n "${PLAN:-}" ] || near "English seed 0, $ENEND against $EN17 (10k)" \
            "$(python scripts/ops/run_ids.py --bpb $(en_args "$ENEND" 10000 0) 2>/dev/null)" \
            "$(python scripts/ops/run_ids.py --bpb $(en_args "$EN17" 10000 0) 2>/dev/null)"; then
        stamp "1  sanity passed — seeds 1–3"
        for s in 1 2 3; do en_run 10000 "$s" en10k; done
    else
        stamp "1  SANITY FAILED — the rest of piece 1, and piece 3, are skipped"; STATE[en]=no
    fi
fi
if on 2; then
    stamp "2  UberText fiction: $F17 and $FEND nulls at $FFRAC, 10k, seed 0"
    fic_set 0
    a=$(python scripts/ops/run_ids.py --bpb $(fic_args "$FEND" 0) 2>/dev/null)
    b=$(python scripts/ops/run_ids.py --bpb $(fic_args "$F17" 0) 2>/dev/null)
    if [ -z "${PLAN:-}" ] && [ -n "${FIC_ONLY:-}" ] && { [ -z "$a" ] || [ -z "$b" ]; } && [ -n "$a$b" ]; then
        stamp "2  sanity deferred — only $FIC_ONLY runs now; its seed 0 is checked against the other's when both exist"
        for s in 1 2 3; do fic_set "$s"; done
    elif [ -n "${PLAN:-}" ] || near "fiction seed 0, $FEND against $F17" "$a" "$b"; then
        stamp "2  sanity passed — seeds 1–3"
        for s in 1 2 3; do fic_set "$s"; done
    else
        stamp "2  SANITY FAILED — the rest of piece 2 is skipped"
    fi
fi
if on 3; then
    stamp "3  English with endings apart at 20k, seeds 0–3"
    for s in 0 1 2 3; do en_run 20000 "$s" en20k; done
fi
if on 4; then
    stamp "4  English with endings apart at full training (50k), seed 0"
    en_run 50000 0 en50k
    if [ -n "${PLAN:-}" ] || near "English seed 0, $ENEND against $EN17 (50k)" \
            "$(python scripts/ops/run_ids.py --bpb $(en_args "$ENEND" 50000 0) 2>/dev/null)" \
            "$(python scripts/ops/run_ids.py --bpb $(en_args "$EN17" 50000 0) 2>/dev/null)"; then
        stamp "4  sanity passed — seeds 1–3"
        for s in 1 2 3; do en_run 50000 "$s" en50k; done
    else
        stamp "4  SANITY FAILED — the rest of piece 4 is skipped"
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
