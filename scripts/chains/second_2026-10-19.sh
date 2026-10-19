#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 19 October 2026 (journey §33.117): the thesis's list of 19 October,
# item 4 — the exposure comparison in a wider model, read on the law's axis.
#
#   w512red       width 512, ~38M tokens, 10k, seeds 0–3, under BPE 17k and endings apart: the null
#                 (twins of the thesis machine's), case and the empty slot              P264–P266
#
# Guards as in second_followup_2026-10-18.sh: ready; twin (the first seed's null within TWIN_TOL of
# the thesis machine's); gpu, disk, END_BY. Every run is a RESUMABLE PIECE. No tables here: exports/
# is refreshed on the thesis machine, which holds every record.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-19.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"      # PLAN on the thesis machine
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/second_2026-10-19.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-w512red}
MIN_FREE_GB=${MIN_FREE_GB:-5}
MIN_GPU_GB=${MIN_GPU_GB:-6}
TWIN_TOL=${TWIN_TOL:-0.01}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 nice -n 10 python -m morphlm.cli run --pipeline bpe"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }

# the cells: each prints a run's arguments for a seed
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
S10="$(steps 10000)"; W="-o model.d_model=512"
tok() { echo "$U -o data.corpus_fraction=$1 -o data.bpe_path=data/$2.json"; }
null_b_w512()  { echo "--facts mconst $(tok 0.05 bpe_uk_17k) $C $S10 $W -o train.seed=$1"; }
case_b_w512()  { echo "--facts ucase $(tok 0.05 bpe_uk_17k) $C $S10 $W -o train.seed=$1"; }
empty_b_w512() { echo "--facts mconst $(tok 0.05 bpe_uk_17k) $C $S10 $W -o frozen.feature_off_step=0 -o train.seed=$1"; }
null_e_w512()  { echo "--facts mconst $(tok 0.05 bpeend_uk_17k) $C $S10 $W -o train.seed=$1"; }
case_e_w512()  { echo "--facts ucase $(tok 0.05 bpeend_uk_17k) $C $S10 $W -o train.seed=$1"; }
empty_e_w512() { echo "--facts mconst $(tok 0.05 bpeend_uk_17k) $C $S10 $W -o frozen.feature_off_step=0 -o train.seed=$1"; }
# the thesis machine's run of the piece's first cell at its first seed (result.json)
REF_w512red=0.880271    # 4d86892143a8, BPE 17k's null at width 512 and 10k, seed 0
minutes() {     # on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        w512red) echo 18 ;;             # width 512: 31.6 min for 20k (17 October), so ~16 min for 10k
    esac
}
free_gb() { [ -n "${FREE_GB:-}" ] && { echo "$FREE_GB"; return; }; df -BG --output=avail . | tail -n 1 | tr -dc '0-9'; }
gpu_free_gb() {
    [ -n "${GPU_FREE_GB:-}" ] && { echo "$GPU_FREE_GB"; return; }
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}'
}
DEADLINE=""
if [ -n "${END_BY:-}" ]; then
    DEADLINE=$(date -d "$END_BY" +%s) || { echo "END_BY not understood: $END_BY — not starting"; exit 1; }
fi
CLOCK=$(date +%s)
now() { if [ -n "${PLAN:-}" ]; then echo "$CLOCK"; else date +%s; fi; }
LEFT=0; PLANNED=0; PLANNED_MIN=0; STOP=""
declare -A STATE DONE_PIECE

fits() {
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
gpu_ok() {      # wait while someone else holds the GPU; give up at END_BY
    [ -n "${PLAN:-}" ] && return 0
    local f
    while f=$(gpu_free_gb); [ "${f:-0}" -lt "$MIN_GPU_GB" ]; do
        if [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
            stamp "GPU: ${f:-?} GB free, below $MIN_GPU_GB GB, until END_BY — no further run starts"; STOP=gpu; return 1
        fi
        echo "  [gpu] ${f:-?} GB free, below $MIN_GPU_GB GB — waiting five minutes ($(date '+%T'))"
        sleep "${GPU_WAIT_SECONDS:-300}"
    done
}
planned() {
    PLANNED=$((PLANNED + $2)); PLANNED_MIN=$((PLANNED_MIN + $1)); CLOCK=$((CLOCK + $1 * 60))
    echo "  [would run] $3 — $2 run(s), ~$1 min; by $(date -d "@$CLOCK" '+%a %H:%M')"
}
ready() {       # ready ARGSTRING — every split's ids and codes for these run arguments exist; checked once a cell
    local key=${1% -o train.seed=*}
    [ -n "${PLAN:-}" ] && return 0
    if [ -z "${STATE[$key]:-}" ]; then
        if python scripts/ops/cells_ready.py $1; then STATE[$key]=yes
        else echo "  [not ready] ${key} — its caches are not all here"; STATE[$key]=no; fi
    fi
    [ "${STATE[$key]}" = yes ]
}
runset() {      # runset KIND SEED ARGSTRING... — one seed's runs, all or none
    local kind=$1 s=$2 a pending=() mins=0
    shift 2
    [ -n "$STOP" ] && return 0
    for a in "$@"; do python scripts/ops/run_ids.py --done $a || pending+=("$a"); done
    [ ${#pending[@]} -eq 0 ] && return 0
    for a in "${pending[@]}"; do ready "$a" || { echo "  [skipped] $kind, seed $s"; return 0; }; done
    mins=$(( ${#pending[@]} * $(minutes "$kind") ))
    fits "$mins" "$kind, seed $s" || { LEFT=$((LEFT + ${#pending[@]})); return 0; }
    if [ -n "${PLAN:-}" ]; then planned "$mins" ${#pending[@]} "$kind, seed $s"; return 0; fi
    for a in "${pending[@]}"; do disk_ok && gpu_ok || return 0; $RUN $a; done
}
bpb() { python scripts/ops/run_ids.py --bpb $1 2>/dev/null; }
within() {      # within LABEL GOT WANT TOL WHAT
    if [ -n "${PLAN:-}" ]; then return 0; fi
    if [ -z "$2" ] || [ -z "$3" ]; then echo "  [guard] $1: no finished run to compare (${2:-none} against ${3:-none})"; return 1; fi
    if awk -v a="$2" -v b="$3" -v t="$4" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= t) }'; then
        echo "  [guard] $1: $2 against $5 $3 — within $4, passes"
    else echo "  [guard] $1: $2 against $5 $3 — NOT within $4, fails"; return 1; fi
}
cells() {       # cells SEED FN... — each cell's arguments for a seed, one per line
    local s=$1 f
    shift
    for f in "$@"; do "$f" "$s"; done
}
twinpiece() {   # twinpiece NAME KIND "SEEDS" REF FN... — every cell at each seed; after the first
                # seed, the first cell's run must end within TWIN_TOL of REF
    local name=$1 kind=$2 seeds=$3 ref=$4 first s
    local -a args
    shift 4
    first=${seeds%% *}
    local rest=""; [[ "$seeds" == *" "* ]] && rest=${seeds#* }    # a piece of one seed has none left
    stamp "$name, seed $first"
    mapfile -t args < <(cells "$first" "$@")
    runset "$kind" "$first" "${args[@]}"
    if within "$name, seed $first" "$(bpb "${args[0]}")" "$ref" "$TWIN_TOL" "the thesis machine's"; then
        stamp "$name: twin check passed — seeds ${rest:-(none left)}"
        for s in $rest; do
            mapfile -t args < <(cells "$s" "$@")
            runset "$kind" "$s" "${args[@]}"
        done
    else
        stamp "$name: TWIN CHECK FAILED — its other seeds are skipped"
    fi
}
run_piece() {
    case "$1" in
        w512red) twinpiece "w512red: item 4 — case, the null and the empty slot at width 512, 10k, under BPE 17k and endings apart" \
                     w512red "0 1 2 3" "$REF_w512red" null_b_w512 case_b_w512 empty_b_w512 null_e_w512 case_e_w512 empty_e_w512 ;;
    esac
}
if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }chain prepared 19 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"
for p in $PIECES; do run_piece "$p"; done

if [ -n "${PLAN:-}" ]; then
    stamp "PLAN: $PLANNED run(s), ~$PLANNED_MIN min of GPU, ending ~$(date -d "@$CLOCK" '+%a %F %H:%M')${END_BY:+; $LEFT run(s) left for a later start}"
    exit 0
fi
case "$STOP" in
    disk) stamp "CHAIN STOPPED — too little disk; free space and start it again: it resumes" ;;
    gpu)  stamp "CHAIN STOPPED — the GPU stayed busy until END_BY; start it again: it resumes" ;;
    *) if [ "$LEFT" -gt 0 ]; then stamp "CHAIN ENDED BY $END_BY — $LEFT run(s) left for the next start"
       else stamp "CHAIN COMPLETE"; fi ;;
esac
