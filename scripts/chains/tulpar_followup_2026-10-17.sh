#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The thesis machine's afternoon, prepared 17 October 2026 (journey §33.105): what follows
# tulpar_2026-10-17.sh, chosen when the empty slot and the twins have reported. PIECES picks them;
# each is a cell family at full training (151M tokens or WikiText-103's training split, 50k steps),
# paired by seed with this machine's runs. Time-boxed by END_BY.
#
#   masks         dropout's masks and fixed arithmetic on this GPU, recorded for comparison with the
#                 second machine's (scripts/audits/dropout_masks.py; seconds)
#   nodropout     the twins' mechanism: BPE 17k's null at ~38M and 10k with dropout off, seeds 0–1,
#                 run on both machines — what the GPUs' rounding alone does
#   fullempty     G-78 at full training: the null switched off from step 0 under endings apart,
#                 151M tokens, 50k, seeds 0–3, beside this machine's null and case there
#   fullemptybpe  the same under BPE 17k, where case ends below its null
#   item7         the thesis's list, item 7: `ucaseprep` under BPE 17k at full training, seeds 0–3
#                 (its codes derived on 17 October)
#   item8         item 8: frequency band and part of speech from the word form at 50k, seeds 2–3
#
# Guards as in tulpar_2026-10-17.sh: ready; sanity (a piece's first run within 0.05 of this
# machine's run of a neighbouring cell); gpu, disk, END_BY. Every run is a RESUMABLE PIECE.
#
#   PLAN=1 PIECES="fullempty item8" END_BY=… bash scripts/chains/tulpar_followup_2026-10-17.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"      # PLAN on the thesis machine
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/tulpar_followup_2026-10-17.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-masks nodropout fullempty item8}
MIN_FREE_GB=${MIN_FREE_GB:-5}
MIN_GPU_GB=${MIN_GPU_GB:-5}       # an 8 GB GPU; a run needs ~3.5 GB
C="-o frozen.trainable=true -o frozen.causal_identity=true"
UKE="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpeend_uk_17k.json"
RUN="env MORPHLM_NO_BUILD=1 nice -n 10 python -m morphlm.cli run --pipeline bpe"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }

# the cells: each prints a run's arguments for a seed (no steps: the default 50k)
U2="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.2"
fempty_e() { echo "--facts mconst $U2 -o data.bpe_path=data/bpeend_uk_17k.json $C -o frozen.feature_off_step=0 -o train.seed=$1"; }
fempty_b() { echo "--facts mconst $U2 -o data.bpe_path=data/bpe_uk_17k.json $C -o frozen.feature_off_step=0 -o train.seed=$1"; }
prep_b()   { echo "--facts ucaseprep $U2 -o data.bpe_path=data/bpe_uk_17k.json $C -o train.seed=$1"; }
mfreq_en() { echo "--facts mfreq -o data.bpe_path=data/bpe_17k.json $C -o train.seed=$1"; }
plex_en()  { echo "--facts poslex -o data.bpe_path=data/bpe_17k.json $C -o train.seed=$1"; }
U05B="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpe_uk_17k.json"
nodrop()   { echo "--facts mconst $U05B $C $(steps 10000) -o model.dropout=0 -o train.seed=$1"; }
# this machine's run of a neighbouring cell at the piece's first seed (result.json)
REF_nodropout=1.089060       # e9f3f153ec20, the same null with dropout, seed 0
REF_fullempty=0.926563       # 57f76ed51760, the endings-apart null at 151M and 50k, seed 0
REF_fullemptybpe=0.941868    # c58ba0662048, the BPE 17k null there, seed 0
REF_item7=0.944875           # bac159440880, case under BPE 17k there, seed 0
REF_item8=1.338828           # 0096ec5ce160, the English null at 50k, seed 2
minutes() {     # on this machine, from the night of 16 October, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        fullempty|fullemptybpe|item7|item8) echo 60 ;;   # 50k at full training: 54–58 min each here
        nodropout) echo 13 ;;                            # 10k at width 128: 11–12 min
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
setpiece() {    # setpiece NAME KIND "SEEDS" REF FN... — every cell at each seed; after the first
                # seed, the first cell's run must end within 0.05 of REF, a run of this machine
    local name=$1 kind=$2 seeds=$3 ref=$4 first s
    local -a args
    shift 4
    first=${seeds%% *}
    local rest=""; [[ "$seeds" == *" "* ]] && rest=${seeds#* }    # a piece of one seed has none left
    stamp "$name, seed $first"
    mapfile -t args < <(cells "$first" "$@")
    runset "$kind" "$first" "${args[@]}"
    if within "$name, seed $first" "$(bpb "${args[0]}")" "$ref" 0.05 "this machine's"; then
        stamp "$name: sanity passed — seeds ${rest:-(none left)}"
        for s in $rest; do
            mapfile -t args < <(cells "$s" "$@")
            runset "$kind" "$s" "${args[@]}"
        done
    else
        stamp "$name: SANITY FAILED — its other seeds are skipped"
    fi
}
MASKS_DONE=scratch/tulpar_followup_2026-10-17.masks_done
run_piece() {
    case "$1" in
        masks)
            stamp "masks: dropout's masks and fixed arithmetic on this GPU"
            if [ -n "${PLAN:-}" ] || [ -f "$MASKS_DONE" ] || [ -n "$STOP" ]; then return 0; fi
            gpu_ok || return 0
            python scripts/audits/dropout_masks.py && mkdir -p scratch && date '+%F %T' > "$MASKS_DONE" ;;
        nodropout)
            setpiece "nodropout: BPE 17k's null at 10k with dropout off, seeds 0–1" nodropout "0 1" "$REF_nodropout" nodrop ;;
        fullempty)
            setpiece "fullempty: G-78 at full training — the null switched off from step 0, endings apart, 151M, 50k" \
                fullempty "0 1 2 3" "$REF_fullempty" fempty_e ;;
        fullemptybpe)
            setpiece "fullemptybpe: the same under BPE 17k" fullemptybpe "0 1 2 3" "$REF_fullemptybpe" fempty_b ;;
        item7)
            setpiece "item7: case with prepositions from their form, BPE 17k, 151M, 50k" item7 "0 1 2 3" "$REF_item7" prep_b ;;
        item8)
            setpiece "item8: frequency band and part of speech from the word form, English, 50k, seeds 2–3" \
                item8 "2 3" "$REF_item8" mfreq_en plex_en ;;
    esac
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }afternoon chain prepared 17 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"
for p in $PIECES; do run_piece "$p"; done

if [ -n "${PLAN:-}" ]; then
    stamp "PLAN: $PLANNED run(s), ~$PLANNED_MIN min of GPU, ending ~$(date -d "@$CLOCK" '+%a %F %H:%M')${END_BY:+; $LEFT run(s) left for a later start}"
    exit 0
fi
stamp "tables"
bash scripts/ops/refresh_exports.sh ;
case "$STOP" in
    disk) stamp "CHAIN STOPPED — too little disk; free space and start it again: it resumes" ;;
    gpu)  stamp "CHAIN STOPPED — the GPU stayed busy until END_BY; start it again: it resumes" ;;
    *) if [ "$LEFT" -gt 0 ]; then stamp "CHAIN ENDED BY $END_BY — $LEFT run(s) left for the next start"
       else stamp "CHAIN COMPLETE"; fi ;;
esac
