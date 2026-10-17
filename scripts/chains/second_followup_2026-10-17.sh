#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's follow-up, prepared 17 October 2026 (journey §33.105): after
# night_2026-10-17.sh, the twins' mechanism and the remaining comparisons. Time-boxed by END_BY.
#
#   masks      dropout's masks and fixed arithmetic on this GPU, for comparison with the thesis
#              machine's record (scripts/audits/dropout_masks.py; seconds)
#   nodropout  BPE 17k's null at ~38M and 10k with dropout off, seeds 0–1 — twins with dropout off,
#              run on both machines: what the GPUs' rounding alone does
#   w512twins  width 512 at 20k, seed 1: both nulls as twins of the thesis machine's — item 1's
#              four seeds are two per machine
#   proberemoval  the thesis's list, item 5: part of speech off from the start and at 5,000, 20k,
#              seeds 0–3, rerun with a checkpoint every 1,000 steps and probed as seed 4 was (every
#              layer; the word's tag and the previous word's) — each run a twin of the thesis machine's
#   probetarget   item 5: part of speech as an auxiliary target, real and shuffled, 20k, seeds 0–3,
#              likewise
#   g78at10k   G-78 at 10k (optional): under endings apart the null and case as twins and the empty
#              slot; under BPE 17k the empty slot (the null's twins at 10k: the calibration's, and
#              seed 3)
#
#   PLAN=1 END_BY=… bash scripts/chains/second_followup_2026-10-17.sh
# Knobs as in night_2026-10-17.sh.
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"      # PLAN on the thesis machine
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/second_followup_2026-10-17.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-masks nodropout w512twins proberemoval probetarget}
MIN_FREE_GB=${MIN_FREE_GB:-5}
MIN_GPU_GB=${MIN_GPU_GB:-6}
TWIN_TOL=${TWIN_TOL:-0.01}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
UKE="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpeend_uk_17k.json"
UKB="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpe_uk_17k.json"
RUN="env MORPHLM_NO_BUILD=1 nice -n 10 python -m morphlm.cli run --pipeline bpe"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }

# the cells: each prints a run's arguments for a seed
U05="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05"
nodrop()   { echo "--facts mconst $UKB $C $(steps 10000) -o model.dropout=0 -o train.seed=$1"; }
w512_b()   { echo "--facts mconst $U05 -o data.bpe_path=data/bpe_uk_17k.json $C $(steps 20000) -o model.d_model=512 -o train.seed=$1"; }
w512_e()   { echo "--facts mconst $U05 -o data.bpe_path=data/bpeend_uk_17k.json $C $(steps 20000) -o model.d_model=512 -o train.seed=$1"; }
null_e10() { echo "--facts mconst $UKE $C $(steps 10000) -o train.seed=$1"; }
case_e10() { echo "--facts ucase $UKE $C $(steps 10000) -o train.seed=$1"; }
empty_e10(){ echo "--facts mconst $UKE $C $(steps 10000) -o frozen.feature_off_step=0 -o train.seed=$1"; }
null_b10() { echo "--facts mconst $UKB $C $(steps 10000) -o train.seed=$1"; }
empty_b10(){ echo "--facts mconst $UKB $C $(steps 10000) -o frozen.feature_off_step=0 -o train.seed=$1"; }
E20="-o data.bpe_path=data/bpe_17k.json $C $(steps 20000)"
off0_p()   { echo "--facts pos $E20 -o frozen.feature_off_step=0 -o train.seed=$1"; }
off5k_p()  { echo "--facts pos $E20 -o frozen.feature_off_step=5000 -o train.seed=$1"; }
tgt_p()    { echo "--facts pos $E20 -o frozen.role=target -o frozen.aux_weight=0.1 -o train.seed=$1"; }
stgt_p()   { echo "--facts pos $E20 -o frozen.role=target -o frozen.aux_weight=0.1 -o frozen.shuffle_codes=true -o train.seed=$1"; }
PROBE_RUN="env MORPHLM_NO_BUILD=1 MORPHLM_CHECKPOINT_EVERY=1000 nice -n 10 python -m morphlm.cli run --pipeline bpe"
# the thesis machine's run of each twins piece's first cell at its first seed (result.json)
REF_nodropout=1.089060     # e9f3f153ec20, the same null with dropout, seed 0 (sanity: within 0.05)
REF_w512twins=0.813491     # a6754f725bcc, the BPE 17k null at width 512 and 20k, seed 1
REF_proberemoval=1.425505  # 7ab5b53eb521, part of speech off from the start at 20k, seed 0
REF_probetarget=1.431079   # 6c2b70565e34, part of speech as a target at 20k, seed 0
REF_g78at10k=1.071363     # 6dda8d847d8a, the endings-apart null at 10k, seed 0
minutes() {     # on this machine, from the night of 16 October, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        nodropout|g78at10k) echo 5 ;;   # 10k at width 128: 4.1 min (the calibration)
        proberemoval|probetarget) echo 10 ;;   # 20k at width 128: 8.4 min, and its checkpoints
        w512twins)          echo 36 ;;  # 20k at width 512: ~32 min on 16 October
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
MASKS_DONE=scratch/second_followup_2026-10-17.masks_done
probe_all() {   # probe_all "PROBE ARGS" FN... — probe every finished run of these cells (seeds 0–3), once each
    local pargs=$1 f s id tag
    shift
    [ -n "${PLAN:-}" ] && { echo "  [would probe] $pargs: the runs of $* at seeds 0–3"; return 0; }
    [ -n "$STOP" ] && return 0
    tag=pos; [[ "$pargs" == *--previous* ]] && tag=posprev
    for s in 0 1 2 3; do
        for f in "$@"; do
            id=$(python scripts/ops/run_ids.py $($f "$s")) || continue
            [ -f "results/$id/result.json" ] && [ -d "results/$id/ckpt" ] || continue
            ls results/audits/probe_${id}_${tag}_*.json > /dev/null 2>&1 && continue
            gpu_ok || return 0
            python scripts/analysis/probe.py --run "$id" $pargs || echo "  [failed] probe $id $pargs"
        done
    done
}
run_piece() {
    case "$1" in
        masks)
            stamp "masks: dropout's masks and fixed arithmetic on this GPU"
            if [ -n "${PLAN:-}" ] || [ -f "$MASKS_DONE" ] || [ -n "$STOP" ]; then return 0; fi
            gpu_ok || return 0
            python scripts/audits/dropout_masks.py && mkdir -p scratch && date '+%F %T' > "$MASKS_DONE" ;;
        nodropout)
            # a sanity check (0.05), not a twin check: dropout off changes the model
            TWIN_TOL=0.05 twinpiece "nodropout: BPE 17k's null at 10k with dropout off, seeds 0–1" nodropout "0 1" "$REF_nodropout" nodrop ;;
        w512twins)
            twinpiece "w512twins: width 512 at 20k, seed 1 — both nulls as twins" w512twins "1" "$REF_w512twins" w512_b w512_e ;;
        proberemoval)
            RUN="$PROBE_RUN" twinpiece "proberemoval: item 5 — part of speech off from the start and at 5,000, 20k, with checkpoints" \
                proberemoval "0 1 2 3" "$REF_proberemoval" off0_p off5k_p
            probe_all "--fact pos" off0_p off5k_p
            probe_all "--fact pos --previous" off0_p off5k_p ;;
        probetarget)
            RUN="$PROBE_RUN" twinpiece "probetarget: item 5 — part of speech as a target, real and shuffled, 20k, with checkpoints" \
                probetarget "0 1 2 3" "$REF_probetarget" tgt_p stgt_p
            probe_all "--fact pos" tgt_p stgt_p ;;
        g78at10k)
            # the first cell is the endings-apart null's twin; its thesis-machine run at seed 0
            twinpiece "g78at10k: G-78 at 10k — the null and case as twins, the empty slot under both tokenizers" \
                g78at10k "0 1 2 3" "$REF_g78at10k" null_e10 case_e10 empty_e10 null_b10 empty_b10 ;;
    esac
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }follow-up prepared 17 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"
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
