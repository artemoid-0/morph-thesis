#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's long chain, prepared 18 October 2026 (journey §33.112): about 15 hours, run
# without anyone watching. Each comparison is within this machine; a cell the thesis machine already
# holds is repeated here as a twin. Time-boxed by END_BY.
#
#   g78uni g78bpe4k g78bpe8k g78bpe32k   G-78 across tokenizers, ~38M, 10k, seeds 0–3: the null and case
#                 as twins and the empty slot, under Unigram 17k and BPE 4k, 8k, 32k — the exposure claim
#                 (§33.110)                                                        P251–P255
#   g78bpe17k     case under BPE 17k at 10k as a twin (its null and empty slot are here already)
#   g78small g78mid   G-78 under BPE 17k at ~1.5M and ~7.6M tokens, 10k: null and case as twins, the
#                 empty slot — the claims on less data                             P256
#   widthprobes   the thesis's list, item 5's last part: the null at widths 128 and 256, 10k and 20k,
#                 seeds 0–3, with a checkpoint every 1,000 steps, then probed (the word's tag and the
#                 previous word's) as seed 4 was                                   P257–P258
#   item7         item 7: case and its word form as twins, and case with prepositions from their form,
#                 151M tokens, BPE 17k, 50k, seeds 0–3                             P250
#   item8         item 8: the null as a twin, frequency band and part of speech from the word form,
#                 English, 50k, seeds 2–3                                          P259
#   then the tables.
#
# Guards as in second_followup_2026-10-17.sh: ready; twin (a piece's first twin within TWIN_TOL of
# the thesis machine's run); gpu, disk, END_BY. Every run is a RESUMABLE PIECE.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-18.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"      # PLAN on the thesis machine
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/second_2026-10-18.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-g78uni g78bpe4k g78bpe8k g78bpe32k g78bpe17k g78small g78mid widthprobes item7 item8}
MIN_FREE_GB=${MIN_FREE_GB:-5}
MIN_GPU_GB=${MIN_GPU_GB:-6}
TWIN_TOL=${TWIN_TOL:-0.01}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 nice -n 10 python -m morphlm.cli run --pipeline bpe"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }

# the cells: each prints a run's arguments for a seed
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
S10="$(steps 10000)"; S20="$(steps 20000)"
tok() { echo "$U -o data.corpus_fraction=$1 -o data.bpe_path=data/$2.json"; }
null_uni()  { echo "--facts mconst $(tok 0.05 unigram_uk_17k) $C $S10 -o train.seed=$1"; }
case_uni()  { echo "--facts ucase $(tok 0.05 unigram_uk_17k) $C $S10 -o train.seed=$1"; }
empty_uni() { echo "--facts mconst $(tok 0.05 unigram_uk_17k) $C $S10 -o frozen.feature_off_step=0 -o train.seed=$1"; }
null_b4()   { echo "--facts mconst $(tok 0.05 bpe_uk_4k) $C $S10 -o train.seed=$1"; }
case_b4()   { echo "--facts ucase $(tok 0.05 bpe_uk_4k) $C $S10 -o train.seed=$1"; }
empty_b4()  { echo "--facts mconst $(tok 0.05 bpe_uk_4k) $C $S10 -o frozen.feature_off_step=0 -o train.seed=$1"; }
null_b8()   { echo "--facts mconst $(tok 0.05 bpe_uk_8k) $C $S10 -o train.seed=$1"; }
case_b8()   { echo "--facts ucase $(tok 0.05 bpe_uk_8k) $C $S10 -o train.seed=$1"; }
empty_b8()  { echo "--facts mconst $(tok 0.05 bpe_uk_8k) $C $S10 -o frozen.feature_off_step=0 -o train.seed=$1"; }
null_b32()  { echo "--facts mconst $(tok 0.05 bpe_uk_32k) $C $S10 -o train.seed=$1"; }
case_b32()  { echo "--facts ucase $(tok 0.05 bpe_uk_32k) $C $S10 -o train.seed=$1"; }
empty_b32() { echo "--facts mconst $(tok 0.05 bpe_uk_32k) $C $S10 -o frozen.feature_off_step=0 -o train.seed=$1"; }
case_b17()  { echo "--facts ucase $(tok 0.05 bpe_uk_17k) $C $S10 -o train.seed=$1"; }
null_s()    { echo "--facts mconst $(tok 0.002 bpe_uk_17k) $C $S10 -o train.seed=$1"; }
case_s()    { echo "--facts ucase $(tok 0.002 bpe_uk_17k) $C $S10 -o train.seed=$1"; }
empty_s()   { echo "--facts mconst $(tok 0.002 bpe_uk_17k) $C $S10 -o frozen.feature_off_step=0 -o train.seed=$1"; }
null_m()    { echo "--facts mconst $(tok 0.01 bpe_uk_17k) $C $S10 -o train.seed=$1"; }
case_m()    { echo "--facts ucase $(tok 0.01 bpe_uk_17k) $C $S10 -o train.seed=$1"; }
empty_m()   { echo "--facts mconst $(tok 0.01 bpe_uk_17k) $C $S10 -o frozen.feature_off_step=0 -o train.seed=$1"; }
EN="-o data.bpe_path=data/bpe_17k.json"
w128_10()   { echo "--facts mconst $EN $C $S10 -o train.seed=$1"; }
w128_20()   { echo "--facts mconst $EN $C $S20 -o train.seed=$1"; }
w256_10()   { echo "--facts mconst $EN $C $S10 -o model.d_model=256 -o train.seed=$1"; }
w256_20()   { echo "--facts mconst $EN $C $S20 -o model.d_model=256 -o train.seed=$1"; }
case_f()    { echo "--facts ucase $(tok 0.2 bpe_uk_17k) $C -o train.seed=$1"; }
prep_f()    { echo "--facts ucaseprep $(tok 0.2 bpe_uk_17k) $C -o train.seed=$1"; }
lex_f()     { echo "--facts ucaselex $(tok 0.2 bpe_uk_17k) $C -o train.seed=$1"; }
null_en()   { echo "--facts mconst $EN $C -o train.seed=$1"; }
mfreq_en()  { echo "--facts mfreq $EN $C -o train.seed=$1"; }
plex_en()   { echo "--facts poslex $EN $C -o train.seed=$1"; }
PROBE_RUN="env MORPHLM_NO_BUILD=1 MORPHLM_CHECKPOINT_EVERY=1000 nice -n 10 python -m morphlm.cli run --pipeline bpe"
# the thesis machine's run of each piece's first cell at its first seed (result.json)
REF_g78uni=1.088992     # d81e089f0452, Unigram 17k's null at 10k, seed 0
REF_g78bpe4k=1.223948   # 24b3759248eb, BPE 4k's null
REF_g78bpe8k=1.154163   # e1e912f50286, BPE 8k's null
REF_g78bpe32k=1.044922  # 6ef7f8764712, BPE 32k's null
REF_g78bpe17k=1.086235  # 87041e3a147b, case under BPE 17k at 10k
REF_g78small=1.266133   # 476eb4c1baeb, BPE 17k's null at ~1.5M
REF_g78mid=1.103724     # 4606f1e94a11, BPE 17k's null at ~7.6M
REF_widthprobes=1.526245  # 3e7d5e2e10b8, the English null at 10k, seed 0
REF_item7=0.944875      # bac159440880, case under BPE 17k at 151M and 50k, seed 0
REF_item8=1.338828      # 0096ec5ce160, the English null at 50k, seed 2
minutes() {     # on this machine, from 16–17 October, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        g78*) echo 5 ;;                 # 10k at width 128: 4.1–4.3 min
        widthprobes) echo 10 ;;         # per run: 4.3 (128, 10k), 8.5 (128, 20k), ~7 (256, 10k), ~14 (256, 20k)
        item7|item8) echo 24 ;;         # 50k at width 128: 20.9 min
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
        g78uni)    twinpiece "g78uni: G-78 under Unigram 17k at 10k" g78uni "0 1 2 3" "$REF_g78uni" null_uni case_uni empty_uni ;;
        g78bpe4k)  twinpiece "g78bpe4k: G-78 under BPE 4k at 10k" g78bpe4k "0 1 2 3" "$REF_g78bpe4k" null_b4 case_b4 empty_b4 ;;
        g78bpe8k)  twinpiece "g78bpe8k: G-78 under BPE 8k at 10k" g78bpe8k "0 1 2 3" "$REF_g78bpe8k" null_b8 case_b8 empty_b8 ;;
        g78bpe32k) twinpiece "g78bpe32k: G-78 under BPE 32k at 10k" g78bpe32k "0 1 2 3" "$REF_g78bpe32k" null_b32 case_b32 empty_b32 ;;
        g78bpe17k) twinpiece "g78bpe17k: case under BPE 17k at 10k as a twin" g78bpe17k "0 1 2 3" "$REF_g78bpe17k" case_b17 ;;
        g78small)  twinpiece "g78small: G-78 under BPE 17k at ~1.5M, 10k" g78small "0 1 2 3" "$REF_g78small" null_s case_s empty_s ;;
        g78mid)    twinpiece "g78mid: G-78 under BPE 17k at ~7.6M, 10k" g78mid "0 1 2 3" "$REF_g78mid" null_m case_m empty_m ;;
        widthprobes)
            RUN="$PROBE_RUN" twinpiece "widthprobes: item 5 — the English null at widths 128 and 256, 10k and 20k, with checkpoints" \
                widthprobes "0 1 2 3" "$REF_widthprobes" w128_10 w128_20 w256_10 w256_20
            probe_all "--fact pos" w128_10 w128_20 w256_10 w256_20
            probe_all "--fact pos --previous" w128_10 w128_20 w256_10 w256_20 ;;
        item7)     twinpiece "item7: case, its word form and prepositions from their form, 151M, BPE 17k, 50k" item7 "0 1 2 3" "$REF_item7" case_f prep_f lex_f ;;
        item8)     twinpiece "item8: the null, frequency band and part of speech from the word form, English, 50k, seeds 2–3" item8 "2 3" "$REF_item8" null_en mfreq_en plex_en ;;
    esac
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }long chain prepared 18 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"
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
