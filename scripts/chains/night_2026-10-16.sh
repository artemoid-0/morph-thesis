#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, prepared 16 October 2026 (journey §33.101) for a second machine (an RTX 5090
# Laptop GPU, calibrated against the first: §33.101). The thesis's list of 15 October (§33.97,
# §33.98), the items whose comparisons are all new runs, in the list's order as far as data allows;
# a seed's pair always on this one machine. Time-boxed by END_BY.
#
#   w512     item 1: width 512 at 20k, seeds 2–3 — scale_2026-10-14.sh (STEPS=20000 SEEDS="2 3"),
#            its width guard reading seed 0's pair                               P211–P213
#   posoff   item 4a: English part of speech off from the start and off at 5,000, 50k steps,
#            seeds 0–3, a seed's pair at a time (the 20k design of §33.42 at full training) P214–P215
#   fiction  item 3: both fiction nulls at fraction 0.4 (~133M tokens), 50k, holdout 500,
#            seeds 0–3 — only once scripts/data/fiction_full.sh has verified the data;
#            tried again after the next piece if it is not ready yet          P216–P217
#   caseoff  item 4b: Ukrainian case under endings apart, off from the start and off at 5,000,
#            ~38M tokens, 20k, seeds 0–3                                       P218–P219
#   english  item 6: English plain baselines of both tokenizers at 50k, seeds 4–7  P220–P221
#   then the tables (seconds).
#
# Guards, each skipping only what depends on it:
#   ready      a piece's caches exist (its ids and the codes of its fact, both splits)
#   sanity     after a piece's first seed, its pair ends within 0.05 of each other
#   gpu        a run starts only with MIN_GPU_GB free on the GPU; otherwise the chain waits in
#              five-minute steps (someone else's job), and stops at END_BY
#   disk, END_BY  as in every chain
# None branches on an effect's size. Every run is a RESUMABLE PIECE (run_ids.py --done).
#
#   PLAN=1 END_BY=$(date -d '+11 hours' '+%F %T') bash scripts/chains/night_2026-10-16.sh
#   END_BY=$(date -d '+11 hours' '+%F %T') nohup bash scripts/chains/night_2026-10-16.sh > /dev/null 2>&1 < /dev/null &
# Knobs: PIECES, END_BY, PLAN, MIN_FREE_GB, MIN_GPU_GB, WAIT_RUNS; MINUTES, FREE_GB, GPU_FREE_GB, GPU_WAIT_SECONDS (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"      # PLAN on the thesis machine
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/night_2026-10-16.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-w512 posoff fiction caseoff english}
MIN_FREE_GB=${MIN_FREE_GB:-5}
MIN_GPU_GB=${MIN_GPU_GB:-6}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
UKE="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpeend_uk_17k.json"
FIC="-o data.corpus=ukfiction -o data.corpus_holdout=500 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.4"
RUN="env MORPHLM_NO_BUILD=1 nice -n 10 python -m morphlm.cli run --pipeline bpe"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }

# the cells: each prints a run's arguments for a seed
posoff()  { echo "--facts pos -o data.bpe_path=data/bpe_17k.json $C -o frozen.feature_off_step=$1 -o train.seed=$2"; }
caseoff() { echo "--facts ucase $UKE $C $(steps 20000) -o frozen.feature_off_step=$1 -o train.seed=$2"; }
fiction() { echo "--facts mconst $FIC -o data.bpe_path=data/$1.json $C -o train.seed=$2"; }
english() { echo "-o data.bpe_path=data/$1.json $C -o train.seed=$2"; }
minutes() {     # on this machine: the 4-minute 10k run of the calibration, scaled, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        posoff|english) echo 24 ;;   # 50k at width 128: ~5 × 4.1 min
        fiction)        echo 26 ;;   # the same, with fiction's 2.4× larger validation split
        caseoff)        echo 10 ;;   # 20k at width 128
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
pairset() {     # pairset KIND KEY SEED ARGSTRING... — one seed's runs, all or none
    local kind=$1 key=$2 s=$3 a pending=() mins=0
    shift 3
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
near() {
    if [ -n "${PLAN:-}" ]; then return 0; fi
    if [ -z "$2" ] || [ -z "$3" ]; then echo "  [guard] $1: no finished run to compare (${2:-none} against ${3:-none})"; return 1; fi
    if awk -v a="$2" -v b="$3" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= 0.05) }'; then
        echo "  [guard] $1: $2 against $3 — within 0.05, passes"
    else echo "  [guard] $1: $2 against $3 — NOT within 0.05, fails"; return 1; fi
}
piece() {       # piece NAME KIND KEY "SEEDS" A_FN A_ARG B_FN B_ARG — a two-cell piece, first seed checked
    local name=$1 kind=$2 key=$3 seeds=$4 fa=$5 xa=$6 fb=$7 xb=$8 first s
    first=${seeds%% *}
    stamp "$name, seed $first"
    pairset "$kind" "$key" "$first" "$($fa "$xa" "$first")" "$($fb "$xb" "$first")"
    if near "$name, seed $first" "$(bpb "$($fa "$xa" "$first")")" "$(bpb "$($fb "$xb" "$first")")"; then
        stamp "$name: sanity passed — seeds ${seeds#* }"
        for s in ${seeds#* }; do pairset "$kind" "$key" "$s" "$($fa "$xa" "$s")" "$($fb "$xb" "$s")"; done
    else
        stamp "$name: SANITY FAILED — its other seeds are skipped"
    fi
}
fiction_ready() { [ -n "${PLAN:-}" ] || [ -f external/ubertext/FULL_VERIFIED ]; }
run_piece() {
    case "$1" in
        w512)
            stamp "w512: item 1 — width 512 at 20k, seeds 2–3 (scale_2026-10-14.sh)"
            if [ -n "${PLAN:-}" ]; then
                local out m n
                out=$(MINUTES=${MINUTES:-40} STEPS=20000 SEEDS="2 3" PLAN=1 END_BY="${END_BY:-}" \
                    bash scripts/chains/scale_2026-10-14.sh)
                echo "$out" | grep "would run\|left for"
                m=$(echo "$out" | grep -o "PLAN: [0-9]* run(s), ~[0-9]* min" | grep -o "~[0-9]*" | tr -d '~')
                n=$(echo "$out" | grep -o "PLAN: [0-9]* run" | grep -o "[0-9]*")
                [ -n "$m" ] && planned "$m" "${n:-0}" "w512, seeds 2–3 (the plan above)"
                return 0
            fi
            [ -n "$STOP" ] && return 0
            MINUTES=${MINUTES:-40} STEPS=20000 SEEDS="2 3" WAIT_RUNS=0 END_BY="${END_BY:-}" \
                bash scripts/chains/scale_2026-10-14.sh ;;
        posoff)
            piece "posoff: item 4a — English part of speech off at 0 and at 5,000, 50k" posoff posoff \
                "0 1 2 3" posoff 0 posoff 5000 ;;
        caseoff)
            piece "caseoff: item 4b — Ukrainian case off at 0 and at 5,000 under endings apart, 20k" caseoff caseoff \
                "0 1 2 3" caseoff 0 caseoff 5000 ;;
        fiction)
            if fiction_ready; then
                piece "fiction: item 3 — both fiction nulls at 0.4, 50k" fiction fiction \
                    "0 1 2 3" fiction bpe_ukf_17k fiction bpeend_ukf_17k
                DONE_PIECE[fiction]=1
            else
                stamp "fiction: its data is not verified yet (scripts/data/fiction_full.sh) — tried again after the next piece"
            fi ;;
        english)
            piece "english: item 6 — English plain baselines at 50k, seeds 4–7" english english \
                "4 5 6 7" english bpe_17k english bpeend_17k ;;
    esac
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }night chain prepared 16 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"
FIC_WAITING=""
for p in $PIECES; do
    run_piece "$p"
    # fiction, if its data was not verified at its turn, is tried after each later piece, and last
    [ "$p" = fiction ] && [ -z "${DONE_PIECE[fiction]:-}" ] && FIC_WAITING=1
    if [ -n "$FIC_WAITING" ] && [ "$p" != fiction ] && fiction_ready; then run_piece fiction; FIC_WAITING=""; fi
done
if [ -n "$FIC_WAITING" ] && fiction_ready; then run_piece fiction; fi

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
