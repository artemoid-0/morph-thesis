#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, prepared 17 October 2026 (journey §33.103) for the second machine (§33.101).
# Two questions the night of 16 October left (§33.102), then what remains of the thesis's list
# for this machine. Time-boxed by END_BY.
#
#   twins     Ukrainian, endings apart, ~38M tokens, 20k, seeds 0–3: the null and case as TWINS of
#             the thesis machine's runs (the same arguments and seeds), and an EMPTY SLOT — the
#             null switched off from step 0, so from the null's own initial weights. The twins
#             measure the machines' difference at 20k; the empty slot answers, on one machine,
#             whether it ends better than the null (§33.102: 0.0036, across machines)  P222, P224–P225
#   entwins   English at 50k, seeds 0–1: both plain baselines as twins (each with the arguments
#             the thesis machine used), the machines' difference at 50k and the lead  P223
#   evals     BLiMP on English seeds 4–7 (item 6) and MultiBLiMP on width 512 at 20k, seeds 2–3
#             (item 1): their checkpoints are here; data sent, not downloaded  P227–P228
#   fiction   item 3: both fiction nulls at fraction 0.4, 50k, seeds 0–3 — once
#             scripts/data/fiction_full.sh has verified the data; tried again after each later
#             piece                                                           P216–P217
#   bpetwins  the twins piece under BPE 17k: the null as a twin, and its empty slot  P222, P226
#   then the tables (seconds).
#
# A twin keeps the thesis machine's run id where the arguments still give it; the September
# English baseline (BPE 17k) gets a new id — its configuration differs only in fields added
# since, at defaults that reproduce it (G-69). Twins are compared, never pooled: on the thesis
# machine they live in results/audits/device_twins_<date>/, not results/.
#
# Guards, each skipping only what depends on it:
#   ready      a piece's caches exist (its ids and the codes of its fact, both splits)
#   twin       after a twins piece's first seed, its first twin ends within TWIN_TOL (0.01) of the
#              thesis machine's run: a broken setup misses by far more, the machines' difference
#              by far less (≤ 0.0005 at 10k, §33.101)
#   sanity     fiction's first pair ends within 0.05 of each other
#   gpu        a run starts only with MIN_GPU_GB free on the GPU; otherwise the chain waits in
#              five-minute steps (someone else's job), and stops at END_BY
#   disk, END_BY  as in every chain
# None branches on an effect's size. Every run is a RESUMABLE PIECE (run_ids.py --done); the
# evaluations leave a mark in scratch/ and are not repeated.
#
#   PLAN=1 END_BY=$(date -d '+11 hours' '+%F %T') bash scripts/chains/night_2026-10-17.sh
#   END_BY=$(date -d '+11 hours' '+%F %T') nohup bash scripts/chains/night_2026-10-17.sh > /dev/null 2>&1 < /dev/null &
# Knobs: PIECES, END_BY, PLAN, MIN_FREE_GB, MIN_GPU_GB, WAIT_RUNS, TWIN_TOL; MINUTES, FREE_GB,
#        GPU_FREE_GB, GPU_WAIT_SECONDS (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"      # PLAN on the thesis machine
fi
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh
    LOG=logs/night_2026-10-17.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-twins entwins evals fiction bpetwins}
MIN_FREE_GB=${MIN_FREE_GB:-5}
MIN_GPU_GB=${MIN_GPU_GB:-6}
TWIN_TOL=${TWIN_TOL:-0.01}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
UKE="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpeend_uk_17k.json"
UKB="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.05 -o data.bpe_path=data/bpe_uk_17k.json"
FIC="-o data.corpus=ukfiction -o data.corpus_holdout=500 -o data.spacy_model=uk_core_news_sm -o data.corpus_fraction=0.4"
RUN="env MORPHLM_NO_BUILD=1 nice -n 10 python -m morphlm.cli run --pipeline bpe"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }

# the cells: each prints a run's arguments for a seed
null_e()  { echo "--facts mconst $UKE $C $(steps 20000) -o train.seed=$1"; }
case_e()  { echo "--facts ucase $UKE $C $(steps 20000) -o train.seed=$1"; }
empty_e() { echo "--facts mconst $UKE $C $(steps 20000) -o frozen.feature_off_step=0 -o train.seed=$1"; }
null_b()  { echo "--facts mconst $UKB $C $(steps 20000) -o train.seed=$1"; }
empty_b() { echo "--facts mconst $UKB $C $(steps 20000) -o frozen.feature_off_step=0 -o train.seed=$1"; }
en_bpe()  { echo "-o data.bpe_path=data/bpe_17k.json -o train.seed=$1"; }         # as run in September
en_end()  { echo "-o data.bpe_path=data/bpeend_17k.json $C -o train.seed=$1"; }   # as run on 12 October
fiction() { echo "--facts mconst $FIC -o data.bpe_path=data/$1.json $C -o train.seed=$2"; }
# the thesis machine's run of each twins piece's first cell at its first seed (result.json)
REF_twins=0.997431      # 8f398c5a967e, the endings-apart null at 20k, seed 0
REF_entwins=1.347706    # 8ff1b23b16d0, the BPE 17k baseline at 50k, seed 0
REF_bpetwins=1.013954   # 59edbb34487d, the BPE 17k null at 20k, seed 0
minutes() {     # on this machine, from the night of 16 October, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        twins|bpetwins) echo 10 ;;   # 20k at width 128: 8.4 min each on 16 October
        entwins)        echo 24 ;;   # 50k at width 128: 20.9 min each
        fiction)        echo 26 ;;   # the same, with fiction's 2.4× larger validation split
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
    stamp "$name, seed $first"
    mapfile -t args < <(cells "$first" "$@")
    runset "$kind" "$first" "${args[@]}"
    if within "$name, seed $first" "$(bpb "${args[0]}")" "$ref" "$TWIN_TOL" "the thesis machine's"; then
        stamp "$name: twin check passed — seeds ${seeds#* }"
        for s in ${seeds#* }; do
            mapfile -t args < <(cells "$s" "$@")
            runset "$kind" "$s" "${args[@]}"
        done
    else
        stamp "$name: TWIN CHECK FAILED — its other seeds are skipped"
    fi
}
piece() {       # piece NAME KIND "SEEDS" A_FN A_ARG B_FN B_ARG — a two-cell piece, first seed checked
    local name=$1 kind=$2 seeds=$3 fa=$4 xa=$5 fb=$6 xb=$7 first s
    first=${seeds%% *}
    stamp "$name, seed $first"
    runset "$kind" "$first" "$($fa "$xa" "$first")" "$($fb "$xb" "$first")"
    if within "$name, seed $first" "$(bpb "$($fa "$xa" "$first")")" "$(bpb "$($fb "$xb" "$first")")" 0.05 ""; then
        stamp "$name: sanity passed — seeds ${seeds#* }"
        for s in ${seeds#* }; do runset "$kind" "$s" "$($fa "$xa" "$s")" "$($fb "$xb" "$s")"; done
    else
        stamp "$name: SANITY FAILED — its other seeds are skipped"
    fi
}
EVALS_DONE=scratch/night_2026-10-17.evals_done
evals() {       # BLiMP (item 6, seeds 4–7) and MultiBLiMP (item 1, width 512 at 20k, seeds 2–3)
    local d=scratch/mb_w512_s23 id f
    if [ -n "${PLAN:-}" ]; then
        [ -f "$EVALS_DONE" ] || planned 15 0 "evals — BLiMP on 8 checkpoints, MultiBLiMP on 4"
        return 0
    fi
    [ -n "$STOP" ] && return 0
    [ -f "$EVALS_DONE" ] && { echo "  [done] the evaluations ran before ($EVALS_DONE)"; return 0; }
    for f in external/blimp/anaphor_gender_agreement.jsonl external/multiblimp/ukr/data.tsv; do
        [ -f "$f" ] || { echo "  [not ready] $f is not here — evaluations skipped"; return 0; }
    done
    disk_ok && gpu_ok || return 0
    nice -n 10 python scripts/eval/blimp_eval.py --all --local external/blimp --runs \
        f8b6493b28dc 0f77800048f2 46dacef2d41c c43d5daf3da4 85e6bf6f8a5a 7cd76965bb65 887b5d008438 0e00fdf8ed0c \
        || { echo "  [failed] BLiMP"; return 0; }
    # MultiBLiMP takes labels; seed 0's record here is a guard reference without a checkpoint, so
    # the four runs of seeds 2–3 are read from a directory of their own
    mkdir -p "$d"
    for id in 7c67de57b1be f518b23725ad 3836680f746a 665fb7c3d0bd; do ln -sfn "$PWD/results/$id" "$d/$id"; done
    gpu_ok || return 0
    nice -n 10 python scripts/eval/multiblimp_eval.py --results-dir "$d" --labels \
        'bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.05_st20k_w512' 'bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.05_st20k_w512' \
        || { echo "  [failed] MultiBLiMP"; return 0; }
    mkdir -p "$(dirname "$EVALS_DONE")" && date '+%F %T' > "$EVALS_DONE"
}
fiction_ready() { [ -n "${PLAN:-}" ] || [ -f external/ubertext/FULL_VERIFIED ]; }
run_piece() {
    case "$1" in
        twins)
            twinpiece "twins: Ukrainian, endings apart, 20k — the null and case as twins, and the empty slot" \
                twins "0 1 2 3" "$REF_twins" null_e case_e empty_e ;;
        entwins)
            twinpiece "entwins: English at 50k — both plain baselines as twins" \
                entwins "0 1" "$REF_entwins" en_bpe en_end ;;
        evals)
            stamp "evals: BLiMP on English seeds 4–7, MultiBLiMP on width 512 at 20k, seeds 2–3"
            evals ;;
        fiction)
            if fiction_ready; then
                piece "fiction: item 3 — both fiction nulls at 0.4, 50k" fiction \
                    "0 1 2 3" fiction bpe_ukf_17k fiction bpeend_ukf_17k
                DONE_PIECE[fiction]=1
            else
                stamp "fiction: its data is not verified yet (scripts/data/fiction_full.sh) — tried again after the next piece"
            fi ;;
        bpetwins)
            twinpiece "bpetwins: Ukrainian, BPE 17k, 20k — the null as a twin, and its empty slot" \
                bpetwins "0 1 2 3" "$REF_bpetwins" null_b empty_b ;;
    esac
}

if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "${PLAN:+PLAN — nothing runs: }night chain prepared 17 October — pieces: $PIECES${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"
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
