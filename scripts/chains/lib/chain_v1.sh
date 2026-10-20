# chain_v1.sh — the body every chain repeated, kept once: version 1.
# ─────────────────────────────────────────────────────────────────────────────
# FROZEN ONCE A DATED CHAIN SOURCES IT. A dated chain is the record of what ran, so the code it
# ran must not change under it: a change goes into chain_v2.sh, which later chains source, and
# this file stays as it is. The chains written before it (scripts/chains/*.sh up to
# second_2026-10-19.sh) keep their own copies and stay untouched.
#
# Taken from the eight chains of 16–19 October, which repeated it word for word, and
# scale_2026-10-14.sh before them: the preamble, the helpers, ONE seed-piece runner in place of
# their six (piece, setpiece and twinpiece, each in two versions; a piece of one seed now runs
# once), the wait for a training run, and the PLAN and STOP ending.
#
# A chain that uses it (rehearsed with the stub python of scripts/rehearsal/, as every chain is):
#
#     cd "$(dirname "$0")/../.." || exit 1
#     source scripts/chains/lib/chain_v1.sh        # outside PLAN: the guard, and the log in logs/
#     PIECES=${PIECES-twins fiction}
#     null_b() { echo "--facts mconst -o data.bpe_path=data/bpe_uk_17k.json ... -o train.seed=$1"; }
#     case_b() { echo "--facts ucase -o data.bpe_path=data/bpe_uk_17k.json ... -o train.seed=$1"; }
#     fic()    { echo "--facts mconst ... -o data.bpe_path=data/$1.json -o train.seed=$2"; }
#     minutes() {                                  # a run of each KIND here, with a margin
#         [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
#         case "$1" in twins) echo 5 ;; fiction) echo 26 ;; esac
#     }
#     run_piece() {
#         case "$1" in
#             twins)   seedpiece "twins: ..." twins "0 1 2 3" twin:1.086235 null_b case_b ;;
#             fiction) seedpiece "fiction: ..." fiction "0 1 2 3" pair "fic bpe_ukf_17k" "fic bpeend_ukf_17k" ;;
#         esac
#     }
#     chain_main "night chain prepared 20 October"
#
# A CELL prints a run's arguments for a seed, which it takes last and which ends them
# (`-o train.seed=N`); a cell given with words before it ("fic bpe_ukf_17k") is called with them
# first. Every run is a RESUMABLE PIECE: a seed's runs already finished (run_ids.py --done) are not
# run again, so a chain started again does only what is left.
#
# Knobs, from the environment: PIECES, END_BY, PLAN (1: say what would run, run nothing),
# MIN_FREE_GB (5), MIN_GPU_GB (6), TWIN_TOL (0.01), SANITY_TOL (0.05), TWIN_WHAT (the words that
# name a twin's original in the log: "the other machine's"), WAIT_RUNS (1: first wait while a
# training run goes), TABLES (0; 1: refresh exports/ at the end). A chain gives a knob another
# default after the source line, e.g. MIN_GPU_GB=${MIN_GPU_GB:-5}; RUN, the command a run starts
# with, it sets outright. Rehearsal hooks: MINUTES, FREE_GB, GPU_FREE_GB, GPU_WAIT_SECONDS
# (tests/test_chainlib.py).
# ─────────────────────────────────────────────────────────────────────────────

[ -f scripts/chains/lib/chain_v1.sh ] || { echo "chain_v1.sh: source it from the project root, after the chain's cd"; exit 1; }

# ── the preamble ─────────────────────────────────────────────────────────────
# the project's python, whatever shell started the chain (guard.sh does the same outside PLAN)
if ! command -v python >/dev/null 2>&1 && [ -x "$HOME/miniconda3/envs/morph/bin/python" ]; then
    export PATH="$HOME/miniconda3/envs/morph/bin:$PATH"
fi
CHAIN_NAME=$(basename "$0" .sh)
if [ -z "${PLAN:-}" ]; then
    source scripts/chains/guard.sh                 # one copy of a chain at a time
    LOG=logs/$CHAIN_NAME.log
    exec > >(tee -a "$LOG") 2>&1
fi
export PYTHONUNBUFFERED=1
DEADLINE=""
if [ -n "${END_BY:-}" ]; then
    DEADLINE=$(date -d "$END_BY" +%s) || { echo "END_BY not understood: $END_BY — not starting"; exit 1; }
fi
CLOCK=$(date +%s)
LEFT=0; PLANNED=0; PLANNED_MIN=0; STOP=""; SET_STATE=""
declare -A STATE DONE_PIECE
RUN="env MORPHLM_NO_BUILD=1 nice -n 10 python -m morphlm.cli run --pipeline bpe"

# ── the helpers ──────────────────────────────────────────────────────────────
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
free_gb() { [ -n "${FREE_GB:-}" ] && { echo "$FREE_GB"; return; }; df -BG --output=avail . | tail -n 1 | tr -dc '0-9'; }
gpu_free_gb() {
    [ -n "${GPU_FREE_GB:-}" ] && { echo "$GPU_FREE_GB"; return; }
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{print int($1/1024)}'
}
now() { if [ -n "${PLAN:-}" ]; then echo "$CLOCK"; else date +%s; fi; }
fits() {        # fits MINUTES WHAT — END_BY lets a set of MINUTES start now
    if [ -n "$DEADLINE" ] && [ $(( $(now) + $1 * 60 )) -gt "$DEADLINE" ]; then
        echo "  [left for the next chain] $2 — would end after $END_BY"; return 1
    fi
}
disk_ok() {
    [ -n "${PLAN:-}" ] && return 0
    if [ "$(free_gb)" -lt "${MIN_FREE_GB:-5}" ]; then
        stamp "DISK: $(free_gb) GB free, below ${MIN_FREE_GB:-5} GB — no further run starts"; STOP=disk; return 1
    fi
}
gpu_ok() {      # wait while someone else holds the GPU; give up at END_BY
    [ -n "${PLAN:-}" ] && return 0
    local f
    while f=$(gpu_free_gb); [ "${f:-0}" -lt "${MIN_GPU_GB:-6}" ]; do
        if [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
            stamp "GPU: ${f:-?} GB free, below ${MIN_GPU_GB:-6} GB, until END_BY — no further run starts"; STOP=gpu; return 1
        fi
        echo "  [gpu] ${f:-?} GB free, below ${MIN_GPU_GB:-6} GB — waiting five minutes ($(date '+%T'))"
        sleep "${GPU_WAIT_SECONDS:-300}"
    done
}
planned() {     # planned MINUTES N WHAT — PLAN's account of a set that would run
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
minutes() {     # minutes KIND — every chain defines its own; this one answers only the rehearsal hook
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    return 1
}
runset() {      # runset KIND SEED ARGSTRING... — one seed's runs, all or none. SET_STATE says what became
                # of them: done, ran, planned, left (END_BY), skipped (not ready, or no minutes for KIND),
                # stopped (disk, GPU)
    local kind=$1 s=$2 a per mins pending=()
    shift 2
    SET_STATE=stopped
    [ -n "$STOP" ] && return 0
    for a in "$@"; do python scripts/ops/run_ids.py --done $a || pending+=("$a"); done
    SET_STATE=done
    [ ${#pending[@]} -eq 0 ] && return 0
    SET_STATE=left
    if [ -n "${LEFT_ONLY:-}" ]; then LEFT=$((LEFT + ${#pending[@]})); return 0; fi
    SET_STATE=skipped
    for a in "${pending[@]}"; do ready "$a" || { echo "  [skipped] $kind, seed $s"; return 0; }; done
    per=$(minutes "$kind")
    [[ "$per" =~ ^[0-9]+$ ]] || { echo "  [skipped] $kind, seed $s — minutes() gives no time for $kind"; return 0; }
    mins=$(( ${#pending[@]} * per ))
    SET_STATE=left
    fits "$mins" "$kind, seed $s" || { LEFT=$((LEFT + ${#pending[@]})); return 0; }
    SET_STATE=planned
    if [ -n "${PLAN:-}" ]; then planned "$mins" ${#pending[@]} "$kind, seed $s"; return 0; fi
    SET_STATE=ran
    for a in "${pending[@]}"; do disk_ok && gpu_ok || { SET_STATE=stopped; return 0; }; $RUN $a; done
}
bpb() { python scripts/ops/run_ids.py --bpb $1 2>/dev/null; }
within() {      # within LABEL GOT WANT TOL WHAT — GOT ends within TOL of WANT, the run WHAT names
    if [ -n "${PLAN:-}" ]; then return 0; fi
    if [ -z "$2" ] || [ -z "$3" ]; then echo "  [guard] $1: no finished run to compare (${2:-none} against ${3:-none})"; return 1; fi
    if awk -v a="$2" -v b="$3" -v t="$4" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d <= t) }'; then
        echo "  [guard] $1: $2 against ${5:+$5 }$3 — within $4, passes"
    else echo "  [guard] $1: $2 against ${5:+$5 }$3 — NOT within $4, fails"; return 1; fi
}
cells() {       # cells SEED CELL... — each cell's run arguments for a seed, one per line
    local s=$1 c
    shift
    for c in "$@"; do $c "$s"; done      # unquoted: a cell may carry words before the seed
}

# ── the seed-piece runner ────────────────────────────────────────────────────
seedpiece() {   # seedpiece NAME KIND "SEEDS" CHECK CELL... — every cell at each seed, a seed's runs all
                # or none; the first seed first, and CHECK on its runs decides whether the others run:
                #   twin:REF  the first cell's run ends within TWIN_TOL of REF, the same run on the other machine
                #   near:REF  the first cell's run ends within SANITY_TOL of REF, a run of this machine
                #   pair      the first two cells' runs end within SANITY_TOL of each other
                #   none      no check
    local name=$1 kind=$2 seeds=$3 check=$4 first rest="" s word ok=1
    local -a args
    shift 4
    first=${seeds%% *}
    [[ "$seeds" == *" "* ]] && rest=${seeds#* }       # a piece of one seed has none left
    stamp "$name, seed $first"
    mapfile -t args < <(cells "$first" "$@")
    runset "$kind" "$first" "${args[@]}"
    case "$SET_STATE" in
        stopped) return 0 ;;
        skipped) stamp "$name: seed $first is skipped (above) — its other seeds too"; return 0 ;;
        left)    if [ -n "$rest" ]; then           # they wait for the first: counted, not run
                     for s in $rest; do mapfile -t args < <(cells "$s" "$@"); LEFT_ONLY=1 runset "$kind" "$s" "${args[@]}"; done
                 fi
                 stamp "$name: seed $first is left for the next start${rest:+, and seeds $rest with it}"; return 0 ;;
    esac
    local other="the other machine's"
    case "$check" in
        twin:*) word="twin check"
                within "$name, seed $first" "$(bpb "${args[0]}")" "${check#twin:}" "${TWIN_TOL:-0.01}" \
                    "${TWIN_WHAT:-$other}" || ok=0 ;;
        near:*) word="sanity"
                within "$name, seed $first" "$(bpb "${args[0]}")" "${check#near:}" "${SANITY_TOL:-0.05}" "this machine's" || ok=0 ;;
        pair)   word="sanity"
                [ ${#args[@]} -ge 2 ] || { stamp "$name: a pair check needs two cells — its other seeds are skipped"; return 0; }
                within "$name, seed $first" "$(bpb "${args[0]}")" "$(bpb "${args[1]}")" "${SANITY_TOL:-0.05}" "" || ok=0 ;;
        none)   word="" ;;
        *)      stamp "$name: unknown check '$check' — its other seeds are skipped"; return 0 ;;
    esac
    if [ "$ok" = 1 ]; then
        [ -n "$word" ] && stamp "$name: $word passed — seeds ${rest:-(none left)}"
        for s in $rest; do
            mapfile -t args < <(cells "$s" "$@")
            runset "$kind" "$s" "${args[@]}"
        done
    else
        stamp "$name: ${word^^} FAILED — its other seeds are skipped"
    fi
}

# ── the run of a chain ───────────────────────────────────────────────────────
chain_wait_runs() {     # outside PLAN, unless WAIT_RUNS=0: wait while a training run still goes
    if [ -z "${PLAN:-}" ] && [ "${WAIT_RUNS:-1}" = 1 ]; then
        stamp "waiting while a training run is still going"
        while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
    fi
}
chain_end() {   # PLAN's account; otherwise the tables (TABLES=1) and how the chain ended
    if [ -n "${PLAN:-}" ]; then
        stamp "PLAN: $PLANNED run(s), ~$PLANNED_MIN min of GPU, ending ~$(date -d "@$CLOCK" '+%a %F %H:%M')${END_BY:+; $LEFT run(s) left for a later start}"
        exit 0
    fi
    if [ "${TABLES:-0}" = 1 ]; then
        stamp "tables"
        bash scripts/ops/refresh_exports.sh || echo "  [tables] exports/ NOT refreshed — see above"
    fi
    case "$STOP" in
        disk) stamp "CHAIN STOPPED — too little disk; free space and start it again: it resumes" ;;
        gpu)  stamp "CHAIN STOPPED — the GPU stayed busy until END_BY; start it again: it resumes" ;;
        *) if [ "$LEFT" -gt 0 ]; then stamp "CHAIN ENDED BY $END_BY — $LEFT run(s) left for the next start"
           else stamp "CHAIN COMPLETE"; fi ;;
    esac
}
chain_main() {  # chain_main "WHAT" — wait for a training run to end, run each piece of PIECES, end
    declare -F run_piece >/dev/null || { echo "chain_v1.sh: the chain defines no run_piece"; exit 1; }
    chain_wait_runs
    stamp "${PLAN:+PLAN — nothing runs: }$1 — pieces: ${PIECES:-}${END_BY:+; sets must end by $END_BY}; $(free_gb) GB free"
    local p
    for p in ${PIECES:-}; do run_piece "$p"; done
    chain_end
}
