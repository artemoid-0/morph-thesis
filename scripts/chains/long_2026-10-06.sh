#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Long chain, 6 October 2026 (journey §33.51). ~9.7 h, 24 runs in 10 pieces.
#
# Every run is a RESUMABLE PIECE: a run whose cell and seed already have a
# finished result in results/ is skipped (scripts/ops/run_ids.py --done), and
# so is a probe already recorded. Stop the chain at any time; start it again
# and it carries on where it stopped — only the run in progress is repeated.
# PIECES="3 5" runs just those pieces. Priority order, first looks first:
#
#   1  D6  flat slot under rotary positions, seed 0                 24 min  P104
#   2  D6  zeros ADDED and flat ADDED, rotary positions, seed 0     48 min  P105
#   3  D3  off from the start and off at 5,000 (20k), seed 4, with
#          checkpoints; then the part-of-speech probe on both       47 min  P107
#   4  D4  real and shuffled targets (20k), seed 4, with
#          checkpoints; then the part-of-speech probe on both       49 min  P108
#   5  D6  zero slot under rotary positions, seeds 1–3              73 min  P106
#   6  D6  flat slot under rotary positions, seeds 1–3              73 min
#   7  D6  zeros ADDED and flat ADDED, rotary positions, seed 1     48 min
#   8  D1  word form, frequency band, structure at 20k, seeds 2–3  136 min  P109
#   9  D6  256 wide: plain baseline at 20k, seed 4 (its null
#          exists — the checkpointed run of §33.49)                 38 min  P91 P92
#  10  D6  zeros ADDED and flat ADDED, rotary positions, seed 2     48 min
#   then tables and the learning figure.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-06.sh >/dev/null 2>&1 &
#   PIECES="2 4" nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-06.sh >/dev/null 2>&1 &
# Knobs: PIECES, WAIT_RUNS (0 skips the wait), ENEXTRA (rehearsal)
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/long_2026-10-06.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

PIECES=${PIECES-1 2 3 4 5 6 7 8 9 10}
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S20="-o train.max_steps=20000 -o train.lr_horizon=20000"
ROPE="-o model.position=rope"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

on() { case " $PIECES " in *" $1 "*) return 0 ;; esac; return 1; }
run() {     # one run — skipped if results/ already holds its cell and seed
    if python scripts/ops/run_ids.py --done "$@"; then return 0; fi
    $RUN "$@" ;
}
probe() {   # the part-of-speech probe on the run these arguments make
    local id
    id=$(python scripts/ops/run_ids.py "$@" | tail -1)
    if ls results/audits/probe_"${id}"_pos_*.json >/dev/null 2>&1; then
        echo "  [skip] probe of $id already recorded"; return 0
    fi
    python scripts/analysis/probe.py --run "$id" --fact pos ;
}

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "long chain of 6 October starts — pieces: $PIECES"

if on 1; then
    stamp "1  D6: flat slot under rotary positions, seed 0"
    run --facts mflat $EN $C $S20 $ROPE -o train.seed=0
fi

if on 2; then
    stamp "2  D6: zeros ADDED, then flat ADDED, under rotary positions, seed 0"
    for f in mzero mflat; do run --facts $f $EN $C $S20 $ROPE -o frozen.inject=add -o train.seed=0; done
fi

if on 3; then
    stamp "3  D3: off from the start and off at 5,000, seed 4, checkpoints; then the probe"
    for k in 0 5000; do
        MORPHLM_CHECKPOINT_EVERY=1000 run --facts pos $EN $C $S20 -o frozen.feature_off_step=$k -o train.seed=4
    done
    for k in 0 5000; do probe --facts pos $EN $C $S20 -o frozen.feature_off_step=$k -o train.seed=4; done
fi

if on 4; then
    stamp "4  D4: real and shuffled targets, seed 4, checkpoints; then the probe"
    MORPHLM_CHECKPOINT_EVERY=1000 run --facts pos $EN $C $S20 -o frozen.role=target -o train.seed=4
    MORPHLM_CHECKPOINT_EVERY=1000 run --facts pos $EN $C $S20 -o frozen.role=target -o frozen.shuffle_codes=true -o train.seed=4
    probe --facts pos $EN $C $S20 -o frozen.role=target -o train.seed=4
    probe --facts pos $EN $C $S20 -o frozen.role=target -o frozen.shuffle_codes=true -o train.seed=4
fi

if on 5; then
    stamp "5  D6: zero slot under rotary positions, seeds 1 2 3"
    for s in 1 2 3; do run --facts mzero $EN $C $S20 $ROPE -o train.seed=$s; done
fi

if on 6; then
    stamp "6  D6: flat slot under rotary positions, seeds 1 2 3"
    for s in 1 2 3; do run --facts mflat $EN $C $S20 $ROPE -o train.seed=$s; done
fi

if on 7; then
    stamp "7  D6: zeros ADDED and flat ADDED under rotary positions, seed 1"
    for f in mzero mflat; do run --facts $f $EN $C $S20 $ROPE -o frozen.inject=add -o train.seed=1; done
fi

if on 8; then
    stamp "8  D1: word form, frequency band and structure at 20k, seeds 2 3"
    for f in poslex mfreq mstruct; do for s in 2 3; do run --facts $f $EN $C $S20 -o train.seed=$s; done; done
fi

if on 9; then
    stamp "9  D6: 256 wide — plain baseline at 20k, seed 4"
    run $EN $C $S20 -o model.d_model=256 -o train.seed=4
fi

if on 10; then
    stamp "10  D6: zeros ADDED and flat ADDED under rotary positions, seed 2"
    for f in mzero mflat; do run --facts $f $EN $C $S20 $ROPE -o frozen.inject=add -o train.seed=2; done
fi

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
