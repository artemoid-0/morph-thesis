#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Long chain, 5 October 2026 (journey §33.49). ~7.5 h, 19 runs. The screen's
# promising tests at seeds 1–3, the probe extended, D1's second seed — all
# long-chain-ready (code run on this machine, design fixed). Priority order; a
# chain cut short keeps the first, and stopping between runs loses nothing:
#   1  D4  part of speech as an auxiliary target, real and shuffled, 1–3  P98
#   2  D6  rotary positions: plain baseline and null, seeds 1–3           P99
#   3  D3  weaning to step 5,000, seeds 1–3                              P100
#   4  D2  the 128- and 256-wide nulls at a 20k budget, seed 4, with
#          checkpoints every 1,000 steps; then the part-of-speech probe  P101
#   5  D1  the 256-wide model with its slot tiled ×2, seed 1             P102
#   then tables and the learning figure.
# Waits while any training run is still going, then starts.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/long_2026-10-05.sh >/dev/null 2>&1 &
# Rehearsal knobs: WAIT_RUNS (0 skips the wait), SEEDS, ENEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/long_2026-10-05.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

SEEDS=${SEEDS-1 2 3}
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S20="-o train.max_steps=20000 -o train.lr_horizon=20000"
S10="-o train.max_steps=10000 -o train.lr_horizon=10000"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "long chain of 5 October starts"

stamp "1  D4: the auxiliary target, real and shuffled, seeds ${SEEDS:-(none)}"
for s in $SEEDS; do
    $RUN --facts pos $EN $C $S20 -o frozen.role=target -o train.seed=$s ;
    $RUN --facts pos $EN $C $S20 -o frozen.role=target -o frozen.shuffle_codes=true -o train.seed=$s ;
done

stamp "2  D6: rotary positions, plain baseline and null, seeds ${SEEDS:-(none)}"
for s in $SEEDS; do
    $RUN $EN $C $S20 -o model.position=rope -o train.seed=$s ;
    $RUN --facts mconst $EN $C $S20 -o model.position=rope -o train.seed=$s ;
done

stamp "3  D3: weaning to 5,000, seeds ${SEEDS:-(none)}"
for s in $SEEDS; do $RUN --facts pos $EN $C $S20 -o frozen.feature_dropout_until=5000 -o train.seed=$s ; done

stamp "4  D2: the nulls at a 20k budget with checkpoints, then the probe"
for w in 128 256; do
    MORPHLM_CHECKPOINT_EVERY=1000 $RUN --facts mconst $EN $C $S20 -o model.d_model=$w -o train.seed=4 ;
done
for id in $(python scripts/ops/run_ids.py --facts mconst $EN $C $S20 -o train.seed=4 --widths 128 256 2>/dev/null); do
    python scripts/analysis/probe.py --run $id --fact pos ;
done

stamp "5  D1: 256 wide, slot tiled x2, seed 1 — null, then part of speech"
for f in mconst pos; do $RUN --facts $f $EN $C $S10 -o model.d_model=256 -o frozen.amplify=2 -o train.seed=1 ; done

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
