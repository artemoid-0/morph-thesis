#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Day block E, 5 October 2026 (journey §33.48). ~80 min. Starts itself when the
# screen (screen_2026-10-05.sh) has finished — waiting on the CHAIN, not on
# "no training": between a chain's runs there is a gap of seconds.
#   1  D2  the 128- and 256-wide nulls at a 10k budget, seed 4 (a new seed:
#          re-running a recorded cell would add a duplicate, §33.48), with
#          checkpoints every 500 steps; then the part-of-speech probe   P96
#   2  D1  the 256-wide model with its slot tiled ×2 (16 of 256 dims — the
#          base model's share): null and part of speech, 10k, seed 0       P97
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/day_2026-10-05.sh >/dev/null 2>&1 &
# Rehearsal knobs: WAIT_FOR (empty skips the wait), ENEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/day_2026-10-05.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

WAIT_FOR=${WAIT_FOR-'^bash scripts/chains/screen_2026-10-05.sh'}
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S="-o train.max_steps=10000 -o train.lr_horizon=10000"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ -n "$WAIT_FOR" ]; then
    stamp "waiting for the screen to finish"
    while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 60; done
fi
stamp "day block E starts"

stamp "1  D2: the nulls with checkpoints every 500 steps, then the probe"
for w in 128 256; do
    MORPHLM_CHECKPOINT_EVERY=500 $RUN --facts mconst $EN $C $S -o model.d_model=$w -o train.seed=4 ;
done
for id in $(python scripts/ops/run_ids.py --facts mconst $EN $C $S -o train.seed=4 --widths 128 256 2>/dev/null); do
    python scripts/analysis/probe.py --run $id --fact pos ;
done

stamp "2  D1: 256 wide, slot tiled x2 — null, then part of speech"
for f in mconst pos; do
    $RUN --facts $f $EN $C $S -o model.d_model=256 -o frozen.amplify=2 -o train.seed=0 ;
done

stamp "BLOCK COMPLETE — python scripts/ops/pack_results.py, then send the zip"
