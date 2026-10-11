#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Screening block, 5 October 2026 (journey §33.47). ~2.7 h, 7 runs at a 20k
# budget, seed 0 unless noted — one sharp test per open direction, the most
# upside first; stopping between runs loses nothing:
#   1  D4  part of speech as an auxiliary TARGET, real and shuffled      P95
#   2  D6  rotary positions: plain baseline and null                      P93
#   3  D3  weaning: part of speech dropped at a rate reaching 1 at 5,000  P94
#   4  G-70  English inflection, seeds 2–3                                P90
# Waits while any training run is still going, then starts.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/screen_2026-10-05.sh >/dev/null 2>&1 &
# Rehearsal knobs: WAIT_RUNS (0 skips the wait), ENEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/screen_2026-10-05.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
S="-o train.max_steps=20000 -o train.lr_horizon=20000"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "screen of 5 October starts"

stamp "1  D4: part of speech as an auxiliary target — real, then shuffled"
$RUN --facts pos $EN $C $S -o frozen.role=target -o train.seed=0 ;
$RUN --facts pos $EN $C $S -o frozen.role=target -o frozen.shuffle_codes=true -o train.seed=0 ;

stamp "2  D6: rotary positions — plain baseline, then null"
$RUN $EN $C $S -o model.position=rope -o train.seed=0 ;
$RUN --facts mconst $EN $C $S -o model.position=rope -o train.seed=0 ;

stamp "3  D3: weaning — part of speech dropped at a rate reaching 1 at step 5,000"
$RUN --facts pos $EN $C $S -o frozen.feature_dropout_until=5000 -o train.seed=0 ;

stamp "4  G-70: English inflection at a 20k budget, seeds 2 3"
for s in 2 3; do $RUN --facts mfeatlex $EN $C $S -o train.seed=$s ; done

stamp "SCREEN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
