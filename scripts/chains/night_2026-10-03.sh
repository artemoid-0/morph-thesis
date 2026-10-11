#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 2–3 October 2026 (journey §33.40–§33.41). ~11 h. Night-ready
# items across four directions, in priority order — a night cut short keeps
# the first, and stopping it between runs loses nothing:
#   1  D3  G-68 replication: part of speech switched on at 5,000, and off from
#          the start, at a 20k budget, seeds 1–3                            P81
#   2  D1  the width pilot's seeds 1–3: 256-wide null and part of speech, 10k P82
#   3  D1  a narrower model: 64-wide null and part of speech, 10k, seeds 0–3  P83
#   4  D6  the architecture effect against the schedule: plain baseline at a
#          10k budget (seeds 1–3; seed 0 is the September run) and at 20k
#          (seeds 0–3) — the nulls exist                                P84 P85
#   5  D3  shedding reliance: part of speech off at 5,000 of a 20k budget,
#          seeds 0–3                                                        P86
#   6  D1  the 256-wide model at a 20k budget, seed 0, with its baseline (D6)
#   then tables and the learning figure.
# Waits while any training run is still going, then starts.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/night_2026-10-03.sh >/dev/null 2>&1 &
# Rehearsal knobs (defaults are the night): WAIT_RUNS (0 skips the wait),
#   G68_SEEDS, W256_SEEDS, W64_SEEDS, B10_SEEDS, B20_SEEDS, OFF5_SEEDS, W20_SEEDS, ENEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/night_2026-10-03.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

G68_SEEDS=${G68_SEEDS-1 2 3}
W256_SEEDS=${W256_SEEDS-1 2 3}
W64_SEEDS=${W64_SEEDS-0 1 2 3}
B10_SEEDS=${B10_SEEDS-1 2 3}
B20_SEEDS=${B20_SEEDS-0 1 2 3}
OFF5_SEEDS=${OFF5_SEEDS-0 1 2 3}
W20_SEEDS=${W20_SEEDS-0}
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    # a training process's command line STARTS with python -m morphlm.cli run
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "night of 2–3 October starts"

stamp "1  G-68: on at 5,000 and off from the start, 20k budget, seeds ${G68_SEEDS:-(none)}"
for s in $G68_SEEDS; do for k in frozen.feature_off_step=0 frozen.feature_on_step=5000; do
    $RUN --facts pos $EN $C $(steps 20000) -o $k -o train.seed=$s ;
done; done

stamp "2  256 wide, 10k budget, seeds ${W256_SEEDS:-(none)}"
for s in $W256_SEEDS; do for f in mconst pos; do
    $RUN --facts $f $EN $C $(steps 10000) -o model.d_model=256 -o train.seed=$s ;
done; done

stamp "3  64 wide, 10k budget, seeds ${W64_SEEDS:-(none)}"
for s in $W64_SEEDS; do for f in mconst pos; do
    $RUN --facts $f $EN $C $(steps 10000) -o model.d_model=64 -o train.seed=$s ;
done; done

stamp "4  D6: plain baseline at 10k (seeds ${B10_SEEDS:-(none)}) and 20k (seeds ${B20_SEEDS:-(none)})"
for s in $B10_SEEDS; do $RUN $EN $C $(steps 10000) -o train.seed=$s ; done
for s in $B20_SEEDS; do $RUN $EN $C $(steps 20000) -o train.seed=$s ; done

stamp "5  D3: part of speech off at 5,000 of a 20k budget, seeds ${OFF5_SEEDS:-(none)}"
for s in $OFF5_SEEDS; do
    $RUN --facts pos $EN $C $(steps 20000) -o frozen.feature_off_step=5000 -o train.seed=$s ;
done

stamp "6  256 wide, 20k budget, seeds ${W20_SEEDS:-(none)}, with its plain baseline"
for s in $W20_SEEDS; do
    for f in mconst pos; do $RUN --facts $f $EN $C $(steps 20000) -o model.d_model=256 -o train.seed=$s ; done
    $RUN $EN $C $(steps 20000) -o model.d_model=256 -o train.seed=$s ;
done

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
