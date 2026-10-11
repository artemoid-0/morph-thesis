#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 3–4 October 2026 (journey §33.43). ~12 h. Night-ready items, in
# priority order — a night cut short keeps the first; stopping between runs
# loses nothing:
#   1  D1/D6  the 256-wide model at a 20k budget, seeds 1–3: null, part of
#             speech, plain baseline — its collapse and its architecture
#             effect at four seeds                                  P87 P88
#   2  D1     English inflection, part of speech from the word form,
#             frequency band and structure at a 20k budget, seeds 0–3 — a
#             third schedule for every feature (the 20k null exists)    P89
#   then tables and the learning figure.
# Waits while any training run is still going, then starts.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/night_2026-10-04.sh >/dev/null 2>&1 &
# Rehearsal knobs (defaults are the night): WAIT_RUNS (0 skips the wait),
#   WIDE_SEEDS, FEAT_SEEDS, ENEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/night_2026-10-04.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

WIDE_SEEDS=${WIDE_SEEDS-1 2 3}
FEAT_SEEDS=${FEAT_SEEDS-0 1 2 3}
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    # a training process's command line STARTS with python -m morphlm.cli run
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "night of 3–4 October starts"

stamp "1  256 wide, 20k budget, seeds ${WIDE_SEEDS:-(none)}: null, part of speech, plain baseline"
for s in $WIDE_SEEDS; do
    for f in mconst pos; do $RUN --facts $f $EN $C $(steps 20000) -o model.d_model=256 -o train.seed=$s ; done
    $RUN $EN $C $(steps 20000) -o model.d_model=256 -o train.seed=$s ;
done

for f in mfeatlex poslex mfreq mstruct; do
    stamp "2  English $f at a 20k budget, seeds ${FEAT_SEEDS:-(none)}"
    for s in $FEAT_SEEDS; do $RUN --facts $f $EN $C $(steps 20000) -o train.seed=$s ; done
done

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
