#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 2–3 October 2026 (journey §33.36). ~7 h. The night-ready items
# of docs/DIRECTIONS.md, in priority order — a night cut short keeps the first:
#   1  English part of speech at a 10k budget, seeds 2–3, with its null   P70
#   2  the switch at seeds 2–3: off from the start, off at step 2,500      P71
#   3  D1: English part of speech at 5k and 20k budgets, seeds 0–3         P72
#   4  D1: Ukrainian case at a 20k budget, seeds 2–3                       P73
#   then tables and the learning figure.
# Waits while any training run is still going (Block C1), then starts.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/night_2026-10-02.sh >/dev/null 2>&1 &
#   status any time: bash scripts/ops/chain_status.sh
# Rehearsal knobs (defaults are the night): WAIT_RUNS (0 skips the wait),
#   EN_SEEDS, SW_SEEDS, LAW_SEEDS, UK_SEEDS, ENEXTRA, UKEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/night_2026-10-02.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

EN_SEEDS=${EN_SEEDS-2 3}
SW_SEEDS=${SW_SEEDS-2 3}
LAW_SEEDS=${LAW_SEEDS-0 1 2 3}
UK_SEEDS=${UK_SEEDS-2 3}
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
UK="-o data.corpus=ukwiki -o data.corpus_fraction=0.2 -o data.bpe_path=data/bpe_uk_17k.json -o data.spacy_model=uk_core_news_sm ${UKEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ "${WAIT_RUNS:-1}" = 1 ]; then
    stamp "waiting while a training run is still going"
    # a training process's command line STARTS with python -m morphlm.cli run
    # (possibly with python's full path); a shell merely mentioning it never does
    while pgrep -f '^(\S*/)?python[0-9.]* -m morphlm\.cli run' >/dev/null; do sleep 60; done
fi
stamp "night of 2 October starts"

stamp "1  English part of speech at a 10k budget, seeds ${EN_SEEDS:-(none)}"
for s in $EN_SEEDS; do for f in mconst pos; do
    $RUN --facts $f $EN $C $(steps 10000) -o train.seed=$s ;
done; done

stamp "2  the switch, off from the start and at 2,500, seeds ${SW_SEEDS:-(none)}"
for s in $SW_SEEDS; do for k in 0 2500; do
    $RUN --facts pos $EN $C $(steps 10000) -o frozen.feature_off_step=$k -o train.seed=$s ;
done; done

stamp "3  D1: English part of speech at 5k and 20k budgets, seeds ${LAW_SEEDS:-(none)}"
for n in 5000 20000; do for s in $LAW_SEEDS; do for f in mconst pos; do
    $RUN --facts $f $EN $C $(steps $n) -o train.seed=$s ;
done; done; done

stamp "4  D1: Ukrainian case at a 20k budget, seeds ${UK_SEEDS:-(none)}"
for s in $UK_SEEDS; do for f in mconst ucase; do
    $RUN --facts $f $UK $C $(steps 20000) -o train.seed=$s ;
done; done

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
