#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Follow-on chain, 2 October 2026 (journey §33.37). ~4.8 h. Starts itself when
# night_2026-10-02.sh has finished. Night-ready items, in priority order:
#   1  C1 at seeds 2–3: part of speech off at 7,500, on at 5,000         P74
#   2  English inflection at a 10k budget, seeds 0–3                     P75
#   3  English structure at a 10k budget, seeds 0–3 (the law's control)  P77
#   4  Ukrainian case at a 5k budget, seeds 0–3, with its null           P76
#   5  English frequency band at a 10k budget, seeds 0–3                 P78
#   6  English part of speech from the word form, 10k budget, seeds 0–3  P78
#   then tables and the learning figure.
# The English cells are compared with the 10k nulls already run at seeds 0–3.
#
#   nohup systemd-inhibit --what=sleep:idle bash scripts/chains/day_2026-10-02.sh >/dev/null 2>&1 &
# Rehearsal knobs (defaults are the day): WAIT_FOR, C1_SEEDS, SEEDS, ENEXTRA, UKEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/guard.sh
export PYTHONUNBUFFERED=1
exec > >(tee -a logs/day_2026-10-02.log) 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

WAIT_FOR=${WAIT_FOR-'^bash scripts/chains/night_2026-10-02.sh'}
C1_SEEDS=${C1_SEEDS-2 3}
SEEDS=${SEEDS-0 1 2 3}
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
UK="-o data.corpus=ukwiki -o data.corpus_fraction=0.2 -o data.bpe_path=data/bpe_uk_17k.json -o data.spacy_model=uk_core_news_sm ${UKEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
steps() { echo "-o train.max_steps=$1 -o train.lr_horizon=$1"; }
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ -n "$WAIT_FOR" ]; then
    stamp "waiting for $WAIT_FOR to finish"
    while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 60; done
fi
stamp "follow-on of 2 October starts"

stamp "1  C1 at seeds ${C1_SEEDS:-(none)}: off at 7,500, on at 5,000"
for s in $C1_SEEDS; do
    $RUN --facts pos $EN $C $(steps 10000) -o frozen.feature_off_step=7500 -o train.seed=$s ;
    $RUN --facts pos $EN $C $(steps 10000) -o frozen.feature_on_step=5000 -o train.seed=$s ;
done

for f in mfeatlex mstruct; do
    stamp "2–3  English $f at a 10k budget, seeds ${SEEDS:-(none)}"
    for s in $SEEDS; do $RUN --facts $f $EN $C $(steps 10000) -o train.seed=$s ; done
done

stamp "4  Ukrainian case at a 5k budget, seeds ${SEEDS:-(none)}"
for s in $SEEDS; do for f in mconst ucase; do
    $RUN --facts $f $UK $C $(steps 5000) -o train.seed=$s ;
done; done

for f in mfreq poslex; do
    stamp "5–6  English $f at a 10k budget, seeds ${SEEDS:-(none)}"
    for s in $SEEDS; do $RUN --facts $f $EN $C $(steps 10000) -o train.seed=$s ; done
done

stamp "tables and the learning figure"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
python scripts/analysis/learning_law.py --out exports/figures ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
