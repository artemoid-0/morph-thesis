#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 30 September – 1 October 2026 (journey §33.24). ~13 h.
# Everything here has run on this machine, so there are no gates.
#
#   U  the Ukrainian family at seeds 1, 2 and 3 — null (mconst), contextual
#      case, lexical case, baseline — one whole seed at a time, so a night cut
#      short still leaves complete matched families (~55 min per run) P44-P47
#   E  last, and the part to drop if the machine is needed in the morning:
#      English part of speech at seeds 2 and 3, the rest of P29       P29
#   then temporal profiles and tables.
#
#   nohup bash scripts/chains/night_2026-09-30.sh >/dev/null 2>&1 &
#   in the morning: python scripts/ops/pack_results.py
# Rehearsal knobs (defaults are the night): UK_SEEDS, EN_SEEDS, UKEXTRA, ENEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
exec > >(tee -a "logs/night_2026-09-30.log") 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

UK_SEEDS=${UK_SEEDS-1 2 3}
EN_SEEDS=${EN_SEEDS-2 3}
UK="-o data.corpus=ukwiki -o data.corpus_fraction=0.2 -o data.bpe_path=data/bpe_uk_17k.json -o data.spacy_model=uk_core_news_sm ${UKEXTRA:-}"
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

for s in $UK_SEEDS; do
    stamp "U  the Ukrainian family, seed $s"
    for f in mconst ucase ucaselex; do
        $RUN --facts $f $UK $C -o train.seed=$s ;
    done
    $RUN $UK -o train.seed=$s ;
done

stamp "E  English part of speech, seeds ${EN_SEEDS:-(none)}"
for s in $EN_SEEDS; do
    $RUN --facts pos $EN $C -o train.seed=$s ;
done

stamp "profiles, tables"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
stamp "NIGHT CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
