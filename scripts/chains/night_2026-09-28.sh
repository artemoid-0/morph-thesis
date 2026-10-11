#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 28–29 September 2026 (journey §33.16). ~9.5 h.
# Every comparison tonight is INITIALIZATION-MATCHED: cells of identical shape
# start from identical weights at the same seed (§33.15).
#   A  the additive family at seeds 2 and 3: the null's two vectors, one flat
#      vector, structure — each against the matched control (mzero ADDED),
#      which already ran at those seeds                                 P36
#   B  why the zero slot works: without position / with position only, seeds
#      0 and 1, matched with mzero (slotted). New model code, so derived and
#      smoke-tested here first; if that fails, only B is skipped       P27, P28
#   C  temporal profiles, tables
#   bash scripts/chains/night_2026-09-28.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
exec > >(tee -a "logs/night_2026-09-28.log") 2>&1
C="-o data.bpe_path=data/bpe_17k.json -o frozen.trainable=true -o frozen.causal_identity=true"
ADD="-o frozen.inject=add"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

stamp "A  the additive family, seeds 2 and 3 (matched with the control)"
for s in 2 3; do
    $RUN --facts mconst $C $ADD -o train.seed=$s ;
    $RUN --facts mflat $C $ADD -o train.seed=$s ;
    $RUN --facts mstruct $C $ADD -o train.seed=$s ;
done

stamp "B  the zero slot's position: derive, smoke, then seeds 0 and 1"
if python scripts/data/derive_lexical_codes.py --fact mstruct --out-fact mzeronopos --transform const --keep mstruct --verify && \
   python scripts/data/derive_lexical_codes.py --fact mstruct --out-fact mzeroposonly --transform const --keep mstruct --verify && \
   python scripts/ops/smoke_model.py --cells mzeronopos mzeroposonly ; then
    $RUN --facts mzeronopos $C -o train.seed=0 ;
    $RUN --facts mzeroposonly $C -o train.seed=0 ;
    $RUN --facts mzeronopos $C -o train.seed=1 ;
    $RUN --facts mzeroposonly $C -o train.seed=1 ;
else
    stamp "B SKIPPED — derivation or smoke test failed; see above"
fi

stamp "C  profiles, tables"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
stamp "NIGHT CHAIN COMPLETE — send results/, results/audits/, logs/"
