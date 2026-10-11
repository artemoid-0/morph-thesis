#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 27–28 September 2026 — REBUILT after the null (journey §33.10):
# `mconst`, an 8-dim slot carrying no information, gained as much as structure
# at seed 0. The night asks whether that holds at every seed, and how the
# benefit depends on the slot's width. Safe runs only: every feature below was
# derived and smoke-tested (the widths) or trained (mconst, mfreq) in a day
# block first. ~8.5 h.
#   A  the null at seeds 1, 2, 3 — structure has 0–3, so every seed pairs  P22
#   B  mfreq at seed 1 (the day's unrun test)                                P19
#   C  width dose-response: T=16 ×2, T=32 seed 1 (seed 0 ran by day), T=64 ×2 P23
#   D  temporal profiles, tables
#   bash scripts/chains/night_2026-09-27.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
exec > >(tee -a "logs/night_2026-09-27.log") 2>&1
C="-o data.bpe_path=data/bpe_17k.json -o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

stamp "A  the null at seeds 1, 2, 3"
$RUN --facts mconst $C -o train.seed=1 ;
$RUN --facts mconst $C -o train.seed=2 ;
$RUN --facts mconst $C -o train.seed=3 ;

stamp "B  mfreq at seed 1"
$RUN --facts mfreq $C -o train.seed=1 ;

stamp "C  the null's width: 16, 32, 64"
$RUN --facts mconst16 $C -o train.seed=0 ;
$RUN --facts mconst16 $C -o train.seed=1 ;
$RUN --facts mconst32 $C -o train.seed=1 ;
$RUN --facts mconst64 $C -o train.seed=0 ;
$RUN --facts mconst64 $C -o train.seed=1 ;

stamp "D  profiles, tables"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
stamp "NIGHT CHAIN COMPLETE — send results/, results/audits/, logs/"
