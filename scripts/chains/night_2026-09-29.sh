#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 29–30 September 2026 (journey §33.23). ~13 h on two tracks.
#
#   CPU  U2a  the Ukrainian train cache (~9 h; the pilot built validation at
#             this speed), then U2b: the null `mconst` and the leak-free case
#             `ucaselex` derived with --verify, and the health check as a gate —
#             ucase, mconst, ucaselex present and >= 99.9% constant within
#             words, on train and validation                              P38
#   GPU  E    meanwhile, English seeds 2 and 3 for claims resting on seeds 0-1:
#             the zero slot, the empty slot, position only (the position
#             channel, §33.19) and shuffled structure (Q2)          P29, P43
#   GPU  U3   when both are done: the first Ukrainian family at seed 0 —
#             mconst, ucase, ucaselex, then the baseline — only if the CPU
#             track passed its gate and the smoke test passes      P39, P40
#   then temporal profiles and tables.
#
# Start it so it survives the terminal:
#   nohup bash scripts/chains/night_2026-09-29.sh >/dev/null 2>&1 &
# Logs: logs/night_2026-09-29.log (this track, U3) and
#       logs/night_2026-09-29_cpu.log (U2a, U2b). In the morning:
#   python scripts/ops/pack_results.py
#
# Rehearsal knobs — the defaults ARE the night; set them only to rehearse:
#   F, UKTOK, UKEXTRA (runs, cache), DEXTRA (derivations), CEXTRA (checks),
#   EN_SEEDS (empty skips E), ENEXTRA
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
LOG=logs/night_2026-09-29.log
CPULOG=logs/night_2026-09-29_cpu.log
exec > >(tee -a "$LOG") 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

F=${F:-0.2}
UKTOK=${UKTOK:-data/bpe_uk_17k.json}
EN_SEEDS=${EN_SEEDS-2 3}
UK="-o data.corpus=ukwiki -o data.corpus_fraction=$F -o data.bpe_path=$UKTOK -o data.spacy_model=uk_core_news_sm ${UKEXTRA:-}"
D="--corpus ukwiki --corpus-fraction $F --bpe-path $UKTOK ${DEXTRA:-}"
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

stamp "night of 29 September — CPU track in the background, its log: $CPULOG"

# ── CPU: U2a, U2b and the gate ───────────────────────────────────────────────
(
    stamp "U2a  the Ukrainian train cache"
    python -m morphlm.cli cache --pipeline bpe --facts ucase $UK || exit 1
    stamp "U2b  the null and the leak-free case, verified"
    python scripts/data/derive_lexical_codes.py --fact ucase --out-fact mconst --transform const --keep ucase $D --verify || exit 1
    python scripts/data/derive_lexical_codes.py --fact ucase --out-fact ucaselex $D --verify || exit 1
    stamp "U2b  the gate: the family's facts on both splits"
    for split in train validation; do
        python scripts/data/check_cache.py --split $split --bpe-path $UKTOK --corpus ukwiki --corpus-fraction $F ${CEXTRA:-} \
            --require-facts ucase mconst ucaselex --require-constant 99.9 || exit 1
    done
    stamp "CPU TRACK PASSED"
) > "$CPULOG" 2>&1 &
CPU_PID=$!

# ── GPU: English seeds 2 and 3 while the cache builds ────────────────────────
stamp "E  English seeds ${EN_SEEDS:-(none)}: the position channel, shuffled structure"
for s in $EN_SEEDS; do
    $RUN --facts mzero $EN $C -o train.seed=$s ;
    $RUN --facts mzeronopos $EN $C -o train.seed=$s ;
    $RUN --facts mzeroposonly $EN $C -o train.seed=$s ;
    $RUN --facts mstructshuf $EN $C -o train.seed=$s ;
done

stamp "E done — waiting for the CPU track"
wait "$CPU_PID"
CPU_OK=$?
echo "CPU track exit status: $CPU_OK — its milestones:"
grep -E "^=== |cached\]|alignment rate|lexicon:|GATE|Traceback|Error" "$CPULOG" | tail -n 40

# ── GPU: the first Ukrainian family, gated ───────────────────────────────────
if [ "$CPU_OK" -eq 0 ] && \
   python scripts/ops/smoke_model.py --cells mconst ucase ucaselex --bpe-path "$UKTOK" \
       --set data.corpus=ukwiki data.corpus_fraction=$F ; then
    stamp "U3  the first Ukrainian family, seed 0"
    for f in mconst ucase ucaselex; do
        $RUN --facts $f $UK $C -o train.seed=0 ;
    done
    $RUN $UK -o train.seed=0 ;
else
    stamp "U3 SKIPPED — the CPU track (status $CPU_OK) or the smoke test failed; see $CPULOG and above"
fi

stamp "profiles, tables"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
stamp "NIGHT CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
