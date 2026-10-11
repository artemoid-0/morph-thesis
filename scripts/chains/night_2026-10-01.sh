#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Follow-on chain, night of 30 September – 1 October 2026 (journey §33.29).
# Starts itself when night_2026-09-30.sh has finished. ~5 h after that.
#
#   CPU  L0  little-data Ukrainian caches: corpus fractions 0.002 and 0.01
#            (~1.5M and ~7.6M train tokens; validation is the same split as
#            always), the null's codes, and the health gate on both splits
#   GPU  E   meanwhile, English inflection bundles, seed 2              P59
#   GPU  L   gated: the little-data family — null and contextual case, both
#            sizes, seeds 0-3, 10k steps with the cosine over 10k   P57, P58
#   GPU  E   English inflection bundles, seed 3                         P59
#   then profiles and tables.
#
#   nohup bash scripts/chains/night_2026-10-01.sh >/dev/null 2>&1 &
# Rehearsal knobs (defaults are the night): WAIT_FOR, SIZES, L_SEEDS,
#   EN_SEEDS_A, EN_SEEDS_B, UKTOK, UKEXTRA, DEXTRA, CEXTRA, ENEXTRA, LSTEPS_N
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
export PYTHONUNBUFFERED=1            # logs stream live (the 30 Sep log came in bursts)
mkdir -p logs
LOG=logs/night_2026-10-01.log
CPULOG=logs/night_2026-10-01_cpu.log
exec > >(tee -a "$LOG") 2>&1
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

# the chain it follows, matched from the START of its command line — a bare
# substring would also match an editor or pager showing the file, forever
WAIT_FOR=${WAIT_FOR-'^bash scripts/chains/night_2026-09-30.sh'}
SIZES=${SIZES-0.002 0.01}
L_SEEDS=${L_SEEDS-0 1 2 3}
EN_SEEDS_A=${EN_SEEDS_A-2}
EN_SEEDS_B=${EN_SEEDS_B-3}
UKTOK=${UKTOK:-data/bpe_uk_17k.json}
N=${LSTEPS_N:-10000}
UKBASE="-o data.corpus=ukwiki -o data.bpe_path=$UKTOK -o data.spacy_model=uk_core_news_sm ${UKEXTRA:-}"
LSTEPS="-o train.max_steps=$N -o train.lr_horizon=$N"
EN="-o data.bpe_path=data/bpe_17k.json ${ENEXTRA:-}"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"

if [ -n "$WAIT_FOR" ]; then
    stamp "waiting for $WAIT_FOR to finish"
    while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 60; done
fi
stamp "follow-on chain starts — CPU track in the background, its log: $CPULOG"

# ── CPU: the little-data caches, the null's codes, the gate ─────────────────
(
    for F in $SIZES; do
        D="--corpus ukwiki --corpus-fraction $F --bpe-path $UKTOK ${DEXTRA:-}"
        stamp "L0  little-data cache, corpus fraction $F"
        python -m morphlm.cli cache --pipeline bpe --facts ucase $UKBASE -o data.corpus_fraction=$F || exit 1
        python scripts/data/derive_lexical_codes.py --fact ucase --out-fact mconst --transform const --keep ucase $D --verify || exit 1
        for split in train validation; do
            python scripts/data/check_cache.py --split $split --bpe-path $UKTOK --corpus ukwiki --corpus-fraction $F ${CEXTRA:-} \
                --require-facts ucase mconst --require-constant 99.9 || exit 1
        done
    done
    stamp "CPU TRACK PASSED"
) > "$CPULOG" 2>&1 &
CPU_PID=$!

# ── GPU ───────────────────────────────────────────────────────────────────────
stamp "E  English inflection bundles, seed(s) ${EN_SEEDS_A:-(none)}"
for s in $EN_SEEDS_A; do $RUN --facts mfeatlex $EN $C -o train.seed=$s ; done

stamp "E done — waiting for the CPU track"
wait "$CPU_PID"
CPU_OK=$?
echo "CPU track exit status: $CPU_OK — its milestones:"
grep -E "^=== |cached\]|alignment rate|GATE|Traceback|Error" "$CPULOG" | tail -n 30

if [ "$CPU_OK" -eq 0 ] && \
   python scripts/ops/smoke_model.py --cells mconst ucase --bpe-path "$UKTOK" --set data.corpus=ukwiki data.corpus_fraction=0.01 ; then
    for F in $SIZES; do
        stamp "L  little data, corpus fraction $F: null and contextual case, $N steps"
        for s in $L_SEEDS; do
            for f in mconst ucase; do
                $RUN --facts $f $UKBASE -o data.corpus_fraction=$F $C $LSTEPS -o train.seed=$s ;
            done
        done
    done
else
    stamp "L SKIPPED — the CPU track (status $CPU_OK) or the smoke test failed; see $CPULOG"
fi

stamp "E  English inflection bundles, seed(s) ${EN_SEEDS_B:-(none)}"
for s in $EN_SEEDS_B; do $RUN --facts mfeatlex $EN $C -o train.seed=$s ; done

stamp "profiles, tables"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;
stamp "CHAIN COMPLETE — python scripts/ops/pack_results.py, then send the zip"
