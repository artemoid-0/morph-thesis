#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Night chain, 26–27 September 2026.  Plan and predictions: docs/record/journey/README.md
# §33 (recorded BEFORE this ran).  Expected wall time ~10.5 h, +2 h with the tail.
#
#   bash scripts/chains/night_2026-09-26.sh            # core, ~10.5 h
#   TAIL=1 bash scripts/chains/night_2026-09-26.sh     # + headline seed 4, ~12.5 h
#
# Run it inside tmux, and make sure the laptop will not suspend.
#
# TWO LANES AT ONCE
#   CPU lane (background): pre-flight gate, then the cache rebuild (~5 h) that
#     adds the inflectional facts minfl / magr / mfeat.
#   GPU lane (foreground): everything that needs only the EXISTING caches.
#   `wait` joins them; phase B needs the rebuilt cache.
# The rebuild is safe beside training: it never rewrites the ids file a running
# job has memory-mapped, and it writes new files atomically.
#
# FAILURE HANDLING — the night never stops, it re-routes:
#   * steps are joined with `;`, so one failed step never ends the chain;
#   * the rebuild starts only if the pre-flight passes (&&);
#   * rebuild failed      -> fallback lane (more seeds on existing caches);
#   * derivation refused  -> the contextual variant (mfeat) instead of mfeatlex;
#   * MORPHLM_NO_BUILD=1 on every training run: a missing cache fails at once
#     instead of silently starting a second multi-hour rebuild.
# ─────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs
# everything this chain prints is also kept in a log
exec > >(tee -a "logs/night_2026-09-26.log") 2>&1
C="-o data.bpe_path=data/bpe_17k.json -o frozen.trainable=true -o frozen.causal_identity=true"
BPE="-o data.bpe_path=data/bpe_17k.json"
RUN="env MORPHLM_NO_BUILD=1 python -m morphlm.cli run --pipeline bpe"
BLIMP="env MORPHLM_NO_BUILD=1 python scripts/eval/blimp_eval.py --all --skip-missing --reference bpe_baseline_bpe_17k"
stamp() { echo; echo "=== $(date '+%F %T')  $*"; }

# ═════ PHASE A ═══════════════════════════════════════════════════════════════
stamp "CPU lane started: pre-flight, then rebuild  (log: logs/rebuild.log)"
( python scripts/data/preflight_rebuild.py && \
  python -m morphlm.cli cache --pipeline bpe --facts mfeat $BPE ) > logs/rebuild.log 2>&1 &
REBUILD=$!

stamp "A1 tagger causality check (CPU, minutes)"
python scripts/audits/morph_causality_check.py --sentences 3000 ;

stamp "A2 BLiMP, all 67 paradigms, every existing checkpoint of the key cells"
$BLIMP --labels \
    bpe_baseline_bpe_17k \
    bpe_mstruct-LEARN-CAUSAL_bpe_17k \
    bpe_mstruct-LEARN-CAUSAL-Smorphology_bpe_17k \
    bpe_mstruct-LEARN-CAUSAL-Snames_bpe_17k \
    bpe_mlemma+mstruct-LEARN-CAUSAL_bpe_17k \
    bpe_mlemma+mstruct-LEARN_bpe_17k ;

stamp "A3 pos, causal — the coarse-class control, seeds 0 and 1"
$RUN --facts pos $C -o train.seed=0 ;
$RUN --facts pos $C -o train.seed=1 ;

stamp "A4 headline pair, seed 3"
$RUN $BPE -o train.seed=3 ;
$RUN --facts mstruct $C -o train.seed=3 ;

stamp "waiting for the rebuild"
wait $REBUILD
REBUILD_OK=$?
stamp "rebuild finished with exit code $REBUILD_OK  (0 = success)"

# ═════ PHASE B ═══════════════════════════════════════════════════════════════
if [ "$REBUILD_OK" -eq 0 ]; then
    stamp "B1 derive the strictly causal lexical variant, with verification"
    if python scripts/data/derive_lexical_codes.py --verify ; then
        stamp "B2 mfeatlex (strictly causal inflection), seeds 0 and 1"
        $RUN --facts mfeatlex $C -o train.seed=0 ;
        $RUN --facts mfeatlex $C -o train.seed=1 ;
        stamp "B3 mstruct + mfeatlex — does inflection add to structure? seeds 0 and 1"
        $RUN --facts mstruct,mfeatlex $C -o train.seed=0 ;
        $RUN --facts mstruct,mfeatlex $C -o train.seed=1 ;
    else
        stamp "B1 REFUSED — running the contextual variant mfeat instead (tagger-leak caveat applies)"
        $RUN --facts mfeat $C -o train.seed=0 ;
        $RUN --facts mfeat $C -o train.seed=1 ;
        $RUN --facts mstruct,mfeat $C -o train.seed=0 ;
        $RUN --facts mstruct,mfeat $C -o train.seed=1 ;
    fi
else
    stamp "REBUILD FAILED — see logs/rebuild.log.  Fallback lane on existing caches."
    $RUN $BPE -o train.seed=4 ;
    $RUN --facts mstruct $C -o train.seed=4 ;
    $RUN --facts pos $C -o train.seed=2 ;
    $RUN --facts mstruct $C -o train.seed=5 ;
    $RUN $BPE -o train.seed=5 ;
fi

if [ -n "$TAIL" ] && [ "$REBUILD_OK" -eq 0 ]; then
    stamp "TAIL headline pair, seed 4"
    $RUN $BPE -o train.seed=4 ;
    $RUN --facts mstruct $C -o train.seed=4 ;
fi

stamp "B4 BLiMP on every new cell, with its references"
$BLIMP --labels \
    bpe_baseline_bpe_17k \
    bpe_mstruct-LEARN-CAUSAL_bpe_17k \
    bpe_pos-LEARN-CAUSAL_bpe_17k \
    bpe_mfeatlex-LEARN-CAUSAL_bpe_17k \
    bpe_mstruct+mfeatlex-LEARN-CAUSAL_bpe_17k \
    bpe_mfeat-LEARN-CAUSAL_bpe_17k \
    bpe_mstruct+mfeat-LEARN-CAUSAL_bpe_17k ;

stamp "B5 temporal profiles and thesis tables"
python scripts/analysis/temporal_profile.py ;
python scripts/analysis/export_results.py --out exports/ ;

stamp "NIGHT CHAIN COMPLETE"
echo "Send back: results_N.zip of results/ and results/audits/, plus logs/."
