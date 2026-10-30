#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 31 October 2026, after second_2026-10-31.sh (journey §33.143): the
# text's list of limits of 30 October, item 2 — width 1024 at full training. At 20k on the 151M-token
# corpus the tokenizer's gain at width 1024 is +0.0082 (1.09%) and still rising (§33.141); at full
# training it is measured at 5M and 30M parameters only (1.75%, 0.97%). Does it last at 85M?
#
#   full1024  the nulls at width 1024, 151M tokens, 50k, the 1,000-step warm-up, under BPE 17k and
#             endings apart, seeds 0–3                                                  P327–P328
#   mb1024    MultiBLiMP on the eight, once all are here                                P329
#
# The nulls, as at 5M and 30M parameters at full training and in width 1024's cells at 10k and 20k, so
# that the gain compares like with like along both axes; at 10k the nulls overstate it at this width
# by 0.0032 against the empty slots (§33.139), which the text states beside it. A seed's pair runs
# together, so a chain cut short keeps whole pairs. Guards: ready; pair (the first seed's two nulls
# end within SANITY_TOL of each other); gpu, disk, END_BY — a run starts only if it can end before
# END_BY. Every run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_wide_2026-10-31.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-full1024 mb1024}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
null() {        # null TOKENIZER SEED — width 1024, 151M tokens, 50k, the longer warm-up
    echo "--facts mconst $U -o data.corpus_fraction=0.2 -o data.bpe_path=data/$1.json $C $(steps 50000) -o model.d_model=1024 -o train.warmup=1000 -o train.seed=$2"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in full1024) echo 235 ;; esac     # width 1024: 88.7–89.3 min at 20k (24 October), so ~223 at 50k
}

# exits 0 when a MultiBLiMP record of labels ending in its argument is already here
MB_RECORD='import glob, json, sys
for p in glob.glob("results/audits/multiblimp_eval_*.json"):
    try:
        a = json.load(open(p)).get("args") or {}
    except (OSError, ValueError):
        continue
    if any(str(l).endswith(sys.argv[1]) for l in a.get("labels") or []):
        sys.exit(0)
sys.exit(1)'

multiblimp_1024() {     # MultiBLiMP on the eight nulls: once all eight are here, once
    local s t missing=0
    stamp "mb1024: MultiBLiMP on width 1024's nulls at full training, seeds 0–3"
    if python -c "$MB_RECORD" _ukwiki0.2_wu1000_w1024; then echo "  [skip] mb1024: a record of it is already here"; return 0; fi
    for s in 0 1 2 3; do
        for t in bpe_uk_17k bpeend_uk_17k; do
            python scripts/ops/run_ids.py --done $(null $t $s) >/dev/null || missing=$((missing + 1))
        done
    done
    if [ "$missing" -gt 0 ]; then echo "  [skipped] mb1024: $missing of the 8 runs are not here"; return 0; fi
    if [ -n "${PLAN:-}" ]; then echo "  [would run] mb1024 — minutes"; return 0; fi
    gpu_ok || return 0
    python scripts/eval/multiblimp_eval.py --machine second \
        --labels bpe_mconst-LEARN-CAUSAL_bpe_uk_17k_ukwiki0.2_wu1000_w1024 \
                 bpe_mconst-LEARN-CAUSAL_bpeend_uk_17k_ukwiki0.2_wu1000_w1024 \
        || echo "  [mb1024] MultiBLiMP FAILED — the chain goes on"
}

run_piece() {
    case "$1" in
        full1024) seedpiece "full1024: item 2 — the nulls at width 1024, full training, a 1,000-step warm-up" full1024 \
                      "0 1 2 3" pair "null bpe_uk_17k" "null bpeend_uk_17k" ;;
        mb1024)   multiblimp_1024 ;;
    esac
}

chain_main "the chain of 31 October, after the short one: width 1024 at full training"
