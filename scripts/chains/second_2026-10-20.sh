#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 20 October 2026 (journey §33.119): the thesis's list of 19 October,
# item 1 — the tokenizer's gain beyond 30M parameters, at width 1024.
#
#   w1024_10k   width 1024 (85.3M parameters), ~38M tokens, 10k, seeds 0–3: the null under BPE 17k
#               and under endings apart                                               P267–P270
#   w1024_20k   the same at 20k, after the 10k half; what END_BY leaves runs on a later night  P271
#
# The fit check on this machine (scripts/ops/fit_check.py): 7.2 GB at peak, 4.0 steps a second,
# ~43 min per 10k run. Guards: ready; pair (the first seed's two nulls end within SANITY_TOL of
# each other); gpu, disk, END_BY — a run starts only if it can end before END_BY. Every run is a
# RESUMABLE PIECE: what a run of the chain finished is never run again. No tables here: exports/ is
# refreshed on the thesis machine, which holds every record.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-20.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-w1024_10k w1024_20k}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
W="-o model.d_model=1024"
null() {        # null TOKENIZER STEPS SEED
    echo "--facts mconst $U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$1.json $C $(steps $2) $W -o train.seed=$3"
}

minutes() {     # a run of each kind, with a margin over the fit check's 43 and 86 minutes
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in w1024_10k) echo 48 ;; w1024_20k) echo 94 ;; esac
}

run_piece() {
    case "$1" in
        w1024_10k) seedpiece "w1024_10k: item 1 — the nulls at width 1024, ~38M tokens, 10k" w1024_10k "0 1 2 3" pair \
                       "null bpe_uk_17k 10000" "null bpeend_uk_17k 10000" ;;
        w1024_20k) seedpiece "w1024_20k: item 1 — the nulls at width 1024, ~38M tokens, 20k" w1024_20k "0 1 2 3" pair \
                       "null bpe_uk_17k 20000" "null bpeend_uk_17k 20000" ;;
    esac
}

chain_main "the chain of 20 October: item 1, the tokenizer's gain at width 1024"
