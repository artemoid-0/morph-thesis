#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 25 November 2026 (journey §33.160): the review's remark on the learning
# rate of the larger models. The 30M model (width 512) at full training, 151M tokens and 50k: the nulls
# under BPE 17k and endings apart at a peak rate of 1e-3 instead of 3e-4, all else as before (the
# 100-step warm-up, the cosine schedule).
#
#   lr512     the two nulls at 1e-3, seeds 0–3, a seed's two runs together            P364–P366
#
# Each against this machine's nulls at 3e-4 (§33.142). Guards: ready; near (the first seed's BPE 17k
# null ends within SANITY_TOL of its run at 3e-4 here, 0.7447: a rate that diverges stops the chain before
# it spends the other seeds); gpu, disk, END_BY — a run starts only if it can end before END_BY. Every
# run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-11-25.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-lr512}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
r512() {        # r512 TOKENIZER SEED — the null at width 512, 151M tokens, 50k, the peak rate 1e-3
    echo "--facts mconst $U -o data.corpus_fraction=0.2 -o data.bpe_path=data/$1.json $C $(steps 50000) -o model.d_model=512 -o train.lr=0.001 -o train.seed=$2"
}
NULL_512=0.744722       # BPE 17k's null at 3e-4, width 512, seed 0, on this machine (run ae4ee76ed3f6)

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        lr512) echo 86 ;;       # width 512 at 50k: 80.2–81.0 min (25–26 October); the rate leaves the cost as it is
    esac
}

run_piece() {
    case "$1" in
        lr512) seedpiece "lr512: the 30M nulls at the peak rate 1e-3, full training" lr512 \
                   "0 1 2 3" "near:$NULL_512" "r512 bpe_uk_17k" "r512 bpeend_uk_17k" ;;
    esac
}

chain_main "the chain of 25 November: the learning rate at 30M parameters"
