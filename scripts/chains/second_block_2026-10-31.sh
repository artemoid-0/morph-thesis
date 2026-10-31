#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 31 October 2026, after second_wide_2026-10-31.sh (journey §33.146): the
# text's list of limits of 30 October, items 7 and 13 — the tokenizer's gain under a modern block and
# in a deeper model. Width and data were varied; the block and the depth (4 layers) never were.
#
#   block10   item 7: the nulls under BPE 17k and endings apart with the LLaMA-style block (rotary
#             positions, RMSNorm, SwiGLU; built, §33.54), width 128, ~38M tokens, 10k, seeds 0–3
#                                                                                       P337–P338
#   deep10    item 13: the nulls at 8 layers of width 512, the 1,000-step warm-up, ~38M tokens, 10k,
#             seeds 0–3                                                                 P339–P340
#
# Each against this machine's nulls of the same width and budget: width 128's standard block, and
# width 512's 4 layers under the same warm-up (§33.132). Guards: ready; pair (the first seed's two
# nulls end within SANITY_TOL of each other); gpu, disk, END_BY — a run starts only if it can end
# before END_BY. Every run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis
# machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_block_2026-10-31.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-block10 deep10}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
null() {        # null TOKENIZER SEED — ~38M tokens, 10k
    echo "--facts mconst $U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$1.json $C $(steps 10000) -o train.seed=$2"
}
block() {       # block TOKENIZER SEED — the LLaMA-style block at width 128
    echo "$(null "$1" "$2") -o model.position=rope -o model.norm=rmsnorm -o model.mlp=swiglu"
}
deep() {        # deep TOKENIZER SEED — 8 layers at width 512, the longer warm-up
    echo "$(null "$1" "$2") -o model.d_model=512 -o model.n_layers=8 -o train.warmup=1000"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        block10) echo 6 ;;      # width 128 at 10k: 4.1–4.4 min; the block's weights are as many
        deep10) echo 32 ;;      # 4 layers at width 512: 16.0–17.6 min; twice the layers, the output layer once
    esac
}

run_piece() {
    case "$1" in
        block10) seedpiece "block10: item 7 — the nulls under the LLaMA-style block, width 128, 10k" block10 \
                     "0 1 2 3" pair "block bpe_uk_17k" "block bpeend_uk_17k" ;;
        deep10)  seedpiece "deep10: item 13 — the nulls at 8 layers of width 512, 10k" deep10 \
                     "0 1 2 3" pair "deep bpe_uk_17k" "deep bpeend_uk_17k" ;;
    esac
}

chain_main "the chain of 31 October, after width 1024: the modern block and depth"
