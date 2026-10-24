#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 24 October 2026 (journey §33.136): the text's request of 24 October,
# item 2 — the empty slot at width 1024. The tokenizer's gain at 85M parameters is measured between
# the two tokenizers' same-shape nulls (§33.132); method point 10 asks for the empty slot at every
# model size, and at 30M the two references agree (§33.118).
#
#   empty1024   the empty slot (the null switched off from step 0) at width 1024, ~38M tokens, 10k,
#               a 1,000-step warm-up, under BPE 17k and endings apart, seeds 0–3   P307–P309
#
# Its partners, the nulls with the same warm-up, ran on 22 October (second_2026-10-22.sh). The width
# 1024 runs without repeating the text (item 1) follow in second_wide_2026-10-24.sh, started after this
# chain lets go of its lock. Guards: ready; pair (the first seed's two empty slots end within
# SANITY_TOL of each other); gpu, disk, END_BY — a run starts only if it can end before END_BY. Every
# run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-24.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-empty1024}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
empty() {       # empty TOKENIZER SEED — the null switched off from step 0, width 1024, the longer warm-up
    echo "--facts mconst $U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$1.json $C $(steps 10000) -o model.d_model=1024 -o train.warmup=1000 -o frozen.feature_off_step=0 -o train.seed=$2"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in empty1024) echo 48 ;; esac      # width 1024, 10k: 44.1–44.4 min (20 and 22 October)
}

run_piece() {
    case "$1" in
        empty1024) seedpiece "empty1024: item 2 — the empty slot at width 1024 with a 1,000-step warm-up, ~38M tokens, 10k" empty1024 \
                       "0 1 2 3" pair "empty bpe_uk_17k" "empty bpeend_uk_17k" ;;
    esac
}

chain_main "the chain of 24 October: the empty slot at width 1024"
