#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain for item 2 of the thesis's list of 19 October (journey §33.125):
# number and gender as context-aware features. Started after the chain of 21 October
# (second_2026-10-21.sh) has let go of its lock, so the two never share the GPU.
#
#   gender_b, gender_e   gender under BPE 17k and under endings apart, ~38M tokens, 10k, seeds 0–3:
#                        tagged in context (`ugender`), from the word form (`ugenderlex`) and from a
#                        causal tagger (`ugendercausal`)                               P286–P291
#   number_b, number_e   the same for number (`unumber`, `unumberlex`, `unumbercausal`)
#
# The references, the null and the empty slot at 10k under both tokenizers, are already on this
# machine (G-78). Guards: ready (the derived codes, scripts/data/derive_lexical_codes.py and
# derive_causal_ud.py); pair (the first seed's first two cells end within SANITY_TOL of each
# other); gpu, disk, END_BY. Every run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on
# the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_item2_2026-10-21.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-gender_b gender_e number_b number_e}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
cell() {        # cell FACT TOKENIZER SEED — width 128, ~38M tokens, 10k
    echo "--facts $1 $U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$2.json $C $(steps 10000) -o train.seed=$3"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in gender_b|gender_e|number_b|number_e) echo 5 ;; esac     # width 128, 10k: 4.1–4.4 min
}

run_piece() {
    case "$1" in
        gender_b) seedpiece "gender_b: item 2 — gender under BPE 17k: in context, from the word form, causal" gender_b \
                      "0 1 2 3" pair "cell ugender bpe_uk_17k" "cell ugenderlex bpe_uk_17k" "cell ugendercausal bpe_uk_17k" ;;
        gender_e) seedpiece "gender_e: item 2 — gender under endings apart: in context, from the word form, causal" gender_e \
                      "0 1 2 3" pair "cell ugender bpeend_uk_17k" "cell ugenderlex bpeend_uk_17k" "cell ugendercausal bpeend_uk_17k" ;;
        number_b) seedpiece "number_b: item 2 — number under BPE 17k: in context, from the word form, causal" number_b \
                      "0 1 2 3" pair "cell unumber bpe_uk_17k" "cell unumberlex bpe_uk_17k" "cell unumbercausal bpe_uk_17k" ;;
        number_e) seedpiece "number_e: item 2 — number under endings apart: in context, from the word form, causal" number_e \
                      "0 1 2 3" pair "cell unumber bpeend_uk_17k" "cell unumberlex bpeend_uk_17k" "cell unumbercausal bpeend_uk_17k" ;;
    esac
}

chain_main "item 2: number and gender as context-aware features"
