#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 31 October 2026 (journey §33.143): the text's list of limits of 30
# October, the items whose code and data are ready. Every run is the 5M model at 151M tokens and 50k
# unless noted.
#
#   lr0       item 1, the learning rate: the nulls under BPE 17k and endings apart at 1e-3 and 3e-3,
#             and BPE 17k's null at the default 3e-4 as the twin of the thesis machine's, seed 0
#                                                                                       P321
#   lrbest    the better rate's seeds 1–3, decided on lr0: the rate at which BPE 17k's null ends lowest
#             at seed 0, of the three. If it is the default, nothing runs; otherwise both nulls at that
#             rate                                                                      P322–P323
#   plain128  item 12, the tokenizer without a slot: plain baselines under both tokenizers, seeds 0–3,
#             BPE 17k's the twins of the thesis machine's; and BPE 17k's null at the default rate, seeds
#             1–3, the twins too, so that the gain between the nulls at the default rate is paired by
#             seed on this machine (its endings-apart nulls at that rate are here already) for items 1 and 12
#                                                                                       P324–P325
#   wu128     item 14: the nulls under both tokenizers with the 1,000-step warm-up, ~38M tokens, 10k,
#             seeds 0–3                                                                 P326
#
# The rate is chosen at the budget the claim is about: a rate best at 10k need not be best at 50k
# under the cosine schedule. It is chosen for BPE 17k, the reference, so the comparison answers
# whether the reference was held back by its rate. Guards: ready; twin (the first run ends within
# TWIN_TOL of the thesis machine's); lrbest runs only if lr0's twin passed; pair (wu128's first two
# runs end within SANITY_TOL of each other); gpu, disk, END_BY — a run starts only if it can end
# before END_BY. Every run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis
# machine. Width 1024 at full training follows: second_wide_2026-10-31.sh.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_2026-10-31.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-lr0 lrbest plain128 wu128}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
n50() {         # n50 RATE TOKENIZER SEED — the null, 151M tokens, 50k; the default rate adds nothing
    local lr=""
    [ "$1" != 0.0003 ] && lr="-o train.lr=$1"
    echo "--facts mconst $U -o data.corpus_fraction=0.2 -o data.bpe_path=data/$2.json $C $(steps 50000) $lr -o train.seed=$3"
}
p50() {         # p50 TOKENIZER SEED — the plain baseline, no slot, 151M tokens, 50k
    echo "$U -o data.corpus_fraction=0.2 -o data.bpe_path=data/$1.json $(steps 50000) -o train.seed=$2"
}
w10() {         # w10 TOKENIZER SEED — the null, ~38M tokens, 10k, the 1,000-step warm-up
    echo "--facts mconst $U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$1.json $C $(steps 10000) -o train.warmup=1000 -o train.seed=$2"
}
NULL_TWIN=0.941868      # BPE 17k's null at 3e-4, seed 0, on the thesis machine (run c58ba0662048)
PLAIN_TWIN=0.947546     # BPE 17k's plain baseline, seed 0, on the thesis machine (run b1515377c268)

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        lr0|lrbest|plain128) echo 24 ;;     # the 5M model at 50k: 20.5–21.2 min (16–24 October)
        wu128) echo 6 ;;                    # at 10k on ~38M tokens: 4.1–4.4 min
    esac
}

best_rate() {   # the rate at which BPE 17k's null ends lowest at seed 0 here, of the three; fails if one is missing
    local r b best="" low=""
    for r in 0.0003 0.001 0.003; do
        b=$(bpb "$(n50 $r bpe_uk_17k 0)")
        [ -n "$b" ] || return 1
        if [ -z "$low" ] || awk -v b="$b" -v l="$low" 'BEGIN { exit !(b < l) }'; then low=$b; best=$r; fi
    done
    echo "$best"
}

lr_best() {     # lrbest: decided on lr0's seed 0
    local best
    stamp "lrbest: decided on lr0 — the rate at which BPE 17k's null ends lowest at seed 0"
    if [ -n "${PLAN:-}" ]; then
        echo "  [plan] lrbest: decided once lr0 is here — planned as if a new rate wins"; best=0.003
    else
        within "lrbest: lr0's twin" "$(bpb "$(n50 0.0003 bpe_uk_17k 0)")" "$NULL_TWIN" "${TWIN_TOL:-0.01}" \
            "the other machine's" || { echo "  [skipped] lrbest: lr0's twin is missing or failed its check"; return 0; }
        best=$(best_rate) || { echo "  [skipped] lrbest: seed 0 at the three rates is not all here"; return 0; }
        if [ "$best" = 0.0003 ]; then echo "  [decided] lrbest does not run: the default rate is the best of the three"; return 0; fi
        echo "  [decided] lrbest runs at $best"
    fi
    seedpiece "lrbest: both nulls at rate $best, seeds 1–3" lrbest "1 2 3" none "n50 $best bpe_uk_17k" "n50 $best bpeend_uk_17k"
}

run_piece() {
    case "$1" in
        lr0)      seedpiece "lr0: the nulls at rates 1e-3 and 3e-3, and BPE 17k's at the default as the twin" lr0 \
                      "0" twin:$NULL_TWIN "n50 0.0003 bpe_uk_17k" "n50 0.001 bpe_uk_17k" "n50 0.001 bpeend_uk_17k" \
                      "n50 0.003 bpe_uk_17k" "n50 0.003 bpeend_uk_17k" ;;
        lrbest)   lr_best ;;
        plain128) seedpiece "plain128: plain baselines under both tokenizers, and BPE 17k's null at the default rate" plain128 \
                      "0 1 2 3" twin:$PLAIN_TWIN "p50 bpe_uk_17k" "p50 bpeend_uk_17k" "n50 0.0003 bpe_uk_17k" ;;
        wu128)    seedpiece "wu128: the nulls with the 1,000-step warm-up, ~38M tokens, 10k" wu128 \
                      "0 1 2 3" pair "w10 bpe_uk_17k" "w10 bpeend_uk_17k" ;;
    esac
}

chain_main "the chain of 31 October: the learning rate, the plain baselines, the warm-up at width 128"
