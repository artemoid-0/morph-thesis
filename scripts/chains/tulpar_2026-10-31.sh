#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The thesis machine's chain of 31 October 2026 (journey §33.144): the first tier of the text's list
# of limits of 30 October, on this machine, where every run each is compared with already is. The
# second machine holds width 1024 at full training until Sunday.
#
#   shuf      item 10, shuffled case: case from the word form, its values reassigned among word types
#             of similar frequency (`ucaseshuf`), under BPE 17k, ~38M tokens, 10k, seeds 0–3
#                                                                                       P330–P331
#   const     item 11, the switch to a constant: English part of speech set to one class at step
#             5,000 of 20k (frozen.feature_const_step), seeds 0–4, the switch to zeros' seeds
#                                                                                       P332–P333
#   cutshift  item 9, where the cut falls: the null under BPE 17k trained with the rule's words cut
#             one letter off its boundary (`bpeshift_uk_17k`), ~38M tokens, 10k, seeds 0–3
#                                                                                       P334–P335
#
# cutshift's data is built beside the GPU work by the command whose PID is PREP_PID (the tokenizer,
# its statistics, then prepare_condition.py); the piece waits for it, then runs only if its caches
# are all here. Guards: ready; near (a piece's first run ends within SANITY_TOL of a run of this
# machine's that it is compared with); gpu, disk, END_BY — a run starts only if it can end before
# END_BY. Every run is a RESUMABLE PIECE.
#
#   PLAN=1 END_BY=… bash scripts/chains/tulpar_2026-10-31.sh
#   nohup systemd-inhibit --what=sleep:idle env PREP_PID=… END_BY=… bash scripts/chains/tulpar_2026-10-31.sh >/dev/null 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-shuf const cutshift}
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
uk10() {        # uk10 FACT TOKENIZER SEED — ~38M tokens, 10k
    echo "--facts $1 $U -o data.corpus_fraction=0.05 -o data.bpe_path=data/$2.json $C $(steps 10000) -o train.seed=$3"
}
const20() {     # const20 SEED — English part of speech, 20k, set to one class at step 5,000
    echo "--facts pos -o data.bpe_path=data/bpe_17k.json $C $(steps 20000) -o frozen.feature_const_step=5000 -o train.seed=$1"
}
LEX_B=1.088812          # case from the word form under BPE 17k, seed 0, this machine
OFF5000=1.429887        # English part of speech switched to zeros at 5,000 of 20k, seed 0, this machine
NULL_E=1.071363         # the endings-apart null at 10k, seed 0, this machine

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in
        shuf|cutshift) echo 13 ;;           # 10k on ~38M tokens: 10.8–11.8 min (October)
        const) echo 25 ;;                   # English at 20k: 21.5–23.1 min
    esac
}

wait_prep() {   # cutshift: wait while the data's build runs (its PID), until END_BY
    [ -n "${PLAN:-}" ] && return 0
    [ -n "${PREP_PID:-}" ] || return 0
    stamp "cutshift: waiting for its data's build (PID $PREP_PID)"
    while kill -0 "$PREP_PID" 2>/dev/null; do
        if [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "  [cutshift] END_BY reached while its data was built"; return 1; fi
        sleep "${PREP_WAIT_SECONDS:-120}"
    done
}

run_piece() {
    case "$1" in
        shuf)     seedpiece "shuf: shuffled case, from the word form, under BPE 17k, 10k" shuf \
                      "0 1 2 3" near:$LEX_B "uk10 ucaseshuf bpe_uk_17k" ;;
        const)    seedpiece "const: English part of speech set to one class at step 5,000 of 20k" const \
                      "0 1 2 3 4" near:$OFF5000 const20 ;;
        cutshift) wait_prep || return 0
                  seedpiece "cutshift: the null under BPE 17k with the cut one letter off the boundary, 10k" cutshift \
                      "0 1 2 3" near:$NULL_E "uk10 mconst bpeshift_uk_17k" ;;
    esac
}

chain_main "the thesis machine's chain of 31 October: shuffled case, the switch to a constant, the cut moved"
