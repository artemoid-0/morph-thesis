#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 23 October 2026 (journey §33.134): the text's request of 22 October,
# item 2 — English part of speech at full training from a causal tagger. At 10k and 20k four fifths
# of what the sentence's part of speech adds over the word form is the tagger's leak (§33.99,
# §33.107); at full training case's whole lead is (§33.81). Does English part of speech go the same way?
#
#   enfull   part of speech tagged in context (`pos`), from the word form (`poslex`) and from a causal
#            tagger (`poscausal`), WikiText-103, 50k, seeds 0–3; the word form's seeds 2–3 ran here
#            on 18 October. Its first run repeats the thesis machine's 06b2a3bb429b (twin check)
#                                                                                   P304–P306
#   ennull   the null, seeds 0–1 (2–3 ran here on 18 October): the reference the split needs; its
#            first run repeats the thesis machine's c8fe0eb7cfe8 (twin check)
#   ensplit  G-82's split of what the causal tagger adds over the word form, once all are here
#
# The thesis machine's runs of these cells predate fields the configuration gained later, so their
# repeats here take new run ids. Guards: ready; twin; gpu, disk, END_BY — a run starts only if it can
# end before END_BY. Every run is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the
# thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_en_2026-10-23.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-enfull ennull ensplit}
TWIN_WHAT="the thesis machine's"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
en() {          # en FACT SEED — width 128, WikiText-103, 50k
    echo "--facts $1 -o data.bpe_path=data/bpe_17k.json $C $(steps 50000) -o train.seed=$2"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in enfull|ennull) echo 24 ;; esac         # English, 50k, width 128: 21.0 min (18 October)
}

# exits 0 when a loss_by_kind record of these facts on WikiText-103 is already here
LBK_RECORD='import glob, json, sys
for p in glob.glob("results/audits/loss_by_kind_*.json"):
    try:
        a = json.load(open(p)).get("args") or {}
    except (OSError, ValueError):
        continue
    if a.get("facts") == sys.argv[1:3] and a.get("corpus") == "wikitext103" and a.get("previous"):
        sys.exit(0)
sys.exit(1)'

split_en() {            # G-82's split for English part of speech: once all its runs are here, once
    local s c missing=0
    stamp "ensplit: what poscausal adds over poslex, by the word the feature describes"
    if python -c "$LBK_RECORD" poscausal poslex; then echo "  [skip] ensplit: a record of it is already here"; return 0; fi
    for s in 0 1 2 3; do
        for c in mconst poscausal poslex; do
            python scripts/ops/run_ids.py --done $(en $c $s) >/dev/null || missing=$((missing + 1))
        done
    done
    if [ "$missing" -gt 0 ]; then echo "  [skipped] ensplit: $missing of the 12 runs are not here"; return 0; fi
    if [ -n "${PLAN:-}" ]; then echo "  [would run] ensplit — minutes"; return 0; fi
    gpu_ok || return 0
    python scripts/analysis/loss_by_kind.py --machine second --corpus wikitext103 --budget '' --tokenizers bpe_17k \
        --facts poscausal poslex --previous --previous-facts pos poscausal poslex --unseen-lexicon pos \
        || echo "  [ensplit] the split FAILED — the chain goes on"
}

run_piece() {
    case "$1" in
        enfull)  seedpiece "enfull: item 2 — English part of speech at full training: in context, from the word form, causal" enfull \
                     "0 1 2 3" twin:1.340535 "en pos" "en poslex" "en poscausal" ;;
        ennull)  seedpiece "ennull: item 2 — the English null at full training" ennull \
                     "0 1 2 3" twin:1.340149 "en mconst" ;;
        ensplit) split_en ;;
    esac
}

chain_main "the chain of 23 October: English part of speech at full training"
