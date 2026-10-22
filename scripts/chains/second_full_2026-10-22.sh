#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# The second machine's chain of 22 October 2026, after second_2026-10-22.sh (journey §33.131): the
# text's request of 22 October, item 3 — Ukrainian gender and number at full training. At 10k what
# a causal tagger adds over the word form is not context (G-82, §33.129); at full training case's
# whole lead over the word form is the tagger's leak (§33.81). Does the same hold for gender and number?
#
#   gender_f   gender under endings apart, 151M tokens, 50k, seeds 0–3: tagged in context (`ugender`),
#              from the word form (`ugenderlex`) and from a causal tagger (`ugendercausal`)  P300–P303
#   number_f   the same for number (`unumber`, `unumberlex`, `unumbercausal`)
#   null_f     the null, the same stream and budget: the reference the split needs; its first seed is
#              the thesis machine's run 57f76ed51760 repeated (twin check)
#   split_f    G-82's split of what the causal tagger adds over the word form, once all are here
#
# Endings apart, where case's full training was measured (§33.81). Its stream came from the thesis
# machine; the derived codes were made here (scripts/data/derive_lexical_codes.py,
# derive_causal_ud.py). Guards: ready; pair (the first seed's first two cells end within SANITY_TOL
# of each other); twin; gpu, disk, END_BY — a run starts only if it can end before END_BY. Every run
# is a RESUMABLE PIECE. No tables here: exports/ is refreshed on the thesis machine.
#
#   PLAN=1 END_BY=… bash scripts/chains/second_full_2026-10-22.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/../.." || exit 1
source scripts/chains/lib/chain_v1.sh

PIECES=${PIECES-gender_f number_f null_f split_f}
TWIN_WHAT="the thesis machine's"
C="-o frozen.trainable=true -o frozen.causal_identity=true"
U="-o data.corpus=ukwiki -o data.corpus_holdout=2000 -o data.spacy_model=uk_core_news_sm"
full() {        # full FACT SEED — width 128, endings apart, 151M tokens, 50k
    echo "--facts $1 $U -o data.corpus_fraction=0.2 -o data.bpe_path=data/bpeend_uk_17k.json $C $(steps 50000) -o train.seed=$2"
}

minutes() {     # a run of each kind on this machine, with a margin
    [ -n "${MINUTES:-}" ] && { echo "$MINUTES"; return; }
    case "$1" in gender_f|number_f|null_f) echo 24 ;; esac     # 151M, 50k, width 128: 20.5–20.6 min (18 October)
}

# exits 0 when a loss_by_kind record of these facts at this budget is already here
LBK_RECORD='import glob, json, sys
for p in glob.glob("results/audits/loss_by_kind_*.json"):
    try:
        a = json.load(open(p)).get("args") or {}
    except (OSError, ValueError):
        continue
    if a.get("facts") == sys.argv[1:3] and a.get("fraction") == "0.2" and a.get("previous"):
        sys.exit(0)
sys.exit(1)'

split_full() {          # G-82's split at full training: once all of a feature's runs are here, once
    local f s c missing
    for f in ugender unumber; do
        stamp "split_f: what ${f}causal adds over ${f}lex, by the word the feature describes"
        if python -c "$LBK_RECORD" ${f}causal ${f}lex; then echo "  [skip] split_f, $f: a record of it is already here"; continue; fi
        missing=0
        for s in 0 1 2 3; do
            for c in mconst ${f}causal ${f}lex; do
                python scripts/ops/run_ids.py --done $(full $c $s) >/dev/null || missing=$((missing + 1))
            done
        done
        if [ "$missing" -gt 0 ]; then echo "  [skipped] split_f, $f: $missing of the 12 runs are not here"; continue; fi
        if [ -n "${PLAN:-}" ]; then echo "  [would run] split_f, $f — minutes"; continue; fi
        gpu_ok || return 0
        python scripts/analysis/loss_by_kind.py --machine second --fraction 0.2 --budget '' \
            --tokenizers bpeend_uk_17k --facts ${f}causal ${f}lex \
            --previous --previous-facts pos ${f}causal ${f}lex --unseen-lexicon $f \
            || echo "  [split_f] the split of $f FAILED — the chain goes on"
    done
}

run_piece() {
    case "$1" in
        gender_f) seedpiece "gender_f: item 3 — gender at full training under endings apart: in context, from the word form, causal" gender_f \
                      "0 1 2 3" pair "full ugender" "full ugenderlex" "full ugendercausal" ;;
        number_f) seedpiece "number_f: item 3 — number at full training under endings apart: in context, from the word form, causal" number_f \
                      "0 1 2 3" pair "full unumber" "full unumberlex" "full unumbercausal" ;;
        null_f)   seedpiece "null_f: item 3 — the null at full training under endings apart" null_f \
                      "0 1 2 3" twin:0.926563 "full mconst" ;;
        split_f)  split_full ;;
    esac
}

chain_main "the chain of 22 October, after the warm-up: gender and number at full training"
