"""
bpe.py — Pipeline 1: BPE baseline, the anchor cell.

NEW CAPABILITY: context alignment.
----------------------------------
A BPE token sits inside some word, and that word has a dependency role and a
POS tag. Aligning BPE tokens to spaCy tokens by CHARACTER OFFSET lets the BPE
pipeline carry the same context facts as the morpheme pipelines.

That matters structurally: it decouples the thesis's two legs. Previously only
morpheme pipelines could be frozen, so 'does segmentation help' and 'does
injection help' were entangled. With BPE+context the matrix becomes

                no-freeze   static-freeze   context-freeze
        bpe         y             -               y
        morph       y             y               y
        det         y             y               y

so injection can be tested independently of segmentation, and the interaction
(does linguistic freezing help MORE when the units are already linguistic?)
becomes a question you can actually answer.

COST NOTE: unlike morph/det — which already run the full spaCy pipeline for
NER and simply discard tagger/parser output — BPE pays a genuinely new spaCy
pass when context facts are requested. With no context facts requested, spaCy
is never loaded and this pipeline behaves exactly as before.
"""

from __future__ import annotations


class BPEPipeline:
    """HuggingFace BPE wrapped in the shared pipeline interface."""

    def __init__(self, tokenizer_path=None, vocab_size=10000,
                 spacy_model='en_core_web_sm', want_context=False,
                 verbose=True):
        from tokenizers import Tokenizer

        self.vocab_size = vocab_size
        self.want_context = want_context
        self._spacy_model = spacy_model
        self._nlp = None
        self.align_ok = 0
        self.align_fail = 0

        if tokenizer_path:
            if verbose:
                print(f'  Loading BPE tokenizer from {tokenizer_path}...')
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            if verbose:
                print(f'  Vocabulary: {self.tokenizer.get_vocab_size()} tokens')
        else:
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import Whitespace
            self.tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
            self.tokenizer.pre_tokenizer = Whitespace()
            if verbose:
                print('  BPE tokenizer ready for training.')

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, texts, save_path=None):
        from tokenizers.trainers import BpeTrainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'],
            show_progress=True,
        )
        print(f'  Training BPE (vocab_size={self.vocab_size})...')
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        print(f'  Done. Actual vocab size: {self.tokenizer.get_vocab_size()}')
        if save_path:
            self.tokenizer.save(save_path)
            print(f'  Saved to {save_path}')

    # ── context alignment ─────────────────────────────────────────────────────

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self._spacy_model)
        return self._nlp

    def _char_annotation_map(self, text):
        """
        Build a per-character lookup: char index -> (dep, pos, ent_label).

        Character-level rather than token-level because BPE boundaries do not
        respect word boundaries; a BPE token's start offset is enough to find
        the word containing it.
        """
        doc = self.nlp(text)
        n = len(text)
        deps = [None] * n
        poss = [None] * n
        ents = [None] * n
        for tok in doc:
            start, end = tok.idx, tok.idx + len(tok.text)
            label = tok.ent_type_ if tok.ent_type_ else None
            for c in range(start, min(end, n)):
                deps[c] = tok.dep_
                poss[c] = tok.pos_
                ents[c] = label
        return deps, poss, ents

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode(self, text):
        encoding = self.tokenizer.encode(text)

        deps = poss = ents = None
        if self.want_context:
            deps, poss, ents = self._char_annotation_map(text)

        result = []
        for idx, (token, token_id) in enumerate(
                zip(encoding.tokens, encoding.ids)):
            dep = pos = ent = None
            if self.want_context:
                start, end = encoding.offsets[idx]
                if 0 <= start < len(deps):
                    dep, pos, ent = deps[start], poss[start], ents[start]
                    if dep is None:
                        self.align_fail += 1
                    else:
                        self.align_ok += 1
                else:
                    self.align_fail += 1
            result.append({
                'token_id': token_id,
                'token': token,
                'category': 'bpe',
                'role': 'bpe',
                'entity_label': ent,
                'word': token,
                'dep': dep,
                'pos': pos,
            })
        return result

    def encode_ids(self, text):
        return self.tokenizer.encode(text).ids

    def alignment_rate(self):
        """Fraction of BPE tokens successfully aligned to a spaCy token."""
        total = self.align_ok + self.align_fail
        return (self.align_ok / total) if total else None

    def pretty_print(self, text):
        units = self.encode(text)
        print(f'Input: {text}')
        print('  ' + ' | '.join(u['token'] for u in units))
        if self.want_context:
            print('  deps: ' + ' | '.join(str(u['dep'] or '-') for u in units))
        print()

    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()
