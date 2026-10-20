"""
bpe.py — Pipeline 1: BPE baseline, the anchor cell.

CONTEXT ALIGNMENT. A BPE token sits inside some word, and that word has a
dependency role, a POS tag, a morphology. Aligning BPE tokens to spaCy tokens by
CHARACTER OFFSET lets BPE carry the context facts too, so injection can be
tested independently of segmentation (journey §15, "The structural payoff").

Unlike morph/det, which parse with spaCy anyway, BPE pays a new spaCy pass when
context facts are requested; with none requested spaCy is never loaded.
"""

from __future__ import annotations

from ..features.inflection import inflection_of, agreement_of


#: The context annotations a BPE token carries, in token-dict order: the spaCy
#: parse's (entity_label, dep, pos, minfl, magr, mfeat, udfeats) and the
#: WordAnalyser's (mroot, mhash, mlemma, mstruct, mrole).
ANNOTATED = ('entity_label', 'dep', 'pos', 'mroot', 'mhash', 'mlemma', 'mstruct',
             'mrole', 'minfl', 'magr', 'mfeat', 'udfeats')


def _first_annotated(arr, start, end):
    """
    First annotated character in a token's span. Scans the span: ByteLevel
    folds the preceding space into word-initial tokens, whose start offset is
    whitespace (bugs.md #14, journey §23).
    """
    if arr is None:
        return None
    n = len(arr)
    for c in range(max(start, 0), min(end, n)):
        if arr[c] is not None:
            return arr[c]
    return None


class BPEPipeline:
    """HuggingFace BPE wrapped in the shared pipeline interface."""

    def __init__(self, tokenizer_path=None, vocab_size=10000,
                 spacy_model='en_core_web_sm', want_context=False,
                 analyser=None, analyser_factory=None, verbose=True):
        from tokenizers import Tokenizer

        self.vocab_size = vocab_size
        self.want_context = want_context
        # WordAnalyser, present only when morphological facts are wanted.
        # This is what lets a BPE-tokenized model receive morphology WITHOUT
        # adopting morphological segmentation — the Level-2 design. Given as a
        # factory, it is built on first use (`analyser`).
        self._analyser = analyser
        self._analyser_factory = analyser_factory
        self._spacy_model = spacy_model
        self._nlp = None
        self.align_ok = 0
        self.align_fail = 0

        if tokenizer_path:
            if verbose:
                print(f'  Loading BPE tokenizer from {tokenizer_path}...')
            # load_tokenizer: a tokenizer trained on ending-marked text comes
            # wrapped, so the text encoded here is marked as its training was
            from ..data.tokenization import load_tokenizer
            self.tokenizer = load_tokenizer(tokenizer_path)
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

    def train(self, texts, save_path=None, pre_tokenizer='bytelevel',
              algorithm='bpe', presplit=None):
        """
        Train a subword tokenizer.

        algorithm='bpe'      greedy pairwise merges, the GPT-2 standard.
        algorithm='unigram'  prunes a large seed vocabulary by likelihood; its
            units align more closely with morphology (Bostrom & Durrett, 2020),
            so it is the third arm of Q1 (journey §20).

        pre_tokenizer='bytelevel'  (default) LOSSLESS: whitespace is folded into
            the following token as a Ġ prefix, so decode(encode(text)) == text
            and BPB is a true compression rate (journey §17).
        pre_tokenizer='whitespace' discards whitespace; kept only to reproduce
            the legacy anchor.

        presplit — an `endings.EndingRule`: `texts` arrive marked at every
        stem|ending cut, and the pre-tokenizer splits on the mark and removes it
        before ByteLevel, so no merge is learned across a cut. The round trip is
        then checked on marked Ukrainian text as well: decoding gives back the
        text without its marks.
        """
        from tokenizers.trainers import BpeTrainer, UnigramTrainer
        if algorithm == 'unigram':
            from tokenizers import Tokenizer
            from tokenizers.models import Unigram
            self.tokenizer = Tokenizer(Unigram())
        if presplit is not None and pre_tokenizer != 'bytelevel':
            raise ValueError('a presplit tokenizer is byte-level only')
        if pre_tokenizer == 'bytelevel':
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            # add_prefix_space MUST be False for exact round-trip. With True,
            # a space is prepended before encoding and the decoder faithfully
            # returns it, so decode(encode(t)) == ' ' + t. That is not
            # information loss (a constant prefix) but it breaks the
            # losslessness check and shifts per-line byte counts.
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            if presplit is not None:
                from ..data.endings import pre_tokenizer as marked_pre_tokenizer
                self.tokenizer.pre_tokenizer = marked_pre_tokenizer()
            self.tokenizer.decoder = ByteLevelDecoder()
            alphabet = ByteLevel.alphabet()
        elif pre_tokenizer == 'whitespace':
            from tokenizers.pre_tokenizers import Whitespace
            self.tokenizer.pre_tokenizer = Whitespace()
            alphabet = []
        else:
            raise ValueError(f'unknown pre_tokenizer {pre_tokenizer!r}')

        if algorithm == 'unigram':
            trainer = UnigramTrainer(
                vocab_size=self.vocab_size,
                special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'],
                initial_alphabet=alphabet,
                unk_token='[UNK]',
                show_progress=True,
            )
        else:
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'],
                initial_alphabet=alphabet,
                show_progress=True,
            )
        print(f'  Training {algorithm.upper()} (vocab_size={self.vocab_size}, '
              f'pre_tokenizer={pre_tokenizer})...')
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        print(f'  Done. Actual vocab size: {self.tokenizer.get_vocab_size()}')
        if pre_tokenizer == 'bytelevel':
            probes = [
                'The cat sat on the mat.',
                'Researchers found 42 yards, including 3 touchdowns.',
                'Multi-word  spacing\tand "quotes" (parens) 1990s.',
            ]
            pairs = [(t, t) for t in probes]
            if presplit is not None:
                from ..data.endings import mark_text
                for t in ('Вона прочитала книгами, а студентам — важливих новин 2020 року.',
                          'Об’єктами м’ясо НАТО… «Компанія» — Шевченка'):
                    pairs.append((mark_text(t, presplit)[0], t))
            bad = []
            for src, t in pairs:
                d = self.tokenizer.decode(self.tokenizer.encode(src).ids)
                if d != t:
                    bad.append((t, d))
            if bad:
                print('  ROUND-TRIP FAILED — this tokenizer is NOT lossless:')
                for t, d in bad:
                    print(f'    in : {t!r}')
                    print(f'    out: {d!r}')
                raise RuntimeError(
                    'ByteLevel BPE failed its round-trip check; refusing to '
                    'save a tokenizer that cannot reconstruct its input.')
            print(f'  round-trip verified on {len(pairs)} probes: LOSSLESS')

        if save_path:
            self.tokenizer.save(save_path)
            print(f'  Saved to {save_path}')

    # ── context alignment ─────────────────────────────────────────────────────

    @property
    def analyser(self):
        """The WordAnalyser, built by its factory the first time it is read; None without one."""
        if self._analyser is None and self._analyser_factory is not None:
            self._analyser = self._analyser_factory()
            self._analyser_factory = None
        return self._analyser

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            # English keeps every component (entity context needs the NER,
            # lemma facts the lemmatizer). The UD facts need only the
            # morphologizer and parser; Ukrainian's lemmatizer is a Python
            # dictionary lookup and its NER unused, so both are skipped —
            # the codes are identical (docs/reference/ukrainian_pipeline.md, block U1).
            skip = [] if self._spacy_model.startswith('en_') else ['ner', 'lemmatizer']
            self._nlp = spacy.load(self._spacy_model, disable=skip)
            # A paragraph is tagged whole, however long: fiction at 0.4 holds the
            # Bible as one line of 2.7M characters, past spaCy's default 1M. The
            # Ukrainian pipeline needs ~3 GB per million characters (measured on
            # it); 4M covers the corpus's longest document (3.6M). No text below
            # the default limit is touched, so no cache built before changes.
            self._nlp.max_length = max(self._nlp.max_length, 4_000_000)
        return self._nlp

    def _char_annotation_map(self, text):
        """
        Per character, the annotation of the spaCy word covering it, for every
        key in ANNOTATED: {key: [value or None for each character]}.

        Character-level rather than token-level because BPE boundaries do not
        respect word boundaries; `encode` reads each token's span from it.
        """
        doc = self.nlp(text)
        n = len(text)
        ann = {key: [None] * n for key in ANNOTATED}
        analyser = self.analyser
        for tok in doc:
            start, end = tok.idx, tok.idx + len(tok.text)
            alpha = tok.is_alpha
            word = {
                'dep': tok.dep_,
                'pos': tok.pos_,
                'entity_label': tok.ent_type_ if tok.ent_type_ else None,
                # INFLECTION of the word (Number=Plur, Tense=Past, ...): the
                # classifier of the morpheme tokenizer's inflection tokens, as a
                # word-level context fact (G-28, journey §32)
                'minfl': inflection_of(tok),
                # agreement (Number, Person), the full UD bundle and every UD
                # feature by name (ucase...), for alphabetic words only — the
                # coverage of the other morphological facts
                'magr': agreement_of(tok) if alpha else None,
                'mfeat': (str(tok.morph) or None) if alpha else None,
                'udfeats': (tok.morph.to_dict() or None) if alpha else None,
            }
            regions = ()
            if analyser is not None and alpha:
                root, word['mstruct'], regions = analyser.analyse(tok.text)
                word['mroot'] = analyser.root_code(root)
                word['mhash'] = analyser.hash_code(tok.text)
                word['mlemma'] = analyser.lemma_code(tok.lemma_)
            for key, value in word.items():
                column = ann[key]
                for c in range(start, min(end, n)):
                    column[c] = value
            for a, b, role in regions:
                for c in range(start + a, min(start + b, n)):
                    ann['mrole'][c] = role
        return ann

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode(self, text):
        encoding = self.tokenizer.encode(text)
        ann = self._char_annotation_map(text) if self.want_context else None

        result = []
        # read once: a tokenizers.Encoding builds a new list on every `.offsets`,
        # so indexing it per token made a paragraph cost the square of its length
        # — hours on fiction's long paragraphs (journey §33.87); the outputs are the same
        offsets = encoding.offsets
        for idx, (token, token_id) in enumerate(
                zip(encoding.tokens, encoding.ids)):
            t = {'token_id': token_id, 'token': token, 'category': 'bpe',
                 'role': 'bpe', 'entity_label': None, 'word': token}
            if ann is None:
                t.update(dict.fromkeys(ANNOTATED))
            else:
                start, end = offsets[idx]
                for key in ANNOTATED:
                    t[key] = _first_annotated(ann[key], start, end)
                if t['dep'] is None:
                    self.align_fail += 1
                else:
                    self.align_ok += 1
            result.append(t)
        return result

    def encode_ids(self, text):
        return self.tokenizer.encode(text).ids

    def alignment_rate(self):
        """Fraction of BPE tokens successfully aligned to a spaCy token."""
        total = self.align_ok + self.align_fail
        return (self.align_ok / total) if total else None

    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()
