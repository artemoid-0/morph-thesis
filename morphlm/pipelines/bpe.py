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

from ..features.inflection import inflection_of, agreement_of


def _first_annotated(arr, start, end):
    """
    First annotated character inside a token's span.

    CRITICAL, and the source of a bug that would have silently voided every
    BPE + context-fact experiment: ByteLevel folds the preceding space INTO the
    following token, so a word-initial token spans " cat" and its START OFFSET
    lands on whitespace, which no spaCy token covers. Indexing by start offset
    therefore returned None for every word-initial token — 78% of the BPE
    stream (211,300 of 270,848, measured in category_bits). Scanning the span
    fixes it: 17% -> 100% alignment on a synthetic sentence.

    Same class of error as the region-attribution bug in coverage_economics
    (journey section 19). Alignment logic is the default suspect here.
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
                 analyser=None, verbose=True):
        from tokenizers import Tokenizer

        self.vocab_size = vocab_size
        self.want_context = want_context
        # WordAnalyser, present only when morphological facts are wanted.
        # This is what lets a BPE-tokenized model receive morphology WITHOUT
        # adopting morphological segmentation — the Level-2 design.
        self.analyser = analyser
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

        algorithm='unigram'  a probabilistic model that prunes a large seed
            vocabulary by likelihood. Bostrom & Durrett (2020) found Unigram
            recovers subword units aligning more closely with morphology than
            BPE's greedy merging, and matching or beating it downstream. It is
            the natural third arm: if the morpheme pipeline also loses to
            Unigram, the Q1 result is not an artifact of one particular
            statistical tokenizer, which is the obvious objection to a
            two-arm comparison.

        pre_tokenizer='bytelevel'  (default, GPT-2 standard)
            LOSSLESS: whitespace is folded into the following token as a Ġ
            prefix, so decode(encode(text)) == text exactly. Required for BPB to
            be a genuine compression rate comparable to published numbers.

        pre_tokenizer='whitespace'  (the original choice)
            Splits on word/non-word boundaries and DISCARDS whitespace, so it cannot
            reconstruct the input. Kept only to reproduce the legacy anchor.

        The coverage audit showed whitespace is 18.8% of corpus bytes. Both the
        old BPE and the old morpheme pipelines dropped it, so it never biased
        Q1 — but it meant neither BPB was a real compression rate.

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
        mroot = [None] * n
        mhash = [None] * n
        mlemma = [None] * n
        mstruct = [None] * n
        mrole = [None] * n
        minfl = [None] * n
        magr = [None] * n
        mfeat = [None] * n
        udf = [None] * n
        for tok in doc:
            start, end = tok.idx, tok.idx + len(tok.text)
            label = tok.ent_type_ if tok.ent_type_ else None
            # INFLECTION of the word (Number=Plur, Tense=Past, ...), read from
            # spaCy's morphologizer on the parse the pipeline already runs. The
            # classifier is the one the morpheme tokenizer uses for its own
            # inflection tokens (features/inflection.py); here it becomes a
            # word-level context fact for the BPE model — G-28, journey §32.
            infl = inflection_of(tok)
            # agreement (Number, Person) and the full UD bundle, for alphabetic
            # words only — the same coverage as the other morphological facts
            agr = agreement_of(tok) if tok.is_alpha else None
            feats = (str(tok.morph) or None) if tok.is_alpha else None
            # every UD feature by name, for the language-neutral blocks (ucase...)
            ud = (tok.morph.to_dict() or None) if tok.is_alpha else None
            for c in range(start, min(end, n)):
                deps[c] = tok.dep_
                poss[c] = tok.pos_
                ents[c] = label
                minfl[c] = infl
                magr[c] = agr
                mfeat[c] = feats
                udf[c] = ud
            if self.analyser is None or not tok.is_alpha:
                continue
            root, struct, regions = self.analyser.analyse(tok.text)
            rcode = self.analyser.root_code(root)
            hcode = self.analyser.hash_code(tok.text)
            lcode = self.analyser.lemma_code(tok.lemma_)
            for c in range(start, min(end, n)):
                mroot[c] = rcode
                mhash[c] = hcode
                mlemma[c] = lcode
                mstruct[c] = struct
            for a, b, role in regions:
                for c in range(start + a, min(start + b, n)):
                    mrole[c] = role
        return (deps, poss, ents, mroot, mhash, mlemma, mstruct, mrole,
                minfl, magr, mfeat, udf)

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode(self, text):
        encoding = self.tokenizer.encode(text)

        deps = poss = ents = mroot = mhash = mlemma = mstruct = mrole = minfl = \
            magr = mfeat = udf = None
        if self.want_context:
            (deps, poss, ents, mroot, mhash, mlemma,
             mstruct, mrole, minfl, magr, mfeat, udf) = self._char_annotation_map(text)

        result = []
        # read once: a tokenizers.Encoding builds a new list on every `.offsets`,
        # so indexing it per token made a paragraph cost the square of its length
        # — hours on fiction's long paragraphs (journey §33.87); the outputs are the same
        offsets = encoding.offsets
        for idx, (token, token_id) in enumerate(
                zip(encoding.tokens, encoding.ids)):
            dep = pos = ent = None
            mr = mh = ml = ms = mrl = mi = ma = mf = uf = None
            if self.want_context:
                start, end = offsets[idx]
                dep = _first_annotated(deps, start, end)
                pos = _first_annotated(poss, start, end)
                ent = _first_annotated(ents, start, end)
                mr = _first_annotated(mroot, start, end)
                mh = _first_annotated(mhash, start, end)
                ml = _first_annotated(mlemma, start, end)
                ms = _first_annotated(mstruct, start, end)
                mrl = _first_annotated(mrole, start, end)
                mi = _first_annotated(minfl, start, end)
                ma = _first_annotated(magr, start, end)
                mf = _first_annotated(mfeat, start, end)
                uf = _first_annotated(udf, start, end)
                if dep is None:
                    self.align_fail += 1
                else:
                    self.align_ok += 1
            result.append({
                'token_id': token_id,
                'token': token,
                'category': 'bpe',
                'role': 'bpe',
                'entity_label': ent,
                'word': token,
                'dep': dep,
                'pos': pos,
                'mroot': mr,
                'mhash': mh,
                'mlemma': ml,
                'mstruct': ms,
                'mrole': mrl,
                'minfl': mi,
                'magr': ma,
                'mfeat': mf,
                'udfeats': uf,
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
