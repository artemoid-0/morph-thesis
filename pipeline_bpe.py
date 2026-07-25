"""
pipeline_bpe.py

Pipeline 1 — BPE baseline.
Trains a BPE tokenizer on WikiText-103 and provides the same
encode / encode_ids / pretty_print interface as the other pipelines.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPEPipeline:
    """
    Wraps HuggingFace BPE tokenizer with the same interface
    as the morpheme-based pipelines for direct comparison.
    """

    def __init__(self, tokenizer_path=None, vocab_size=10000):
        """
        Load a saved tokenizer from tokenizer_path, or set up
        an untrained one ready for train().
        """
        self.vocab_size = vocab_size

        if tokenizer_path:
            print(f'  Loading BPE tokenizer from {tokenizer_path}...')
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f'  Vocabulary: {self.tokenizer.get_vocab_size()} tokens')
        else:
            self.tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
            self.tokenizer.pre_tokenizer = Whitespace()
            print('  BPE tokenizer ready for training.')

    def train(self, texts, save_path=None):
        """Train BPE on a list of text strings."""
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

    def encode(self, text):
        """
        Encode text and return list of token dicts matching
        the format of morpheme pipelines for comparison.
        """
        encoding = self.tokenizer.encode(text)
        result = []
        for token, token_id in zip(encoding.tokens, encoding.ids):
            result.append({
                'token_id':     token_id,
                'token':        token,
                'category':     'bpe',
                'role':         'bpe',
                'entity_label': None,
                'word':         token,
            })
        return result

    def encode_ids(self, text):
        return self.tokenizer.encode(text).ids

    def pretty_print(self, text):
        units = self.encode(text)
        tokens = ' | '.join(u['token'] for u in units)
        print(f'Input: {text}')
        print(f'  {tokens}')
        print()

    def vocab_size_actual(self):
        return self.tokenizer.get_vocab_size()