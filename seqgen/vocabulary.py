import pickle
import os.path
from pathlib import Path


class Vocabulary(object):
    def __init__(self,
                 vocab_filename,
                 vocab_file=None,
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 vocab_from_file=False):
        """Initialize the vocabulary.
        Args:
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        if vocab_file is not None:
            self.vocab_file = vocab_file
        else:
            p = Path(vocab_filename)
            self.vocab_file = p.stem
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.vocab_filename = vocab_filename
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.load_vocab()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def load_vocab(self):
        """
        Load vocabluary from a text file
        """
        with open(self.vocab_filename, "r") as fp:
            content = fp.read()
            tokens = content.split("\n")
            for token in tokens:
                self.add_word(token)
                
    def encode(self, word):
        """
        Translate a word to an index
        """
        return self.word2idx[word]

    def encode_sequence(self, sequence):
        """
        Translate a sequence of words to a sequence of indices
        """
        return [self.word2idx[word] for word in sequence]

    def decode(self, index):
        """
        Translate an index to a word
        """
        return self.idx2word[index]

    def decode_sequence(self, sequence):
        """
        Translate a sequence of indices to a sequence of words
        """
        return [self.idx2word[index] for index in sequence]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
