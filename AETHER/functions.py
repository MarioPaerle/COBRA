import numpy as np
import spacy
from spacy_syllables import SpacySyllables


nlp = spacy.load("en_core_web_sm")
if "syllables" not in nlp.pipe_names:
    nlp.add_pipe("syllables", after="tagger")


class Tokenizer:
    """A Simple Tokenizer :)"""
    def __init__(self, splitter=' ', undefined=0):
        self._undefined = undefined
        self._splitter = splitter
        self._tokens_dict = dict()

    def tokenize(self, text):
        tokens = text.split(self._splitter)
        return [self._tokens_dict.get(t, self._undefined) for t in tokens]

    def add_token(self, token, bypass=False):
        token = token.strip().lower()
        if token not in self._tokens_dict:
            self._tokens_dict[token] = len(self._tokens_dict) + 1
        else:
            if not bypass:
                raise ValueError("Token already exists in the tokenizer")

    def __str__(self):
        return f"Tokenizer(splitter = {self._splitter}, undefined = {self._undefined}), dict_size = {len(self._tokens_dict)}"

    def __getattribute__(self, item):
        data = object.__getattribute__(self, '_tokens_dict')
        if item in data:
            return data[item]
        return object.__getattribute__(self, item)

    def __len__(self):
        return len(self._tokens_dict)

    def __getitem__(self, item):
        return self._tokens_dict[item]

    def __call__(self, text):
        return self.tokenize(text)

    def __add__(self, other):
        self.add_token(other, bypass=True)
        return self



def tokenize_into_syllables(text):
    """
    Tokenizes input text into syllables using spaCy and spacy-syllables.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of syllables.
    """

    doc = nlp(text)

    syllables = []
    for token in doc:
        token_syllables = token._.syllables
        if token_syllables:
            syllables.extend(token_syllables)
        else:
            syllables.append(token.text)

    return syllables
