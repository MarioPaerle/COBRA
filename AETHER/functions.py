import numpy as np

class Tokenizer:
    """A Simple Tokenizer :)"""
    def __init__(self, splitter=' ', undefined=0):
        self.undefined = undefined
        self.splitter = splitter
        self.tokens_dict = dict()

    def tokenize(self, text):
        tokens = text.split(self.splitter)
        return [self.tokens_dict.get(t, self.undefined) for t in tokens]

    def add_token(self, token, bypass=False):
        token = token.strip().lower()
        if token not in self.tokens_dict:
            self.tokens_dict[token] = len(self.tokens_dict) + 1
        else:
            if not bypass:
                raise ValueError("Token already exists in the tokenizer")

    def __str__(self):
        return f"Tokenizer(splitter = {self.splitter}, undefined = {self.undefined}), dict_size = {len(self.tokens_dict)}"

    def __getattribute__(self, item):
        data = object.__getattribute__(self, 'tokens_dict')
        if item in data:
            return data[item]
        return object.__getattribute__(self, item)

    def __len__(self):
        return len(self.tokens_dict)

    def __getitem__(self, item):
        return self.tokens_dict[item]

    def __call__(self, text):
        return self.tokenize(text)

    def __add__(self, other):
        self.add_token(other, bypass=True)
        return self

