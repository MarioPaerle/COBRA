from datasets import load_dataset
from functions import Tokenizer

"""Variational Auto Embedder Transformer for Hybrid sEmantic Representation"""


ds = load_dataset("roneneldan/TinyStories")

ds_train = ds['train']

sample = ds_train[0]['text']
print(len(sample))

tok = Tokenizer()

for word in sample.split():
    tok += word


print(tok)
print(tok.tokens_dict)