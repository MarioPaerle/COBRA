from datasets import load_dataset
from functions import Tokenizer, tokenize_into_syllables
import joblib
import pyphen

dic = pyphen.Pyphen(lang='en_EN')

"""Variational Auto Embedder Transformer for Hybrid sEmantic Representation"""


ds = load_dataset("roneneldan/TinyStories")

ds_train = ds['train']


tok = Tokenizer(undefined=0, splitter=' ')

print("Started Tokenizer")
for i, story in enumerate(ds_train):
    text = story['text']
    for word in tokenize_into_syllables(text):
        tok += word

    if i % 1000 == 0:
        print(i)

    if i == 10_000:
        break

print(tok)

input('save tok >>>   ')
joblib.dump(tok, 'datas/tok1.pkl')