from datasets import load_dataset
from functions import Tokenizer, tokenize_into_syllables
import joblib
import pyphen
import itertools
dic = pyphen.Pyphen(lang='en_EN')

"""Variational Auto Embedder Transformer for Hybrid sEmantic Representation"""
"https://huggingface.co/datasets/glaiveai/reasoning-v1-20m/viewer/default/train"



ds_stream = load_dataset("glaiveai/reasoning-v1-20m", split="train", streaming=True)
ds = list(itertools.islice(ds_stream, 10))
print(ds)
print(len(ds))
ds_train = ds
print(len(ds_train))
print([d['response'] for d in ds])
input('start tokenizer >>>  ')

tok = Tokenizer(undefined=0, splitter=' ')

print("Started Tokenizer")
for i, story in enumerate(ds_train):
    text = story['response']
    for word in tokenize_into_syllables(text):
        tok += word

    if i % 1000 == 0:
        print(i)

    if i == 10_000:
        break

print(tok)

input('save tok >>>   ')
joblib.dump(tok, 'datas/tok1.pkl')