from datasets import load_dataset
from functions import Tokenizer, tokenize_into_syllables
import joblib
import pyphen
import itertools
from transformers import AutoTokenizer

dic = pyphen.Pyphen(lang='en_EN')

"""Variational Auto Embedder Transformer for Hybrid sEmantic Representation"""
"https://huggingface.co/datasets/glaiveai/reasoning-v1-20m/viewer/default/train"


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ds_stream = load_dataset("open-thoughts/OpenThoughts2-1M", streaming=True, split="train")
dataset = ds_stream.map(lambda x: tokenizer(" ".join([c['value'] for c in x['conversations']]), return_tensors="pt"))
texts = []
for i, piece in enumerate(ds_stream):
    text = " ".join([c['value'] for c in piece['conversations']])[:512]
    texts.append(tokenizer(text))
    if i % 100 == 0:
        print(i)
    if i == 1000:
        break

input(' save >>>   ')
joblib.dump(texts, 'datas/OT_ex1.pkl')