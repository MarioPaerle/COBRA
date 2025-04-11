"""Train Example implementation"""
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import Tremo

vocab_size = 30 # Il numero di singoli token
context = 10    # Grandezza del contesto
n_texts = 64    # Il numero di testi presenti nel dataset
epochs = 1000
batch_size = 64
tokenized = torch.randint(0, vocab_size, (n_texts, context)) # questo è solo simulato in realtà servirà caricare testo
                                                             # per testo e tokenizzarlo

flattened = tokenized.flatten(start_dim=0)
print(flattened.shape)
# NO RAM FRIENDLY
# x = flattened[:-1]
# y = flattened[1:]

# FAREMO POI COSì SPRECANDO LA METà DI RAM
# x = flattened[i-context:i]
# y = flattened[i-context+1:i+1]
# questo for i da 1 a n-1

dataloader = DataLoader(flattened, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for batch in dataloader:
        print(batch.shape)
        batch = batch.unfoo


