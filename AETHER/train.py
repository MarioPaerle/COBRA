import joblib
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from models import Tremo
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
import torch.nn.functional as F



class TokenizedDataset(Dataset):
    def __init__(self, data):
        """
        data: lista di sequenze tokenizzate (es. [[1, 5, 6, 2], [4, 7, 8], ...])
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Riceve un batch di sequenze e le padda al massimo di lunghezza presente nel batch.
    Viene usato il valore 0 come pad-token.
    """
    batch = [torch.tensor(seq, dtype=torch.long) for seq in batch]
    max_length = max(seq.size(0) for seq in batch)
    padded_batch = torch.stack([
        torch.cat([seq, torch.zeros(max_length - seq.size(0), dtype=torch.long)])
        for seq in batch
    ])
    return padded_batch


class BucketBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        """
        data_source: Dataset contenente le sequenze tokenizzate
        batch_size: Numero di sequenze per batch
        drop_last: Se True, scarta l'ultimo batch se non raggiunge batch_size
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Crea la lista degli indici ordinati per lunghezza della sequenza
        self.sorted_indices = sorted(range(len(data_source)), key=lambda idx: len(data_source[idx]))

    def __iter__(self):
        # Dividi gli indici ordinati in batch di dimensione batch_size
        batches = [self.sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(self.sorted_indices), self.batch_size)]

        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        # Se desiderato, randomizza l'ordine dei batch (senza modificare l'ordine interno)
        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

tokenized_texts = [t['input_ids'] for t in joblib.load('datas/OT_ex1.pkl')]

dataset = TokenizedDataset(tokenized_texts)


batch_size = 10
bucket_sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=False)
emb_size = 128

dataloader = DataLoader(dataset, batch_sampler=bucket_sampler, collate_fn=collate_fn)
model = Tremo(vocab_size=30_522, n_layers=1, dim_feedforward=10, embedding_size=emb_size, pos_encoder=Summer(PositionalEncoding1D(emb_size)))
epochs = 1
criterion = nn.MSELoss()
optim = Adam(model.parameters())
mean_losses = []

for epoch in range(epochs):
    losses = []
    for i, batch in enumerate(dataloader):
        optim.zero_grad()
        x = batch[:, :-1]
        y = batch[:, 1:]
        with torch.no_grad():
            y_onehot = model.mean_embedder(y)

        y_pred, mean, var = model(x)

        loss = criterion(y_pred, y_onehot)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if i % 1 == 0:
            print(f"Step {i}: {loss.item()}")
    print(f"Epoch {i}: {np.mean(losses)}")

