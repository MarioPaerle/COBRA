import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np


# 1. Creazione del Dataset: lista di sequenze tokenizzate (liste di interi)
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


# 2. Funzione di collate per il padding dinamico
def collate_fn(batch):
    """
    Riceve un batch di sequenze e le padda al massimo di lunghezza presente nel batch.
    Viene usato il valore 0 come pad-token.
    """
    # Converti ogni sequenza in un tensore
    batch = [torch.tensor(seq, dtype=torch.long) for seq in batch]
    # Calcola la lunghezza massima del batch
    max_length = max(seq.size(0) for seq in batch)
    # Applica il padding ad ogni sequenza e crea un tensore batch
    padded_batch = torch.stack([
        torch.cat([seq, torch.zeros(max_length - seq.size(0), dtype=torch.long)])
        for seq in batch
    ])
    return padded_batch


# 3. Implementazione del BucketBatchSampler
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


# 4. Dati di esempio: lista di testi tokenizzati di lunghezza variabile
tokenized_texts = [
    [1, 5, 6, 2],
    [4, 7, 8],
    [9, 3, 2, 6, 7],
    [3, 2],
    [8, 4, 5, 7, 2, 9],
    [6, 3, 8, 1],
    # puoi aggiungere ulteriori sequenze
]

# Creazione del dataset
dataset = TokenizedDataset(tokenized_texts)

# Creazione del BucketBatchSampler
batch_size = 2  # imposta il batch size secondo necessità
bucket_sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=False)

# Creazione del DataLoader con la funzione collate custom
dataloader = DataLoader(dataset, batch_sampler=bucket_sampler, collate_fn=collate_fn)

# 5. Ciclo di esempio sul DataLoader
for batch in dataloader:
    print("Batch shape:", batch.shape)
    print(batch)
