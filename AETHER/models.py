import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from matplotlib import pyplot as plt
from torch import optim


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = residual + self.dropout1(attn_output)

        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = residual + self.dropout2(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int = 1, norm: nn.Module = None):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Tremo(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 n_layers,
                 pos_encoder,
                 dim_feedforward=10):
        super(Tremo, self).__init__()
        self.mean_embedder = nn.Embedding(vocab_size, embedding_size)
        self.var_embedder = nn.Embedding(vocab_size, embedding_size)
        self.block = TransformerDecoder(
            TransformerDecoderLayer(embedding_size, 1, dim_feedforward=dim_feedforward),
            num_layers=n_layers)
        self.pos_encoder = pos_encoder

    def forward(self, x: torch.Tensor):
        mean = self.mean_embedder(x)
        var = self.var_embedder(x)
        eps = torch.randn_like(var)
        z = mean + var * eps
        z = self.pos_encoder(z)
        out = self.block(z)
        return out, mean, var


class AutoTrainer:
    """This Class will make the train easier"""
    def __init__(self, model, optimizer='Adam', lr=0.001, loss='NLLLoss'):
        self.loss = loss
        self.model = model
        if isinstance(optimizer, str):
            if optimizer == 'Adam':
                self.optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer == 'AdamW':
                self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer # This will assume model.parameter() is already been passed to\ the optimizer

        if isinstance(loss, str):
            if loss == 'NLLLoss':
                self.loss = nn.NLLLoss()
            elif loss == 'MSE':
                self.loss = nn.MSELoss()
            else:
                raise ValueError('loss should be NLLLoss or MSE')
        else:
            self.loss = loss

        self.epoch_losses = []
        self.losses = []
        self.dataloader = None
        self.step = 0
        self.batch_index = 0
        self.epochs = 0

    def step(self, batch):
        """Step is meant to be called inside of the epoch loop"""
        self.optimizer.zero_grad()

        X = batch[:-1]
        Y = batch[1:]
        pred = self.model(X)

        loss = self.loss(Y, pred)
        self.losses.append(loss.item())

        loss.backward()
        self.optimizer.step()
        self.step += 1
        self.batch_index += 1

    def epoch_step(self):
        self.epoch_losses.append(sum(self.losses) / len(self.losses))
        self.losses = []
        self.batch_index = 0
        self.epochs += 1

    def add_loss(self, loss):
        self.losses.append(loss)

    def add_epoch_loss(self, loss):
        self.epoch_losses.append(loss)

    def plot_epoch_losses(self):
        plt.plot(self.epoch_losses)
        plt.show()




if __name__ == "__main__":
    d_model = 512
    nhead = 8
    dim_feedforward = 2048
    num_layers = 6
    dropout = 0.1

    decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
    final_norm = nn.LayerNorm(d_model)
    transformer_decoder = TransformerDecoder(decoder_layer, num_layers, norm=final_norm)
    tremolino = Tremo(10, 4, 5)


    x = torch.randn(10, 32, d_model)
    xx = torch.randint(0, 10, size=(11, 32))

    tgt_mask = generate_square_subsequent_mask(x.size(0))

    output = transformer_decoder(x, tgt_mask=tgt_mask)

    print("Output shape:", output.shape)  # Expected shape: [10, 32, 512]
    print(tremolino(xx))