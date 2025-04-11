import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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
                 dim_feedforward=10,):
        super(Tremo, self).__init__()
        self.mean_embedder = nn.Embedding(vocab_size, embedding_size)
        self.var_embedder = nn.Embedding(vocab_size, embedding_size)
        self.block = TransformerDecoder(
            TransformerDecoderLayer(embedding_size, 1, dim_feedforward=dim_feedforward),
            num_layers=n_layers)

    def forward(self, x: torch.Tensor):
        mean = self.mean_embedder(x)
        var = self.var_embedder(x)
        eps = torch.randn_like(var)
        z = mean + var * eps
        out = self.block(z)
        return out, mean, var


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