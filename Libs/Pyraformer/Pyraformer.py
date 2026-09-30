import torch
import torch.nn as nn
from Libs.layers.Pyraformer_EncDec import Encoder


class Pyraformer(nn.Module):

    def __init__(self, configs, window_size=[4, 4], inner_size=5):

        super().__init__()
        self.d_model = configs.d_model
        self.encoder = Encoder(configs, window_size, inner_size)

        self.act = torch.nn.functional.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            (len(window_size) + 1) * self.d_model * configs.seq_len, configs.num_class)


    def forward(self, x_enc):
        x_enc = x_enc.transpose(-1, -2)
        enc_out = self.encoder(x_enc, x_mark_enc=None)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)

        return output
