import torch
import torch.nn as nn
from Libs.layers.Embed import Emb


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)

        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Encoder(nn.Module):

    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 2, 1)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)

        return y_2


class PatchMLP(nn.Module):

    def __init__(self, configs):
        super(PatchMLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = 1
        self.output_attention = configs.output_attention

        self.decompsition = series_decomp(13)

        # Embedding
        self.emb = Emb(configs.seq_len, configs.d_model)
        self.seasonal_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for i in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            Encoder(configs.d_model, configs.enc_in)
            for i in range(configs.e_layers)
        ])

        self.projector1 = nn.Linear(configs.d_model, configs.num_class)

        self.projector2 = nn.Linear(configs.num_class * configs.enc_in, configs.num_class)

    def forward(self, x_enc):
        x_enc = x_enc.transpose(-1, -2)
        B = x_enc.shape[0]
        x = x_enc.permute(0, 2, 1)
        x = self.emb(x)
        seasonal_init, trend_init = self.decompsition(x)
        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)
        x = seasonal_init + trend_init
        dec_out = self.projector1(x).squeeze().reshape(B, -1)
        dec_out = self.projector2(dec_out).squeeze()
        return dec_out
