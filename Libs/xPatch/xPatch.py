import torch.nn as nn

from Libs.layers.decomp import DECOMP
from Libs.layers.network import Network


class xPatch(nn.Module):
    def __init__(self, configs):
        super(xPatch, self).__init__()

        # Parameters
        seq_len = configs.seq_len  # lookback window L
        pred_len = configs.seq_len  # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in  # input channels
        num_class = configs.num_class
        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha  # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta  # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch, c_in, num_class)

    def forward(self, x):
        x = x.transpose(-1, -2)

        if self.ma_type == 'reg':  # If no decomposition, directly pass the input to the network
            x = self.net(x, x)
            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        return x
