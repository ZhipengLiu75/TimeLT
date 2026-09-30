import numpy as np
from sklearn.cluster import KMeans
from torch import nn
import torch
from torch.nn import Module, Parameter
import math
import torch.nn.functional as F
import warnings


warnings.filterwarnings("ignore")

torch.set_printoptions(sci_mode=False, precision=2)


class GRUmodel(nn.Module):
    def __init__(self, inputsize=10, hiddensize=128, outsize=32):
        super(GRUmodel, self).__init__()
        self.rnn1 = nn.GRU(inputsize, hiddensize, 1, batch_first=True, )
        self.relu1 = nn.ReLU()
        self.rnn2 = nn.GRU(hiddensize, outsize, 1, batch_first=True, )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out, h = self.rnn1(x)
        out = self.relu1(out)
        out, h = self.rnn2(out)
        out = self.relu2(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim

        self.W_Q = nn.Linear(input_dim, self.input_dim)
        self.W_K = nn.Linear(input_dim, self.input_dim)
        self.W_V = nn.Linear(input_dim, hidden_dim)

    def forward(self, X, tau=.5):
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention_weights = F.softmax(attention_scores * tau, dim=-1, )
        output = torch.bmm(attention_weights, V)
        return output


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        seq_len = args.seq_len
        indim = args.indim
        classes = args.classes

        self.T = seq_len
        self.down_sampling_layers = args.down_sampling_layers
        self.temporalchannel = args.channel
        self.temporal_size = args.temporal_size
        self.method1 = False
        self.scale_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=self.temporalchannel, kernel_size=(1, 3), padding=(0, 1))
                for i in range(self.down_sampling_layers + 1)
            ])

        self.temporalmapping = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.temporal_size * indim, self.temporal_size),
                              nn.Linear(self.temporal_size, self.temporal_size), )
                for i in range(self.down_sampling_layers + 1)
            ])

        self.tduplicate = nn.Sequential(nn.Linear(self.temporal_size, self.temporal_size))

        self.normalize_layers = torch.nn.ModuleList(
            [
                nn.BatchNorm2d(self.temporalchannel)
                for i in range(self.down_sampling_layers + 1)
            ])

        self.temporal_normalizes = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(self.temporal_size)
                for i in range(self.down_sampling_layers + 1)
            ])

        self.temporal = GRUmodel(self.temporalchannel, self.temporalchannel, self.temporal_size)


        self.attention = nn.Parameter((torch.ones(1, indim, 1)))

        self.predict = nn.Sequential(
            nn.Linear(self.temporal_size * (self.down_sampling_layers + 1), classes)
        )

    def multi_scale_process(self, x, down_sampling_method='avg', down_sampling_window=2):
        if self.down_sampling_layers == 0:
            return [x], None
        enc_in = 9
        down_pool = None
        if down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(down_sampling_window)
        elif down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=enc_in, out_channels=enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=down_sampling_window,
                                  padding_mode='circular',
                                  bias=False).to('cuda')
        x_enc_ori = x
        temporal_list = []
        spectral_list = []
        temporal_list.append(x.permute(0, 2, 1))

        fft_data = torch.fft.fft(x, dim=-1)
        spectral_list.append(torch.abs(fft_data).permute(0, 2, 1))

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            b, n, t = x_enc_sampling.shape

            temporal_list.append(x_enc_sampling.permute(0, 2, 1))
            fft_data = torch.fft.fft(x_enc_sampling, dim=-1)
            spectral_list.append(torch.abs(fft_data).permute(0, 2, 1))

            x_enc_ori = x_enc_sampling
        return temporal_list, spectral_list

    def sharp_softmax(self, logits, tau=1e-10):
        return F.softmax(logits / tau, dim=-1)

    def get_relation(self, x):
        right = torch.transpose(x, -1, -2)
        A = torch.matmul(x, right).squeeze()
        A = torch.softmax(A, dim=-1)
        return A

    def forward(self, x):
        b, n, t = x.shape
        temporal_list, spectral_list = self.multi_scale_process(x)

        convs_list = []
        for i, x in enumerate(temporal_list):
            x = self.scale_convs[i](torch.unsqueeze(torch.transpose(x, -1, -2), 1))
            x = self.normalize_layers[i](x)
            convs_list.append(x.permute(0, 2, 3, 1).reshape(b * n, -1, self.temporalchannel))

        tem_list = []
        for i, x in enumerate(convs_list):
            z = self.temporal(x)[:, -1].reshape(b, n, -1) * self.attention
            tem_list.append(z)

        temporal_mapping_list = []
        for i, z in enumerate(tem_list):
            z = self.temporalmapping[i](z.reshape(b, -1))
            z = self.temporal_normalizes[i](z)
            temporal_mapping_list.append(z)

        out = torch.stack(temporal_mapping_list, dim=-1)
        embedding = out.reshape(out.shape[0], -1)
        out = self.predict(embedding)
        return out, embedding
