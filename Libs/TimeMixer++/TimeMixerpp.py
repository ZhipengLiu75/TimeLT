
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.fft

from Libs.layers.AxialAttention import RowAttention, ColAttention
from Libs.layers.Embed import DataEmbedding_wo_pos
from Libs.layers.Conv_Blocks import Inception_Block_V1, Inception_Trans_Block_V1
from Libs.layers.SelfAttention_Family import FullAttention, AttentionLayer
from Libs.layers.StandardNorm import Normalize


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    if len(frequency_list) < k:
        k = len(frequency_list)
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    index = np.where(period > 0)
    top_list = top_list[index]
    period = period[period > 0]
    return period, abs(xf).mean(-1)[:, top_list], top_list


class MultiScaleSeasonCross(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSeasonCross, self).__init__()
        self.cross_conv_season = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels, stride=(configs.down_sampling_window, 1)),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels))

    def forward(self, season_list):
        B, N, _, _ = season_list[0].size()


        # cross high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = []
        out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))
        for i in range(len(season_list) - 1):
            out_low_res = self.cross_conv_season(out_high)
            out_low = out_low + out_low_res#[:,:,:out_low.shape[2]]
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 3, 1).reshape(B, -1, N))


        return out_season_list


class MultiScaleTrendCross(nn.Module):
    def __init__(self, configs):
        super(MultiScaleTrendCross, self).__init__()

        self.cross_trans_conv_season = Inception_Trans_Block_V1(configs.d_model, configs.d_ff,
                                                                num_kernels=configs.num_kernels,
                                                                stride=(configs.down_sampling_window, 1))
        self.cross_trans_conv_season_restore = nn.Sequential(
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, trend_list):
        B, N, _, _ = trend_list[0].size()
        # cross low->high
        trend_list.reverse()
        out_low = trend_list[0]
        out_high = trend_list[1]
        out_trend_list = []
        out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        for i in range(len(trend_list) - 1):
            out_high_res = self.cross_trans_conv_season(out_low, output_size=out_high.size())
            out_high_res = self.cross_trans_conv_season_restore(out_high_res)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list) - 1:
                out_high = trend_list[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 3, 1).reshape(B, -1, N))

        out_trend_list.reverse()
        return out_trend_list


class MixerBlock(nn.Module):
    def __init__(self, configs):
        super(MixerBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.down_sampling_window = configs.down_sampling_window
        self.k = configs.top_k
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.row_attn_2d_trend = RowAttention(configs.d_model, configs.d_ff)
        self.col_attn_2d_season = ColAttention(configs.d_model, configs.d_ff)
        self.multi_scale_season_conv = MultiScaleSeasonCross(configs)
        self.multi_scale_trend_conv = MultiScaleTrendCross(configs)

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)
        period_list, period_weight, top_list = FFT_for_Period(x_list[-1], self.k)


        res_list = []
        for i in range(len(period_list)):
            period = period_list[i]
            season_list = []
            trend_list = []


            for x in x_list:
                out = self.time_imaging(x, period)

                season, trend = self.dual_axis_attn(out)
                season_list.append(season)
                trend_list.append(trend)



            out_list = self.multi_scale_mixing(season_list, trend_list, length_list)
            res_list.append(out_list)

        res_list_new = []
        for i in range(len(x_list)):
            list = []
            for j in range(len(period_list)):
                list.append(res_list[j][i])
            res = torch.stack(list, dim=-1)
            res_list_new.append(res)

        res_list_agg = []
        for x, res in zip(x_list, res_list_new):
            res = self.multi_reso_mixing(period_weight, x, res)
            res = self.layer_norm(res)
            res_list_agg.append(res)
        return res_list_agg

    def time_imaging(self, x, period):
        B, T, N = x.size()
        out, length = self.__conv_padding(x, period)

        out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

        return out

    def dual_axis_attn(self, out):
        trend = self.row_attn_2d_trend(out)
        season = self.col_attn_2d_season(out)
        return season, trend

    def multi_scale_mixing(self, season_list, trend_list, length_list):

        out_season_list = self.multi_scale_season_conv(season_list)
        out_trend_list = self.multi_scale_trend_conv(trend_list)
        out_list = []
        for out_season, out_trend, length in zip(out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out_list.append(out[:, :length, :])
        return out_list

    def __conv_padding(self, x, period, down_sampling_window=1):
        B, T, N = x.size()

        # if T % (period * down_sampling_window) != 0:

        length = ((T // (period * down_sampling_window)) + 1) * period * down_sampling_window
        padding = torch.zeros([B, (length - T), N]).to(x.device)
        out = torch.cat([x, padding], dim=1)

        # else:
        #     length = T
        #     out = x
        return out, length

    def multi_reso_mixing(self, period_weight, x, res):
        B, T, N = x.size()
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimeMixerpp(nn.Module):
    def __init__(self, configs):
        super(TimeMixerpp, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.encoder_model = nn.ModuleList([MixerBlock(configs)
                                            for _ in range(configs.e_layers)])

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model,configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.dropout)

        self.revin_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in,
                          affine=True, subtract_last=False
                          )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.layer = configs.e_layers

        if self.configs.channel_mixing:
            d_time_model = configs.seq_len // (configs.down_sampling_window ** configs.down_sampling_layers)
            self.channel_mixing_attention = AttentionLayer(FullAttention(False, attention_dropout=self.configs.dropout,
                                                                         output_attention=self.configs.output_attention),
                                                           d_time_model, self.configs.n_heads)

        if self.configs.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            if self.channel_independence:
                in_channels = 1
            else:
                in_channels = self.configs.enc_in
            self.down_pool = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=3, padding=padding,
                                       stride=self.configs.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)
        else:
            raise ValueError('Downsampling method is error,only supporting the max, avg, conv1D')


        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
            configs.d_model * configs.seq_len, configs.num_class, bias=True)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # B,T,C -> B,C,T
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            if self.configs.down_sampling_method == 'conv' and i == 0 and self.channel_independence:
                x_enc_ori = x_enc_ori.contiguous().reshape(B * N, T, 1).permute(0, 2, 1).contiguous()
            x_enc_sampling = self.down_pool(x_enc_ori)

            if self.configs.down_sampling_method == 'conv':
                x_enc_sampling_list.append(
                    x_enc_sampling.reshape(B, N, T // (self.down_sampling_window ** (i + 1))).permute(0, 2, 1))
            else:
                x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))

            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc



    def forward(self, x_enc):
        x_enc=x_enc.transpose(-1,-2)
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc
        enc_out_list = []

        for x in x_list:
            enc_out = self.enc_embedding(x)  # [B,T,C]
            enc_out_list.append(enc_out)



        for i in range(self.layer):
            enc_out_list = self.encoder_model[i](enc_out_list)


        enc_out = enc_out_list[0]
        output = self.act(enc_out)
        output = self.dropout(output)

        output = output.reshape(output.shape[0], -1)

        output = self.projection(output)  # (batch_size, num_classes)
        return output




