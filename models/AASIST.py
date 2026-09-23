"""
AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
Official architecture implementation compatible with ClovaAI AASIST.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_via_conv1d(conv2d_layer: nn.Conv2d, input_tensor: torch.Tensor) -> torch.Tensor:
    """Computes Conv2d via slice-wise Conv1d to prevent multi-gigabyte im2col memory allocation on large sequence widths."""
    B, C_in, H, W = input_tensor.shape
    C_out = conv2d_layer.out_channels
    kh, kw = conv2d_layer.kernel_size
    ph, pw = conv2d_layer.padding
    sh, sw = conv2d_layer.stride

    if ph > 0 or pw > 0:
        input_tensor = F.pad(input_tensor, (pw, pw, ph, ph))

    padded_H = input_tensor.shape[2]
    weight = conv2d_layer.weight
    bias = conv2d_layer.bias

    out_H = (padded_H - kh) // sh + 1
    row_outputs = []
    for h in range(out_H):
        h_start = h * sh
        row_out = None
        for k in range(kh):
            x_slice = input_tensor[:, :, h_start + k, :]
            w_slice = weight[:, :, k, :]
            c1d = F.conv1d(x_slice, w_slice, stride=sw)
            if row_out is None:
                row_out = c1d
            else:
                row_out = row_out + c1d
        if bias is not None:
            row_out = row_out + bias.view(1, C_out, 1)
        row_outputs.append(row_out.unsqueeze(2))

    return torch.cat(row_outputs, dim=2)


class SincConv_fast(nn.Module):
    """Sinc-layer convolution for raw audio waveforms."""
    @staticmethod
    def to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports in_channels=1, got {in_channels}")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)

        mel = torch.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(hz[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter(torch.diff(hz).unsqueeze(1))

        n_ = torch.linspace(1, (self.kernel_size - 1) / 2, steps=int((self.kernel_size - 1) / 2))
        self.register_buffer("n_", n_.unsqueeze(0))

        window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(0, self.kernel_size) / self.kernel_size)
        self.register_buffer("window_", window_)

    def forward(self, waveforms):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0].unsqueeze(1)

        f_times_t_low = torch.matmul(low, self.n_) / self.sample_rate
        f_times_t_high = torch.matmul(high, self.n_) / self.sample_rate

        band_pass_left = ((torch.sin(2 * math.pi * f_times_t_high) - torch.sin(2 * math.pi * f_times_t_low)) / (2 * math.pi * self.n_)) * 2
        band_pass_center = 2 * band
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band)

        filters = band_pass * self.window_
        filters = filters.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=False)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1], out_channels=nb_filts[1], kernel_size=(2, 3), padding=(0, 1))

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1], padding=(0, 1), kernel_size=(1, 3))
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = conv2d_via_conv1d(self.conv1, out)
        out = self.bn2(out)
        out = self.selu(out)
        out = conv2d_via_conv1d(self.conv2, out)

        if self.downsample:
            identity = conv2d_via_conv1d(self.conv_downsample, identity)

        out = out + identity
        out = self.mp(out)
        return out


class GraphAttentionSquare(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.Tensor(out_dim, 1))
        nn.init.xavier_uniform_(self.att_weight)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SELU(inplace=False)

    def forward(self, x):
        att_h = torch.tanh(self.att_proj(x))
        att = torch.matmul(att_h, self.att_weight)
        att = F.softmax(att, dim=1)

        x_att = x * att
        h_with_att = self.proj_with_att(x_att)
        h_without_att = self.proj_without_att(x)

        h = h_with_att + h_without_att
        h = h.transpose(1, 2).contiguous()
        h = self.bn(h)
        h = h.transpose(1, 2).contiguous()
        h = self.act(h)
        return h


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.Tensor(out_dim, 1))
        nn.init.xavier_uniform_(self.att_weight11)
        nn.init.xavier_uniform_(self.att_weight22)
        nn.init.xavier_uniform_(self.att_weight12)
        nn.init.xavier_uniform_(self.att_weightM)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SELU(inplace=False)

    def forward(self, x1, x2, master):
        h1 = self.proj_type1(x1)
        h2 = self.proj_type2(x2)

        att_h1 = torch.tanh(self.att_proj(h1))
        att11 = torch.matmul(att_h1, self.att_weight11)
        att11 = F.softmax(att11, dim=1)
        h1_att = h1 * att11

        att_h2 = torch.tanh(self.att_proj(h2))
        att22 = torch.matmul(att_h2, self.att_weight22)
        att22 = F.softmax(att22, dim=1)
        h2_att = h2 * att22

        h1_out = self.proj_with_att(h1_att) + self.proj_without_att(h1)
        h2_out = self.proj_with_att(h2_att) + self.proj_without_att(h2)

        h_cat = torch.cat([h1_out, h2_out], dim=1)
        h_cat = h_cat.transpose(1, 2).contiguous()
        h_cat = self.bn(h_cat)
        h_cat = h_cat.transpose(1, 2).contiguous()
        h_cat = self.act(h_cat)

        out1 = h_cat[:, :x1.shape[1], :]
        out2 = h_cat[:, x1.shape[1]:, :]

        att_hm = torch.tanh(self.att_projM(master))
        att_m = torch.matmul(att_hm, self.att_weightM)
        att_m = F.softmax(att_m, dim=1)
        master_att = master * att_m

        master_out = self.proj_with_attM(master_att) + self.proj_without_attM(master)
        master_out = master_out.transpose(1, 2).contiguous()
        master_out = self.bn(master_out)
        master_out = master_out.transpose(1, 2).contiguous()
        master_out = self.act(master_out)

        return out1, out2, master_out


class GraphPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x):
        scores = self.proj(x)
        scores = F.softmax(scores, dim=1)
        pooled = torch.sum(x * scores, dim=1)
        return pooled


class Model(nn.Module):
    def __init__(self, d_args: dict):
        super().__init__()

        self.d_args = d_args
        filt_0 = d_args["filts"][0]
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=False)

        self.SincConv = SincConv_fast(
            out_channels=filt_0,
            kernel_size=d_args.get("first_conv", 128),
            in_channels=1
        )

        self.encoder = nn.ModuleList([
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][1], first=True)),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][2])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][3])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][4])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][4])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][4]))
        ])

        gat_dims = d_args.get("gat_dims", [64, 32])
        self.GAT_layer_S = GraphAttentionSquare(in_dim=64, out_dim=gat_dims[0])
        self.GAT_layer_T = GraphAttentionSquare(in_dim=64, out_dim=gat_dims[0])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(in_dim=gat_dims[0], out_dim=gat_dims[1])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(in_dim=gat_dims[1], out_dim=gat_dims[1])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(in_dim=gat_dims[0], out_dim=gat_dims[1])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(in_dim=gat_dims[1], out_dim=gat_dims[1])

        self.pool_S = GraphPool(in_dim=gat_dims[0])
        self.pool_T = GraphPool(in_dim=gat_dims[0])

        self.pool_hS1 = GraphPool(in_dim=gat_dims[1])
        self.pool_hT1 = GraphPool(in_dim=gat_dims[1])
        self.pool_hS2 = GraphPool(in_dim=gat_dims[1])
        self.pool_hT2 = GraphPool(in_dim=gat_dims[1])

        self.pos_S = nn.Parameter(torch.randn(1, 23, gat_dims[0]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        out_dim = 160
        self.out_layer = nn.Linear(out_dim, 2)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.SincConv(x)
        x = x.unsqueeze(1).contiguous()
        x = self.first_bn(x)
        x = self.selu(x)

        for layer in self.encoder:
            x = layer(x)

        x_S = x.max(dim=-1)[0].permute(0, 2, 1).contiguous()
        x_T = x.max(dim=2)[0].permute(0, 2, 1).contiguous()

        if x_S.shape[1] != 23:
            x_S = F.adaptive_avg_pool1d(x_S.permute(0, 2, 1), 23).permute(0, 2, 1).contiguous()

        h_S = self.GAT_layer_S(x_S + self.pos_S)
        h_T = self.GAT_layer_T(x_T)

        out_S = self.pool_S(h_S)
        out_T = self.pool_T(h_T)

        h_S1, h_T1, m1 = self.HtrgGAT_layer_ST11(h_S, h_T, self.master1)
        h_S1, h_T1, m1 = self.HtrgGAT_layer_ST12(h_S1, h_T1, m1)
        out_hS1 = self.pool_hS1(h_S1)
        out_hT1 = self.pool_hT1(h_T1)

        h_S2, h_T2, m2 = self.HtrgGAT_layer_ST21(h_S, h_T, self.master2)
        h_S2, h_T2, m2 = self.HtrgGAT_layer_ST22(h_S2, h_T2, m2)
        out_hS2 = self.pool_hS2(h_S2)
        out_hT2 = self.pool_hT2(h_T2)

        last_hidden = torch.cat([out_hS1, out_hT1, out_hS2, out_hT2, m1.squeeze(1)], dim=1)
        logits = self.out_layer(last_hidden)

        return last_hidden, logits
