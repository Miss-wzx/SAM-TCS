# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -------------------------
# 1D 位置编码（Positional Encoding）
# Transformer 需要位置信息，这里用正弦余弦编码
# -------------------------
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=16384):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为buffer，不参与训练
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1), :].to(x.device)


# -------------------------
# 残差卷积块（Conv1D）
# 包含两层Conv1d + BN + GELU + 残差连接
# -------------------------
class ResConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        # 如果通道或步长不匹配，增加下采样层保证残差尺寸一致
        if in_ch != out_ch or stride != 1:
            self.down = nn.Conv1d(in_ch, out_ch, 1, stride=stride)
        else:
            self.down = None

    def forward(self, x):
        identity = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.act(out)
        return out


# -------------------------
# Transformer Block（带残差）
# 输入：(B, T, D)，输出：(B, T, D)
# -------------------------
class TransformerBlock1D(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, T, D)
        x2 = x.permute(1, 0, 2)  # (T, B, D) 符合MultiheadAttention要求
        attn_out, _ = self.self_attn(x2, x2, x2, need_weights=True)
        attn_out = attn_out.permute(1, 0, 2)  # (B, T, D)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# -------------------------
# 测量层（可训练测量矩阵）
# 输入：原始信号 (B, N)
# 输出：压缩后信号 (B, M)
# -------------------------
class MeasurementLayer(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.N = N
        self.M = M
        self.linear = nn.Linear(N, M, bias=False)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))  # 初始化权重

    def forward(self, x):
        # x: (B, N)
        return self.linear(x)  # (B, M)


# -------------------------
# 初始重构层（伪逆映射）
# 输入：测量结果 (B, M)
# 输出：初步重构 (B, 1, N)
# -------------------------
class InitialReconstructor(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.linear = nn.Linear(M, N, bias=True)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, y):
        x_init = self.linear(y)  # (B, N)
        return x_init.unsqueeze(1)  # (B, 1, N)


# -------------------------
# 完整模型：CS-Transformer-1D
# 包含测量、初始重构、卷积编码器、Transformer、解码器
# -------------------------
class CSTransformer1D(nn.Module):
    def __init__(self, N=4096, M=1024, d_model=256, conv_channels=[32, 128, 512], trans_layers=2, nhead=8):
        super().__init__()
        self.N = N
        self.M = M
        # 压缩测量 + 初始重构
        self.measure = MeasurementLayer(N, M)
        self.init_recon = InitialReconstructor(N, M)

        # 编码器：多层卷积+池化
        self.enc_convs = nn.ModuleList()
        in_ch = 1
        for ch in conv_channels:
            self.enc_convs.append(ResConv1D(in_ch, ch, kernel=9))
            self.enc_convs.append(nn.MaxPool1d(2))  # 下采样2倍
            in_ch = ch

        # Conv投影到Transformer维度
        self.project_in = nn.Conv1d(conv_channels[-1], d_model, kernel_size=1)
        self.pos_enc = PositionalEncoding1D(d_model, max_len=N // (2 ** len(conv_channels)) + 10)

        # Transformer堆叠
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock1D(d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=0.1)
            for _ in range(trans_layers)
        ])

        # 投影回卷积通道
        self.project_out = nn.Conv1d(d_model, conv_channels[-1], kernel_size=1)

        # 解码器：反卷积/上采样
        self.dec_convs = nn.ModuleList()
        rev_channels = conv_channels[::-1]
        for i in range(len(rev_channels) - 1):
            self.dec_convs.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
            self.dec_convs.append(ResConv1D(rev_channels[i], rev_channels[i + 1], kernel=9))

        # 输出层，恢复到单通道
        self.final_conv = nn.Conv1d(rev_channels[-1], 1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(248)
        self.fc_out = nn.Linear(248, N)

    def forward(self, x):
        # 输入：原始信号 (B, 1, N)
        B = x.shape[0]

        x_flat = x.view(B, self.N)  # (B, N)
        y = self.measure(x_flat)  # (B, M)
        x_init = self.init_recon(y)  # (B, 1, N)

        # 编码器
        out = x_init
        for module in self.enc_convs:
            out = module(out)

        # 投影到Transformer
        out = self.project_in(out)  # (B, d_model, T)
        out = out.permute(0, 2, 1)  # (B, T, d_model)
        out = self.pos_enc(out)

        # Transformer精炼
        for blk in self.transformer_blocks:
            out = blk(out)

        # 投影回卷积通道
        out = out.permute(0, 2, 1)  # (B, d_model, T)
        out = self.project_out(out)  # (B, C_last, T)

        # 解码器上采样
        for module in self.dec_convs:
            out = module(out)

        out = self.final_conv(out)  # (B, 1, N)
        out = self.pool(out)  # [B, 1, 248]
        out = out.squeeze(1)  # [B, 248]
        out = self.fc_out(out)  # (B, 500)
        # x_hat = x_init + out  # 两者现在对齐
        # 残差连接：预测的增量 + 初始重构
        out = out.unsqueeze(1)
        x_hat = x_init + out

        # x_hat = x_hat.squeeze(1)

        return x_hat, y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(torch.cuda.is_available())

    model = CSTransformer1D(N=3072, M=128).to(device)

    x = torch.randn(2, 1, 3072).to(device)

    with torch.no_grad():
        x_hat, _ = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", x_hat.shape)
    print("latent_dim shape:", _.shape)
