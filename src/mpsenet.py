"""MP-SENet (Lu et al., Interspeech 2023 / long version 2025) 생성기.

원본: https://github.com/yxlu-0102/MP-SENet (MIT License, Copyright (c) 2023 Ye-Xin Lu)
models/model.py, models/transformer.py, utils.py, dataset.py의 추론에 필요한 부분만 옮겼다.
state_dict 키는 원본과 같아서 공개 가중치(best_ckpt/g_best_vb, g_best_dns)를 그대로 불러올 수 있다.

legacy_attention=True(기본)는 원본의 batch_first 버그를 그대로 재현한다. 공개 가중치는 이 동작으로
학습됐으므로 그 가중치를 쓸 때는 True로 두고, 파일을 한 개씩(batch 1) 처리해야 학습 때와 같다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class FFN(nn.Module):
    def __init__(self, d_model, batch_first):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model * 2, 1, bidirectional=True, batch_first=batch_first)
        self.linear = nn.Linear(d_model * 2 * 2, d_model)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        return self.linear(F.leaky_relu(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, batch_first=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=batch_first)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, batch_first)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt, need_weights=False)
        x = x + xt
        x = x + self.ffn(self.norm2(x))
        return self.norm3(x)


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(self.pad1(x))
        b, c, h, w = out.shape
        out = out.view((b, self.r, c // self.r, h, w)).permute(0, 2, 3, 4, 1)
        return out.contiguous().view((b, c // self.r, h, -1))


class DenseBlock(nn.Module):
    def __init__(self, ch, kernel_size=(2, 3), depth=4):
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d((1, 1, 2 ** i, 0), value=0.),
                nn.Conv2d(ch * (i + 1), ch, kernel_size, dilation=(2 ** i, 1)),
                nn.InstanceNorm2d(ch, affine=True),
                nn.PReLU(ch))
            for i in range(depth)])

    def forward(self, x):
        skip = x
        for layer in self.dense_block:
            x = layer(skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, ch, in_channel):
        super().__init__()
        self.dense_conv_1 = nn.Sequential(nn.Conv2d(in_channel, ch, (1, 1)),
                                          nn.InstanceNorm2d(ch, affine=True), nn.PReLU(ch))
        self.dense_block = DenseBlock(ch, depth=4)
        self.dense_conv_2 = nn.Sequential(nn.Conv2d(ch, ch, (1, 3), (1, 2), padding=(0, 1)),
                                          nn.InstanceNorm2d(ch, affine=True), nn.PReLU(ch))

    def forward(self, x):
        return self.dense_conv_2(self.dense_block(self.dense_conv_1(x)))


class MaskDecoder(nn.Module):
    def __init__(self, ch, n_fft, beta):
        super().__init__()
        self.dense_block = DenseBlock(ch, depth=4)
        self.mask_conv = nn.Sequential(SPConvTranspose2d(ch, ch, (1, 3), 2),
                                       nn.InstanceNorm2d(ch, affine=True), nn.PReLU(ch),
                                       nn.Conv2d(ch, 1, (1, 2)))
        self.lsigmoid = LearnableSigmoid2d(n_fft // 2 + 1, beta=beta)

    def forward(self, x):
        x = self.mask_conv(self.dense_block(x))
        return self.lsigmoid(x.permute(0, 3, 2, 1).squeeze(-1))  # [B, F, T]


class PhaseDecoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dense_block = DenseBlock(ch, depth=4)
        self.phase_conv = nn.Sequential(SPConvTranspose2d(ch, ch, (1, 3), 2),
                                        nn.InstanceNorm2d(ch, affine=True), nn.PReLU(ch))
        self.phase_conv_r = nn.Conv2d(ch, 1, (1, 2))
        self.phase_conv_i = nn.Conv2d(ch, 1, (1, 2))

    def forward(self, x):
        x = self.phase_conv(self.dense_block(x))
        x = torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))
        return x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]


class TSTransformerBlock(nn.Module):
    def __init__(self, ch, batch_first):
        super().__init__()
        self.time_transformer = TransformerBlock(ch, 4, batch_first)
        self.freq_transformer = TransformerBlock(ch, 4, batch_first)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_transformer(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


class MPNet(nn.Module):
    """원본 MPNet과 같은 구조·키. 입력·출력은 압축된 크기와 위상 [B, F, T]."""

    def __init__(self, dense_channel=64, n_fft=400, beta=2.0, num_tsblocks=4, legacy_attention=True):
        super().__init__()
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.TSTransformer = nn.ModuleList(
            [TSTransformerBlock(dense_channel, batch_first=not legacy_attention) for _ in range(num_tsblocks)])
        self.mask_decoder = MaskDecoder(dense_channel, n_fft, beta)
        self.phase_decoder = PhaseDecoder(dense_channel)

    def forward(self, noisy_amp, noisy_pha):
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        x = self.dense_encoder(x)
        for block in self.TSTransformer:
            x = block(x)
        amp = noisy_amp * self.mask_decoder(x)
        pha = self.phase_decoder(x)
        com = torch.stack((amp * torch.cos(pha), amp * torch.sin(pha)), dim=-1)
        return amp, pha, com


class MPSENet(nn.Module):
    """파형 입출력 래퍼: RMS 정규화 → STFT(400/100, 0.3 압축) → MPNet → iSTFT → 원래 음량."""

    def __init__(self, n_fft=400, hop=100, win=400, compress_factor=0.3, dense_channel=64, beta=2.0,
                 num_tsblocks=4, legacy_attention=True):
        super().__init__()
        self.n_fft, self.hop, self.win, self.compress = n_fft, hop, win, compress_factor
        self.generator = MPNet(dense_channel, n_fft, beta, num_tsblocks, legacy_attention)
        self.register_buffer("window", torch.hann_window(win))

    def mag_pha(self, wave):
        spec = torch.view_as_real(torch.stft(wave, self.n_fft, self.hop, self.win, window=self.window,
                                             center=True, pad_mode="reflect", return_complex=True))
        mag = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        pha = torch.atan2(spec[..., 1] + 1e-10, spec[..., 0] + 1e-5)
        return mag.pow(self.compress), pha

    def to_wave(self, mag, pha, length):
        mag = mag.pow(1.0 / self.compress)
        com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        return torch.istft(com, self.n_fft, self.hop, self.win, window=self.window, center=True, length=length)

    def enhance(self, noisy):
        """noisy: [B, N]. legacy_attention 가중치는 B=1로 호출해야 학습 때와 같다."""
        norm = torch.sqrt(noisy.shape[-1] / noisy.pow(2).sum(-1, keepdim=True).clamp_min(1e-12))
        mag, pha = self.mag_pha(noisy * norm)
        amp_g, pha_g, _ = self.generator(mag, pha)
        return self.to_wave(amp_g, pha_g, noisy.shape[-1]) / norm

    def load_original_checkpoint(self, path, map_location="cpu"):
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.generator.load_state_dict(state["generator"] if "generator" in state else state)
        return self
