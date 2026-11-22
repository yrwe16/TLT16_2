"""PCS(Perceptual Contrast Stretching) 후처리.

Chao et al., "Perceptual Contrast Stretching on Target Feature for Speech Enhancement" (Interspeech 2022)의
방식: 출력 파형의 STFT(512/256) 크기에 log1p를 취하고 주파수 대역별 가중치(1.0~1.4)를 곱한 뒤
expm1로 되돌린다. 위상은 그대로 둔다. alpha로 강도를 조절한다(0이면 원본, 1이면 논문 가중치).
학습이 필요 없는 결정적 후처리다.
"""
import numpy as np
import torch

N_FFT, HOP = 512, 256


def pcs_weights(alpha=1.0):
    w = np.ones(N_FFT // 2 + 1, dtype=np.float32)
    w[0:3] = 1.0
    w[3:6] = 1.070175439
    w[6:9] = 1.182456140
    w[9:12] = 1.287719298
    w[12:138] = 1.4
    w[138:166] = 1.322807018
    w[166:200] = 1.238596491
    w[200:241] = 1.161403509
    w[241:256] = 1.077192982
    w[256] = 1.0
    return 1.0 + alpha * (w - 1.0)


def apply_pcs(wave, alpha=1.0):
    """wave: [T] 또는 [B, T] float tensor/ndarray. 같은 길이로 돌려준다."""
    as_numpy = isinstance(wave, np.ndarray)
    x = torch.as_tensor(wave, dtype=torch.float32)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[None]
    window = torch.hann_window(N_FFT, device=x.device)
    spec = torch.stft(x, N_FFT, HOP, window=window, return_complex=True)
    mag, pha = spec.abs(), spec.angle()
    w = torch.from_numpy(pcs_weights(alpha)).to(x.device)[None, :, None]
    mag = torch.expm1(torch.log1p(mag) * w)
    y = torch.istft(torch.polar(mag, pha), N_FFT, HOP, window=window, length=x.shape[-1])
    y = y[0] if squeeze else y
    return y.numpy() if as_numpy else y
