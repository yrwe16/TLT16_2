"""torch-pesq를 typeguard/torchtyping/torchaudio 없이 쓰기 위한 대체 구현."""
import numpy as np
import torch


def typechecked(fn):
    return fn


class TensorType:
    def __class_getitem__(cls, item):
        return torch.Tensor


_IR_CACHE = {}


def _impulse_response(b, a, n):
    """IIR 필터 b/a의 임펄스 응답 앞부분 n개(numpy 직접 계산, float64)."""
    b, a = np.asarray(b, np.float64) / a[0], np.asarray(a, np.float64) / a[0]
    y = np.zeros(n)
    for t in range(n):  # 직접형 차분방정식, 입력은 t=0의 임펄스
        acc = b[t] if t < len(b) else 0.0
        for k in range(1, min(len(a), t + 1)):
            acc -= a[k] * y[t - k]
        y[t] = acc
    return y


def lfilter(signal, a, b, clamp=False, taps=8192):
    """torchaudio.functional.lfilter 대체: 임펄스 응답을 taps개로 자른 FIR을 FFT 합성곱으로 적용.

    쓰이는 두 필터(사전 강조 2차, 325~3250 Hz 5차 Butterworth 대역통과)는 안정 필터라
    8192표본 뒤 꼬리 에너지가 무시할 수준이다. 미분 가능하다.
    """
    key = (tuple(b.detach().cpu().tolist()), tuple(a.detach().cpu().tolist()), taps)
    if key not in _IR_CACHE:
        _IR_CACHE[key] = torch.tensor(_impulse_response(key[0], key[1], taps), dtype=torch.float32)
    h = _IR_CACHE[key].to(signal.device)
    n = signal.shape[-1]
    size = 1 << (n + taps - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(signal, size) * torch.fft.rfft(h, size), size)[..., :n]
    return y


class Spectrogram(torch.nn.Module):
    def __init__(self, win_length, n_fft, hop_length, window_fn, power, normalized, center):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop_length, win_length
        self.register_buffer("window", window_fn(win_length), persistent=False)

    def forward(self, x):
        spec = torch.stft(x, self.n_fft, self.hop, self.win, window=self.window.to(x.device),
                          center=False, return_complex=True)
        return spec.abs().pow(2)
