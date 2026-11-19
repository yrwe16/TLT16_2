"""모델 정의. MaskCNN은 예제 노트북 3.2절과 같은 구조·state_dict 키를 쓴다."""
import torch
from torch import nn


class MaskCNN(nn.Module):
    """log(1+|Y|) -> Conv1d 3층 -> Sigmoid 마스크 -> 마스크 * Y (입력 위상 유지)."""

    def __init__(self, n_fft=512, hop=128, hidden_channels=(64, 64), kernel_sizes=(5, 3, 1)):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        bins = n_fft // 2 + 1
        c1, c2 = hidden_channels
        k1, k2, k3 = kernel_sizes
        self.mask_net = nn.Sequential(
            nn.Conv1d(bins, c1, kernel_size=k1, padding=k1 // 2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.ReLU(),
            nn.Conv1d(c2, bins, kernel_size=k3, padding=k3 // 2),
            nn.Sigmoid(),
        )
        self.register_buffer("window", torch.hann_window(n_fft))

    def spectrum(self, wave):
        return torch.stft(wave, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
                          return_complex=True)

    def forward(self, noisy):
        spectrum = self.spectrum(noisy)
        mask = self.mask_net(torch.log1p(spectrum.abs()))
        return mask * spectrum

    def istft(self, spectrum, length):
        return torch.istft(spectrum, n_fft=self.n_fft, hop_length=self.hop,
                           window=self.window, length=length)

    def enhance(self, noisy):
        return self.istft(self(noisy), noisy.shape[-1])


MODELS = {"mask_cnn": MaskCNN}


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)
