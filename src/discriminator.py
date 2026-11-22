"""MP-SENet 메트릭 판별기 (원본 models/discriminator.py, MIT License, Copyright (c) 2023 Ye-Xin Lu).

(clean 압축 크기, 비교 대상 압축 크기)를 받아 정규화 PESQ (PESQ-1)/3.5 를 예측한다.
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True), nn.PReLU(dim),
            spectral_norm(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True), nn.PReLU(dim * 2),
            spectral_norm(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True), nn.PReLU(dim * 4),
            spectral_norm(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True), nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            spectral_norm(nn.Linear(dim * 8, dim * 4)), nn.Dropout(0.3), nn.PReLU(dim * 4),
            spectral_norm(nn.Linear(dim * 4, 1)),
            LearnableSigmoid1d(1),
        )

    def forward(self, x, y):
        return self.layers(torch.stack((x, y), dim=1))
