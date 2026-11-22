"""torch-pesq (https://github.com/audiolabs/torch-pesq, commit 3aac3c8, MIT License) 이식본.

변경: typeguard/torchtyping/torchaudio 의존 제거(_compat.py), 리샘플링 제거(16 kHz 전용).
"""
from .loss import PesqLoss
