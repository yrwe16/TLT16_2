"""학습·검증 데이터셋. 예제 노트북 2절의 PairedSpeech와 같은 캐시 형식.

캐시는 원본 데이터 폴더가 아니라 이 저장소의 cache/ 에 저장한다(원본 폴더 수정 금지).
"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import SAMPLE_RATE, read_wav


def _load_waveforms(root, split, manifest, cache_path, sample_count):
    shape = (len(manifest), 2, sample_count)  # [쌍, noisy/clean, 표본]
    if cache_path.is_file():
        try:
            waveforms = np.load(cache_path, allow_pickle=False)
            if waveforms.shape == shape and waveforms.dtype == np.float32:
                return waveforms
        except (OSError, ValueError, EOFError):
            pass
    waveforms = np.empty(shape, dtype=np.float32)
    for index, row in enumerate(manifest.itertuples(index=False)):
        noisy = read_wav(root / row.noisy_path)
        clean = read_wav(root / row.clean_path)
        if noisy.size != sample_count or clean.size != sample_count:
            raise ValueError(f"파형 길이가 일치하지 않습니다: {row.segment_id}")
        waveforms[index, 0] = noisy
        waveforms[index, 1] = clean
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / cache_path.name
        np.save(temporary_path, waveforms, allow_pickle=False)
        temporary_path.replace(cache_path)
    print(f"{split}: 캐시 저장 완료 ({len(manifest):,}쌍) -> {cache_path}", flush=True)
    return waveforms


class PairedSpeech(Dataset):
    def __init__(self, root, split, cache_dir):
        root = Path(root)
        manifest_bytes = (root / split / "samples.csv").read_bytes()
        self.manifest = pd.read_csv(BytesIO(manifest_bytes))
        if (self.manifest.empty or self.manifest.segment_id.duplicated().any()
                or not self.manifest["split"].eq(split).all()):
            raise ValueError(f"{split} manifest의 샘플 ID 또는 분할을 확인하세요.")
        lengths = self.manifest["num_samples"]
        sample_count = int(lengths.iloc[0])
        if sample_count <= 0 or not lengths.eq(sample_count).all():
            raise ValueError(f"{split} manifest의 표본 수가 같아야 합니다.")
        manifest_key = sha256(manifest_bytes).hexdigest()[:20]
        cache_path = Path(cache_dir) / f"{split}_v1_{SAMPLE_RATE}_{manifest_key}.npy"
        self.waveforms = torch.from_numpy(
            _load_waveforms(root, split, self.manifest, cache_path, sample_count))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        noisy, clean = self.waveforms[index]
        return noisy, clean
