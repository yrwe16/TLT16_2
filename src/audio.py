"""공통 유틸: conda DLL 경로, WAV 입출력, PESQ/SNR 계산.

이 모듈을 가장 먼저 import하면 conda 환경의 python.exe를 직접 실행해도
soundfile이 libsndfile.dll을 찾을 수 있다(예제 노트북 1절과 같은 처리).
"""
import os
import sys
from pathlib import Path

if sys.platform == "win32" and (Path(sys.prefix) / "conda-meta").is_dir():
    _conda_bin = str(Path(sys.prefix) / "Library" / "bin")
    os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")
    _dll_handle = os.add_dll_directory(_conda_bin)

import numpy as np
import soundfile as sf
from pesq import pesq

SAMPLE_RATE = 16_000


def read_wav(path):
    """16 kHz 모노 WAV를 float32 numpy 배열로 읽는다."""
    samples, rate = sf.read(str(path), dtype="float32")
    if rate != SAMPLE_RATE or samples.ndim != 1:
        raise ValueError(f"16 kHz 모노 WAV가 필요합니다: {path} (rate={rate}, shape={samples.shape})")
    return samples


def write_wav(path, samples):
    """16 kHz 모노 float32 WAV로 저장한다."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"모노 1차원 배열이 필요합니다: shape={samples.shape}")
    sf.write(str(path), samples, SAMPLE_RATE, subtype="FLOAT")


def pesq_wb(clean, estimate):
    score = float(pesq(SAMPLE_RATE, clean, estimate, mode="wb"))
    if not np.isfinite(score) or score < 0:
        raise ValueError(f"유효하지 않은 PESQ-WB 점수: {score}")
    return score


def snr_db(clean, estimate):
    clean = clean.astype(np.float64)
    error = clean - estimate
    signal_energy = np.sum(clean ** 2)
    error_energy = np.sum(error ** 2)
    if signal_energy == 0:
        raise ValueError("무음 기준 파형의 SNR은 정의되지 않습니다.")
    return np.inf if error_energy == 0 else 10 * np.log10(signal_energy / error_energy)
