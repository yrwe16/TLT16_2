# TLT16 · 신호 및 시스템 2차 챌린지: 딥러닝 음성 잡음 제거

잡음 섞인 16 kHz 모노 음성(noisy)에서 배경 잡음을 줄이는 딥러닝 모델입니다.
`input/`의 모든 WAV를 처리해 `output/`에 **같은 파일명, 16 kHz 모노, 같은 표본 수**로 저장합니다.

## 현재 제출 모델

| 항목 | 값 |
| :--- | :--- |
| 모델 | **MP-SENet** (Lu et al., 2023, MIT License): 크기·위상을 함께 복원하는 STFT 기반 모델, 2,263,372 파라미터 |
| 후처리 | PCS(Perceptual Contrast Stretching), 강도 alpha 0.3 (`src/pcs.py`) |
| 가중치 | `weights/mpsenet_pesqloss_pcs.pt` (Git LFS), 설정 `weights/mpsenet_pesqloss_pcs.json` |
| **검증 PESQ-WB** | **3.4824** (검증 164개 평균, `pesq==0.0.4` wb). 잡음 입력 1.3947 |
| 선택 기준 | 검증 평균 PESQ-WB가 가장 높은 epoch, PCS 강도는 검증셋에서 0~1 중 선택(0.3~0.5 구간은 거의 같음) |
| CPU 추론 | 검증 164개 266초(파일당 약 1.7초), GPU와 같은 점수 |

### 단계별 검증 점수

| 단계 | 검증 PESQ-WB |
| :--- | :--- |
| 잡음 입력 | 1.3947 |
| 예제 MaskCNN 기준선 | 1.7255 |
| MP-SENet 공개 DNS 가중치, 학습 없이 | 3.1757 |
| + 우리 학습셋으로 미세조정(동적 리믹싱 0.5, PESQ 판별기 0.05) | 3.2936 |
| + Zeroth Korean clean 음성(46.5시간)에 우리 학습셋 잡음을 섞어 추가 학습 | 3.3356 |
| + 미분 가능한 PESQ 손실(0.1)로 추가 학습 | 3.4529 |
| **+ PCS 후처리 (alpha 0.3)** | **3.4824** |

미세조정 ablation: 학습셋만 3.2799, +동적 리믹싱 3.2857, +PESQ 판별기 3.2936.

## 1. 실행 환경

- 학습: Windows 11, NVIDIA RTX 4090, Python 3.12, PyTorch 2.15.0+cu132
- 추론: **GPU가 없으면 자동으로 CPU에서 실행**됩니다. CPU 추론도 검증셋에서 같은 점수(3.4824)를 확인했습니다.

```bash
conda env create -f environment.yml
conda activate denoise
# GPU (CUDA 13.2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
# 또는 CPU만
pip install torch
```

가중치는 Git LFS로 저장되어 있습니다. 클론 전에 `git lfs install`을 한 번 실행하거나, 클론 후 `git lfs pull`로 받으세요.

## 2. 추론 (제출 형식)

```bash
python infer.py
```

- 기본 경로: 입력 `./input/`, 출력 `./output/`, 설정 `./weights/mpsenet_pesqloss_pcs.json`
- 다른 경로: `python infer.py --input <입력 폴더> --output <출력 폴더> --device cpu`
- noisy 입력과 저장된 가중치만 사용하며, 추가 다운로드가 없습니다. 파일마다 하나씩(batch 1) 처리합니다.
- 처리 후 **파일명 일치, 누락/추가 WAV, 16 kHz 모노, 표본 수, NaN/Inf**를 자동 검사합니다. 하나라도 어긋나면 오류 목록을 출력하고 종료 코드 1로 끝납니다.
- `output/`에 입력에 없는 이전 WAV가 남아 있으면 실행을 멈춥니다. 지운 뒤 다시 실행하세요.
- 예제 기준선으로 돌리려면 `--config weights/mask_cnn_baseline.json`.

## 3. 검증셋 자체 검증

```bash
python selftest_validation.py --data-root <배포 데이터 폴더>
python evaluate.py --model mpsenet --ckpt weights/mpsenet_pesqloss_pcs.pt --by-noise   # PCS 없이 모델만
```

`selftest_validation.py`는 검증 noisy 164개를 임시 `input/`에 복사하고 `infer.py`를 실행한 뒤, 저장된 `output/` WAV를 clean과 비교해 평균 PESQ-WB·SNR과 잡음 종류별 점수를 출력합니다.

## 4. 학습

### 4.1 외부 데이터 준비

1. MP-SENet 원본 코드와 공개 가중치: <https://github.com/yxlu-0102/MP-SENet> 를 `external/MP-SENet/`에 클론(`best_ckpt/g_best_dns` 사용).
2. Zeroth Korean(OpenSLR 40, CC BY 4.0): <https://www.openslr.org/40/> 의 `zeroth_korean.tar.gz`(10.3 GB)를 `external/zeroth/`에 풀기.
3. 2초 구간 캐시 만들기: `python build_ext_speech.py --src external/zeroth --name zeroth` → `cache/zeroth_32000.npy`

### 4.2 미세조정

```bash
# 1) 공개 DNS 가중치 → 우리 학습셋 (동적 리믹싱 + PESQ 판별기)
python finetune_mpsenet.py --run-name ft_dns_remix_disc --epochs 30 --remix-prob 0.5 --metric-weight 0.05
# 2) + Zeroth 외부 음성
python finetune_mpsenet.py --run-name ft_ext_zeroth --init runs/ft_dns_remix_disc/g_best --epochs 30 \
    --steps-per-epoch 1000 --ext-speech cache/zeroth_32000.npy --ext-prob 0.5 --remix-prob 0.5 \
    --metric-weight 0.05 --patience 6
# 3) + 미분 가능한 PESQ 손실 (제출 모델, best epoch 19)
python finetune_mpsenet.py --run-name ft_ext_pesqloss --init runs/ft_ext_zeroth/g_best --epochs 20 \
    --steps-per-epoch 1000 --lr 4e-5 --lr-decay 0.985 --ext-speech cache/zeroth_32000.npy --ext-prob 0.5 \
    --remix-prob 0.5 --metric-weight 0.05 --pesq-loss-weight 0.1 --patience 10
```

결과는 `runs/<실험 이름>/`에 저장됩니다(`g_best`, `history.csv`, `summary.json`, `train.log`). 제출할 가중치는 `weights/`로 복사하고 같은 이름의 `.json` 설정을 함께 둡니다.

| 설정 | 값 |
| :--- | :--- |
| 시작 가중치 | MP-SENet 공개 DNS 가중치(`g_best_dns`) |
| STFT | `n_fft` 400, `hop` 100, 크기 0.3 압축, 입력 RMS 정규화 |
| 손실 | 크기 MSE 0.9, anti-wrapping 위상 0.3, 복소 MSE 0.1, 일관성 0.1, 시간 L1 0.2, PESQ 판별기 0.05 |
| 최적화 | AdamW lr 5e-5, betas (0.8, 0.99), epoch마다 ×0.98, batch 4, 2초 구간 |
| 동적 리믹싱 | 학습셋 잡음(noisy − clean)을 다른 clean에 SNR U[−5, 20] dB로 섞음, 시간 뒤집기. 검증셋은 사용하지 않음 |
| 외부 음성 | 샘플마다 50% 확률로 Zeroth clean + 학습셋 잡음 |
| 종료 | 검증 PESQ-WB가 6~8 epoch 동안 개선되지 않으면 조기 종료 |
| seed | 1234 |

추가 손실(선택): `--pesq-loss-weight`(미분 가능한 PESQ 근사 손실, `src/torch_pesq/`), `--sisdr-weight`(음의 SI-SDR). 제출 모델은 PESQ 손실 0.1을 사용했고, SI-SDR(0.01)은 개선이 없어 쓰지 않았습니다.

공개 가중치는 원본 코드의 attention `batch_first` 설정 그대로 학습되어 있어 `legacy_attention=True`로 맞추고, 추론은 파일당 batch 1로 합니다.

예제 MaskCNN 기준선은 `python train.py --data-root <배포 데이터 폴더> --run-name <실험 이름>`(seed 1, best epoch 84, 1.7255).

## 5. 개인 PC 경로를 고쳐야 하는 위치

| 파일 | 위치 | 기본값 |
| :--- | :--- | :--- |
| `train.py`, `finetune_mpsenet.py`, `evaluate.py` | `DEFAULT_DATA_ROOT` 또는 `--data-root` | `C:\deep_learning_noise_reduction_challenge` |
| `selftest_validation.py` | `DEFAULT_DATA_ROOT` 또는 `--data-root` | 위와 같음 |

`infer.py`는 저장소 기준 상대 경로만 사용하므로 수정할 필요가 없습니다.

## 6. 폴더 구조

```
infer.py                 제출용 추론 (input/ → output/)
finetune_mpsenet.py      MP-SENet 미세조정
build_ext_speech.py      외부 clean 음성 캐시 만들기
evaluate.py              검증셋 평가(SNR·잡음 종류별)
train.py                 예제 MaskCNN 기준선 학습
selftest_validation.py   검증셋으로 제출 형식·점수 자체 검증
src/audio.py             WAV 입출력, PESQ-WB, SNR
src/data.py              학습·검증 데이터셋(캐시는 cache/)
src/mpsenet.py           MP-SENet 모델(원본 MIT 코드 이식)
src/discriminator.py     PESQ 메트릭 판별기(원본 MIT 코드)
src/pcs.py               PCS 후처리
src/torch_pesq/          미분 가능한 PESQ 손실(torch-pesq 0.1.2 이식, MIT, scipy·torchaudio 의존 제거)
src/models.py            예제 MaskCNN
weights/                 제출 가중치(*.pt, Git LFS)와 설정(*.json)
```

학습 데이터 WAV와 `samples.csv`는 개인 식별 정보가 있어 저장소에 포함하지 않습니다(`.gitignore`). 외부 데이터(`external/`)와 캐시(`cache/`)도 저장소에 넣지 않습니다.

## 7. 사용한 외부 자료

- MP-SENet 코드·공개 가중치: Ye-Xin Lu et al., "MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra", Interspeech 2023. MIT License.
- Zeroth Korean: OpenSLR 40, CC BY 4.0.
- torch-pesq: <https://github.com/audiolabs/torch-pesq> (MIT), 학습용 PESQ 손실로만 사용.
- PCS: Chao et al., "Perceptual Contrast Stretching on Target Feature for Speech Enhancement", Interspeech 2022.
