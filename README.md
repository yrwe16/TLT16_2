# TLT16 · 신호 및 시스템 2차 챌린지: 딥러닝 음성 잡음 제거

잡음 섞인 16 kHz 모노 음성(noisy)에서 배경 잡음을 줄이는 딥러닝 모델입니다.
`input/`의 모든 WAV를 처리해 `output/`에 **같은 파일명, 16 kHz 모노, 같은 표본 수**로 저장합니다.

## 현재 제출 모델

| 항목 | 값 |
| :--- | :--- |
| 모델 | `MaskCNN` (조교 예제 구조: STFT 로그 크기 → Conv1d 3층 → Sigmoid 마스크, 111,361 파라미터) |
| 가중치 | `weights/mask_cnn_baseline.pt` (Git LFS), 설정 `weights/mask_cnn_baseline.json` |
| **검증 PESQ-WB** | **1.7255** (검증 164개 평균, `pesq==0.0.4` wb). 잡음 입력 1.3947 |
| 검증 SNR | 13.42 dB (잡음 입력 9.63 dB) |
| 선택 기준 | 검증 평균 PESQ-WB가 가장 높은 epoch |

## 1. 실행 환경

- 학습: Windows 11, NVIDIA RTX 4090, Python 3.12, PyTorch 2.15.0+cu132
- 추론: **GPU가 없으면 자동으로 CPU에서 실행**됩니다. CPU 추론도 검증셋에서 같은 점수(1.7255)를 확인했습니다.

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

- 기본 경로: 입력 `./input/`, 출력 `./output/`, 설정 `./weights/mask_cnn_baseline.json`
- 다른 경로: `python infer.py --input <입력 폴더> --output <출력 폴더> --device cpu`
- noisy 입력과 저장된 가중치만 사용하며, 추가 다운로드가 없습니다.
- 처리 후 **파일명 일치, 누락/추가 WAV, 16 kHz 모노, 표본 수, NaN/Inf**를 자동 검사합니다. 하나라도 어긋나면 오류 목록을 출력하고 종료 코드 1로 끝납니다.
- `output/`에 입력에 없는 이전 WAV가 남아 있으면 실행을 멈춥니다. 지운 뒤 다시 실행하세요.

## 3. 검증셋 자체 검증

```bash
python selftest_validation.py --data-root <배포 데이터 폴더>
```

검증 noisy 164개를 임시 `input/`에 복사하고 `infer.py`를 실행한 뒤, 저장된 `output/` WAV를 clean과 비교해 평균 PESQ-WB·SNR과 잡음 종류별 점수를 출력합니다.

## 4. 학습

```bash
python train.py --data-root <배포 데이터 폴더> --run-name <실험 이름>
```

결과는 `runs/<실험 이름>/`에 저장됩니다(`best.pt`, `history.csv`, `summary.json`, `train.log`). 제출할 가중치는 `weights/`로 복사하고 같은 이름의 `.json` 설정을 함께 둡니다.

| 설정 | 값 |
| :--- | :--- |
| seed | 1 (기준선 seed 분산 확인용으로 42·1·2를 학습해 최고 검증 점수를 선택) |
| STFT | `n_fft` 512, `hop` 128, Hann 창 |
| 손실 | 로그 크기 L1: `L1(log1p|Ŝ|, log1p|S|)` |
| 최적화 | Adam, lr 1e-3, betas (0.9, 0.999), weight decay 0, batch 16 |
| 학습률 조정 | ReduceLROnPlateau(검증 PESQ 기준, ×0.5, patience 5, min 1e-5) |
| 종료 | 최대 200 epoch, 최소 30 epoch 후 PESQ 개선(≥0.001)이 20 epoch 없으면 조기 종료 |
| 학습 epoch | 104 epoch에서 조기 종료, **best epoch 84** |
| DataLoader | `num_workers=0` (Windows) |

같은 설정의 seed별 결과: seed 42 1.6725, seed 1 1.7255, seed 2 1.6993 (평균 1.699).

## 5. 개인 PC 경로를 고쳐야 하는 위치

| 파일 | 위치 | 기본값 |
| :--- | :--- | :--- |
| `train.py` | `DEFAULT_DATA_ROOT` 또는 `--data-root` | `C:\deep_learning_noise_reduction_challenge` |
| `selftest_validation.py` | `DEFAULT_DATA_ROOT` 또는 `--data-root` | 위와 같음 |

`infer.py`는 저장소 기준 상대 경로만 사용하므로 수정할 필요가 없습니다.

## 6. 폴더 구조

```
infer.py                 제출용 추론 (input/ → output/)
train.py                 학습
selftest_validation.py   검증셋으로 제출 형식·점수 자체 검증
src/audio.py             WAV 입출력, PESQ-WB, SNR
src/data.py              학습·검증 데이터셋(캐시는 cache/)
src/models.py            모델 정의
weights/                 제출 가중치(*.pt, Git LFS)와 설정(*.json)
```

학습 데이터 WAV와 `samples.csv`는 개인 식별 정보가 있어 저장소에 포함하지 않습니다(`.gitignore`).
