"""제출용 추론: input/ 의 모든 WAV를 처리해 output/ 에 같은 파일명으로 저장한다.

- noisy 입력과 저장된 가중치만 사용한다(추가 다운로드 없음).
- GPU가 없으면 자동으로 CPU에서 실행한다(map_location).
- 저장 후 파일명 일치, 16 kHz 모노, 표본 수 일치, 유한값 여부를 검사하고
  하나라도 어긋나면 0이 아닌 종료 코드로 끝난다.

예) python infer.py
    python infer.py --input input --output output --config weights/mask_cnn_baseline.json --device cpu
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from src.audio import SAMPLE_RATE, read_wav, write_wav  # DLL 경로 설정을 위해 가장 먼저 import

import numpy as np
import soundfile as sf
import torch

from src.models import build_model

REPO = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=REPO / "input")
    p.add_argument("--output", type=Path, default=REPO / "output")
    p.add_argument("--config", type=Path, default=REPO / "weights" / "mask_cnn_baseline.json")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    return p.parse_args()


def list_wavs(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def load_model(config_path, device):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = build_model(config["model"], **config.get("model_kwargs", {}))
    state = torch.load(config_path.parent / config["weights"], map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), config


@torch.inference_mode()
def enhance(model, noisy, device):
    """임의 길이의 1차원 파형을 복원하고 입력과 같은 길이로 돌려준다."""
    length = noisy.shape[0]
    pad = max(0, model.n_fft - length)  # 아주 짧은 입력도 STFT가 가능하도록
    wave = torch.from_numpy(np.pad(noisy, (0, pad))).unsqueeze(0).to(device)
    out = model.enhance(wave)[0, :length].float().cpu().numpy()
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("출력에 NaN/Inf가 있습니다.")
    return out.astype(np.float32)


def verify(inputs, output_dir):
    """제출 형식 검사. 문제 목록을 돌려준다(비어 있으면 통과)."""
    problems = []
    expected = {p.name for p in inputs}
    produced = {p.name for p in list_wavs(output_dir)}
    problems += [f"누락: {name}" for name in sorted(expected - produced)]
    problems += [f"추가 WAV: {name}" for name in sorted(produced - expected)]
    for src in inputs:
        dst = output_dir / src.name
        if not dst.is_file():
            continue
        info_in, info_out = sf.info(str(src)), sf.info(str(dst))
        if info_out.samplerate != SAMPLE_RATE or info_out.channels != 1:
            problems.append(f"형식 오류 {dst.name}: {info_out.samplerate} Hz, {info_out.channels} ch")
        if info_out.frames != info_in.frames:
            problems.append(f"표본 수 불일치 {dst.name}: 입력 {info_in.frames}, 출력 {info_out.frames}")
        data, _ = sf.read(str(dst), dtype="float32")
        if not np.all(np.isfinite(data)):
            problems.append(f"NaN/Inf 포함: {dst.name}")
    return problems


def main():
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    if not args.input.is_dir():
        sys.exit(f"입력 폴더가 없습니다: {args.input}")
    inputs = list_wavs(args.input)
    if not inputs:
        sys.exit(f"입력 폴더에 WAV가 없습니다: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    stale = [p for p in list_wavs(args.output) if p.name not in {q.name for q in inputs}]
    if stale:
        sys.exit(f"output/에 입력에 없는 WAV {len(stale)}개가 있습니다(예: {stale[0].name}). "
                 "이전 결과를 지운 뒤 다시 실행하세요.")

    model, config = load_model(args.config, device)
    print(f"model {config['model']} | weights {config['weights']} | device {device} | files {len(inputs)}", flush=True)
    started = perf_counter()
    for index, src in enumerate(inputs, 1):
        noisy = read_wav(src)
        write_wav(args.output / src.name, enhance(model, noisy, device))
        if index % 50 == 0 or index == len(inputs):
            print(f"  {index}/{len(inputs)} files", flush=True)
    print(f"추론 완료: {perf_counter() - started:.1f}s", flush=True)

    problems = verify(inputs, args.output)
    if problems:
        print(f"형식 검사 실패 ({len(problems)}건):")
        for line in problems[:50]:
            print("  " + line)
        sys.exit(1)
    print(f"형식 검사 통과: {len(inputs)}개 파일, 파일명·16 kHz 모노·표본 수 일치")


if __name__ == "__main__":
    main()
