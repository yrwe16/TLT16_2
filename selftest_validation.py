"""제출 형식 자체 검증: 검증셋 noisy를 임시 input/ 으로 복사 → infer.py 실행 → output/ 을 clean과 비교.

input/, output/ 은 .gitignore 대상인 임시 폴더이며, 이 스크립트는 그 안의 WAV를 지우고 다시 만든다.

예) python selftest_validation.py                 # 자동 장치
    python selftest_validation.py --device cpu    # 조교 PC처럼 CPU만으로
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from src.audio import pesq_wb, read_wav, snr_db  # DLL 경로 설정을 위해 가장 먼저 import

import pandas as pd

REPO = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"C:\deep_learning_noise_reduction_challenge")  # 개인 PC 경로: --data-root로 변경


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="validation")
    p.add_argument("--config", type=Path, default=REPO / "weights" / "mpsenet_pesqloss_pcs.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--report", type=Path, default=None, help="파일별 결과 CSV 저장 경로(선택)")
    return p.parse_args()


def reset_wavs(folder):
    folder.mkdir(exist_ok=True)
    for path in folder.glob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            path.unlink()


def main():
    args = parse_args()
    input_dir, output_dir = REPO / "input", REPO / "output"
    manifest = pd.read_csv(args.data_root / args.split / "samples.csv")

    reset_wavs(input_dir)
    reset_wavs(output_dir)
    for row in manifest.itertuples(index=False):
        shutil.copy2(args.data_root / row.noisy_path, input_dir / Path(row.noisy_path).name)
    print(f"{args.split} noisy {len(manifest)}개를 {input_dir}에 복사", flush=True)

    command = [sys.executable, str(REPO / "infer.py"), "--input", str(input_dir), "--output", str(output_dir),
               "--config", str(args.config), "--device", args.device]
    result = subprocess.run(command, cwd=REPO)
    if result.returncode != 0:
        sys.exit(f"infer.py 실패 (exit {result.returncode})")

    rows = []
    for row in manifest.itertuples(index=False):
        name = Path(row.noisy_path).name
        clean = read_wav(args.data_root / row.clean_path)
        noisy = read_wav(input_dir / name)
        enhanced = read_wav(output_dir / name)  # 저장된 WAV를 다시 읽어 실제 제출물 기준으로 채점
        rows.append({"file": name, "noise_category": row.noise_category, "target_snr_db": row.target_snr_db,
                     "input_pesq_wb": pesq_wb(clean, noisy), "output_pesq_wb": pesq_wb(clean, enhanced),
                     "input_snr_db": snr_db(clean, noisy), "output_snr_db": snr_db(clean, enhanced)})
    detail = pd.DataFrame(rows)
    if args.report:
        detail.to_csv(args.report, index=False, encoding="utf-8-sig")

    print(f"\nPESQ-WB 평균: 입력 {detail.input_pesq_wb.mean():.4f} -> 출력 {detail.output_pesq_wb.mean():.4f} "
          f"({len(detail)}개 파일, PESQ 오류 0건)")
    print(f"SNR 평균:     입력 {detail.input_snr_db.mean():.2f} dB -> 출력 {detail.output_snr_db.mean():.2f} dB")
    by_noise = detail.groupby("noise_category")[["input_pesq_wb", "output_pesq_wb"]].mean()
    by_noise["delta"] = by_noise.output_pesq_wb - by_noise.input_pesq_wb
    print("\n잡음 종류별 평균 PESQ-WB:")
    print(by_noise.sort_values("delta").round(3).to_string())


if __name__ == "__main__":
    main()
