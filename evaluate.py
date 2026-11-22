"""검증셋 평가: 모델로 noisy를 복원하고 PESQ-WB(병렬), SNR, 잡음 종류·SNR별 점수를 출력한다.

예) python evaluate.py --model mpsenet --ckpt external/MP-SENet/best_ckpt/g_best_dns
    python evaluate.py --model mask_cnn --ckpt weights/mask_cnn_baseline.pt
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from src.audio import pesq_wb_many, snr_db  # DLL 경로 설정을 위해 가장 먼저 import

import numpy as np
import pandas as pd
import torch

from src.data import PairedSpeech

REPO = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"C:\deep_learning_noise_reduction_challenge")  # 개인 PC 경로: --data-root로 변경


def load_model(name, ckpt, device):
    if name == "mpsenet":
        from src.mpsenet import MPSENet
        model = MPSENet().load_original_checkpoint(ckpt, map_location=device)
    elif name == "mask_cnn":
        from src.models import MaskCNN
        model = MaskCNN()
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        raise ValueError(name)
    return model.to(device).eval()


@torch.inference_mode()
def enhance_all(model, noisy_batch, device, batch_size=1):
    outputs = []
    for start in range(0, len(noisy_batch), batch_size):
        wave = noisy_batch[start:start + batch_size].to(device)
        outputs.append(model.enhance(wave).float().cpu())
    return torch.cat(outputs).numpy()


def report(manifest, noisy, clean, enhanced, label):
    t = perf_counter()
    in_pesq = pesq_wb_many(clean, noisy)
    out_pesq = pesq_wb_many(clean, enhanced)
    detail = manifest[["segment_id", "noise_category", "target_snr_db"]].copy()
    detail["input_pesq_wb"], detail["output_pesq_wb"] = in_pesq, out_pesq
    detail["input_snr_db"] = [snr_db(c, n) for c, n in zip(clean, noisy)]
    detail["output_snr_db"] = [snr_db(c, e) for c, e in zip(clean, enhanced)]
    print(f"[{label}] PESQ-WB {detail.input_pesq_wb.mean():.4f} -> {detail.output_pesq_wb.mean():.4f} | "
          f"SNR {detail.input_snr_db.mean():.2f} -> {detail.output_snr_db.mean():.2f} dB | "
          f"PESQ {perf_counter() - t:.1f}s")
    by_snr = detail.groupby("target_snr_db")[["input_pesq_wb", "output_pesq_wb"]].mean().round(3)
    print("  SNR별:", "  ".join(f"{int(k)}dB {r.input_pesq_wb:.2f}->{r.output_pesq_wb:.2f}"
                               for k, r in by_snr.iterrows()))
    return detail


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["mpsenet", "mask_cnn"])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--by-noise", action="store_true")
    args = p.parse_args()

    data = PairedSpeech(args.data_root, args.split, REPO / "cache")
    noisy, clean = data.waveforms[:, 0], data.waveforms[:, 1]
    model = load_model(args.model, args.ckpt, args.device)
    t = perf_counter()
    enhanced = enhance_all(model, noisy, torch.device(args.device))
    print(f"{args.model} {args.ckpt.name} | {sum(q.numel() for q in model.parameters()):,} params | "
          f"{len(data)} files enhanced in {perf_counter() - t:.1f}s on {args.device}")
    if not np.all(np.isfinite(enhanced)):
        raise FloatingPointError("출력에 NaN/Inf가 있습니다.")
    detail = report(data.manifest, noisy.numpy(), clean.numpy(), enhanced, args.ckpt.name)
    if args.by_noise:
        by = detail.groupby("noise_category")[["input_pesq_wb", "output_pesq_wb"]].mean()
        by["delta"] = by.output_pesq_wb - by.input_pesq_wb
        print(by.sort_values("delta").round(3).to_string())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.report, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
