"""학습 스크립트. 기본값은 예제 노트북 4.2절의 기준선 설정과 같다.

예) python train.py --run-name baseline_cnn_gpu
결과: runs/<run-name>/{best.pt, history.csv, summary.json, train.log}
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from src.audio import pesq_wb  # DLL 경로 설정을 위해 가장 먼저 import

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import PairedSpeech
from src.models import build_model

REPO = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"C:\deep_learning_noise_reduction_challenge")  # 개인 PC 경로: --data-root로 변경


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-dir", type=Path, default=REPO / "cache")
    p.add_argument("--run-name", default=None)
    p.add_argument("--model", default="mask_cnn")
    p.add_argument("--n-fft", type=int, default=512)
    p.add_argument("--hop", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.001)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def log_mag_l1(estimate, target):
    return nn.functional.l1_loss(torch.log1p(estimate.abs()), torch.log1p(target.abs()))


def run_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        loss = log_mag_l1(model(noisy), model.spectrum(clean))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss가 발산했습니다: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * noisy.size(0)
    return total / len(loader.dataset)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    total, scores = 0.0, []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        estimate = model(noisy)
        total += log_mag_l1(estimate, model.spectrum(clean)).item() * noisy.size(0)
        enhanced = model.istft(estimate, noisy.shape[-1]).cpu().numpy()
        for reference, restored in zip(clean.cpu().numpy(), enhanced):
            scores.append(pesq_wb(reference, restored))
    if len(scores) != len(loader.dataset):
        raise RuntimeError("검증 전체 파일의 PESQ 계산이 필요합니다.")
    return total / len(loader.dataset), float(np.mean(scores))


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPO / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    t0 = perf_counter()
    train_set = PairedSpeech(args.data_root, "train", args.cache_dir)
    val_set = PairedSpeech(args.data_root, "validation", args.cache_dir)
    log(f"data loaded in {perf_counter() - t0:.1f}s | train {len(train_set)} | val {len(val_set)}")

    torch.manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args.model, n_fft=args.n_fft, hop=args.hop).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=args.min_lr)

    checkpoint = run_dir / "best.pt"
    best_pesq, stopping_reference = -float("inf"), -float("inf")
    best_epoch, stale_epochs, history = 0, 0, []
    stop_reason = "max_epochs"
    log(f"start | torch {torch.__version__} | device {device} | model {args.model} ({n_params:,} params) | "
        f"args {vars(args)}")
    started = perf_counter()
    train_time = val_time = 0.0
    for epoch in range(1, args.max_epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        t = perf_counter()
        train_loss = run_epoch(model, train_loader, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += perf_counter() - t
        t = perf_counter()
        val_loss, val_pesq = validate(model, val_loader, device)
        val_time += perf_counter() - t

        if val_pesq > best_pesq:
            best_pesq, best_epoch = val_pesq, epoch
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.replace(checkpoint)
        if val_pesq >= stopping_reference + args.min_delta:
            stopping_reference, stale_epochs = val_pesq, 0
        else:
            stale_epochs += 1
        scheduler.step(val_pesq)

        elapsed = perf_counter() - started
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "validation_pesq_wb": val_pesq, "learning_rate": lr, "best_epoch": best_epoch,
                        "best_pesq_wb": best_pesq, "stale_epochs": stale_epochs, "elapsed_seconds": elapsed})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(f"epoch {epoch:3d} | train {train_loss:.5f} | val loss {val_loss:.5f} | val PESQ {val_pesq:.4f} | "
            f"best {best_pesq:.4f} (ep {best_epoch}) | lr {lr:.2g} | {elapsed:.0f}s")
        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            stop_reason = "early_stopping"
            break

    elapsed = perf_counter() - started
    summary = {
        "run_name": run_name, "stop_reason": stop_reason, "epochs_completed": epoch,
        "best_epoch": best_epoch, "best_validation_pesq_wb": best_pesq, "validation_files": len(val_set),
        "elapsed_seconds": elapsed, "train_seconds": train_time, "validation_seconds": val_time,
        "parameters": n_params, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch_version": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done ({stop_reason}, {epoch} epochs, {elapsed:.0f}s; train {train_time:.0f}s, val {val_time:.0f}s) "
        f"best epoch {best_epoch} PESQ-WB {best_pesq:.4f}")
    log_file.close()


if __name__ == "__main__":
    sys.exit(main())
