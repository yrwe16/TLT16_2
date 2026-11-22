"""MP-SENet 미세조정: 공개 가중치(기본 DNS)에서 시작해 우리 학습셋으로 이어서 학습한다.

손실은 원본 train.py와 같다: 크기 0.9, 위상 0.3, 복소 0.1, 일관성 0.1, 시간 0.2, 메트릭 0.05.
- --remix-prob: 배치 샘플마다 이 확률로 동적 리믹싱(학습셋 clean + 다른 학습 쌍의 잡음(noisy-clean),
  SNR U[--snr-low, --snr-high]). 나머지는 원래 쌍. 검증셋은 리믹싱에 쓰지 않는다.
- --ext-speech: build_ext_speech.py로 만든 외부 clean 음성 배열(.npy). 배치 샘플마다 --ext-prob 확률로
  외부 clean + 우리 학습셋 잡음(noisy-clean)을 SNR U[--snr-low, --snr-high]로 섞어 쓴다(리믹싱보다 먼저 결정).
- --metric-weight 0 이면 판별기를 쓰지 않는다.
- --sisdr-weight: 음의 SI-SDR(dB) 손실 가중치(0이면 끔).
- --pesq-loss-weight: 미분 가능한 PESQ 손실(src/torch_pesq, torch-pesq 이식본) 가중치(0이면 끔).
  무음 등으로 값이 유한하지 않은 샘플은 그 배치에서 제외한다.
매 epoch 끝에 검증 164개 전체(3초, batch 1)의 PESQ-WB를 계산해 최고 가중치를 runs/<name>/g_best 에 저장.

예) python finetune_mpsenet.py --run-name ft_dns_remix --remix-prob 0.5
"""
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

from src.audio import pesq_pair_or_nan, pesq_wb_many  # DLL 경로 설정을 위해 가장 먼저 import

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import PairedSpeech
from src.discriminator import MetricDiscriminator
from src.mpsenet import MPSENet

REPO = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(r"C:\deep_learning_noise_reduction_challenge")  # 개인 PC 경로: --data-root로 변경


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--init", type=Path, default=REPO / "external/MP-SENet/best_ckpt/g_best_dns")
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--segment", type=int, default=32000, help="학습 구간 길이(표본), 원본 2초")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-decay", type=float, default=0.98, help="epoch마다 곱함")
    p.add_argument("--remix-prob", type=float, default=0.0)
    p.add_argument("--ext-speech", type=Path, default=None)
    p.add_argument("--ext-prob", type=float, default=0.0)
    p.add_argument("--steps-per-epoch", type=int, default=None, help="기본: 학습 쌍 수 / batch")
    p.add_argument("--snr-low", type=float, default=-5.0)
    p.add_argument("--snr-high", type=float, default=20.0)
    p.add_argument("--metric-weight", type=float, default=0.05)
    p.add_argument("--sisdr-weight", type=float, default=0.0)
    p.add_argument("--pesq-loss-weight", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=8, help="검증 PESQ 개선 없이 기다릴 epoch")
    p.add_argument("--pesq-workers", type=int, default=12)
    return p.parse_args()


def neg_si_sdr(ref, est, eps=1e-8):
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    target = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps) * ref
    ratio = target.pow(2).sum(-1) / ((est - target).pow(2).sum(-1) + eps)
    return -10 * torch.log10(ratio + eps).mean()


def anti_wrap(x):
    return torch.abs(x - torch.round(x / (2 * math.pi)) * 2 * math.pi)


def phase_loss(pha_r, pha_g):
    ip = anti_wrap(pha_r - pha_g).mean()
    gd = anti_wrap(torch.diff(pha_r, dim=1) - torch.diff(pha_g, dim=1)).mean()
    iaf = anti_wrap(torch.diff(pha_r, dim=2) - torch.diff(pha_g, dim=2)).mean()
    return ip + gd + iaf


class Sampler:
    """GPU에 올린 학습 파형에서 2초 구간 배치를 만든다(원래 쌍 또는 동적 리믹싱)."""

    def __init__(self, waveforms, device, args, generator):
        self.noisy = waveforms[:, 0].to(device)
        self.clean = waveforms[:, 1].to(device)
        self.noise = self.noisy - self.clean
        self.n, self.length = self.clean.shape
        self.args, self.g, self.device = args, generator, device
        self.ext = None
        if args.ext_speech is not None and args.ext_prob > 0:
            ext = np.load(args.ext_speech, mmap_mode="r")
            if ext.shape[1] < args.segment:
                raise ValueError(f"외부 음성 구간({ext.shape[1]})이 --segment보다 짧습니다.")
            self.ext = torch.from_numpy(np.ascontiguousarray(ext)).pin_memory()  # CPU float16 [M, L]
            self.ext_g = torch.Generator().manual_seed(args.seed + 1)

    def _ext_clean(self, B):
        rows = torch.randint(0, len(self.ext), (B,), generator=self.ext_g)
        L = self.ext.shape[1]
        start = int(torch.randint(0, L - self.args.segment + 1, (1,), generator=self.ext_g))
        x = self.ext[rows, start:start + self.args.segment].to(self.device, non_blocking=True).float()
        return x

    def _crop(self, source, idx, start):
        offsets = start[:, None] + torch.arange(self.args.segment, device=self.device)[None, :]
        return torch.gather(source[idx], 1, offsets)

    def batch(self):
        a, B, dev = self.args, self.args.batch_size, self.device
        rand = lambda *s: torch.rand(*s, generator=self.g, device=dev)
        idx = torch.randint(0, self.n, (B,), generator=self.g, device=dev)
        start = (rand(B) * (self.length - a.segment + 1)).long()
        clean = self._crop(self.clean, idx, start)
        noisy = self._crop(self.noisy, idx, start)
        use_ext = torch.zeros(B, dtype=torch.bool, device=dev)
        if self.ext is not None:
            use_ext = rand(B) < a.ext_prob
            if use_ext.any():
                clean = torch.where(use_ext[:, None], self._ext_clean(B), clean)
        remix = (rand(B) < a.remix_prob) | use_ext
        if remix.any():
            n_idx = torch.randint(0, self.n, (B,), generator=self.g, device=dev)
            n_start = (rand(B) * (self.length - a.segment + 1)).long()
            noise = self._crop(self.noise, n_idx, n_start)
            noise = torch.where(rand(B, 1) < 0.5, noise, noise.flip(-1))  # 시간 뒤집기
            snr = a.snr_low + (a.snr_high - a.snr_low) * rand(B, 1)
            p_clean = clean.pow(2).mean(-1, keepdim=True)
            p_noise = noise.pow(2).mean(-1, keepdim=True).clamp_min(1e-10)
            scale = torch.sqrt(p_clean / (p_noise * 10 ** (snr / 10)))
            remixed = clean + scale * noise
            ok = remix & (p_clean.squeeze(-1) > 1e-6)  # 거의 무음 구간은 원래 쌍 유지
            noisy = torch.where(ok[:, None], remixed, noisy)
            bad = use_ext & ~ok  # 외부 clean이 무음이면 원래 쌍으로 되돌린다
            if bad.any():
                clean = torch.where(bad[:, None], self._crop(self.clean, idx, start), clean)
                noisy = torch.where(bad[:, None], self._crop(self.noisy, idx, start), noisy)
        norm = torch.sqrt(a.segment / noisy.pow(2).sum(-1, keepdim=True).clamp_min(1e-8))
        return clean * norm, noisy * norm


@torch.inference_mode()
def validate(model, noisy, clean, device, pool):
    model.eval()
    outs = [model.enhance(noisy[i:i + 1].to(device)).float().cpu() for i in range(len(noisy))]
    enhanced = torch.cat(outs).numpy()
    if not np.all(np.isfinite(enhanced)):
        raise FloatingPointError("검증 출력에 NaN/Inf")
    scores = list(pool.map(pesq_pair_or_nan, list(zip(clean.numpy(), enhanced)), chunksize=4))
    if np.isnan(scores).any():
        raise RuntimeError("검증 PESQ 계산 실패")
    model.train()
    return float(np.mean(scores))


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    run_dir = REPO / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    train_set = PairedSpeech(args.data_root, "train", REPO / "cache")
    val_set = PairedSpeech(args.data_root, "validation", REPO / "cache")
    val_noisy, val_clean = val_set.waveforms[:, 0], val_set.waveforms[:, 1]
    g = torch.Generator(device=device).manual_seed(args.seed)
    sampler = Sampler(train_set.waveforms, device, args, g)

    model = MPSENet().load_original_checkpoint(args.init, map_location=device).to(device)
    gen = model.generator
    disc = MetricDiscriminator().to(device)
    opt_g = torch.optim.AdamW(gen.parameters(), args.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), args.lr * 10, betas=(0.8, 0.99))
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, args.lr_decay)
    steps_per_epoch = args.steps_per_epoch or len(train_set) // args.batch_size
    pool = ProcessPoolExecutor(args.pesq_workers)
    use_disc = args.metric_weight > 0
    pesq_loss_fn = None
    if args.pesq_loss_weight > 0:
        from src.torch_pesq import PesqLoss
        pesq_loss_fn = PesqLoss(1.0, sample_rate=16000).to(device)

    log(f"start | init {args.init.name} | args {vars(args)}")
    best = validate(model, val_noisy, val_clean, device, pool)
    torch.save({"generator": gen.state_dict()}, run_dir / "g_best")
    best_epoch, stale, history = 0, 0, [{"epoch": 0, "val_pesq_wb": best}]
    log(f"epoch   0 | val PESQ {best:.4f} (초기 가중치)")
    started = perf_counter()
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = perf_counter()
        sums = {"gen": 0.0, "disc": 0.0, "mag": 0.0, "pha": 0.0, "time": 0.0, "sisdr": 0.0, "pesq": 0.0}
        for step in range(steps_per_epoch):
            clean, noisy = sampler.batch()
            clean_mag, clean_pha = model.mag_pha(clean)
            clean_com = torch.stack((clean_mag * torch.cos(clean_pha), clean_mag * torch.sin(clean_pha)), -1)
            noisy_mag, noisy_pha = model.mag_pha(noisy)
            mag_g, pha_g, com_g = gen(noisy_mag, noisy_pha)
            audio_g = model.to_wave(mag_g, pha_g, clean.shape[-1])
            mag_hat, pha_hat = model.mag_pha(audio_g)
            com_hat = torch.stack((mag_hat * torch.cos(pha_hat), mag_hat * torch.sin(pha_hat)), -1)

            if use_disc:
                pairs = list(zip(clean.detach().cpu().numpy(), audio_g.detach().cpu().numpy()))
                scores = np.array(list(pool.map(pesq_pair_or_nan, pairs)))
                opt_d.zero_grad()
                one = torch.ones(args.batch_size, device=device)
                loss_d = F.mse_loss(disc(clean_mag, clean_mag).flatten(), one)
                if not np.isnan(scores).any():
                    target = torch.tensor((scores - 1) / 3.5, dtype=torch.float32, device=device)
                    loss_d = loss_d + F.mse_loss(disc(clean_mag, mag_hat.detach()).flatten(), target)
                loss_d.backward()
                opt_d.step()
                sums["disc"] += loss_d.item()

            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_pha = phase_loss(clean_pha, pha_g)
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_stft = F.mse_loss(com_g, com_hat) * 2
            loss_time = F.l1_loss(clean, audio_g)
            loss = loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1 + loss_stft * 0.1 + loss_time * 0.2
            if args.sisdr_weight > 0:
                loss_sisdr = neg_si_sdr(clean, audio_g)
                loss = loss + args.sisdr_weight * loss_sisdr
                sums["sisdr"] += loss_sisdr.item()
            if pesq_loss_fn is not None:
                per = pesq_loss_fn(clean, audio_g).float()
                finite = torch.isfinite(per)
                if finite.any():
                    loss_pesq = per[finite].mean()
                    loss = loss + args.pesq_loss_weight * loss_pesq
                    sums["pesq"] += loss_pesq.item()
            if use_disc:
                loss_metric = F.mse_loss(disc(clean_mag, mag_hat).flatten(), torch.ones(args.batch_size, device=device))
                loss = loss + args.metric_weight * loss_metric
            if not torch.isfinite(loss):
                log(f"loss 발산: epoch {epoch} step {step} loss {loss.item()} → 중단")
                raise SystemExit(1)
            opt_g.zero_grad()
            loss.backward()
            opt_g.step()
            for k, v in (("gen", loss), ("mag", loss_mag), ("pha", loss_pha), ("time", loss_time)):
                sums[k] += v.item()
        sched_g.step()
        train_s = perf_counter() - t0
        val = validate(model, val_noisy, val_clean, device, pool)
        if val > best:
            best, best_epoch, stale = val, epoch, 0
            torch.save({"generator": gen.state_dict()}, run_dir / "g_best")
        else:
            stale += 1
        avg = {k: v / steps_per_epoch for k, v in sums.items()}
        history.append({"epoch": epoch, "val_pesq_wb": val, **{f"train_{k}": v for k, v in avg.items()},
                        "lr": opt_g.param_groups[0]["lr"], "elapsed": perf_counter() - started})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(f"epoch {epoch:3d} | gen {avg['gen']:.4f} mag {avg['mag']:.4f} pha {avg['pha']:.3f} disc {avg['disc']:.4f} sisdr {avg['sisdr']:.2f} pesqL {avg['pesq']:.3f} | "
            f"val PESQ {val:.4f} | best {best:.4f} (ep {best_epoch}) | train {train_s:.0f}s total {perf_counter() - started:.0f}s")
        if stale >= args.patience:
            log("조기 종료")
            break

    summary = {"run_name": args.run_name, "best_epoch": best_epoch, "best_validation_pesq_wb": best,
               "initial_validation_pesq_wb": history[0]["val_pesq_wb"], "epochs_completed": epoch,
               "elapsed_seconds": perf_counter() - started, "args": {k: str(v) for k, v in vars(args).items()}}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done | best epoch {best_epoch} PESQ-WB {best:.4f}")
    pool.shutdown()


if __name__ == "__main__":
    main()
