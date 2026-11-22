"""여러 MP-SENet 체크포인트의 가중치를 똑같은 비율로 평균해 하나로 저장한다(checkpoint averaging).

예) 제출 가중치 만들기: EMA 실험의 8~18 epoch EMA 가중치 11개 평균
    python average_checkpoints.py --out weights/mpsenet_final.pt runs/ft_pesq_ema/g_ep{08..18}_ema
"""
import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("inputs", type=Path, nargs="+")
    args = p.parse_args()
    states = []
    for path in args.inputs:
        s = torch.load(path, map_location="cpu", weights_only=True)
        states.append(s["generator"] if "generator" in s else s)
    avg = {k: (sum(s[k].float() for s in states) / len(states)) if v.is_floating_point() else v
           for k, v in states[0].items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": avg}, args.out)
    print(f"{len(states)}개 평균 → {args.out}")


if __name__ == "__main__":
    main()
