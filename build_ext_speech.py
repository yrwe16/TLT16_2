"""외부 clean 음성(예: Zeroth Korean FLAC)을 16 kHz 2초 구간 배열로 만들어 cache/에 저장한다.

- 하위 폴더의 *.flac / *.wav 를 모두 읽는다(16 kHz가 아니면 오류).
- 파일마다 겹치지 않는 --segment 표본 구간으로 자르고 나머지는 버린다.
- 거의 무음인 구간(RMS가 파일 최대 구간 RMS보다 --silence-db 이상 낮음)은 버린다.
- 결과: cache/<name>_<segment>.npy (float16, [M, segment]) 와 같은 이름의 .json(출처·개수).
학습 데이터(수강생 녹음)는 쓰지 않으며, 생성물은 .gitignore(cache/, *.npy)로 커밋되지 않는다.

예) python build_ext_speech.py --src external/zeroth --name zeroth
"""
import argparse
import json
from pathlib import Path

from src.audio import SAMPLE_RATE  # DLL 경로 설정을 위해 가장 먼저 import

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--segment", type=int, default=32000)
    p.add_argument("--silence-db", type=float, default=30.0)
    p.add_argument("--max-hours", type=float, default=None)
    args = p.parse_args()

    files = sorted([*args.src.rglob("*.flac"), *args.src.rglob("*.wav")])
    if not files:
        raise SystemExit(f"오디오 파일이 없습니다: {args.src}")
    seg, chunks, kept_files, dropped = args.segment, [], 0, 0
    limit = None if args.max_hours is None else int(args.max_hours * 3600 * SAMPLE_RATE / seg)
    for i, f in enumerate(files):
        x, rate = sf.read(str(f), dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            raise ValueError(f"16 kHz가 아닙니다: {f} ({rate})")
        x = x.mean(1)
        x = x - x.mean()
        n = len(x) // seg
        if n == 0:
            continue
        c = x[: n * seg].reshape(n, seg)
        rms = np.sqrt((c ** 2).mean(1) + 1e-12)
        keep = 20 * np.log10(rms / rms.max()) > -args.silence_db
        keep &= rms > 1e-4
        dropped += int((~keep).sum())
        if keep.any():
            chunks.append(c[keep].astype(np.float16))
            kept_files += 1
        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{len(files)} files, {sum(len(c) for c in chunks)} chunks", flush=True)
        if limit and sum(len(c) for c in chunks) >= limit:
            break
    data = np.concatenate(chunks)[:limit]
    out = REPO / "cache" / f"{args.name}_{seg}.npy"
    out.parent.mkdir(exist_ok=True)
    np.save(out, data)
    meta = {"source": str(args.src), "files_scanned": i + 1, "files_used": kept_files, "chunks": len(data),
            "dropped_silent": dropped, "hours": len(data) * seg / SAMPLE_RATE / 3600, "segment": seg}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
