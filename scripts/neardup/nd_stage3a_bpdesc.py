"""근사 중복 3a단계 — 세 번째 후보 생성기용 기술자. 밝기 변환·화소 잡음에 둔감하게 "큰 구조" 만 남긴다.

안쪽 상자 → 절반 해상도 → 가우시안 차(σ 4−16, 전 해상도로 약 8~32px 규모) → 행·열 평균 제거 → 64x36 으로 면적 평균.
출력: _workspace/.../bpdesc_64x36.npy (float32, 영상 순서는 thumbs_ids.txt 와 같다).
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
X0, Y0, X1, Y1 = 102, 58, 1178, 662
csv.field_size_limit(10_000_000)


def desc(rel: str) -> np.ndarray:
    cv2.setNumThreads(1)
    with Image.open(ROOT / rel) as im:
        im.draft("L", (640, 360))                      # JPEG DCT 단계 1/2 축소 — 여기서 쓰는 규모에는 충분하다
        a = np.asarray(im.convert("L"), dtype=np.float32)
    sy, sx = a.shape[0] / 720.0, a.shape[1] / 1280.0
    a = a[round(Y0 * sy):round(Y1 * sy), round(X0 * sx):round(X1 * sx)]
    bp = cv2.GaussianBlur(a, (0, 0), 4.0 * sx * 2) - cv2.GaussianBlur(a, (0, 0), 16.0 * sx * 2)
    bp -= bp.mean(axis=1, keepdims=True)
    bp -= bp.mean(axis=0, keepdims=True)
    return cv2.resize(bp, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def main() -> int:
    ids = (W / "thumbs_ids.txt").read_text(encoding="utf-8").split("\n")[:-1]
    rel = {}
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rel[r["image_id"]] = r["rel_path"]
    out = np.zeros((len(ids), 36, 64), dtype=np.float32)
    t0 = time.perf_counter()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for i, d in enumerate(pool.imap(desc, [rel[k] for k in ids], chunksize=64)):
            out[i] = d
            if i % 10000 == 0:
                print(f"  {i:,}/{len(ids):,} {time.perf_counter() - t0:.0f}s", flush=True)
    np.save(W / "bpdesc_64x36.npy", out)
    print(f"완료 {len(ids):,}장 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
