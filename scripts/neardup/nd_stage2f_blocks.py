"""근사 중복 2f단계 — 과병합 안전장치. 정합 봉우리는 영상의 일부(필름 식별 문자·표식)만 같아도 뾰족해질 수 있다.
같은 그림이면 **전역에서** 맞아야 한다 → 정합 뒤 4x6 격자 블록별 대역 통과 상관을 본다.

사용: python nd_stage2f_blocks.py <tag> [workers] [입력 csv 이름]   입력 feat3_<tag>.csv → 출력 feat5_<tag>.csv
      (train·val 안쪽은 2d′ 가 feat3d_within_trainval.csv 를 쓴다 — 세 번째 인자로 그 이름을 준다)
  blk_n        구조 에너지가 충분한(양쪽 모두 블록 표준편차가 영상 중앙값의 절반 이상) 블록 수 (최대 24)
  blk_frac     그 블록 가운데 상관 >= 0.3 인 비율
  blk_med      그 블록들의 상관 중앙값
  blk_min      그 블록들의 상관 최솟값
"""
from __future__ import annotations

import csv
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
X0, Y0, X1, Y1 = 102, 58, 1178, 662
GY, GX = 4, 6
csv.field_size_limit(10_000_000)
REL: dict[str, str] = {}


def _init() -> None:
    cv2.setNumThreads(1)
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            REL[r["image_id"]] = r["rel_path"]


def bandpass(interior: np.ndarray) -> np.ndarray:
    """안쪽 상자(그레이, float32) → 절반 해상도의 대역 통과 영상. 순수 함수 — 시험은 이것과 compare 만 쓴다."""
    h = cv2.resize(interior, (interior.shape[1] // 2, interior.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(h, (0, 0), 1.5) - cv2.GaussianBlur(h, (0, 0), 8.0)


@lru_cache(maxsize=128)
def load(iid: str) -> np.ndarray:
    with Image.open(ROOT / REL[iid]) as im:
        a = np.asarray(im.convert("L"), dtype=np.float32)[Y0:Y1, X0:X1]
    return bandpass(a)


def features(t):
    ia, ib, dx, dy = t
    return compare(load(ia), load(ib), dx, dy)


def compare(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> list:
    """두 대역 통과 영상을 (dx, dy)(전 해상도 px, 정밀 특징의 rd_dx·rd_dy)로 맞춘 뒤 4x6 블록별 상관. HEAD 순서."""
    h, w = a.shape
    dx, dy = dx // 2, dy // 2
    if abs(dx) > w // 3 or abs(dy) > h // 3:
        dx, dy = 0, 0
    ya0, ya1, xa0, xa1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    A, B = a[ya0:ya1, xa0:xa1], b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    H, Wd = A.shape
    cs, ea, eb = [], [], []
    for gy in range(GY):
        for gx in range(GX):
            P = A[gy * H // GY:(gy + 1) * H // GY, gx * Wd // GX:(gx + 1) * Wd // GX]
            Q = B[gy * H // GY:(gy + 1) * H // GY, gx * Wd // GX:(gx + 1) * Wd // GX]
            p, q = P - P.mean(), Q - Q.mean()
            den = float(np.sqrt((p * p).sum() * (q * q).sum()))
            cs.append(float((p * q).sum() / den) if den > 0 else 0.0)
            ea.append(float(P.std()))
            eb.append(float(Q.std()))
    cs, ea, eb = np.array(cs), np.array(ea), np.array(eb)
    ok = (ea >= 0.5 * np.median(ea)) & (eb >= 0.5 * np.median(eb))
    if not ok.any():
        return [0, 0.0, 0.0, 0.0]
    c = cs[ok]
    return [int(ok.sum()), round(float((c >= 0.3).mean()), 4), round(float(np.median(c)), 4), round(float(c.min()), 4)]


HEAD = ["blk_n", "blk_frac", "blk_med", "blk_min"]


def main() -> int:
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    src = sys.argv[3] if len(sys.argv) > 3 else f"feat3_{tag}.csv"
    rows = list(csv.DictReader((W / src).open(encoding="utf-8")))
    todo = [(r["image_id_a"], r["image_id_b"], int(float(r["rd_dx"])), int(float(r["rd_dy"]))) for r in rows]
    print(f"[{tag}] {len(rows):,}쌍", flush=True)
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init) as pool, (W / f"feat5_{tag}.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["image_id_a", "image_id_b"] + HEAD)
        for k, f in enumerate(pool.imap(features, todo, chunksize=16)):
            w.writerow([rows[k]["image_id_a"], rows[k]["image_id_b"]] + f)
            if k % 3000 == 0:
                print(f"  {k:,}/{len(rows):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(rows):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
