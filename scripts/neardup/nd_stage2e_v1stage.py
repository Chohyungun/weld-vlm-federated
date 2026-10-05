"""근사 중복 2e단계 — 히스토그램 정합 **이전** 판(tiles_v1)에서의 차이. 정합 판은 저대비 영상의 작은 차이를 3~4배로 키운다.

사용: python nd_stage2e_v1stage.py <tag> [workers]   입력 feat3_<tag>.csv(rd_dx, rd_dy 포함) → 출력 feat4_<tag>.csv
feat3 의 이동(rd_dx, rd_dy)으로 정합한 뒤, 안쪽 상자의 겹치는 영역에서 잰다.
  v1_mad · v1_corr              정합 없이(이동 0) 잰 값
  v1_reg_mad · v1_reg_corr      정합 뒤 값
  v1_reg_mad_gain               B 에 선형 밝기 보정(기울기·절편 최소제곱)을 한 뒤의 평균 절대차
  v1_std_a · v1_std_b           영상의 밝기 표준편차(저대비 여부)
"""
from __future__ import annotations

import csv
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
X0, Y0, X1, Y1 = 102, 58, 1178, 662
csv.field_size_limit(10_000_000)
REL: dict[str, str] = {}


def _init() -> None:
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            REL[r["image_id"]] = r["rel_path"].replace("tiles_v1_histmatch", "tiles_v1")


@lru_cache(maxsize=128)
def load(iid: str) -> np.ndarray:
    with Image.open(ROOT / REL[iid]) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)[Y0:Y1, X0:X1]


def _corr(a, b) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def features(t):
    ia, ib, dx, dy = t
    a, b = load(ia), load(ib)
    h, w = a.shape
    out = [round(float(np.abs(a - b).mean()), 3), round(_corr(a, b), 5)]
    if abs(dx) > w // 3 or abs(dy) > h // 3:
        dx, dy = 0, 0
    ya0, ya1, xa0, xa1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    A, B = a[ya0:ya1, xa0:xa1], b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    bm, am = B.mean(), A.mean()
    var = float(((B - bm) ** 2).mean())
    gain = float(((A - am) * (B - bm)).mean() / var) if var > 0 else 1.0
    Bg = (B - bm) * gain + am
    out += [round(float(np.abs(A - B).mean()), 3), round(_corr(A, B), 5), round(float(np.abs(A - Bg).mean()), 3),
            round(float(a.std()), 2), round(float(b.std()), 2)]
    return out


HEAD = ["v1_mad", "v1_corr", "v1_reg_mad", "v1_reg_corr", "v1_reg_mad_gain", "v1_std_a", "v1_std_b"]


def main() -> int:
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rows = list(csv.DictReader((W / f"feat3_{tag}.csv").open(encoding="utf-8")))
    todo = [(r["image_id_a"], r["image_id_b"], int(float(r["rd_dx"])), int(float(r["rd_dy"]))) for r in rows]
    print(f"[{tag}] {len(rows):,}쌍 · 판 tiles_v1", flush=True)
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init) as pool, (W / f"feat4_{tag}.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        head = list(rows[0].keys())
        w.writerow(head + HEAD)
        for k, f in enumerate(pool.imap(features, todo, chunksize=16)):
            w.writerow([rows[k][c] for c in head] + f)
            if k % 3000 == 0:
                print(f"  {k:,}/{len(rows):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(rows):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
