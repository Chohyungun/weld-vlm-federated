"""근사 중복 2d단계 — 정밀 특징. "같은 방사선 사진인가" 를 밝기 변환·작은 이동·압축 이력과 무관하게 본다.

사용: python nd_stage2d_refine.py <tag> [workers]    입력 feat2_<tag>.csv 에서 가능성이 있는 쌍만 골라 feat3_<tag>.csv 를 쓴다.

특징 (전부 안쪽 상자, 절반 해상도에서 계산 · 이동은 전 해상도 px 로 적는다)
  rd_peak · rd_dx · rd_dy   행·열 평균을 뺀 대역 통과 영상의 정규화 교차상관 최댓값과 그 이동.
                            행·열 추세를 빼는 이유: 용접 비드의 수평 경계는 다른 영상끼리도 같은 높이에 있어 상관을 부풀린다.
  rd_side · rd_sharp        봉우리 주변(±8)을 뺀 나머지의 99.9 분위와, 봉우리와의 차. 같은 그림이면 한 점에서만 맞는다(뾰족하다).
  ov                        정합 뒤 겹치는 면적 비율.
                            **주의 — 이동이 너무 크면(절반 해상도 폭·높이의 1/3, 전 해상도로 |dx|>358 · |dy|>200)
                            정합을 포기하고 `ov` 가 1.0 으로 되돌아간다.** 겹침이 좋아서가 아니라 재지 않았다는 뜻이다.
                            그 범위에서는 `ov`·`reg_*` 를 증거로 쓸 수 없다. 규칙에 쓰려면 이동 상한을 함께 걸어라.
  reg_corr                  정합 뒤 밝기 상관(선형 밝기 변환에 불변)
  reg_mad_tone              B 의 밝기 분포를 A 에 맞춘 뒤(단조 변환 제거)의 평균 절대차 — 그레이 레벨
  reg_fine·reg_mid·reg_coarse  정합 뒤 세 규모(약 1.5~3px · 3~16px · 16~64px)의 대역 통과 상관

`prepare()` 와 `compare()` 는 파일을 읽지 않는 순수 함수다 — 합성 영상으로 시험한다(tests/test_neardup_features.py).
"""
from __future__ import annotations

import csv
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
X0, Y0, X1, Y1 = 102, 58, 1178, 662            # scripts/run_border_mask.py 의 interior_box(1280, 720) — 테두리는 상수 채움이다
csv.field_size_limit(10_000_000)
REL: dict[str, str] = {}

HEAD = ["rd_peak", "rd_dx", "rd_dy", "rd_side", "rd_sharp", "ov", "reg_corr", "reg_mad_tone", "reg_fine", "reg_mid", "reg_coarse"]


class Prepared(NamedTuple):
    half: np.ndarray        # 절반 해상도 밝기
    fine: np.ndarray        # 가우시안 차 0.7−1.5
    mid: np.ndarray         # 가우시안 차 1.5−8
    coarse: np.ndarray      # 가우시안 차 8−32
    spectrum: np.ndarray    # 행·열 평균을 뺀 mid 의 rfft2
    norm: float             # 그 영상의 L2 노름
    shape: tuple[int, int]


def _init() -> None:
    cv2.setNumThreads(1)
    with (ROOT / "data/interim/manifest_v1/manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            REL[r["image_id"]] = r["rel_path"]


def _dog(h: np.ndarray, s1: float, s2: float) -> np.ndarray:
    return cv2.GaussianBlur(h, (0, 0), s1) - cv2.GaussianBlur(h, (0, 0), s2)


def prepare(interior: np.ndarray) -> Prepared:
    """안쪽 상자의 전 해상도 밝기(float32) → 비교에 쓰는 표현들."""
    a = np.asarray(interior, dtype=np.float32)
    h = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
    mid = _dog(h, 1.5, 8.0)
    rd = mid - mid.mean(axis=1, keepdims=True)
    rd = rd - rd.mean(axis=0, keepdims=True)
    return Prepared(h, _dog(h, 0.7, 1.5), mid, _dog(h, 8.0, 32.0), np.fft.rfft2(rd), float(np.sqrt((rd * rd).sum())), rd.shape)


@lru_cache(maxsize=128)
def load(iid: str) -> Prepared:
    with Image.open(ROOT / REL[iid]) as im:
        return prepare(np.asarray(im.convert("L"), dtype=np.float32)[Y0:Y1, X0:X1])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _overlap(a: np.ndarray, b: np.ndarray, dx: int, dy: int):
    """a[y, x] 와 b[y - dy, x - dx] 가 같은 자리가 되게 자른다(이동 dx, dy 는 절반 해상도 px)."""
    h, w = a.shape
    ya0, ya1 = max(0, dy), min(h, h + dy)
    xa0, xa1 = max(0, dx), min(w, w + dx)
    return a[ya0:ya1, xa0:xa1], b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]


def compare(pa: Prepared, pb: Prepared) -> list:
    """두 표현의 정밀 특징. 반환 순서는 HEAD 와 같다."""
    # 구조 에너지가 사실상 0(행·열 추세를 뺀 대역 통과 표준편차 < 0.001)이면 판정하지 않는다 — 평탄한 영상끼리는
    # 부동소수 잔차만으로 봉우리가 1 이 된다. 실데이터의 최솟값은 0.037 이라 이 가드는 실제 쌍에는 걸리지 않는다.
    floor = 1e-3 * float(np.sqrt(pa.shape[0] * pa.shape[1]))
    if pa.norm <= floor or pb.norm <= floor:
        return [0.0, 0, 0, 0.0, 0.0, 1.0, 0.0, 99.0, 0.0, 0.0, 0.0]
    xc = np.fft.irfft2(pa.spectrum * np.conj(pb.spectrum), s=pa.shape) / (pa.norm * pb.norm)
    hh, ww = xc.shape
    py, px = divmod(int(np.argmax(xc)), ww)
    dy = py if py <= hh // 2 else py - hh
    dx = px if px <= ww // 2 else px - ww
    peak = float(xc[py, px])
    m = np.ones_like(xc, dtype=bool)
    for yy in range(py - 8, py + 9):
        for xx in range(px - 8, px + 9):
            m[yy % hh, xx % ww] = False
    side = float(np.percentile(xc[m], 99.9))
    # 겹침이 너무 작으면 정합하지 않는다
    dx_r, dy_r = (0, 0) if abs(dx) > ww // 3 or abs(dy) > hh // 3 else (dx, dy)
    a, b = _overlap(pa.half, pb.half, dx_r, dy_r)
    ov = a.size / pa.half.size
    q = np.linspace(0, 100, 257)                       # 단조 밝기 변환 제거: B 의 분위를 A 의 분위에 맞춘다
    bm = np.interp(b, np.percentile(b, q), np.percentile(a, q))
    out = [round(peak, 5), dx * 2, dy * 2, round(side, 5), round(peak - side, 5), round(ov, 4),
           round(_corr(a, b), 5), round(float(np.abs(a - bm).mean()), 3)]
    for xa, xb in ((pa.fine, pb.fine), (pa.mid, pb.mid), (pa.coarse, pb.coarse)):
        p, r = _overlap(xa, xb, dx_r, dy_r)
        out.append(round(_corr(p, r), 5))
    return out


def features(pair) -> list:
    return compare(load(pair[0]), load(pair[1]))


def main() -> int:
    tag = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with (W / f"feat2_{tag}.csv").open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        head = rd.fieldnames
        rows = [r for r in rd if r.get("bp_std_b")]
    sel = [r for r in rows if float(r["bp_peak"]) >= 0.30 or float(r["grad_corr"]) >= 0.55
           or float(r["corr"]) >= 0.97 or float(r["thumb_rmsd"]) <= 4.0]
    # 걸러진 쪽에 같은 그림이 없는지 보려고 무작위 대조 표본도 같이 계산한다
    chosen = {id(r) for r in sel}
    rest = [r for r in rows if id(r) not in chosen]
    rng = np.random.default_rng(20260921)
    ctrl = [rest[i] for i in rng.choice(len(rest), size=min(600, len(rest)), replace=False)] if rest else []
    print(f"[{tag}] 전체 {len(rows):,}쌍 → 정밀 대상 {len(sel):,}쌍 + 걸러진 쪽 대조 표본 {len(ctrl):,}쌍", flush=True)
    todo = [(r, 1) for r in sel] + [(r, 0) for r in ctrl]
    todo.sort(key=lambda t: (t[0]["image_id_a"], t[0]["image_id_b"]))
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init) as pool, (W / f"feat3_{tag}.csv").open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow([*head, "selected", *HEAD])
        for k, f in enumerate(pool.imap(features, [(t[0]["image_id_a"], t[0]["image_id_b"]) for t in todo], chunksize=16)):
            r, s = todo[k]
            w.writerow([r[c] for c in head] + [s] + f)
            if k % 1000 == 0:
                print(f"  {k:,}/{len(todo):,} {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"완료 {len(todo):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
