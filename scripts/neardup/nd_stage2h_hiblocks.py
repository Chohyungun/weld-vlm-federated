"""근사 중복 2h단계 — **같은 필름의 다른 프레임**을 가려내는 고에너지 블록 특징.

왜 필요한가. 4×6 블록 일치(`blk_frac`)는 "필름 식별 문자만 같은 다른 그림" 은 걸러 내지만,
**같은 필름을 다시 찍은 다른 프레임** 은 못 거른다 — 하단 인쇄 문자열과 큰 명암이 같아 대부분의 블록이 맞고,
실제로 내용이 다른 곳은 구조가 강한 몇 블록뿐이라 비율로 보면 묻힌다. 과병합을 겨눈 검토에서
`rd_sharp` 0.38~0.64 · `blk_frac` 0.78~0.86 인 다른 프레임 셋을 그렇게 찾아냈다.

그래서 **비율이 아니라 최악의 블록** 을 본다. 격자를 8×12 로 잘게 쓰고, 자기 영상에서 구조가 강한
블록(자기 블록 에너지 중앙값의 1.5배 이상)만 골라 그 가운데 **상관의 최솟값** 을 적는다. 같은 사진이면
구조가 강한 곳일수록 잘 맞아야 한다. 다른 프레임이면 바로 그 자리가 음의 상관으로 튄다.

사용: python nd_stage2h_hiblocks.py <feat3 csv 이름> [워커]   → feat6_<같은 꼬리>.csv
      워커 기본 1. 올리기 전에 여유 메모리를 먼저 본다(하나가 약 0.7GB 를 쓴다).
출력 열
  hi_n            구조가 강한 블록 수 (최대 96)
  hi_min          그 블록들의 상관 최솟값
  hi_min_energy   최솟값을 낸 블록의 에너지비(자기 영상 중앙값 대비) — 강한 곳에서 어긋날수록 크다
  hi_badfrac      그 블록 가운데 상관 < 0.3 인 비율
"""
from __future__ import annotations

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_stage2f_blocks as blocks

W = blocks.W
GY, GX = 8, 12
HEAD = ["hi_n", "hi_min", "hi_min_energy", "hi_badfrac"]
ENERGY_FACTOR = 1.5          # 자기 영상 블록 에너지 중앙값의 몇 배 이상을 "구조가 강한" 으로 볼 것인가
BAD = 0.3
FLUSH = 200          # 이만큼마다 디스크에 내린다
csv.field_size_limit(10_000_000)


def _done_pairs(out: Path) -> set:
    """이미 계산해 적어 둔 쌍. 산출 CSV 자체가 진행 기록이다 — 따로 두면 둘이 어긋난다."""
    if not out.exists():
        return set()
    with out.open(encoding="utf-8", newline="") as fh:
        return {(r["image_id_a"], r["image_id_b"]) for r in csv.DictReader(fh)}


def compare(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> list:
    """두 대역 통과 영상을 (dx, dy)로 맞춘 뒤 8×12 격자에서 구조가 강한 블록만 본다. HEAD 순서."""
    h, w = a.shape
    dx, dy = dx // 2, dy // 2
    if abs(dx) > w // 3 or abs(dy) > h // 3:
        return [0, 0.0, 0.0, 0.0]                      # 정합이 수행되지 않는 범위 — 판정하지 않는다
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
    ma, mb = np.median(ea), np.median(eb)
    if ma <= 0 or mb <= 0:
        return [0, 0.0, 0.0, 0.0]
    ratio = np.minimum(ea / ma, eb / mb)               # 양쪽 모두에서 강한 블록만
    hi = ratio >= ENERGY_FACTOR
    if not hi.any():
        return [0, 0.0, 0.0, 0.0]
    c, r = cs[hi], ratio[hi]
    k = int(np.argmin(c))
    return [int(hi.sum()), round(float(c[k]), 4), round(float(r[k]), 3), round(float((c < BAD).mean()), 4)]


def features(t):
    ia, ib, dx, dy = t
    return compare(blocks.load(ia), blocks.load(ib), dx, dy)


def main() -> int:
    src = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = W / ("feat6_" + src.split("feat3", 1)[-1].lstrip("_d").lstrip("_"))
    rows = list(csv.DictReader((W / src).open(encoding="utf-8")))
    done = _done_pairs(out)
    rows = [r for r in rows if (r["image_id_a"], r["image_id_b"]) not in done]
    todo = [(r["image_id_a"], r["image_id_b"], int(float(r["rd_dx"])), int(float(r["rd_dy"]))) for r in rows]
    print(f"[{src}] 남은 {len(rows):,}쌍 (이미 적힌 {len(done):,} 건너뜀) → {out.name} · 워커 {workers}", flush=True)
    if not rows:
        print("할 일이 없다", flush=True)
        return 0
    t0 = time.perf_counter()
    buf: list[list] = []
    with Pool(workers, initializer=blocks._init) as pool:
        if not done:
            with out.open("w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow(["image_id_a", "image_id_b", *HEAD])
        for k, f in enumerate(pool.imap(features, todo, chunksize=16)):
            buf.append([rows[k]["image_id_a"], rows[k]["image_id_b"], *f])
            if len(buf) >= FLUSH:
                with out.open("a", encoding="utf-8", newline="") as fh:
                    csv.writer(fh).writerows(buf)
                buf.clear()
                print(f"  {k + 1:,}/{len(rows):,} {time.perf_counter() - t0:.0f}s", flush=True)
    if buf:
        with out.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows(buf)
    print(f"완료 {len(rows):,}쌍 {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
