"""근사 중복 7단계 — **하단 문자 띠와 본문을 따로 잰다.** 남은 오판 계열을 겨눈 열이다.

무엇이 안 잡혔나. 인쇄된 필름 식별 문자열이 같고 겹쳐 찍힌 노출 번호만 다른 쌍은 지금까지의 어떤 열로도
갈리지 않았다 — 봉우리도 뾰족하고 블록도 맞는다. 같은 필름을 다시 찍었으니 당연하다.

무엇이 다른가. **문자는 필름에 인쇄돼 있어 프레임이 달라도 같고, 본문(용접부)은 프레임이 다르면 달라진다.**
그래서 두 영역을 따로 재면 갈린다. 같은 사진이면 둘 다 맞고, 같은 필름의 다른 프레임이면 문자만 맞는다.

영역은 안쪽 상자의 아래 20%(문자 띠)와 나머지 80%(본문)다. 둘 다 정합 뒤 대역 통과 상관으로 잰다.

사용: python nd_stage7_textband.py <feat3 csv 이름> [워커]   → feat7_<같은 꼬리>.csv
      워커 기본 1. 올리기 전에 여유 메모리를 먼저 본다.
출력 열
  txt_corr    하단 문자 띠의 상관
  body_corr   본문의 상관
  txt_gap     txt_corr - body_corr. **클수록 "문자만 같다"** 는 뜻이다
  txt_energy  문자 띠의 구조 에너지비(본문 대비) — 띠에 아무것도 없으면 txt_corr 은 뜻이 없다
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
BAND = 0.20                  # 안쪽 상자 아래에서 이만큼이 문자 띠다
HEAD = ["txt_corr", "body_corr", "txt_gap", "txt_energy"]
FLUSH = 200          # 이만큼마다 디스크에 내린다
csv.field_size_limit(10_000_000)


def _done_pairs(out: Path) -> set:
    """이미 계산해 적어 둔 쌍. 산출 CSV 자체가 진행 기록이다 — 따로 두면 둘이 어긋난다."""
    if not out.exists():
        return set()
    with out.open(encoding="utf-8", newline="") as fh:
        return {(r["image_id_a"], r["image_id_b"]) for r in csv.DictReader(fh)}


def _corr(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p - p.mean(), q - q.mean()
    den = float(np.sqrt((p * p).sum() * (q * q).sum()))
    return float((p * q).sum() / den) if den > 0 else 0.0


def compare(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> list:
    """정합 뒤 하단 띠와 본문의 상관. HEAD 순서. 정합이 수행되지 않는 범위면 판정하지 않는다."""
    h, w = a.shape
    dx, dy = dx // 2, dy // 2
    if abs(dx) > w // 3 or abs(dy) > h // 3:
        return [0.0, 0.0, 0.0, 0.0]
    ya0, ya1, xa0, xa1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    A, B = a[ya0:ya1, xa0:xa1], b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    cut = int(A.shape[0] * (1 - BAND))
    if cut <= 0 or cut >= A.shape[0]:
        return [0.0, 0.0, 0.0, 0.0]
    t = _corr(A[cut:], B[cut:])
    body = _corr(A[:cut], B[:cut])
    eb = float(A[:cut].std())
    en = float(A[cut:].std()) / eb if eb > 0 else 0.0
    return [round(t, 4), round(body, 4), round(t - body, 4), round(en, 3)]


def bandpass_in(interior: np.ndarray) -> np.ndarray:
    """안쪽 상자(그레이, float32) → 절반 해상도 대역 통과. 2f 와 같은 표현을 쓴다 — 시험은 이것과 compare 만 쓴다."""
    return blocks.bandpass(interior)


def features(t):
    ia, ib, dx, dy = t
    return compare(blocks.load(ia), blocks.load(ib), dx, dy)


def main() -> int:
    src = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = W / ("feat7_" + src.split("feat3", 1)[-1].lstrip("_d").lstrip("_"))
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
