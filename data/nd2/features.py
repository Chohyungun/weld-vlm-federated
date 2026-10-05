"""규칙 입력 — 두 가설 정합과 **겹친 자리만으로** 잰 특징(미니스펙 2판 3-2). 순수 함수(파일을 읽지 않는다).

v1 의 정밀 단계(2d)는 원형 상관이라 이동이 폭의 1/3 을 넘으면 정합을 포기하고 겹침 1.0 을 냈고, 큰 이동은 반대 부호로
접혔다. 여기서는 선형 · 겹친 자리 정규화 상관(`ncc.ncc_map`)으로 모든 이동을 재고, 후보 생성기가 낸 이동과 그 접힌 짝을
가설로 두어 가설마다 작은 창 안에서 가장 잘 맞는 이동을 고른다.

표현은 v1 과 같다 — 안쪽 상자(전 해상도 1076 × 604)를 절반으로 줄여 가우시안 차(1.5 − 8)를 낸 대역 통과(블록 · 문자 띠 ·
2f · 7 과 같은 표현)와, 그 행 · 열 평균을 뺀 것(정합 — 2d 와 같은 표현). 이동은 절반 해상도 px 로 다루고 전 해상도로 적는다.
"""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from data.nd2.ncc import best_in_window, ncc_map

#: 거친 구조 기술자(64 × 36)의 한 칸 — 안쪽 상자 1076 × 604 를 64 × 36 으로 줄였다(가로 16.81 · 세로 16.78 px)
CELL_X_FULL, CELL_Y_FULL = 1076 / 64, 604 / 36
DESC_W, DESC_H = 64, 36
GY, GX = 4, 6
TEXT_BAND = 0.20
HEAD = ["hyp", "dx", "dy", "shift_px", "overlap", "ncc", "ncc_side", "ncc_sharp", "alt_ncc", "fold_gap",
        "reg_corr", "blk_n", "blk_frac", "blk_med", "blk_min", "txt_corr", "body_corr", "txt_gap", "txt_energy"]


class Prepared(NamedTuple):
    half: np.ndarray     # 절반 해상도 밝기
    mid: np.ndarray      # 대역 통과(가우시안 차 1.5 − 8)
    rd: np.ndarray       # mid 의 행 · 열 평균을 뺀 것


def prepare(interior: np.ndarray) -> Prepared:
    a = np.asarray(interior, dtype=np.float32)
    h = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
    mid = cv2.GaussianBlur(h, (0, 0), 1.5) - cv2.GaussianBlur(h, (0, 0), 8.0)
    rd = mid - mid.mean(axis=1, keepdims=True)
    rd = rd - rd.mean(axis=0, keepdims=True)
    return Prepared(h, mid, rd)


def overlap(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """a[y, x] 와 b[y - dy, x - dx] 가 같은 자리가 되게 자른다(상한 없음)."""
    h, w = a.shape
    ya0, ya1, xa0, xa1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    return a[ya0:ya1, xa0:xa1], b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]


def _corr(p: np.ndarray, q: np.ndarray) -> float:
    p = p - p.mean()
    q = q - q.mean()
    den = float(np.sqrt((p * p).sum() * (q * q).sum()))
    return float((p * q).sum() / den) if den > 0 else 0.0


def hypotheses(cell_dx: int, cell_dy: int) -> list[tuple[int, int]]:
    """생성기의 칸 이동과 그 접힌 짝(가로 64 · 세로 36 칸 고리) — 칸 단위, 중복 없이 정해진 순서."""
    xs = [cell_dx] + ([cell_dx - DESC_W] if cell_dx > 0 else [cell_dx + DESC_W] if cell_dx < 0 else [])
    ys = [cell_dy] + ([cell_dy - DESC_H] if cell_dy > 0 else [cell_dy + DESC_H] if cell_dy < 0 else [])
    out = []
    for y in ys:
        for x in xs:
            if (x, y) not in out:
                out.append((x, y))
    return out


def hypotheses_of(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """생성기 여럿(G4 · P · G3 의 영 이동)이 낸 칸 이동마다 그 접힌 짝까지 — 중복 없이 들어온 순서대로."""
    out: list[tuple[int, int]] = []
    for cx, cy in cells:
        for h in hypotheses(cx, cy):
            if h not in out:
                out.append(h)
    return out


def register(pa: Prepared, pb: Prepared, cell_dx: int | list[tuple[int, int]], cell_dy: int | None = None, *,
             window_cells: float, min_overlap: float) -> dict | None:
    """가설마다 ± window_cells 칸 안에서 겹친 자리 상관이 가장 큰 이동. 가장 좋은 가설과 둘째 가설의 값을 함께.

    `cell_dx` 에 (가로, 세로) 칸 이동의 목록을 주면 그 모두와 접힌 짝을 가설로 둔다. `hyp` 는 그 목록의 순번이다.
    """
    cells = cell_dx if isinstance(cell_dx, list) else [(cell_dx, int(cell_dy or 0))]
    r, dys, dxs = ncc_map(pa.rd, pb.rd, min_overlap)
    half_x = CELL_X_FULL / 2
    half_y = CELL_Y_FULL / 2
    win = round(window_cells * max(half_x, half_y))
    found = []
    for k, (hx, hy) in enumerate(hypotheses_of(cells)):
        b = best_in_window(r, dys, dxs, round(hy * half_y), round(hx * half_x), win)
        if b is not None:
            found.append((b[0], k, b[1], b[2]))
    if not found:
        return None
    found.sort(key=lambda t: (-t[0], t[1]))
    best = found[0]
    alt = max((f[0] for f in found[1:]
               if abs(f[2] - best[2]) > 2 * win or abs(f[3] - best[3]) > 2 * win), default=float("nan"))
    _, k, dx, dy = best
    # 봉우리 둘레(± 8)를 뺀 나머지의 99.9 분위 — v1 의 rd_side 와 같은 뜻(겹침 하한을 넘는 이동만)
    iy, ix = int(np.searchsorted(dys, dy)), int(np.searchsorted(dxs, dx))
    mask = ~np.isnan(r)
    mask[max(0, iy - 8):iy + 9, max(0, ix - 8):ix + 9] = False
    side = float(np.percentile(r[mask], 99.9)) if mask.any() else float("nan")
    return {"hyp": k, "dx": dx, "dy": dy, "ncc": best[0], "ncc_side": side, "alt_ncc": alt, "map_shape": pa.rd.shape}


def blocks(a: np.ndarray, b: np.ndarray) -> list:
    """겹친 자리(같은 꼴의 두 대역 통과 조각)의 4 × 6 블록별 상관 — v1 2f 와 같은 셈."""
    H, W = a.shape
    cs, ea, eb = [], [], []
    for gy in range(GY):
        for gx in range(GX):
            P = a[gy * H // GY:(gy + 1) * H // GY, gx * W // GX:(gx + 1) * W // GX]
            Q = b[gy * H // GY:(gy + 1) * H // GY, gx * W // GX:(gx + 1) * W // GX]
            if P.size == 0:
                cs.append(0.0)
                ea.append(0.0)
                eb.append(0.0)
                continue
            cs.append(_corr(P, Q))
            ea.append(float(P.std()))
            eb.append(float(Q.std()))
    cs, ea, eb = np.array(cs), np.array(ea), np.array(eb)
    ok = (ea >= 0.5 * np.median(ea)) & (eb >= 0.5 * np.median(eb)) & (ea > 0) & (eb > 0)
    if not ok.any():
        return [0, 0.0, 0.0, 0.0]
    c = cs[ok]
    return [int(ok.sum()), round(float((c >= 0.3).mean()), 4), round(float(np.median(c)), 4), round(float(c.min()), 4)]


def textband(a: np.ndarray, b: np.ndarray) -> list:
    """겹친 자리의 아래 20 %(문자 띠)와 나머지(본문) — v1 7 과 같은 셈."""
    cut = int(a.shape[0] * (1 - TEXT_BAND))
    if cut <= 0 or cut >= a.shape[0]:
        return [0.0, 0.0, 0.0, 0.0]
    t, body = _corr(a[cut:], b[cut:]), _corr(a[:cut], b[:cut])
    eb = float(a[:cut].std())
    en = float(a[cut:].std()) / eb if eb > 0 else 0.0
    return [round(t, 4), round(body, 4), round(t - body, 4), round(en, 3)]


def features(pa: Prepared, pb: Prepared, cell_dx: int | list[tuple[int, int]], cell_dy: int | None = None, *,
             window_cells: float, min_overlap: float, fold_eps: float | None) -> tuple[list | None, str]:
    """(HEAD 순서의 값, 사유). 정합을 못 하면 (None, 사유) — 사유는 01 문서 2절의 닫힌 목록.

    `fold_eps` 가 None 이면 접힘 미결을 가르지 않고 `fold_gap` 만 적는다(회차 1 — 판정 없음). `min_overlap` 은 계산 바닥이다 —
    규칙의 겹침 하한은 개발 판독에서 고른다.
    """
    reg = register(pa, pb, cell_dx, cell_dy, window_cells=window_cells, min_overlap=min_overlap)
    if reg is None:
        return None, "no_overlap_structure"
    dx, dy = reg["dx"], reg["dy"]
    h, w = pa.rd.shape
    ov = max(0, h - abs(dy)) * max(0, w - abs(dx)) / (h * w)
    gap = reg["ncc"] - reg["alt_ncc"] if np.isfinite(reg["alt_ncc"]) else float("nan")
    if fold_eps is not None and np.isfinite(gap) and gap < fold_eps:
        return None, "fold_ambiguous"
    A, B = overlap(pa.mid, pb.mid, dx, dy)
    Ah, Bh = overlap(pa.half, pb.half, dx, dy)
    row = [reg["hyp"], dx * 2, dy * 2, max(abs(dx), abs(dy)) * 2, round(ov, 4), round(reg["ncc"], 5),
           round(reg["ncc_side"], 5), round(reg["ncc"] - reg["ncc_side"], 5),
           (round(reg["alt_ncc"], 5) if np.isfinite(reg["alt_ncc"]) else ""), (round(gap, 5) if np.isfinite(gap) else ""),
           round(_corr(Ah, Bh), 5), *blocks(A, B), *textband(A, B)]
    return row, "full"
