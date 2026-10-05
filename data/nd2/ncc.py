"""겹친 자리만으로 잰 정규화 상관(피어슨) — 모든 이동에서 한 번에. 순수 함수.

원형 상관(G4 · v1 의 2d)은 이동이 크면 반대편이 감겨 들어와 겹친 넓이와 무관한 값을 낸다. 여기서는 영으로 채워
**선형** 상관을 내고, 겹친 자리의 합 · 제곱합은 누적 합 표로 구해 겹친 자리마다 평균을 뺀 피어슨 상관을 만든다.

이동의 뜻은 v1 과 같다 — `(dx, dy)` 이면 `a[y, x]` 와 `b[y - dy, x - dx]` 가 같은 자리다.
"""
from __future__ import annotations

import numpy as np


def _rect_sums(img: np.ndarray, y0: np.ndarray, y1: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """누적 합 표로 직사각형 합. y0..y1 · x0..x1 은 같은 꼴의 배열(끝은 열린 구간)."""
    s = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = np.cumsum(np.cumsum(img, axis=0, dtype=np.float64), axis=1)
    return s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]


def ncc_map(a: np.ndarray, b: np.ndarray, min_overlap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """같은 꼴의 두 영상 → (상관 지도, dy 축, dx 축). 겹친 넓이가 `min_overlap`(전체 넓이 대비) 아래인 이동은 NaN.

    지도의 [i, j] 는 이동 (dx = dxs[j], dy = dys[i]) 의 값이다.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("두 영상의 꼴이 다르다")
    h, w = a.shape
    H, W = 2 * h, 2 * w
    fa = np.fft.rfft2(a, s=(H, W))
    fb = np.fft.rfft2(b, s=(H, W))
    c = np.fft.irfft2(fa * np.conj(fb), s=(H, W))          # c[k] = Σ_n a[n + k] b[n]
    dys = np.arange(-(h - 1), h)
    dxs = np.arange(-(w - 1), w)
    DY, DX = np.meshgrid(dys, dxs, indexing="ij")
    C = c[DY % H, DX % W]
    ya0, ya1 = np.maximum(0, DY), np.minimum(h, h + DY)
    xa0, xa1 = np.maximum(0, DX), np.minimum(w, w + DX)
    yb0, yb1, xb0, xb1 = ya0 - DY, ya1 - DY, xa0 - DX, xa1 - DX
    N = ((ya1 - ya0) * (xa1 - xa0)).astype(np.float64)
    Sa, Saa = _rect_sums(a, ya0, ya1, xa0, xa1), _rect_sums(a * a, ya0, ya1, xa0, xa1)
    Sb, Sbb = _rect_sums(b, yb0, yb1, xb0, xb1), _rect_sums(b * b, yb0, yb1, xb0, xb1)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = C - Sa * Sb / N
        den = np.sqrt(np.maximum(Saa - Sa * Sa / N, 0) * np.maximum(Sbb - Sb * Sb / N, 0))
        r = num / den
    r[(N < min_overlap * h * w) | ~np.isfinite(r) | (den <= 1e-12)] = np.nan
    return r, dys, dxs


def overlap_fraction(h: int, w: int, dx: int, dy: int) -> float:
    return max(0, h - abs(dy)) * max(0, w - abs(dx)) / (h * w)


def best_in_window(r: np.ndarray, dys: np.ndarray, dxs: np.ndarray, cy: int, cx: int, half: int) -> tuple[float, int, int] | None:
    """(cx, cy) 를 가운데로 ± half 안에서 가장 큰 값과 그 이동. 창 안이 모두 NaN 이면 None. 동률은 작은 |이동| · 작은 (dy, dx) 순."""
    iy0, iy1 = np.searchsorted(dys, cy - half), np.searchsorted(dys, cy + half, side="right")
    ix0, ix1 = np.searchsorted(dxs, cx - half), np.searchsorted(dxs, cx + half, side="right")
    sub = r[iy0:iy1, ix0:ix1]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return None
    m = np.nanmax(sub)
    cand = [(abs(int(dys[iy0 + i])) + abs(int(dxs[ix0 + j])), int(dys[iy0 + i]), int(dxs[ix0 + j]))
            for i, j in zip(*np.nonzero(sub == m))]
    _, dy, dx = min(cand)
    return float(m), dx, dy
