"""거친 구조 기술자(64 × 36) 위의 후보 생성기 — G3(영 이동 상관) · G4(원형 교차상관 봉우리) · `P` 탐색(겹친 자리 정규화
상관). 순수 함수(파일을 읽지 않는다). 정의는 v1 과 같다(`scripts/neardup/nd_stage3b_bppairs.py` · `nd_stage3d_shiftscan.py`).

`P` 탐색이 따로 있는 까닭(미니스펙 2판 3-1 (다)) — G4 봉우리는 대략 "겹친 넓이 × 실제 상관" 이라 겹침이 작은 부분 중첩은
문턱 아래로 빠진다. `P` 탐색은 영으로 채운 선형 상관을 겹친 자리마다 정규화해 겹침과 무관한 상관을 낸다.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from data.nd2.ncc import ncc_map

GH, GW = 36, 64


def normalize(D: np.ndarray) -> np.ndarray:
    """(n, 36, 64) → 장마다 칸 전체 평균을 빼고 L2 정규화(노름 < 1e-3 이면 0). G3 · G4 에 같은 정규화를 쓴다."""
    D = np.asarray(D, dtype=np.float32).copy()
    D -= D.mean(axis=(1, 2), keepdims=True)
    nrm = np.sqrt((D * D).sum(axis=(1, 2)))
    flat = nrm < 1e-3
    nrm[flat] = 1.0
    D /= nrm[:, None, None]
    D[flat] = 0
    return D


def spectra(Dn: np.ndarray, workers: int) -> np.ndarray:
    return sfft.rfft2(Dn, workers=workers).astype(np.complex64)


def g4_peaks(fa: np.ndarray, FB: np.ndarray, workers: int, chunk: int = 8192) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """기술자 하나의 스펙트럼 대 여럿 → (봉우리, 세로 칸 이동, 가로 칸 이동). 이동은 a 기준 b 의 위치(v1 과 같다)."""
    val = np.empty(len(FB), dtype=np.float32)
    arg = np.empty(len(FB), dtype=np.int64)
    for s in range(0, len(FB), chunk):
        xc = sfft.irfft2(fa[None] * np.conj(FB[s:s + chunk]), s=(GH, GW), workers=workers).reshape(-1, GH * GW)
        arg[s:s + chunk] = xc.argmax(axis=1)
        val[s:s + chunk] = xc[np.arange(len(xc)), arg[s:s + chunk]]
    py, px = np.divmod(arg, GW)
    dy = np.where(py <= GH // 2, py, py - GH)
    dx = np.where(px <= GW // 2, px, px - GW)
    return val, dy.astype(np.int32), dx.astype(np.int32)


def g3_corr(Dn_a: np.ndarray, Dn_b: np.ndarray) -> np.ndarray:
    """정규화한 기술자 묶음 둘의 영 이동 상관(행렬). (na, 36, 64) × (nb, 36, 64) → (na, nb)."""
    return Dn_a.reshape(len(Dn_a), -1) @ Dn_b.reshape(len(Dn_b), -1).T


def psearch(a: np.ndarray, b: np.ndarray, min_overlap: float) -> tuple[float, int, int, float]:
    """원 기술자(정규화 전) 둘 → (겹친 자리 정규화 상관의 최댓값, 가로 칸, 세로 칸, 그 이동의 겹침). 유효한 이동이 없으면 NaN."""
    r, dys, dxs = ncc_map(a, b, min_overlap)
    if np.all(np.isnan(r)):
        return float("nan"), 0, 0, 0.0
    i, j = np.unravel_index(np.nanargmax(r), r.shape)
    dy, dx = int(dys[i]), int(dxs[j])
    ov = max(0, GH - abs(dy)) * max(0, GW - abs(dx)) / (GH * GW)
    return float(r[i, j]), dx, dy, ov


class _Win:
    """기술자 꼴(36 × 64)의 모든 이동에서 a 쪽 · b 쪽 겹친 창의 누적 합 표 색인(한 번만 만든다)."""

    def __init__(self, h: int = GH, w: int = GW, min_overlap: float = 0.2) -> None:
        dys, dxs = np.arange(-(h - 1), h), np.arange(-(w - 1), w)
        DY, DX = np.meshgrid(dys, dxs, indexing="ij")
        ya0, ya1 = np.maximum(0, DY), np.minimum(h, h + DY)
        xa0, xa1 = np.maximum(0, DX), np.minimum(w, w + DX)
        self.a = (ya0, ya1, xa0, xa1)
        self.b = (ya0 - DY, ya1 - DY, xa0 - DX, xa1 - DX)
        self.N = ((ya1 - ya0) * (xa1 - xa0)).astype(np.float64)
        self.valid = self.N >= min_overlap * h * w
        self.idx = np.flatnonzero(self.valid.ravel())
        self.dy, self.dx = DY.ravel()[self.idx], DX.ravel()[self.idx]
        self.cy, self.cx = DY.ravel()[self.idx] % (2 * h), DX.ravel()[self.idx] % (2 * w)
        self.h, self.w = h, w

    def sums(self, X: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(n, h, w) → 유효 이동마다 창 합 · 제곱합 (n, k)."""
        y0, y1, x0, x1 = (t.ravel()[self.idx] for t in (self.a if side == "a" else self.b))
        out = []
        for Z in (X, X * X):
            S = np.zeros((len(X), self.h + 1, self.w + 1), dtype=np.float64)
            S[:, 1:, 1:] = np.cumsum(np.cumsum(Z, axis=1, dtype=np.float64), axis=2)
            out.append(S[:, y1, x1] - S[:, y0, x1] - S[:, y1, x0] + S[:, y0, x0])
        return out[0], out[1]


def psearch_batch(a: np.ndarray, B: np.ndarray, win: _Win, workers: int, chunk: int = 512
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """원 기술자 하나 대 여럿 — 겹친 자리 정규화 상관의 최댓값과 그 이동(가로 · 세로 칸). `psearch` 와 같은 값(시험)."""
    h, w = win.h, win.w
    a = np.asarray(a, dtype=np.float64)
    fa = sfft.rfft2(a, s=(2 * h, 2 * w), workers=workers)
    Sa, Saa = win.sums(a[None], "a")
    Sa, Saa = Sa[0], Saa[0]
    N = win.N.ravel()[win.idx]
    Va = np.maximum(Saa - Sa * Sa / N, 0)
    val = np.full(len(B), np.nan)
    sx = np.zeros(len(B), dtype=np.int32)
    sy = np.zeros(len(B), dtype=np.int32)
    for s in range(0, len(B), chunk):
        X = np.asarray(B[s:s + chunk], dtype=np.float64)
        C = sfft.irfft2(fa[None] * np.conj(sfft.rfft2(X, s=(2 * h, 2 * w), workers=workers)), s=(2 * h, 2 * w),
                        workers=workers)[:, win.cy, win.cx]
        Sb, Sbb = win.sums(X, "b")
        with np.errstate(invalid="ignore", divide="ignore"):
            den = np.sqrt(Va[None] * np.maximum(Sbb - Sb * Sb / N[None], 0))
            r = (C - Sa[None] * Sb / N[None]) / den
        r[~np.isfinite(r) | (den <= 1e-12)] = -np.inf
        k = r.argmax(axis=1)
        m = r[np.arange(len(r)), k]
        ok = np.isfinite(m)
        val[s:s + chunk] = np.where(ok, m, np.nan)
        sx[s:s + chunk], sy[s:s + chunk] = win.dx[k], win.dy[k]
    return val, sy, sx
