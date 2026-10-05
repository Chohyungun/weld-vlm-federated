"""간선 v2 — 겹친 자리 정규화 상관과 두 가설 정합(합성 영상, 실데이터를 읽지 않는다)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from data.nd2 import features as F
from data.nd2.ncc import best_in_window, ncc_map, overlap_fraction


def _brute(a, b, dx, dy):
    h, w = a.shape
    ya0, ya1, xa0, xa1 = max(0, dy), min(h, h + dy), max(0, dx), min(w, w + dx)
    p = a[ya0:ya1, xa0:xa1].astype(np.float64)
    q = b[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx].astype(np.float64)
    p, q = p - p.mean(), q - q.mean()
    return float((p * q).sum() / np.sqrt((p * p).sum() * (q * q).sum()))


def test_겹친_자리_상관이_손셈과_같다():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(9, 13)), rng.normal(size=(9, 13))
    r, dys, dxs = ncc_map(a, b, min_overlap=0.1)
    for dx, dy in [(0, 0), (3, -2), (-5, 4), (8, 1), (-12, -3)]:
        i, j = int(np.searchsorted(dys, dy)), int(np.searchsorted(dxs, dx))
        if overlap_fraction(9, 13, dx, dy) >= 0.1:
            assert r[i, j] == pytest.approx(_brute(a, b, dx, dy), abs=1e-9)
        else:
            assert np.isnan(r[i, j])


def test_겹침_하한_아래는_비운다():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(10, 10))
    r, dys, dxs = ncc_map(a, a, min_overlap=0.5)
    i, j = int(np.searchsorted(dys, 0)), int(np.searchsorted(dxs, 6))
    assert np.isnan(r[i, j])                                   # 겹침 0.4
    i, j = int(np.searchsorted(dys, 0)), int(np.searchsorted(dxs, 4))
    assert not np.isnan(r[i, j])                               # 겹침 0.6


def _texture(h, w, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(h, w)).astype(np.float32)
    return cv2.GaussianBlur(x, (0, 0), 3.0) * 40 + 120


def test_옮긴_조각의_이동을_찾는다():
    big = _texture(80, 140, 3)
    a = big[10:50, 20:100]
    b = big[10 + 6:50 + 6, 20 - 9:100 - 9]                    # a[y, x] = big[y + 10, x + 20] = b[y - 6, x + 9] → (dx, dy) = (-9, 6)
    r, dys, dxs = ncc_map(a, b, min_overlap=0.2)
    m, dx, dy = best_in_window(r, dys, dxs, 0, 0, 30)
    assert (dx, dy) == (-9, 6)
    assert m > 0.999


def test_접힌_짝을_만든다():
    assert F.hypotheses(38, 0) == [(38, 0), (-26, 0)]
    assert F.hypotheses(-5, 3) == [(-5, 3), (59, 3), (-5, -33), (59, -33)]
    assert F.hypotheses(0, 0) == [(0, 0)]


def _panorama_pair(shift_full: int, seed: int = 5):
    """한 장의 넓은 그림에서 가로로 shift_full px 떨어진 두 안쪽 상자(전 해상도 604 × 1076)."""
    pano = _texture(604, 1076 + shift_full + 8, seed)
    a = pano[:, shift_full:shift_full + 1076]
    b = pano[:, :1076]
    return a, b                                                 # a[y, x] = pano[y, x + s] = b[y, x + s] → dx = -s


def test_큰_이동은_바른_가설로_정합한다():
    s = 640
    a, b = _panorama_pair(s)
    pa, pb = F.prepare(a), F.prepare(b)
    true_cell = -s / F.CELL_X_FULL                              # 약 -38 칸 — 원형 상관에서는 +26 칸으로 접혀 보인다
    folded = round(true_cell) + F.DESC_W
    row, why = F.features(pa, pb, folded, 0, window_cells=1, min_overlap=0.2, fold_eps=0.02)
    assert why == "full"
    d = dict(zip(F.HEAD, row))
    assert abs(d["dx"] + s) <= 2 and abs(d["dy"]) <= 2         # 전 해상도 px — 접힌 쪽이 아니라 참 이동
    assert d["overlap"] == pytest.approx(1 - s / 1076, abs=0.01)
    assert d["ncc"] > 0.95
    assert d["blk_med"] > 0.9


def test_다른_그림은_맞지_않는다():
    a = _texture(604, 1076, 7)
    b = _texture(604, 1076, 8)
    row, _why = F.features(F.prepare(a), F.prepare(b), 3, 1, window_cells=1, min_overlap=0.2, fold_eps=0.0)
    d = dict(zip(F.HEAD, row))
    assert d["ncc"] < 0.3
    assert d["blk_med"] < 0.3


def test_가설이_모두_겹침_하한_밖이면_사유를_낸다():
    a, b = _panorama_pair(1000)
    # 가설 -60 칸과 접힌 짝 +4 칸(절반 해상도 약 34 px) 모두 겹침 0.99 밖이다
    row, why = F.features(F.prepare(a), F.prepare(b), -60, 0, window_cells=1, min_overlap=0.99, fold_eps=0.02)
    assert row is None and why == "no_overlap_structure"


def test_참_겹침이_작으면_접힌_가설이_골라져도_상관이_낮다():
    a, b = _panorama_pair(1000)                                  # 참 겹침 약 7 % — 하한 0.2 아래
    row, _why = F.features(F.prepare(a), F.prepare(b), -60, 0, window_cells=1, min_overlap=0.2, fold_eps=0.0)
    d = dict(zip(F.HEAD, row))
    assert d["hyp"] == 1 and d["ncc"] < 0.3                      # 판정 단계에서 아님이 될 값


def test_여러_생성기의_이동을_함께_가설로_둔다():
    s = 640
    a, b = _panorama_pair(s)
    pa, pb = F.prepare(a), F.prepare(b)
    true_cell = round(-s / F.CELL_X_FULL)
    # G3 의 영 이동(틀린 가설)과 P 의 참 이동을 함께 준다 — 참 이동이 골라진다
    row, why = F.features(pa, pb, [(0, 0), (true_cell, 0)], window_cells=1, min_overlap=0.05, fold_eps=None)
    d = dict(zip(F.HEAD, row))
    assert why == "full" and abs(d["dx"] + s) <= 2
    assert F.hypotheses_of([(0, 0), (true_cell, 0)]) == [(0, 0), (true_cell, 0), (true_cell + 64, 0)]


def test_접힘_미결은_회차_1_에서_가르지_않는다():
    a, b = _panorama_pair(640)
    row, why = F.features(F.prepare(a), F.prepare(b), 26, 0, window_cells=1, min_overlap=0.05, fold_eps=None)
    assert why == "full" and row is not None
