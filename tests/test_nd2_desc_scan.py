"""간선 v2 — 기술자 수준 생성기(G3 · G4 · P 탐색)의 합성 시험. 실데이터를 읽지 않는다."""
from __future__ import annotations

import numpy as np
import pytest

from data.nd2 import desc_scan as S


def _desc(seed: int, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    import cv2
    x = cv2.GaussianBlur(rng.normal(size=(36, w)).astype(np.float32), (0, 0), 1.2)
    return x


def test_묶음_P_탐색이_한_쌍_셈과_같다():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(36, 64))
    B = rng.normal(size=(7, 36, 64))
    win = S._Win(min_overlap=0.2)
    val, dy, dx = S.psearch_batch(a, B, win, workers=1, chunk=3)
    for k in range(len(B)):
        v, x, y, _ = S.psearch(a, B[k], 0.2)
        assert val[k] == pytest.approx(v, abs=1e-9)
        assert (dx[k], dy[k]) == (x, y)


@pytest.mark.parametrize("shift", [0, 8, 20, 32, 44])
def test_P_탐색은_겹침이_작아도_같은_자리를_찾는다(shift):
    pano = _desc(1, 64 + 48)
    a, b = pano[:, shift:shift + 64], pano[:, :64]                  # a[y, x] = b[y, x + shift] → dx = -shift
    v, dx, dy, ov = S.psearch(a, b, 0.2)
    assert (dx, dy) == (-shift, 0)
    assert v == pytest.approx(1.0, abs=1e-6)
    assert ov == pytest.approx(1 - shift / 64)


def test_G4_봉우리는_겹침에_따라_줄고_P_탐색은_줄지_않는다():
    pano = _desc(2, 64 + 48)
    Dn = S.normalize(np.stack([pano[:, :64], pano[:, 40:104]]))
    F = S.spectra(Dn, workers=1)
    g4, _, _ = S.g4_peaks(F[0], F[1:], workers=1)
    v, *_ = S.psearch(pano[:, :64], pano[:, 40:104], 0.2)
    assert g4[0] < 0.6 and v > 0.99                                   # 겹침 37.5 % — G4 는 문턱 0.45 근처로 떨어진다


def test_G4_이동의_부호와_접힘():
    pano = _desc(3, 64 + 20)
    a, b = pano[:, 10:74], pano[:, :64]                                # a[y, x] = b[y, x + 10]
    Dn = S.normalize(np.stack([a, b]))
    F = S.spectra(Dn, workers=1)
    _val, dy, dx = S.g4_peaks(F[0], F[1:], workers=1)
    assert (int(dx[0]), int(dy[0])) == (-10, 0)


def test_G3_은_정규화한_내적():
    rng = np.random.default_rng(4)
    D = rng.normal(size=(5, 36, 64)).astype(np.float32)
    Dn = S.normalize(D)
    G = S.g3_corr(Dn, Dn)
    assert np.allclose(np.diag(G), 1.0, atol=1e-5)
    x, y = D[1] - D[1].mean(), D[2] - D[2].mean()
    assert G[1, 2] == pytest.approx(float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())), abs=1e-5)


def test_평탄한_기술자는_0():
    D = np.zeros((2, 36, 64), dtype=np.float32)
    assert np.all(S.normalize(D) == 0)
    win = S._Win()
    val, _, _ = S.psearch_batch(np.zeros((36, 64)), D, win, workers=1)
    assert np.all(np.isnan(val))


def test_G4_는_늘_G3_이상이다():
    rng = np.random.default_rng(9)
    D = S.normalize(rng.normal(size=(40, 36, 64)).astype(np.float32))
    F = S.spectra(D, workers=1)
    g4, _, _ = S.g4_peaks(F[0], F[1:], workers=1)
    g3 = S.g3_corr(D[:1], D[1:])[0]
    assert np.all(g4 >= g3 - 1e-5)
