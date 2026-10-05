"""근사 중복 판정의 핵심 — "같은 방사선 사진인가" 를 가르는 정밀 특징(`scripts/neardup/nd_stage2d_refine.py`).

실데이터를 읽지 않는다. 합성 영상으로 성질만 고정한다.

- 같은 그림은 밝기 변환·잡음·작은 이동이 달라도 정합 봉우리가 뾰족하다(`rd_sharp`), 이동이 복원된다.
- **수평 구조만 같은 다른 그림**(용접 비드의 경계가 같은 높이에 있는 서로 다른 영상)은 뾰족하지 않다 —
  행·열 평균을 빼는 이유가 이것이다. 이 시험이 그 설계를 지킨다.
- 증폭된 판의 "평균 절대차" 는 같은 그림을 놓친다(65~10번 문서의 실측과 같은 방향).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.neardup import nd_stage2d_refine as R

H, W = 604, 1076                                   # 안쪽 상자 크기(1280x720 에서 테두리 8% 를 뺀 것)
IDX = {k: i for i, k in enumerate(R.HEAD)}


def _texture(rng: np.random.Generator, sigma: float) -> np.ndarray:
    a = rng.standard_normal((H, W)).astype(np.float32)
    return cv2.GaussianBlur(a, (0, 0), sigma)


def _weld(rng: np.random.Generator, band_center: float = 300.0) -> np.ndarray:
    """가로로 긴 밝은 띠(비드) + 중간 규모 얼룩 + 미세 잡음 + 결함 몇 개."""
    y = np.arange(H, dtype=np.float32)[:, None]
    band = 60.0 * np.exp(-((y - band_center) ** 2) / (2 * 70.0**2))
    img = 110.0 + band + 14.0 * _texture(rng, 6.0) / 0.05 * 0.05 + 40.0 * _texture(rng, 5.0) + 3.0 * rng.standard_normal((H, W)).astype(np.float32)
    for _ in range(5):
        cy, cx, r = int(rng.integers(80, H - 80)), int(rng.integers(80, W - 80)), int(rng.integers(6, 18))
        cv2.circle(img, (cx, cy), r, float(img[cy, cx] - 35.0), -1)
    return np.clip(img, 0, 255).astype(np.float32)


def _variant(img: np.ndarray, rng: np.random.Generator, *, dx: int, dy: int, gamma: float, noise: float) -> np.ndarray:
    """같은 그림의 다른 사본 — 이동 · 밝기 변환 · 잡음."""
    out = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** gamma
    return np.clip(out + noise * rng.standard_normal(img.shape).astype(np.float32), 0, 255).astype(np.float32)


def _f(a: np.ndarray, b: np.ndarray) -> dict:
    v = R.compare(R.prepare(a), R.prepare(b))
    return {k: v[i] for k, i in IDX.items()}


def test_같은_그림은_밝기와_잡음이_달라도_뾰족하다():
    rng = np.random.default_rng(1)
    a = _weld(rng)
    f = _f(a, _variant(a, rng, dx=0, dy=0, gamma=0.7, noise=6.0))
    assert f["rd_sharp"] >= 0.5 and (f["rd_dx"], f["rd_dy"]) == (0, 0)
    assert f["reg_coarse"] >= 0.9


def test_증폭된_판의_평균_절대차는_같은_그림을_놓친다():
    """밝기 변환이 다르면 평균 절대차는 크다 — 그래도 같은 그림이다. '평균 절대차 < 2' 기준의 반례."""
    rng = np.random.default_rng(2)
    a = _weld(rng)
    b = _variant(a, rng, dx=0, dy=0, gamma=0.6, noise=4.0)
    assert float(np.abs(a - b).mean()) > 10.0
    assert _f(a, b)["rd_sharp"] >= 0.5


@pytest.mark.parametrize("dx, dy", [(6, -4), (-10, 8), (14, 0)])
def test_작은_이동은_복원된다(dx, dy):
    rng = np.random.default_rng(3)
    a = _weld(rng)
    f = _f(a, _variant(a, rng, dx=dx, dy=dy, gamma=0.9, noise=3.0))
    assert f["rd_sharp"] >= 0.4
    # a[y, x] 가 b[y - dy', x - dx'] 와 맞는다 — b 는 a 를 (dx, dy) 만큼 민 것이므로 복원값은 (-dx, -dy) 다. 절반 해상도라 ±2px.
    assert abs(f["rd_dx"] + dx) <= 2 and abs(f["rd_dy"] + dy) <= 2
    assert f["ov"] >= 0.95


def test_수평_구조만_같은_다른_그림은_뾰족하지_않다():
    """비드가 같은 높이에 있는 서로 다른 영상 — 행 평균을 빼지 않으면 상관이 부풀어 같은 그림으로 오인한다."""
    a = _weld(np.random.default_rng(4), band_center=300.0)
    b = _weld(np.random.default_rng(5), band_center=300.0)          # 같은 띠, 다른 질감·다른 결함
    f = _f(a, b)
    assert f["rd_sharp"] < 0.2
    assert f["reg_coarse"] > f["rd_sharp"]                          # 거친 규모의 밝기 상관은 띠 때문에 높게 남는다 — 그것에 기대면 안 된다


def test_서로_다른_그림은_뾰족하지_않다():
    f = _f(_weld(np.random.default_rng(6), 200.0), _weld(np.random.default_rng(7), 420.0))
    assert f["rd_sharp"] < 0.2


def test_구조가_없는_영상은_판정_불능으로_떨어진다():
    flat = np.full((H, W), 120.0, dtype=np.float32)
    v = R.compare(R.prepare(flat), R.prepare(flat))
    assert v[IDX["rd_sharp"]] == 0.0 and v[IDX["rd_peak"]] == 0.0   # 구조 에너지 0 → 같은 그림이라고 말하지 않는다


def test_특징_순서가_머리글과_같다():
    rng = np.random.default_rng(8)
    a = _weld(rng)
    assert len(R.compare(R.prepare(a), R.prepare(a))) == len(R.HEAD)
    assert R.HEAD[:5] == ["rd_peak", "rd_dx", "rd_dy", "rd_side", "rd_sharp"]
