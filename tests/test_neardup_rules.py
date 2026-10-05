"""근사 중복 — 과병합 안전장치(블록 일치)와 규칙 식 검사기.

실데이터를 읽지 않는다.

- 블록 일치(`scripts/neardup/nd_stage2f_blocks.py`): 같은 그림은 **전역에서** 맞는다. 필름 식별 문자 같은 작은 영역만 같은
  서로 다른 그림은 정합 봉우리가 서더라도 블록 대부분이 맞지 않는다.
- 규칙 식(`scripts/neardup/nd_stage5_lists.py`): 제외 기준은 화소·해시 특징으로만 쓴다. 정답 라벨·분할·참여자 같은 열 이름은
  규칙에 들어오는 순간 거부한다 — 지시의 제약("정답 라벨과 모델 출력을 기준에 쓰지 않는다")을 코드로 지킨다.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.neardup import nd_stage2d_refine as R
from scripts.neardup import nd_stage2f_blocks as B
from scripts.neardup import nd_stage5_lists as S
from scripts.neardup import nd_stage7_textband as X

H, W = 604, 1076
RI = {k: i for i, k in enumerate(R.HEAD)}
BI = {k: i for i, k in enumerate(B.HEAD)}


def _weld(rng: np.random.Generator, band_center: float) -> np.ndarray:
    y = np.arange(H, dtype=np.float32)[:, None]
    band = 60.0 * np.exp(-((y - band_center) ** 2) / (2 * 70.0**2))
    tex = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 5.0)
    img = 110.0 + band + 40.0 * tex + 3.0 * rng.standard_normal((H, W)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.float32)


def _stamp(img: np.ndarray, seed: int = 99) -> np.ndarray:
    """필름 식별 문자를 흉내 낸 고대비 조각(180x56)을 왼쪽 위에 찍는다 — 서로 다른 그림에 **같은** 조각이 들어간다."""
    g = np.random.default_rng(seed)
    glyph = (g.random((7, 22)) > 0.5).astype(np.float32)
    patch = cv2.resize(glyph, (176, 56), interpolation=cv2.INTER_NEAREST) * 150.0 + 60.0
    out = img.copy()
    out[40:96, 60:236] = patch
    return out


def _both(a: np.ndarray, b: np.ndarray) -> tuple[dict, dict]:
    r = R.compare(R.prepare(a), R.prepare(b))
    blk = B.compare(B.bandpass(a), B.bandpass(b), int(r[RI["rd_dx"]]), int(r[RI["rd_dy"]]))
    return {k: r[i] for k, i in RI.items()}, {k: blk[i] for k, i in BI.items()}


def test_같은_그림은_전역에서_맞는다():
    rng = np.random.default_rng(11)
    a = _stamp(_weld(rng, 300.0))
    b = np.roll(np.roll(a, -4, axis=0), 6, axis=1)
    b = np.clip(255.0 * (b / 255.0) ** 0.8 + 4.0 * rng.standard_normal(a.shape).astype(np.float32), 0, 255).astype(np.float32)
    r, blk = _both(a, b)
    assert r["rd_sharp"] >= 0.4
    assert blk["blk_n"] >= 12 and blk["blk_frac"] >= 0.8


def test_식별_문자만_같은_다른_그림은_블록에서_걸러진다():
    a = _stamp(_weld(np.random.default_rng(12), 300.0))
    b = _stamp(_weld(np.random.default_rng(13), 300.0))            # 다른 그림, 같은 자리에 같은 문자
    r, blk = _both(a, b)
    assert (r["rd_dx"], r["rd_dy"]) == (0, 0) and r["rd_sharp"] >= 0.5   # 문자가 정합을 끌어당긴다 — 봉우리만 보면 속는다
    assert r["reg_mid"] >= 0.5                                   # 정합 뒤 전역 상관도 고대비 조각 하나에 끌려간다 — 블록만이 거른다
    assert blk["blk_frac"] <= 0.25                                 # 24칸 가운데 문자가 걸친 몇 칸만 맞는다


def test_블록_특징_순서가_머리글과_같다():
    assert B.HEAD == ["blk_n", "blk_frac", "blk_med", "blk_min"]


@pytest.mark.parametrize("expr", [
    "has_defect == 1", "material == 0 and rd_sharp >= 0.5", "split == 1", "client == 1", "n_defects > 0", "score >= 0.5",
])
def test_규칙은_라벨과_분할_열을_거부한다(expr):
    with pytest.raises(ValueError, match="쓸 수 없는 이름"):
        S.compile_rule(expr)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('x')", "rd_sharp.real >= 0", "[rd_sharp][0] >= 0", "(lambda: 1)()", "abs(rd_dx, key=1) >= 0",
    "rd_sharp(1)",
])
def test_규칙은_허용한_구문만_받는다(expr):
    with pytest.raises((TypeError, ValueError)):
        S.compile_rule(expr)


def test_규칙은_없는_특징을_거짓으로_둔다():
    rule, names = S.compile_rule("mad < 2 and corr > 0.99")
    assert names == ["corr", "mad"]
    assert rule({"mad": "1.5", "corr": "0.995"}) is True
    assert rule({"mad": "", "corr": "0.995"}) is False             # G3 만 찾은 쌍에는 mad 가 없다 — 빈 값을 0 으로 읽으면 전부 통과한다
    assert rule({"corr": "0.995"}) is False


def test_규칙은_이동_상한을_쓸_수_있다():
    rule, _ = S.compile_rule("rd_sharp >= 0.4 and max(abs(rd_dx), abs(rd_dy)) <= 32")
    assert rule({"rd_sharp": "0.5", "rd_dx": "-30", "rd_dy": "4"}) is True
    assert rule({"rd_sharp": "0.5", "rd_dx": "-34", "rd_dy": "4"}) is False


def test_저장된_정의는_전부_화소와_해시만_쓴다():
    defs = json.loads((Path(S.__file__).resolve().parent / "definitions.json").read_text(encoding="utf-8"))["definitions"]
    assert len({d["key"] for d in defs}) == len(defs) >= 1
    for d in defs:
        _, names = S.compile_rule(d["rule"])
        assert set(names) <= S.ALLOWED


# ------------------------------ 하단 문자 띠와 본문을 따로 재기 (같은 필름의 다른 프레임)

def _with_band(img, band, seed=7):
    """아래 20% 를 band 로 갈아 끼운다 — 필름에 인쇄된 문자 띠를 흉내 낸 것이다."""
    out = img.copy()
    out[int(H * 0.8):] = band
    return out


def _band(seed):
    g = np.random.default_rng(seed)
    b = cv2.GaussianBlur(g.standard_normal((int(H * 0.2) + 1, W)).astype(np.float32), (0, 0), 1.5) * 60 + 120
    return np.clip(b[: H - int(H * 0.8)], 0, 255)


def test_같은_필름의_다른_프레임은_문자_띠만_맞는다():
    """본문이 다른데 문자 띠가 같으면 `txt_gap` 이 크다 — 이것이 남은 오판 계열을 가르는 열이다."""
    band = _band(1)
    a = _with_band(_weld(np.random.default_rng(21), 300.0), band)
    b = _with_band(_weld(np.random.default_rng(22), 300.0), band)      # 다른 본문, 같은 문자 띠
    v = X.compare(X.bandpass_in(a), X.bandpass_in(b), 0, 0)
    g = {k: v[i] for i, k in enumerate(X.HEAD)}
    assert g["txt_corr"] >= 0.8 and g["body_corr"] <= 0.3
    assert g["txt_gap"] >= 0.5


def test_같은_사진은_문자_띠와_본문이_함께_맞는다():
    band = _band(2)
    a = _with_band(_weld(np.random.default_rng(23), 300.0), band)
    rng = np.random.default_rng(24)
    b = np.clip(255.0 * (a / 255.0) ** 0.85 + 3.0 * rng.standard_normal(a.shape).astype(np.float32), 0, 255).astype(np.float32)
    v = X.compare(X.bandpass_in(a), X.bandpass_in(b), 0, 0)
    g = {k: v[i] for i, k in enumerate(X.HEAD)}
    assert g["txt_corr"] >= 0.8 and g["body_corr"] >= 0.8
    assert g["txt_gap"] <= 0.2


def test_문자_띠가_비어_있으면_그_상관은_뜻이_없다():
    """띠에 구조가 없으면 `txt_energy` 가 낮다 — 그 경우 `txt_corr` 가 1.0 이어도 증거가 되지 못한다.

    0 이 아니라 '훨씬 낮다' 로 건다. 대역 통과가 본문과 띠의 경계를 조금 물고 들어오기 때문에 완전한 0 은 나오지 않는다.
    """
    def energy(band):
        a = _with_band(_weld(np.random.default_rng(25), 300.0), band)
        v = X.compare(X.bandpass_in(a), X.bandpass_in(a), 0, 0)
        return {k: v[i] for i, k in enumerate(X.HEAD)}

    flat = energy(np.full((H - int(H * 0.8), W), 120.0, dtype=np.float32))
    printed = energy(_band(1))
    assert flat["txt_corr"] == printed["txt_corr"] == 1.0          # 상관만 보면 둘이 구별되지 않는다
    assert flat["txt_energy"] < 0.5 and printed["txt_energy"] > 1.5   # 에너지가 그 둘을 가른다
    assert flat["txt_energy"] < printed["txt_energy"] / 3


def test_문자_띠_특징_순서가_머리글과_같다():
    assert X.HEAD == ["txt_corr", "body_corr", "txt_gap", "txt_energy"]

