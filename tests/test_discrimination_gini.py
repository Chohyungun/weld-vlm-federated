"""임계 독립 판별력 `Δ_AUC = 2·AUROC − 1` — 승격 후보의 성질 (17번 §12-5).

이 파일이 지키는 것은 셋이다.

1. **지름길이 정확히 0** — 출처 고정 구간에서 상수 점수를 주는 예측기는 0 을 넘을 수 없다.
   이것이 `Δ` 와 이 지표를 함께 쓸 수 있는 유일한 이유다. 근사 0 이 아니라 **정확히** 0 이어야
   한다(부동소수 오차 없이). 넘으면 지름길이 통과할 수 있는 축이 되고 대조가 무의미해진다.
2. **임계와 무관** — 같은 순위를 주는 점수라면 값이 같아야 한다. 이 지표의 존재 이유다.
3. **`image_score` 가 발화 정의의 연속판** — `image_score(r) >= c` 가 임계 `c` 로 자른 뒤의
   `fires(r)` 와 **항상** 같아야 한다. 어긋나면 곡선과 임계 독립판이 다른 것을 재게 된다.
"""

from __future__ import annotations

import math

import pytest

from evaluation.detect_infer import filter_by_conf
from evaluation.discrimination import fires, gini, image_score
from evaluation.schema import Defect, PredictionRecord


def _rec(image_id: str, scores: list[float], *, parse_ok: bool = True) -> PredictionRecord:
    return PredictionRecord(
        schema_version="1.3",
        image_id=image_id,
        cell="sep_central",
        seed=1,
        defects=[
            Defect(iso_code="100", bbox_px=(1.0, 1.0, 2.0, 2.0), score=s)
            for s in scores
        ],
        verdict="불합격" if scores else "합격",
        cited_clauses=[],
        parse_ok=parse_ok,
        parse_error=None if parse_ok else "schema_violation",
    )


# --------------------------------------------------------------------------------------
# 1 — 지름길은 정확히 0
# --------------------------------------------------------------------------------------

def test_상수_점수_예측기는_정확히_0():
    """출처만 읽는 예측기는 구간 안에서 상수가 된다. 근사 0 이 아니라 정확히 0 이어야 한다."""
    sc = dict.fromkeys([f"i{n}" for n in range(50)], 0.7)
    d, n = [f"i{n}" for n in range(30)], [f"i{n}" for n in range(30, 50)]
    assert gini(sc, d, n) == 0.0


def test_전부_미발화도_정확히_0():
    """아무것도 예측하지 않는 칸 — 이것도 상수라 0 이다. `image_score` 가 −1 로 묶는다."""
    recs = [_rec(f"i{n}", []) for n in range(20)]
    sc = {r.image_id: image_score(r) for r in recs}
    assert gini(sc, [f"i{n}" for n in range(10)],
                [f"i{n}" for n in range(10, 20)]) == 0.0


def test_파싱_실패는_어떤_임계에서도_발화하지_않는다():
    r = _rec("x", [0.99], parse_ok=False)
    assert image_score(r) < 0.0
    assert not fires(filter_by_conf([r], 0.01)[0])


# --------------------------------------------------------------------------------------
# 2 — 임계 독립성과 부호
# --------------------------------------------------------------------------------------

def test_완전_분리는_1():
    sc = {"a": 0.9, "b": 0.8, "c": 0.2, "d": 0.1}
    assert gini(sc, ["a", "b"], ["c", "d"]) == pytest.approx(1.0)


def test_완전_역전은_음수_1():
    """정상에서 더 높게 발화하면 음수다 — 0 이 바닥이 아니다."""
    sc = {"a": 0.1, "b": 0.2, "c": 0.8, "d": 0.9}
    assert gini(sc, ["a", "b"], ["c", "d"]) == pytest.approx(-1.0)


def test_단조_변환에_불변():
    """순위만 보는 지표다 — 점수를 단조 변환해도 값이 같아야 한다."""
    sc = {"a": 0.9, "b": 0.4, "c": 0.5, "d": 0.1}
    d, n = ["a", "b"], ["c", "d"]
    warped = {k: v**3 for k, v in sc.items()}
    assert gini(warped, d, n) == pytest.approx(gini(sc, d, n))


def test_동점은_절반으로_센다():
    """결함 1장과 정상 1장이 동점이면 AUROC 0.5 → Δ_AUC 0."""
    assert gini({"a": 0.5, "b": 0.5}, ["a"], ["b"]) == 0.0


def test_한쪽_층이_비면_nan():
    """재표집에서 한쪽이 통째로 안 뽑히는 경우다. 0 으로 대치하면 CI 가 0 쪽으로 끌린다."""
    assert math.isnan(gini({"a": 0.5}, ["a"], []))


# --------------------------------------------------------------------------------------
# 3 — image_score 가 발화 정의의 연속판이다
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("conf", [0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.9, 1.0])
def test_점수_이상이_임계_필터_후_발화와_같다(conf):
    """`image_score(r) >= c` ⟺ `fires(filter_by_conf([r], c)[0])`.

    이 동치가 깨지면 Δ 곡선(임계로 자름)과 Δ_AUC(점수 순위)가 서로 다른 발화 개념을 쓴다.
    """
    recs = [
        _rec("a", [0.9, 0.3]),
        _rec("b", [0.05]),
        _rec("c", []),
        _rec("d", [0.25]),
        _rec("e", [0.5, 0.5]),
        _rec("f", [1.0], parse_ok=False),
    ]
    for r in recs:
        cut = filter_by_conf([r], conf)[0]
        assert (image_score(r) >= conf) == fires(cut), f"{r.image_id} @ {conf}"
