"""표본 설계 가중과 대표 지표 M1 — `evaluation/sample_weights.py`(25번 §2-5 · 실험설계 8-5). 합성 입력만 쓴다."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import sample_weights as SW  # noqa: E402
from evaluation.eval_list import canonical_bytes  # noqa: E402
from evaluation.metrics.box_f1 import SENSITIVITY_REGISTERED, build_counts, score_m1  # noqa: E402

C = ("100", "2011", "301", "401")


def box(x, y, w=10, h=10):
    return (float(x), float(y), float(x + w), float(y + h))


IDS = ("a", "b", "c", "d")
STRATA = {"a": ("ST|1", 10, 2), "b": ("ST|1", 10, 2), "c": ("ST|0", 3, 1), "d": ("AL|2-3", 4, 4)}


def side_bytes(ids=IDS, strata=STRATA, header=SW.SIDE_HEADER) -> bytes:
    rows = [header] + [f"{i},{strata[i][0]},{strata[i][1]},{strata[i][2]}" for i in ids]
    return ("\n".join(rows) + "\n").encode("utf-8")


def table():
    gold = {"a": [("2011", box(0, 0))], "b": [("100", box(0, 0))], "c": [], "d": [("2011", box(0, 0))]}
    pred = {"a": [("2011", box(0, 0))], "b": [], "c": [("2011", box(50, 50))], "d": [("2011", box(0, 0))]}
    return build_counts(pred, gold, C), {i: {c for c, _ in g} for i, g in gold.items()}


def test_곁_파일을_읽고_가중은_G_over_n() -> None:
    side = SW.read_side(side_bytes(), canonical_bytes(IDS))
    assert side["a"].weight == Fraction(10, 2) and side["c"].weight == Fraction(3, 1) and side["d"].weight == 1
    t, _ = table()
    w = SW.weights_for(t, side)
    assert list(w) == [5.0, 5.0, 3.0, 1.0]


@pytest.mark.parametrize("bad", [
    side_bytes(header="image_id,stratum,G,n"),
    side_bytes().replace(b"\n", b"\r\n"),
    side_bytes()[:-1],
    b"\xef\xbb\xbf" + side_bytes(),
    side_bytes(strata={**STRATA, "b": ("ST|1", 10, 3)}),            # 같은 층에 다른 n_h
    side_bytes(strata={**STRATA, "d": ("AL|2-3", 4, 5)}),           # n_h > G_h
    side_bytes(strata={**STRATA, "d": ("AL|2-3", 4, 0)}),           # n_h = 0
    side_bytes(strata={**STRATA, "d": ("AL|2-3", "04", 4)}),        # 10진 정규형이 아님
    side_bytes(ids=("b", "a", "c", "d")),                             # 목록과 순서가 다름
    side_bytes(ids=("a", "b", "c")),                                  # 목록보다 짧음
])
def test_어긋난_곁_파일은_거부한다(bad) -> None:
    with pytest.raises(SW.SampleSideError):
        SW.read_side(bad, canonical_bytes(IDS))


def test_계수표와_표본의_집합이_다르면_거부한다() -> None:
    t, _ = table()
    side = SW.read_side(side_bytes(ids=("a", "b", "c")), canonical_bytes(("a", "b", "c")))
    with pytest.raises(SW.SampleSideError):
        SW.weights_for(t, side)


def test_대표_지표는_가중_M1_과_가중_없는_값을_함께() -> None:
    t, gold_codes = table()
    side = SW.read_side(side_bytes(), canonical_bytes(IDS))
    out = SW.representative(t, gold_codes=gold_codes, failed_image_ids=[], side=side)
    assert out["representative_metric"] == SW.REPRESENTATIVE_METRIC and out["weighting"] == SW.WEIGHTING_SAMPLE
    w = np.array([5.0, 5.0, 3.0, 1.0])
    assert out["main"]["m1_macro_f1"] == score_m1(t, w).macro_f1
    assert out["main_unweighted"]["m1_macro_f1"] == score_m1(t).macro_f1
    # 손 계산: 기공 TP 5+1, FP 3, FN 0 → 12/15 · 균열 TP 0, FN 5 → 0 · 나머지 둘은 지지량 0 이라 평균 밖
    assert out["main"]["per_class_f1"]["2011"] == pytest.approx(12 / 15)
    assert out["main"]["m1_macro_f1"] == pytest.approx((12 / 15 + 0) / 2)
    assert out["main"]["classes_in_macro"] == ["100", "2011"]


def test_가중이_전부_1_이면_가중_없는_값과_같다() -> None:
    t, gold_codes = table()
    ones = {i: (s, 1, 1) for i, (s, _, _) in STRATA.items()}
    side = SW.read_side(side_bytes(strata=ones), canonical_bytes(IDS))
    out = SW.representative(t, gold_codes=gold_codes, failed_image_ids=["c"], side=side)
    assert out["main"] == out["main_unweighted"] and out["sensitivity"] == out["sensitivity_unweighted"]


def test_민감도는_등록_변형으로_같은_가중() -> None:
    t, gold_codes = table()
    side = SW.read_side(side_bytes(), canonical_bytes(IDS))
    out = SW.representative(t, gold_codes=gold_codes, failed_image_ids=["b", "c"], side=side)
    assert out["sensitivity_variant"] == SENSITIVITY_REGISTERED
    # 실패 b(결함) · c(정상) 에 클래스마다 오탐 1, 가중 5 · 3 → 기공 FP 3 + 5 + 3 = 11
    assert out["sensitivity"]["per_class_counts"]["2011"]["fp"] == pytest.approx(11.0)
    assert out["sensitivity"]["per_class_counts"]["100"]["fn"] == pytest.approx(5.0), "미탐은 주 채점 그대로"


def test_전량이면_가중_없음() -> None:
    t, gold_codes = table()
    out = SW.representative(t, gold_codes=gold_codes, failed_image_ids=[], side=None)
    assert out["weighting"] == SW.WEIGHTING_NONE and "main_unweighted" not in out
    assert out["main"]["m1_macro_f1"] == score_m1(t).macro_f1
