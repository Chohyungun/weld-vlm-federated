"""서술 지표 틀(`evaluation.narrative_metrics`) — 분모 · 해당 없음 · 설계 가중 · 보조 둘 · 추첨의 미정의 · 대응 차 방향.

검사기는 가짜다 — 서술 글마다 (걸린 종류, 필요 사유, 인정된 사유)를 미리 정해 둔다. 기대는 아래 일곱 이미지에서 손으로 셈했다.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import narrative_metrics as NM
from evaluation import prereg_unified as P
from evaluation.narrative_extract import NarrativeExtract

TABLE = {
    "n1": ({"나"}, {"r_dim", "r_note"}, {"r_dim"}),
    "n2": (set(), {"r_dim"}, {"r_dim"}),
    "n3": ({"가", "나"}, {"r_dim"}, set()),
    "n7": (set(), set(), set()),
}


class Fake:
    def check(self, narrative, *, material, gold_codes, structured_codes):
        f, r, found = TABLE[narrative]
        return SimpleNamespace(flagged=f, required=r, found=found)

    def required_for(self, material, codes):
        return {"r_dim"} if codes else set()


def ex(state, narrative=None, front=None, stop=None):
    return NarrativeExtract(state, narrative, front, stop, "json_decode" if state == "record_fail" else None)


ITEMS = [
    NM.ImageItem("i1", ex("complete", "n1", True, "eos"), "ST", ("2011",), ("2011",), weight=2.0),
    NM.ImageItem("i2", ex("complete", "n2", False, "eos"), "ST", ("2011",), ("2011",)),
    NM.ImageItem("i3", ex("truncated", "n3", False, "length"), "ST", ("2011",), ("2011",)),
    NM.ImageItem("i4", ex("missing", "", False, "eos"), "ST", ("2011",), ()),
    NM.ImageItem("i5", ex("record_fail"), "ST", ("2011",), ()),
    NM.ImageItem("i6", ex("validation_error"), "ST", (), ()),
    NM.ImageItem("i7", ex("complete", "n7", False, "eos"), "ST", (), ()),
]


@pytest.fixture(scope="module")
def rows():
    return NM.build_rows(ITEMS, Fake())


def test_분모와_해당_없음(rows) -> None:
    p = NM.point_estimates(rows)
    u = {k: v["unweighted"] for k, v in p.items() if isinstance(v, dict) and "unweighted" in v}
    assert (u["forbidden.나"]["value"], u["forbidden.나"]["n_eligible"]) == (0.5, 4)
    assert u["forbidden.가"]["value"] == 0.25 and u["forbidden.any"]["value"] == 0.5
    assert u["forbidden.다"] == {"value": 0.0, "state": "finite", "numerator": 0.0, "denominator": 4.0, "n_eligible": 4}
    assert u["reservation.r_dim"]["value"] == 1.0 and u["reservation.r_note"]["value"] == 0.0
    assert u["reservation.r_scope"] == {"value": None, "state": "해당 없음", "numerator": 0.0, "denominator": 0.0,
                                        "n_eligible": 0}
    assert u["reservation.all"]["value"] == 0.5 and u["reservation.all"]["n_eligible"] == 2
    assert u["reservation_gold.r_dim"]["value"] == 1.0 and u["reservation_gold.r_dim"]["n_eligible"] == 2
    assert u["state.complete"]["value"] == pytest.approx(3 / 7) and u["state.validation_error"]["value"] == pytest.approx(1 / 7)
    assert u["front_text"]["value"] == pytest.approx(1 / 5)                 # 분모 = 구조 유효(미기재 · 잘림 · 완결)
    assert p["aux.required_size_distribution"] == {"0": 1, "1": 1, "2": 1}
    assert p["counts"]["n_images"] == 7 and p["counts"]["n_complete"] == 3


def test_설계_가중(rows) -> None:
    w = NM.point_estimates(rows)["forbidden.나"]["weighted"]
    assert (w["numerator"], w["denominator"], w["value"]) == (3.0, 5.0, 0.6)


def test_서술_없는_이미지는_검사하지_않는다() -> None:
    class Loud(Fake):
        def check(self, narrative, **kw):
            assert narrative in TABLE
            return super().check(narrative, **kw)

    NM.build_rows(ITEMS, Loud())


def test_검사기가_모르는_이름을_내면_멈춘다() -> None:
    class Bad(Fake):
        def check(self, narrative, **kw):
            return SimpleNamespace(flagged={"바"}, required=set(), found=set())

    with pytest.raises(ValueError, match="모르는 이름"):
        NM.build_rows(ITEMS, Bad())


def test_추첨의_분모가_0_이면_NaN(rows) -> None:
    w_point = np.array([[r.weight for r in rows]])
    no_complete = np.array([[0.0 if r.state == "complete" else 1.0 for r in rows]])
    d = NM.draws(rows, np.vstack([w_point, no_complete]))
    assert d["forbidden.나"][0] == pytest.approx(0.6)
    assert math.isnan(d["reservation.all"][1]) and not math.isnan(d["reservation.all"][0])
    with pytest.raises(ValueError):
        NM.draws(rows, np.ones((2, 3)))


def test_대응_차는_둘_다_정의된_추첨에서만이고_하한_밑이면_방향을_말하지_않는다() -> None:
    pa = {"value": 0.6, "state": "finite"}
    pb = {"value": 0.2, "state": "finite"}
    a = np.array([0.6, 0.7, np.nan, 0.5, 0.65])
    b = np.array([0.2, 0.1, 0.3, np.nan, 0.25])
    got = NM.paired_difference(pa, pb, a, b, alpha=0.05, min_defined_draws=3)
    assert got["interval"]["n_defined"] == 3 and got["interval"]["n_undefined"] == 2
    assert got["direction"] == "위" and got["diff"] == pytest.approx(0.4)
    short = NM.paired_difference(pa, pb, a, b, alpha=0.05, min_defined_draws=4)
    assert short["direction"] is None and "하한" in short["direction_reason"]
    none = NM.paired_difference({"value": None, "state": "해당 없음"}, pb, a, b, alpha=0.05, min_defined_draws=1)
    assert none["state"] == "해당 없음" and none["interval"] is None


def test_이름_목록은_등록_어휘_및_엄격_경로의_방향_낱말과_같다() -> None:
    """같은 목록을 두 곳에 적었다 — 갈리면 등록이 받는 이름과 틀이 내는 이름이 달라진다."""
    from evaluation.strict_main import DIRECTION_KO

    assert NM.KINDS == P.NARRATIVE_KINDS and NM.REASONS == P.NARRATIVE_REASONS
    assert NM.DIRECTION_KO == DIRECTION_KO
