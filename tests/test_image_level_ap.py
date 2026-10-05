"""이미지 수준 macro-AP — Macro-F1 의 임계 독립 대응물 (판정 1 · 22번 §1-2).

Macro-F1 은 운용 임계 한 점의 값이라 임계가 바뀌면 칸의 **순서까지** 뒤집혔다
(17번 §3-5). 이 지표는 전 구간을 적분하므로 그 자리에서 흔들리지 않아야 하고,
**그 불변성 자체가 첫 시험이다.**
"""

from __future__ import annotations

import pytest

from evaluation.metrics.detection import image_level_ap

CLASSES = ("100", "2011")


def _ap(scored, gold, classes=CLASSES):
    return image_level_ap(scored, gold, classes)


# --------------------------------------------------------------------------------------
# 임계 독립성 — 존재 이유
# --------------------------------------------------------------------------------------

def test_점수_하한을_올려도_AP_는_변하지_않는다():
    """운용 임계로 미리 자른 입력과 자르지 않은 입력이 같은 AP 를 준다 —
    잘린 것이 전부 순위 아래쪽인 한. 이것이 '임계 독립'의 내용이다."""
    gold = {"a": {"100"}, "b": {"100"}, "c": set(), "d": set()}
    full = {"a": [("100", 0.9)], "b": [("100", 0.6)],
            "c": [("100", 0.05)], "d": [("100", 0.02)]}
    cut = {k: [(c, s) for c, s in v if s >= 0.1] for k, v in full.items()}
    assert _ap(full, gold)["per_class_ap"]["100"] == pytest.approx(1.0)
    assert _ap(cut, gold)["per_class_ap"]["100"] == pytest.approx(1.0)


def test_Macro_F1_이_뒤집히는_상황에서_AP_는_순서를_유지한다():
    """모델 A 는 저신뢰 박스를 많이 내고 B 는 적게 낸다. 높은 임계에서는 B 가 이기지만
    순위 품질은 A 가 낫다 — AP 가 그 사실을 임계 없이 말한다."""
    gold = {f"p{i}": {"100"} for i in range(4)} | {f"n{i}": set() for i in range(4)}
    # A: 양성을 전부 상위에 둔다(순위 완벽), 점수 자체는 낮다
    a = {**{f"p{i}": [("100", 0.30 - 0.01 * i)] for i in range(4)},
         **{f"n{i}": [("100", 0.20 - 0.01 * i)] for i in range(4)}}
    # B: 점수는 높지만 음성 하나가 최상위에 온다(순위 나쁨)
    b = {"n0": [("100", 0.99)], **{f"p{i}": [("100", 0.90 - 0.01 * i)] for i in range(4)},
         **{f"n{i}": [("100", 0.10)] for i in range(1, 4)}}
    ap_a = _ap(a, gold)["per_class_ap"]["100"]
    ap_b = _ap(b, gold)["per_class_ap"]["100"]
    assert ap_a == pytest.approx(1.0)
    assert ap_a > ap_b


# --------------------------------------------------------------------------------------
# 정의 — macro 규칙은 score_detection 과 같다
# --------------------------------------------------------------------------------------

def test_GT_양성이_없는_클래스는_평균에서_빠지고_기록된다():
    gold = {"a": {"100"}, "b": set()}
    out = _ap({"a": [("100", 0.9), ("2011", 0.8)]}, gold)
    assert out["per_class_ap"]["2011"] is None
    assert out["skipped_classes"] == ["2011"]
    assert out["macro_ap"] == pytest.approx(out["per_class_ap"]["100"])


def test_완벽한_순위는_1_이고_역순은_낮다():
    gold = {"a": {"100"}, "b": {"100"}, "c": set(), "d": set()}
    perfect = {"a": [("100", 0.9)], "b": [("100", 0.8)],
               "c": [("100", 0.2)], "d": [("100", 0.1)]}
    reverse = {"a": [("100", 0.1)], "b": [("100", 0.2)],
               "c": [("100", 0.8)], "d": [("100", 0.9)]}
    assert _ap(perfect, gold)["per_class_ap"]["100"] == pytest.approx(1.0)
    assert _ap(reverse, gold)["per_class_ap"]["100"] < 0.5


def test_예측이_전혀_없으면_AP_는_0_이다():
    """양성이 있는데 아무것도 못 냈다 — 재현율 0. 분모에서 빼면 낙관적으로 잡힌다."""
    gold = {"a": {"100"}, "b": set()}
    assert _ap({}, gold)["per_class_ap"]["100"] == pytest.approx(0.0)


def test_이미지_점수는_그_클래스_박스의_최댓값이다():
    gold = {"a": {"100"}, "b": set()}
    out = _ap({"a": [("100", 0.1), ("100", 0.95)], "b": [("100", 0.5)]}, gold)
    assert out["per_class_ap"]["100"] == pytest.approx(1.0)   # 0.95 > 0.5 라 완벽한 순위


def test_다른_클래스_점수는_섞이지_않는다():
    gold = {"a": {"100"}, "b": set()}
    out = _ap({"a": [("2011", 0.99)], "b": [("100", 0.5)]}, gold)
    assert out["per_class_ap"]["100"] == pytest.approx(0.0)   # 100 은 b(음성)만 점수가 있다


def test_동점은_한_덩어리로_반영돼_입력_순서에_의존하지_않는다():
    gold = {"a": {"100"}, "b": set()}
    x = _ap({"a": [("100", 0.5)], "b": [("100", 0.5)]}, gold)["per_class_ap"]["100"]
    y = _ap({"b": [("100", 0.5)], "a": [("100", 0.5)]}, gold)["per_class_ap"]["100"]
    assert x == pytest.approx(y) == pytest.approx(0.5)


def test_채점_클래스_밖_코드는_무시된다():
    gold = {"a": {"100"}, "b": set()}
    out = _ap({"a": [("100", 0.9), ("9999", 0.99)]}, gold)
    assert set(out["per_class_ap"]) == set(CLASSES)


# --------------------------------------------------------------------------------------
# 단일 채점기 배선
# --------------------------------------------------------------------------------------

def test_score_records_가_macro_ap_를_싣는다():
    from evaluation.schema import SCHEMA_VERSION, PredictionRecord
    from evaluation.score import score_records

    recs = [
        PredictionRecord(
            schema_version=SCHEMA_VERSION, image_id=i, cell="sep_fed", seed=0,
            defects=[{"iso_code": "100", "bbox_px": [0.0, 0.0, 10.0, 10.0], "score": s}]
            if s is not None else [],
            verdict="판정불가", cited_clauses=[], parse_ok=True, coord_space="ABS_ORIG",
        )
        for i, s in (("a", 0.9), ("b", 0.2), ("c", None))
    ]
    gold_codes = {"a": {"100"}, "b": set(), "c": set()}
    gold_boxes = {"a": [("100", (0.0, 0.0, 10.0, 10.0))], "b": [], "c": []}
    out = score_records(recs, gold_codes, gold_boxes, ["100"])
    assert out["macro_ap"] == pytest.approx(1.0)
    assert out["per_class_ap"]["100"] == pytest.approx(1.0)


def test_점수가_없는_칸은_macro_ap_가_None_이다():
    """생성형은 신뢰도를 내지 않는다 — 순위가 없으므로 0 이 아니라 정의되지 않음이다."""
    from evaluation.schema import SCHEMA_VERSION, PredictionRecord
    from evaluation.score import score_records

    recs = [PredictionRecord(
        schema_version=SCHEMA_VERSION, image_id="a", cell="uni_fed", seed=0,
        defects=[{"iso_code": "100", "bbox_px": [0.0, 0.0, 10.0, 10.0], "score": None}],
        verdict="합격", cited_clauses=[], parse_ok=True, coord_space="ABS_ORIG")]
    out = score_records(recs, {"a": {"100"}}, {"a": [("100", (0.0, 0.0, 10.0, 10.0))]}, ["100"])
    assert out["scores_present"] is False
    assert out["macro_ap"] is None
