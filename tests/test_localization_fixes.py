"""위치 지표 수정 — 체크리스트 10·11·12 (80번 D5·D6·D9·D16).

세 성질을 **시험으로 고정한다.** 셋 다 이전 정의에서는 반드시 실패해야 하는 것들이라,
회귀가 나면 여기서 먼저 깨진다.
"""

from __future__ import annotations

import pytest

from evaluation.metrics.localization import (
    NOT_APPLICABLE,
    coco_map,
    match_image,
    score_bbox_iou,
)
from evaluation.score import coord_health

G = (10.0, 10.0, 20.0, 20.0)


def _one(pred_boxes, gold_boxes=(("2011", G),)):
    return score_bbox_iou({"i": list(pred_boxes)}, {"i": list(gold_boxes)})


# --------------------------------------------------------------------------------------
# 11 — 예측 수에 대한 단조성 제거
# --------------------------------------------------------------------------------------

def test_extra_predictions_are_penalised() -> None:
    """정답 박스에 **엉뚱한 박스를 더 얹으면 값이 내려가야 한다.**

    이전 정의는 분모가 GT 수로 고정이라 정답 하나만 맞으면 1.0 이었고, 쓰레기를
    아무리 더 얹어도 1.0 이었다(80번 D6).
    """
    perfect = _one([("2011", G)])
    plus_junk = _one([("2011", G), ("2011", (500.0, 500.0, 520.0, 520.0))])
    assert perfect.mean_penalized == pytest.approx(1.0)
    assert plus_junk.mean_penalized < perfect.mean_penalized
    assert plus_junk.mean_penalized == pytest.approx(0.5)
    # 옛 정의는 여전히 1.0 이다 — 이름을 바꿔 남겨 둔 이유가 이 대조다.
    assert plus_junk.mean_all == pytest.approx(1.0)


def test_random_boxes_do_not_climb() -> None:
    """무작위 박스를 늘려도 값이 오르지 않는다. 이전 정의는 10박스에서 0.6123 이었다."""
    prev = None
    for n in (1, 3, 10, 30):
        boxes = [("2011", (100.0 * i, 100.0 * i, 100.0 * i + 10, 100.0 * i + 10))
                 for i in range(1, n + 1)]
        v = _one(boxes).mean_penalized
        if prev is not None:
            assert v <= prev + 1e-12, f"n={n} 에서 값이 올랐다"
        prev = v


def test_false_positives_on_normal_images_count() -> None:
    """정상 이미지(GT 0건)의 오탐이 분모에 들어간다 (80번 D9).

    이전에는 `gold` 키에 정상 이미지가 없어 위치 축이 오탐에 **면역**이었다.
    """
    rep = score_bbox_iou(
        {"defect": [("2011", G)], "normal": [("2011", (0.0, 0.0, 5.0, 5.0))]},
        {"defect": [("2011", G)], "normal": []},
    )
    assert rep.n_pred == 2
    assert rep.n_pred_unmatched == 1
    assert rep.mean_penalized == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# 10 — 겹침 0 배정
# --------------------------------------------------------------------------------------

def test_zero_overlap_pair_is_not_a_match() -> None:
    """Hungarian 이 묶어도 겹침이 0 이면 매칭이 아니다 (80번 D5)."""
    matches, n_unmatched = match_image(
        [("2011", (900.0, 900.0, 910.0, 910.0))], [("2011", G)], "i")
    assert len(matches) == 1
    m = matches[0]
    assert m.assigned is True
    assert m.matched is False
    assert m.iou == 0.0
    assert n_unmatched == 0


def test_zero_overlap_is_still_counted_not_hidden() -> None:
    """매칭에서 빼되 **통계로는 남는다.** 안 그러면 붕괴가 '판정 불가'로 숨는다."""
    rep = _one([("2011", (900.0, 900.0, 910.0, 910.0))])
    assert rep.n_matched == 0
    assert rep.n_assigned == 1
    assert rep.n_zero_overlap_assigned == 1
    assert rep.zero_overlap_frac == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# 10 — `coord_health` 판별력: 붕괴와 저성능이 다른 판정을 받는가
# --------------------------------------------------------------------------------------

def _collapsed_inputs(n: int = 40):
    """좌표계 붕괴 — 클래스·개수는 맞는데 위치가 통째로 어긋난다(규약 불일치)."""
    gold = {f"i{k}": [("2011", (100.0, 100.0, 200.0, 200.0))] for k in range(n)}
    pred = {f"i{k}": [("2011", (0.1, 0.1, 0.2, 0.2))] for k in range(n)}
    return pred, gold


def _low_performance_inputs(n: int = 40):
    """저성능 — 대부분 못 찾지만 찾은 것은 제대로 겹친다."""
    gold = {f"i{k}": [("2011", (100.0, 100.0, 200.0, 200.0))] for k in range(n)}
    pred = {f"i{k}": ([("2011", (105.0, 105.0, 195.0, 195.0))] if k < 5 else [])
            for k in range(n)}
    return pred, gold


def test_collapse_and_low_performance_get_different_verdicts() -> None:
    """**판별력 시험.** 두 입력이 같은 판정을 받으면 진단이 목적을 수행하지 못한다."""
    c = coord_health(score_bbox_iou(*_collapsed_inputs()).as_dict())
    lo = coord_health(score_bbox_iou(*_low_performance_inputs()).as_dict())
    assert "붕괴" in c["verdict"], c["verdict"]
    assert "붕괴" not in lo["verdict"], lo["verdict"]
    assert c["verdict"] != lo["verdict"]


def test_collapse_is_not_labelled_unjudgeable() -> None:
    """붕괴가 '판정 불가'로 이름만 바꿔 숨지 않는다 (재검 경고 그대로)."""
    c = coord_health(score_bbox_iou(*_collapsed_inputs()).as_dict())
    assert "판정 불가" not in c["verdict"], c["verdict"]
    assert c["n_assigned"] == 40
    assert c["zero_overlap_frac"] == pytest.approx(1.0)


def test_no_prediction_is_unjudgeable_not_collapse() -> None:
    """예측이 없으면 좌표를 잴 재료가 없는 것이지 붕괴가 아니다."""
    gold = {f"i{k}": [("2011", G)] for k in range(10)}
    h = coord_health(score_bbox_iou({}, gold).as_dict())
    assert "판정 불가" in h["verdict"]
    assert "붕괴" not in h["verdict"]


# --------------------------------------------------------------------------------------
# 12 — mAP 모집단과 score
# --------------------------------------------------------------------------------------

def test_map_is_not_applicable_without_scores() -> None:
    """신뢰도 없는 칸의 mAP 는 산출하지 않는다 (80번 D16)."""
    gold = {"i": [("2011", G)]}
    out = coco_map({"i": [("2011", G, 0.0)]}, gold, ["2011"], scores_present=False)
    assert out["map_50"] == NOT_APPLICABLE
    assert out["map_50_95"] == NOT_APPLICABLE
    assert "신뢰도" in out["note"]


def test_map_population_includes_normal_images() -> None:
    """정상 이미지가 gold 키에 있으면 mAP 모집단에 들어간다."""
    gold = {"d": [("2011", G)], "n": []}
    out = coco_map({"d": [("2011", G, 0.9)], "n": [("2011", G, 0.9)]}, gold, ["2011"])
    assert out["n_pred_boxes"] == 2, "정상 이미지 오탐이 mAP 입력에서 빠졌다"
