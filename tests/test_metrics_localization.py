"""위치 지표 테스트. 스펙 §10 "BBox-IoU" 행.

**박스 순서를 섞어도 같은 값**이 나오는지가 핵심이다 — 채점기가 비결정적이면
`check-scorer` 의 비트 단위 일치 요구가 성립하지 않는다.
"""

from __future__ import annotations

import pytest

from evaluation.metrics.localization import (
    iou,
    match_image,
    score_bbox_iou,
    to_coco_xywh,
)


def test_identical_boxes_iou_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_disjoint_boxes_iou_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)


def test_touching_boxes_iou_zero():
    assert iou([0, 0, 10, 10], [10, 0, 20, 10]) == pytest.approx(0.0)


def test_half_overlap():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_iou_is_not_clipped_to_image():
    """이미지 밖으로 나간 예측은 자연 벌점을 받아야 한다.
    클램프는 IoU 를 올리는 방향으로만 작동해 답을 고쳐주는 셈이다 (§3-4 순서 5)."""
    inside = iou([0, 0, 10, 10], [0, 0, 10, 10])
    spilled = iou([0, 0, 10, 10], [-10, 0, 10, 10])
    assert spilled < inside


# --- 매칭 -----------------------------------------------------------------------

def test_different_class_never_matches():
    m, n_unmatched = match_image([("100", [0, 0, 10, 10])], [("2011", [0, 0, 10, 10])])
    assert [x.matched for x in m] == [False]
    assert m[0].iou == pytest.approx(0.0)
    # 클래스가 다른 예측은 짝을 못 찾은 예측으로 센다 — 위치 축 분모에 들어간다(80번 D6).
    assert n_unmatched == 1


def test_unmatched_gold_included_with_zero():
    """분모에서 빼면 못 찾은 결함이 지표에서 사라진다."""
    m, n_unmatched = match_image([], [("2011", [0, 0, 10, 10])])
    assert len(m) == 1 and m[0].iou == pytest.approx(0.0)
    assert n_unmatched == 0


def test_hungarian_picks_global_optimum_not_greedy():
    """greedy 는 순서에 따라 다른 답을 낸다. 밀집 이미지에서 재현성이 깨진다."""
    gold = [("2011", [0, 0, 10, 10]), ("2011", [100, 100, 110, 110])]
    pred = [("2011", [100, 100, 110, 110]), ("2011", [0, 0, 10, 10])]
    m, n_unmatched = match_image(pred, gold)
    assert all(x.iou == pytest.approx(1.0) for x in m)
    assert n_unmatched == 0


def test_box_order_does_not_change_result():
    gold = {"a": [("2011", [0, 0, 10, 10]), ("2011", [50, 50, 60, 60])]}
    pred_a = {"a": [("2011", [0, 0, 10, 10]), ("2011", [50, 50, 62, 62])]}
    pred_b = {"a": [("2011", [50, 50, 62, 62]), ("2011", [0, 0, 10, 10])]}
    assert score_bbox_iou(pred_a, gold).mean_all == pytest.approx(
        score_bbox_iou(pred_b, gold).mean_all
    )


# --- 집계 -----------------------------------------------------------------------

def test_two_definitions_reported_separately():
    """미매칭 포함 평균과 매칭쌍 평균은 다른 수다. 정의를 안 밝히면 같은 이름을 단다."""
    gold = {"a": [("2011", [0, 0, 10, 10]), ("301", [0, 0, 10, 10])]}
    pred = {"a": [("2011", [0, 0, 10, 10])]}
    r = score_bbox_iou(pred, gold)
    assert r.mean_matched == pytest.approx(1.0)
    assert r.mean_all == pytest.approx(0.5)
    assert (r.n_gold, r.n_matched) == (2, 1)


def test_coord_suspect_flag_on_collapse():
    """IoU 0.9대 → 0.05대 붕괴의 자동 신호. 실패가 아니라 플래그다."""
    gold = {"a": [("2011", [0, 0, 100, 100])]}
    pred = {"a": [("2011", [98, 98, 200, 200])]}
    assert score_bbox_iou(pred, gold).coord_suspect is True


def test_coord_suspect_off_on_healthy_scores():
    gold = {"a": [("2011", [0, 0, 100, 100])]}
    pred = {"a": [("2011", [2, 2, 98, 98])]}
    assert score_bbox_iou(pred, gold).coord_suspect is False


def test_empty_gold_is_safe():
    assert score_bbox_iou({}, {}).n_gold == 0


# --- COCO 변환 --------------------------------------------------------------------

def test_coco_conversion_is_zero_based_wh():
    """Pascal VOC `+1` 규약이 섞이면 mAP 만 조용히 어긋난다."""
    assert to_coco_xywh([10.0, 20.0, 40.0, 55.0]) == (10.0, 20.0, 30.0, 35.0)


def test_tiny_box_survives_conversion():
    """슬래그는 수~수십 px 다. 초소형에서 무너지면 최소 클래스만 왜곡된다."""
    assert to_coco_xywh([10.0, 10.0, 11.0, 11.0]) == (10.0, 10.0, 1.0, 1.0)
