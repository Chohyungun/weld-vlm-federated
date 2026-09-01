"""**단일 채점기.** 다섯 칸이 전부 이 한 함수를 통과한다 (불변조건 3-7).

칸별 채점 코드를 따로 짜는 순간 공정성 주장이 무너지므로, 칸을 구분해도 되는 지점은
어댑터 하나뿐이다(`evaluation/adapters.py`). 어댑터는 원시 출력 형식의 차이만 흡수하고,
계약 #4 레코드가 만들어진 뒤로는 검출 3칸과 통합형 2칸이 **비트 단위로 같은 경로**를 탄다.

65번(검출 3칸)은 이 함수를 `scripts/probe/score_detection_cells.py` 안에 두고 돌렸다.
66번에서 통합형 2칸이 합류하면서 그 사본이 두 벌이 될 위험이 생겨 여기로 올렸다 —
65번의 산출 수치는 바뀌지 않는다(재채점으로 동일성을 확인한다).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from evaluation.metrics.detection import class_jaccard, score_detection
from evaluation.metrics.localization import coco_map, score_bbox_iou
from evaluation.schema import PredictionRecord


def score_records(
    records: Sequence[PredictionRecord],
    gold_codes: Mapping[str, Sequence[str] | set[str]],
    gold_boxes: Mapping[str, Sequence[tuple[str, tuple[float, float, float, float]]]],
    classes: Sequence[str],
) -> dict:
    """계약 #4 레코드 한 벌을 채점한다. 칸 이름으로 분기하는 코드는 여기에 없다.

    파싱 실패 레코드는 `defects=[]` 이므로 자동으로 빈 예측 집합이 되어 **미검출로
    계상된다** — 통계에서 빠지면 오답보다 낙관적으로 잡힌다(스펙 §4-1).
    """
    pred_codes = {r.image_id: sorted(r.iso_codes) for r in records}
    pred_boxes = {
        r.image_id: [(d.iso_code, tuple(d.bbox_px)) for d in r.defects if d.bbox_px]
        for r in records
    }
    pred_scored = {
        r.image_id: [
            (d.iso_code, tuple(d.bbox_px), d.score or 0.0)
            for d in r.defects if d.bbox_px
        ]
        for r in records
    }
    det = score_detection(pred_codes, {k: sorted(v) for k, v in gold_codes.items()}, classes)
    loc = score_bbox_iou(pred_boxes, gold_boxes)
    ap = coco_map(pred_scored, gold_boxes, classes)
    return {
        **det.as_dict(),
        "miss_rate": 1.0 - det.defect_recall,
        "class_jaccard": class_jaccard(pred_codes, gold_codes),
        **loc.as_dict(),
        **ap,
    }


def failure_breakdown(records: Sequence[PredictionRecord]) -> dict:
    """파싱 실패율과 사유별 분해. **오답 처리와 별도로 반드시 보고한다**(불변조건 3-4)."""
    n = len(records)
    counts: dict[str, int] = {}
    for r in records:
        if not r.parse_ok and r.parse_error:
            counts[r.parse_error] = counts.get(r.parse_error, 0) + 1
    n_fail = sum(counts.values())
    return {
        "n_records": n,
        "n_parse_fail": n_fail,
        "parse_fail_rate": n_fail / n if n else 0.0,
        "by_reason": counts,
    }


def coord_health(metrics: Mapping[str, object]) -> dict:
    """좌표계 건강 판정 — 매칭쌍 IoU 대 전체 IoU 분리(65번과 같은 방법).

    전체 IoU 가 낮은 것은 미검출로도 설명되지만, **매칭쌍 IoU 까지 0.0x 로 무너지면**
    그것은 성능이 아니라 좌표계 불일치의 서명이다(함정 #4: 0.938 → 0.055).
    """
    matched = float(metrics.get("bbox_iou_matched_only") or 0.0)
    n_matched = int(metrics.get("n_matched") or 0)
    suspect = bool(metrics.get("coord_suspect"))
    if n_matched == 0:
        verdict = "판정 불가 — 매칭쌍 0건(모델이 해당 클래스를 한 번도 맞히지 못함)"
    elif suspect or matched < 0.10:
        verdict = "붕괴 의심 — 매칭쌍 IoU 가 0.1 이하다. 함정 #4 발현 가능성"
    elif matched < 0.25:
        verdict = "경계 — 매칭쌍 IoU 가 낮다. 오버레이 육안 확인 권고"
    else:
        verdict = "건강 — 매칭쌍 IoU 정상 범위. 전체 IoU 하락은 미검출로 설명된다"
    return {
        "bbox_iou_matched_only": matched,
        "n_matched": n_matched,
        "n_gold": int(metrics.get("n_gold") or 0),
        "coord_suspect": suspect,
        "verdict": verdict,
    }
