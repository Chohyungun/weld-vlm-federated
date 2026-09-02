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

COLLAPSE_ZERO_FRAC = 0.90
"""배정쌍 중 겹침 0 의 비율이 이 이상이면 좌표계 붕괴로 본다.

0.9 인 이유: 정상 학습된 모델도 어려운 이미지에서 완전 빗나간 배정을 몇 건 낸다.
붕괴는 그것과 달리 **거의 전부**가 겹침 0 이다 — 규약이 어긋나면 예외가 없다.
"""


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
    # 신뢰도가 하나라도 실재하는가. 생성형 칸은 전부 None 이라 mAP 를 내지 않는다(D16).
    scores_present = any(
        d.score is not None for r in records for d in r.defects if d.bbox_px
    )
    det = score_detection(pred_codes, {k: sorted(v) for k, v in gold_codes.items()}, classes)
    loc = score_bbox_iou(pred_boxes, gold_boxes)
    ap = coco_map(pred_scored, gold_boxes, classes, scores_present=scores_present)
    return {
        **det.as_dict(),
        "miss_rate": 1.0 - det.defect_recall,
        "class_jaccard": class_jaccard(pred_codes, gold_codes, classes),
        "scores_present": scores_present,
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
    """좌표계 건강 판정 — **"좌표계가 무너졌다"와 "위치를 못 맞혔다"를 가른다.**

    이전 판은 `bbox_iou_matched_only` 와 `n_matched` 만 봤다. Hungarian 이 겹침 0 인 쌍도
    매칭으로 배정했으므로 붕괴가 "낮은 매칭 IoU"로 섞여 들어왔고, 겹침 0 을 매칭에서 빼자
    이번엔 `n_matched=0` → "판정 불가"로 이름만 바꿔 다시 숨었다(80번 D5의 재검 경고).

    그래서 **배정 통계를 본다.** 클래스가 맞아 쌍으로 묶였는데(`n_assigned`) 겹침이
    거의 전무하면(`zero_overlap_frac`) 그것은 성능이 아니라 규약이다 — 모델이 맞는
    클래스를 맞는 이미지에서 찾아 놓고 위치만 통째로 어긋난 상태다. 반대로 애초에
    예측이 거의 없으면(`n_pred` 작음) 좌표를 판정할 재료가 없는 것이지 붕괴가 아니다.
    """
    matched = float(metrics.get("bbox_iou_matched_only") or 0.0)
    n_matched = int(metrics.get("n_matched") or 0)
    n_assigned = int(metrics.get("n_assigned") or 0)
    n_zero = int(metrics.get("n_zero_overlap_assigned") or 0)
    n_pred = int(metrics.get("n_pred") or 0)
    n_gold = int(metrics.get("n_gold") or 0)
    zero_frac = n_zero / n_assigned if n_assigned else 0.0
    suspect = bool(metrics.get("coord_suspect"))

    if n_pred == 0:
        verdict = "판정 불가 — 예측 박스 0건. 좌표를 잴 재료가 없다(미검출)"
    elif n_assigned == 0:
        verdict = (
            "판정 불가 — 클래스가 맞는 쌍이 0건. 좌표 이전에 분류가 안 맞았다"
        )
    elif zero_frac >= COLLAPSE_ZERO_FRAC:
        verdict = (
            f"붕괴 의심 — 클래스가 맞아 묶인 {n_assigned}쌍 중 {n_zero}쌍"
            f"({zero_frac:.1%})이 겹침 0 이다. 위치가 통째로 어긋난 서명(함정 #4)"
        )
    elif suspect or matched < 0.10:
        verdict = "붕괴 의심 — 매칭쌍 IoU 가 0.1 이하다. 함정 #4 발현 가능성"
    elif matched < 0.25:
        verdict = "경계 — 매칭쌍 IoU 가 낮다. 오버레이 육안 확인 권고"
    else:
        verdict = "건강 — 매칭쌍 IoU 정상 범위. 전체 IoU 하락은 미검출로 설명된다"
    return {
        "bbox_iou_matched_only": matched,
        "n_matched": n_matched,
        "n_assigned": n_assigned,
        "n_zero_overlap_assigned": n_zero,
        "zero_overlap_frac": zero_frac,
        "n_pred": n_pred,
        "n_gold": n_gold,
        "coord_suspect": suspect,
        "verdict": verdict,
    }
