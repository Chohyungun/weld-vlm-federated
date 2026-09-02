"""위치 지표 — BBox-IoU. 스펙 §4-6.

좌표는 **원본 이미지 픽셀**로 들어온다. C가 역변환을 끝내 제출하므로 이 모듈에는 좌표
변환 산식이 없다(§3-4). `letterbox`·`smart_resize`·`1000` 같은 상수가 이 파일에 등장하면
코드 리뷰에서 거부한다 — 그것이 이중 역변환을 구조적으로 막는 장치다.

매칭은 **Hungarian 1:1**이다. greedy 를 쓰지 않는 이유는 결함이 밀집한 이미지에서 순서에
따라 다른 답을 내기 때문이다 — 채점기가 비결정적이면 `check-scorer` 의 비트 단위 일치
요구가 성립하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Sequence[float]


def iou(a: Box, b: Box) -> float:
    """xyxy 두 박스의 IoU. **클리핑하지 않는다** — 이미지 밖으로 나간 예측은 자연 벌점을
    받아야 하고, 클램프는 IoU 를 올리는 방향으로만 작동해 답을 고쳐주는 셈이 된다(§3-4)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


MATCH_MIN_IOU = 0.0
"""매칭으로 인정하는 겹침의 **하한(초과)**. 겹침이 정확히 0 인 배정은 매칭이 아니다.

Hungarian 은 비용이 같으면 아무 쌍이나 배정하므로, 겹침 0 인 쌍도 `matched=True` 로
들어왔다. 함정 #4(좌표계 붕괴)를 잡으라고 만든 `bbox_iou_matched_only` 가 그 탓에
"위치를 못 맞힘"과 "좌표계가 무너짐"을 구분하지 못했다(80번 D5).

**하한을 0 초과로 두는 것만으로는 부족하다.** 겹침 0 을 매칭에서 빼면 진짜 붕괴가
`n_matched=0` → "판정 불가"로 이름만 바꿔 다시 숨는다. 그래서 배정 자체의 통계
(`n_assigned`·`n_zero_overlap_assigned`)를 함께 남기고, `coord_health` 판정 트리가
그 둘을 본다. 지표 수정과 판정 트리 수정은 한 묶음이다.
"""


@dataclass(frozen=True)
class Match:
    image_id: str
    iso_code: str
    iou: float
    matched: bool
    """겹침이 실제로 있는 매칭. `assigned` 와 다르다."""
    assigned: bool = False
    """Hungarian 이 쌍으로 묶었는가. 겹침 0 이어도 True 일 수 있다."""


def match_image(
    pred: Sequence[tuple[str, Box]],
    gold: Sequence[tuple[str, Box]],
    image_id: str = "",
) -> tuple[tuple[Match, ...], int]:
    """같은 `iso_code` 안에서만 Hungarian 1:1 매칭한다.

    클래스가 틀리면 위치가 맞아도 매칭하지 않는다. **미매칭 GT 는 IoU 0 으로 남긴다** —
    분모에서 빼면 못 찾은 결함이 지표에서 사라진다.

    Returns:
        (GT 기준 매칭 목록, **짝을 못 찾은 예측 박스 수**). 두 번째 값이 새로 생겼다 —
        남는 예측에 벌점을 매기려면 그 개수를 알아야 한다(80번 D6).
    """
    out: list[Match] = []
    n_pred_unmatched = 0
    codes = sorted({c for c, _ in gold} | {c for c, _ in pred})
    for code in codes:
        p = [b for c, b in pred if c == code]
        g = [b for c, b in gold if c == code]
        if not g:
            # GT 가 없는 클래스의 오검출. 검출 지표(FP)가 잡지만 **위치 축에서도
            # 분모에 들어가야 한다** — 안 그러면 정상 이미지 오탐이 위치 지표에 면역이다.
            n_pred_unmatched += len(p)
            continue
        if not p:
            out.extend(Match(image_id, code, 0.0, False, False) for _ in g)
            continue
        cost = np.zeros((len(g), len(p)), dtype=float)
        for i, gb in enumerate(g):
            for j, pb in enumerate(p):
                cost[i, j] = -iou(gb, pb)
        rows, cols = linear_sum_assignment(cost)
        assigned_g = set(rows.tolist())
        assigned_p = set(cols.tolist())
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            v = float(-cost[i, j])   # numpy 스칼라를 흘리지 않는다 — jsonl 직렬화·`is False` 대조
            out.append(Match(image_id, code, v, v > MATCH_MIN_IOU, True))
        out.extend(
            Match(image_id, code, 0.0, False, False)
            for i in range(len(g)) if i not in assigned_g
        )
        n_pred_unmatched += len(p) - len(assigned_p)
    return tuple(out), n_pred_unmatched


@dataclass(frozen=True)
class BBoxIoUReport:
    """위치 지표 세 정의. **이름이 다르면 다른 양이다** — 표에 정의를 병기한다."""

    mean_penalized: float
    """**주 보고값** `bbox_iou` — 분모가 (GT 박스 + 짝 못 찾은 예측 박스)다.

    이전 정의(`mean_all`)는 분모가 GT 수로 고정이라 **예측을 더 낼수록 값이 올랐다.**
    빗나간 박스 하나면 0.1429 인데 정답 박스를 하나 더 얹으면 1.0000 이 되고, 무작위
    10박스가 0.6123 이었다(80번 D6). conf 를 0.01 까지 내리는 스윕과 곱해지면 분리형만
    벌점 없이 단조 상승한다. 남는 예측을 분모에 넣어 그 성질을 없앤다.
    """
    mean_all: float
    """이전 정의 `bbox_iou_gold_anchored` — 미매칭 GT 를 0 으로 포함한 GT 기준 평균.

    65·66번이 `bbox_iou` 라는 이름으로 실은 값이 이것이다. **이름을 바꿔 남긴다** —
    지우면 과거 산출물과의 회귀 대조가 불가능해진다.
    """
    mean_matched: float
    """매칭쌍만의 평균. 선행 연구(WeldLLM 0.953) 비교용.

    **겹침 0 배정은 이제 여기 안 들어간다**(80번 D5). 그래서 65·66번 값과 다르다.
    """
    n_gold: int
    n_matched: int
    coord_suspect: bool
    """매칭쌍 median IoU ≤ 0.1 이면 True. **실패가 아니라 플래그다** — 진짜 저성능을
    채점기가 기각하면 안 되므로 판단은 사람이 오버레이로 한다(§3-3c)."""
    n_matched_ge_50: int = 0
    """매칭쌍 중 IoU ≥ 0.5 인 건수. mAP@0.5 가 세는 것과 같은 문턱이라, mAP 가 낮을 때
    "위치가 나쁜 것"과 "점수 순위가 없는 것"을 가르는 재료가 된다."""
    n_assigned: int = 0
    """Hungarian 이 묶은 쌍 수. 겹침 0 도 포함한다."""
    n_zero_overlap_assigned: int = 0
    """묶였으나 겹침이 0 인 쌍. **좌표계 붕괴의 서명**이다 — 클래스는 맞는데 위치가
    완전히 어긋난 상태. 이 수가 `n_assigned` 를 거의 다 차지하면 성능이 아니라 규약이다."""
    n_pred: int = 0
    n_pred_unmatched: int = 0

    @property
    def zero_overlap_frac(self) -> float:
        return self.n_zero_overlap_assigned / self.n_assigned if self.n_assigned else 0.0

    def as_dict(self) -> dict:
        return {
            "bbox_iou": self.mean_penalized,
            "bbox_iou_gold_anchored": self.mean_all,
            "bbox_iou_matched_only": self.mean_matched,
            "n_gold": self.n_gold,
            "n_matched": self.n_matched,
            "n_matched_ge_50": self.n_matched_ge_50,
            "n_assigned": self.n_assigned,
            "n_zero_overlap_assigned": self.n_zero_overlap_assigned,
            "zero_overlap_frac": self.zero_overlap_frac,
            "n_pred": self.n_pred,
            "n_pred_unmatched": self.n_pred_unmatched,
            "coord_suspect": self.coord_suspect,
        }


COORD_SUSPECT_MEDIAN = 0.1


def score_bbox_iou(
    pred: Mapping[str, Sequence[tuple[str, Box]]],
    gold: Mapping[str, Sequence[tuple[str, Box]]],
) -> BBoxIoUReport:
    """전 이미지의 BBox-IoU. 정의 세 벌을 **함께** 낸다.

    어느 정의로 산출했는지 표에 반드시 병기한다 — 안 밝히면 0.9 와 0.5 가 같은 이름을 단다.

    **모집단은 `gold` 의 키 전량이다.** 정상 이미지(GT 박스 0건)도 반드시 키로 들어와야
    한다. 빠지면 그 이미지의 오탐이 위치 축에서 사라진다(80번 D9) — 호출부가
    `evaluation.cells.load_population` 에서 back-fill 한다.
    """
    matches: list[Match] = []
    n_pred_unmatched = 0
    n_pred = 0
    for img in sorted(gold):
        p = pred.get(img, ())
        n_pred += len(p)
        ms, unmatched = match_image(p, gold[img], img)
        matches.extend(ms)
        n_pred_unmatched += unmatched
    if not matches and not n_pred:
        return BBoxIoUReport(0.0, 0.0, 0.0, 0, 0, False)

    all_ious = [m.iou for m in matches]
    matched = [m.iou for m in matches if m.matched]
    n_assigned = sum(1 for m in matches if m.assigned)
    n_zero = sum(1 for m in matches if m.assigned and not m.matched)
    median = float(np.median(matched)) if matched else 0.0
    denom = len(matches) + n_pred_unmatched
    return BBoxIoUReport(
        mean_penalized=(float(np.sum(matched)) / denom) if denom else 0.0,
        mean_all=float(np.mean(all_ious)) if all_ious else 0.0,
        mean_matched=float(np.mean(matched)) if matched else 0.0,
        n_gold=len(matches),
        n_matched=len(matched),
        coord_suspect=bool(matched) and median <= COORD_SUSPECT_MEDIAN,
        n_matched_ge_50=sum(1 for v in matched if v >= 0.5),
        n_assigned=n_assigned,
        n_zero_overlap_assigned=n_zero,
        n_pred=n_pred,
        n_pred_unmatched=n_pred_unmatched,
    )


def to_coco_xywh(box: Box) -> tuple[float, float, float, float]:
    """xyxy → COCO xywh. 좌상단 원점, **0-기준 연속 좌표** — Pascal VOC `+1` 규약 혼입 금지.

    지표 파일에서 직접 산술하지 말고 이 함수를 쓴다. 같은 공간 안의 표기 변환이므로
    좌표계 변환(C 소유)이 아니라 D 소유다(§3-4).
    """
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


# --- mAP (pycocotools) ---------------------------------------------------------

NOT_APPLICABLE = "NOT_APPLICABLE"
"""산출하지 않았다는 표식. `None` 과 구분한다 — `None` 은 "0건이라 못 냈다"이고
이쪽은 **"이 칸에서는 정의되지 않는다"**이다. 표에 빈칸으로 두면 둘이 섞인다."""


def coco_map(
    pred: Mapping[str, Sequence[tuple[str, Box, float]]],
    gold: Mapping[str, Sequence[tuple[str, Box]]],
    classes: Sequence[str],
    *,
    scores_present: bool = True,
) -> dict:
    """mAP@0.5 / @0.5:0.95 — pycocotools `COCOeval`, 기본 설정 그대로(커스텀 금지).

    좌표 규약(스펙 §3-4): xywh 0-기준 연속, `image_id` 는 정렬 문자열 → 1..N 고정 매핑,
    `category_id` 는 `classes` 명시 순서 → 1..K. 상수 score 동률의 입력 순서 의존을 없애기
    위해 이미지·박스 순서를 정렬로 고정한다.

    Args:
        pred: image_id → [(iso_code, xyxy_box, score)]
        gold: image_id → [(iso_code, xyxy_box)]. **정상 이미지도 빈 리스트로 들어와야
            한다** — 빠지면 그 이미지의 오탐이 mAP 모집단에서 사라진다(80번 D9).
        scores_present: 예측이 실제 신뢰도를 가지는가. 생성 모델은 신뢰도를 내지 않으므로
            전 박스가 같은 점수가 되는데, mAP 는 **순위 지표**라 그 상태에서 나온 수는
            다른 칸의 mAP 와 같은 표에 실을 수 없다(80번 D16). `False` 면 산출하지 않고
            `NOT_APPLICABLE` 을 돌려준다 — 0 으로 채우면 "위치를 못 맞혔다"로 오독된다.
    """
    import contextlib
    import io as _io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    image_ids = sorted(gold)
    if not scores_present:
        return {
            "map_50_95": NOT_APPLICABLE, "map_50": NOT_APPLICABLE,
            "n_gt_boxes": sum(len(gold[i]) for i in image_ids),
            "n_pred_boxes": sum(len(pred.get(i, ())) for i in image_ids),
            "note": ("예측에 신뢰도가 없다. mAP 는 순위 지표라 상수 점수에서 산출하면 "
                     "다른 칸과 비교 불가능한 수가 된다 — 산출하지 않는다"),
        }
    img_map = {iid: i + 1 for i, iid in enumerate(image_ids)}
    cat_map = {c: i + 1 for i, c in enumerate(classes)}

    gt = {
        "info": {}, "licenses": [],
        "images": [{"id": img_map[i], "file_name": i} for i in image_ids],
        "categories": [{"id": v, "name": k} for k, v in cat_map.items()],
        "annotations": [],
    }
    ann_id = 1
    for iid in image_ids:
        for code, box in gold[iid]:
            if code not in cat_map:
                continue
            x, y, w, h = to_coco_xywh(box)
            gt["annotations"].append({
                "id": ann_id, "image_id": img_map[iid], "category_id": cat_map[code],
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1

    dt = []
    for iid in image_ids:
        for code, box, score in pred.get(iid, ()):
            if code not in cat_map:
                continue
            x, y, w, h = to_coco_xywh(box)
            dt.append({
                "image_id": img_map[iid], "category_id": cat_map[code],
                "bbox": [x, y, w, h], "score": float(score),
            })

    if not gt["annotations"]:
        return {"map_50_95": None, "map_50": None, "note": "GT 박스 0건 — 산출 불가"}
    with contextlib.redirect_stdout(_io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dt) if dt else COCO()
        if not dt:
            coco_dt.dataset = {**gt, "annotations": []}
            coco_dt.createIndex()
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {
        "map_50_95": float(ev.stats[0]),
        "map_50": float(ev.stats[1]),
        "n_gt_boxes": len(gt["annotations"]),
        "n_pred_boxes": len(dt),
    }
