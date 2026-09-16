"""`map_50` 두 번째 구현 — 대표 수치 채택에 남은 검산용 (53번 미니스펙 · 57번 게이트 · A 58번 정의표 승인).

지금 채점의 `map_50` 은 pycocotools 2.0.11 한 구현에서만 나온다. 이 모듈은 **같은 정의**(53번 §3 정의표 17항 +
A 58번 보강 · #18)를 pycocotools 없이 다시 짠다. 정의를 맞춰야 값이 다를 때 결함인지 정의 차이인지 가릴 수 있다.
검산 결과는 대표 채택 조건의 판정에만 쓰고 `map_50` 을 대체하지 않는다(53번 §9).

두 입구가 같은 알고리즘(`evaluate_map50`)으로 들어간다.

- **S(공유 입력)** `map50_shared` — 기존 채점의 입력 결과(레코드·정답·xywh 변환)를 그대로 받는다. 차이가 나면 알고리즘 쪽이다.
- **I(독립 입력)** `read_gold_independent` · `read_pred_independent` · `map50_independent` — 파일 형식 문서만 보고
  동결 스냅샷과 하한 레코드를 직접 읽는다. 문서에 없어 기존 동작에서 가져온 규약은 `CONVENTIONS` 에 출처와 함께 적고
  결과에 싣는다. 계약 #2 의 "승인 로더만" 규칙에 대한 예외는 이 입력 함수와 검산 실행기에만 적용된다(57번 00:15 추기).

**import 경계.** pycocotools · `evaluation.recovery_ci` · `evaluation.score` · `evaluation.metrics.localization` ·
`scripts.probe.*` 를 쓰지 않는다. 입력 규약을 기존 코드에서 가져오지 않도록 `data.*` 와 다른 `evaluation.*` 도 쓰지 않는다.
이 모듈을 import 하는 파일은 자기 시험과 검산 실행기뿐이다. 둘 다 시험이 고정한다.

**정의 요약**(IoU 문턱 0.5 한 개, 면적 'all', 무시 정답 없음).

1. 이미지 목록 = 정답 사전의 키 전부(정상 이미지 포함), 문자열 정렬 순서(#8).
2. 이미지 × 카테고리마다 검출을 `-score` 안정 정렬 후 상위 100건만 쓴다(#5·#6). 정답·검출이 모두 없으면 건너뛴다(#17).
3. IoU = 면적식(xywh, 오른쪽 끝을 `x + w` 로 다시 계산, `+1` 없음)(#11).
4. 매칭 = 점수 순 탐욕. 검출마다 아직 매칭되지 않은 정답 가운데 **IoU 가 문턱 이상인 것 중 최대**, 동률이면 뒤쪽(#9·#10).
   문턱은 `min(0.5, 1 − 1e−10)` 이고 같음은 매칭이다. 매칭은 id 의 참/거짓이 아니라 명시적 표로 기록한다(#18).
5. 카테고리마다 이미지 순서로 이어 붙여 `-score` 안정 정렬(#7). 정답 0 이면 그 카테고리는 −1(평균에서 제외, #14).
   누적 → `rc = tp / npig` · `pr = tp / (fp + tp + ε)`(#4) → 뒤에서부터 포락선(#3) → `linspace(0, 1, 101)`(#2) 격자에서
   `searchsorted(left)`, 범위 밖은 0(#1). 검출이 없으면 정밀도 0 행(#15).
6. `map_50` = (101, K) 격자를 C 순서로 펴서 −1 이 아닌 값을 한 번에 평균한다(#16). 카테고리별 AP 평균도 따로 낸다.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

Box = tuple[float, float, float, float]
"""COCO xywh `(x, y, w, h)`. 좌상 원점, 0-기준 연속 좌표, `+1` 없음(13_spec_D §3-4)."""

GoldMap = Mapping[str, Sequence[tuple[str, Box]]]
PredMap = Mapping[str, Sequence[tuple[str, Box, float]]]

IOU_THRESHOLD = 0.5
"""pycocotools `iouThrs[0]` — `linspace(.5, .95, 10)` 의 첫 값은 정확히 0.5 다."""
MATCH_THRESHOLD = min(IOU_THRESHOLD, 1 - 1e-10)
"""#9 — 매칭 시작 문턱. 0.5 에서는 0.5 그대로다."""
MAX_DETS = 100
"""#5 — 이미지 × 카테고리당 점수 상위 건수. pycocotools 기본 `maxDets[-1]`, `map_50` 은 100 조각."""
REC_THRS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
"""#2 — 재현율 격자 101점. 십진 정확값이 아니라 같은 `linspace` 부동소수 값을 쓴다."""
EPS = np.spacing(1)
"""#4 — 정밀도 분모의 ε. 이 때문에 완벽 매칭의 AP 도 1.0 이 아니다(A 58번 §3)."""
BOUNDARY_TOL = 1e-9
"""#11 — 문턱 경계 쌍 계수의 폭 `|IoU − 0.5| < 1e−9`."""
AREA_RANGE_ALL = (0.0, 1e10)
"""#12 — 'all' 면적 범위. 여기 밖의 정답·미매칭 검출은 무시 대상이지만 실데이터에는 없어야 한다(단언)."""


# ======================================================================================
# 알고리즘 — 모드 공통
# ======================================================================================

@dataclass(frozen=True)
class Map50Result:
    map_50: float
    """#16 pycocotools 순서 — (101, K) 격자를 C 순서로 펴서 −1 제외 한 번에 `np.mean`."""
    map_50_class_mean: float
    """#16 카테고리별 AP(101점 평균)의 평균. 정답 없는 카테고리 제외."""
    ap_by_class: dict[str, float | None]
    classes: tuple[str, ...]
    precision: np.ndarray
    """(101, K). 정답 없는 카테고리 열은 −1."""
    counts: dict = field(default_factory=dict)


def _check_box(box, where: str) -> Box:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise ValueError(f"{where}: 박스는 네 값이어야 한다 — {box!r}")
    out = []
    for v in box:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"{where}: 박스 값이 수가 아니다 — {box!r}")  # noqa: TRY004 — 입력 형식 오류는 ValueError 하나로 모은다
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"{where}: 박스 값이 유한하지 않다 — {box!r}")
        out.append(f)
    return (out[0], out[1], out[2], out[3])


def _check_score(score, where: str) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float, np.integer, np.floating)):
        raise ValueError(f"{where}: 점수가 수가 아니다 — {score!r}")  # noqa: TRY004 — 위와 같은 이유
    f = float(score)
    if not math.isfinite(f):
        raise ValueError(f"{where}: 점수가 유한하지 않다 — {score!r}")
    return f


def _iou_matrix(dts: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """(D, G) IoU — #11 면적식. `maskUtils.iou` 의 bbox 경로와 같은 연산 순서를 따른다.

    교집합 폭 `min(dw + dx, gw + gx) − max(dx, gx)`, 0 이하이면 IoU 0. 합집합 `(da + ga) − i`.
    덧셈은 교환법칙이 성립하므로 `dw + dx` 와 `dx + dw` 는 같은 비트다.
    """
    dx, dy, dw, dh = (dts[:, j:j + 1] for j in range(4))       # (D, 1)
    gx, gy, gw, gh = (gts[None, :, j] for j in range(4))         # (1, G)
    w = np.minimum(dw + dx, gw + gx) - np.maximum(dx, gx)
    h = np.minimum(dh + dy, gh + gy) - np.maximum(dy, gy)
    inter = w * h
    da = dw * dh
    ga = gw * gh
    union = (da + ga) - inter
    ok = (w > 0) & (h > 0)
    out = np.zeros(np.broadcast(w, h).shape, dtype=np.float64)
    np.divide(inter, union, out=out, where=ok)
    return out


def evaluate_map50(gold: GoldMap, pred: PredMap, classes: Sequence[str]) -> Map50Result:
    """알고리즘 입구(§ 모듈 정의 요약 1~6). 입력 형식은 `GoldMap`·`PredMap` 주석을 따른다."""
    classes = tuple(classes)
    if len(set(classes)) != len(classes) or not classes:
        raise ValueError(f"클래스 목록이 비었거나 중복이 있다: {classes!r}")
    if not all(isinstance(c, str) for c in classes):
        raise ValueError("클래스는 문자열이어야 한다")
    unknown_imgs = set(pred) - set(gold)
    if unknown_imgs:
        raise ValueError(f"정답에 없는 이미지의 검출이 있다: {sorted(unknown_imgs)[:5]} 외 {max(0, len(unknown_imgs) - 5)}")
    image_ids = sorted(gold)                                        # #8 문자열 정렬
    kidx = {c: k for k, c in enumerate(classes)}
    K, R = len(classes), len(REC_THRS)

    counts: dict = {
        "n_images": len(image_ids),
        "gt_by_class": {c: 0 for c in classes}, "det_by_class": {c: 0 for c in classes},
        "det_evaluated_by_class": {c: 0 for c in classes},
        "det_truncated": 0, "pairs_empty": 0, "iou_ties": 0, "boundary_pairs": 0,
        "dropped_class_gt": 0, "dropped_class_det": 0, "ignored_gt": 0, "ignored_det": 0,
    }

    # 카테고리별 이미지 순서의 평가 결과: (점수 배열, 매칭 불 배열, 정답 수)
    per_class: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in classes]
    for img in image_ids:
        g_by = [[] for _ in classes]
        for n, item in enumerate(gold[img]):
            code, box = item
            k = kidx.get(code)
            if k is None:
                counts["dropped_class_gt"] += 1
                continue
            g_by[k].append(_check_box(box, f"정답 {img}#{n}"))
        d_by = [[] for _ in classes]
        for n, item in enumerate(pred.get(img, ())):
            code, box, score = item
            k = kidx.get(code)
            if k is None:
                counts["dropped_class_det"] += 1
                continue
            d_by[k].append((_check_box(box, f"검출 {img}#{n}"), _check_score(score, f"검출 {img}#{n}")))
        for k, cls in enumerate(classes):
            gts, dts = g_by[k], d_by[k]
            counts["gt_by_class"][cls] += len(gts)
            counts["det_by_class"][cls] += len(dts)
            if not gts and not dts:
                counts["pairs_empty"] += 1                          # #17 — 평가하지 않는다
                continue
            order = sorted(range(len(dts)), key=lambda j: -dts[j][1])   # #6 안정 정렬(Timsort)
            kept = order[:MAX_DETS]                                 # #5 이미지 × 카테고리 상위 100
            counts["det_truncated"] += len(order) - len(kept)
            counts["det_evaluated_by_class"][cls] += len(kept)
            d_boxes = np.array([dts[j][0] for j in kept], dtype=np.float64).reshape(len(kept), 4)
            d_scores = np.array([dts[j][1] for j in kept], dtype=np.float64)
            g_boxes = np.array(gts, dtype=np.float64).reshape(len(gts), 4)
            _assert_area_all(d_boxes, g_boxes, img, cls)
            matched = np.zeros(len(kept), dtype=bool)
            if len(kept) and len(gts):
                ious = _iou_matrix(d_boxes, g_boxes)
                counts["boundary_pairs"] += int(np.count_nonzero(np.abs(ious - IOU_THRESHOLD) < BOUNDARY_TOL))
                gt_taken = [False] * len(gts)
                for di in range(len(kept)):
                    best = MATCH_THRESHOLD
                    m = -1
                    row = ious[di]
                    for gi in range(len(gts)):
                        if gt_taken[gi]:
                            continue
                        v = row[gi]
                        if v < best:                                # #9 같음은 매칭 · #10 같음이면 뒤쪽으로 갱신
                            continue
                        if m != -1 and v == best:
                            counts["iou_ties"] += 1
                        best = v
                        m = gi
                    if m == -1:
                        continue
                    matched[di] = True                              # #18 명시적 표 — id 진릿값에 기대지 않는다
                    gt_taken[m] = True
            per_class[k].append((d_scores, matched, len(gts)))

    precision = -np.ones((R, K), dtype=np.float64)
    for k in range(K):
        entries = per_class[k]
        if not entries:
            continue
        scores = np.concatenate([e[0] for e in entries])
        inds = np.argsort(-scores, kind="mergesort")               # #7 이미지 순서 → 안의 순서 안정
        dtm = np.concatenate([e[1] for e in entries])[inds]
        npig = int(sum(e[2] for e in entries))
        if npig == 0:
            continue                                                # #14 −1 그대로(평균 제외)
        tps = dtm
        fps = np.logical_not(dtm)
        tp = np.cumsum(tps).astype(dtype=float)
        fp = np.cumsum(fps).astype(dtype=float)
        nd = len(tp)
        rc = tp / npig
        pr = (tp / (fp + tp + EPS)).tolist()                       # #4
        for i in range(nd - 1, 0, -1):                              # #3 포락선
            if pr[i] > pr[i - 1]:  # noqa: PLR1730 — pycocotools 와 같은 비교 형태를 그대로 둔다
                pr[i - 1] = pr[i]
        q = [0.0] * R
        for ri, pi in enumerate(np.searchsorted(rc, REC_THRS, side="left")):   # #1
            if pi >= nd:
                break                                               # 오름차순이라 뒤도 전부 범위 밖 → 0
            q[ri] = pr[pi]
        precision[:, k] = np.array(q)

    flat = precision[precision > -1]                                # #16 C 순서(R 우선, 그 안에서 K)
    map_50 = float(np.mean(flat)) if flat.size else -1.0
    ap_by_class: dict[str, float | None] = {}
    for k, cls in enumerate(classes):
        col = np.ascontiguousarray(precision[:, k])
        ap_by_class[cls] = None if col[0] == -1 else float(np.mean(col))
    present = [v for v in ap_by_class.values() if v is not None]
    class_mean = float(np.mean(present)) if present else -1.0
    return Map50Result(map_50=map_50, map_50_class_mean=class_mean, ap_by_class=ap_by_class,
                       classes=classes, precision=precision, counts=counts)


def _assert_area_all(d_boxes: np.ndarray, g_boxes: np.ndarray, img: str, cls: str) -> None:
    """#12·#13 — 'all' 범위 밖 박스가 있으면 무시 처리 규칙이 필요해진다. 이 검산은 그 경로를 짜지 않았으므로 멈춘다."""
    lo, hi = AREA_RANGE_ALL
    for name, boxes in (("검출", d_boxes), ("정답", g_boxes)):
        if len(boxes):
            area = boxes[:, 2] * boxes[:, 3]
            if np.any((area < lo) | (area > hi)):
                raise ValueError(f"{img} {cls}: 'all' 면적 범위 밖 {name} 박스 — 무시 규칙이 필요하다(이 구현 밖)")


# ======================================================================================
# S 모드 — 공유 입력
# ======================================================================================

def map50_shared(gold_xywh: GoldMap, pred_xywh: PredMap, classes: Sequence[str]) -> Map50Result:
    """기존 채점 입력(`read_records`·`read_gold`·`to_coco_xywh` 결과)을 그대로 받는다.

    None 점수 치환·클래스 목록·xyxy→xywh 는 호출자가 이미 한 상태여야 한다. 형식은 `evaluate_map50` 이 검사한다.
    """
    if not isinstance(gold_xywh, Mapping) or not isinstance(pred_xywh, Mapping):
        raise TypeError("정답·검출은 image_id → 목록 사전이어야 한다")
    return evaluate_map50(gold_xywh, pred_xywh, classes)


# ======================================================================================
# I 모드 — 독립 입력 (57번 00:15 예외 범위)
# ======================================================================================

CONVENTIONS: dict[str, dict[str, str]] = {
    "D3_record_identity": {
        "rule": "모든 레코드의 cell·client·seed 가 한 값이고 expect 와 같다", "source": "doc",
        "doc": "prediction.schema.json 필드"},
    "D4_float_coords": {
        "rule": "bbox_px 를 파일값 그대로 쓴다(반올림·스냅·클리핑 없음)", "source": "doc-mismatch",
        "doc": "13_spec_D §3-4 는 제출 전 정수화라 적지만 실물은 실수다. 채점기가 반올림을 더하지 않는다는 규칙은 같은 절의 문서"},
    "D5_image_order": {
        "rule": "이미지 순서 = 정답 image_id 문자열 정렬, defects 순서 보존", "source": "doc", "doc": "13_spec_D §3-4"},
    "D6_score_none": {
        "rule": "점수 None: score_none='error' 면 멈춤, 'zero' 면 0.0 치환", "source": "code-if-zero",
        "doc": "문서 없음(13_spec_D §3-4 는 통합형 1.0 상수만 적는다). 0.0 치환은 기존 채점 동작"},
    "D7_bbox_null": {
        "rule": "bbox_px 가 null 인 결함은 위치 지표에서 뺀다", "source": "doc", "doc": "13_spec_D §3-5"},
    "D8_classes": {
        "rule": "클래스 = label_map eval_spaces.<eval_space>.defect_types 순서의 iso_code", "source": "doc",
        "doc": "configs/label_map.yaml L4 eval_spaces · 13_spec_D §3-4"},
    "D9_gold_order": {
        "rule": "한 이미지 안 정답 순서 = annotations.csv 파일 행 순서", "source": "code",
        "doc": "문서 없음 — 기존 read_gold 동작 · 53번 #10. 파일은 ann_id 를 문자열 순서로 둔다"},
    "D10_gold_exclusion": {
        "rule": "geom_valid 가 'True' 가 아닌 정답 행을 뺀다. True 인데 bbox 칸이 비면 멈춘다", "source": "doc",
        "doc": "10_spec_A §5(geom_valid=false 는 채점 제외). 기존 코드는 bbox 유무 기준 — 두 기준의 차를 센다"},
    "D11_gold_bounds": {
        "rule": "정답 bbox 는 정수이고 0 ≤ x1 < x2 ≤ width, 0 ≤ y1 < y2 ≤ height", "source": "doc",
        "doc": "10_spec_A §2-4 IV10"},
    "D12_class_drop": {
        "rule": "클래스 목록 밖 정답·검출은 버리고 센다", "source": "doc", "doc": "label_map eval_spaces.excluded"},
    "D14_localization": {
        "rule": "평가 이미지의 has_localization 은 'True'", "source": "doc", "doc": "10_spec_A §2-3 N1"},
    "D15_direct_csv": {
        "rule": "CSV 를 csv 모듈로 직접 읽는다(빈 문자열 보존)", "source": "exception",
        "doc": "계약 #2 로더 규칙의 예외 — 57번 00:15 추기"},
    "D16_encoding": {
        "rule": "UTF-8, BOM 이면 멈춤", "source": "measured", "doc": "문서 없음 — 실물 확인"},
    "D17_record_ids": {
        "rule": "레코드 image_id 집합 = 평가 이미지 집합, 중복 없음", "source": "doc", "doc": "10_spec_A §2-1 #20 · 평가 모집단"},
    "D18_coord_space": {
        "rule": "coord_space 는 ABS_ORIG", "source": "doc", "doc": "13_spec_D §3-4"},
    "D19_parse_fail": {
        "rule": "parse_ok=false 레코드는 결함 없음으로 계산", "source": "doc", "doc": "13_spec_D §2-5 · 구현설계 §8-1"},
}
"""I 모드 입력 규약과 출처(D `input_conventions_I모드_D.md` 번호). `source` 가 `code` 인 항목은 **독립 확인이 아니다.**"""

_MANIFEST_COLS = ("image_id", "split", "has_localization", "width_px", "height_px")
_ANN_COLS = ("ann_id", "image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "geom_valid")
_BBOX_COLS = ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")


@dataclass(frozen=True)
class GoldInputs:
    gold: dict[str, list[tuple[str, Box]]]
    classes: tuple[str, ...]
    image_ids: tuple[str, ...]
    report: dict


@dataclass(frozen=True)
class PredInputs:
    pred: dict[str, list[tuple[str, Box, float]]]
    report: dict


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_text(path: Path):
    """UTF-8 로 연다. BOM 이 있으면 멈춘다(D16)."""
    path = Path(path)
    with path.open("rb") as fh:
        if fh.read(3) == b"\xef\xbb\xbf":
            raise ValueError(f"{path.name}: BOM 이 있다 — 형식이 문서와 다르다")
    return path.open("r", encoding="utf-8", newline="")


def _require_cols(fieldnames, need, name: str) -> None:
    missing = [c for c in need if c not in (fieldnames or ())]
    if missing:
        raise ValueError(f"{name}: 열이 없다 {missing}")


def _int_cell(value: str, where: str) -> int:
    s = value.strip()
    if not s or not (s.isdigit() or (s[0] == "-" and s[1:].isdigit())):
        raise ValueError(f"{where}: 정수가 아니다 — {value!r}")
    return int(s)


def read_gold_independent(snapshot_dir: Path, label_map_path: Path, *, eval_space: str = "main_rt",
                          split: str = "eval") -> GoldInputs:
    """동결 스냅샷에서 평가 이미지와 정답 박스를 직접 읽는다(무결성은 실행기가 먼저 확인한다 — 57번 00:15 판정 3)."""
    snapshot_dir, label_map_path = Path(snapshot_dir), Path(label_map_path)

    with _open_text(label_map_path) as fh:
        lm = yaml.safe_load(fh)
    space = (lm.get("eval_spaces") or {}).get(eval_space)
    if not space or not space.get("defect_types"):
        raise ValueError(f"label_map 에 eval_spaces.{eval_space}.defect_types 가 없다")
    types = lm.get("defect_types") or {}
    classes = []
    for key in space["defect_types"]:
        code = (types.get(key) or {}).get("iso_code")
        if not isinstance(code, str) or not code:
            raise ValueError(f"label_map defect_types.{key}.iso_code 가 문자열이 아니다: {code!r}")
        classes.append(code)
    if len(set(classes)) != len(classes):
        raise ValueError(f"평가 공간 {eval_space} 의 iso_code 가 겹친다: {classes}")

    sizes: dict[str, tuple[int, int]] = {}
    split_counts: Counter = Counter()
    n_manifest = 0
    with _open_text(snapshot_dir / "manifest.csv") as fh:
        rd = csv.DictReader(fh)
        _require_cols(rd.fieldnames, _MANIFEST_COLS, "manifest.csv")
        seen: set[str] = set()
        for row in rd:
            n_manifest += 1
            iid = row["image_id"]
            if iid in seen:
                raise ValueError(f"manifest.csv: image_id 중복 {iid}")
            seen.add(iid)
            split_counts[row["split"]] += 1
            if row["split"] != split:
                continue
            if row["has_localization"] != "True":
                raise ValueError(f"manifest.csv: {iid} has_localization={row['has_localization']!r} — 위치 지표 대상이 아니다")
            sizes[iid] = (_int_cell(row["width_px"], f"{iid} width_px"), _int_cell(row["height_px"], f"{iid} height_px"))
    if not sizes:
        raise ValueError(f"manifest.csv: split={split!r} 이미지가 없다")

    image_ids = tuple(sorted(sizes))
    gold: dict[str, list[tuple[str, Box]]] = {i: [] for i in image_ids}
    excluded: Counter = Counter()
    rule_diff = Counter()
    n_rows = 0
    with _open_text(snapshot_dir / "annotations.csv") as fh:
        rd = csv.DictReader(fh)
        _require_cols(rd.fieldnames, _ANN_COLS, "annotations.csv")
        for row in rd:
            iid = row["image_id"]
            if iid not in sizes:
                continue
            n_rows += 1
            has_bbox = [bool(row[c].strip()) for c in _BBOX_COLS]
            valid = row["geom_valid"] == "True"
            if not valid:
                excluded[f"geom_valid={row['geom_valid']}"] += 1           # D10 문서 규칙
                if all(has_bbox):
                    rule_diff["excluded_but_has_bbox"] += 1                 # bbox 유무 기준(기존 코드)이면 남았을 행
                elif any(has_bbox):
                    rule_diff["excluded_partial_bbox"] += 1
                continue
            if not all(has_bbox):
                raise ValueError(f"annotations.csv: {row['ann_id']} geom_valid=True 인데 bbox 칸이 비었다")
            x1, y1, x2, y2 = (_int_cell(row[c], f"{row['ann_id']} {c}") for c in _BBOX_COLS)
            w_img, h_img = sizes[iid]
            if not (0 <= x1 < x2 <= w_img and 0 <= y1 < y2 <= h_img):
                raise ValueError(f"annotations.csv: {row['ann_id']} bbox {x1, y1, x2, y2} 가 IV10 을 어긴다")
            gold[iid].append((row["iso_code"], (float(x1), float(y1), float(x2 - x1), float(y2 - y1))))

    kept = Counter(code for boxes in gold.values() for code, _ in boxes)
    digest_line = None
    lock = snapshot_dir / "SNAPSHOT.sha256"
    if lock.exists():
        for ln in lock.read_text(encoding="utf-8").splitlines():
            if ln.startswith("# snapshot_digest"):
                digest_line = ln.split()[-1]
    report = {
        "files_sha256": {"manifest.csv": _sha256(snapshot_dir / "manifest.csv"),
                         "annotations.csv": _sha256(snapshot_dir / "annotations.csv"),
                         "label_map.yaml": _sha256(label_map_path)},
        "snapshot_digest_line": digest_line,
        "eval_space": eval_space, "split": split, "classes": list(classes),
        "n_manifest_rows": n_manifest, "split_counts": dict(split_counts),
        "n_images": len(image_ids),
        "n_gold_rows": n_rows, "gold_excluded": dict(excluded),
        "gold_rule_disagreement": dict(rule_diff),
        "gold_boxes_by_code": dict(kept), "n_gold_boxes": sum(kept.values()),
        "n_images_with_box": sum(1 for v in gold.values() if v),
    }
    return GoldInputs(gold=gold, classes=tuple(classes), image_ids=image_ids, report=report)


def read_pred_independent(records_path: Path, gold: GoldInputs, *, expect: Mapping[str, object] | None = None,
                          score_none: Literal["error", "zero"] = "error") -> PredInputs:
    """하한 레코드 jsonl 하나를 직접 읽는다(계약 #4). 경로와 해시 고정은 실행기 몫이다."""
    if score_none not in ("error", "zero"):
        raise ValueError(f"score_none 은 'error' 또는 'zero' 다: {score_none!r}")
    records_path = Path(records_path)
    wanted = set(gold.image_ids)
    pred: dict[str, list[tuple[str, Box, float]]] = {}
    seen: set[str] = set()
    identity: set[tuple] = set()
    stats: Counter = Counter()
    coord: Counter = Counter()
    with _open_text(records_path) as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                raise ValueError(f"{records_path.name}:{lineno} 빈 줄 — 한 줄이 한 이미지여야 한다")
            rec = json.loads(line)
            for key in ("image_id", "cell", "seed", "defects", "parse_ok"):
                if key not in rec:
                    raise ValueError(f"{records_path.name}:{lineno} 필수 필드 {key} 가 없다")
            iid = rec["image_id"]
            if iid in seen:
                raise ValueError(f"{records_path.name}: image_id 중복 {iid}")
            seen.add(iid)
            identity.add((rec["cell"], rec.get("client"), rec["seed"]))
            coord[rec.get("coord_space")] += 1
            if rec.get("coord_space") != "ABS_ORIG":
                raise ValueError(f"{records_path.name}:{lineno} coord_space={rec.get('coord_space')!r} — ABS_ORIG 가 아니다")
            stats["records"] += 1
            if rec["parse_ok"] is not True:
                stats["parse_fail_records"] += 1                    # D19 결함 없음으로 계산
                stats["parse_fail_defects_ignored"] += len(rec["defects"] or [])
                continue
            boxes = []
            for n, d in enumerate(rec["defects"]):
                stats["defects"] += 1
                b = d.get("bbox_px")
                if b is None:
                    stats["bbox_null"] += 1                         # D7
                    continue
                x1, y1, x2, y2 = _check_box(b, f"{iid} 결함 {n}")
                if not (x1 < x2 and y1 < y2):
                    raise ValueError(f"{iid} 결함 {n}: 퇴화 박스 {b!r} — 어댑터가 걸렀어야 한다")
                s = d.get("score")
                if s is None:
                    stats["score_none"] += 1
                    if score_none == "error":
                        raise ValueError(f"{iid} 결함 {n}: 점수가 없다(score_none='error')")
                    s = 0.0                                         # D6 코드 규약
                s = _check_score(s, f"{iid} 결함 {n}")
                boxes.append((d["iso_code"], (x1, y1, x2 - x1, y2 - y1), s))   # D4 파일값 그대로
            if boxes:
                pred[iid] = boxes
    if len(identity) != 1:
        raise ValueError(f"{records_path.name}: 칸·참여자·시드가 한 값이 아니다 {sorted(map(str, identity))}")
    cell, client, seed = next(iter(identity))
    if expect is not None:
        got = {"cell": cell, "client": client, "seed": seed}
        diff = {k: (got.get(k), v) for k, v in expect.items() if got.get(k) != v}
        if diff:
            raise ValueError(f"{records_path.name}: 기대한 칸과 다르다 {diff}")
    missing, extra = wanted - seen, seen - wanted
    if missing or extra:
        raise ValueError(f"{records_path.name}: 평가 이미지와 집합이 다르다 — 누락 {len(missing)} · 초과 {len(extra)}")
    report = {
        "file": records_path.name, "sha256": _sha256(records_path),
        "cell": cell, "client": client, "seed": seed,
        "coord_space": {str(k): v for k, v in coord.items()},
        "counts": dict(stats),
        "boxes_by_code": dict(Counter(code for bs in pred.values() for code, _, _ in bs)),
        "score_none_policy": score_none,
    }
    return PredInputs(pred=pred, report=report)


def map50_independent(gold: GoldInputs, pred: PredInputs) -> tuple[Map50Result, dict]:
    """I 모드 한 칸. 둘째 값은 입력 보고와 규약 출처 목록이다."""
    result = evaluate_map50(gold.gold, pred.pred, gold.classes)
    used_zero = pred.report["score_none_policy"] == "zero" and pred.report["counts"].get("score_none", 0) > 0
    conventions = []
    for key, c in CONVENTIONS.items():
        independent = c["source"] != "code" and not (key == "D6_score_none" and used_zero)
        conventions.append({"id": key, **c, "independent": independent})
    report = {"gold": gold.report, "pred": pred.report, "conventions": conventions,
              "not_independent": [c["id"] for c in conventions if not c["independent"]],
              "definition_source": "53번 §3 정의표(A 58번 승인) — pycocotools 2.0.11 동작에 맞춘 정의이며 문서가 아니다"}
    return result, report
