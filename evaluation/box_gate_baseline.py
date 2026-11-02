"""등록 M1 대조선 — ID 구간 × 클래스 빈도 게이트의 중앙값 상자. **화소를 한 번도 보지 않는다.**

실험설계 8-5 의 주 대조선을 그대로 구현한다. 구간은 `evaluation.strata.bins_for`(촬영 ID 의 `idq512`, 절단점은 동결본
train+val)이고, 구간 b · 클래스 c 의 빈도는 적합 이미지 가운데 c 를 가진 이미지의 비율, 상자는 그 구간 · 클래스 정답 상자의
좌표별 중앙값이다. 빈도가 문턱 t 이상인 (b, c) 에만 상자 하나를 낸다. t 는 train 적합 · val 채점으로 고르고, 고른 t 를 고정해
train+val 로 다시 적합한다(`FIT_SPLIT`). 평가셋으로 문턱이나 적합을 고르지 않는다.

| 칸 | 값 |
|---|---|
| 문턱 후보 | 0.1 … 0.8, 0.1 간격 — 처음 계산(2026-09-20 설계 개정 교차 검수 §7)의 후보 그대로 |
| 문턱 비교 | 정확한 정수식 `10 × (c 를 가진 장 수) ≥ k × (구간 장 수)`(t = k/10) — 부동소수 경계에서 갈리지 않는다 |
| 고르는 값 | M1(공통 채점기 `build_counts` · `score_m1`, 정답 지지량 매크로). 같으면 작은 t |
| 정답 | 채점 클래스의 원 코드 그대로(엄격 채점과 같다 — 별칭을 합치지 않는다). 상자 네 칸이 빈 행은 빈도에는 세고 상자에는 넣지 않는다 |
| 예측 줄 | `{"image_id", "defects": [{"iso_code", "bbox_px"}]}` — 엄격 채점의 대조선 판독기(`strict_main.read_baseline_predictions`) 꼴, 목록 순서 |

**평가 정답을 읽지 않는다.** 정답은 `data.manifest_view.read_annotations_view` 로만 읽고, 그 판독기는 평가 id 가 섞이면 주석 파일을
열기 전에 거부한다. 평가 이미지에 쓰는 것은 id 의 구간 번호뿐이다. 이 대조선도 상한이 아니다 — 더 강한 무화소 규칙이 나오면
등록 대조선을 올린다(실험설계 8-5).
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluation.metrics.box_f1 import M1Report, build_counts, score_m1
from evaluation.strata import bins_for

RULE_ID = "id_bin_class_freq_gate_median_box"
"""등록 칸 `baseline.rule_id` 에 들어갈 이름. 구간 수 · 문턱은 따로 `strata_k` · `threshold_t` 에 적는다."""
STRATA_K = 512
"""등록된 족 `idq512` — 여기서 다시 고르지 않는다."""
THRESHOLD_TENTHS: tuple[int, ...] = tuple(range(1, 9))
"""문턱 후보 t = k/10 의 k."""
FIT_SPLIT = "train_then_refit_trainval"
"""등록 어휘 `baseline.fit_split` 의 값 — train 적합, val 로 t 선택, train+val 재적합."""
FIT_SPLITS = ("train", "val")


@dataclass(frozen=True)
class GateFit:
    """적합이 끝난 게이트 한 벌. 세 표 모두 **적합 모집단의 정답으로만** 만든다."""

    classes: tuple[str, ...]
    n_images: Mapping[int, int]
    """구간 → 적합 이미지 수(정상 이미지 포함)."""
    n_with: Mapping[tuple[int, str], int]
    """(구간, 클래스) → 그 클래스를 가진 적합 이미지 수."""
    box: Mapping[tuple[int, str], tuple[float, float, float, float]]
    """(구간, 클래스) → 정답 상자의 좌표별 중앙값. 상자가 하나도 없는 칸은 없다."""

    def fires(self, b: int, c: str, tenths: int) -> bool:
        n = self.n_images.get(b, 0)
        return n > 0 and (b, c) in self.box and 10 * self.n_with.get((b, c), 0) >= tenths * n

    def exact_boundary_cells(self) -> int:
        """빈도가 어떤 후보 문턱과 정확히 같은 (구간, 클래스) 수 — 경계를 '이상' 으로 읽는 결정이 닿는 칸."""
        return sum(1 for (b, _c), m in self.n_with.items()
                   if any(10 * m == k * self.n_images[b] for k in THRESHOLD_TENTHS))


def fit_gate(fit_ids: Sequence[str], codes_of: Mapping[str, Iterable[str]],
             boxes_of: Mapping[str, Sequence[tuple[str, Sequence[float]]]], bin_of: Mapping[str, int],
             classes: Sequence[str]) -> GateFit:
    """적합. `codes_of` 는 이미지 수준 코드(상자 없는 행 포함), `boxes_of` 는 상자가 있는 행이다. 채점 클래스 밖 코드는 버린다."""
    ids = list(fit_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("적합 id 가 겹친다")
    cls = tuple(classes)
    scope = set(cls)
    n_images: Counter = Counter()
    n_with: Counter = Counter()
    coords: dict[tuple[int, str], list] = defaultdict(list)
    for i in ids:
        b = bin_of[i]
        n_images[b] += 1
        for c in set(codes_of.get(i, ())) & scope:
            n_with[(b, c)] += 1
        for c, bx in boxes_of.get(i, ()):
            if c in scope:
                coords[(b, c)].append(bx)
    box = {key: tuple(float(statistics.median(float(bx[j]) for bx in v)) for j in range(4)) for key, v in coords.items()}
    return GateFit(cls, dict(n_images), dict(n_with), box)


def predict_gate(fit: GateFit, ids: Iterable[str], bin_of: Mapping[str, int],
                 tenths: int) -> dict[str, list[tuple[str, list[float]]]]:
    """이미지 → 예측 상자(클래스 순서). 이미지에 쓰는 것은 구간 번호뿐이다."""
    return {i: [(c, list(fit.box[(bin_of[i], c)])) for c in fit.classes if fit.fires(bin_of[i], c, tenths)] for i in ids}


def m1(pred: Mapping[str, Sequence], gold_boxes: Mapping[str, Sequence], ids: Sequence[str],
       classes: Sequence[str]) -> M1Report:
    """모델과 같은 채점기 — 모집단은 `ids` 전량(정상 이미지는 빈 정답)."""
    return score_m1(build_counts({i: pred[i] for i in ids}, {i: list(gold_boxes.get(i, ())) for i in ids}, list(classes)))


def threshold_curve(fit: GateFit, ids: Sequence[str], bin_of: Mapping[str, int], gold_boxes: Mapping[str, Sequence],
                    classes: Sequence[str]) -> dict[int, float | None]:
    return {k: m1(predict_gate(fit, ids, bin_of, k), gold_boxes, ids, classes).macro_f1 for k in THRESHOLD_TENTHS}


def pick_threshold(curve: Mapping[int, float | None]) -> int:
    """가장 높은 M1 의 k. 같으면 작은 k. 값이 하나도 없으면 고르지 않는다."""
    vals = [v for v in curve.values() if v is not None]
    if not vals:
        raise ValueError("문턱 후보의 M1 이 모두 정의되지 않았다 — 고르지 않는다")
    best = max(vals)
    return min(k for k, v in curve.items() if v == best)


def prediction_bytes(pred: Mapping[str, Sequence[tuple[str, Sequence[float]]]], order: Sequence[str]) -> bytes:
    """JSON 줄 — 목록 순서, 한 줄 한 이미지, UTF-8 · LF · 끝 개행."""
    lines = [json.dumps({"image_id": i, "defects": [{"iso_code": c, "bbox_px": [float(x) for x in b]} for c, b in pred[i]]},
                        ensure_ascii=False, separators=(",", ":")) for i in order]
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_fit_gold(snapshot: Path, ids: Sequence[str]) -> tuple[dict[str, set[str]], dict[str, list], int]:
    """적합 분할의 정답 — `(이미지 수준 코드, 상자, 상자 없는 행 수)`. 판독기가 평가 id 를 주석 파일을 열기 전에 거부한다."""
    import pandas as pd

    from data.manifest_view import read_annotations_view

    df = read_annotations_view(snapshot, list(ids))
    codes: dict[str, set[str]] = defaultdict(set)
    boxes: dict[str, list] = defaultdict(list)
    no_box = 0
    for row in df.itertuples(index=False):
        codes[row.image_id].add(str(row.iso_code))
        vals = (row.bbox_x1_px, row.bbox_y1_px, row.bbox_x2_px, row.bbox_y2_px)
        if any(pd.isna(v) for v in vals):
            no_box += 1
            continue
        boxes[row.image_id].append((str(row.iso_code), tuple(float(v) for v in vals)))
    return dict(codes), dict(boxes), no_box


def build(snapshot: Path, list_ids: Sequence[str], classes: Sequence[str]) -> tuple[bytes, dict]:
    """등록 절차 그대로 — train 적합 · val 선택 · train+val 재적합 · `list_ids`(평가 분할) 예측. `(예측 바이트, 기록)`.

    기록의 `fit_curve_trainval` 은 train+val 로 적합하고 train+val 로 채점한 곡선이다 — 고르는 데 쓰지 않는다(처음 계산의
    '적합 자료에서 고르면' 값과 맞대는 자기 검사다).
    """
    from data.manifest_view import read_manifest_columns

    snapshot = Path(snapshot)
    m = read_manifest_columns(snapshot, ["image_id", "split"], split_filter=("train", "val", "eval"))
    split_of = dict(zip(m["image_id"].astype(str), m["split"].astype(str), strict=True))
    order = list(list_ids)
    if len(set(order)) != len(order) or any(split_of.get(i) != "eval" for i in order):
        raise ValueError("예측 목록은 겹치지 않는 평가 분할 id 여야 한다")
    train = sorted(i for i, s in split_of.items() if s == "train")
    val = sorted(i for i, s in split_of.items() if s == "val")
    trainval = [*train, *val]
    bin_of = bins_for([*trainval, *order], STRATA_K, snapshot)
    codes, boxes, no_box = read_fit_gold(snapshot, trainval)
    fit_tr = fit_gate(train, codes, boxes, bin_of, classes)
    val_curve = threshold_curve(fit_tr, val, bin_of, boxes, classes)
    k = pick_threshold(val_curve)
    fit_tv = fit_gate(trainval, codes, boxes, bin_of, classes)
    fit_curve = threshold_curve(fit_tv, trainval, bin_of, boxes, classes)
    pred = predict_gate(fit_tv, order, bin_of, k)
    raw = prediction_bytes(pred, order)
    per_class = Counter(c for v in pred.values() for c, _ in v)
    meta = {
        "rule_id": RULE_ID, "strata_k": STRATA_K, "threshold_t": k / 10, "fit_split": FIT_SPLIT,
        "threshold_candidates": [x / 10 for x in THRESHOLD_TENTHS],
        "selection": "train 적합 · val 의 M1 최대(같으면 작은 t)",
        "val_curve": {f"{x / 10:.1f}": v for x, v in val_curve.items()},
        "fit_curve_trainval": {f"{x / 10:.1f}": v for x, v in fit_curve.items()},
        "classes": list(classes),
        "n_train": len(train), "n_val": len(val), "n_lines": len(order),
        "n_fit_rows_without_box": no_box,
        "n_fire_cells": sum(1 for (b, c) in fit_tv.box if fit_tv.fires(b, c, k)),
        "n_exact_boundary_cells": {"train": fit_tr.exact_boundary_cells(), "trainval": fit_tv.exact_boundary_cells()},
        "n_images_with_boxes": sum(1 for v in pred.values() if v),
        "n_pred_boxes": sum(per_class.values()),
        "n_pred_boxes_per_class": {c: per_class.get(c, 0) for c in classes},
    }
    return raw, meta
