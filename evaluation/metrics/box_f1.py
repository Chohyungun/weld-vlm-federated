"""대표 지표 후보 **M1** — 클래스 일치 박스 매칭 F1@IoU 0.5. 07번 미니스펙 §2 · §12-7 · §15-10.

## 무엇을 재나

이미지마다, 클래스마다, 같은 클래스의 정답 박스와 예측 박스를 **겹침 0.5 이상이면 이을 수 있는 짝**으로 보고
**가장 많이 짝지을 수 있는 수**를 TP 로 센다. 남는 예측이 FP, 남는 정답이 FN 이다.

기존 `metrics/localization.py` 의 매칭은 **IoU 합이 최대가 되는 배정**이라 여기 쓸 수 없다 —
겹침 0.9 인 한 쌍을 고르느라 0.6 짜리 두 쌍을 버리면 짝 수가 줄고, 그만큼 TP 가 준다.

## 단일 채점기

핵심 함수는 `(예측 박스, 정답 박스, 클래스)` 만 받는다. **칸도, 모델 종류도, 대조선인지도 모른다.**
통합형 다섯 모델과 무화소 대조선이 같은 함수로 계수를 얻는 것이 불변조건 3-7 을 지키는 방법이다(§12-7).

## 왜 계수표가 기본 단위인가

`M1` 을 바로 내지 않고 **(이미지 × 클래스) 계수표**를 먼저 만든다. 그 위에서

- 전체·부분집합·출처별 값은 **표를 나눠 더하면** 나온다
- 가중 채점은 행에 가중을 곱하면 된다(§15-10)
- 묶음 부트스트랩은 중복도를 곱해 더하면 된다 — 재표집마다 매칭을 다시 풀지 않는다

지표를 먼저 내면 위 셋이 전부 따로 구현된다.

## 경계

문턱은 **수학적 `IoU ≥ 1/2`** 다. 나눗셈을 하지 않고 `2·교집합 ≥ 합집합` 으로 판정한다 —
부동소수 나눗셈이 경계에서 어느 쪽으로 떨어질지에 값이 걸리지 않게. 경계 가까이(상대 1e-9)면
**유리수로 다시 판정**하고 그 쌍 수를 보고한다.
독립 검산의 두 구현이 같은 규칙을 쓴다 — 한쪽만 유리수를 쓰면 경계쌍에서 갈린다(§12-2).

### 무엇의 유리수인가 — **이진(binary64) 규약**

유리수 재판정은 `Fraction(float)` 이므로 **JSON 을 읽어 얻은 배정밀도 값의 정확한 유리수**다.
기록된 십진 문자열이 뜻하는 유리수가 아니다. 둘은 갈릴 수 있다 — 정답 `[0, 0, 0.3, 0.1]` ·
예측 `[0.1, 0, 0.4, 0.1]` 은 **십진 기하로는 IoU 가 정확히 1/2** 이지만 배정밀도에서는
`2·교집합 − 합집합 = −2 / 2^55` 로 1/2 미만이다.

**정본은 이진이다.** 근거 둘.

1. **읽는 쪽이 하나로 정해진다.** 산출물은 JSON 이고, 표준 파서로 읽으면 누구나 같은 배정밀도
   값을 얻는다. 채점기가 실제로 들고 있는 값 위에서 술어를 정확히 판정하는 것이 이 규약이다.
   십진을 정본으로 하려면 `parse_float=Decimal` 이 어댑터부터 채점기까지 이어져야 하고,
   1.4 레코드는 그 문자열을 보존하지 않는다.
2. **임의 오차로 덮지 않는다.** 경계 판정은 여전히 정확하다 — 근사가 들어가는 자리는
   십진→이진 변환 한 번뿐이고, 그 변환은 결정적이다.

대가는 위 예시처럼 **십진으로 정확히 1/2 인 쌍이 후보에서 빠질 수 있다**는 것이다.
그 사실을 시험으로 고정해 둔다(숨기지 않는다). 규약을 십진으로 바꾸려면 어댑터의 `parse_float`
와 여기 `Fraction` 인자를 함께 바꾸고 **새 채점 지문으로 등록**한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Sequence[float]

BOUNDARY_REL = 1e-9
"""`2·교집합` 과 `합집합` 의 차가 이 상대 크기 안이면 유리수로 다시 본다."""

COORD_NUMERIC_CONVENTION = "binary64"
"""경계 판정이 어느 수 위에서 이뤄지는가. 등록 블록의 `matching_rule_id` 와 함께 박는다."""


def _inter_union(a: Box, b: Box) -> tuple[float, float]:
    """교집합·합집합 넓이. **클리핑하지 않는다** — 이미지 밖으로 나간 예측은 자연 벌점을 받아야 한다."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    inter = iw * ih if iw > 0 and ih > 0 else 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter, union


def _exact_candidate(a: Box, b: Box) -> bool:
    """유리수로 `2·교집합 ≥ 합집합` 을 판정한다.

    `Fraction(float)` 은 **그 배정밀도 값의** 정확한 유리수다 — 기록된 십진 문자열의 유리수가
    아니다. 이진 규약을 쓴다는 뜻이고, 모듈 머리말에 근거와 대가를 적었다.
    """
    ax1, ay1, ax2, ay2 = (Fraction(v) for v in a)
    bx1, by1, bx2, by2 = (Fraction(v) for v in b)
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    inter = iw * ih if iw > 0 and ih > 0 else Fraction(0)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return union > 0 and 2 * inter >= union


def is_candidate(a: Box, b: Box) -> tuple[bool, bool]:
    """`(짝지을 수 있는가, 경계에서 다시 본 쌍인가)`.

    빠른 길은 float 이고, 경계 가까이만 유리수로 되본다. 겹침이 0 이면 어떤 경우에도 후보가 아니다.
    """
    inter, union = _inter_union(a, b)
    if union <= 0 or inter <= 0:
        return False, False
    slack = 2 * inter - union
    if abs(slack) <= BOUNDARY_REL * max(1.0, union):
        return _exact_candidate(a, b), True
    return slack > 0, False


def max_matching(gold: Sequence[Box], pred: Sequence[Box]) -> tuple[int, int]:
    """같은 클래스 안에서 **가장 많이 짝지을 수 있는 수**와 경계쌍 수.

    후보 그래프의 최대 매칭이다. 최대 매칭이 여럿이어도 **크기는 정의상 하나**라
    2차 기준(겹침 합)이나 동률 규칙이 계수를 바꾸지 못한다 — 시험이 그 사실을 본다.
    """
    if not gold or not pred:
        return 0, 0
    cand = np.zeros((len(gold), len(pred)), dtype=np.int8)
    n_boundary = 0
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            ok, boundary = is_candidate(g, p)
            n_boundary += boundary
            cand[i, j] = 1 if ok else 0
    if not cand.any():
        return 0, n_boundary
    rows, cols = linear_sum_assignment(-cand.astype(np.int32))
    # 배정된 쌍 가운데 **후보인 것만** 센다. 나머지 칸이 0 이라 배정은 언제나 min(n, m) 쌍이 된다.
    return int(cand[rows, cols].sum()), n_boundary


@dataclass
class CountTable:
    """(이미지 × 클래스) 계수표. **모든 값이 여기서 파생된다.**"""

    classes: tuple[str, ...]
    image_ids: tuple[str, ...]
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    boundary: np.ndarray | None = None
    """(이미지 × 클래스) 경계 재판정 쌍 수. **칸마다 세어 둔다** — 총계만 들고 있으면
    부분집합에서 되살릴 수 없어 `subset` 이 조용히 0 을 보고하게 된다."""
    pred_per_image: np.ndarray | None = None
    """이미지마다의 예측 박스 수. 같은 이유로 총계가 아니라 칸으로 둔다."""
    notes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = (len(self.image_ids), len(self.classes))
        for name in ("tp", "fp", "fn"):
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name} 의 모양이 {arr.shape} 다 — {shape} 여야 한다")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} 에 유한하지 않은 값이 있다")
            if (arr < 0).any():
                raise ValueError(f"{name} 에 음수가 있다 — 계수는 0 이상이다")
        if self.boundary is None:
            self.boundary = np.zeros(shape, dtype=np.int64)
        elif self.boundary.shape != shape:
            raise ValueError(f"boundary 의 모양이 {self.boundary.shape} 다 — {shape} 여야 한다")
        if self.pred_per_image is None:
            self.pred_per_image = np.zeros(len(self.image_ids), dtype=np.int64)
        elif self.pred_per_image.shape != (len(self.image_ids),):
            raise ValueError(f"pred_per_image 의 모양이 {self.pred_per_image.shape} 다")

    @property
    def n_boundary_pairs(self) -> int:
        return int(self.boundary.sum())

    @property
    def boundary_cells(self) -> tuple[tuple[str, str], ...]:
        """경계에서 유리수로 다시 본 (이미지, 클래스). 0 이 아니면 목록을 산출물에 남긴다."""
        rows, cols = np.nonzero(self.boundary)
        return tuple((self.image_ids[r], self.classes[c])
                     for r, c in zip(rows.tolist(), cols.tolist(), strict=True))

    @property
    def n_pred_boxes(self) -> int:
        return int(self.pred_per_image.sum())

    def index_of(self) -> dict[str, int]:
        return {i: n for n, i in enumerate(self.image_ids)}

    def subset(self, image_ids) -> CountTable:
        """부분집합의 계수표. **표를 나눠 더하는 것이 곧 부분집합 채점이다.**

        진단 계수도 함께 고른다. 전체 id 로 다시 고르면 **경계쌍 수·예측 수·notes 가 그대로**여야
        한다 — 그렇지 않으면 "부분집합에서는 경계 재판정이 없었다" 는 거짓 진단이 나간다.
        """
        idx = self.index_of()
        keep = [idx[i] for i in image_ids]
        return CountTable(self.classes, tuple(self.image_ids[k] for k in keep),
                          self.tp[keep], self.fp[keep], self.fn[keep],
                          self.boundary[keep], self.pred_per_image[keep],
                          notes={**self.notes, "from": "subset", "n_subset": len(keep)})

    def totals(self, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """클래스별 합. `weights` 는 **이미지마다의 가중**이다(부표본의 `N_h/n_h`, 재표집의 중복도).

        가중이 없으면 전부 1 이다 — 전량 채점에서는 가중 없는 합과 같다(§15-10 의 항등).
        """
        if weights is None:
            return self.tp.sum(0), self.fp.sum(0), self.fn.sum(0)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.image_ids),):
            raise ValueError(f"가중의 길이가 {w.shape} 다 — 이미지 수 {len(self.image_ids)} 여야 한다")
        if not np.isfinite(w).all():
            raise ValueError("가중에 유한하지 않은 값이 있다 — NaN·Inf 는 계수와 지지량을 오염시킨다")
        if (w < 0).any():
            # 0 은 정상이다(재표집에서 안 뽑힌 이미지·부표본 밖). 음수는 F1 을 1 위로 올린다.
            raise ValueError("가중에 음수가 있다 — 0 은 되지만 음수는 안 된다")
        return (self.tp * w[:, None]).sum(0), (self.fp * w[:, None]).sum(0), (self.fn * w[:, None]).sum(0)


def build_counts(
    pred_boxes: Mapping[str, Sequence[tuple[str, Box]]],
    gold_boxes: Mapping[str, Sequence[tuple[str, Box]]],
    classes: Sequence[str],
) -> CountTable:
    """계수표를 만든다. **모집단은 `gold_boxes` 의 키 전량**이다 — 정상 이미지도 들어온다.

    빠지면 그 이미지의 오탐이 지표에서 사라진다. 대상마다 예측이 정확히 하나여야 한다 —
    없는 것은 빈 목록으로, 여분은 오류로 다룬다(호출부가 먼저 거부한다).
    """
    cls = tuple(classes)
    ids = tuple(sorted(gold_boxes))
    extra = set(pred_boxes) - set(ids)
    if extra:
        raise ValueError(f"정답에 없는 이미지의 예측이 {len(extra)}건 있다")

    tp = np.zeros((len(ids), len(cls)), dtype=np.int32)
    fp = np.zeros_like(tp)
    fn = np.zeros_like(tp)
    boundary = np.zeros((len(ids), len(cls)), dtype=np.int64)
    pred_per_image = np.zeros(len(ids), dtype=np.int64)

    for r, image_id in enumerate(ids):
        g_all = gold_boxes.get(image_id, ())
        p_all = pred_boxes.get(image_id, ())
        pred_per_image[r] = len(p_all)
        for c, code in enumerate(cls):
            g = [b for cc, b in g_all if cc == code]
            p = [b for cc, b in p_all if cc == code]
            if not g and not p:
                continue
            matched, n_b = max_matching(g, p)
            boundary[r, c] = n_b
            tp[r, c] = matched
            fp[r, c] = len(p) - matched
            fn[r, c] = len(g) - matched

    return CountTable(cls, ids, tp, fp, fn, boundary, pred_per_image,
                      notes={"n_images": len(ids),
                             "coord_numeric_convention": COORD_NUMERIC_CONVENTION})


def sensitivity_counts(
    base: CountTable,
    *,
    gold_codes: Mapping[str, Sequence[str] | set[str]],
    failed_image_ids,
) -> CountTable:
    """민감도 변형 — **정상 이미지의 레코드 실패에 클래스마다 오탐 1건**을 더한다(§12-8).

    주 채점은 실패를 빈 예측으로 둔다. 그러면 정상 이미지의 실패는 벌점을 하나도 받지 않는다 —
    불변조건 3-4 의 "오답 처리" 와 어긋나는 자리이고, 그 크기를 재는 것이 이 변형이다.

    **채점 4클래스로 고정한다**(그 모집단의 평균 대상 목록이 아니라). 목록이 모집단마다 달라지면
    micro 가 함께 움직여 민감도의 뜻이 흐려진다.
    """
    idx = base.index_of()
    fp = base.fp.copy()
    n_added = 0
    for image_id in failed_image_ids:
        if image_id not in idx:
            continue
        if set(gold_codes.get(image_id, ())):
            continue        # 결함 이미지의 실패는 주 채점과 같다(정답이 FN 으로 남는다)
        fp[idx[image_id], :] += 1
        n_added += 1
    return CountTable(base.classes, base.image_ids, base.tp, fp, base.fn,
                      base.boundary, base.pred_per_image,
                      notes={**base.notes, "variant": "sensitivity",
                             "n_normal_failures_penalised": n_added,
                             "fp_per_failure": len(base.classes)})


@dataclass(frozen=True)
class M1Report:
    """M1 과 그 재료. **평균에 넣은 클래스 목록을 값과 함께 들고 다닌다.**"""

    macro_f1: float | None
    micro_f1: float | None
    per_class_f1: dict[str, float | None]
    per_class_counts: dict[str, dict[str, float]]
    classes_in_macro: tuple[str, ...]
    """실제로 평균에 들어간 클래스. 고정 대상 모드에서 미정의면 **의도한 대상**이 들어온다."""
    classes_without_support: tuple[str, ...]
    n_boundary_pairs: int = 0
    macro_support_rule: str = "gold_support"
    """`gold_support` 는 그 모집단의 지지량으로 대상을 정한다(점추정). `fixed` 는 미리 정한
    목록을 쓴다(재표집) — 둘을 섞으면 추첨마다 나누는 클래스 수가 달라진다."""
    undefined_reason: str | None = None
    """`macro_f1` 이 `None` 인 사유. `support` 는 고정 대상 중 지지량 0 이 생긴 것이고,
    `no_support` 는 어느 클래스에도 지지량이 없는 것이다. 0 점과 구분한다."""

    @property
    def label(self) -> str:
        """`4결함 매크로` 인가 아닌가. 평균 대상이 넷보다 적으면 그렇게 부르지 않는다(§2-1)."""
        k = len(self.classes_in_macro)
        return "4결함 매크로" if k == 4 else f"{k}결함 매크로"

    def as_dict(self) -> dict:
        return {"m1_macro_f1": self.macro_f1, "micro_f1": self.micro_f1,
                "label": self.label,
                "per_class_f1": self.per_class_f1, "per_class_counts": self.per_class_counts,
                "classes_in_macro": list(self.classes_in_macro),
                "classes_without_support": list(self.classes_without_support),
                "n_boundary_pairs": self.n_boundary_pairs,
                "macro_support_rule": self.macro_support_rule,
                "undefined_reason": self.undefined_reason}


def _f1(tp: float, fp: float, fn: float) -> float | None:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else None


def score_m1(table: CountTable, weights: np.ndarray | None = None, *,
             fixed_classes: Sequence[str] | None = None) -> M1Report:
    """계수표에서 M1 을 낸다.

    평균 대상은 **정답 지지량(TP+FN)이 있는 클래스**다. 예측으로 정하지 않는다 —
    모델이 어떤 클래스를 안 내면 그 클래스가 평균에서 빠져 점수가 올라간다.

    Args:
        fixed_classes: 평균 대상을 **미리 고정**한다. 재표집이 쓰는 모드다.

    재표집에서 대상을 매번 다시 정하면 계약이 깨진다. 정답이 각각 A·B 인 이미지 둘에서 A 만
    맞춘 표는 비가중 macro 가 0.5 인데, 가중 `[1, 0]` 을 주면 B 의 지지량이 0 이 되어 B 가
    빠지고 macro 가 1 로 **올라간다.** 그 추첨은 점수가 좋아진 것이 아니라 **잴 수 없는** 것이다.
    고정 대상 모드는 그런 추첨을 `undefined_reason="support"` 로 돌려준다.

    점추정에서 모집단을 바꿔 대상이 달라지는 것은 정당한 모집단 변경이고 이와 다르다 —
    그때는 `fixed_classes` 를 주지 않는다.
    """
    tp, fp, fn = table.totals(weights)
    per_f1: dict[str, float | None] = {}
    per_counts: dict[str, dict[str, float]] = {}
    in_macro: list[str] = []
    no_support: list[str] = []

    for c, code in enumerate(table.classes):
        support = float(tp[c] + fn[c])
        per_counts[code] = {"tp": float(tp[c]), "fp": float(fp[c]), "fn": float(fn[c]),
                            "support": support}
        value = _f1(tp[c], fp[c], fn[c])
        per_f1[code] = value
        (in_macro if support > 0 else no_support).append(code)

    micro = _f1(tp.sum(), fp.sum(), fn.sum())
    if fixed_classes is None:
        scored = [per_f1[c] for c in in_macro if per_f1[c] is not None]
        macro = (sum(scored) / len(scored)) if scored else None
        reason = None if macro is not None else "no_support"
        return M1Report(macro, micro, per_f1, per_counts, tuple(in_macro), tuple(no_support),
                        table.n_boundary_pairs, "gold_support", reason)

    target = tuple(fixed_classes)
    unknown = [c for c in target if c not in table.classes]
    if unknown:
        raise ValueError(f"계수표에 없는 고정 평균 대상: {unknown}")
    missing = tuple(c for c in target if per_counts[c]["support"] <= 0)
    if missing:
        return M1Report(None, micro, per_f1, per_counts, target, missing,
                        table.n_boundary_pairs, "fixed", "support")
    macro = sum(per_f1[c] for c in target) / len(target)
    return M1Report(macro, micro, per_f1, per_counts, target, (),
                    table.n_boundary_pairs, "fixed", None)


def cluster_weights(image_ids: Sequence[str], cluster_of: Mapping[str, str],
                    weight_of: Mapping[str, float]) -> np.ndarray:
    """덩어리마다의 가중을 **이미지마다의 가중**으로 편다.

    덩어리는 묶음이 아니라 **연결 성분**이다(§18-4) — 같은 묶음과 근사 중복 쌍을 함께 이은 단위다.

    값의 **유한·비음수**만 여기서 본다. 양수여야 하는가(층화 추정의 `N_h/n_h`)·정수여야 하는가
    (재표집 중복도)는 쓰는 쪽 조건이라 호출부가 본다.
    """
    out = np.array([float(weight_of[cluster_of[i]]) for i in image_ids], dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("덩어리 가중에 유한하지 않은 값이 있다")
    if (out < 0).any():
        raise ValueError("덩어리 가중에 음수가 있다")
    return out
