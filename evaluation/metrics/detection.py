"""검출 지표 — Macro-F1(주 지표) · 결함 recall · 클래스별 P/R/F1 · 혼동행렬. 스펙 §4-1·§4-2.

**단위는 이미지 수준 다중라벨이다. 박스 단위가 아니다.** 이유 둘: 통합형은 생성 모델이라
같은 결함을 두 번 언급할 수 있어 박스 개수를 신뢰할 수 없고, 현장의 1차 관심사가
"이 사진에 균열이 있는가"이지 "균열이 몇 개인가"가 아니다.

**"정상"은 클래스가 아니다.** 결함 4종이 전부 음성인 상태로 표현한다. 정상을 5번째
클래스로 두면 다수 클래스가 하나 더 들어가 Macro-F1 이 낙관적으로 부풀려진다.

전부 순수 함수다 — 파일 IO·전역 상태를 쓰지 않아 mock 픽스처로 지금 테스트할 수 있다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassScore:
    iso_code: str
    support: int          # GT 양성 이미지 수
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float | None:
        """예측 양성이 0건이면 정밀도는 정의되지 않는다. 0.0 으로 두면 평균이 왜곡된다."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return 0.0 if self.support else None
        return 2 * p * r / (p + r)


@dataclass(frozen=True)
class DetectionReport:
    per_class: tuple[ClassScore, ...]
    macro_f1: float
    defect_recall: float
    skipped_classes: tuple[str, ...] = field(default=())
    """GT 표본이 0건이라 macro 평균에서 제외한 클래스. **제외 사실을 반드시 보고한다** —
    조용히 빼면 평균이 좋아 보인다(A 리뷰 Minor-2 처방)."""

    def as_dict(self) -> dict:
        return {
            "macro_f1": self.macro_f1,
            "defect_recall": self.defect_recall,
            "skipped_classes": list(self.skipped_classes),
            "per_class": {
                c.iso_code: {
                    "support": c.support, "tp": c.tp, "fp": c.fp, "fn": c.fn,
                    "precision": c.precision, "recall": c.recall, "f1": c.f1,
                }
                for c in self.per_class
            },
        }


def _as_sets(
    records: Mapping[str, Iterable[str]], image_ids: Sequence[str]
) -> dict[str, frozenset[str]]:
    return {img: frozenset(records.get(img, ())) for img in image_ids}


def score_detection(
    pred: Mapping[str, Iterable[str]],
    gold: Mapping[str, Iterable[str]],
    classes: Sequence[str],
) -> DetectionReport:
    """이미지별 클래스 집합 두 벌을 받아 검출 지표를 낸다.

    Args:
        pred: image_id → 예측 ISO 코드들. **파싱 실패 레코드는 빈 집합으로 넘긴다**
            (그래야 미검출로 계상된다 — 통계에서 빠지면 오답보다 낙관적으로 잡힌다).
        gold: image_id → 정답 ISO 코드들. 정상 이미지는 빈 집합.
        classes: 평균 대상 결함 클래스. `label_map.yaml` 에서 온다 — 여기에 하드코딩하지
            않는다(불변조건 1-8).

    Returns:
        `DetectionReport`. macro 평균은 **GT 표본이 있는 클래스만** 대상으로 하고,
        제외된 클래스를 `skipped_classes` 로 함께 보고한다.
    """
    image_ids = sorted(set(gold))
    p = _as_sets(pred, image_ids)
    g = _as_sets(gold, image_ids)

    scores: list[ClassScore] = []
    for code in classes:
        tp = sum(1 for i in image_ids if code in p[i] and code in g[i])
        fp = sum(1 for i in image_ids if code in p[i] and code not in g[i])
        fn = sum(1 for i in image_ids if code not in p[i] and code in g[i])
        scores.append(ClassScore(code, support=tp + fn, tp=tp, fp=fp, fn=fn))

    scored = [s for s in scores if s.support > 0]
    skipped = tuple(s.iso_code for s in scores if s.support == 0)
    macro = sum(s.f1 or 0.0 for s in scored) / len(scored) if scored else 0.0

    # 결함 recall 은 (이미지, 클래스) 양성쌍 기준 micro — 현장에서 미검출이 오검출보다 위험해
    # 별도 지표로 뗀다. macro 와 달리 희소 클래스가 자동으로 낮은 가중을 받는다.
    positives = sum(s.tp + s.fn for s in scores)
    hits = sum(s.tp for s in scores)
    recall = hits / positives if positives else 0.0

    return DetectionReport(tuple(scores), macro, recall, skipped)


def confusion_pairs(
    pred: Mapping[str, Iterable[str]],
    gold: Mapping[str, Iterable[str]],
    classes: Sequence[str],
) -> dict[str, dict[str, int]]:
    """클래스별 2×2 혼동행렬. 다중라벨이라 단일 N×N 행렬이 성립하지 않는다."""
    image_ids = sorted(set(gold))
    p = _as_sets(pred, image_ids)
    g = _as_sets(gold, image_ids)
    out: dict[str, dict[str, int]] = {}
    for code in classes:
        out[code] = {
            "tp": sum(1 for i in image_ids if code in p[i] and code in g[i]),
            "fp": sum(1 for i in image_ids if code in p[i] and code not in g[i]),
            "fn": sum(1 for i in image_ids if code not in p[i] and code in g[i]),
            "tn": sum(1 for i in image_ids if code not in p[i] and code not in g[i]),
        }
    return out


def class_jaccard(
    pred: Mapping[str, Iterable[str]],
    gold: Mapping[str, Iterable[str]],
    classes: Sequence[str] | None = None,
) -> float:
    """이미지별 |∩|/|∪| 의 평균. 스펙 §4-7.

    **양쪽 공집합(정상을 정상으로 예측)이면 1.0 으로 정의한다.** 이 규약을 안 정하면
    정상 이미지 비율(RT 강재 44%)이 지표를 통째로 좌우한다.

    `classes` 를 주면 **채점 공간 안으로 제한한다.** 안 주던 시절, 통합형이 낸 채점
    클래스 밖 코드(2012·402)가 합집합 분모만 키웠다 — `score_detection` 은 그 코드를
    순회하지 않아 FP 로도 세지 않는데 여기서만 벌점이 붙었고, 분리형은 nc=4 라 물리적으로
    그런 코드를 못 내므로 **통합형에만 붙는 벌점**이었다(80번 D8). 어댑터도 같은 코드를
    버리지만(`evaluation.policy`), 지표 쪽에도 걸어 둔다 — 한 겹이 뚫려도 대칭이 유지된다.
    """
    image_ids = sorted(set(gold))
    if not image_ids:
        return 0.0
    scope = frozenset(classes) if classes is not None else None
    total = 0.0
    for img in image_ids:
        a, b = frozenset(pred.get(img, ())), frozenset(gold.get(img, ()))
        if scope is not None:
            a, b = a & scope, b & scope
        union = a | b
        total += 1.0 if not union else len(a & b) / len(union)
    return total / len(image_ids)
