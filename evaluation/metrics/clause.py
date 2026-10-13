"""조항 정확도 축 — 검색 top-1/3 · 인용 일치율 · 판정 정합성 · 무근거 인용률. 스펙 §4-3·§4-5·§4-8·§4-9.

근거 접지 두 지표(Class-Jaccard·BBox-IoU)는 **검출 결과와의 일치**를 잰다. 검출이 맞다는
전제에서만 의미가 있고, 인용한 규정이 실제 표준과 맞는지는 재지 않는다. 세 번째 축을
세우는 것이 이 연구의 평가 쪽 기여이며, 이 모듈이 그 축이다.

**단위는 (이미지 × 결함코드) 쌍이다** (게이트 #8 결정 ②). 이미지 단위로 재면 다결함
이미지에서 첫 결함만 질의하는 구현이 통과해 버리고, RT 강재는 기공이 45%라 다수 클래스로
편중된 값이 소수 클래스(슬래그·융합불량)의 실패를 은폐한다.

**판정 로직을 여기서 재구현하지 않는다.** `judge`·`aggregate_verdicts`·`applicable_row`는
`corpus.rules.limit_eval`에서 import한다. 골격 생성기와 채점기가 같은 함수를 써야
gold와 재계산이 어긋나지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from corpus.rules.limit_eval import (
    aggregate_verdicts,
    applicable_row,
    judge,
    rows_for_clause,
)
from corpus.rules.schema import FAIL, PASS, InspectionMethod

UNDECIDABLE = "판정불가"


@dataclass(frozen=True)
class GoldPair:
    """(이미지 × 결함코드) 하나에 대응하는 정답. `corpus/derived/gold_clauses.csv`에서 온다.

    D가 `limits.csv`를 직접 순회해 만들지 않는다 — B의 `derive_gold_clauses` 산출물을
    읽기만 한다(개발규약 불변조건 1-7, B 스펙 §1-4 금지 규칙 ④).
    """

    image_id: str
    iso_code: str
    clause_id: str
    row_id: str | None = None
    """판정 정합성 재계산에 쓸 `limits.csv` 행 식별자(rule_id)."""


@dataclass(frozen=True)
class RetrievalReport:
    top1: float
    top3: float
    n_pairs: int
    per_code: dict[str, dict[str, float | int]]
    gt_coverage: float
    """GT에 있는 쌍 중 실제로 질의가 나간 비율. **분모가 '질의한 쌍'이라 검출이 나쁠수록
    top-1이 올라가는 착시**가 생기므로 반드시 병기한다(§4-8)."""
    n_gt_pairs: int

    def as_dict(self) -> dict:
        return {
            "clause_top1": self.top1, "clause_top3": self.top3,
            "n_pairs": self.n_pairs, "gt_code_coverage": self.gt_coverage,
            "n_gt_pairs": self.n_gt_pairs, "per_code": self.per_code,
        }


def _gold_map(gold: Iterable[GoldPair]) -> dict[tuple[str, str], GoldPair]:
    return {(g.image_id, g.iso_code): g for g in gold}


def score_retrieval(
    retrieved: Mapping[tuple[str, str], Sequence[str]],
    gold: Iterable[GoldPair],
) -> RetrievalReport:
    """조항 검색 top-1/top-3. 분리형 전용 — 통합형은 검색을 붙이지 않는다.

    Args:
        retrieved: (image_id, iso_code) → 검색기가 반환한 조항 ID 목록(순서 유지).
            **예측 결함 기준**이다. 모델이 질의하지 않은 쌍은 키가 없다.
        gold: 정답 쌍들.
    """
    gm = _gold_map(gold)
    hit1 = hit3 = 0
    per: dict[str, dict[str, float | int]] = {}
    for (img, code), got in sorted(retrieved.items()):
        g = gm.get((img, code))
        if g is None:
            # 검출이 틀린 결함코드로 던진 질의 — 정답 조항 자체가 없다.
            # 오답으로 계상한다(검색기 탓이 아니어도 파이프라인 결과는 틀렸다).
            slot = per.setdefault(code, {"n": 0, "hit1": 0, "hit3": 0})
            slot["n"] += 1
            continue
        ok1 = bool(got) and got[0] == g.clause_id
        ok3 = g.clause_id in list(got)[:3]
        hit1 += ok1
        hit3 += ok3
        slot = per.setdefault(code, {"n": 0, "hit1": 0, "hit3": 0})
        slot["n"] += 1
        slot["hit1"] += int(ok1)
        slot["hit3"] += int(ok3)

    n = len(retrieved)
    for slot in per.values():
        slot["top1"] = slot["hit1"] / slot["n"] if slot["n"] else 0.0
        slot["top3"] = slot["hit3"] / slot["n"] if slot["n"] else 0.0

    gt_pairs = set(gm)
    queried = set(retrieved)
    coverage = len(gt_pairs & queried) / len(gt_pairs) if gt_pairs else 0.0
    return RetrievalReport(
        top1=hit1 / n if n else 0.0,
        top3=hit3 / n if n else 0.0,
        n_pairs=n,
        per_code=per,
        gt_coverage=coverage,
        n_gt_pairs=len(gt_pairs),
    )


@dataclass(frozen=True)
class CitationReport:
    match_rate: float
    precision: float
    n_pairs: int
    per_code: dict[str, dict[str, float | int]]

    def as_dict(self) -> dict:
        return {
            "citation_match_rate": self.match_rate,
            "citation_precision": self.precision,
            "n_pairs": self.n_pairs,
            "per_code": self.per_code,
        }


def score_citation(
    cited_by_image: Mapping[str, Sequence[str]],
    gold: Iterable[GoldPair],
) -> CitationReport:
    """인용 일치율. **두 구조 공통**이라 RQ2의 핵심 비교 지표다.

    분모는 GT 결함 쌍이고, 인용 집합은 **이미지 수준** 하나다. 결함별로 인용을 쪼개 내라고
    요구하면 통합형의 출력 형식을 바꿔야 하고, 그건 5칸 공통 프롬프트 고정을 깨는 대가가
    얻는 것보다 크다(§2-2).

    정밀도를 함께 내는 이유: 안 보면 "여러 개 인용하면 이기는" 지표가 된다.
    """
    gm = _gold_map(gold)
    hit = 0
    per: dict[str, dict[str, float | int]] = {}
    for (img, code), g in sorted(gm.items()):
        ok = g.clause_id in set(cited_by_image.get(img, ()))
        hit += ok
        slot = per.setdefault(code, {"n": 0, "hit": 0})
        slot["n"] += 1
        slot["hit"] += int(ok)
    for slot in per.values():
        slot["rate"] = slot["hit"] / slot["n"] if slot["n"] else 0.0

    gold_by_image: dict[str, set[str]] = {}
    for (img, _), g in gm.items():
        gold_by_image.setdefault(img, set()).add(g.clause_id)
    cited_total = correct = 0
    for img, cites in cited_by_image.items():
        allowed = gold_by_image.get(img, set())
        for c in cites:
            cited_total += 1
            correct += c in allowed

    n = len(gm)
    return CitationReport(
        match_rate=hit / n if n else 0.0,
        precision=correct / cited_total if cited_total else 0.0,
        n_pairs=n,
        per_code=per,
    )


@dataclass(frozen=True)
class ConsistencyReport:
    rate: float
    n_scored: int
    excluded: dict[str, int] = field(default_factory=dict)
    """분모에서 뺀 건수와 사유. **빼는 것을 숨기면 지표가 조용히 부풀려진다**(§4-3)."""
    error_types: dict[str, int] = field(default_factory=dict)
    """오류 유형 분해 — 조항 오검색 / 대소 비교 실패 / 이중 실패. 기준선 미달 시
    어느 쪽인지 구분되지 않으면 해석이 불가능하다."""

    def as_dict(self) -> dict:
        return {
            "verdict_consistency": self.rate, "n_scored": self.n_scored,
            "excluded": self.excluded, "error_types": self.error_types,
        }


@dataclass(frozen=True)
class DefectMeasure:
    """판정 정합성 재계산의 입력 한 건."""

    iso_code: str
    row: object | None
    """`corpus.rules.schema.LimitRow`. 인용 조항에서 선택한 applicable_row."""
    measured: Decimal | None
    basis_value: Decimal | None = None
    gold_clause_id: str | None = None
    cited_clause_id: str | None = None
    inspection_method: str | None = None
    """이 이미지의 검사 방식(manifest `modality`). 주 실험은 RT 한정.

    **행이 이미 선택된 채로 들어오므로 채점기가 축을 재확인한다.** 표면(VT) 행을 내부(RT)
    이미지에 적용해도 부등식은 멀쩡히 계산되고 형식상 정상으로 보인다 — 판정 정합성이
    틀린 채로 높게 나오는 최악의 형태이며, 계약 #3에 이 컬럼을 넣은 이유가 그것이다.
    """


class AxisMismatch(ValueError):
    """선택된 허용치 행의 검사 방식이 이미지의 검사 방식과 다르다."""


def row_matches_method(row: object, method: str | None) -> bool:
    """행의 `inspection_method` 가 질의 축과 맞는가. `ALL` 행은 어느 축에도 응한다."""
    if method is None:
        return True
    row_method = getattr(row, "inspection_method", None)
    if row_method is None:
        return True
    return row_method in (InspectionMethod(method), InspectionMethod.ALL)


def select_row(
    table: object,
    *,
    clause_id: str,
    defect_code: str,
    material: str,
    inspection_method: str,
    quality_scheme: str,
    quality_level: str,
    thickness_mm: Decimal,
    limit_type: str | None = None,
):
    """판정 정합성 재계산의 **행 선택 단일 경로**. 스펙 §4-3.

    인용 조항으로 후보를 좁힌 뒤(`rows_for_clause`) 골격 내장 키로 `applicable_row` 를
    재사용한다. 두 함수 모두 `corpus.rules` 의 것이며 채점기가 자체 pandas 순회로 행을 고르지 않는다.

    **검사 방식이 키에 들어간다.** 축이 빠지면 같은 결함코드·같은 두께의 표면 행과 내부
    행이 한 후보 집합에 섞여, 잘못된 허용치를 집고도 부등식이 정상 계산된다.
    """
    candidates = rows_for_clause(table, clause_id)
    if not candidates:
        raise LookupError(f"인용 조항이 허용치 표에 없다: {clause_id}")
    return applicable_row(
        candidates,
        defect_code,
        material,
        inspection_method,
        quality_scheme,
        quality_level,
        thickness_mm,
        limit_type,
    )


def score_consistency(
    predicted_verdicts: Mapping[str, str],
    measures: Mapping[str, Sequence[DefectMeasure]],
) -> ConsistencyReport:
    """인용 조항의 허용치와 실측값으로 부등식을 재계산해 verdict와 대조한다.

    **조항을 맞게 찾고도 결론을 틀릴 수 있다.** 그 실패를 잡는 것이 이 지표의 목적이며,
    규칙 기반 재계산이라 LLM 심판이 개입하지 않는다.

    이미지 합성은 `aggregate_verdicts`(B 소유)를 호출한다 — 보수 규칙(한 결함이라도
    불합격이면 불합격)을 D가 재구현하지 않는다.
    """
    ok = 0
    scored = 0
    excluded: dict[str, int] = {}
    errors: dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for image_id, predicted in sorted(predicted_verdicts.items()):
        items = list(measures.get(image_id, ()))
        if not items:
            # 결함 0개 → 합성 규칙상 합격. 정상 이미지의 정상 판정도 채점 대상이다.
            recomputed = aggregate_verdicts([])
        else:
            if any(m.cited_clause_id is None for m in items):
                drop("근거 없음")
                continue
            if any(m.row is None for m in items):
                drop("인용 조항이 허용치 표에 없음")
                continue
            if any(not row_matches_method(m.row, m.inspection_method) for m in items):
                # 축이 어긋난 행으로도 부등식은 멀쩡히 계산된다 — 조용히 채점하면
                # 틀린 값이 정상처럼 집계된다. 배관 오류로 분류해 분모에서 뺀다.
                drop("검사 방식 불일치(축 오적용)")
                continue
            if any(m.measured is None for m in items):
                drop("실측값 없음(스케일 부재)")
                continue
            try:
                per_defect = [
                    judge(m.row, m.measured, m.basis_value).verdict for m in items
                ]
            except (ValueError, AssertionError):
                drop("판정 불능")
                continue
            recomputed = aggregate_verdicts(per_defect)

        scored += 1
        if predicted == UNDECIDABLE:
            # 조항이 실제로 존재하는데 판정불가를 냈으면 부정합.
            # (검색 0건이라 판정불가인 경우는 위에서 '근거 없음'으로 이미 빠진다.)
            errors["판정 회피"] = errors.get("판정 회피", 0) + 1
            continue
        if predicted == recomputed:
            ok += 1
            continue

        # 오류 유형 분해 — 조항을 틀렸나, 대소 비교를 틀렸나.
        wrong_clause = any(
            m.gold_clause_id is not None and m.cited_clause_id != m.gold_clause_id
            for m in items
        )
        key = "이중 실패" if wrong_clause and items else ("조항 오검색" if wrong_clause else "대소 비교 실패")
        errors[key] = errors.get(key, 0) + 1

    return ConsistencyReport(
        rate=ok / scored if scored else 0.0,
        n_scored=scored,
        excluded=excluded,
        error_types=errors,
    )


@dataclass(frozen=True)
class UngroundedReport:
    rate: float
    nonexistent: int
    """색인에 없는 조항 — 지어낸 인용. **통합형이 생성문 안에서 인용을 만들어내는 정도**를
    정량화하는 것이 이 지표의 목적이므로 이쪽이 더 중요한 신호다."""
    irrelevant: int
    """실존하지만 무관 — 결함코드·두께구간·품질수준 중 하나라도 어긋남."""
    n_citations: int

    def as_dict(self) -> dict:
        return {
            "ungrounded_rate": self.rate,
            "ungrounded_nonexistent": self.nonexistent,
            "ungrounded_irrelevant": self.irrelevant,
            "n_citations": self.n_citations,
        }


def score_ungrounded(
    cited_by_image: Mapping[str, Sequence[str]],
    index_clause_ids: Iterable[str],
    gold: Iterable[GoldPair],
) -> UngroundedReport:
    """무근거 인용률. 1차(미실존)와 2차(무관)를 **분해해서** 보고한다.

    없는 조항을 지어내는 것과 있는 조항을 잘못 고르는 것은 다른 실패다.
    """
    index = set(index_clause_ids)
    gold_by_image: dict[str, set[str]] = {}
    for g in gold:
        gold_by_image.setdefault(g.image_id, set()).add(g.clause_id)

    nonexistent = irrelevant = total = 0
    for img, cites in sorted(cited_by_image.items()):
        allowed = gold_by_image.get(img, set())
        for c in cites:
            total += 1
            if c not in index:
                nonexistent += 1
            elif c not in allowed:
                irrelevant += 1
    return UngroundedReport(
        rate=(nonexistent + irrelevant) / total if total else 0.0,
        nonexistent=nonexistent,
        irrelevant=irrelevant,
        n_citations=total,
    )


__all__ = [
    "FAIL",
    "PASS",
    "UNDECIDABLE",
    "AxisMismatch",
    "CitationReport",
    "ConsistencyReport",
    "DefectMeasure",
    "GoldPair",
    "RetrievalReport",
    "UngroundedReport",
    "row_matches_method",
    "score_citation",
    "score_consistency",
    "score_retrieval",
    "score_ungrounded",
    "select_row",
]
