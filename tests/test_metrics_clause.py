"""조항 정확도 축 테스트. 스펙 §10 "조항 검색 / 인용 일치율 / 판정 정합성 / 무근거 인용률" 행.

**쌍 단위 정의가 실제로 편중을 드러내는지**가 핵심이다 — 첫 결함만 질의하는 구현이
통과해 버리면 소수 클래스의 조항 검색 실패가 은폐된다.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from corpus.rules.schema import (
    ExtractionPath,
    InspectionMethod,
    LimitOp,
    LimitRow,
    LimitRule,
    LimitType,
    Material,
    QualityScheme,
    Scope,
    Unit,
)
from evaluation.metrics.clause import (
    DefectMeasure,
    GoldPair,
    row_matches_method,
    score_citation,
    score_consistency,
    score_retrieval,
    score_ungrounded,
    select_row,
)


def row(rule_id: str, limit: str, op: LimitOp = LimitOp.LE,
        method: InspectionMethod = InspectionMethod.RT) -> LimitRow:
    return LimitRow(
        rule_id=rule_id, canonical=True, scope=Scope.ACTIVE,
        defect_code="2011", material=Material.ST, inspection_method=method,
        thickness_min=Decimal(8), thickness_max=Decimal(25),
        quality_scheme=QualityScheme.ISO5817, quality_level="C",
        limit_type=LimitType.DIAMETER, limit_rule=LimitRule.CONST,
        limit_value=Decimal(limit), limit_op=op, unit=Unit.MM,
        clause_id="IACS47-3.2.1", source_doc="IACS Rec.47",
        filled_h=False, filled_v=False, extraction_path=ExtractionPath.D,
    )


GOLD = [
    GoldPair("img1", "2011", "IACS47-3.2.1"),
    GoldPair("img1", "301", "IACS47-3.4.1"),   # 다결함 이미지의 두 번째 결함
    GoldPair("img2", "2011", "IACS47-3.2.1"),
]


# --- 검색 top-1/3 (쌍 단위) ---------------------------------------------------

def test_retrieval_counts_pairs_not_images():
    """img1 은 결함이 2개이므로 쌍은 3개다. 이미지 단위면 2로 세어 편중이 숨는다."""
    retrieved = {
        ("img1", "2011"): ["IACS47-3.2.1"],
        ("img1", "301"): ["IACS47-3.4.1"],
        ("img2", "2011"): ["IACS47-3.2.1"],
    }
    r = score_retrieval(retrieved, GOLD)
    assert r.n_pairs == 3
    assert r.top1 == pytest.approx(1.0)


def test_first_defect_only_implementation_is_caught():
    """다수 클래스(기공)만 질의하는 구현. 커버리지가 이것을 드러내야 한다."""
    retrieved = {
        ("img1", "2011"): ["IACS47-3.2.1"],
        ("img2", "2011"): ["IACS47-3.2.1"],
    }
    r = score_retrieval(retrieved, GOLD)
    assert r.top1 == pytest.approx(1.0)          # 질의한 것만 보면 만점인데
    assert r.gt_coverage == pytest.approx(2 / 3)  # 슬래그 쌍이 통째로 빠졌다
    assert "301" not in r.per_code


def test_per_code_breakdown_exposes_minority_class():
    retrieved = {
        ("img1", "2011"): ["IACS47-3.2.1"],
        ("img2", "2011"): ["IACS47-3.2.1"],
        ("img1", "301"): ["IACS47-9.9.9"],   # 슬래그만 틀림
    }
    r = score_retrieval(retrieved, GOLD)
    assert r.per_code["2011"]["top1"] == pytest.approx(1.0)
    assert r.per_code["301"]["top1"] == pytest.approx(0.0)


def test_top3_counts_within_first_three():
    retrieved = {("img2", "2011"): ["X", "Y", "IACS47-3.2.1"]}
    r = score_retrieval(retrieved, GOLD)
    assert r.top1 == pytest.approx(0.0)
    assert r.top3 == pytest.approx(1.0)


def test_query_on_wrong_defect_code_counts_as_miss():
    """검출이 틀린 코드로 질의하면 검색기 탓이 아니어도 파이프라인 결과는 틀렸다."""
    retrieved = {("img2", "401"): ["IACS47-5.1.1"]}
    r = score_retrieval(retrieved, GOLD)
    assert r.top1 == pytest.approx(0.0)
    assert r.n_pairs == 1


# --- 인용 일치율 ---------------------------------------------------------------

def test_citation_is_pair_unit_with_image_level_set():
    cited = {"img1": ["IACS47-3.2.1", "IACS47-3.4.1"], "img2": ["IACS47-3.2.1"]}
    r = score_citation(cited, GOLD)
    assert r.n_pairs == 3
    assert r.match_rate == pytest.approx(1.0)


def test_citation_partial_on_multi_defect_image():
    cited = {"img1": ["IACS47-3.2.1"], "img2": ["IACS47-3.2.1"]}
    r = score_citation(cited, GOLD)
    assert r.match_rate == pytest.approx(2 / 3)


def test_citation_precision_penalises_shotgun_citing():
    """여러 개 인용해 맞히는 전략을 정밀도가 잡는다."""
    cited = {"img2": ["IACS47-3.2.1", "X", "Y", "Z"]}
    r = score_citation(cited, [GoldPair("img2", "2011", "IACS47-3.2.1")])
    assert r.match_rate == pytest.approx(1.0)
    assert r.precision == pytest.approx(0.25)


def test_empty_citation_is_mismatch():
    r = score_citation({}, GOLD)
    assert r.match_rate == pytest.approx(0.0)


# --- 판정 정합성 ---------------------------------------------------------------

def m(code: str, measured: str, rule: LimitRow, *, gold="IACS47-3.2.1", cited="IACS47-3.2.1"):
    return DefectMeasure(
        iso_code=code, row=rule, measured=Decimal(measured),
        gold_clause_id=gold, cited_clause_id=cited,
    )


def test_all_pass_image_is_pass():
    r5 = row("R1", "3.0")
    rep = score_consistency({"img1": "합격"}, {"img1": [m("2011", "2.0", r5)]})
    assert rep.rate == pytest.approx(1.0)
    assert rep.n_scored == 1


def test_conservative_rule_one_fail_makes_image_fail():
    """검사 실무 관례이자 B 골격 생성기와 공유하는 규칙 (게이트 #6 결정 F)."""
    r5 = row("R1", "3.0")
    measures = {"img1": [m("2011", "2.0", r5), m("301", "9.0", r5)]}
    ok = score_consistency({"img1": "불합격"}, measures)
    bad = score_consistency({"img1": "합격"}, measures)
    assert ok.rate == pytest.approx(1.0)
    assert bad.rate == pytest.approx(0.0)


def test_all_fail_image_is_fail():
    r5 = row("R1", "3.0")
    rep = score_consistency({"img1": "불합격"}, {"img1": [m("2011", "9.0", r5)]})
    assert rep.rate == pytest.approx(1.0)


def test_boundary_equality_is_pass_under_le():
    """경계 사례가 corpus 의 20%다. 여기서 흔들리면 지표가 실제보다 나빠 보인다."""
    r5 = row("R1", "3.0", LimitOp.LE)
    rep = score_consistency({"img1": "합격"}, {"img1": [m("2011", "3.0", r5)]})
    assert rep.rate == pytest.approx(1.0)


def test_normal_image_with_no_defect_is_pass():
    rep = score_consistency({"img9": "합격"}, {"img9": []})
    assert rep.rate == pytest.approx(1.0)
    assert rep.n_scored == 1


def test_no_citation_excluded_from_denominator():
    r5 = row("R1", "3.0")
    d = DefectMeasure("2011", r5, Decimal("2.0"), cited_clause_id=None)
    rep = score_consistency({"img1": "합격"}, {"img1": [d]})
    assert rep.n_scored == 0
    assert rep.excluded == {"근거 없음": 1}


def test_missing_measurement_excluded_and_reported():
    """스케일 부재 경로. 빼되 뺀 사실을 보고한다."""
    r5 = row("R1", "3.0")
    d = DefectMeasure("2011", r5, None, gold_clause_id="IACS47-3.2.1",
                      cited_clause_id="IACS47-3.2.1")
    rep = score_consistency({"img1": "합격"}, {"img1": [d]})
    assert rep.n_scored == 0
    assert rep.excluded == {"실측값 없음(스케일 부재)": 1}


def test_error_type_breakdown_wrong_clause():
    """조항을 틀렸는지 대소 비교를 틀렸는지 구분되지 않으면 해석이 불가능하다."""
    r5 = row("R1", "3.0")
    d = m("2011", "9.0", r5, gold="IACS47-3.2.1", cited="IACS47-OTHER")
    rep = score_consistency({"img1": "합격"}, {"img1": [d]})
    assert rep.error_types.get("이중 실패") == 1


def test_error_type_breakdown_comparison_failure():
    r5 = row("R1", "3.0")
    d = m("2011", "9.0", r5)          # 조항은 맞고 결론만 틀림
    rep = score_consistency({"img1": "합격"}, {"img1": [d]})
    assert rep.error_types.get("대소 비교 실패") == 1


def test_undecidable_when_clause_exists_is_evasion():
    r5 = row("R1", "3.0")
    rep = score_consistency({"img1": "판정불가"}, {"img1": [m("2011", "2.0", r5)]})
    assert rep.rate == pytest.approx(0.0)
    assert rep.error_types.get("판정 회피") == 1


# --- 검사축 적용 (계약 #3, 게이트 #13 결정 L) -------------------------------------

def test_axis_mismatch_excluded_from_denominator():
    """VT 행을 RT 이미지에 적용하면 부등식은 멀쩡히 계산된다 — 조용히 채점하면
    틀린 값이 정상처럼 집계된다. 판정 정합성이 틀린 채 높게 나오는 최악의 형태."""
    vt = row("R-VT", "3.0", method=InspectionMethod.VT)
    d = DefectMeasure("2011", vt, Decimal("2.0"), gold_clause_id="IACS47-3.2.1",
                      cited_clause_id="IACS47-3.2.1", inspection_method="RT")
    rep = score_consistency({"img1": "합격"}, {"img1": [d]})
    assert rep.n_scored == 0
    assert rep.excluded == {"검사 방식 불일치(축 오적용)": 1}


def test_matching_axis_is_scored_normally():
    rt = row("R-RT", "3.0", method=InspectionMethod.RT)
    d = DefectMeasure("2011", rt, Decimal("2.0"), gold_clause_id="IACS47-3.2.1",
                      cited_clause_id="IACS47-3.2.1", inspection_method="RT")
    rep = score_consistency({"img1": "합격"}, {"img1": [d]})
    assert rep.n_scored == 1
    assert rep.rate == pytest.approx(1.0)


def test_all_axis_row_answers_any_image():
    any_row = row("R-ALL", "3.0", method=InspectionMethod.ALL)
    assert row_matches_method(any_row, "RT")
    assert row_matches_method(any_row, "VT")


def test_row_selection_goes_through_b_functions():
    """행 선택 단일 경로 — D가 자체 순회로 고르지 않는다. 축이 키에 들어간다."""
    rt = row("R-RT", "3.0", method=InspectionMethod.RT)
    vt = row("R-VT", "9.0", method=InspectionMethod.VT)
    picked = select_row(
        [rt, vt], clause_id="IACS47-3.2.1", defect_code="2011", material="ST",
        inspection_method="RT", quality_scheme="iso5817", quality_level="C",
        thickness_mm=Decimal(12),
    )
    assert picked.rule_id == "R-RT"


def test_row_selection_rejects_unknown_clause():
    rt = row("R-RT", "3.0")
    with pytest.raises(LookupError):
        select_row(
            [rt], clause_id="NOPE", defect_code="2011", material="ST",
            inspection_method="RT", quality_scheme="iso5817", quality_level="C",
            thickness_mm=Decimal(12),
        )


# --- 무근거 인용률 --------------------------------------------------------------

INDEX = ["IACS47-3.2.1", "IACS47-3.4.1", "IACS47-9.9.9"]


def test_nonexistent_clause_is_hallucination():
    r = score_ungrounded({"img2": ["MADE-UP-1"]}, INDEX, GOLD)
    assert (r.nonexistent, r.irrelevant) == (1, 0)
    assert r.rate == pytest.approx(1.0)


def test_existing_but_irrelevant_clause():
    r = score_ungrounded({"img2": ["IACS47-9.9.9"]}, INDEX, GOLD)
    assert (r.nonexistent, r.irrelevant) == (0, 1)


def test_correct_citation_is_grounded():
    r = score_ungrounded({"img2": ["IACS47-3.2.1"]}, INDEX, GOLD)
    assert r.rate == pytest.approx(0.0)


def test_two_failure_kinds_reported_separately():
    """지어낸 것과 잘못 고른 것은 다른 실패다."""
    r = score_ungrounded({"img2": ["MADE-UP", "IACS47-9.9.9"]}, INDEX, GOLD)
    assert (r.nonexistent, r.irrelevant, r.n_citations) == (1, 1, 2)
