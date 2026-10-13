"""계약 #4 스키마 테스트. 스펙 §10 "스키마" 행.

계약이 지키는 것은 두 가지다: **오타 필드를 조용히 삼키지 않는 것**과 **어떤 값도
보정하지 않는 것**. 둘 다 어기면 "채웠다고 믿었는데 안 채워진" 사고가 난다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.schema import (
    SCHEMA_VERSION,
    PredictionRecord,
    failed_record,
    json_schema,
    parse_record,
)

SCHEMA_PATH = Path("evaluation/prediction.schema.json")


def rec(**overrides) -> dict:
    base = {
        "schema_version": SCHEMA_VERSION,
        "image_id": "aihub71761:RT/ST/w00001_f0.png",
        "cell": "sep_central",
        "seed": 0,
        "defects": [
            {
                "iso_code": "2011",
                "bbox_px": [10.0, 20.0, 40.0, 55.0],
                "score": 0.87,
                "size_px": 30.0,
                "size_basis": "equiv_diameter",
                "retrieved": ["IACS47-3.2.1", "IACS47-3.2.2"],
            }
        ],
        "verdict": "합격",
        "cited_clauses": ["IACS47-3.2.1"],
        "parse_ok": True,
    }
    base.update(overrides)
    return base


# --- 기본 통과 -----------------------------------------------------------------

def test_valid_record_parses():
    r = PredictionRecord.model_validate(rec())
    assert r.image_id.endswith("w00001_f0.png")
    assert r.defects[0].iso_code == "2011"


def test_empty_defects_means_no_defect_predicted():
    r = PredictionRecord.model_validate(rec(defects=[], cited_clauses=[]))
    assert r.defects == []
    assert r.iso_codes == frozenset()


# --- 미지 필드 거부 (additionalProperties: false) --------------------------------

def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(rec(bbox_normalized=[0, 0, 1, 1]))


def test_unknown_defect_field_rejected():
    """네이티브 좌표 필드가 실수로 되살아나는 경로를 막는다 (게이트 §5-1)."""
    payload = rec()
    payload["defects"][0]["bbox_2d"] = [1, 2, 3, 4]
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(payload)


# --- enum 위반 ------------------------------------------------------------------

def test_verdict_enum_violation_rejected():
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(rec(verdict="pass"))


def test_verdict_undecidable_is_valid():
    """'판정불가'는 오답이 아니라 정상 경로다 — 검색 0건·스케일 부재에서 나온다."""
    r = PredictionRecord.model_validate(rec(verdict="판정불가", cited_clauses=[]))
    assert r.verdict == "판정불가"


def test_coord_space_uses_c_vocabulary():
    r = PredictionRecord.model_validate(rec(coord_space="NORM_1000"))
    assert r.coord_space == "NORM_1000"


def test_old_coord_space_vocabulary_rejected():
    """D 초안 어휘(original_px 등)는 게이트 #6에서 폐기됐다."""
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(rec(coord_space="original_px"))


# --- bbox 구조 검증 --------------------------------------------------------------

def test_bbox_with_three_elements_rejected():
    payload = rec()
    payload["defects"][0]["bbox_px"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(payload)


def test_bbox_null_allowed_for_riawelc():
    """위치 라벨 없는 시나리오. 필드를 없애면 스키마가 갈라지고 채점기가 둘이 된다."""
    payload = rec()
    payload["defects"][0]["bbox_px"] = None
    r = PredictionRecord.model_validate(payload)
    assert r.defects[0].bbox_px is None


@pytest.mark.parametrize("bad", [
    [40.0, 20.0, 10.0, 55.0],   # x1 > x2
    [10.0, 55.0, 40.0, 20.0],   # y1 > y2
    [10.0, 20.0, 10.0, 55.0],   # 폭 0
])
def test_degenerate_bbox_rejected(bad):
    payload = rec()
    payload["defects"][0]["bbox_px"] = bad
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(payload)


def test_degenerate_bbox_is_not_silently_swapped():
    """자동 스왑 금지 — 보정하면 채점기가 답을 고쳐주는 것이다."""
    payload = rec()
    payload["defects"][0]["bbox_px"] = [40.0, 20.0, 10.0, 55.0]
    parsed = parse_record(payload)
    assert parsed == (None, "schema_violation")


def test_nan_bbox_rejected():
    payload = rec()
    payload["defects"][0]["bbox_px"] = [float("nan"), 20.0, 40.0, 55.0]
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(payload)


# --- 교차 필드 규칙 ---------------------------------------------------------------

def test_sep_local_requires_client():
    """sep_local 만 모델이 3개다. RQ3(C3 편익) 분해에 필요하다."""
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(rec(cell="sep_local", client=None))


def test_non_local_cell_rejects_client():
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(rec(cell="sep_central", client="C1"))


def test_unified_cell_rejects_retrieved():
    """통합형에 검색을 붙이지 않는 비대칭이 근거 신뢰도 비교의 대비축이다."""
    payload = rec(cell="uni_central")
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(payload)


def test_unified_cell_without_retrieved_is_valid():
    payload = rec(cell="uni_central")
    payload["defects"][0]["retrieved"] = None
    payload["defects"][0]["score"] = None  # 생성 모델이라 신뢰도가 없다
    r = PredictionRecord.model_validate(payload)
    assert r.defects[0].retrieved is None


def test_retrieved_capped_at_top_k():
    payload = rec()
    payload["defects"][0]["retrieved"] = ["a", "b", "c", "d"]
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(payload)


def test_parse_failure_requires_reason():
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(rec(parse_ok=False, parse_error=None))


def test_parse_ok_rejects_reason():
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate(rec(parse_ok=True, parse_error="truncated"))


# --- 쌍 단위 지표의 기본 단위 -------------------------------------------------------

def test_pairs_are_image_by_iso_code():
    payload = rec()
    payload["defects"].append({"iso_code": "301", "bbox_px": [1.0, 1.0, 5.0, 5.0]})
    r = PredictionRecord.model_validate(payload)
    assert r.pairs() == (
        (r.image_id, "2011"),
        (r.image_id, "301"),
    )


def test_pairs_deduplicate_repeated_codes():
    """같은 결함을 두 번 언급해도 질의는 한 번이다 — 쌍 분모가 부풀지 않아야 한다."""
    payload = rec()
    payload["defects"].append({"iso_code": "2011", "bbox_px": [1.0, 1.0, 5.0, 5.0]})
    r = PredictionRecord.model_validate(payload)
    assert len(r.pairs()) == 1


# --- 실패 레코드 -------------------------------------------------------------------

def test_failed_record_is_countable_not_dropped():
    r = failed_record("img1", "uni_fed", 0, "no_json")
    assert r.parse_ok is False
    assert r.parse_error == "no_json"
    assert r.defects == []
    assert r.verdict == "판정불가"


def test_parse_record_returns_reason_on_bad_json():
    assert parse_record("{not json") == (None, "json_decode")


# --- JSON Schema 동기화 -------------------------------------------------------------

def test_committed_json_schema_matches_model():
    """스키마 파일을 손으로 고치는 경로를 막는다. 두 곳이 갈리면 C와 D가 다른 계약을 본다."""
    assert SCHEMA_PATH.exists(), f"{SCHEMA_PATH} 가 없다 — 생성 스크립트를 돌려라"
    committed = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert committed == json_schema()
