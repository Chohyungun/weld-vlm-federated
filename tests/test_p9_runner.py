"""P9 브리지 테스트 — 공통 스키마 출력 → 교차 출처 오탐 판정.

핵심 셋: 오탐 정의가 다섯 칸에 하나인가, 결측을 '깨끗함'으로 읽지 않는가,
N-band 가 집계에서 빠지되 세어지는가.
"""

from __future__ import annotations

import pytest

from evaluation.probes.cross_source import BAND, CROP, TILE
from evaluation.probes.p9_runner import (
    EvalNormalContext,
    contexts_from_snapshot,
    is_false_positive,
    p9_all_cells,
    p9_for_cell,
)
from evaluation.schema import SCHEMA_VERSION, PredictionRecord, failed_record


def rec(image_id: str, codes=(), cell="sep_central", seed=0) -> PredictionRecord:
    return PredictionRecord(
        schema_version=SCHEMA_VERSION,
        image_id=image_id,
        cell=cell,
        seed=seed,
        defects=[
            {"iso_code": c, "bbox_px": [1.0, 1.0, 9.0, 9.0], "score": 0.9,
             "retrieved": None}
            for c in codes
        ],
        verdict="불합격" if codes else "합격",
        cited_clauses=[],
        parse_ok=True,
    )


def ctx(iid: str, src: str, group: str | None = None) -> EvalNormalContext:
    return EvalNormalContext(iid, group or f"g_{iid}", src)


def make_contexts(n_crop=30, n_tile=30):
    out = [ctx(f"c{i}", CROP) for i in range(n_crop)]
    out += [ctx(f"t{i}", TILE) for i in range(n_tile)]
    return out


# --- 오탐 정의 (다섯 칸 공통 단일) --------------------------------------------------

def test_fp_is_predicted_defect_on_normal():
    assert is_false_positive(rec("a", ["2011"]))
    assert not is_false_positive(rec("a"))


def test_parse_failure_is_not_a_false_positive():
    """파싱 실패는 예측 집합 공집합(스펙 §4-1) — 오탐이 아니라 실패율이 센다."""
    assert not is_false_positive(failed_record("a", "uni_fed", 0, "no_json"))


def test_duplicate_codes_still_one_fp():
    assert is_false_positive(rec("a", ["2011", "2011"]))


# --- 스냅샷 → 맥락 ----------------------------------------------------------------

def test_contexts_take_eval_normals_only():
    rows = [
        {"image_id": "a", "split": "eval", "has_defect": "False", "group_id": "g1"},
        {"image_id": "b", "split": "eval", "has_defect": "True", "group_id": "g2"},
        {"image_id": "c", "split": "train", "has_defect": "False", "group_id": "g3"},
    ]
    got, missing = contexts_from_snapshot(rows, {"a": CROP, "b": TILE, "c": TILE})
    assert [c.image_id for c in got] == ["a"]
    assert missing == ()


def test_missing_provenance_is_reported_not_dropped():
    rows = [{"image_id": "a", "split": "eval", "has_defect": "False", "group_id": "g1"}]
    got, missing = contexts_from_snapshot(rows, {})
    assert got == () and missing == ("a",)


# --- 칸 하나의 P9 ------------------------------------------------------------------

def test_equivalent_when_fp_rates_match():
    """같은 FP 율이면 동등 판정. 표본은 실데이터 규모(수백 묶음)에 맞춘다 —
    묶음 40개로는 CI 가 δ 보다 넓어 TOST 가 동등성을 주장하지 못하는데,
    그것은 결함이 아니라 저검정력 통과를 막는 성질이다."""
    contexts = make_contexts(200, 400)
    records = [rec(c.image_id, ["2011"] if i % 10 == 0 else [])
               for i, c in enumerate(contexts)]
    r = p9_for_cell(records, contexts)
    assert r.report.tost.equivalent
    assert "한계 문단" in r.verdict


def test_small_sample_cannot_claim_equivalence_even_if_rates_match():
    """FP 율이 같아도 묶음이 적으면 CI 가 δ 를 넘어 동등 판정이 나오지 않는다."""
    contexts = make_contexts(40, 40)
    records = [rec(c.image_id, ["2011"] if i % 10 == 0 else [])
               for i, c in enumerate(contexts)]
    r = p9_for_cell(records, contexts)
    assert not r.report.tost.equivalent


def test_not_equivalent_when_fp_concentrates_on_tiles():
    """오탐이 출처로 쏠리는 경우 — 크롭 한정본 재검토 신호."""
    contexts = make_contexts(200, 400)
    records = [
        rec(c.image_id, ["2011"] if c.provenance == TILE and i % 2 == 0 else [])
        for i, c in enumerate(contexts)
    ]
    r = p9_for_cell(records, contexts)
    assert not r.report.tost.equivalent
    assert "재검토" in r.verdict


def test_band_excluded_from_aggregation_but_counted():
    contexts = make_contexts(30, 30) + [ctx(f"b{i}", BAND) for i in range(5)]
    records = [rec(c.image_id) for c in contexts]
    r = p9_for_cell(records, contexts)
    assert r.n_band_excluded == 5
    assert BAND not in {TILE, CROP}
    assert r.breakdown["전체"]["n_images"] == 60      # band 5 는 집계 밖


def test_missing_prediction_counted_not_treated_clean():
    """채점 누락을 '오탐 아님'으로 채우면 오탐률이 조용히 내려간다."""
    contexts = make_contexts(30, 30)
    records = [rec(c.image_id) for c in contexts[:50]]     # 10장 누락
    r = p9_for_cell(records, contexts)
    assert r.n_missing_prediction == 10
    assert r.breakdown["전체"]["n_images"] == 50


def test_mixed_cells_rejected():
    contexts = make_contexts(5, 5)
    records = [rec("c0", cell="sep_central"), rec("t0", cell="sep_fed")]
    with pytest.raises(ValueError):
        p9_for_cell(records, contexts)


def test_underpowered_crop_side_is_undecidable():
    """N-crop 묶음이 적으면 '판정 불가' — 통과가 아니다."""
    contexts = make_contexts(5, 40)
    records = [rec(c.image_id) for c in contexts]
    r = p9_for_cell(records, contexts, min_clusters=20)
    assert not r.report.tost.equivalent
    assert "판정 불가" in r.verdict


# --- 전 칸 일괄 --------------------------------------------------------------------

def test_all_cells_split_by_cell_and_seed():
    contexts = make_contexts(30, 30)
    records = []
    for cell in ("sep_central", "uni_central"):
        records += [rec(c.image_id, cell=cell) for c in contexts]
    records += [rec(c.image_id, cell="sep_central", seed=1) for c in contexts]
    summary = p9_all_cells(records, contexts)
    assert [(r.cell, r.seed) for r in summary.results] == [
        ("sep_central", 0), ("sep_central", 1), ("uni_central", 0),
    ]


def test_summary_carries_no_proof_caveat():
    """'통과 = 신호 없음의 증명'으로 읽히지 않게 하는 문장이 결과에 붙어 다닌다."""
    contexts = make_contexts(30, 30)
    summary = p9_all_cells([rec(c.image_id) for c in contexts], contexts)
    assert any("증명이 아니라" in c for c in summary.caveats)


def test_same_function_serves_separated_and_unified():
    """칸 이름으로 분기하지 않는다 — 통합형 레코드도 같은 경로를 탄다."""
    contexts = make_contexts(30, 30)
    uni = []
    for c in contexts:
        r = rec(c.image_id, cell="uni_fed")
        uni.append(r)
    result = p9_for_cell(uni, contexts)
    assert result.cell == "uni_fed"
    assert result.report.tost.equivalent
