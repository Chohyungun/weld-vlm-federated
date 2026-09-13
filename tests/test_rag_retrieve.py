"""조항 검색 테스트. 스펙 §10 "검색" 행.

**계약 #3 변경(검사축)의 회귀 테스트**가 핵심이다 — 같은 결함코드·같은 두께의 표면(VT)·
내부(RT) 기공 행이 공존할 때 RT 질의가 RT 행만 집어야 한다. 축이 없으면 잘못된 행을
집어도 형식상 정상으로 보이고, 판정 정합성이 틀린 채로 높게 나온다.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from rag.retrieve import (
    NO_CLAUSE,
    Chunk,
    EmbeddingTrial,
    Query,
    chunk_from_meta,
    filter_chunks,
    index_stats,
    pick_embedding,
    retrieve,
)


def chunk(cid: str, method="RT", codes=("2011",), tmin="8", tmax="25",
          scheme="iso5817", levels=("C",), scope="active") -> Chunk:
    methods = () if method is None else (method,) if isinstance(method, str) else tuple(method)
    return Chunk(
        chunk_id=cid, doc="IACS Rec.47", clause_id=cid,
        inspection_methods=methods, defect_codes=tuple(codes),
        thickness_min=None if tmin is None else Decimal(tmin),
        thickness_max=None if tmax is None else Decimal(tmax),
        quality_scheme=scheme, quality_levels=tuple(levels), scope=scope,
    )


def q(method="RT", code="2011", t="12", level="C") -> Query:
    return Query(
        inspection_method=method, defect_code=code,
        material="ST",
        thickness_mm=None if t is None else Decimal(t),
        quality_scheme="iso5817", quality_level=level,
    )


# --- 검사축 (계약 #3 변경 회귀) --------------------------------------------------

def test_rt_query_does_not_pick_vt_chunk():
    chunks = [chunk("RT-1", method="RT"), chunk("VT-1", method="VT")]
    assert retrieve(chunks, q(method="RT")).chunk_ids == ("RT-1",)


def test_surface_and_internal_porosity_coexist_without_collision():
    """같은 결함코드·같은 두께 구간인데 검사축만 다른 두 행.
    축이 없으면 후보 2건이 되어 top-1 이 흔들리고, 잘못된 허용치를 집는다."""
    chunks = [chunk("INTERNAL", method="RT"), chunk("SURFACE", method="VT")]
    r = retrieve(chunks, q(method="RT"))
    assert r.chunk_ids == ("INTERNAL",)
    assert r.n_candidates == 1
    assert r.used_dense is False        # 임베딩 호출조차 없다


def test_vt_chunk_stays_in_index_for_ungrounded_check():
    """필터에서 탈락하되 색인에는 남는다 — 빼면 무근거 인용 1차 판정 기준 집합이 좁아진다."""
    chunks = [chunk("RT-1", method="RT"), chunk("VT-1", method="VT")]
    assert index_stats(chunks).methods == {"RT": 1, "VT": 1}


def test_method_null_chunk_matches_any():
    chunks = [chunk("ANY", method=None)]
    assert retrieve(chunks, q(method="VT")).chunk_ids == ("ANY",)


def test_all_method_chunk_answers_both_axes():
    """검사 방법과 무관한 조항은 RT·VT 어느 질의에도 응한다 (B InspectionMethod.ALL)."""
    chunks = [chunk("ANY", method="ALL")]
    assert retrieve(chunks, q(method="RT")).chunk_ids == ("ANY",)
    assert retrieve(chunks, q(method="VT")).chunk_ids == ("ANY",)


def test_chunk_from_derive_chunk_meta_carries_axis():
    """B 의 derive_chunk_meta 출력을 그대로 받는다 — 필드명이 어긋나면 축 필터가
    조용히 통과해 버리므로 이 변환이 유일한 지점이다."""
    meta = {
        "clause_id": "IACS47-3.2.1",
        "source_docs": ["IACS Rec.47"],
        "defect_codes": ["2011"],
        "inspection_methods": ["RT"],
        "quality_schemes": ["iso5817"],
        "quality_levels": ["C"],
        "thickness_min": "8",
        "thickness_max": "25",
        "scope": "active",
    }
    c = chunk_from_meta(meta)
    assert c.inspection_methods == ("RT",)
    assert retrieve([c], q(method="RT")).chunk_ids == ("IACS47-3.2.1",)
    assert retrieve([c], q(method="VT")).chunk_ids == ()


# --- 두께 반구간 ----------------------------------------------------------------

def test_thickness_lower_bound_inclusive():
    assert filter_chunks([chunk("A", tmin="8", tmax="25")], q(t="8"))


def test_thickness_upper_bound_exclusive():
    """양끝 포함이면 경계 두께에서 두 조항이 동시에 걸린다."""
    assert not filter_chunks([chunk("A", tmin="8", tmax="25")], q(t="25"))


def test_adjacent_intervals_never_both_match():
    chunks = [chunk("LOW", tmin="8", tmax="25"), chunk("HIGH", tmin="25", tmax="50")]
    r = retrieve(chunks, q(t="25"))
    assert r.chunk_ids == ("HIGH",)


def test_open_upper_bound():
    assert filter_chunks([chunk("A", tmin="25", tmax=None)], q(t="999"))


# --- 등급 어휘 (grade_map 경유) ---------------------------------------------------

GRADE_MAP = {"iacs": {"C": ["STD"]}, "iso5817": {"C": ["C"]}}


def test_iacs_chunk_unreachable_without_grade_map():
    """정확 매칭만 쓰면 IACS 유래 청크가 전부 탈락한다 — 색인이 주력 문서라 치명적이다."""
    chunks = [chunk("IACS-1", scheme="iacs", levels=("STD",))]
    assert retrieve(chunks, q(level="C")).chunk_ids == ()


def test_grade_map_makes_iacs_chunk_reachable():
    chunks = [chunk("IACS-1", scheme="iacs", levels=("STD",))]
    r = retrieve(chunks, q(level="C"), GRADE_MAP)
    assert r.chunk_ids == ("IACS-1",)


def test_level_free_clause_always_matches():
    assert filter_chunks([chunk("A", levels=())], q(level="C"))


# --- 후보 수에 따른 분기 -----------------------------------------------------------

def test_zero_candidates_returns_no_clause_reason():
    """환각으로 채우지 않는 것이 이 경로의 목적이다."""
    r = retrieve([chunk("A", codes=("301",))], q(code="2011"))
    assert r.found is False
    assert r.reason == NO_CLAUSE


def test_single_candidate_skips_embedding():
    r = retrieve([chunk("A")], q())
    assert (r.n_candidates, r.used_dense) == (1, False)


def test_multiple_candidates_use_dense_and_cap_at_top_k():
    chunks = [chunk(f"C{i}", tmin="8", tmax="25") for i in range(5)]
    r = retrieve(chunks, q())
    assert len(r.chunk_ids) == 3
    assert r.used_dense is True


def test_ranker_is_used_when_provided():
    chunks = [chunk("A"), chunk("B")]
    seen = {}

    def rank(text, cands):
        seen["text"] = text
        return sorted(cands, key=lambda c: c.chunk_id, reverse=True)

    r = retrieve(chunks, q(), rank=rank)
    assert r.chunk_ids == ("B", "A")
    assert "결함코드=2011" in seen["text"]


def test_default_order_is_deterministic_not_input_order():
    a = retrieve([chunk("Z"), chunk("A")], q()).chunk_ids
    b = retrieve([chunk("A"), chunk("Z")], q()).chunk_ids
    assert a == b == ("A", "Z")


# --- 질의 직렬화 ------------------------------------------------------------------

def test_query_serialisation_is_deterministic():
    """같은 키는 언제나 같은 문자열 — 이것이 '자연어 질의 생성'과 다른 점이다."""
    assert q().to_text() == q().to_text()


def test_query_serialisation_includes_inspection_method():
    assert "검사방식=RT" in q(method="RT").to_text()


# --- 임베딩 선정 ------------------------------------------------------------------

def test_embedding_picked_by_measured_top1():
    trials = [EmbeddingTrial("KURE-v1", 0.97, 0.99, 100),
              EmbeddingTrial("BGE-M3", 0.93, 0.99, 100)]
    assert pick_embedding(trials) == "KURE-v1"


def test_tie_breaks_to_bge_m3():
    trials = [EmbeddingTrial("KURE-v1", 0.97, 0.99, 100),
              EmbeddingTrial("BGE-M3", 0.97, 0.99, 100)]
    assert pick_embedding(trials) == "BGE-M3"


def test_refuses_to_pick_without_measurement():
    """벤치마크 순위로 고르지 않는다."""
    with pytest.raises(ValueError):
        pick_embedding([])


# --- 색인 통계 --------------------------------------------------------------------

def test_excluded_chunks_counted_separately():
    chunks = [chunk("A"), chunk("B", scope="excluded")]
    s = index_stats(chunks)
    assert (s.n_active, s.n_excluded) == (1, 1)
