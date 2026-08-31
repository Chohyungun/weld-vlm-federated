"""색인 구축 · 임베딩 하네스 · 판정부 배선 테스트. 지시서 과제 1~3.

전부 CPU·스텁으로 돈다. GPU 는 총괄 신호 대기.
"""

from __future__ import annotations

import json
from decimal import Decimal

import numpy as np
import pytest

from rag.embedding_trial import decide, make_dense_ranker, run_trial
from rag.index import (
    MissingEmbeddingModel,
    build_index,
    load_chunks,
    load_rag_config,
    queries_from_gold_rows,
)
from rag.judge import (
    UNDECIDABLE,
    judge_image,
    load_prompt_template,
    parse_generation,
    run_judge,
)
from rag.retrieve import Chunk, Query, RetrievalResult


def meta(clause: str, method="RT", codes=("2011",), tmin="8", tmax="25",
         levels=("ALL",), scheme="none", scope="active") -> dict:
    return {
        "clause_id": clause, "source_docs": ["KR-RULES-P2"],
        "defect_codes": list(codes), "materials": ["ST"],
        "inspection_methods": [method], "quality_schemes": [scheme],
        "quality_levels": list(levels), "thickness_min": tmin,
        "thickness_max": tmax, "scope": scope, "rule_ids": [f"R-{clause}"],
    }


def write_chunk_meta(tmp_path, metas):
    p = tmp_path / "chunk_meta.jsonl"
    p.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in metas),
                 encoding="utf-8")
    return p


def q(code="2011", t="12", method="RT", level="ALL", scheme="none") -> Query:
    return Query(inspection_method=method, defect_code=code,
                 thickness_mm=Decimal(t), quality_scheme=scheme, quality_level=level)


# --- 과제 1: 색인 구축 -------------------------------------------------------------

def test_config_is_single_source_for_top_k():
    cfg = load_rag_config()
    assert cfg.top_k == 3
    assert cfg.no_hit_output == "해당 조항 없음"


def test_embedding_model_starts_unset():
    """실측 전이다. null 이어야 하고, 값이 생기면 실측이 끝났다는 뜻이다."""
    assert load_rag_config().embedding_model is None


def test_index_builds_from_chunk_meta_contract(tmp_path):
    p = write_chunk_meta(tmp_path, [meta("KRA27-T15"), meta("KRA27-T16", codes=("100",))])
    idx = build_index(p)
    assert len(idx.chunks) == 2


def test_duplicate_clause_ids_rejected(tmp_path):
    """조항 단위 청킹이 깨지면(같은 조항이 두 청크) 색인이 거부한다."""
    p = write_chunk_meta(tmp_path, [meta("KRA27-T15"), meta("KRA27-T15")])
    with pytest.raises(ValueError):
        load_chunks(p)


def test_single_candidate_needs_no_embedding(tmp_path):
    """후보 0~1개 경로는 임베딩 모델 없이 돌아야 한다 — 지시서 요구."""
    p = write_chunk_meta(tmp_path, [meta("KRA27-T15")])
    idx = build_index(p)
    r = idx.search(q())
    assert r.chunk_ids == ("KRA27-T15",)
    assert not r.used_dense


def test_zero_candidates_needs_no_embedding(tmp_path):
    p = write_chunk_meta(tmp_path, [meta("KRA27-T15", codes=("100",))])
    r = build_index(p).search(q(code="2011"))
    assert not r.found


def test_multiple_candidates_without_model_fail_loudly(tmp_path):
    """사전순 폴백으로 낸 top-1 이 실측처럼 읽히면 선정 절차가 무의미해진다."""
    p = write_chunk_meta(tmp_path, [meta("A"), meta("B")])
    with pytest.raises(MissingEmbeddingModel):
        build_index(p).search(q())


def test_snapshot_digest_is_deterministic(tmp_path):
    p = write_chunk_meta(tmp_path, [meta("A"), meta("B", codes=("100",))])
    assert build_index(p).snapshot_digest() == build_index(p).snapshot_digest()


# --- 과제 2: 임베딩 하네스 ----------------------------------------------------------

GOLD_ROWS = [
    {"rule_id": "R1", "clause_id": "C-2011", "defect_code": "2011", "material": "ST",
     "inspection_method": "RT", "thickness_min": "8", "thickness_max": "25",
     "quality_scheme": "none", "quality_level": "ALL", "limit_type": "직경"},
    {"rule_id": "R2", "clause_id": "C-100", "defect_code": "100", "material": "ST",
     "inspection_method": "RT", "thickness_min": "8", "thickness_max": "25",
     "quality_scheme": "none", "quality_level": "ALL", "limit_type": "길이"},
]


def test_query_generation_is_deterministic():
    a = queries_from_gold_rows(GOLD_ROWS, n=100, seed=20260825)
    b = queries_from_gold_rows(GOLD_ROWS, n=100, seed=20260825)
    assert len(a) == 100
    assert [(x.to_text(), g) for x, g in a] == [(x.to_text(), g) for x, g in b]


def test_queries_stay_inside_half_open_interval():
    for query, _ in queries_from_gold_rows(GOLD_ROWS, n=100, seed=1):
        assert Decimal(8) <= query.thickness_mm < Decimal(25)


def test_boundary_queries_included():
    """경계 사례 20% — 상한 직전 두께가 섞여 있어야 한다."""
    qs = [x for x, _ in queries_from_gold_rows(GOLD_ROWS, n=100, seed=20260825)]
    near_top = sum(1 for x in qs if x.thickness_mm > Decimal(24))
    assert near_top >= 10


def stub_embedder(vocab: dict[str, list[float]]):
    def embed(texts):
        return np.array([vocab.get(t, [1.0, 0.0]) for t in texts], dtype=float)
    return embed


def test_trial_scores_top1_with_stub_embedder():
    chunks = (
        Chunk(chunk_id="C-2011", doc="d", clause_id="C-2011",
              inspection_methods=("RT",), defect_codes=("2011",),
              thickness_min=Decimal(8), thickness_max=Decimal(25),
              quality_scheme="none", quality_levels=("ALL",), text="기공 조항"),
        Chunk(chunk_id="C-2011b", doc="d", clause_id="C-2011b",
              inspection_methods=("RT",), defect_codes=("2011",),
              thickness_min=Decimal(8), thickness_max=Decimal(25),
              quality_scheme="none", quality_levels=("ALL",), text="다른 조항"),
    )
    queries = [(q(), "C-2011")] * 10
    vocab = {"기공 조항": [1.0, 0.0], "다른 조항": [0.0, 1.0]}
    good = stub_embedder({**vocab, queries[0][0].to_text(): [1.0, 0.0]})
    bad = stub_embedder({**vocab, queries[0][0].to_text(): [0.0, 1.0]})
    r_good = run_trial("good", queries, chunks, good)
    r_bad = run_trial("bad", queries, chunks, bad)
    assert r_good.trial.top1 == pytest.approx(1.0)
    assert r_bad.trial.top1 == pytest.approx(0.0)
    assert r_good.n_dense == 10


def test_trial_counts_lookup_only_queries_separately():
    """후보 1개 질의는 임베딩과 무관하다 — 그 몫이 크면 선택의 영향 자체가 작다."""
    chunks = (
        Chunk(chunk_id="C-100", doc="d", clause_id="C-100",
              inspection_methods=("RT",), defect_codes=("100",),
              thickness_min=Decimal(8), thickness_max=Decimal(25),
              quality_scheme="none", quality_levels=("ALL",), text="균열"),
    )
    r = run_trial("m", [(q(code="100"), "C-100")] * 5, chunks,
                  stub_embedder({}))
    assert r.n_lookup_only == 5 and r.n_dense == 0
    assert r.trial.top1 == pytest.approx(1.0)


def test_decide_picks_measured_winner():
    chunks = ()
    a = run_trial("BAAI/bge-m3", [], chunks, stub_embedder({}))
    # 실측 없는 선정은 pick_embedding 이 거부한다 — 빈 결과로 확인
    with pytest.raises(ValueError):
        decide([])
    d = decide([a])
    assert d["winner"] == "BAAI/bge-m3"


def test_dense_ranker_pushes_textless_chunks_down():
    """본문 없는 청크가 우연히 1위가 되지 않는다."""
    chunks = [
        Chunk(chunk_id="EMPTY", doc="d", clause_id="EMPTY",
              inspection_methods=("RT",), defect_codes=("2011",), text=""),
        Chunk(chunk_id="FULL", doc="d", clause_id="FULL",
              inspection_methods=("RT",), defect_codes=("2011",), text="본문"),
    ]
    rank = make_dense_ranker(stub_embedder({"본문": [0.0, 1.0]}))
    ordered = rank("질의", chunks)
    assert ordered[0].chunk_id == "FULL"


# --- 과제 3: 판정부 배선 -----------------------------------------------------------

def hit(ids) -> RetrievalResult:
    return RetrievalResult(tuple(ids), True, len(ids))


def no_hit() -> RetrievalResult:
    return RetrievalResult((), False, 0, "해당 조항 없음")


def chunk_for(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, doc="d", clause_id=cid,
                 inspection_methods=("RT",), defect_codes=("2011",), text="조항 본문")


TEMPLATE, _DIGEST = load_prompt_template()
DEFECTS = [{"iso_code": "2011", "size_px": 12.0}]


def test_zero_hit_skips_generation_entirely():
    """검색 0건 경로 — 생성을 부르지 않고 결정론적으로 판정불가."""
    calls = []

    def gen(prompt):
        calls.append(prompt)
        return "{}"

    out = judge_image("img1", DEFECTS, no_hit(), [], gen, TEMPLATE)
    assert calls == []                      # 생성 미호출
    assert out.verdict == UNDECIDABLE
    assert out.basis == "해당 조항 없음"
    assert not out.generated and out.parse_ok


def test_generation_called_exactly_once():
    """greedy 1회 — 스키마 위반이어도 다시 부르지 않는다."""
    calls = []

    def bad_gen(prompt):
        calls.append(prompt)
        return "이건 JSON 이 아니다"

    out = judge_image("img1", DEFECTS, hit(["C-1"]), [chunk_for("C-1")], bad_gen, TEMPLATE)
    assert len(calls) == 1
    assert not out.parse_ok and out.parse_error == "no_json"
    assert out.verdict == UNDECIDABLE       # 오답 처리


def test_valid_generation_parsed():
    def gen(prompt):
        return '{"verdict": "판정불가", "cited_clauses": ["C-1"], "basis": "t≤10 에서 직경 4mm 이하"}'

    out = judge_image("img1", DEFECTS, hit(["C-1"]), [chunk_for("C-1")], gen, TEMPLATE)
    assert out.parse_ok
    assert out.cited_clauses == ("C-1",)
    assert out.retrieved == ("C-1",)


def test_code_fenced_json_is_extracted_but_not_corrected():
    """추출은 관대하게(코드펜스 벗김), 검증은 엄격하게(enum 위반은 오답)."""
    fenced = '```json\n{"verdict": "판정불가", "cited_clauses": [], "basis": "x"}\n```'
    obj, err = parse_generation(fenced)
    assert err is None and obj["verdict"] == UNDECIDABLE

    wrong_enum = '{"verdict": "pass", "cited_clauses": [], "basis": "x"}'
    obj2, err2 = parse_generation(wrong_enum)
    assert obj2 is None and err2 == "schema_violation"


def test_hallucinated_clause_is_kept_not_filtered():
    """제시 밖 조항 인용을 지우지 않는다 — 무근거 인용률이 재야 할 신호다."""
    def gen(prompt):
        return '{"verdict": "판정불가", "cited_clauses": ["MADE-UP"], "basis": "x"}'

    out = judge_image("img1", DEFECTS, hit(["C-1"]), [chunk_for("C-1")], gen, TEMPLATE)
    assert out.cited_clauses == ("MADE-UP",)


def test_run_judge_aggregates_failure_rate():
    items = [
        ("a", DEFECTS, hit(["C-1"]), [chunk_for("C-1")]),
        ("b", DEFECTS, hit(["C-1"]), [chunk_for("C-1")]),
        ("c", DEFECTS, no_hit(), []),
    ]
    responses = iter(['{"verdict": "판정불가", "cited_clauses": [], "basis": "x"}', "쓰레기"])

    report = run_judge(items, lambda p: next(responses))
    assert report.as_dict()["n_generated"] == 2
    assert report.as_dict()["n_no_hit"] == 1
    assert report.failure_counts == {"no_json": 1}
    assert report.failure_rate == pytest.approx(1 / 3)


def test_prompt_hash_is_stable_and_reported():
    _, d1 = load_prompt_template()
    _, d2 = load_prompt_template()
    assert d1 == d2 and len(d1) == 64
    report = run_judge([], lambda p: "")
    assert report.prompt_sha256 == d1
