"""corpus/derived/ 실체화 회귀 방지.

파생물이 원천과 어긋난 채 남는 것이 최악이라, 다음을 고정한다:
- 재실행이 같은 바이트를 낸다(결정론 — 타임스탬프 없음). **바이트로 비교한다** —
  read_text() 비교는 win32 개행 변환에 눈이 멀어 CRLF 산출물을 통과시켰다 (74번 재검 3)
- 조항 본문이 비어 있지 않고, 원문 표기 필드를 전재하지 않는다 (과제 4 · 규약 2-5)
- 산출물마다 원천 sha256 이 박혀 있고 실제 원천과 일치한다
- 형식 규약: jsonl 첫 줄 _meta 헤더, gold CSV 의 # 주석은 pandas comment="#" 로 읽힌다
- gold 헤더에 평가 자산(D6) 경고가 있다
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from corpus.rules import materialize_derived as M

REPO = Path(__file__).resolve().parents[2]
CHUNK = REPO / "corpus/derived/chunk_meta.jsonl"
TEXTS = REPO / "corpus/derived/clause_texts.json"
GOLD = REPO / "corpus/derived/gold_clauses.csv"
CRLF = bytes([13, 10])


def test_derived_files_exist_and_match_source():
    assert CHUNK.exists() and GOLD.exists(), "실체화 산출물이 없다 — materialize_derived 실행"
    src_sha = M.sha256_file(M.DEFAULT_CSV)

    head = json.loads(CHUNK.read_text(encoding="utf-8").splitlines()[0])
    assert "_meta" in head
    assert head["_meta"]["source_sha256"] == src_sha, "chunk_meta 가 낡았다 — 재파생 필요"

    gold_head = GOLD.read_text(encoding="utf-8").splitlines()[:3]
    assert any(src_sha in l for l in gold_head), "gold 가 낡았다 — 재파생 필요"
    assert any("평가 자산" in l and "학습" in l for l in gold_head), "D6 경고 주석이 없다"

def _rendered():
    from corpus.rules import clause_text as CT
    from corpus.rules import limit_eval, limits_loader

    src = M.DEFAULT_CSV
    table = limits_loader.load_limits(str(src), pilot=True)
    sha = M.sha256_file(src)
    names = M.defect_names()
    return {
        CHUNK: M.render_chunk_meta(limit_eval.derive_chunk_meta(table, names),
                                   src, sha, True),
        TEXTS: M.render_clause_texts(CT.derive_clause_texts(table.rows, names),
                                     src, sha, True),
        GOLD: M.render_gold(limit_eval.derive_gold_clauses(table), src, sha, True),
    }


def test_rerender_is_byte_identical():
    """이름대로 바이트로 본다. read_text() 비교는 CRLF 산출물을 조용히 통과시킨다."""
    for path, content in _rendered().items():
        assert path.read_bytes() == content.encode("utf-8"), f"{path.name} 바이트 불일치"


def test_derived_files_are_lf():
    """.gitattributes 가 eol=lf 인데 산출물이 CRLF 면 재파생 바이트가 커밋본과 갈린다."""
    for path in (CHUNK, TEXTS, GOLD):
        assert CRLF not in path.read_bytes(), f"{path.name} 이 CRLF 다"


def test_every_chunk_carries_clause_text():
    """본문이 비면 dense 정렬이 걸려도 정렬할 재료가 없다 (임베딩 선정 변별 불가)."""
    rows = [json.loads(x) for x in CHUNK.read_text(encoding="utf-8").splitlines()
            if x.strip()]
    data = [r for r in rows if "_meta" not in r]
    assert data
    for c in data:
        assert c.get("text"), f"{c['clause_id']} 본문 없음"
        assert c["clause_id"] in c["text"]
    doc = json.loads(TEXTS.read_text(encoding="utf-8"))
    assert set(doc["clauses"]) == {c["clause_id"] for c in data}
    assert all(doc["clauses"][c["clause_id"]] == c["text"] for c in data)


def test_clause_text_is_restatement_not_verbatim():
    """규약 2-5 — 원문 표기가 실린 필드(limit_expr·source_row_label·note)는 옮기지 않는다."""
    from corpus.rules import limits_loader

    table = limits_loader.load_limits(str(M.DEFAULT_CSV), pilot=True)
    doc = json.loads(TEXTS.read_text(encoding="utf-8"))["clauses"]
    for r in table.rows:
        for field in (r.limit_expr, r.source_row_label, r.note):
            if field and len(str(field)) >= 8:
                assert str(field) not in doc[r.clause_id], (
                    f"{r.rule_id}: 원문 표기 필드가 본문에 전재됐다")


def test_inequality_direction_comes_from_limit_op():
    """`limit_op` 를 읽는다 — 고정 문자열이면 부등식 방향 게이트가 자기참조가 된다 (M9)."""
    from corpus.rules import clause_text as CT
    from corpus.rules import limits_loader
    from corpus.rules.schema import LimitOp

    table = limits_loader.load_limits(str(M.DEFAULT_CSV), pilot=True)
    row = next(r for r in table.rows if r.limit_rule.value == "const")
    assert "이하" in CT.criterion_ko(row)
    assert "미만" in CT.criterion_ko(row.model_copy(update={"limit_op": LimitOp.LT}))


def test_gold_csv_readable_with_comment_convention():
    df = pd.read_csv(GOLD, comment="#", dtype=str)
    assert len(df) == 12
    assert "inspection_method" in df.columns  # 검사축이 키에 있다 (계약 #3 변경)
    assert set(df["inspection_method"]) <= {"RT", "VT", "ALL"}


def test_chunk_meta_rows_parse_and_skip_meta():
    rows = [json.loads(l) for l in CHUNK.read_text(encoding="utf-8").splitlines() if l.strip()]
    data = [r for r in rows if "_meta" not in r]
    assert len(data) == rows[0]["_meta"]["n_chunks"] == 4
    for c in data:
        assert c["inspection_methods"], "메타 필터 축 결손"
        assert c["clause_id"] and c["rule_ids"]
