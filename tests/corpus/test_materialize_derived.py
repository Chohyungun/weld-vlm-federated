"""corpus/derived/ 실체화 회귀 방지.

파생물이 원천과 어긋난 채 남는 것이 최악이라, 다음을 고정한다:
- 재실행이 같은 바이트를 낸다(결정론 — 타임스탬프 없음)
- 산출물마다 원천 sha256 이 박혀 있고 실제 원천과 일치한다
- 형식 규약: jsonl 첫 줄 _meta 헤더, gold CSV 의 # 주석은 pandas comment="#" 로 읽힌다
- gold 헤더에 평가 자산(D6) 경고가 있다
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from corpus.rules import materialize_derived as M

REPO = Path(__file__).resolve().parents[2]
CHUNK = REPO / "corpus/derived/chunk_meta.jsonl"
GOLD = REPO / "corpus/derived/gold_clauses.csv"


def test_derived_files_exist_and_match_source():
    assert CHUNK.exists() and GOLD.exists(), "실체화 산출물이 없다 — materialize_derived 실행"
    src_sha = M.sha256_file(M.DEFAULT_CSV)

    head = json.loads(CHUNK.read_text(encoding="utf-8").splitlines()[0])
    assert "_meta" in head
    assert head["_meta"]["source_sha256"] == src_sha, "chunk_meta 가 낡았다 — 재파생 필요"

    gold_head = GOLD.read_text(encoding="utf-8").splitlines()[:3]
    assert any(src_sha in l for l in gold_head), "gold 가 낡았다 — 재파생 필요"
    assert any("평가 자산" in l and "학습" in l for l in gold_head), "D6 경고 주석이 없다"


def test_rerender_is_byte_identical():
    from corpus.rules import limit_eval, limits_loader

    src = M.DEFAULT_CSV
    table = limits_loader.load_limits(str(src), pilot=True)
    sha = M.sha256_file(src)
    assert M.render_chunk_meta(limit_eval.derive_chunk_meta(table), src, sha, True) == \
        CHUNK.read_text(encoding="utf-8")
    assert M.render_gold(limit_eval.derive_gold_clauses(table), src, sha, True) == \
        GOLD.read_text(encoding="utf-8")


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
