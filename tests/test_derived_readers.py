"""B 파생물 소비 경로 — 헤더 레코드·주석 줄 처리. 66번 신규.

62번 파생물은 `_meta` 헤더(jsonl)와 `#` 주석 3줄(csv)을 이고 있다. 그대로 먹이면
색인에 빈 조항이 실리거나 CSV 컬럼이 통째로 어긋나는데, 둘 다 **조용히** 일어난다.
"""

from __future__ import annotations

import json

import pytest

from evaluation.gold import entries_from_derived, read_derived_csv
from rag.index import load_chunks

META = {"_meta": {"source_csv": "limits_v0_pilot.csv", "n_chunks": 1, "pilot": True}}
CHUNK = {
    "clause_id": "KRA27-T15", "source_docs": ["KR-RULES-P2"], "defect_codes": ["2011"],
    "inspection_methods": ["RT"], "quality_schemes": ["none"], "quality_levels": ["ALL"],
    "thickness_min": "0.00", "thickness_max": "100.01", "scope": "active",
}


def test_meta_헤더는_청크가_되지_않는다(tmp_path) -> None:
    p = tmp_path / "chunk_meta.jsonl"
    p.write_text(json.dumps(META) + "\n" + json.dumps(CHUNK) + "\n", encoding="utf-8")
    chunks = load_chunks(p)
    assert [c.chunk_id for c in chunks] == ["KRA27-T15"]


def test_meta_헤더가_둘이면_이어붙은_파생물로_보고_멈춘다(tmp_path) -> None:
    p = tmp_path / "chunk_meta.jsonl"
    p.write_text("\n".join([json.dumps(META), json.dumps(CHUNK), json.dumps(META)]) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="_meta 헤더가 2줄"):
        load_chunks(p)


def test_gold_csv_주석줄을_헤더로_읽지_않는다(tmp_path) -> None:
    p = tmp_path / "gold_clauses.csv"
    p.write_text(
        "# 평가 자산(D6) 경고\n# 원천: limits_v0_pilot.csv sha256=abc\n"
        "defect_code,material,inspection_method,thickness_min,thickness_max,"
        "quality_scheme,quality_level,limit_type,clause_id,rule_id\n"
        "2011,ST,RT,0.00,10.01,none,ALL,직경,KRA27-T15,KRA27-T15-01\n",
        encoding="utf-8",
    )
    rows = read_derived_csv(p)
    assert len(rows) == 1
    assert rows[0]["clause_id"] == "KRA27-T15"
    entries = entries_from_derived(rows)
    assert entries[0].inspection_method == "RT"


def test_iso_codes_구분자는_세미콜론이다() -> None:
    """`|` 로 쪼개면 다중 라벨이 `"2011;301"` 이라는 없는 코드 하나로 집계된다.

    같은 매니페스트의 `strata_key` 가 `|` 를 쓰기 때문에 실제로 한 번 틀렸다 —
    동결 평가셋 자명하한이 0.2160 대신 0.2142 로 나왔다(71번 §10).
    """
    from evaluation.eval_set import ISO_SEP, parse_iso_codes

    assert ISO_SEP == ";"
    assert parse_iso_codes("2011;301") == ("2011", "301")
    assert parse_iso_codes("2011") == ("2011",)
    assert parse_iso_codes("") == ()
    assert parse_iso_codes(None) == ()
    # strata_key 의 구분자를 그대로 넣어도 쪼개지지 않는다 — 열을 바꿔 읽으면 티가 난다
    assert parse_iso_codes("AL|__normal__") == ("AL|__normal__",)


def test_크기_없는_결함은_프롬프트에_미상으로_실린다() -> None:
    """`None` 이 그대로 실리면 모델이 'None' 을 근거 문장에 옮긴다(66번 시운전 실측)."""
    from rag.judge import build_prompt

    p = build_prompt("{defects}|{clauses}", [{"iso_code": "2011", "size_px": None}], [])
    assert "크기(px) 미상" in p and "None" not in p


def test_실물_파생물이_소비된다() -> None:
    """실측 경로 — 픽스처가 아니라 커밋된 B 산출물 자체를 한 번 읽는다."""
    chunks = load_chunks("corpus/derived/chunk_meta.jsonl")
    assert {c.chunk_id for c in chunks} == {
        "KRA27-3D", "KRA27-S", "KRA27-T15", "KRA27-T16"
    }
    rows = read_derived_csv("corpus/derived/gold_clauses.csv")
    assert len(rows) == 12
    assert {r["clause_id"] for r in rows} == {
        "KRA27-3D", "KRA27-S", "KRA27-T15", "KRA27-T16"
    }
