"""테스트 픽스처 — limits.csv 픽스처 생성 헬퍼.

스펙 §4-8 테스트가 쓰는 최소 유효 행을 기준으로, 컬럼 override 로 위반을 주입한다.
"""

from __future__ import annotations

import csv
import datetime as _dt
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from corpus.rules.limits_loader import EXPECTED_COLUMNS, load_limits
from corpus.rules.schema import LimitRow, LimitsTable

# ---------------------------------------------------------------------------
# CSV 픽스처 (로더 테스트용 — 문자열 셀)
# ---------------------------------------------------------------------------

BASE_CSV_ROW: dict[str, str] = {
    "rule_id": "KR-3.2.1-001",
    "canonical": "true",
    "scope": "active",
    "defect_code": "2011",
    "material": "ST",
    "inspection_method": "RT",
    "thickness_min": "8",
    "thickness_max": "25",
    "quality_scheme": "iso5817",
    "quality_level": "C",
    "limit_type": "직경",
    "limit_rule": "const",
    "limit_value": "3",
    "limit_factor": "",
    "limit_cap": "",
    "ratio_basis": "",
    "limit_op": "le",
    "unit": "mm",
    "clause_id": "KR-3.2.1",
    "source_doc": "KR-RULES-P2",
    "source_page": "14",
    "source_table_id": "T1",
    "source_row_label": "r1",
    "limit_expr": "≤ 3 mm",
    "note": "",
    "restated_from": "",
    "filled_h": "false",
    "filled_v": "false",
    "extraction_path": "D",
    "verified_by": "tester",
    "verified_date": "2026-08-17",
}


def csv_row(**over: str) -> dict[str, str]:
    """기본 유효 행에 컬럼 override 를 얹는다.

    `inspection_method` 를 명시하지 않으면 `rule_id` 에서 유추한다 — rule_id 에 `VT` 가
    들어 있으면 VT, 그 밖에는 RT. 검사 방법 축(결정 #13) 이전에 작성된 호출부가
    `csv_row(rule_id="KR-VT-001", ...)` 만으로 RT/VT 병존 픽스처를 만들 수 있게 하는
    **테스트 편의**다. 실제 CSV 에는 이런 유추가 없고 컬럼이 필수값이다 — 원문이 특정하지
    않은 조항까지 RT 로 추정해 채우면 축을 만든 이유가 사라진다.
    """
    merged = dict(BASE_CSV_ROW)
    merged.update(over)
    if "inspection_method" not in over:
        merged["inspection_method"] = "VT" if "VT" in merged["rule_id"] else "RT"
    return merged


@pytest.fixture
def write_limits(tmp_path: Path):
    """행 dict 목록을 limits.csv 로 쓴다. drop_columns 로 헤더 누락을 주입할 수 있다."""

    def _write(rows: list[dict[str, str]], name: str = "limits.csv",
               drop_columns: tuple[str, ...] = ()) -> Path:
        cols = [c for c in EXPECTED_COLUMNS if c not in drop_columns]
        path = tmp_path / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        return path

    return _write


@pytest.fixture
def sources_path(tmp_path: Path) -> Path:
    """문서 등록부 픽스처 — OPEN 2건 + PAID 1건 (V5 게이트 검증용)."""
    data = {
        "documents": [
            {"doc_id": "KR-RULES-P2", "license_class": "OPEN",
             "allowed_use": ["table_extract", "corpus_text", "rag_index"], "sha256": "f" * 64},
            {"doc_id": "IACS47", "license_class": "OPEN",
             "allowed_use": ["table_extract"], "sha256": "e" * 64},
            {"doc_id": "PAID-DOC", "license_class": "PAID", "allowed_use": []},
        ]
    }
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture
def registry() -> list[str]:
    """색인 등재 조항 목록 픽스처 (V8)."""
    return ["KR-3.2.1", "KR-3.2.2", "IACS47-3.2.1"]


@pytest.fixture
def load(write_limits, sources_path, registry):
    """픽스처 행들로 CSV 를 쓰고 로드까지 한 번에. 기본은 pilot=True + 등록부/조항 목록 제공."""

    def _load(rows: list[dict[str, str]], *, pilot: bool = True,
              sources=..., reg=..., drop_columns: tuple[str, ...] = ()):
        path = write_limits(rows, drop_columns=drop_columns)
        return load_limits(
            path,
            sources_yaml=sources_path if sources is ... else sources,
            clause_registry=registry if reg is ... else reg,
            pilot=pilot,
        )

    return _load


# ---------------------------------------------------------------------------
# 직접 구성 헬퍼 (limit_eval 테스트용 — 타입 있는 값)
# ---------------------------------------------------------------------------

_BASE_ROW_KWARGS: dict = {
    "rule_id": "T-001",
    "canonical": True,
    "scope": "active",
    "defect_code": "2011",
    "material": "ST",
    "inspection_method": "RT",
    "thickness_min": Decimal("8"),
    "thickness_max": Decimal("25"),
    "quality_scheme": "iso5817",
    "quality_level": "C",
    "limit_type": "직경",
    "limit_rule": "const",
    "limit_value": Decimal("3"),
    "limit_factor": None,
    "limit_cap": None,
    "ratio_basis": None,
    "limit_op": "le",
    "unit": "mm",
    "clause_id": "KR-3.2.1",
    "source_doc": "KR-RULES-P2",
    "source_page": 14,
    "source_table_id": "T1",
    "source_row_label": "r1",
    "limit_expr": "≤ 3 mm",
    "note": None,
    "restated_from": None,
    "filled_h": False,
    "filled_v": False,
    "extraction_path": "D",
    "verified_by": "tester",
    "verified_date": _dt.date(2026, 8, 17),
}


def make_row(**over) -> LimitRow:
    merged = dict(_BASE_ROW_KWARGS)
    merged.update(over)
    return LimitRow(**merged)


def make_table(*rows: LimitRow) -> LimitsTable:
    """테스트 전용 테이블 (공식 로더 경유 아님 — 픽스처 예외)."""
    return LimitsTable(rows=tuple(rows), sha256="0" * 64, source_path="<fixture>", pilot=True)
