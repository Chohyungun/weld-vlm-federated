"""등록부 검증기 테스트 — 통과 1건 + 위반마다 1건.

"ERROR 0" 은 검증기가 실제로 잡을 때만 의미가 있다. 고의로 깨뜨린 사본으로
각 관문이 발동하는지 확인한다.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate_registry import Report, check_schema, check_semantics

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "data" / "registry" / "datasets" / "riawelc.json"


@pytest.fixture(scope="module")
def base() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _run(doc: dict, name: str = "riawelc") -> Report:
    rep = Report()
    check_schema(doc, name, rep)
    check_semantics(doc, REGISTRY.with_name(f"{name}.json"), rep)
    return rep


def test_committed_registry_is_clean(base: dict) -> None:
    rep = _run(base)
    assert not rep.errors, "\n".join(rep.errors)


def test_schema_layer_actually_runs(base: dict) -> None:
    """jsonschema 가 설치돼 있어야 스키마 층이 돈다 — 미설치면 조용히 반만 검사한다."""
    assert check_schema(base, "riawelc", Report()) is True


def test_filename_mismatch(base: dict) -> None:
    rep = _run(base, name="lohi_weld")
    assert any("파일명" in e for e in rep.errors)


def test_typo_key_rejected(base: dict) -> None:
    d = copy.deepcopy(base)
    d["pipeline"]["eligable"] = True          # 오타
    assert any("Additional properties" in e or "eligable" in e for e in _run(d).errors)


def test_computed_value_rejected(base: dict) -> None:
    d = copy.deepcopy(base)
    d["fields_declared"]["material"]["group_id"] = "grp_x"
    assert any("계산 결과" in e or "Additional" in e for e in _run(d).errors)


def test_archive_sha256_is_allowed(base: dict) -> None:
    """아카이브 체크섬은 정당한 기록이다 — 이미지별 sha256 과 구별해야 한다."""
    rep = _run(base)
    assert not any("archives" in e and "계산 결과" in e for e in rep.errors)


def test_thickness_value_rejected(base: dict) -> None:
    """게이트 #7 결정 G — 등록부에 두께 값을 적으면 실측 동기가 사라진다."""
    d = copy.deepcopy(base)
    d["fields_declared"]["thickness_mm"] = {
        "status": "constant", "constant_value": 12, "evidence": "가정",
    }
    assert any("게이트 #7" in e for e in _run(d).errors)


def test_coverage_must_be_null(base: dict) -> None:
    d = copy.deepcopy(base)
    d["fields_declared"]["material"]["coverage"] = 24407
    assert any("coverage" in e for e in _run(d).errors)


def test_status_requires_companion_fields(base: dict) -> None:
    d = copy.deepcopy(base)
    d["fields_declared"]["px_per_mm"] = {"status": "unverified"}   # how_to_verify·owner·due 없음
    errs = _run(d).errors
    assert any("how_to_verify" in e for e in errs)


def test_iso_code_in_proposed_l2_rejected(base: dict) -> None:
    d = copy.deepcopy(base)
    d["label_vocabulary"]["labels"][0]["proposed_l2"] = "2011"
    assert any("ISO 코드" in e or "pattern" in e.lower() for e in _run(d).errors)


def test_unknown_l2_key_rejected(base: dict) -> None:
    d = copy.deepcopy(base)
    d["label_vocabulary"]["labels"][0]["proposed_l2"] = "undercut"
    assert any("계약 #1 L2" in e for e in _run(d).errors)


def test_normal_label_must_not_have_l2(base: dict) -> None:
    d = copy.deepcopy(base)
    for lab in d["label_vocabulary"]["labels"]:
        if lab["is_normal"]:
            lab["proposed_l2"] = "crack"
    assert any("정상인데" in e for e in _run(d).errors)


def test_eligible_without_contract_mapping_rejected(base: dict) -> None:
    """계약 #1 이 실물 라벨을 모르는데 eligible 이면 ingest 가 즉사한다 — 그 전에 잡는다."""
    d = copy.deepcopy(base)
    d["pipeline"]["eligible"] = True
    d["pipeline"]["raw_subdir"] = "riawelc"
    d["pipeline"]["adapter"] = "RiawelcAdapter"
    d["pipeline"]["adapter_state"] = "ready"
    # 계약 #1 이 2026-08-21 에 실물과 맞춰졌으므로, 관문이 살아 있는지 보려면
    # 계약이 모르는 라벨을 넣어야 한다.
    d["label_vocabulary"]["labels"].append(
        {"raw": "Difetto3", "is_normal": False, "proposed_l2": "crack", "count_announced": 1}
    )
    errs = _run(d).errors
    assert any("unmapped_policy" in e for e in errs), errs


def test_eligible_with_matching_contract_passes(base: dict) -> None:
    """실물과 계약이 맞으면 통과해야 한다 — 관문이 항상 막기만 하면 쓸모가 없다."""
    d = copy.deepcopy(base)
    d["pipeline"].update(
        eligible=True, raw_subdir="riawelc", adapter="RiawelcAdapter",
        adapter_state="ready", exclusion_reason=None,
    )
    assert not [e for e in _run(d).errors if "unmapped_policy" in e]


def test_absolute_path_rejected(base: dict) -> None:
    d = copy.deepcopy(base)
    d["acquisition"]["staging_path_hint"] = "G:\\공유 드라이브\\대한산업공학회_추계학술대회\\weld-fl-datasets\\riawelc"
    assert any("절대경로" in e for e in _run(d).errors)


def test_counts_verified_cannot_be_true(base: dict) -> None:
    d = copy.deepcopy(base)
    d["counts_announced"]["verified"] = True
    assert any("verified" in e for e in _run(d).errors)


def test_samples_examples_capped_at_five(base: dict) -> None:
    d = copy.deepcopy(base)
    d["samples"]["examples"] = [{"n": i} for i in range(6)]
    assert any("maxItems" in e or "too long" in e.lower() for e in _run(d).errors)


def test_citation_required_without_citations(base: dict) -> None:
    d = copy.deepcopy(base)
    d["provenance"]["citations"] = []
    assert any("citations" in e for e in _run(d).errors)


# ---- S0 바이트 층 (파싱보다 먼저) ----------------------------------------------------


def _bytes_report(tmp_path: Path, raw: bytes) -> Report:
    from scripts.validate_registry import check_bytes

    f = tmp_path / "riawelc.json"
    f.write_bytes(raw)
    rep = Report()
    check_bytes(f, rep)
    return rep


def test_clean_bytes_pass(tmp_path: Path) -> None:
    assert not _bytes_report(tmp_path, b'{"a": 1}\n').errors


def test_bom_rejected(tmp_path: Path) -> None:
    rep = _bytes_report(tmp_path, b"\xef\xbb\xbf" + b'{"a": 1}\n')
    assert any("BOM" in e for e in rep.errors)


def test_crlf_rejected(tmp_path: Path) -> None:
    rep = _bytes_report(tmp_path, b'{\r\n  "a": 1\r\n}\r\n')
    assert any("CRLF" in e for e in rep.errors)


def test_smart_quote_rejected(tmp_path: Path) -> None:
    rep = _bytes_report(tmp_path, '{"a": “1”}\n'.encode())
    assert any("스마트쿼트" in e for e in rep.errors)


def test_reviewer_must_differ_from_owner(base: dict) -> None:
    d = copy.deepcopy(base)
    d["authoring"]["review_status"] = "reviewed"
    d["authoring"]["reviewer"] = d["authoring"]["owner"]
    assert any("검토자가 같다" in e for e in _run(d).errors)
