"""계약 #1 테스트. 스펙 §7-1."""

from __future__ import annotations

import ast
import copy
import re
from pathlib import Path

import pytest
import yaml

from data.label_map import (
    DEFAULT_LABEL_MAP,
    LabelMap,
    LabelMapError,
    UnmappedLabelError,
    load_label_map,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
#: 파이프라인 코드. 라벨 문자열이 여기 들어가면 안 된다.
PIPELINE_DIRS = ("data", "corpus", "detection", "vlm", "fl", "rag", "evaluation", "tracking")


@pytest.fixture(scope="module")
def lm() -> LabelMap:
    return load_label_map()


@pytest.fixture()
def raw() -> dict:
    return yaml.safe_load(DEFAULT_LABEL_MAP.read_text(encoding="utf-8"))


def _build(raw: dict) -> LabelMap:
    return LabelMap(raw, DEFAULT_LABEL_MAP)


def test_l2_keys_unique_and_ascii(lm: LabelMap) -> None:
    for key in lm.defect_types:
        assert key.isascii() and key == key.lower()
        assert re.fullmatch(r"[a-z][a-z0-9_]*", key), key


def test_train_class_id_dense_and_unique(lm: LabelMap) -> None:
    ids = sorted(dt.train_class_id for dt in lm.defect_types.values())
    assert ids == list(range(len(lm.defect_types)))


def test_normal_not_in_defect_types(lm: LabelMap) -> None:
    """정상은 결함 유형이 아니다 — L1/L2 계층 분리 회귀 방지."""
    assert "normal" not in lm.defect_types
    for level in lm.verdict_levels:
        assert level not in lm.defect_types


def test_source_mapping_targets_exist(lm: LabelMap) -> None:
    for spec in lm.sources.values():
        for target in spec.mapping.values():
            assert target in lm.defect_types


def test_eval_space_subset_of_l2(lm: LabelMap) -> None:
    for space in lm.eval_spaces.values():
        assert set(space.defect_types) <= set(lm.defect_types)
        # 모든 L2 키의 포함/제외가 명시돼 있어야 한다
        assert set(space.defect_types) | set(space.excluded) == set(lm.defect_types)


def test_iso_code_alt_no_collision(lm: LabelMap) -> None:
    seen: dict[str, str] = {}
    for key, dt in lm.defect_types.items():
        for code in (dt.iso_code, *dt.iso_code_alt):
            assert seen.setdefault(code, key) == key


def test_riawelc_lp_excluded_from_main_rt(lm: LabelMap) -> None:
    """열린질문 Q5 — 용입불량은 L2 에 정의하되 주 실험 라벨 공간에서 제외."""
    assert "lack_of_penetration" in lm.defect_types
    assert "lack_of_penetration" not in lm.eval_spaces["main_rt"].defect_types
    assert "lack_of_penetration" in lm.sources["riawelc"].mapping.values()


def test_unseen_space_is_common_three(lm: LabelMap) -> None:
    space = lm.eval_spaces["unseen_riawelc_common3"]
    assert set(space.defect_types) == {"crack", "porosity"}   # + normal = 3


def test_mapping_and_lookup(lm: LabelMap) -> None:
    assert lm.to_defect_type("aihub71761", "기공") == "porosity"
    # RIAWELC 실물 폴더명은 Difetto{1,2,4} + NoDifetto (2026-08-21 실물 검증).
    # 번호 순서대로 짝지으면 균열↔용입불량이 뒤바뀐다 — 그 회귀를 여기서 막는다.
    assert lm.to_defect_type("riawelc", "Difetto2") == "porosity"
    assert lm.to_defect_type("riawelc", "Difetto1") == "crack"
    assert lm.to_defect_type("riawelc", "Difetto4") == "lack_of_penetration"
    assert lm.to_defect_type("aihub71761", "정상") is None
    assert lm.is_normal_label("riawelc", "NoDifetto")
    assert lm.iso_code("porosity") == "2011"


def test_iso_codes_accessor(lm: LabelMap) -> None:
    """계약 소비 트랙(B의 limits.csv V1)이 참조하는 코드 집합 접근자."""
    with_alt = lm.iso_codes()
    without = lm.iso_codes(include_alt=False)
    assert without == {"100", "2011", "301", "401", "402"}
    assert with_alt == without | {"2012"}
    assert isinstance(with_alt, frozenset)


def test_defect_type_for_code_resolves_alias(lm: LabelMap) -> None:
    assert lm.defect_type_for_code("2011") == "porosity"
    assert lm.defect_type_for_code("2012") == "porosity"   # 별칭도 대표 유형으로
    assert lm.defect_type_for_code("402") == "lack_of_penetration"
    with pytest.raises(LabelMapError, match="out_of_scope"):
        lm.defect_type_for_code("5011")                    # 언더컷 — 사상표 밖


def test_unmapped_label_fails_loudly(lm: LabelMap) -> None:
    with pytest.raises(UnmappedLabelError):
        lm.to_defect_type("aihub71761", "언더컷")
    # 교체 전 값(CR/PO/LP/ND)은 이제 없는 폴더다. 되살아나면 여기서 걸린다.
    with pytest.raises(UnmappedLabelError):
        lm.to_defect_type("riawelc", "PO")


def test_riawelc_difetto3_absent(lm: LabelMap) -> None:
    """Difetto3 은 원본에 없는 결번이다 — 누락이 아니라 원저자가 안 쓴 번호다."""
    assert "Difetto3" not in lm.sources["riawelc"].mapping
    with pytest.raises(UnmappedLabelError):
        lm.to_defect_type("riawelc", "Difetto3")


def test_dense_ids_preserve_relative_order(lm: LabelMap) -> None:
    dense = lm.dense_ids("main_rt")
    assert sorted(dense.values()) == list(range(len(dense)))
    by_orig = sorted(dense, key=lambda k: lm.train_class_id(k))
    assert [dense[k] for k in by_orig] == list(range(len(dense)))


# ---- 위반 케이스: 검증기가 실제로 잡는지 -------------------------------------------


def test_rejects_normal_in_defect_types(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["defect_types"]["normal"] = {
        "iso_code": "000", "iso_code_alt": [], "name_ko": "정상",
        "name_en": "Normal", "train_class_id": 5,
    }
    with pytest.raises(LabelMapError, match="계층 분리"):
        _build(bad)


def test_rejects_sparse_train_class_id(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["defect_types"]["crack"]["train_class_id"] = 9
    with pytest.raises(LabelMapError, match="조밀"):
        _build(bad)


def test_rejects_iso_code_collision(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["defect_types"]["crack"]["iso_code_alt"] = ["2011"]
    with pytest.raises(LabelMapError, match="중복 배정"):
        _build(bad)


def test_rejects_mapping_to_unknown_type(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["sources"]["riawelc"]["mapping"]["XX"] = "undercut"
    with pytest.raises(LabelMapError, match="L2 에 없다"):
        _build(bad)


def test_rejects_incomplete_eval_space(raw: dict) -> None:
    bad = copy.deepcopy(raw)
    bad["eval_spaces"]["main_rt"]["excluded"] = []
    with pytest.raises(LabelMapError, match="미결정"):
        _build(bad)


def test_no_hardcoded_label_strings() -> None:
    """파이프라인 코드에 한글 라벨·ISO 코드 리터럴이 없어야 한다 (불변조건 1-8).

    사상표 자체와 테스트·스크립트는 예외다. 스크립트는 도구이고 테스트는 픽스처다.
    """
    lm = load_label_map()
    forbidden = {dt.name_ko for dt in lm.defect_types.values()}
    forbidden |= {dt.iso_code for dt in lm.defect_types.values()}
    forbidden |= {c for dt in lm.defect_types.values() for c in dt.iso_code_alt}

    offenders: list[str] = []
    for d in PIPELINE_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            # 주석·독스트링은 설명이므로 제외한다. 실제 코드가 쓰는 문자열 상수만 본다.
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and node.value in forbidden
                ):
                    offenders.append(
                        f"{py.relative_to(REPO_ROOT)}:{node.lineno}: {node.value!r}"
                    )
    assert not offenders, f"라벨 하드코딩: {offenders}"
