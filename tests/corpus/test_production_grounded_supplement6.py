"""Independent supplement-6 regression expectations, without reference vectors.

All annotations are synthetic. Labels come from the repository's label map; no
model, external worktree, sealed cases, monkeypatch, or policy-wrapper identity
is involved. Expected acceptance/reasons are stated from the synthetic facts.
"""

from collections import Counter
import hashlib
from pathlib import Path

import pytest

from corpus.production import judge_grounded_v7 as grounded
from data.label_map import load_label_map


REFERENCE_SHA256 = "07ce544f974e95692800314413a1c87f4ad6d741a54cf239404e979713e00962"
LABELS = load_label_map().defect_types


def _name(key="porosity"):
    return LABELS[key].name_ko


def _task(*groups):
    """groups contain (label-map key, explicit nine-zone name, count)."""
    defects, positions = [], []
    for key, zone, count in groups:
        entry = LABELS[key]
        for _ in range(count):
            defects.append({"type": entry.iso_code, "bbox_px": [1, 2, 3, 4]})
            positions.append({"name": entry.name_ko, "where": zone})
    return {
        "kind": "image_report",
        "locked_target": {"defects": defects},
        "evidence": {
            "defect_counts_by_name": dict(Counter(p["name"] for p in positions)),
            "defect_positions": positions,
        },
    }


def _three_task():
    return _task(("porosity", "위-왼쪽", 2), ("porosity", "아래-오른쪽", 1))


def _three_description():
    return f"{_name()} 2개가 위쪽 왼쪽에 있고 {_name()} 1개가 아래쪽 오른쪽에 있다."


def test_transplanted_file_is_the_approved_supplement6_bytes():
    assert hashlib.sha256(Path(grounded.__file__).read_bytes()).hexdigest() == REFERENCE_SHA256


@pytest.mark.parametrize("template", [
    "{name} 3개가 위쪽 왼쪽에 2개, 아래쪽 오른쪽에 1개 표시되어 있다.",
    "{name} 3개 가운데 2개는 위쪽 왼쪽에, 나머지는 아래쪽 오른쪽에 있다.",
    "결함은 모두 3개이다. {base}",
    "세 개의 결함이 표시되어 있다. {base}",
    "{base} 결함은 두 개의 구역에 표시되어 있다.",
    "{base} 주석에는 두 개의 위치가 표시되어 있다.",
])
def test_total_zone_counts_remainder_and_region_counts_are_not_double_counted(template):
    text = template.format(name=_name(), base=_three_description())
    assert grounded.grounded_codes(_three_task(), text) == []


@pytest.mark.parametrize("suffix,reason", [
    ("결함은 모두 4개이다.", "count_mismatch"),
    ("네 개의 결함이 표시되어 있다.", "count_mismatch"),
    ("{name}은 모두 4개이다.", "count_mismatch"),
    ("{name} 3개가 위쪽 왼쪽에 있다.", "zone_count_mismatch"),
    ("{name} 2개가 아래쪽 오른쪽에 있다.", "statement_conflict"),
])
def test_wrong_totals_and_conflicting_zone_counts_are_rejected(suffix, reason):
    text = _three_description() + " " + suffix.format(name=_name())
    assert reason in grounded.grounded_codes(_three_task(), text)


@pytest.mark.parametrize("zone,phrase", [
    ("위-왼쪽", "위쪽 왼쪽 칸"),
    ("가운데-오른쪽", "가운데 오른쪽 칸"),
    ("아래-가운데", "아래쪽 띠의 가운데 칸"),
    ("위-오른쪽", "오른쪽 세로 띠의 위쪽 구역"),
    ("가운데-가운데", "영상 가운데"),
])
def test_zone_followed_by_cell_and_band_wording_does_not_add_a_false_region(zone, phrase):
    task = _task(("porosity", zone, 1))
    assert grounded.grounded_codes(task, f"{phrase}에 {_name()} 1개가 있다.") == []


def test_column_claim_accepts_only_the_column_containing_all_locked_boxes():
    task = _task(("porosity", "위-왼쪽", 1), ("porosity", "아래-왼쪽", 1))
    base = f"{_name()} 1개가 위쪽 왼쪽에 있고 {_name()} 1개가 아래쪽 왼쪽에 있다."
    assert grounded.grounded_codes(task, base + f" {_name()}은 왼쪽 열에 있다.") == []
    assert "band_contradicts_locked" in grounded.grounded_codes(
        task, base + f" {_name()}은 오른쪽 열에 있다.")


def test_height_union_accepts_both_rows_and_rejects_an_incomplete_band():
    task = _task(("porosity", "위-왼쪽", 1), ("porosity", "가운데-오른쪽", 1))
    base = f"{_name()} 1개가 위쪽 왼쪽 칸에 있고 {_name()} 1개가 가운데 오른쪽 칸에 있다."
    assert grounded.grounded_codes(task, base + f" {_name()}은 위쪽과 가운데 높이에 있다.") == []
    assert "band_contradicts_locked" in grounded.grounded_codes(
        task, base + f" {_name()}은 아래쪽 높이에 있다.")


@pytest.mark.parametrize("suffix", [
    "{name}은 없다.",
    "위쪽 왼쪽에는 {name}이 없다.",
    "위쪽 높이에는 결함이 없다.",
    "오른쪽 열에는 결함이 없다.",
])
def test_absence_claims_cannot_erase_existing_locked_defects(suffix):
    assert "absence_contradicts_locked" in grounded.grounded_codes(
        _three_task(), _three_description() + " " + suffix.format(name=_name()))


def test_absence_of_a_different_label_or_empty_region_is_not_a_new_defect():
    base = _three_description()
    assert grounded.grounded_codes(_three_task(), base + f" {_name('crack')}은 없다.") == []
    assert grounded.grounded_codes(_three_task(), base + " 가운데 높이에는 결함이 없다.") == []


@pytest.mark.parametrize("text", [
    "표시가 생긴 위치는 주석에 기록되어 있다.",
    "설명의 시작은 주석에 근거한다.",
    "밝은 정도는 주석만으로 확인할 수 없다.",
    "둥근 모양인지는 주석만으로 알 수 없다.",
])
def test_attribute_word_boundaries_and_explicit_unknown_statements_are_allowed(text):
    assert grounded.attribute_codes(_three_task(), text) == []


@pytest.mark.parametrize("text", [
    "밝은 표시가 있다.",
    "둥근 모양이다.",
    "작은 표시가 있다.",
    "긴 표시가 있다.",
])
def test_positive_visual_attributes_not_present_in_annotations_are_rejected(text):
    assert grounded.attribute_codes(_three_task(), text) == ["unannotated_attribute"]


@pytest.mark.parametrize("label_key", sorted(LABELS))
def test_every_label_is_read_from_the_canonical_map(label_key):
    task = _task((label_key, "가운데-가운데", 1))
    assert grounded.grounded_codes(task, f"{_name(label_key)} 1개가 영상 가운데에 있다.") == []
