"""조항 축 (나) — 주 조항 하나 + 보조 조항의 분해(07 4판 §13 · §13-5 · §14-3, 등록 칸 `clause_axis.multi_clause`).

기대는 구현 전에 고정한 사례 파일(`evaluation/vectors/clause_axis_roles.v1.json`)에서만 읽는다. 합성 코드(갑 · 을 · 병 · 정)와 같은 꼴의
신판 코드(08 §7 시험 5 의 표)가 한 기대를 쓴다 — 생성 쪽 시험 5 가 같은 입력을 쓴다. 신판 역할은 생성 쪽 `derive_clause_roles` 의
대역이다. 실제 신판 표와의 결합 시험은 생성 쪽 병합 뒤에 따로 둔다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from corpus.rules import limits_loader
from evaluation import clause_axis as CA
from evaluation.gold import GoldEntry
from evaluation.metrics.clause import GoldPair

REPO = Path(__file__).resolve().parents[1]
CASES_REL = "evaluation/vectors/clause_axis_roles.v1.json"
CASES_SHA256 = "78d297adf7f61f2c1e2eca25a51d8c6b86acf018a098c3bfe10a0786a4a12a8b"
"""구현 전 고정 커밋(91e83979)의 바이트 — 기대를 바꾸려면 새 판의 파일을 내고 기대 변경 표에 적는다."""
CASES = json.loads((REPO / CASES_REL).read_bytes().decode("utf-8"))
LIMITS_V1 = REPO / "corpus/rules/limits_v0_pilot.csv"


def roles_of(world: str) -> dict:
    """사례 파일의 역할 목록 → 생성 쪽 출력의 꼴(07e §5) → `roles_from_derived`."""
    derived = {(r["material"], r["code"]): {"main": r["main"], "auxiliary": [tuple(a) for a in r["auxiliary"]],
                                            "defect_class": None, "conditions": ()}
               for r in CASES["worlds"][world]["roles"]}
    return CA.roles_from_derived(derived)


def counts(world: str, images: list[dict]) -> dict:
    roles = roles_of(world)
    material_of = {im["image"]: im["material"] for im in images}
    gt = {im["image"]: im["gt_codes"] for im in images}
    cited = {im["image"]: im["cited"] for im in images}
    pairs, skipped = CA.gold_pairs_from_roles(roles, material_of, gt)
    aux = CA.auxiliary_only_by_image(roles, material_of, gt)
    rep = CA.clause_axis_report(cited, pairs, CASES["worlds"][world]["index"], auxiliary_only=aux)
    return {"n_gold_pairs": rep["citation_match_rate"]["denominator"], "match_hits": rep["citation_match_rate"]["numerator"],
            "skipped": skipped, "n_citations": rep["n_citations"], "correct": rep["citation_precision"]["numerator"],
            "nonexistent": rep["ungrounded_rate"]["nonexistent"], "irrelevant": rep["ungrounded_rate"]["irrelevant"],
            "auxiliary_only": rep["ungrounded_rate"]["auxiliary_only"]}


CELLS = [(c, w) for c in CASES["cases"] for w in ("synthetic", "edition2") if c["images"][w] is not None]


def test_사례_파일은_구현_전에_고정한_바이트다() -> None:
    assert hashlib.sha256((REPO / CASES_REL).read_bytes()).hexdigest() == CASES_SHA256
    assert CASES["kind"] == "clause_axis_roles_cases" and CASES["version"] == 1
    assert sorted(CASES["count_keys"]) == sorted(CASES["cases"][0]["expect"])


@pytest.mark.parametrize(("case", "world"), CELLS, ids=[f"{c['id']}-{w}" for c, w in CELLS])
def test_이미지_단위_규칙의_수가_사례의_기대와_같다(case: dict, world: str) -> None:
    assert counts(world, case["images"][world]) == case["expect"]


@pytest.mark.parametrize(("case", "world"), CELLS, ids=[f"{c['id']}-{w}" for c, w in CELLS])
def test_가중이_모두_1_이면_보조_전용의_가중_합이_무가중_수와_같다(case: dict, world: str) -> None:
    images = case["images"][world]
    roles = roles_of(world)
    material_of = {im["image"]: im["material"] for im in images}
    gt = {im["image"]: im["gt_codes"] for im in images}
    cited = {im["image"]: im["cited"] for im in images}
    pairs, _ = CA.gold_pairs_from_roles(roles, material_of, gt)
    aux = CA.auxiliary_only_by_image(roles, material_of, gt)
    got = CA.clause_axis_weighted(cited, pairs, CASES["worlds"][world]["index"], {im: 1.0 for im in material_of},
                                  auxiliary_only=aux)
    assert got["ungrounded_rate"]["auxiliary_only"] == float(case["expect"]["auxiliary_only"])


def test_주_없음_보조_있음은_신판_표에서_나지_않아도_합성으로_남는다() -> None:
    case = next(c for c in CASES["cases"] if c["id"] == "rule3a_no_primary_with_auxiliary")
    assert case["images"]["edition2"] is None and case["images"]["synthetic"]
    assert "합성 시험으로 유지" in case["edition2_absent"]


def test_보조_전용은_다른_코드의_주인_조항을_빼고_낸다() -> None:
    roles = roles_of("edition2")
    got = CA.auxiliary_only_by_image(roles, {"a": "ST", "b": "ST", "c": "AL"},
                                     {"a": ["2011", "301"], "b": ["2011"], "c": ["2011"]})
    assert got == {"a": frozenset(), "b": frozenset({"KRA27-T16"}), "c": frozenset()}


def test_보조_전용이_주_합집합과_겹치면_받지_않는다() -> None:
    with pytest.raises(ValueError, match="보조 전용 집합"):
        CA.clause_axis_report({"a": ["P"]}, (GoldPair("a", "갑", "P"),), {"P"}, auxiliary_only={"a": {"P"}})


def _entry(material: str, code: str, clause: str) -> GoldEntry:
    return GoldEntry(defect_code=code, material=material, inspection_method="RT", thickness_min=None, thickness_max=None,
                     quality_scheme="none", quality_level="", limit_type="length", clause_id=clause, rule_id=f"{clause}-1")


def test_한_코드에_주_조항이_둘이면_멈춘다() -> None:
    stop = next(c for c in CASES["stop_cases"] if c["id"] == "two_primary_stop")
    assert stop["expect"] == {"error": "multiple_main"}
    entries = [_entry(e["material"], e["code"], e["clause"]) for e in stop["entries"]]
    material = {img: v["material"] for img, v in stop["gt"].items()}
    gt = {img: v["gt_codes"] for img, v in stop["gt"].items()}
    with pytest.raises(CA.MultipleMainClauses):
        CA.roles_from_entries(entries, materials={"ST"})
    with pytest.raises(CA.MultipleMainClauses):                       # 예전의 '복수 조항' 건너뜀이 멈춤으로
        CA.gold_pairs_without_thickness(entries, material, gt)
    with pytest.raises(CA.MultipleMainClauses):                       # 대조선도 같은 식
        CA.citations_from_codes(entries, material, gt)


@pytest.mark.parametrize("stop", [c for c in CASES["stop_cases"] if "roles" in c], ids=lambda c: c["id"])
def test_역할의_꼴이_틀리면_받지_않는다(stop: dict) -> None:
    assert stop["expect"] == {"error": "roles_form"}
    derived = {(r["material"], r["code"]): {"main": r["main"], "auxiliary": [tuple(a) for a in r["auxiliary"]],
                                            "defect_class": None, "conditions": ()} for r in stop["roles"]}
    with pytest.raises(CA.RolesFormError):
        CA.roles_from_derived(derived)


def test_역할의_칸이_빠지거나_열쇠가_틀리면_받지_않는다() -> None:
    with pytest.raises(CA.RolesFormError):
        CA.roles_from_derived({("ST", "2011"): {"main": "KRA27-T15", "auxiliary": []}})
    with pytest.raises(CA.RolesFormError):
        CA.roles_from_derived({"ST|2011": {"main": "KRA27-T15", "auxiliary": [], "defect_class": None, "conditions": ()}})


@pytest.mark.parametrize("world", ["synthetic", "edition2"])
def test_대조선은_주_조항만_인용한다(world: str) -> None:
    case = CASES["baseline_case"]
    assert CA.citations_from_roles(roles_of(world), case["material_of"], case["codes_by_image"][world]) == \
        case["expect"][world]


# ── 1 판 회귀 — 조건 칸이 없는 표에서 (나)는 지금 값과 같다 ────────────────────────────────────

@pytest.fixture(scope="module")
def v1_entries():
    assert hashlib.sha256(LIMITS_V1.read_bytes()).hexdigest() == "334c78811677accb86a2ec14dad24716e16add3b6a14b47abe33ca6fe9e0cccb"
    return CA.entries_from_limits(limits_loader.load_limits(LIMITS_V1, pilot=True))


def test_1_판_역할은_보조가_없고_주는_1_판_조항이다(v1_entries) -> None:
    roles = CA.roles_from_entries(v1_entries, materials={"ST", "AL"})
    assert roles == {("ST", "100"): CA.ClauseRole("KRA27-3D"), ("ST", "2011"): CA.ClauseRole("KRA27-T15"),
                     ("ST", "301"): CA.ClauseRole("KRA27-T16"), ("ST", "401"): CA.ClauseRole("KRA27-T16")}


def test_1_판_표에서_역할의_정답_쌍과_대조선은_지금_함수와_같다(v1_entries) -> None:
    material = {"a": "ST", "b": "ST", "c": "ST", "d": "ST", "e": "AL", "f": "ST"}
    gt = {"a": ["2011", "301"], "b": ["401"], "c": ["100"], "d": ["402"], "e": ["2011"], "f": []}
    roles = CA.roles_from_entries(v1_entries, materials=set(material.values()))
    assert CA.gold_pairs_from_roles(roles, material, gt) == CA.gold_pairs_without_thickness(v1_entries, material, gt)
    codes = {"a": ["301", "2011", "401"], "e": ["2011"], "d": ["402"]}
    assert CA.citations_from_roles(roles, material, codes) == CA.citations_from_codes(v1_entries, material, codes)
    assert CA.auxiliary_only_by_image(roles, material, gt) == {img: frozenset() for img in gt}


def test_보조_전용을_주지_않으면_보고는_지금_바이트_그대로다(v1_entries) -> None:
    material = {"a": "ST", "b": "AL"}
    gt = {"a": ["2011", "301"], "b": ["2011"]}
    pairs, _ = CA.gold_pairs_without_thickness(v1_entries, material, gt)
    cited = {"a": ["KRA27-T15", "KRA27-T99"], "b": ["KRA27-T15"]}
    index = {"KRA27-3D", "KRA27-S", "KRA27-T15", "KRA27-T16"}
    base = CA.clause_axis_report(cited, pairs, index)
    assert "auxiliary_only" not in base["ungrounded_rate"]
    roles = CA.roles_from_entries(v1_entries, materials=set(material.values()))
    with_aux = CA.clause_axis_report(cited, pairs, index, auxiliary_only=CA.auxiliary_only_by_image(roles, material, gt))
    assert with_aux["ungrounded_rate"].pop("auxiliary_only") == 0 and with_aux == base
    w = CA.clause_axis_weighted(cited, pairs, index, {"a": 2.0, "b": 0.5})
    assert "auxiliary_only" not in w["ungrounded_rate"]
    assert replace(CA.ClauseRole("P"), auxiliary=("Q",)) == CA.ClauseRole("P", ("Q",))
