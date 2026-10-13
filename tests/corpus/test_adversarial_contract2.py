"""재검토 — 계약·결정론 렌즈 (finding #1·#4·#5·#6 수정 이후).

이 파일이 담는 것은 세 종류다.

1. **수정 확인** — 앞 단계가 막았다고 한 경로가 실제로 막혔는지 실행으로 재확인한다.
2. **잔여 경로의 회귀 가드** — 같은 결함이 다른 표기·다른 계층에 남아 있던 구멍
   (축의 소비 계층 전파, D 파생 함수 부재, 어휘 락 이중 구현, 단위 표기 변형,
   번호 없는 표준명, counts 사후 변조). 한때 `xfail(strict=True)` 로 고정해 뒀다가
   수정과 함께 마커를 걷고 통과 계약으로 승격했다.
3. **한계 고정** — counts.json 하나만으로는 닫을 수 없는 것들(단계 간 재배분, 자기정합
   위조, 값 충돌로 인한 단위 집합 합쳐짐). 무엇이 무엇으로 닫히는지를 테스트로 못 박아
   두어 나중에 "잡히는 줄 알았다"는 오해를 막는다. 앞 둘은 `cross_check_counts` 가 맡는다.
"""

from __future__ import annotations

import csv as _csv
import hashlib
import inspect
import json
from decimal import Decimal
from pathlib import Path

import pytest

from corpus.generate import numeric_lock as NL
from corpus.generate.counts_builder import (
    COUNTS_TOP_KEYS,
    CountsError,
    build_counts,
    cross_check_counts,
    validate_counts,
    write_counts,
)
from corpus.rules import limit_eval as LE
from corpus.rules import skeleton_gen as SG
from corpus.rules.limits_loader import EXPECTED_COLUMNS, LimitsValidationError, load_limits
from corpus.rules.schema import InspectionMethod, LimitsTable
from corpus.validate import stage1_rules as S1
from tests.corpus.conftest import csv_row, make_row, make_table

D = Decimal
REPO = Path(__file__).resolve().parents[2]
PILOT_CSV = REPO / "corpus" / "rules" / "limits_v0_pilot.csv"
PILOT_SOURCES = REPO / "corpus" / "parse" / "sources.yaml"


# ---------------------------------------------------------------------------
# 공용 픽스처 행 · 헬퍼
# ---------------------------------------------------------------------------

ROW_RT = csv_row()  # 2011 ST RT 직경 const 3 [8,25) iso5817:C, KR-3.2.1
ROW_LOF = csv_row(rule_id="KR-3.2.2-001", defect_code="401", limit_type="길이",
                  clause_id="KR-3.2.2")
# 같은 결함코드·같은 두께 구간의 표면(VT) 기준 — 게이트 #13 이 표현 가능하게 만든 형태
ROW_VT = csv_row(rule_id="KR-VT-9", inspection_method="VT", defect_code="2011",
                 limit_type="직경", limit_value="1", clause_id="KR-3.2.2")


def _stripped(skeletons) -> list[dict]:
    """provenance(limits sha 가 정당하게 달라지는 부분)를 뗀 직렬화 출력."""
    out = []
    for s in skeletons:
        d = s.to_json_dict()
        d.pop("provenance")
        out.append(d)
    return out


def _c_record(table: LimitsTable, seed: int = 7):
    """(c) 경로 골격 1건 + 골격에 정합하는 생성문."""
    sk = SG.generate_corpus_skeletons(table, seed=seed, total=6, cap=6)[0]
    rec = dict(sk.to_json_dict())
    rec["text"] = (
        f"관찰: 직경 {rec['size_mm']} mm 음영. 조회: 두께 {rec['thickness_mm']} mm 기준 "
        f"{rec['clause_id']} 허용 {rec['limit_value']} mm 이하. 판정: {rec['verdict']}."
    )
    return rec


_SK_CONST = {
    "defect_code": "2011", "material": "ST", "quality_scheme": "iso5817",
    "quality_level": "C", "limit_type": "직경", "limit_rule": "const",
    "thickness_mm": "12.00", "size_mm": "2.00", "measured_value": None,
    "clause_id": "KR-3.2.1", "limit_value": "3.00", "margin": "1.00",
    "verdict": "합격", "limit_op": "le", "unit": "mm",
    "limit_factor": None, "limit_cap": None, "ratio_basis": None,
}

_SK_RATIO = {
    "defect_code": "2012", "material": "ST", "quality_scheme": "iso5817",
    "quality_level": "C", "limit_type": "비율", "limit_rule": "const",
    "thickness_mm": "12.00", "size_mm": None, "measured_value": "20.00",
    "clause_id": "KR-3.2.1", "limit_value": "25.00", "margin": "5.00",
    "verdict": "합격", "limit_op": "le", "unit": "percent",
    "limit_factor": None, "limit_cap": None, "ratio_basis": None,
}

_GOOD_C_TEXT = (
    "관찰: 직경 2.00 mm의 원형 음영. 분류: 기공 (ISO 6520-1 2011). "
    "조회: 모재 두께 12 mm 품질수준 C → 조항 KR-3.2.1 기준 3 mm 이하. "
    "대조: 2.00 ≤ 3.00. 판정: 합격."
)


# ===========================================================================
# 1. finding #1 — inspection_method: 로더 계층은 실제로 막혔는가
# ===========================================================================


def test_inspection_method_is_a_frozen_contract_column():
    """계약 컬럼 순서에 실재하고, 스키마 enum 이 RT/VT/ALL 3값으로 닫혀 있다."""
    assert "inspection_method" in EXPECTED_COLUMNS
    assert EXPECTED_COLUMNS.index("inspection_method") == EXPECTED_COLUMNS.index("material") + 1
    assert {m.value for m in InspectionMethod} == {"RT", "VT", "ALL"}


@pytest.mark.parametrize(
    "override, drop",
    [
        ({"inspection_method": ""}, ()),        # 공란 = 필수값 위반
        ({"inspection_method": "UT"}, ()),      # enum 밖
        ({}, ("inspection_method",)),           # 컬럼 자체 누락
    ],
)
def test_inspection_method_required_and_enum_closed(load, override, drop):
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(**override)], drop_columns=drop)
    assert "V1" in {v.check for v in ei.value.violations}


def test_rt_vt_coexistence_loads_and_axis_narrows(load):
    """같은 결함코드·같은 두께 구간의 RT·VT 기준 병존이 로드된다 (finding #1 수정 확인).

    축 뷰는 {질의 축, ALL} 로만 좁히고, 질의 값으로 ALL 을 주는 것은 거부한다.
    """
    table = load([ROW_RT, ROW_VT])
    assert {r.rule_id for r in table.rows} == {"KR-3.2.1-001", "KR-VT-9"}
    assert {r.rule_id for r in table.for_inspection("RT").rows} == {"KR-3.2.1-001"}
    assert {r.rule_id for r in table.for_inspection("VT").rows} == {"KR-VT-9"}
    with pytest.raises(ValueError):
        table.for_inspection("ALL")


def test_all_row_answers_both_axes(load):
    """`ALL` 행은 RT·VT 어느 질의에도 응한다 — 축 도입이 무관 조항을 잘라내면 안 된다."""
    row_all = csv_row(rule_id="KR-ALL-1", inspection_method="ALL", defect_code="401",
                      limit_type="길이", clause_id="KR-3.2.2")
    table = load([ROW_RT, row_all])
    for axis in ("RT", "VT"):
        assert "KR-ALL-1" in {r.rule_id for r in table.for_inspection(axis).rows}


def test_v3_v6_group_keys_carry_the_axis(load):
    """축이 그룹 키에 없으면 정당한 RT/VT 병존이 구간 겹침·이중 정본으로 오탐된다.

    같은 조합에서 축만 다른 두 canonical 행이 로드되는 것이 축 반영의 증거다.
    """
    table = load([ROW_RT, ROW_VT])
    assert len(table.active_canonical) == 2
    # 대조군: 축까지 같으면 여전히 거부된다 (검사가 헐거워진 것이 아니다)
    with pytest.raises(LimitsValidationError) as ei:
        load([ROW_RT, csv_row(rule_id="KR-DUP-1", clause_id="KR-3.2.2", limit_value="1")])
    assert {"V3", "V6"} & {v.check for v in ei.value.violations}


def test_skeleton_and_audit_carry_the_axis(load):
    """골격에 축이 실리고, 다른 축의 골격 묶음은 사후 감사에서 떨어진다."""
    table = load([ROW_RT, ROW_VT])
    rt_sks = SG.generate_corpus_skeletons(table, seed=7, total=20, cap=20,
                                          inspection_method="RT")
    vt_sks = SG.generate_corpus_skeletons(table, seed=7, total=20, cap=20,
                                          inspection_method="VT")
    assert {s.inspection_method for s in rt_sks} == {InspectionMethod.RT}
    assert {s.inspection_method for s in vt_sks} == {InspectionMethod.VT}
    assert SG.audit_skeletons(rt_sks, table, inspection_method="RT").ok
    assert not SG.audit_skeletons(vt_sks, table, inspection_method="RT").ok


# ===========================================================================
# 2. finding #1 잔여 — 축이 소비 계층(limit_eval · stage1 · D 파생)에 전파되지 않았다
# ===========================================================================


def test_applicable_row_signature_has_inspection_method():
    """잔여 #1a 수정 확인 — 스펙 §4-2 시그니처
    (table, defect_code, material, inspection_method, scheme, level, t) 복원."""
    params = list(inspect.signature(LE.applicable_row).parameters)
    assert params[:7] == ["table", "defect_code", "material", "inspection_method",
                          "quality_scheme", "quality_level", "t"]


def test_applicable_row_enforces_the_axis_itself():
    """잔여 #1a 수정 확인 — 축 강제가 호출자의 뷰 주입 규율에 달려 있지 않다.

    RT 기준이 직경, VT 기준이 길이로 갈린 표에서 길이 제약을 RT 로 물으면, 뷰를 좁히지
    않아도 후보가 없어 fail-closed 로 떨어진다. 예전에는 예외 없이 VT 행이 선택됐다 —
    축을 만든 이유("잘못된 행을 집어도 형식상 정상으로 보인다")가 이 경로에 남아 있었다.
    """
    rt = make_row(rule_id="RT-2", inspection_method="RT", limit_type="직경",
                  limit_value=D("3"), clause_id="KR-3.2.1")
    vt = make_row(rule_id="VT-2", inspection_method="VT", limit_type="길이",
                  limit_value=D("1"), clause_id="KR-3.2.2")
    table = make_table(rt, vt)
    with pytest.raises(LookupError):
        LE.applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("12"),
                          limit_type="길이")
    got = LE.applicable_row(table, "2011", "ST", "VT", "iso5817", "C", D("12"),
                            limit_type="길이")
    assert got.rule_id == "VT-2" and got.inspection_method is InspectionMethod.VT
    # 뷰를 좁혀 넘겨도 결과는 같다 — 두 겹 방어가 서로 어긋나지 않는다
    with pytest.raises(LookupError):
        LE.applicable_row(table.for_inspection("RT"), "2011", "ST", "RT",
                          "iso5817", "C", D("12"), limit_type="길이")


def test_stage1_accepts_valid_rt_record_against_mixed_table(load):
    """잔여 #1b 수정 확인 — stage1 이 골격에 실린 축으로 행을 조회한다.

    예전에는 축을 좁히지 않은 테이블을 넘기면 정당한 RT/VT 병존이 '복수 행 매칭'이 되어
    정상 RT 레코드가 combo_not_found 로 폐기됐다. 호출자가 뷰를 좁혔는지에 결과가
    달라지면 안 된다.
    """
    table = load([ROW_RT, ROW_VT])
    rec = _c_record(table.for_inspection("RT"))
    assert S1.check_record_c(rec, table.for_inspection("RT")).ok  # 대조군
    assert S1.check_record_c(rec, table).ok
    # 기대 축을 못박으면 다른 축의 레코드는 폐기된다
    assert S1.check_record_c(rec, table, inspection_method="RT").ok
    assert not S1.check_record_c(rec, table, inspection_method="VT").ok


@pytest.mark.parametrize("mutate", ["VT", "완전엉터리", None])
def test_stage1_validates_skeleton_inspection_method(load, mutate):
    """잔여 #1c 수정 확인 — 골격의 축을 위조하거나 지우면 1단계가 잡는다.

    _IDENTITY_KEYS 에 축이 없으면 "VT"·enum 밖 값·키 삭제가 전부 통과했다 — 행 선택이
    조용히 다른 축의 기준으로 넘어가는 경로다.
    """
    table = load([ROW_RT])
    rec = _c_record(table)
    assert S1.check_record_c(rec, table).ok  # 대조군
    tampered = dict(rec)
    if mutate is None:
        tampered.pop("inspection_method")
    else:
        tampered["inspection_method"] = mutate
    assert not S1.check_record_c(tampered, table).ok


@pytest.mark.parametrize(
    "name", ["rows_for_clause", "derive_chunk_meta", "derive_gold_clauses"]
)
def test_d_consumption_contract_functions_exist(name):
    """잔여 #1d 수정 확인 — D 소비 계약 3종이 공유 모듈에 실재한다 (§1-4).

    게이트 #13 변경통지가 derive_chunk_meta 의 inspection_methods[] 추가와
    derive_gold_clauses 키 확장을 D 에 통지했는데, 정작 함수 자체가 없어 통지된 변경을
    반영할 대상이 없었다.
    """
    assert hasattr(LE, name)
    assert name in LE.__all__


def test_derive_chunk_meta_carries_the_axis_and_keeps_excluded_rows(load):
    """chunk_meta: 합집합 집계 + 검사 방법 축 + 두께 반열림(+∞ → null) + excluded 포함."""
    excluded = csv_row(rule_id="KR-EX-1", scope="excluded", defect_code="301",
                       limit_type="길이", clause_id="KR-3.2.2", thickness_max="")
    meta = {m["clause_id"]: m for m in LE.derive_chunk_meta(load([ROW_RT, ROW_VT, excluded]))}
    assert set(meta) == {"KR-3.2.1", "KR-3.2.2"}
    assert meta["KR-3.2.1"]["inspection_methods"] == ["RT"]
    # 같은 조항을 VT 행과 RT excluded 행이 공유한다 — 합집합 집계와 scope 혼재 표기
    assert meta["KR-3.2.2"]["inspection_methods"] == ["RT", "VT"]
    assert meta["KR-3.2.2"]["defect_codes"] == ["2011", "301"]
    assert meta["KR-3.2.2"]["scope"] == "mixed"
    assert meta["KR-3.2.2"]["thickness_max"] is None  # +∞ → null
    assert meta["KR-3.2.1"]["thickness_max"] == "25.00"


def test_derive_gold_clauses_key_includes_the_axis(load):
    """gold_clauses: canonical ∧ active 만, 키에 검사 방법 축 포함 (게이트 #13 통지분)."""
    gold = LE.derive_gold_clauses(load([ROW_RT, ROW_VT]))
    keyed = {(g["defect_code"], g["inspection_method"]): g["clause_id"] for g in gold}
    assert keyed == {("2011", "RT"): "KR-3.2.1", ("2011", "VT"): "KR-3.2.2"}
    assert all("thickness_min" in g and "quality_level" in g for g in gold)


@pytest.mark.parametrize("axis", ["RT", "VT"])
def test_all_axis_row_survives_every_consumption_layer(axis):
    """`ALL` 행(검사 방법 무관 조항)이 생성·감사·1단계를 어느 축으로도 통과한다.

    축을 필수로 만들면서 ALL 행이 조용히 잘려 나가면 무관 조항이 corpus 에서 통째로
    사라진다 — 축 도입이 만들 수 있는 반대 방향의 사고라 경로별로 못 박는다.
    골격에는 행의 값 그대로 "ALL" 이 실리고, 조회 축은 질의자가 정한다.
    """
    table = make_table(make_row(rule_id="R-ALL", inspection_method="ALL",
                                limit_value=D("3")))
    sks = SG.generate_corpus_skeletons(table, seed=1, total=10, cap=10,
                                       inspection_method=axis)
    assert len(sks) == 10
    assert {s.inspection_method for s in sks} == {InspectionMethod.ALL}
    assert SG.audit_skeletons(sks, table, inspection_method=axis).ok

    d = sks[0].to_json_dict()
    rec = dict(d, text=(
        f"관찰: 직경 {d['size_mm']} mm 음영. 조회: 두께 {d['thickness_mm']} mm 기준 "
        f"{d['clause_id']} 허용 {d['limit_value']} mm 이하. 판정: {d['verdict']}."
    ))
    assert S1.check_record_c(rec, table).ok
    assert S1.check_record_c(rec, table, inspection_method=axis).ok


def test_rows_for_clause_narrows_then_applicable_row_reselects(load):
    """§1-4 D 판정 정합성 재계산 경로: 조항으로 좁힌 뒤 축 포함 applicable_row 재사용."""
    table = load([ROW_RT, ROW_VT])
    candidates = LE.rows_for_clause(table, "KR-3.2.2")
    assert [r.rule_id for r in candidates] == ["KR-VT-9"]
    row = LE.applicable_row(candidates, "2011", "ST", "VT", "iso5817", "C", D("12"))
    assert row.rule_id == "KR-VT-9"
    # 같은 후보를 RT 로 물으면 후보가 없다 — 조항만으로 축이 갈리지 않는다는 것이 요점이다
    with pytest.raises(LookupError):
        LE.applicable_row(candidates, "2011", "ST", "RT", "iso5817", "C", D("12"))


# ===========================================================================
# 3. finding #4 — counts.json 사후 변조
# ===========================================================================

_ADOPTED = [
    {"image_id": "i1", "defects": [{"defect_code": "2011"}]},
    {"image_id": "i2", "defects": []},
    {"image_id": "i3", "defects": [{"defect_code": "100"}]},
]
_CLIENT_OF = {"i1": "C1", "i2": "C1", "i3": "C2"}


def _counts(**over):
    kw = dict(
        n_generated=7, discarded={"stage0_numeric_lock": 4},
        limits_sha256="a" * 64, manifest_sha256="b" * 64,
        git_commit="x", date="2026-08-17",
    )
    kw.update(over)
    return build_counts(_ADOPTED, _CLIENT_OF, **kw)


@pytest.mark.parametrize(
    "field, value",
    [("discarded", 999), ("n_generated", 99)],
    ids=["discarded 단독 변조", "n_generated 단독 변조"],
)
def test_counts_single_field_tamper_detected(field, value):
    """finding #4 수정 확인 — n_generated 가 본문에 있어 파일만으로 재검산된다."""
    c = json.loads(json.dumps(_counts()))
    if field == "discarded":
        c["discarded"]["stage2_judge"] = value
    else:
        c["n_generated"] = value
    with pytest.raises(CountsError):
        validate_counts(c, _ADOPTED)


_REPORT = {
    "n_generated": 7,
    "quarantine": 0,
    "stages": {
        "stage0_numeric_lock": {"n_in": 7, "n_pass": 3},
        "stage1_rule": {"n_in": 3, "n_pass": 3},
        "stage2_judge": {"n_in": 3, "n_pass": 3},
        "stage3_expert": {"n_in": 3, "n_pass": 3},
    },
}


def test_counts_cross_check_with_validation_report_passes():
    """정합 3항을 통과한 정상 counts 는 검증 보고서와도 맞는다 (대조군)."""
    cross_check_counts(_counts(), _REPORT)


def test_counts_self_consistent_forgery_needs_the_cross_check():
    """생성량과 폐기량을 함께 맞춰 고치면 정합 3항만으로는 못 잡는다.

    counts.json 안에서만 보면 자기정합이라 통과한다. 잡으려면 validation_report(§6-4)의
    단계별 n_in·n_pass 와 교차 대조해야 하고, 그 대조를 스냅샷 확정 조건에 넣는다.
    """
    c = json.loads(json.dumps(_counts()))
    c["discarded"]["stage2_judge"] = 999
    c["n_generated"] = 999 + 4 + len(_ADOPTED)
    validate_counts(c, _ADOPTED)  # 파일 하나만으로는 예외 없음
    with pytest.raises(CountsError):
        cross_check_counts(c, _REPORT)


def test_counts_stage_redistribution_needs_the_cross_check():
    """합계만 맞추고 단계 간 배분을 바꾸면 합계 검산은 통과한다.

    단계별 폐기 사유 분포는 논문에 싣는 수치라 합계만으로는 방어되지 않는다 — 교차 대조가
    그 자리를 메운다.
    """
    c = json.loads(json.dumps(_counts()))
    c["discarded"]["stage0_numeric_lock"] = 1
    c["discarded"]["stage3_expert"] = 3
    validate_counts(c, _ADOPTED)  # 파일 하나만으로는 예외 없음
    with pytest.raises(CountsError):
        cross_check_counts(c, _REPORT)


def test_counts_duplicate_image_id_rejected():
    """수정 확인 — 채택 레코드의 image_id 유일성 검사.

    중복분은 "합계 = 채택 행수" 를 그대로 만족시키면서 n_k 만 부풀린다 — FedAvg 가중치가
    조용히 틀어지는 형태라 fail-closed 로 막는다.
    """
    dup = _ADOPTED + [_ADOPTED[0]]
    with pytest.raises(CountsError, match="image_id 중복"):
        build_counts(dup, _CLIENT_OF, n_generated=8,
                     discarded={"stage0_numeric_lock": 4},
                     limits_sha256="a" * 64, manifest_sha256="b" * 64,
                     git_commit="x", date="2026-08-17")


def test_counts_schema_diverges_from_spec_7_5():
    """스펙 §7-5 스키마 그대로 만든 counts.json 이 거부된다 (계약 이탈 기록).

    `n_generated` 는 §7-5 스키마에 없는 키이고 최상위 키는 순서까지 엄격 비교된다.
    스키마 변경은 승인 사항이라 승인 기록 확인이 필요하다.
    """
    assert "n_generated" in COUNTS_TOP_KEYS
    spec_shaped = {k: v for k, v in _counts().items() if k != "n_generated"}
    with pytest.raises(CountsError, match="최상위 키 불일치"):
        validate_counts(spec_shaped, _ADOPTED)


def test_write_counts_returned_hash_matches_file_bytes(tmp_path):
    """critical 수정 확인 — 반환 sha256 이 실제 기록된 파일 바이트의 해시다.

    예전에는 write_text 의 플랫폼 개행 변환 때문에 Windows 에서 파일에 CRLF 가 들어가고
    반환값은 LF 텍스트 해시였다. 그 해시를 SNAPSHOT.sha256·논문에 실으면 §8 의 소비처
    해시 대조가 항상 불일치해 C 의 n_k 로드가 통째로 중단된다.
    """
    path = tmp_path / "counts.json"
    returned = write_counts(path, _counts())
    assert returned == hashlib.sha256(path.read_bytes()).hexdigest()
    assert b"\r\n" not in path.read_bytes()  # 플랫폼 무관 바이트


def test_write_counts_roundtrip_still_validates(tmp_path):
    """회귀 — 기록한 파일을 다시 읽어도 키 순서·정합이 유지된다."""
    path = tmp_path / "counts.json"
    write_counts(path, _counts())
    back = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(back.keys()) == COUNTS_TOP_KEYS
    validate_counts(back, _ADOPTED)


# ===========================================================================
# 4. finding #5 — 결함 어휘 락
# ===========================================================================


@pytest.mark.parametrize(
    "term", ["crack", "CRACK", "Crack", "ｃｒａｃｋ", "２０１１"],
    ids=["소문자", "대문자", "원형", "전각영문", "전각ISO코드"],
)
def test_normal_lock_discards_defect_terms_in_any_casing(term):
    """finding #5 수정 확인 — 대소문자·전각 표기 전부 폐기된다."""
    lex = SG.load_defect_lexicon()
    text = f"필름 전반에 {term} 성 지시 없음. 판정: 합격."
    r = NL.check_normal_lock(text, defect_lexicon=lex, thickness_mm=D("12.00"),
                             quality_level="C", expected_verdict="합격")
    assert not r.ok
    assert NL.REASON_DEFECT_HALLUCINATION in r.reasons


def test_normal_lock_control_text_still_passes():
    """회귀 — 결함 어휘가 없는 정상 판정문은 그대로 통과한다."""
    lex = SG.load_defect_lexicon()
    text = "필름 전반에 검출 한계 내 특기할 지시 없음. 가정 두께 12 mm 기준. 판정: 합격."
    r = NL.check_normal_lock(text, defect_lexicon=lex, thickness_mm=D("12.00"),
                             quality_level="C", expected_verdict="합격")
    assert r.ok, r.detail


FULLWIDTH_CRACK = "ｃｒａｃｋ 지시 없음"
FULLWIDTH_CODE = "２０１１ 지시 없음"
NFD_POROSITY = "기공 지시 없음"


@pytest.mark.parametrize(
    "text",
    [FULLWIDTH_CRACK, FULLWIDTH_CODE, NFD_POROSITY],
    ids=["전각영문", "전각ISO코드", "NFD한글"],
)
def test_defect_lexicon_detectors_agree(text):
    """잔여 #5 수정 확인 — 결함 어휘 락의 구현이 하나다.

    두 벌이던 시절에는 발산 축만 대소문자에서 유니코드 정규화로 옮겨 갔다: skeleton_gen
    쪽(공개 API)은 NFKC 정규화가 없어 전각 영문·전각 ISO 코드·NFD 분해 한글을 놓치고
    numeric_lock 쪽은 검출했다. 이제 skeleton_gen 은 재수출만 하므로 같은 함수다.
    """
    lex = SG.load_defect_lexicon()
    assert SG.find_defect_tokens is NL.find_defect_tokens  # 정본은 하나뿐이다
    assert bool(SG.find_defect_tokens(text, lex)) == bool(NL.find_defect_tokens(text, lex))
    assert NL.find_defect_tokens(text, lex), "전각·NFD 표기도 검출돼야 한다"


# ===========================================================================
# 5. finding #6a — 단위 대조
# ===========================================================================


def test_numeric_lock_discards_percent_value_written_as_mm():
    """finding #6a 수정 확인 — 비율 골격의 값을 mm 로 적으면 폐기된다."""
    text = ("관찰: 기공 면적 비율 20.00 mm. 분류: 2012. 조회: 두께 12 mm 기준 "
            "KR-3.2.1 허용 25 mm 이하. 판정: 합격.")
    r = NL.check_numeric_lock(text, _SK_RATIO)
    assert not r.ok and NL.REASON_CHANGED_VALUE in r.reasons


def test_numeric_lock_accepts_correct_percent_notation():
    """회귀 — 올바른 % 표기는 통과한다 (단위 대조가 정상 표본을 죽이지 않는다)."""
    text = ("관찰: 기공 면적 비율 20.00 %. 분류: 2012. 조회: 두께 12 mm 기준 "
            "KR-3.2.1 허용 25 % 이하. 판정: 합격.")
    assert NL.check_numeric_lock(text, _SK_RATIO).ok


@pytest.mark.parametrize(
    "notation", ["20.00 밀리미터", "20.00(mm)", "20.00 미리", "20.00 [mm]"],
    ids=["한글단위", "괄호단위", "구어단위", "대괄호단위"],
)
def test_numeric_lock_unit_notation_variants(notation):
    """잔여 #6a 수정 확인 — 단위 표기 변형으로 대조를 건너뛰지 못한다.

    사전이 {mm, %, percent, 퍼센트} 뿐이던 시절에는 한글 단위명·괄호 표기가 '단위 없음'으로
    읽혀 대조 자체를 지나쳤다 — 같은 단위 혼입이 표기만 바꾸면 통과했다.
    """
    text = (f"관찰: 기공 면적 비율 {notation}. 분류: 2012. 조회: 두께 12 mm 기준 "
            "KR-3.2.1 허용 25 % 이하. 판정: 합격.")
    r = NL.check_numeric_lock(text, _SK_RATIO)
    assert not r.ok and NL.REASON_CHANGED_VALUE in r.reasons


def test_numeric_lock_korean_unit_on_the_right_value_passes():
    """회귀 — 한글 단위명이 골격 단위와 맞으면 통과한다 (사전 확장이 정상 표본을 죽이지 않는다)."""
    text = ("관찰: 직경 2.00 밀리미터의 원형 음영. 조회: 두께 12 mm 기준 KR-3.2.1 허용 "
            "3 mm 이하. 판정: 합격.")
    assert NL.check_numeric_lock(text, _SK_CONST).ok


def test_numeric_lock_unit_sets_merge_on_value_collision():
    """한계 고정 — 허용 집합이 값 기준이라 같은 수치가 두 단위에 걸리면 대조가 무력해진다.

    두께 25 mm 와 허용치 25 % 가 겹치는 골격에서는 허용치를 mm 로 적어도 통과한다.
    슬롯 인식 없이 값만 보는 화이트리스트의 구조적 한계다.
    """
    sk = dict(_SK_RATIO, thickness_mm="25.00", limit_value="25.00", margin="5.00")
    text = ("관찰: 기공 면적 비율 20.00 %. 조회: 두께 25 mm 기준 KR-3.2.1 허용 "
            "25 mm 이하. 분류: 2012. 판정: 합격.")
    assert NL.check_numeric_lock(text, sk).ok


def test_pair_lock_unit_sets_merge_across_defects():
    """한계 고정 — 다결함 페어는 결함별 허용 집합을 합집합으로 합쳐 단위까지 섞인다."""
    d_mm = dict(_SK_CONST, size_mm="2.00", limit_value="3.00", margin="1.00")
    d_pct = dict(_SK_RATIO, measured_value="2.00", limit_value="3.00", margin="1.00",
                 clause_id="KR-3.2.2")
    text = ("관찰: 직경 2.00 mm 음영과 면적 비율 2.00 mm. 조회: 두께 12 mm, "
            "KR-3.2.1 기준 3.00 mm 이하, KR-3.2.2 기준 3.00 % 이하. 여유 1.00 mm. "
            "판정: 합격.")
    assert NL.check_pair_lock(text, [d_mm, d_pct], image_verdict="합격").ok


# ===========================================================================
# 6. finding #6b — 환각 표준 식별자
# ===========================================================================


def test_numeric_lock_control_text_with_whitelisted_standard_passes():
    assert NL.check_numeric_lock(_GOOD_C_TEXT, _SK_CONST).ok


@pytest.mark.parametrize(
    "tail",
    [" 아울러 ISO 9999 및 EN 1290 요건도 충족한다.",
     " 아울러 ISO5817 요건도 충족한다.",
     " 아울러 ISO 6520-1:2007 요건도 충족한다.",
     " 아울러 IACS Rec. 99 요건도 충족한다."],
    ids=["다른 ISO·EN", "공백 없는 표기", "판번호 덧붙임", "다른 IACS 권고"],
)
def test_numeric_lock_discards_hallucinated_standard_ids(tail):
    """finding #6b 수정 확인 — 골격에서 유도되지 않는 표준 식별자는 폐기된다."""
    r = NL.check_numeric_lock(_GOOD_C_TEXT + tail, _SK_CONST)
    assert not r.ok and NL.REASON_EXTRA_VALUE in r.reasons


def test_numeric_lock_keeps_defect_code_outside_the_standard_mask(load):
    """회귀 — 표준 식별자 마스킹이 뒤따르는 결함 코드까지 먹어 치우지 않는다.

    '(ISO 6520-1 2011)' 에서 2011 은 수치 스캔에 남아 필수 출현 검사를 통과해야 한다.
    """
    f4 = dict(_SK_CONST, limit_type="불허", limit_rule="none_permitted",
              limit_value=None, margin=None, unit=None, defect_code="100",
              verdict="불합격", size_mm="2.00")
    text = ("관찰: 선상 지시 2.00 mm. 분류: ISO 6520-1 100. "
            "조회: 조항 KR-3.2.1 은 전 수준 불허. 두께 12.00 mm. 판정: 불합격.")
    assert NL.check_numeric_lock(text, f4).ok


@pytest.mark.parametrize(
    "tail", [" 아울러 AWS 및 ASME 요건도 충족한다.", " 아울러 일본공업규격 요건도 충족한다.",
             " 아울러 JIS 요건도 충족한다."],
    ids=["숫자 없는 표준명", "한글 표준명", "약어 표준명"],
)
def test_numeric_lock_discards_numberless_standard_names(tail):
    """잔여 #6b 수정 확인 — 번호 없는 발행기관 명칭도 화이트리스트 대조 대상이다.

    식별자 정규식이 숫자를 요구해 "AWS 및 ASME 요건도 충족한다" 류가 그대로 통과했다 —
    무근거 인용의 한 형태가 남아 있었다. 한글 표기는 약어로 정규화해 같은 대조를 받는다.
    """
    r = NL.check_numeric_lock(_GOOD_C_TEXT + tail, _SK_CONST)
    assert not r.ok and NL.REASON_EXTRA_VALUE in r.reasons


def test_numeric_lock_standard_body_named_in_korean_is_allowed_when_grounded():
    """회귀 — 골격이 지시한 발행기관은 한글 표기로도 통과한다.

    발행기관 명칭을 무조건 막는 것이 아니라 "골격에 근거가 있는가" 를 본다. 근거가 있으면
    표기(약어/한글)와 무관하게 같은 판단이어야 한다.
    """
    sk = dict(_SK_CONST, source_doc="KS")
    text = _GOOD_C_TEXT + " 한국산업표준 전사분 근거."
    assert NL.check_numeric_lock(text, sk).ok
    # 골격이 지시하지 않은 기관은 한글 표기여도 막힌다
    assert not NL.check_numeric_lock(text.replace("한국산업표준", "일본공업규격"), sk).ok
    # 골격 지시가 없으면 한글 표기도 막힌다 (근거 유무가 기준이다)
    assert not NL.check_numeric_lock(text, _SK_CONST).ok


# ===========================================================================
# 7. G0 V0~V11 항목별 실재 대조 — 문자열 검색이 아니라 행동으로 본다
# ===========================================================================


def _checks_of(load, rows, **kw) -> set[str]:
    with pytest.raises(LimitsValidationError) as ei:
        load(rows, **kw)
    return {v.check for v in ei.value.violations}


@pytest.mark.parametrize(
    "check, rows, kw",
    [
        ("V1", [csv_row(material="XX")], {}),
        ("V1", [csv_row(defect_code="9999")], {}),
        ("V1", [csv_row(source_doc="NOPE")], {}),
        ("V2", [csv_row(thickness_min="25", thickness_max="8")], {}),
        ("V2", [csv_row(limit_value="0")], {}),
        ("V3", [csv_row(rule_id="a", thickness_min="8", thickness_max="25"),
                csv_row(rule_id="b", thickness_min="12", thickness_max="30",
                        clause_id="KR-3.2.2")], {}),
        ("V3", [csv_row(rule_id="a", thickness_min="8", thickness_max="12"),
                csv_row(rule_id="b", thickness_min="20", thickness_max="30",
                        clause_id="KR-3.2.2")], {}),
        ("V5", [csv_row(source_doc="PAID-DOC")], {}),
        ("V6", [csv_row(rule_id="a"), csv_row(rule_id="b")], {}),
        ("V7", [csv_row(limit_factor="0.1")], {}),
        ("V7", [csv_row(limit_type="비율", unit="percent", limit_rule="prop_t",
                        limit_value="", limit_factor="0.1", ratio_basis="t")], {}),
        ("V8", [csv_row(clause_id="KR-9.9.9")], {}),
        ("V9", [csv_row(rule_id="a", canonical="false"),
                csv_row(rule_id="b", canonical="false", limit_value="4",
                        source_doc="IACS47", clause_id="KR-3.2.2")], {}),
    ],
    ids=["V1-enum", "V1-사상표밖", "V1-미등록문서", "V2-두께역전", "V2-비양수한계",
         "V3-겹침", "V3-틈", "V5-유료문서", "V6-중복조합", "V7-파생값중복기재",
         "V7-비율은const", "V8-미등재조항", "V9-미해소충돌"],
)
def test_g0_rejecting_checks_fire(load, check, rows, kw):
    assert check in _checks_of(load, rows, **kw)


def test_g0_v1_rejects_undefined_column(tmp_path, registry):
    """계약 밖 컬럼은 로드 자체를 거부한다 (컬럼 몰래 늘리기 차단)."""
    cols = list(EXPECTED_COLUMNS) + ["bogus"]
    p = tmp_path / "limits_bogus.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({**{c: csv_row().get(c, "") for c in EXPECTED_COLUMNS}, "bogus": "x"})
    with pytest.raises(LimitsValidationError) as ei:
        load_limits(p, clause_registry=registry, pilot=True)
    assert "V1" in {v.check for v in ei.value.violations}


def test_g0_v4_and_v10_report_without_rejecting(load):
    """V4 단조성·V10 실현성은 경보·보고이지 거부가 아니다 — 둘 다 실제로 채워진다."""
    table = load([csv_row(rule_id="b", quality_level="B", limit_value="5"),
                  csv_row(rule_id="c", quality_level="C", limit_value="3",
                          clause_id="KR-3.2.2")])
    assert table.v4_warnings and all(w.startswith("[V4]") for w in table.v4_warnings)
    flagged = load([csv_row(rule_id="z", limit_value="0.05")])
    assert flagged.v10_flags and all(f.startswith("[V10]") for f in flagged.v10_flags)


def test_g0_v11_hash_is_the_file_hash(load, write_limits):
    path = write_limits([csv_row()])
    table = load_limits(path, clause_registry=["KR-3.2.1"], pilot=True)
    assert table.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_g0_v0_binds_accepted_files_to_the_processing_log(tmp_path, registry):
    """V0 승격 결속: 이력에 없는 accepted 파일(무감사 반입)이 빌드를 거부시킨다."""
    root = tmp_path / "corpus"
    (root / "parse" / "accepted").mkdir(parents=True)
    accepted = root / "parse" / "accepted" / "t1.csv"
    accepted.write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "parse" / "processing_log.jsonl").write_text(
        json.dumps({"status": "accepted", "output_path": "accepted/t1.csv",
                    "output_sha256": hashlib.sha256(accepted.read_bytes()).hexdigest(),
                    "doc_id": "KR-RULES-P2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sources = root / "sources.yaml"
    sources.write_text(
        "documents:\n  - doc_id: KR-RULES-P2\n    license_class: OPEN\n"
        "    allowed_use: [table_extract]\n", encoding="utf-8")

    path = root / "limits.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(EXPECTED_COLUMNS), extrasaction="ignore")
        w.writeheader()
        w.writerow({c: csv_row().get(c, "") for c in EXPECTED_COLUMNS})

    table = load_limits(path, sources_yaml=sources, clause_registry=registry, pilot=False)
    assert table.pilot is False and "V0" not in table.checks_skipped

    (root / "parse" / "accepted" / "ghost.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(LimitsValidationError) as ei:
        load_limits(path, sources_yaml=sources, clause_registry=registry, pilot=False)
    assert "V0" in {v.check for v in ei.value.violations}


# ===========================================================================
# 8. 결정론 — 축이 늘어난 뒤에도 유지되는가
# ===========================================================================


def test_same_table_and_seed_twice_is_byte_identical(load):
    table = load([ROW_RT, ROW_LOF, ROW_VT])
    a = SG.generate_corpus_skeletons(table, seed=11, total=200, cap=20)
    b = SG.generate_corpus_skeletons(table, seed=11, total=200, cap=20)
    assert a, "생성 0건이면 검사가 공허하다"
    assert SG.skeletons_jsonl(a) == SG.skeletons_jsonl(b)
    assert SG.skeletons_sha256(a) == SG.skeletons_sha256(b)
    assert SG.skeletons_sha256(a) != SG.skeletons_sha256(
        SG.generate_corpus_skeletons(table, seed=12, total=200, cap=20)
    )


def test_row_shuffle_leaves_output_unchanged(load):
    base = load([ROW_RT, ROW_LOF, ROW_VT])
    shuffled = load([ROW_VT, ROW_LOF, ROW_RT])
    assert base.sha256 != shuffled.sha256  # 전제: 파일 바이트가 실제로 다르다
    for axis in ("RT", "VT"):
        assert _stripped(SG.generate_corpus_skeletons(
            base, seed=11, total=10_000, cap=20, inspection_method=axis)
        ) == _stripped(SG.generate_corpus_skeletons(
            shuffled, seed=11, total=10_000, cap=20, inspection_method=axis))


def test_adding_other_axis_row_leaves_rt_output_unchanged(load):
    """다른 축의 행이 늘어도 RT 조합의 출력은 한 글자도 달라지지 않는다."""
    base = load([ROW_RT, ROW_LOF])
    with_vt = load([ROW_RT, ROW_LOF, ROW_VT])
    assert _stripped(SG.generate_corpus_skeletons(base, seed=11, total=10_000, cap=20)) \
        == _stripped(SG.generate_corpus_skeletons(with_vt, seed=11, total=10_000, cap=20))
    # 추가된 축 자체는 생성 가능해야 한다 (조용히 사라지면 그것대로 결손)
    assert SG.generate_corpus_skeletons(with_vt, seed=11, total=10_000, cap=20,
                                        inspection_method="VT")


def test_adding_unrelated_same_axis_row_leaves_existing_combos_unchanged(load):
    """같은 축의 무관 행 추가 — 조합 정체성 시드라 기존 조합 난수열이 흔들리지 않는다."""
    unrelated = csv_row(rule_id="KR-CRACK-9", defect_code="100", limit_type="불허",
                        limit_rule="none_permitted", limit_value="", unit="",
                        clause_id="KR-3.2.2")
    base = load([ROW_RT, ROW_LOF])
    bigger = load([ROW_RT, ROW_LOF, unrelated])
    kept = [d for d in _stripped(SG.generate_corpus_skeletons(
        bigger, seed=11, total=10_000, cap=20)) if d["defect_code"] != "100"]
    assert kept == _stripped(SG.generate_corpus_skeletons(
        base, seed=11, total=10_000, cap=20))


def test_d4_path_has_no_randomness(load):
    """D4 경로는 난수가 없어 같은 라벨 묶음이면 두 번 돌려도 바이트 동일하다."""
    table = load([ROW_RT])
    labels = [SG.D4Label(f"img{i}", "d0", "porosity", "ST", D("40"), D("12"))
              for i in range(5)]
    a, qa = SG.skeletons_from_labels(table, labels, scale=D("0.05"),
                                     assumptions=SG.D4Assumptions())
    b, qb = SG.skeletons_from_labels(table, labels, scale=D("0.05"),
                                     assumptions=SG.D4Assumptions())
    assert len(a) == 5 and not qa and not qb
    assert SG.skeletons_jsonl(a) == SG.skeletons_jsonl(b)


def test_label_order_does_not_leak_into_other_labels(load):
    """D4 는 라벨 순서를 뒤집어도 각 라벨의 골격이 그대로다 (위치 의존 없음)."""
    table = load([ROW_RT])
    labels = [SG.D4Label(f"img{i}", "d0", "porosity", "ST", D(str(30 + i)), D("12"))
              for i in range(4)]
    fwd, _ = SG.skeletons_from_labels(table, labels, scale=D("0.05"),
                                      assumptions=SG.D4Assumptions())
    rev, _ = SG.skeletons_from_labels(table, list(reversed(labels)), scale=D("0.05"),
                                      assumptions=SG.D4Assumptions())
    assert {s.sample_id: s.to_json_dict() for s in fwd} == \
        {s.sample_id: s.to_json_dict() for s in rev}


# ===========================================================================
# 9. 회귀 — 수정이 기존 계약을 깨지 않았는가
# ===========================================================================


def test_pilot_csv_still_loads_and_audits_clean():
    table = load_limits(PILOT_CSV, sources_yaml=PILOT_SOURCES, pilot=True)
    sks = SG.generate_corpus_skeletons(table, seed=7, total=400, cap=40)
    assert sks
    report = SG.audit_skeletons(sks, table)
    assert report.ok, report.failures


def test_stage1_still_passes_a_well_formed_c_record(load):
    table = load([ROW_RT])
    assert S1.check_record_c(_c_record(table), table).ok


def test_clause_only_gate_still_holds_for_d4(load):
    """§4-7 게이트 회귀 — judge 는 계산되고 직렬화에서만 합부가 가려진다."""
    table = load([ROW_RT])
    sk = SG.skeleton_from_label(
        table, SG.D4Label("img1", "d0", "porosity", "ST", D("40"), D("12")),
        scale=D("0.05"), assumptions=SG.D4Assumptions())
    assert isinstance(sk, SG.Skeleton) and sk.verdict is not None
    d = sk.to_json_dict()
    assert d["verdict"] is None and d["margin"] is None and d["inspection_method"] == "RT"
