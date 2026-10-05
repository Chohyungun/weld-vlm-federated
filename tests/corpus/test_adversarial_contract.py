"""검토 — 결정론·계약 렌즈.

이 파일은 두 종류의 테스트를 담는다:

1. **통과 계약 가드** — 결정론(같은 입력 2회 바이트 동일, PYTHONHASHSEED 교차,
   행 셔플·무관 행 추가 불변), (c) 경로 verdict_mode 게이팅 오적용 부재,
   quarantine 비중단, counts 오염 검출, T01~T35 실재 대조.
2. **회귀 가드** — 한때 xfail(strict) 로 고정해 뒀던 finding #1·#4·#5·#6a·#6b 의 재현
   테스트. 구현이 고쳐지면서 XPASS 로 뒤집혔으므로 마커를 걷고 통과 계약으로 승격했다.
   각 테스트는 어떤 결함을 막고 있는지를 독스트링에 남긴다 — 다시 새면 여기서 걸린다.

finding #2(조합키 충돌)·#3(비율형 상한)은 fail-closed 로 중단하는 현재 동작을 통과
테스트로 못 박아 둔다. 조용한 오생성이 아니라 시끄러운 중단이라는 것이 요점이다.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from corpus.generate.counts_builder import (
    CountsError,
    build_counts,
    validate_counts,
)
from corpus.generate.numeric_lock import check_normal_lock, check_numeric_lock
from corpus.rules.limits_loader import (
    EXPECTED_COLUMNS,
    LimitsValidationError,
    load_limits,
)
from corpus.rules.schema import FAIL, PASS, VerdictMode
from corpus.rules.skeleton_gen import (
    QUARANTINE_DEGENERATE_POLYGON,
    QUARANTINE_EXCLUDED_2012,
    QUARANTINE_NO_APPLICABLE_ROW,
    QUARANTINE_UNMAPPED_DEFECT,
    D4Assumptions,
    D4Label,
    Skeleton,
    SkeletonGenError,
    enumerate_combos,
    find_defect_tokens,
    generate_corpus_skeletons,
    load_defect_lexicon,
    skeletons_from_labels,
    skeletons_jsonl,
    skeletons_sha256,
)
from corpus.validate.stage1_rules import check_record_c
from tests.corpus.conftest import csv_row, make_row, make_table

D = Decimal
REPO = Path(__file__).resolve().parents[2]
PILOT_CSV = REPO / "corpus" / "rules" / "limits_v0_pilot.csv"
PILOT_SOURCES = REPO / "corpus" / "parse" / "sources.yaml"


# ---------------------------------------------------------------------------
# 픽스처 — 공식 로더 경유 3행 테이블 (fixture sha 가 아니라 실제 CSV sha)
# ---------------------------------------------------------------------------

ROW_A = csv_row()  # 2011 ST 직경 const 3 [8,25) iso5817:C, KR-3.2.1
ROW_B = csv_row(rule_id="KR-3.2.2-001", defect_code="401", limit_type="길이",
                clause_id="KR-3.2.2")
ROW_C = csv_row(rule_id="IACS-001", defect_code="301", limit_type="길이",
                clause_id="IACS47-3.2.1", source_doc="IACS47",
                quality_scheme="iacs", quality_level="STD")
ROW_D_UNRELATED = csv_row(rule_id="KR-CRACK-001", defect_code="100",
                          limit_type="불허", limit_rule="none_permitted",
                          limit_value="", unit="")


def _stripped(skeletons) -> list[dict]:
    """provenance(limits sha 가 당연히 달라지는 부분)를 뗀 직렬화 출력."""
    out = []
    for s in skeletons:
        d = s.to_json_dict()
        d.pop("provenance")
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# 1. 결정론 — 같은 (table, seed) 2회 실행 jsonl 바이트 동일 (실 파일럿 CSV)
# ---------------------------------------------------------------------------


def test_attack_same_input_twice_byte_identical_real_pilot_csv():
    t1 = load_limits(PILOT_CSV, sources_yaml=PILOT_SOURCES, pilot=True)
    t2 = load_limits(PILOT_CSV, sources_yaml=PILOT_SOURCES, pilot=True)
    s1 = generate_corpus_skeletons(t1, seed=7, total=400, cap=40)
    s2 = generate_corpus_skeletons(t2, seed=7, total=400, cap=40)
    assert skeletons_jsonl(s1) == skeletons_jsonl(s2)  # 바이트 동일
    assert skeletons_sha256(s1) == skeletons_sha256(s2)


def test_attack_hashseed_cross_process_determinism():
    """PYTHONHASHSEED 를 바꾼 두 서브프로세스가 동일 sha 를 내야 한다 —
    dict/set 순회 순서 의존이 있으면 여기서 갈라진다."""
    child = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from pathlib import Path\n"
        "from corpus.rules.limits_loader import load_limits\n"
        "from corpus.rules.skeleton_gen import generate_corpus_skeletons, skeletons_sha256\n"
        "t = load_limits(r'%s', sources_yaml=r'%s', pilot=True)\n"
        "sk = generate_corpus_skeletons(t, seed=7, total=400, cap=40)\n"
        "print(skeletons_sha256(sk))\n"
    ) % (REPO, PILOT_CSV, PILOT_SOURCES)
    shas = []
    for hashseed in ("0", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONIOENCODING="utf-8")
        r = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, env=env, cwd=str(REPO)
        )
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
        shas.append(r.stdout.decode("utf-8").strip())
    assert shas[0] == shas[1] and len(shas[0]) == 64


# ---------------------------------------------------------------------------
# 2. 조합 정체성 시드 — 행 셔플·무관 행 추가에 기존 조합 출력 불변 (로더 경유)
#    provenance 는 CSV sha 가 정당하게 달라지므로 떼고 비교한다.
# ---------------------------------------------------------------------------


def test_attack_csv_row_shuffle_output_invariant(load):
    base = load([ROW_A, ROW_B, ROW_C])
    shuffled = load([ROW_C, ROW_A, ROW_B])
    assert base.sha256 != shuffled.sha256  # 전제: 실제로 다른 파일 바이트다
    out1 = generate_corpus_skeletons(base, seed=7, total=10_000, cap=20)
    out2 = generate_corpus_skeletons(shuffled, seed=7, total=10_000, cap=20)
    assert _stripped(out1) == _stripped(out2)


def test_attack_unrelated_row_added_existing_combos_invariant(load):
    base = load([ROW_A, ROW_B, ROW_C])
    bigger = load([ROW_A, ROW_B, ROW_C, ROW_D_UNRELATED])
    out1 = generate_corpus_skeletons(base, seed=7, total=10_000, cap=20)
    out2 = generate_corpus_skeletons(bigger, seed=7, total=10_000, cap=20)
    kept = [d for d in _stripped(out2) if d["defect_code"] != "100"]
    assert kept == _stripped(out1)
    # 추가된 조합 자체는 생성됐어야 한다 (무관 행이 사라지면 그것대로 결손)
    assert any(d["defect_code"] == "100" for d in _stripped(out2))


# ---------------------------------------------------------------------------
# 3. verdict_mode 게이팅 — (c) 경로는 항상 full, 게이트가 새면 안 된다
# ---------------------------------------------------------------------------


def test_attack_c_path_never_gated(load):
    table = load([ROW_A, ROW_B, ROW_C])
    out = generate_corpus_skeletons(table, seed=7, total=10_000, cap=20)
    assert out, "생성 0건이면 검사 자체가 공허하다"
    for sk in out:
        assert sk.verdict_mode is VerdictMode.FULL
        d = sk.to_json_dict()
        assert d["verdict"] in (PASS, FAIL)  # 직렬화에서도 게이트되지 않는다
        assert d["verdict_mode"] == "full"
        if sk.limit_rule.value != "none_permitted":
            assert d["margin"] is not None


def test_attack_stage1_rejects_gated_c_record(load):
    """clause_only 게이트가 잘못 (c) 레코드에 섞이면 1단계가 잡아야 한다."""
    table = load([ROW_A])
    sk = generate_corpus_skeletons(table, seed=7, total=6, cap=6)[0]
    rec = dict(sk.to_json_dict())
    rec["text"] = (
        f"관찰: 직경 {rec['size_mm']} mm 음영. 조회: 두께 {rec['thickness_mm']} mm "
        f"기준 KR-3.2.1 허용 {rec['limit_value']} mm 이하. 판정: {rec['verdict']}."
    )
    good = check_record_c(rec, table)
    assert good.ok, good  # 대조군: 정상 full 레코드는 통과

    gated = dict(rec)
    gated["verdict"] = None
    gated["margin"] = None
    gated["text"] = (
        f"관찰: 직경 {rec['size_mm']} mm 음영. 조회: 두께 {rec['thickness_mm']} mm "
        f"기준 KR-3.2.1 허용 {rec['limit_value']} mm 이하."
    )
    bad = check_record_c(gated, table)
    assert not bad.ok
    assert "schema_violation" in bad.reasons


# ---------------------------------------------------------------------------
# 4. quarantine — 이상치가 전체를 중단시키지 않고, 사유 코드가 남는다
# ---------------------------------------------------------------------------


def test_attack_quarantine_mixed_reasons_no_halt(load):
    table = load([ROW_A])
    assumptions = D4Assumptions()  # clause_only 기본
    labels = [
        D4Label("img-bad1", "d0", "undercut", "ST", D("40"), D("12")),   # 사상표 밖
        D4Label("img-bad2", "d0", "2012", "ST", D("40"), D("12")),        # alt 코드 배제
        D4Label("img-good", "d0", "porosity", "ST", D("40"), D("12")),    # 정상
        D4Label("img-bad3", "d0", "porosity", "ST", D("0"), D("12")),     # 퇴화 폴리곤
        D4Label("img-bad4", "d0", "porosity", "ST", D("40"), D("30")),    # 구간 밖 두께
    ]
    oks, bad = skeletons_from_labels(table, labels, scale=D("0.05"), assumptions=assumptions)
    assert len(oks) == 1 and oks[0].image_id == "img-good"
    reasons = {q.image_id: q.reason for q in bad}
    assert reasons == {
        "img-bad1": QUARANTINE_UNMAPPED_DEFECT,
        "img-bad2": QUARANTINE_EXCLUDED_2012,
        "img-bad3": QUARANTINE_DEGENERATE_POLYGON,
        "img-bad4": QUARANTINE_NO_APPLICABLE_ROW,
    }
    # clause_only: judge 는 계산돼 내부에 보유되고, 직렬화만 게이트된다 (§4-7)
    sk = oks[0]
    assert isinstance(sk, Skeleton) and sk.verdict in (PASS, FAIL)
    assert sk.to_json_dict()["verdict"] is None
    assert sk.to_json_dict()["margin"] is None


# ---------------------------------------------------------------------------
# 5. counts.json — 오염 주입이 실제로 걸리는지
# ---------------------------------------------------------------------------

_ADOPTED = [
    {"image_id": "i1", "defects": [{"defect_code": "2011"}]},
    {"image_id": "i2", "defects": []},
    {"image_id": "i3", "defects": [{"defect_code": "100"}]},
]
_CLIENT_OF = {"i1": "C1", "i2": "C1", "i3": "C2"}


def _counts():
    return build_counts(
        _ADOPTED, _CLIENT_OF, n_generated=7,
        discarded={"stage0_numeric_lock": 4},
        limits_sha256="a" * 64, manifest_sha256="b" * 64,
        git_commit="x", date="2026-08-17",
    )


def test_attack_counts_totals_vs_adopted_mismatch_detected():
    counts = _counts()
    with pytest.raises(CountsError):
        validate_counts(counts, _ADOPTED[:-1])  # 채택 행수가 다르면 걸린다


def test_attack_counts_consistent_tamper_still_detected():
    """n_total·n_defect 를 함께 조작해 클라이언트 내부 정합(n_d+n_n=n_t)은 지켜도
    합계=채택 행수 대조가 잡는다."""
    counts = _counts()
    tampered = json.loads(json.dumps(counts))
    tampered["clients"]["C1"]["n_total"] += 1
    tampered["clients"]["C1"]["n_defect"] += 1
    with pytest.raises(CountsError):
        validate_counts(tampered, _ADOPTED)


# ---------------------------------------------------------------------------
# 6. 스펙 T01~T35 실재 대조 — 번호 결손은 곧 커버리지 결손
# ---------------------------------------------------------------------------


def test_attack_spec_test_numbers_all_present():
    corpus_tests = sorted((Path(__file__).parent).glob("test_*.py"))
    blob = "\n".join(
        p.read_text(encoding="utf-8") for p in corpus_tests
        if p.name != Path(__file__).name
    )
    missing = [f"T{n:02d}" for n in range(1, 36) if f"T{n:02d}" not in blob]
    assert not missing, f"스펙 §4-8 테스트 번호 결손: {missing}"


def test_attack_g0_check_ids_all_present_in_loader():
    src = (REPO / "corpus" / "rules" / "limits_loader.py").read_text(encoding="utf-8")
    src += (REPO / "corpus" / "rules" / "schema.py").read_text(encoding="utf-8")
    missing = [f"V{n}" for n in range(0, 12) if not re.search(rf"\bV{n}\b", src)]
    assert not missing, f"G0 검사 ID 결손: {missing}"


# ===========================================================================
# 이하 회귀 가드 — 실행으로 재현했던 결함이 다시 새지 않는지 (finding 번호 참조)
# ===========================================================================


def test_finding1_inspection_method_column_accepted(tmp_path):
    """finding #1 수정 확인 (회귀 가드): 스펙 v1.3 §1-2 5a 의 inspection_method 가 계약
    컬럼이라 로더가 받는다. 예전에는 '미정의 컬럼'으로 로드 자체를 거부했다."""
    import csv as _csv

    cols = list(EXPECTED_COLUMNS) + ["inspection_method"]
    row = dict(csv_row())
    row["inspection_method"] = "RT"
    p = tmp_path / "limits_im.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({c: row.get(c, "") for c in cols})
    table = load_limits(p, pilot=True)
    assert len(table.rows) == 1
    assert table.rows[0].inspection_method.value == "RT"


def test_finding1_rt_vt_coexistence_loadable(load):
    """finding #1 수정 확인 (회귀 가드): RT/VT 기준 병존 — 결정 #13 의 존재 이유 —
    이 표현 가능하다. 예전에는 V6 이중 canonical·V3 구간 겹침으로 로드가 거부됐다."""
    rt = csv_row()  # RT 기공 상당
    vt = csv_row(rule_id="KR-VT-001", clause_id="KR-3.2.2", limit_value="4")  # VT 기공 상당
    table = load([rt, vt])
    assert len(table.rows) == 2
    assert {r.inspection_method.value for r in table.rows} == {"RT", "VT"}


def test_finding2_dual_constraint_combo_aborts_generation():
    """finding #2 재현 (통과 테스트 — 현재의 fail-closed 동작 고정):
    §4-1·V3(D1-8)가 정당하다고 선언한 복수 제약(지름+비율, 같은 조항) 테이블에서
    §4-2 조합키(material·limit_type 미포함)가 충돌해 (c) 생성 전체가 중단된다."""
    r_diam = make_row(rule_id="A-1", defect_code="2012", limit_type="직경",
                      limit_rule="const", limit_value=D("3"), unit="mm",
                      clause_id="KR-3.4.1")
    r_ratio = make_row(rule_id="A-2", defect_code="2012", limit_type="비율",
                       limit_rule="const", limit_value=D("25"), unit="percent",
                       clause_id="KR-3.4.1")
    with pytest.raises(SkeletonGenError, match="조합키 충돌"):
        enumerate_combos(make_table(r_diam, r_ratio))
    # 같은 조항의 ST·AL 행도 동일하게 충돌한다 (material 미포함)
    r_st = make_row(rule_id="B-1", material="ST")
    r_al = make_row(rule_id="B-2", material="AL")
    with pytest.raises(SkeletonGenError, match="조합키 충돌"):
        enumerate_combos(make_table(r_st, r_al))


def test_finding3_high_percent_ratio_row_aborts_whole_run():
    """finding #3 (통과 테스트 — 현재 동작 고정): F5 행 L=95% 는 불합격 버킷
    (1.1L, 2.5L]∩(0,100%] 이 공집합이라 생성이 전체 중단된다.

    중단 자체는 fail-closed 로 옳다. 고친 것은 "발견 시점"이다 — 이제 로더의 V10 이
    비율형 상한 미달을 로드 시점에 실현성 미달로 보고해 scope=excluded 판단 경로를 태운다
    (tests/corpus/test_adversarial_numeric.py 의 V10 비율 상한 테스트). 여기 픽스처는
    로더를 거치지 않는 make_table 이라 보고 없이 생성 시점에 걸린다."""
    r95 = make_row(rule_id="D-1", defect_code="2012", limit_type="비율",
                   limit_rule="const", limit_value=D("95"), unit="percent",
                   clause_id="KR-3.4.9")
    table = make_table(r95)
    with pytest.raises(SkeletonGenError, match="빈 버킷"):
        generate_corpus_skeletons(table, seed=1, total=40, cap=40)


def test_finding4_discarded_tamper_detected():
    """finding #4 수정 확인 (회귀 가드): n_generated 가 본문에 있어 discarded 사후 변조가
    파일만으로 재검산된다. 예전에는 §7-5 정합 3항이 build 시점에만 강제됐다."""
    counts = _counts()
    tampered = json.loads(json.dumps(counts))
    tampered["discarded"]["stage2_judge"] = 999  # 합계 ≠ 생성량 − 채택량
    with pytest.raises(CountsError):
        validate_counts(tampered, _ADOPTED)


def test_finding5_lowercase_english_defect_term_discarded():
    """finding #5 수정 확인 (회귀 가드): 결함 어휘 락이 대소문자를 구분하지 않는다.
    예전에는 프로덕션 경로(check_normal_lock)만 대소문자를 구분해 소문자 'crack' 이 통과했다."""
    lex = load_defect_lexicon()
    text = "필름 전반에 crack 성 지시 없음. 판정: 합격."
    assert find_defect_tokens(text, lex)
    r = check_normal_lock(text, defect_lexicon=lex, thickness_mm=D("12.00"),
                          quality_level="C", expected_verdict="합격")
    assert not r.ok


_SK_RATIO = {
    "defect_code": "2012", "material": "ST", "quality_scheme": "iso5817",
    "quality_level": "C", "limit_type": "비율", "limit_rule": "const",
    "thickness_mm": "12.00", "size_mm": None, "measured_value": "20.00",
    "clause_id": "KR-3.2.1", "limit_value": "25.00", "margin": "5.00",
    "verdict": "합격", "limit_op": "le", "unit": "percent",
    "limit_factor": None, "limit_cap": None, "ratio_basis": None,
}

_SK_CONST = {
    "defect_code": "2011", "material": "ST", "quality_scheme": "iso5817",
    "quality_level": "C", "limit_type": "직경", "limit_rule": "const",
    "thickness_mm": "12.00", "size_mm": "2.00", "measured_value": None,
    "clause_id": "KR-3.2.1", "limit_value": "3.00", "margin": "1.00",
    "verdict": "합격", "limit_op": "le", "unit": "mm",
    "limit_factor": None, "limit_cap": None, "ratio_basis": None,
}


def test_finding6_unit_tamper_discarded():
    """finding #6a 수정 확인 (회귀 가드): 수치 토큰은 숫자+단위다 (§5-4). percent 골격의
    값을 mm 로 표기하면 다른 물리량이므로 폐기된다."""
    text = ("관찰: 기공 면적 비율 20.00 mm. 분류: 2012. 조회: 두께 12 mm 기준 "
            "KR-3.2.1 허용 25 mm 이하. 판정: 합격.")
    r = check_numeric_lock(text, _SK_RATIO)
    assert not r.ok


def test_finding6_hallucinated_standard_ref_discarded():
    """finding #6b 수정 확인 (회귀 가드): 표준 식별자는 마스킹 후 화이트리스트와 대조된다.
    예전에는 수치 스캔 전에 떼어내기만 해 환각 표준이 무조건 통과했다."""
    good = ("관찰: 직경 2.00 mm의 원형 음영. 분류: 기공 (ISO 6520-1 2011). "
            "조회: 모재 두께 12 mm 품질수준 C → 조항 KR-3.2.1 기준 3 mm 이하. "
            "대조: 2.00 ≤ 3.00. 판정: 합격.")
    assert check_numeric_lock(good, _SK_CONST).ok  # 대조군
    r = check_numeric_lock(good + " 아울러 ISO 9999 및 EN 1290 요건도 충족한다.", _SK_CONST)
    assert not r.ok
