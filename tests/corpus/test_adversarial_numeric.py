"""검토 — 수치 의미론 렌즈.

목표는 잘못된 verdict·잘못된 margin·정의되지 않은 동작을 내는 **구체적 입력**을 실행으로
찾는 것이다. 스펙 §1-3(양자화·부등호 규약) · §4-3(그리드 직접 추출·버킷·최소 보장) ·
§4-4(verdict·margin) · §4-9(G2 감사) · §1-2 5a(검사 방법 축)를 공격 대상으로 삼았다.

파일 구성은 두 갈래다.

- **회귀 방어**: 스윕으로 두들겨 보고 실제로 성립한 불변식. 이후 리팩터가 깨뜨리면
  여기서 걸린다.
- **finding 수정 확인**: 한때 `xfail(strict=True)` 로 고정해 둔 재현 테스트 N1~N8.
  구현이 고쳐지면서 XPASS 로 뒤집혔으므로 마커를 걷고 회귀 가드로 승격했다. 각 테스트는
  무엇이 어떻게 틀렸었는지를 독스트링에 남긴다 — 다시 새면 여기서 걸린다.

수치는 전부 Decimal 이고 0.01 그리드 인덱스로 비교한다 (float 직접 비교 금지, §1-3).
"""

from __future__ import annotations

import inspect
from decimal import Decimal as D
from pathlib import Path

import pytest

from corpus.rules import limit_eval as LE
from corpus.rules import skeleton_gen as SG
from corpus.rules.limits_loader import load_limits
from corpus.rules.schema import (
    FAIL,
    PASS,
    LimitRow,
    LimitRule,
    VerdictMode,
    quantize,
)

from tests.corpus.conftest import make_row, make_table

PILOT_CSV = Path(__file__).resolve().parents[2] / "corpus" / "rules" / "limits_v0_pilot.csv"


def const_row(value: str, op: str = "le", **over) -> LimitRow:
    return make_row(rule_id=f"R-{op}-{value}", limit_op=op, limit_value=D(value), **over)


def grid(d: D) -> int:
    """0.01 그리드 정수 인덱스. 스칼라 비교는 전부 이 위에서 한다."""
    return SG._idx(d)


# ===========================================================================
# 회귀 방어 — 스윕으로 확인한 불변식
# ===========================================================================


@pytest.mark.parametrize("op,at_limit", [("le", PASS), ("lt", FAIL)])
def test_equality_and_pm_one_cell(op: str, at_limit: str) -> None:
    """L 정확 일치와 ±0.01 (§1-3 부등호 규약, §4-4 margin 부호).

    등호가 판정을 가르는 유일한 지점이라 le·lt 를 각각 못 박는다.
    """
    row = const_row("3", op)
    assert LE.judge(row, D("2.99")).verdict == PASS
    assert LE.judge(row, D("3.00")).verdict == at_limit
    assert LE.judge(row, D("3.01")).verdict == FAIL
    assert LE.judge(row, D("2.99")).margin == D("0.01")
    assert LE.judge(row, D("3.00")).margin == D("0.00")
    assert LE.judge(row, D("3.01")).margin == D("-0.01")


@pytest.mark.parametrize("op", ["le", "lt"])
def test_bucket_grid_never_crosses_outer_edges(op: str) -> None:
    """0.9L·1.1L·2.5L 외곽 경계 침범 전수 확인 (§4-3 그리드 직접 추출).

    L 을 0.02~20.00 전 그리드로 훑어 각 버킷의 양 끝점이 정의 구간을 벗어나지 않는지 본다.
    양자화가 버킷 소속을 바꾸는 경로가 정말로 없는지가 요지다.
    """
    for cents in range(2, 2001):
        L = quantize(D(cents) * D("0.01"))
        row = const_row(str(L), op)
        try:
            p = SG.bucket_interval(row, L, SG.BUCKET_PASS)
            bl = SG.bucket_interval(row, L, SG.BUCKET_BOUNDARY_LOW)
            eq = SG.bucket_interval(row, L, SG.BUCKET_BOUNDARY_EQ)
            bh = SG.bucket_interval(row, L, SG.BUCKET_BOUNDARY_HIGH)
            f = SG.bucket_interval(row, L, SG.BUCKET_FAIL)
        except SG.SkeletonGenError:
            continue  # 실현 불가 L — finding 쪽에서 따로 다룬다
        Li = grid(L)
        assert D(p[1]) * D("0.01") < D("0.9") * L, f"합격 버킷이 0.9L 을 침범 (L={L})"
        assert D(bl[0]) * D("0.01") >= D("0.9") * L
        assert eq == (Li, Li)
        assert D(bh[1]) * D("0.01") <= D("1.1") * L
        assert D(f[0]) * D("0.01") > D("1.1") * L, f"불합격 버킷 하단이 1.1L 을 포함 (L={L})"
        assert D(f[1]) * D("0.01") <= D("2.5") * L
        # 인접 버킷이 그리드에서 정확히 맞물린다 (겹침·틈 없음)
        for lo_iv, hi_iv in ((p, bl), (bl, eq), (eq, bh), (bh, f)):
            assert lo_iv[1] + 1 == hi_iv[0], f"버킷 불연속 (L={L})"


@pytest.mark.parametrize("op", ["le", "lt"])
def test_bucket_intent_matches_judge_verdict(op: str) -> None:
    """버킷은 의도, verdict 는 judge() 결과다 (§4-3). 둘이 어긋나는 L 이 있는가.

    =L 버킷만 op 에 따라 갈리고, 나머지 네 버킷은 양 끝점에서 의도와 같은 판정이어야 한다.
    """
    for cents in list(range(2, 400)) + [777, 1234, 2000]:
        L = quantize(D(cents) * D("0.01"))
        row = const_row(str(L), op)
        for bucket, expected in (
            (SG.BUCKET_PASS, PASS),
            (SG.BUCKET_BOUNDARY_LOW, PASS),
            (SG.BUCKET_BOUNDARY_HIGH, FAIL),
            (SG.BUCKET_FAIL, FAIL),
        ):
            try:
                lo, hi = SG.bucket_interval(row, L, bucket)
            except SG.SkeletonGenError:
                continue
            for k in (lo, hi):
                got = LE.judge(row, quantize(D(k) * D("0.01"))).verdict
                assert got == expected, f"L={L} {bucket} idx={k} → {got}"
        eq_lo, _ = SG.bucket_interval(row, L, SG.BUCKET_BOUNDARY_EQ)
        assert LE.judge(row, quantize(D(eq_lo) * D("0.01"))).verdict == (
            PASS if op == "le" else FAIL
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 11, 12, 13, 25, 40])
def test_small_allocation_decomposes_to_n_and_keeps_equality_sample(n: int) -> None:
    """소규모 배분의 층 합계 = n, =L 표본 최소 1건 (§4-3 최소 보장, S1-3).

    조합당 12건에서 =L 표본이 0건이 되는 것이 이 규칙의 존재 이유라 n 을 촘촘히 훑는다.
    """
    counts = SG.decompose_counts(n, LimitRule.CONST)
    assert sum(counts.values()) == n
    assert counts[SG.BUCKET_BOUNDARY_EQ] >= 1
    if n >= 3:
        assert counts[SG.BUCKET_BOUNDARY_LOW] >= 1
        assert counts[SG.BUCKET_BOUNDARY_HIGH] >= 1


@pytest.mark.parametrize("n", [1, 2, 3, 7, 11, 40])
def test_equality_sample_actually_materializes(n: int) -> None:
    """분해표가 아니라 **생성된 골격**에 =L 표본이 실제로 있는지 본다."""
    row = const_row("3")
    sks = SG.generate_corpus_skeletons(make_table(row), seed=1, total=n, cap=40)
    assert len(sks) == n
    eq = [s for s in sks if s.sample_bucket == SG.BUCKET_BOUNDARY_EQ]
    assert eq, f"n={n} 에 =L 표본이 없다"
    assert all(s.size_mm == D("3.00") for s in eq)


def test_multi_seed_sweep_no_bucket_invasion_and_no_verdict_drift() -> None:
    """시드 다수 스윕: 생성 표본이 버킷 구간 밖으로 나가거나 judge 재계산과 어긋나는가."""
    invaded: list[tuple] = []
    drifted: list[str] = []
    for op in ("le", "lt"):
        for value in ("0.10", "0.13", "0.25", "1.00", "3.03", "7.77"):
            row = const_row(value, op)
            L = quantize(D(value))
            table = make_table(row)
            for seed in range(20):
                for s in SG.generate_corpus_skeletons(table, seed=seed, total=40, cap=40):
                    lo, hi = SG.bucket_interval(row, L, s.sample_bucket)
                    if not lo <= grid(s.size_mm) <= hi:
                        invaded.append((op, value, seed, s.sample_bucket, str(s.size_mm)))
                    j = LE.judge(row, s.size_mm)
                    if j.verdict != s.verdict or j.margin != s.margin:
                        drifted.append(s.sample_id)
    assert not invaded, f"버킷 침범: {invaded[:5]}"
    assert not drifted, f"verdict·margin 재계산 불일치: {drifted[:5]}"


@pytest.mark.parametrize(
    "factor,cap", [("0.2", "5"), ("0.3", "5"), ("0.25", "3"), ("0.1", "1.5"), ("0.15", "4")]
)
def test_prop_t_cap_crossing_point(factor: str, cap: str) -> None:
    """prop_t_cap 교차점 t* = cap/factor 와 그 ±0.01 (§1-3 유효 한계 정의).

    L 은 어느 t 에서도 cap 을 넘지 않고, 교차점 근방에서 min() 이 뒤바뀌지 않는다.
    """
    row = make_row(
        rule_id="R-pc", limit_rule="prop_t_cap", limit_value=None,
        limit_factor=D(factor), limit_cap=D(cap), ratio_basis="t",
        thickness_min=D("0"), thickness_max=D("200"),
    )
    tstar = quantize(D(cap) / D(factor))
    for t in (tstar - D("0.01"), tstar, tstar + D("0.01")):
        L = LE.effective_limit(row, t)
        assert L is not None and L <= D(cap), f"t={t} 에서 L={L} 이 cap 을 넘는다"
        assert L == quantize(min(D(factor) * t, D(cap)))
    # 교차점 한참 아래는 비례식, 한참 위는 상한
    assert LE.effective_limit(row, tstar / 2) < D(cap)
    assert LE.effective_limit(row, tstar * 2) == quantize(D(cap))


def test_f4_rejects_zero_and_negative_size() -> None:
    """F4(불허)는 크기 0·음수를 판정하지 않는다 (§4-4, T12).

    limit_value=0 으로 흉내내면 "0 ≤ 0 → 합격" 이 되는 것이 이 규칙의 존재 이유다.
    """
    f4 = make_row(
        limit_type="불허", limit_rule="none_permitted", limit_value=None,
        limit_factor=None, limit_cap=None, ratio_basis=None, unit=None,
    )
    for bad in ("0", "-0.01", "-5"):
        with pytest.raises(ValueError):
            LE.judge(f4, D(bad))
    j = LE.judge(f4, D("0.01"))
    assert j.verdict == FAIL and j.margin is None
    # 비F4 는 음수만 거부한다 (크기 0 은 합격 — 규약상 정상)
    with pytest.raises(ValueError):
        LE.judge(const_row("3"), D("-0.01"))


def test_percent_sampling_respects_hundred_percent_cap() -> None:
    """비율형 표본 상한 min(2.5L, 100%) (§4-3)."""
    row = make_row(
        rule_id="R-pct", limit_type="비율", limit_rule="const", unit="percent",
        limit_value=D("50"), limit_factor=None, limit_cap=None, ratio_basis=None,
    )
    sks = SG.generate_corpus_skeletons(make_table(row), seed=0, total=40, cap=40)
    assert sks and all(s.size_mm is None for s in sks), "비율형은 size_mm 에 넣지 않는다"
    assert max(s.measured_value for s in sks) <= D("100.00")


def test_decimal_roundtrip_reselects_the_same_row() -> None:
    """Decimal 문자열 왕복 후 재조회 행이 생성 시와 같은가 (§4-3 S2-3, T29).

    저장 t 의 반올림이 옆 구간 행을 잡으면 D 재계산이 다른 조항으로 붕괴한다 — 두께 구간이
    맞닿은 3행 표에서 확인한다.
    """
    rows = [
        make_row(rule_id="A", clause_id="KR-3.2.1", thickness_min=D("0"),
                 thickness_max=D("8"), limit_value=D("2")),
        make_row(rule_id="B", clause_id="KR-3.2.2", thickness_min=D("8"),
                 thickness_max=D("25"), limit_value=D("3")),
        make_row(rule_id="C", clause_id="KR-3.2.1", thickness_min=D("25"),
                 thickness_max=None, limit_value=D("4")),
    ]
    table = make_table(*rows)
    sks = SG.generate_corpus_skeletons(table, seed=7, total=120, cap=40)
    assert len(sks) == 120
    for s in sks:
        back = SG.skeleton_from_json(s.to_json_dict())
        assert back == s, f"round-trip 손실: {s.sample_id}"
        row = LE.applicable_row(
            table, back.defect_code, back.material, back.inspection_method,
            back.quality_scheme, back.quality_level, back.thickness_mm,
            limit_type=back.limit_type,
        )
        assert row.clause_id == back.clause_id
        j = LE.judge(row, back.size_mm, SG._basis_value(row, back.thickness_mm))
        assert (j.verdict, j.margin) == (back.verdict, back.margin)


def test_allocate_counts_sum_and_cap_invariant() -> None:
    """총량 축소 규약: sum = min(total, 조합수 × cap), 조합별 ≤ cap (§4-3)."""
    row = const_row("3")
    for ncombo in (1, 2, 3, 7, 11, 250, 999):
        combos = [SG.Combo(f"k{i}", i, f"{i:08d}", row) for i in range(ncombo)]
        for total in (0, 1, 10, 999, 10_000):
            for cap in (1, 3, 40):
                counts = SG.allocate_counts(combos, total, cap)
                assert sum(counts.values()) == min(total, ncombo * cap)
                assert not counts or max(counts.values()) <= cap


def test_pilot_csv_rt_generation_audit_passes() -> None:
    """저장소의 v0-pilot CSV 를 RT 축으로 관통시켜 G2 감사가 통과하는지 (기준선)."""
    table = load_limits(PILOT_CSV, pilot=True, clause_registry=None)
    sks = SG.generate_corpus_skeletons(table, seed=0)
    report = SG.audit_skeletons(sks, table)
    assert report.ok, report.failures


# ===========================================================================
# finding — 재현된 결함
# ===========================================================================


def test_finding_applicable_row_takes_inspection_method() -> None:
    """N1 수정 확인: 스펙 §4-2 시그니처대로 축이 필수 인자다.

    방어가 "호출자가 for_inspection 뷰를 넘긴다"는 규율뿐이면 limit_eval 만 import 하는
    소비처(D 채점기)에는 강제가 없다. 인자로 받아야 축 미지정 호출이 애초에 불가능하다.
    질의 값 ALL 은 거부한다 — ALL 은 조항의 속성이지 질의의 값이 아니다.
    """
    params = list(inspect.signature(LE.applicable_row).parameters)
    assert "inspection_method" in params
    assert params.index("inspection_method") == params.index("material") + 1
    with pytest.raises(ValueError):
        LE.applicable_row(make_table(const_row("3")), "2011", "ST", "ALL",
                          "iso5817", "C", D("12"))


def test_finding_rt_query_must_not_pick_vt_row() -> None:
    """N1 수정 확인: KR 부록 2-7 표면 기공(VT, 0.25t cap 1.5)만 실린 표에 RT 로 질의한다.

    RT 기준(직경 3mm)이면 크기 2.0mm 는 합격인데, VT 행이 선택되면 L=1.50 이라 불합격이
    된다. 예외가 나지 않으면 지표가 틀린 채로 산출되므로 fail-closed 여야 한다.
    """
    vt = make_row(
        rule_id="KRA27-S-01", clause_id="KR-3.2.2", inspection_method="VT",
        defect_code="2011", limit_type="직경", limit_rule="prop_t_cap",
        limit_value=None, limit_factor=D("0.25"), limit_cap=D("1.5"),
        ratio_basis="t", thickness_min=D("0"), thickness_max=None,
    )
    table = make_table(vt)
    with pytest.raises(LookupError):
        LE.applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("12"))
    # 같은 표를 VT 로 물으면 정상 선택된다 (축 도입이 행을 잘라먹은 것이 아니다)
    got = LE.applicable_row(table, "2011", "ST", "VT", "iso5817", "C", D("12"))
    assert got.rule_id == "KRA27-S-01"


def test_finding_rt_vt_coexistence_does_not_silently_pick_vt() -> None:
    """N1 수정 확인: RT 직경 행과 VT 길이 행이 공존해도 축이 조용히 갈리지 않는다.

    예전에는 limit_type 이 갈리는 순간 예외 없이 VT 행이 선택됐다 — 조용한 오선택이라
    탐지가 안 됐다. 이제 RT 질의에 길이 제약을 물으면 후보가 없어 fail-closed 로 떨어진다.
    """
    rt = make_row(
        rule_id="KR-RT-01", clause_id="KR-3.2.1", inspection_method="RT",
        defect_code="2011", limit_type="직경", limit_rule="const",
        limit_value=D("3"), thickness_min=D("0"), thickness_max=None,
    )
    vt = make_row(
        rule_id="KRA27-S-02", clause_id="KR-3.2.2", inspection_method="VT",
        defect_code="2011", limit_type="길이", limit_rule="const",
        limit_value=D("1.5"), unit="mm", thickness_min=D("0"), thickness_max=None,
    )
    table = make_table(rt, vt)
    with pytest.raises(LookupError):
        LE.applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("12"),
                          limit_type="길이")
    # 축을 바꾸면 그 축의 행이 잡힌다 — RT 질의에 VT 기준이 섞이지 않는다는 것이 요점이다
    got = LE.applicable_row(table, "2011", "ST", "VT", "iso5817", "C", D("12"),
                            limit_type="길이")
    assert got.rule_id == "KRA27-S-02"
    rt_row = LE.applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("12"))
    assert rt_row.rule_id == "KR-RT-01"
    assert LE.judge(rt_row, D("2.00")).verdict == PASS


def test_finding_pilot_vt_axis_generation_survives() -> None:
    """N2 수정 확인: 저장소 `limits_v0_pilot.csv` 의 VT 축이 살아 있다.

    원인 행은 `KRA27-S-01` (0.25t, cap 3, 두께 [0, ∞)). t=0.00 표본에서 L=0.00 이라
    어떤 버킷도 정의되지 않아 축 전체가 죽었다. 이제 prop 계열의 t 하한을 L ≥ 0.1 을
    만족하는 최소 그리드 값으로 올려 살리고, 그 사실은 감사 경고로 남는다.
    """
    table = load_limits(PILOT_CSV, pilot=True, clause_registry=None)
    assert table.v10_flags, "전제: V10 이 실현성 미달을 이미 잡아뒀다"
    sks = SG.generate_corpus_skeletons(table, seed=0, inspection_method="VT")
    assert sks
    assert min(s.thickness_mm for s in sks) >= D("0.40")  # 0.1 / 0.25
    report = SG.audit_skeletons(sks, table, inspection_method="VT")
    assert report.ok, report.failures
    assert any("t 하한 보정" in w for w in report.warnings), report.warnings


@pytest.mark.parametrize("factor,t_min", [("0.25", "0"), ("0.1", "0"), ("0.5", "0")])
def test_finding_prop_row_starting_at_zero_thickness(factor: str, t_min: str) -> None:
    """N2 수정 확인: t_min=0 인 prop 행 단독도 생성된다.

    경계 두께 배치가 구간 하한을 강제로 넣으므로, 하한 보정이 없으면 시드와 무관하게 죽는다.
    보정 후에는 전 표본이 실현 가능한 L 을 갖는다.
    """
    row = make_row(
        rule_id="R-p0", limit_rule="prop_t", limit_value=None,
        limit_factor=D(factor), limit_cap=None, ratio_basis="t",
        thickness_min=D(t_min), thickness_max=D("25"),
    )
    sks = SG.generate_corpus_skeletons(make_table(row), seed=0, total=10, cap=10)
    assert len(sks) == 10
    assert all(s.limit_value >= SG.FEASIBILITY_MIN_L for s in sks)


def test_finding_prop_cap_below_feasibility_is_named_not_silent() -> None:
    """N2 보충: t 보정으로 살릴 수 없는 행(cap < 실현성 하한)은 원인을 가리키며 중단한다.

    prop_t_cap 은 L = min(factor·t, cap) 이라 cap 자체가 하한 아래면 어떤 t 도 답이 아니다.
    "빈 버킷" 으로 흘려보내면 V10 이 이미 보고한 행인데도 원인이 안 보인다.
    """
    row = make_row(
        rule_id="R-tiny-cap", limit_rule="prop_t_cap", limit_value=None,
        limit_factor=D("0.25"), limit_cap=D("0.05"), ratio_basis="t",
        thickness_min=D("0"), thickness_max=D("25"),
    )
    with pytest.raises(SG.SkeletonGenError, match="실현성 미달"):
        SG.generate_corpus_skeletons(make_table(row), seed=0, total=10, cap=10)


@pytest.mark.parametrize("limit_pct", ["91", "95", "100"])
def test_finding_high_percent_limit_is_reported_at_load(load, limit_pct: str) -> None:
    """N3 수정 확인: 비율형 상한 미달은 **로드 시점에** 실현성 미달로 보고된다.

    min(2.5L, 100%) 규약에서 (1.1L, 2.5L] ∩ (0, 100%] 이 비는 구간이 있다. 예전에는
    V10 이 하한(L ≥ 0.1)만 봐서 로드가 조용히 통과했고, 조합 하나가 생성 1만 건을 막았다.
    이제 로드가 보고 → scope=excluded 로 배제 → 나머지 조합은 정상 생성이라는
    경로가 성립한다. 배제 전에 생성을 걸면 여전히 fail-closed 로 중단되며, 그때의 메시지도
    V10 을 가리킨다.
    """
    from tests.corpus.conftest import csv_row

    ratio = dict(limit_type="비율", unit="percent", limit_value=limit_pct,
                 defect_code="2012")
    table = load([csv_row(rule_id="R-pct", **ratio)])
    assert any("비율 상한" in f for f in table.v10_flags), table.v10_flags
    with pytest.raises(SG.SkeletonGenError, match="빈 버킷|V10"):
        SG.generate_corpus_skeletons(table, seed=0, total=20, cap=20)

    # scope=excluded 로 배제하면 로드도 생성도 막히지 않는다 (판단 경로)
    excluded = load([csv_row(rule_id="R-pct", scope="excluded", **ratio)])
    assert SG.generate_corpus_skeletons(excluded, seed=0, total=20, cap=20) == ()


@pytest.mark.parametrize("n_combo,total", [(200, 2_000), (700, 10_000), (250, 10_000)])
def test_finding_small_per_combo_allocation_passes_pool_ratio_audit(
    n_combo: int, total: int
) -> None:
    """N4 수정 확인: 조합당 배분이 작아도 감사가 떨어지지 않는다.

    §4-3 최소 보장(=L·상·하 각 1건)은 소규모 배분에서 비율보다 우선하므로, n=10 이면
    합격 3(30%)·경계 3(30%)이 되어 명목 40/40/20 을 벗어난다. 감사의 기대값을 상수가
    아니라 이론 분해 합 sum(decompose_counts(n_i)) 로 두면 규칙대로 동작한 생성물이
    통과하고, 명목에서 벗어났다는 사실은 경고로 남아 원인이 드러난다.
    """
    rows = [
        make_row(rule_id=f"S{i:04d}", clause_id="KR-3.2.1", limit_value=D("3"),
                 thickness_min=D(i), thickness_max=D(i + 1))
        for i in range(n_combo)
    ]
    table = make_table(*rows)
    sks = SG.generate_corpus_skeletons(table, seed=0, total=total, cap=40)
    report = SG.audit_skeletons(sks, table)
    assert report.ok, report.failures
    per_combo = total // n_combo
    if per_combo < 15:
        assert any("버킷 비율 목표 미달성" in w for w in report.warnings), report.warnings
    else:  # 조합당 40 이면 정확히 40/40/20 — 경고가 없어야 한다
        assert not any("버킷 비율" in w for w in report.warnings), report.warnings


def test_finding_float_size_px_is_quarantined_not_fatal() -> None:
    """N5 수정 확인: D4 라벨의 float 수치는 격리이지 전체 중단이 아니다.

    §1-3 의 float 금지 가드는 quantize 에 있는데 곱셈이 먼저라 원인 불명의 TypeError 가
    났다. 이상치 1건이 5만 건 생성을 중단시키지 않는다는 §4-2 원칙과 어긋났다.
    """
    table = make_table(make_row(rule_id="R1", limit_value=D("3")))
    labels = [
        SG.D4Label("i1", "d0", "porosity", "ST", D("100"), D("12")),
        SG.D4Label("i2", "d0", "porosity", "ST", 100.0, D("12")),       # float 유입
        SG.D4Label("i3", "d0", "porosity", "ST", D("100"), 12.0),       # 두께 쪽 float
    ]
    oks, bad = SG.skeletons_from_labels(table, labels, D("0.02"), SG.D4Assumptions())
    assert len(oks) == 1 and len(bad) == 2
    assert {q.reason for q in bad} == {SG.QUARANTINE_INVALID_NUMERIC_TYPE}
    assert oks[0].image_id == "i1"


def test_finding_ratio_basis_a_is_held_in_shared_evaluator_too() -> None:
    """N6 수정 확인: ratio_basis=a 보류가 공유 평가기에도 있다.

    B 는 enumerate_combos·_basis_value·D4 세 곳에서 a 행을 막았는데, §1-4 가 D 소비 경로로
    지정한 limit_eval 에는 가드가 없어 같은 행에서 B 는 "평가 보류", D 는 "합격 margin 0.50"
    이 나왔다. 가드를 공유 평가기로 옮겨 두 경로가 같은 답을 낸다.
    """
    a_row = make_row(
        rule_id="R-a", limit_rule="prop_t", limit_value=None,
        limit_factor=D("0.25"), limit_cap=None, ratio_basis="a",
        thickness_min=D("8"), thickness_max=D("25"),
    )
    table = make_table(a_row)
    assert SG.enumerate_combos(table) == ()          # B: 열거에서 제외
    assert SG.a_basis_hold_rows(table)               # B: 보류 큐에 있음
    with pytest.raises(ValueError, match="평가 보류"):
        LE.effective_limit(a_row, D("12"))
    with pytest.raises(ValueError, match="평가 보류"):
        LE.judge(a_row, D("2.50"), D("12"))


def test_finding_s_basis_assumption_is_carried_in_the_judgment() -> None:
    """N6 보충: ratio_basis=s 의 s=t 가정은 반환 구조에 흔적이 남는다.

    a 와 달리 s 는 완전용입 맞대기 가정으로 평가하기로 했으므로(열린 질문 3) 막지 않는다.
    다만 가정 유래 판정을 확정 판정과 구분할 수 없으면 안 되므로 Judgment 에 싣는다.
    """
    s_row = make_row(
        rule_id="R-s", limit_rule="prop_t", limit_value=None,
        limit_factor=D("0.25"), limit_cap=None, ratio_basis="s",
        thickness_min=D("8"), thickness_max=D("25"),
    )
    j = LE.judge(s_row, D("2.50"), D("12"))
    assert j.verdict == PASS and j.basis_assumption == LE.BASIS_ASSUMPTION_S_EQ_T
    t_row = make_row(rule_id="R-t", limit_rule="prop_t", limit_value=None,
                     limit_factor=D("0.25"), limit_cap=None, ratio_basis="t")
    assert LE.judge(t_row, D("2.50"), D("12")).basis_assumption is None


def test_finding_assumed_quality_level_makes_verdict_conditional() -> None:
    """N7 수정 확인: 가정 품질수준이면 verdict_type 이 조건부다.

    판정을 가른 허용치가 가정 품질수준(configs d4_quality_level)에서 선택된 행에서 나오는데,
    verdict_type 이 thickness_source·scale_source 만 보고 quality_source 를 보지 않아
    "확정"으로 표기됐다. clause_only 에서는 게이트로 가려지지만 full 승격 시 그대로 발효된다.
    """
    table = make_table(make_row(rule_id="R-d4", limit_value=D("3")))
    label = SG.D4Label("IMG1", "d0", "porosity", "ST", D("100"), D("12"))
    sk = SG.skeleton_from_label(
        table, label, scale=D("0.02"),
        assumptions=SG.D4Assumptions(verdict_mode=VerdictMode.FULL),
    )
    assert isinstance(sk, SG.Skeleton)
    assert sk.quality_source == SG.SOURCE_ASSUMED
    assert sk.verdict_type == SG.VERDICT_TYPE_CONDITIONAL


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_finding_prop_edge_thickness_spreads_across_buckets(seed: int) -> None:
    """N8 수정 확인: 경계 두께 표본이 합격 버킷에만 몰리지 않는다.

    `_sample_thicknesses` 가 경계 t 를 리스트 앞쪽에 몰고 buckets 가 STRATA 순(합격 먼저)
    이라 zip 이 항상 합격 버킷과 짝지었다. 행 선택 로직을 데이터에 노출한다는 §4-3 의
    의도가 반만 달성됐던 셈이다 — 정작 구간 경계 두께에서 대소 비교가 갈리는 사례를
    모델이 한 번도 보지 못했다. 버킷 순서를 조합 rng 로 섞어 해소한다 (결정론 유지).
    """
    row = make_row(
        rule_id="R-prop", limit_rule="prop_t", limit_value=None,
        limit_factor=D("0.25"), limit_cap=None, ratio_basis="t",
        thickness_min=D("8"), thickness_max=D("25"),
    )
    table = make_table(row)
    sks = SG.generate_corpus_skeletons(table, seed=seed, total=40, cap=40)
    edge_t = {D("8.00"), D("24.99")}
    buckets = {s.sample_bucket for s in sks if s.thickness_mm in edge_t}
    assert buckets and buckets != {SG.BUCKET_PASS}, f"경계 두께 표본이 {buckets} 에만 있다"
    # 결정론: 같은 시드면 같은 배치
    again = SG.generate_corpus_skeletons(table, seed=seed, total=40, cap=40)
    assert SG.skeletons_sha256(sks) == SG.skeletons_sha256(again)
