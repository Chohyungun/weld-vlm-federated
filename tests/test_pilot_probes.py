"""파일럿 프로브 준비 테스트. `52_한사이클_파일럿_계획.md` §2·§5.

**축소 표본이라 고정 통과선을 그대로 대면 틀린다**는 것이 이 모듈의 존재 이유다.
"""

from __future__ import annotations

import pytest

from evaluation.prereg import PREREG
from evaluation.probes.metadata_probe import MetaSample
from evaluation.probes.pilot import (
    expected_constants_note,
    judge_p2,
    judge_p3,
    pilot_probe_report,
    prepare_pilot_gates,
    verify_pilot_constants,
)

CLASSES = ("100", "2011", "301", "401")


def sample(iid: str, codes=()) -> MetaSample:
    return MetaSample(
        image_id=iid, width_px=1280, height_px=720,
        file_bytes=92_160, n_channels=1, quant_table_id=0,
        iso_codes=tuple(codes),
    )


def pilot_samples(n_defect: int, n_normal: int) -> list[MetaSample]:
    out = [sample(f"d{i}", ["2011"]) for i in range(n_defect)]
    out += [sample(f"n{i}") for i in range(n_normal)]
    return out


def stratified_like_rt(n: int = 3000) -> list[MetaSample]:
    """본실험 유병률을 보존한 층화 표본. 클래스별 비율을 RT 실측에서 가져온다."""
    counts = {"100": 2349, "2011": 26967, "301": 2062, "401": 3229}
    total = 62_998
    out: list[MetaSample] = []
    idx = 0
    for code, c in counts.items():
        k = round(n * c / total)
        for _ in range(k):
            out.append(sample(f"d{idx}", [code]))
            idx += 1
    while len(out) < n:
        out.append(sample(f"n{idx}"))
        idx += 1
    return out


# --- 통과선 계산 -----------------------------------------------------------------

def test_stratified_sample_reproduces_prereg_bound():
    """층화가 비율을 보존하면 표본 자명하한이 사전등록 상수와 같아진다."""
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    assert g.matches_prereg
    assert g.trivial_bound == pytest.approx(PREREG.all_positive_macro_f1, abs=0.005)
    assert "일치한다" in g.note


def test_skewed_sample_gets_its_own_gate():
    """유병률이 다르면 고정 상수를 쓰면 안 된다."""
    g = prepare_pilot_gates(pilot_samples(1500, 1500), CLASSES)
    assert not g.matches_prereg
    assert g.p1_prime_gate > PREREG.p1_prime_gate
    assert "다시 잡는다" in g.note


def test_gate_basis_is_always_recorded():
    """값이 같더라도 어떻게 얻었는지를 남긴다. '고정 상수를 갖다 썼다'와 구분되어야 한다."""
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    assert g.gate_basis == "표본 상대(relative)"


def test_gate_is_bound_plus_tolerance():
    g = prepare_pilot_gates(pilot_samples(1500, 1500), CLASSES)
    assert g.p1_prime_gate == pytest.approx(round(g.trivial_bound + 0.005, 4))


def test_no_defect_sample_is_flagged():
    g = prepare_pilot_gates(pilot_samples(0, 100), CLASSES)
    assert "계산할 수 없다" in g.note


def test_prevalence_reported_per_class():
    g = prepare_pilot_gates(pilot_samples(1000, 2000), CLASSES)
    assert g.prevalence["2011"] == pytest.approx(1 / 3)
    assert g.class_counts["2011"] == 1000


# --- P2 출처 판별 ----------------------------------------------------------------

def test_p2_passes_below_gate():
    v = judge_p2(0.52, ci_upper=0.58, n_clusters=120)
    assert v.passed is True and "판별되지 않는다" in v.detail


def test_p2_fails_above_auc_gate():
    v = judge_p2(0.83, ci_upper=0.88, n_clusters=120)
    assert v.passed is False and "후퇴 사다리" in v.detail


def test_p2_fails_when_ci_upper_exceeds_even_if_point_passes():
    """점추정만 보면 통과인데 CI 상한이 넘는 경우. 통과로 보지 않는다."""
    v = judge_p2(0.59, ci_upper=0.71, n_clusters=120)
    assert v.passed is False and "CI 상한" in v.detail


def test_p2_underpowered_is_undecided_not_pass():
    """묶음이 부족하면 '판정 불가'다. 미도달을 통과처럼 쓰지 않는다."""
    v = judge_p2(0.50, ci_upper=0.55, n_clusters=5)
    assert v.passed is None and "판정 불가" in v.detail


def test_p2_without_ci_uses_point_only():
    assert judge_p2(0.55).passed is True


# --- P3 패치셔플·저해상 -----------------------------------------------------------

def test_p3_passes_when_both_conditions_clear():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    v = judge_p3(shuffle_auc=0.55, lowres_macro_f1=0.20, gates=g)
    assert v.passed is True


def test_p3_uses_pilot_gate_not_fixed_constant():
    """유병률이 다른 표본에서 고정 통과선을 쓰면 판정이 뒤집힌다."""
    skewed = prepare_pilot_gates(pilot_samples(1500, 1500), CLASSES)
    v = judge_p3(shuffle_auc=0.55, lowres_macro_f1=0.40, gates=skewed)
    assert v.passed is True                     # 표본 자명하한(0.667) 아래라 통과
    assert v.values["gate"] > PREREG.p1_prime_gate
    assert "표본 상대" in v.detail


def test_p3_fails_on_shuffle_leak():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    v = judge_p3(shuffle_auc=0.82, lowres_macro_f1=0.20, gates=g)
    assert v.passed is False and "규격 외 채널" in v.detail


def test_p3_fails_on_lowres_leak():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    v = judge_p3(shuffle_auc=0.55, lowres_macro_f1=0.45, gates=g)
    assert v.passed is False


# --- 상수 대조 -------------------------------------------------------------------

def test_measurement_reproducing_sample_bound_passes():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    assert verify_pilot_constants(g.trivial_bound, g).passed is True


def test_drifted_measurement_blocks_interpretation():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    v = verify_pilot_constants(g.trivial_bound + 0.05, g)
    assert v.passed is False and "계측을 먼저" in v.detail


# --- 종합 리포트 -----------------------------------------------------------------

def test_report_counts_undecided_separately_from_pass():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    rep = pilot_probe_report(
        [judge_p2(0.50, n_clusters=5), judge_p3(0.55, 0.20, g)], g
    )
    assert rep["n_passed"] == 1 and rep["n_undecided"] == 1
    assert rep["all_clear"] is False


def test_report_all_clear_only_when_everything_passes():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    rep = pilot_probe_report(
        [judge_p2(0.50, ci_upper=0.55, n_clusters=120), judge_p3(0.55, 0.20, g)], g
    )
    assert rep["all_clear"] is True


def test_report_warns_against_using_pilot_as_conclusion():
    g = prepare_pilot_gates(stratified_like_rt(), CLASSES)
    rep = pilot_probe_report([], g)
    assert "결론으로 쓰지 않는다" in rep["note"]


def test_note_shows_both_main_and_pilot_gates():
    note = expected_constants_note(pilot_samples(1500, 1500), CLASSES)
    assert "0.2081" in note and "본실험" in note and "파일럿" in note
