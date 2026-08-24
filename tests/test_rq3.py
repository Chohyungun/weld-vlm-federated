"""RQ3 참여 이득 지표 8종 테스트. `51_RQ3_참여이득_지표설계.md` §1.

**평균이 소규모 참여자의 손해를 가려 주는지**가 이 지표군의 존재 이유이고, 그 상황을
테스트가 직접 만든다.
"""

from __future__ import annotations

import pytest

from evaluation.rq3 import (
    Delta,
    attribute_by_client,
    build_rq3_report,
    format_report,
    rows_to_client_metric,
)


def report(fed=None, solo=None, **kw):
    fed = fed or {"C1": 0.82, "C2": 0.78, "C3": 0.70}
    solo = solo or {"C1": 0.80, "C2": 0.74, "C3": 0.60}
    return build_rq3_report(fed, solo, n_train_samples={"C1": 32000, "C2": 16000,
                                                       "C3": 7156}, **kw)


def atomic(round_idx, client, value, cell="sep_fed", up=1_048_576, down=1_048_576):
    return {
        "run_id": "r", "seed": 0, "cell": cell, "split_hash": "h",
        "client_id": client, "round": round_idx, "n_train_samples": 100,
        "metric_name": "macro_f1", "metric_value": value,
        "bytes_up": up, "bytes_down": down, "wall_time": 1.0,
    }


# --- %p 와 % 병기 ----------------------------------------------------------------

def test_delta_carries_both_units():
    """하나만 쓰면 해석이 갈린다. 0.60 → 0.63 은 +3%p 이자 +5%다."""
    d = Delta(0.03, 0.60)
    assert d.points == pytest.approx(3.0)
    assert d.relative == pytest.approx(5.0)


def test_relative_undefined_when_baseline_zero():
    assert Delta(0.03, 0.0).relative is None


def test_report_string_shows_both():
    assert "%p" in str(Delta(0.03, 0.60)) and "%" in str(Delta(0.03, 0.60))


def test_every_gain_exposes_both_units():
    r = report()
    for v in r.as_dict()["per_client_gain"].values():
        assert "delta_pp" in v and "delta_pct" in v


# --- ① 클라이언트별 이득 ----------------------------------------------------------

def test_per_client_gain_is_against_own_solo_baseline():
    """기준선은 전체 평균이 아니라 그 클라이언트의 단독 성능이다(§2)."""
    r = report()
    assert r.per_client["C3"].absolute == pytest.approx(0.10)
    assert r.per_client["C3"].relative == pytest.approx(100 / 6, abs=0.01)


# --- ② 평균 / ③ 최소 -------------------------------------------------------------

def test_mean_gain():
    r = report()
    assert r.mean_gain.absolute == pytest.approx((0.02 + 0.04 + 0.10) / 3)


def test_min_gain_exposes_the_worst_client():
    """평균만 실으면 손해 보는 참여자가 가려진다."""
    r = report(fed={"C1": 0.90, "C2": 0.78, "C3": 0.55},
               solo={"C1": 0.80, "C2": 0.74, "C3": 0.60})
    assert r.mean_gain.absolute > 0            # 평균은 이득처럼 보이는데
    cid, d = r.min_gain
    assert cid == "C3" and d.absolute < 0      # 소규모는 실제로 손해다


def test_min_gain_none_when_no_clients():
    assert build_rq3_report({}, {}).min_gain is None


# --- ④ 소규모 클라이언트 ----------------------------------------------------------

def test_small_client_gain_is_the_direct_answer():
    r = report()
    assert r.small_client_gain.absolute == pytest.approx(0.10)


def test_small_client_absent_returns_none():
    r = build_rq3_report({"C1": 0.8}, {"C1": 0.7}, small_client="C3")
    assert r.small_client_gain is None


# --- ⑤ 이득 양수 비율 -------------------------------------------------------------

def test_positive_ratio_and_losers_reported():
    r = report(fed={"C1": 0.82, "C2": 0.70, "C3": 0.70},
               solo={"C1": 0.80, "C2": 0.74, "C3": 0.60})
    assert r.positive_ratio == pytest.approx(2 / 3)
    assert r.losers == ("C2",)


def test_all_positive_has_no_losers():
    assert report().losers == ()


# --- ⑥ 성능 격차 감소 -------------------------------------------------------------

def test_disparity_reduction_is_positive_when_gap_narrows():
    r = report()
    assert r.disparity.federated_sd < r.disparity.solo_sd
    assert r.disparity.reduction.absolute > 0


def test_disparity_reduction_negative_when_gap_widens():
    r = report(fed={"C1": 0.95, "C2": 0.78, "C3": 0.50},
               solo={"C1": 0.80, "C2": 0.78, "C3": 0.76})
    assert r.disparity.reduction.absolute < 0


def test_single_client_has_zero_disparity():
    r = build_rq3_report({"C1": 0.8}, {"C1": 0.7})
    assert r.disparity.solo_sd == 0.0


# --- ⑦ 라운드별 궤적 --------------------------------------------------------------

def test_trajectory_finds_first_positive_round():
    rows = [atomic(0, "C3", 0.55), atomic(1, "C3", 0.58), atomic(2, "C3", 0.65)]
    r = report(atomic_rows=rows)
    assert r.first_positive_round("C3") == 2      # solo 0.60 을 넘긴 첫 라운드


def test_trajectory_none_when_never_positive():
    rows = [atomic(0, "C3", 0.40), atomic(1, "C3", 0.50)]
    assert report(atomic_rows=rows).first_positive_round("C3") is None


def test_trajectory_ignores_other_cells():
    rows = [atomic(0, "C3", 0.99, cell="sep_central"), atomic(1, "C3", 0.55)]
    r = report(atomic_rows=rows)
    assert r.first_positive_round("C3") is None


def test_trajectory_ignores_other_metrics():
    row = atomic(0, "C3", 0.99)
    row["metric_name"] = "bytes"
    assert report(atomic_rows=[row]).trajectory == ()


def test_malformed_atomic_rows_are_skipped_not_fatal():
    """로그가 일부 깨져도 나머지 궤적은 나와야 한다."""
    rows = [{"cell": "sep_fed", "metric_name": "macro_f1"}, atomic(1, "C3", 0.65)]
    assert len(report(atomic_rows=rows).trajectory) == 1


# --- ⑧ 통신량 대비 이득 -----------------------------------------------------------

def test_gain_per_mb_uses_cumulative_bytes():
    rows = [atomic(0, "C3", 0.62), atomic(1, "C3", 0.70)]   # 2MB/라운드 × 2라운드
    per_mb = report(atomic_rows=rows).gain_per_mb()
    assert per_mb["C3"] == pytest.approx(10.0 / 4, abs=1e-6)


def test_gain_per_mb_undefined_without_traffic():
    assert report().gain_per_mb()["C3"] is None


# --- 귀속 분해 (§3) ---------------------------------------------------------------

def test_attribution_splits_global_scoring_not_test_sets():
    """클라이언트별 시험셋을 만드는 것이 아니라 하나의 채점 결과를 나눠 보는 것이다."""
    per_image = {"i1": 1.0, "i2": 0.0, "i3": 1.0}
    owner = {"i1": "C1", "i2": "C1", "i3": "C3"}
    got = attribute_by_client(per_image, owner)
    assert got == {"C1": [1.0, 0.0], "C3": [1.0]}


def test_attribution_drops_unattributed_images():
    got = attribute_by_client({"i1": 1.0, "ghost": 0.5}, {"i1": "C1"})
    assert got == {"C1": [1.0]}


# --- 원자 로그 소비 ---------------------------------------------------------------

def test_rows_to_client_metric_takes_last_round():
    """last 채점 원칙과 같은 결이다."""
    rows = [atomic(0, "C1", 0.50), atomic(2, "C1", 0.80), atomic(1, "C1", 0.60)]
    assert rows_to_client_metric(rows, cell="sep_fed", metric_name="macro_f1") == {
        "C1": 0.80
    }


def test_rows_to_client_metric_can_pin_a_round():
    rows = [atomic(0, "C1", 0.50), atomic(1, "C1", 0.60)]
    got = rows_to_client_metric(rows, cell="sep_fed", metric_name="macro_f1",
                                round_idx=0)
    assert got == {"C1": 0.50}


# --- 단일 진입점·보고 ------------------------------------------------------------

def test_report_dict_has_all_eight_indicators():
    d = report(atomic_rows=[atomic(0, "C3", 0.65)]).as_dict()
    for key in ("per_client_gain", "mean_gain", "min_gain", "small_client_gain",
                "positive_ratio", "disparity", "first_positive_round", "gain_per_mb"):
        assert key in d


def test_caveat_states_no_personalisation_layer():
    """'모든 참여자가 이득'을 조건 없이 쓰지 않기 위한 장치(§6-3)."""
    joined = " ".join(report().caveats)
    assert "개인화 계층" in joined and "글로벌 모델" in joined


def test_caveat_flags_client_present_in_only_one_cell():
    r = build_rq3_report({"C1": 0.8, "C9": 0.5}, {"C1": 0.7})
    assert any("C9" in c for c in r.caveats)


def test_formatted_report_shows_per_client_not_only_mean():
    text = format_report(report())
    assert "C1" in text and "C2" in text and "C3" in text
    assert "%p" in text


def test_formatted_report_marks_a_loser():
    text = format_report(report(fed={"C1": 0.82, "C2": 0.70, "C3": 0.70},
                                solo={"C1": 0.80, "C2": 0.74, "C3": 0.60}))
    assert "손해" in text
