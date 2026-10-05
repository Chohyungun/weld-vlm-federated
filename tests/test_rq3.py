"""RQ3 참여 이득 지표 8종 테스트. `51_RQ3_참여이득_지표설계.md` §1.

**평균이 소규모 참여자의 손해를 가려 주는지**가 이 지표군의 존재 이유이고, 그 상황을
테스트가 직접 만든다.
"""

from __future__ import annotations

import pytest

from evaluation.rq3 import (
    CUMULATIVE_BYTES_BASIS,
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


#: `up` 과 `down` 을 따로 받지만 **독립으로 잰 두 값이 아니다.** 실제 원장은 하향을 따로
#: 재지 않아 `bytes_down` 에 상향과 같은 값이 들어 있다(사전실험의 **비어 있지 않은 원장
#: 17개 6,155행 전부** 같았다. 0행 원장 하나를 세면 파일 18개다 — 14번 검수 1-2).
#: 기본값을 같게 둔 것이 그 실물을 따른 것이다. 아래 기대값의 "MB" 는 전부
#: **상향+하향 합**이고 **기록된 상향 값**의 두 배다 — 회선을 잰 값이 아니다.
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
    # 라운드당 2MB = 상향 1MB + 하향 1MB(상향의 복제) → 누적 4MB. 기록된 상향 값은 2MB 다.
    rows = [atomic(0, "C3", 0.62), atomic(1, "C3", 0.70)]
    per_mb = report(atomic_rows=rows).gain_per_mb()
    assert per_mb["C3"] == pytest.approx(10.0 / 4, abs=1e-6)


def test_gain_per_mb_undefined_without_traffic():
    assert report().gain_per_mb()["C3"] is None


def test_gain_per_mb_carries_its_denominator_basis():
    """**표시를 박는 쪽만 두지 않는다** — 읽는 쪽에서 요구하는 자리를 만든다.

    사람이 읽는 문장에만 적으면 `as_dict` 를 소비하는 코드는 분모가 상향의 두 배인 것을
    모른다. 이 프로젝트에서 "표시는 있는데 읽는 쪽이 없다" 를 여러 번 봤다(14번 검수 1-4).
    """
    rows = [atomic(0, "C3", 0.65)]
    basis = report(atomic_rows=rows).as_dict()["gain_per_mb_basis"]
    # id 는 **문자열 리터럴로 고정한다.** 상수를 같은 상수와 맞대면 연결만 보고 값은 못 본다 —
    # 산출을 읽는 코드가 이 id 로 분기하게 되면 값이 조용히 바뀌는 것이 그 코드를 깨뜨린다.
    # 규약을 바꿀 때 이 줄도 같이 고치게 두는 쪽이 맞다. 그 수정이 규약 변경의 표시가 된다.
    assert basis["id"] == "up_plus_down__down_copied_from_up"
    assert basis["id"] == CUMULATIVE_BYTES_BASIS   # 상수가 산출까지 이어지는지도 함께
    assert basis["provisional"] is True          # 팀 결정 전 기본안이다
    assert "따로 재지 않았다" in basis["note"]   # 하향이 독립 실측이 아니라는 사실
    assert "두 배" in basis["note"]              # 그래서 분모가 무엇인지

    # 사람이 읽는 문장에도 값 옆에 같이 나가야 한다.
    text = format_report(report(atomic_rows=rows))
    assert "통신량 대비 이득" in text and "따로 재지 않았다" in text


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


# --- 원자 로그 스키마 확장 내성 (16번 B 검수 — 과제 2) -------------------------------
#
# B(검증 로더 워커 제거)가 착지하며 ④ 원자 로그에 `val_loader_workers` 지표 행을
# 더한다(라운드당 27→30행, 시드 1 원장은 없음). 이 파서는 `metric_name` 으로 고르므로 새 행이
# 궤적 점이나 통신량 누적에 섞이면 안 된다 — 통신량은 (라운드, 클라이언트)당 한 번만 센다.

def _flag_row(round_idx, client, workers, **kw):
    row = atomic(round_idx, client, float(workers), **kw)
    row["metric_name"] = "val_loader_workers"
    return row


def test_val_loader_workers_rows_do_not_touch_trajectory_or_traffic():
    """지표 행이 늘어도 궤적 점 수·누적 바이트가 같다. -1(미계측 관례)·0(B on) 둘 다."""
    base = [atomic(0, "C3", 0.62), atomic(1, "C3", 0.70)]
    with_flags = [base[0], _flag_row(0, "C3", -1), base[1], _flag_row(1, "C3", 0)]
    a, b = report(atomic_rows=base), report(atomic_rows=with_flags)
    assert a.trajectory == b.trajectory
    assert a.gain_per_mb() == b.gain_per_mb()
    # 4 MB — **지표 행이 늘어도 (라운드, 클라이언트)당 한 번만 센다**는 뜻이다.
    # 중복 계수가 없다는 보증이 아니다: 그 4 MB 안에는 상향 2MB 가 상향·하향 두 방향으로
    # 들어가 있다(하향은 따로 잰 값이 아니다 — `CUMULATIVE_BYTES_NOTE`).
    assert b.gain_per_mb()["C3"] == pytest.approx(10.0 / 4, abs=1e-6)


def test_val_loader_workers_flag_is_readable_per_client():
    """속도 축 '표시' 규칙(15번 G9)의 입력 — 플래그를 클라이언트별로 읽을 수 있다."""
    rows = [_flag_row(0, "C1", 0), _flag_row(0, "C3", 0), atomic(0, "C1", 0.5)]
    got = rows_to_client_metric(rows, cell="sep_fed", metric_name="val_loader_workers")
    assert got == {"C1": 0.0, "C3": 0.0}
    assert rows_to_client_metric(rows, cell="sep_fed", metric_name="macro_f1") == {"C1": 0.5}


# --- 원자 로그 v3(R 착지, 13 지표) 내성 (19번 R 검수 — 과제 1(d)) ---------------------------------
#
# R 착지 뒤 ④ 클라이언트 행은 13 지표(v2 + amp·cudnn_deterministic·cudnn_benchmark·batch)다.
# 이 파서는 `metric_name` 으로 고르므로 어떤 버전의 접두가 섞여도 궤적·통신량 누적이 같아야 한다.

_V3_EXTRA = ("val_loader_workers", "amp", "cudnn_deterministic", "cudnn_benchmark", "batch")


def _extra_rows(round_idx, client, values):
    rows = []
    for name, value in zip(_V3_EXTRA, values):
        row = atomic(round_idx, client, float(value))
        row["metric_name"] = name
        rows.append(row)
    return rows


def test_v3_metric_rows_do_not_touch_trajectory_or_traffic():
    """v1 접두(라운드 0, 추가 행 없음) + v3 라운드(라운드 1, 추가 5행)가 섞여도 결과가 같다."""
    base = [atomic(0, "C3", 0.62), atomic(1, "C3", 0.70)]
    mixed = [base[0], base[1], *_extra_rows(1, "C3", (0, 1, 1, 0, 32))]
    a, b = report(atomic_rows=base), report(atomic_rows=mixed)
    assert a.trajectory == b.trajectory
    assert a.gain_per_mb() == b.gain_per_mb()
    assert rows_to_client_metric(mixed, cell="sep_fed", metric_name="batch") == {"C3": 32.0}
    assert rows_to_client_metric(mixed, cell="sep_fed", metric_name="macro_f1") == {"C3": 0.70}


# --- 범위 — 사전실험 분석이다 (30번 3-2~3-4) -----------------------------------------------------

def test_보고의_첫_단서가_범위를_말한다():
    """산출물만 본 사람이 통합형 RQ3 로 읽지 않게 — 두 양은 정의가 다르다(07번 §19-4)."""
    from evaluation.rq3 import SCOPE_NOTE
    r = report()
    assert r.caveats[0] == SCOPE_NOTE
    assert "통합형 본실험의 RQ3" in SCOPE_NOTE and "아니다" in SCOPE_NOTE


def test_지표_행이_없는_원장이면_궤적이_빈_까닭을_적는다():
    """실제 학습 원장은 성능을 싣지 않는다 — 궤적이 조용히 비지 않게."""
    ledger_like = [{**atomic(1, "C3", 0.0), "metric_name": "optimizer_steps"}]
    r = report(atomic_rows=ledger_like)
    assert r.trajectory == ()
    assert any("궤적" in c and "비었다" in c for c in r.caveats)


def test_칸이_없는_행을_연합으로_세지_않는다():
    row = atomic(1, "C3", 0.7)
    del row["cell"]
    assert report(atomic_rows=[row]).trajectory == ()


def test_재개로_겹친_행은_두_번_더하지_않고_멈춘다():
    rows = [atomic(1, "C3", 0.7), atomic(1, "C3", 0.7)]
    with pytest.raises(ValueError, match="둘 이상"):
        report(atomic_rows=rows)


def test_쓰지_않는_단독_칸_인자를_받지_않는다():
    """앞 판은 받고 쓰지 않았다 — 넘기는 쪽은 효과가 있다고 믿는다."""
    with pytest.raises(TypeError):
        report(solo_cell="sep_local")
