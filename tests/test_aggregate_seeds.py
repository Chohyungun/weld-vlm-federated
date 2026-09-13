"""시드 3세트 집계 — 합성 산출물 3벌로 규칙을 고정한다.

지키는 것:
1. **대조선이 시드 사이에 다르면 멈춘다** — 동결본이 바뀐 신호다(24번 §4).
2. **회복률의 평균과 평균의 회복률을 둘 다 낸다** — 다른 수다.
3. **분모 게이트가 사전등록 함수를 실호출한다** — 풀링 sd 로.
4. **부차 지표의 일반식** — 연합÷중앙집중, (연합−로컬평균)÷로컬평균, 클라이언트별 (연합−로컬_c)÷로컬_c.
   최악 참여자가 평균에 가려지지 않게 따로 낸다.
"""

from __future__ import annotations

import pytest

from scripts.probe.aggregate_seeds import (
    DET_TAGS,
    check_baselines_identical,
    client_improvement,
    recovery_table,
    secondary_ratios,
)

BASE = {"classification_axis": {"macro_ap_freq": 0.95}, "position_axis": {"constant_box_map_50": 0.001},
        "primary_rule": "idq512", "self_check_reproduced": True}


def _payload(seed: int, floor: dict[str, float], op: dict[str, float] | None = None) -> dict:
    op = op or {t: 0.5 for t in DET_TAGS}
    return {
        "params": {"seed": seed},
        "content_free_baseline": dict(BASE),
        "threshold_independent": {"per_tag": {
            t: {"map_50": floor[t], "map_50_95": floor[t] / 2, "macro_ap": floor[t] + 0.3}
            for t in DET_TAGS}},
        "metrics": {t: {"macro_f1": op[t], "miss_rate": 1 - op[t], "defect_recall": op[t],
                        "class_jaccard": op[t], "bbox_iou": op[t]} for t in DET_TAGS},
    }


# 시드 1 실측과 같은 구조의 값 — 검산용 (17번 §3-0: 연합 0.2820 · 중앙 0.4926 · C1 0.3584 …)
S1 = {"sep_central": 0.4926, "sep_local_C1": 0.3584, "sep_local_C2": 0.1358,
      "sep_local_C3": 0.0932, "sep_fed": 0.2820}


def three_seeds() -> dict[int, dict]:
    return {
        1: _payload(1, S1),
        2: _payload(2, {t: v * 0.98 for t, v in S1.items()}),
        3: _payload(3, {t: v * 1.01 for t, v in S1.items()}),
    }


def test_대조선이_다르면_멈춘다():
    p = three_seeds()
    p[2]["content_free_baseline"] = {**BASE, "classification_axis": {"macro_ap_freq": 0.94}}
    with pytest.raises(SystemExit, match="동결본"):
        check_baselines_identical(p)


def test_대조선이_같으면_통과하고_값을_돌려준다():
    out = check_baselines_identical(three_seeds())
    assert out["identical_across_seeds"] and out["classification_axis"]["macro_ap_freq"] == 0.95


def test_회복률의_평균과_평균의_회복률을_둘_다_낸다():
    rec = recovery_table(three_seeds(), {})["map_50"]
    assert len(rec["by_seed"]) == 3
    assert rec["by_seed"][0]["recovery_pct"] == pytest.approx(29.0, abs=0.1)
    # 스케일만 다른 시드는 같은 회복률 → 평균도 같고 '평균의 회복률'도 같다
    assert rec["mean_of_seed_recoveries_pct"] == pytest.approx(29.0, abs=0.1)
    assert rec["recovery_of_seed_means_pct"] == pytest.approx(29.0, abs=0.1)


def test_분모_게이트가_풀링_sd_로_판정한다():
    rec = recovery_table(three_seeds(), {})["map_50"]
    g = rec["denominator_gate"]["primary"]
    assert g["seed_sd"] is not None and g["seed_sd_ci"][0] < g["seed_sd"] < g["seed_sd_ci"][1]
    assert g["pass"] is True and g["denominator"] > 3 * g["seed_sd"]
    assert "분모 D 자체" in rec["denominator_gate"]["secondary_denominator_sd"]["seed_sd_definition"]


def test_분모가_시드_잡음_안이면_게이트가_막는다():
    """중앙과 로컬이 거의 같고 시드 잡음이 크면 회복률을 헤드라인으로 싣지 않는다."""
    near = {"sep_central": 0.30, "sep_local_C1": 0.29, "sep_local_C2": 0.30, "sep_local_C3": 0.31,
            "sep_fed": 0.30}
    p = {1: _payload(1, near),
         2: _payload(2, {t: v + 0.03 for t, v in near.items()}),
         3: _payload(3, {t: v - 0.03 for t, v in near.items()})}
    g = recovery_table(p, {})["map_50"]["denominator_gate"]["primary"]
    assert g["pass"] is False


def test_부차_지표_일반식_검산():
    """시드 1 실측 구조에서 연합÷중앙집중 57.2%, 개별학습 대비 +44.0% 가 나와야 한다."""
    sec = secondary_ratios(three_seeds())
    s1 = sec["by_seed"][0]
    assert s1["fed_over_central_pct"] == pytest.approx(57.25, abs=0.05)
    assert s1["improvement_over_local_mean_pct"] == pytest.approx(44.0, abs=0.1)
    assert sec["definitions"]["fed_over_central_pct"] == "연합 ÷ 중앙집중 × 100"


def test_클라이언트별_향상률과_최악_참여자():
    """C1 −21.3% / C2 +107.7% / C3 +202.6% — 평균이 양수여도 C1 은 손해다."""
    cli = client_improvement(three_seeds())
    s1 = cli["by_seed"][0]["by_client"]
    assert s1["sep_local_C1"]["improvement_pct"] == pytest.approx(-21.3, abs=0.1)
    assert s1["sep_local_C2"]["improvement_pct"] == pytest.approx(107.7, abs=0.1)
    assert s1["sep_local_C3"]["improvement_pct"] == pytest.approx(202.6, abs=0.1)
    assert cli["worst_client_by_seed"] == ["sep_local_C1"] * 3
    assert cli["worst_client_stable"] is True
    assert cli["n_seeds_worst_client_loses"] == 3
    assert cli["spread_pct"]["by_seed"][0] == pytest.approx(202.6 + 21.3, abs=0.2)
