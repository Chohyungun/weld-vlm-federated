"""시드 3세트 집계 — 합성 산출물 3벌로 규칙을 고정한다.

지키는 것:
1. **대조선이 시드 사이에 다르면 멈춘다** — 동결본이 바뀐 신호다(24번 §4).
2. **회복률의 평균과 평균의 회복률을 둘 다 낸다** — 다른 수다.
3. **분모 게이트가 사전등록 함수를 실호출한다** — 풀링 sd 로.
4. **부차 지표의 일반식** — 연합÷중앙집중, (연합−로컬평균)÷로컬평균, 로컬 모델별 (연합−로컬_c)÷로컬_c.
   최악 참여자가 평균에 가려지지 않게 따로 낸다.
5. **공통 평가셋 변화와 명목 자기 재질을 구분한다** — 부호가 갈릴 수 있고, 갈리는 것을 산출물이
   말해야 '참여자가 이득을 본다'로 잘못 읽히지 않는다.
6. **채점기 코드 동일성을 주장하지 않는다** — 산출물에 코드 해시가 없다.
"""

from __future__ import annotations

import pytest

from scripts.probe.aggregate_seeds import (
    DET_TAGS,
    check_baselines_identical,
    client_improvement,
    provenance,
    recovery_table,
    secondary_ratios,
)

BASE = {"classification_axis": {"macro_ap_freq": 0.95}, "position_axis": {"constant_box_map_50": 0.001},
        "primary_rule": "idq512", "self_check_reproduced": True}


def _payload(seed: int, floor: dict[str, float], op: dict[str, float] | None = None,
             al: dict[str, float] | None = None) -> dict:
    op = op or {t: 0.5 for t in DET_TAGS}
    # 재질 분해 — AL 은 C3 단독, ST 는 C1∪C2 라 분리 불가
    al = al or {t: floor[t] for t in DET_TAGS}
    return {
        "params": {"seed": seed, "profile": "main", "conf_floor": 0.01},
        "scorer": "evaluation.score.score_records (단일)",
        "decomposition": {
            "axis": "material",
            "client_mapping": {"AL": "C3 (알루미늄 단독 클라이언트)",
                               "ST": "C1 ∪ C2 (강재 두 클라이언트, 평가셋에서 분리 불가)"},
            "by_group": {
                "AL": {"n_images": 1786, "per_tag": {t: {"map_50": al[t]} for t in DET_TAGS}},
                "ST": {"n_images": 10675, "per_tag": {t: {"map_50": floor[t]} for t in DET_TAGS}},
            },
        },
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


def test_분모_게이트가_세_정의로_각각_판정한다():
    """사전등록 문면이 '시드 sd' 를 정의하지 않았다 — 하나를 골라 싣지 않는다."""
    g = recovery_table(three_seeds(), {})["map_50"]["denominator_gate"]
    tw = g["tripwire_by_definition"]
    assert set(tw) == {"pooled_df10", "max_per_cell_df2", "denominator_sd_df2"}
    pooled = g["seed_sd_pooled"]
    assert pooled["ci"][0] < pooled["sd"] < pooled["ci"][1]
    assert tw["pooled_df10"]["pass"] is True and tw["max_per_cell_df2"]["pass"] is True
    assert g["denominator"] > tw["max_per_cell_df2"]["threshold_3sd"]
    assert "분모 D 자체" in tw["denominator_sd_df2"]["seed_sd_definition"]
    # 하위 필드의 recovery_reportable 이 종합 판정으로 읽히지 않게 단서를 단다
    assert "트립와이어 한 줄" in tw["denominator_sd_df2"]["field_caveat"]


def test_분모가_시드_잡음_안이면_게이트가_막는다():
    """중앙과 로컬이 거의 같고 시드 잡음이 크면 회복률을 헤드라인으로 싣지 않는다."""
    near = {"sep_central": 0.30, "sep_local_C1": 0.29, "sep_local_C2": 0.30, "sep_local_C3": 0.31,
            "sep_fed": 0.30}
    p = {1: _payload(1, near),
         2: _payload(2, {t: v + 0.03 for t, v in near.items()}),
         3: _payload(3, {t: v - 0.03 for t, v in near.items()})}
    tw = recovery_table(p, {})["map_50"]["denominator_gate"]["tripwire_by_definition"]
    assert tw["pooled_df10"]["pass"] is False and tw["max_per_cell_df2"]["pass"] is False


def test_부차_지표_일반식_검산():
    """시드 1 실측 구조에서 연합÷중앙집중 57.2%, 개별학습 대비 +44.0% 가 나와야 한다."""
    sec = secondary_ratios(three_seeds())
    s1 = sec["by_seed"][0]
    assert s1["fed_over_central_pct"] == pytest.approx(57.25, abs=0.05)
    assert s1["improvement_over_local_mean_pct"] == pytest.approx(44.0, abs=0.1)
    assert sec["definitions"]["fed_over_central_pct"] == "연합 ÷ 중앙집중 × 100"


def test_공통_평가셋_변화와_최악_참여자():
    """C1 −21.3% / C2 +107.7% / C3 +202.6% — 평균이 양수여도 C1 은 내려간다."""
    cli = client_improvement(three_seeds())
    s1 = cli["by_seed"][0]["by_client"]
    assert s1["sep_local_C1"]["change_on_common_eval_pct"] == pytest.approx(-21.3, abs=0.1)
    assert s1["sep_local_C2"]["change_on_common_eval_pct"] == pytest.approx(107.7, abs=0.1)
    assert s1["sep_local_C3"]["change_on_common_eval_pct"] == pytest.approx(202.6, abs=0.1)
    assert cli["worst_client_by_seed"] == ["sep_local_C1"] * 3
    assert cli["worst_client_stable"] is True
    assert cli["n_seeds_worst_client_down"] == 3
    assert cli["spread_pct"]["by_seed"][0] == pytest.approx(202.6 + 21.3, abs=0.2)


def test_참여자_편익이_아니라고_산출물이_말한다():
    """분모·분자가 모두 공통 평가셋에서 나온 값이다 — 그 사실이 필드로 남아야 한다."""
    cli = client_improvement(three_seeds())
    assert cli["measures"].startswith("공통 평가셋")
    assert "편익이 아니다" in cli["does_not_measure"]


def test_자기_재질_부호가_공통과_갈리면_표시한다():
    """C3 는 공통에서 오르고 자기 재질(AL)에서 내려간다 — 둘 다 실려야 한다."""
    # AL 에서는 C3 로컬이 강하고 연합이 약하다
    al = {"sep_central": 0.49, "sep_local_C1": 0.08, "sep_local_C2": 0.07,
          "sep_local_C3": 0.48, "sep_fed": 0.23}
    p = {n: _payload(n, S1, al=al) for n in (1, 2, 3)}
    cli = client_improvement(p)
    c3 = cli["by_client"]["sep_local_C3"]
    assert c3["change_on_common_eval_pct"]["mean"] > 0
    assert c3["change_on_own_material_pct"]["mean"] == pytest.approx(-52.1, abs=0.5)
    assert c3["sign_disagrees_with_common"] is True
    assert cli["clients_with_sign_disagreement"] == ["sep_local_C3"]


def test_재질이_여러_클라이언트에_걸리면_자기_재질을_내지_않는다():
    """ST = C1 ∪ C2 — 평가셋에 클라이언트 열이 없어 분리할 수 없다(불변조건 1-3)."""
    cli = client_improvement(three_seeds())
    for t in ("sep_local_C1", "sep_local_C2"):
        assert cli["by_client"][t]["change_on_own_material_pct"] is None
        assert "분리 불가" in cli["by_client"][t]["own_material_note"]


def test_출처_블록이_코드_동일성을_주장하지_않는다(tmp_path):
    """산출물에 채점기 코드 해시가 없다 — 그 사실을 블록이 스스로 말해야 한다."""
    prov = provenance(tmp_path, three_seeds())
    assert prov["params_identical_across_seeds"] is True
    assert "확인되지 않는다" in prov["not_verified_here"]
    # 파일이 없으면 해시는 None 이고, 없는 것을 있는 것처럼 적지 않는다
    assert all(v["sha256"] is None for v in prov["inputs"].values())


def test_분모_판정이_셋으로_갈라진다():
    """트립와이어 충족 / 회복률 CI 미산출 / 대표 채택 미결 — 종합 통과로 읽히면 안 된다."""
    v = recovery_table(three_seeds(), {})["map_50"]["denominator_gate"]["verdict"]
    assert v["1_tripwire_3sigma"] == "충족"
    assert "미산출" in v["2_recovery_ci"]
    assert "미결" in v["3_headline_adoption"]
    assert "종합 검증 통과가 아니다" in v["combined"]


def test_모델별_sd_가_풀링보다_먼저_실린다():
    """풀링은 분산 동일 가정이 들어간 보조 요약이다 — 가정을 함께 적는다."""
    g = recovery_table(three_seeds(), {})["map_50"]["denominator_gate"]
    assert set(g["seed_sd_by_cell"]) == set(DET_TAGS)
    assert all(v["n_seeds"] == 3 and v["df"] == 2 for v in g["seed_sd_by_cell"].values())
    assert "가정" in g["seed_sd_pooled"]["assumption"]
    assert g["seed_sd_pooled"]["role"].startswith("보조")
    # 가정 없는 보수적 정의로도 판정한다
    assert "max_per_cell_df2" in g["tripwire_by_definition"]
