"""시드 3세트 집계 — 합성 산출물 3벌로 규칙을 고정한다.

지키는 것:
1. **대조선이 시드 사이에 다르면 멈춘다** — 동결본이 바뀐 신호다(24번 §4).
2. **회복률의 평균과 평균의 회복률을 둘 다 낸다** — 다른 수다.
3. **분모 게이트가 사전등록 함수를 실호출한다** — 세 sd 정의로 각각. 판정은 셋으로 갈라
   적고, 트립와이어 단독 결과(`tripwire_only_reportable`)와 게재 가부(`recovery_reportable`,
   CI 산출·통과 뒤에만 true, 그 전에는 null)를 섞지 않는다.
4. **부차 지표의 일반식** — 연합÷중앙집중, (연합−로컬평균)÷로컬평균, 로컬 모델별 (연합−로컬_c)÷로컬_c.
   최악 참여자가 평균에 가려지지 않게 따로 낸다.
5. **공통 평가셋 변화와 명목 자기 재질을 구분한다** — 부호가 갈릴 수 있고, 갈리는 것을 산출물이
   말해야 '참여자가 이득을 본다'로 잘못 읽히지 않는다.
6. **채점기 코드 해시를 대조한다** — 다르면 멈추고, 없으면 "독립 확인 불가"를 사실대로 적는다.
   소급 계산해 끼워 넣지 않는다.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.probe.aggregate_seeds import (
    DET_TAGS,
    apply_recovery_ci,
    check_baselines_identical,
    client_improvement,
    provenance,
    recovery_table,
    scorer_code_check,
    secondary_ratios,
)

REPO = Path(__file__).resolve().parents[1]

BASE = {"classification_axis": {"macro_ap_freq": 0.95}, "position_axis": {"constant_box_map_50": 0.001},
        "primary_rule": "idq512", "self_check_reproduced": True}


def _rec_pct(v: dict[str, float]) -> float | None:
    lm = (v["sep_local_C1"] + v["sep_local_C2"] + v["sep_local_C3"]) / 3
    d = v["sep_central"] - lm
    return None if d <= 0 else 100.0 * (v["sep_fed"] - lm) / d


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
            "limitation": "픽스처 — 평가셋에 client 열이 없어 재질이 유일한 귀속 축",
            "client_mapping": {"AL": "C3 (알루미늄 단독 클라이언트)",
                               "ST": "C1 ∪ C2 (강재 두 클라이언트, 평가셋에서 분리 불가)"},
            "by_group": {
                "AL": {"n_images": 1786, "per_tag": {t: {"map_50": al[t]} for t in DET_TAGS},
                       "recovery": {"map_50": {"recovery_pct": _rec_pct(al)}}},
                "ST": {"n_images": 10675, "per_tag": {t: {"map_50": floor[t]} for t in DET_TAGS},
                       "recovery": {"map_50": {"recovery_pct": _rec_pct(floor)}}},
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
    # 총괄 확정 정책(35번 과제 3): 트립와이어 단독 결과와 게재 가부를 가른다.
    dsd = tw["denominator_sd_df2"]
    assert dsd["tripwire_only_reportable"] is True
    assert dsd["ci_pass"] is None
    assert dsd["recovery_reportable"] is None, "CI 미산출이면 미판정(null)이어야 한다"
    assert "CI 산출·통과 뒤에만" in dsd["recovery_reportable_policy"]
    assert "미판정" in dsd["field_caveat"]


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


def _with_code(payloads, digests: dict[int, str]):
    for n, d in digests.items():
        payloads[n]["scorer_code"] = {"combined": d, "n_files": 31}
    return payloads


def test_해시가_없으면_멈추지_않고_확인_불가로_적는다(tmp_path):
    """규칙이 생기기 전 산출물이다. 소급 계산해 끼워 넣지 않기로 했으니 멈추면 안 된다."""
    prov = provenance(tmp_path, three_seeds())
    sc = prov["scorer_code"]
    assert prov["params_identical_across_seeds"] is True
    assert sc["verified_same_code"] is False
    assert "독립 확인 불가" in sc["status"] and "소급" in sc["detail"]
    assert sc["seeds_missing_hash"] == ["1", "2", "3"]
    # 파일이 없으면 해시는 None 이고, 없는 것을 있는 것처럼 적지 않는다
    assert all(v["sha256"] is None for v in prov["inputs"].values())


def test_해시가_전부_같으면_확인됨():
    out = scorer_code_check(_with_code(three_seeds(), dict.fromkeys((1, 2, 3), "a" * 64)))
    assert out["verified_same_code"] is True
    assert out["n_distinct"] == 1 and out["seeds_missing_hash"] == []


def test_해시가_다르면_멈춘다():
    """다른 코드로 낸 값을 한 표에 모으는 것이 문제의 시작이다."""
    p = _with_code(three_seeds(), {1: "a" * 64, 2: "a" * 64, 3: "b" * 64})
    with pytest.raises(SystemExit, match="한 표에 모으지 않는다"):
        scorer_code_check(p)


def test_의도한_차이는_기록을_남기고_진행한다():
    """조용히 넘어가는 경로는 두지 않는다 — 플래그를 줘야 하고 산출물에 남는다."""
    p = _with_code(three_seeds(), {1: "a" * 64, 2: "a" * 64, 3: "b" * 64})
    out = scorer_code_check(p, allow_drift=True)
    assert out["status"] == "불일치" and out["drift_allowed"] is True
    assert out["verified_same_code"] is False and out["n_distinct"] == 2


def test_일부만_해시가_있으면_확인되지_않는다():
    """있는 것끼리 같아도 전 시드 대조가 아니다 — 부분 정보라고 적는다."""
    p = three_seeds()
    p[1]["scorer_code"] = {"combined": "a" * 64}
    p[2]["scorer_code"] = {"combined": "a" * 64}
    out = scorer_code_check(p)
    assert out["verified_same_code"] is False
    assert out["seeds_missing_hash"] == ["3"] and "부분 정보" in out["detail"]


def test_분모_판정이_셋으로_갈라진다():
    """트립와이어 충족 / 회복률 CI 미산출 / 대표 채택 미결 — 종합 통과로 읽히면 안 된다."""
    v = recovery_table(three_seeds(), {})["map_50"]["denominator_gate"]["verdict"]
    assert v["1_tripwire_3sigma"] == "충족"
    # ①은 세 정의 **전부**의 결과를 함께 싣는다 — 둘만 보고 "충족"을 내지 않는다
    assert set(v["1_tripwire_by_definition"]) == {"pooled_df10", "max_per_cell_df2",
                                                  "denominator_sd_df2"}
    assert all(v["1_tripwire_by_definition"].values())
    assert "미산출" in v["2_recovery_ci"]
    assert "미결" in v["3_headline_adoption"]
    assert "종합 검증 통과가 아니다" in v["combined"]


def test_한_정의만_실패하면_정의에_따라_갈림():
    """세 정의 중 하나라도 갈리면 "충족"이 아니다 — 정의 선택이 판정을 가르는 상태를
    숨기지 않는다(35번 과제 4). 연합 칸의 시드 분산만 크게 해 칸별 최대 sd 정의는 막히고
    풀링·분모 sd 정의는 통과하는 픽스처다."""
    base = dict(S1)
    p = {n: _payload(n, base) for n in (1, 2, 3)}
    for n, fed in ((1, 0.20), (2, 0.30), (3, 0.40)):       # sd 0.10 → 3·sd 0.30 > D 0.297
        p[n]["threshold_independent"]["per_tag"]["sep_fed"]["map_50"] = fed
    g = recovery_table(p, {})["map_50"]["denominator_gate"]
    tw = g["tripwire_by_definition"]
    assert tw["max_per_cell_df2"]["pass"] is False
    assert tw["pooled_df10"]["pass"] is True
    assert tw["denominator_sd_df2"]["tripwire_3sigma_pass"] is True
    v = g["verdict"]
    assert v["1_tripwire_3sigma"] == "정의에 따라 갈림"
    assert v["1_tripwire_by_definition"] == {"pooled_df10": True, "max_per_cell_df2": False,
                                              "denominator_sd_df2": True}


def test_세_정의_전부_실패하면_불충족():
    near = {"sep_central": 0.30, "sep_local_C1": 0.29, "sep_local_C2": 0.30, "sep_local_C3": 0.31,
            "sep_fed": 0.30}
    p = {1: _payload(1, near),
         2: _payload(2, {t: v + 0.03 for t, v in near.items()}),
         3: _payload(3, {t: v - 0.03 for t, v in near.items()})}
    g = recovery_table(p, {})["map_50"]["denominator_gate"]
    assert g["verdict"]["1_tripwire_3sigma"] == "불충족"
    assert not any(g["verdict"]["1_tripwire_by_definition"].values())


def test_풀링_가정에_짝지음_문장이_있다():
    """같은 시드로 학습한 다섯 모델 값은 짝지어져 있다 — 독립 표본 가정도 성립하지 않는다."""
    g = recovery_table(three_seeds(), {})["map_50"]["denominator_gate"]
    a = g["seed_sd_pooled"]["assumption"]
    assert "짝지어져" in a and "독립 표본" in a and "자유도" in a


def test_모델별_sd_가_풀링보다_먼저_실린다():
    """풀링은 분산 동일 가정이 들어간 보조 요약이다 — 가정을 함께 적는다."""
    g = recovery_table(three_seeds(), {})["map_50"]["denominator_gate"]
    assert set(g["seed_sd_by_cell"]) == set(DET_TAGS)
    assert all(v["n_seeds"] == 3 and v["df"] == 2 for v in g["seed_sd_by_cell"].values())
    assert "가정" in g["seed_sd_pooled"]["assumption"]
    assert g["seed_sd_pooled"]["role"].startswith("보조")
    # 가정 없는 보수적 정의로도 판정한다
    assert "max_per_cell_df2" in g["tripwire_by_definition"]


# ======================================================================================
# C 34번 Important 4 — 변이 5개를 전부 잡는 픽스처
#   m1 평균의 회복률 ↔ 회복률의 평균 치환 · m2 miss_rate 부호 반전 · m3 sd 무시 ·
#   m4 풀링 sd ×2 · m5 대조선 검사가 위치 축을 건너뜀
# ======================================================================================

# 비례가 아닌 세 시드 — 두 회복률이 다르다. 손계산 값을 함께 둔다.
NP = {
    1: {"sep_central": 0.50, "sep_local_C1": 0.30, "sep_local_C2": 0.10, "sep_local_C3": 0.20,
        "sep_fed": 0.30},   # lm 0.20 · D 0.30 · R (0.30−0.20)/0.30 = 0.3333
    2: {"sep_central": 0.40, "sep_local_C1": 0.20, "sep_local_C2": 0.10, "sep_local_C3": 0.15,
        "sep_fed": 0.35},   # lm 0.15 · D 0.25 · R 0.20/0.25 = 0.8000
    3: {"sep_central": 0.60, "sep_local_C1": 0.40, "sep_local_C2": 0.20, "sep_local_C3": 0.30,
        "sep_fed": 0.31},   # lm 0.30 · D 0.30 · R 0.01/0.30 = 0.0333
}


def nonprop() -> dict[int, dict]:
    return {n: _payload(n, v) for n, v in NP.items()}


def test_m1_회복률의_평균과_평균의_회복률이_다른_픽스처에서_각각_손계산과_같다():
    rec = recovery_table(nonprop(), {})["map_50"]
    by = [p["recovery_pct"] for p in rec["by_seed"]]
    assert by == pytest.approx([33.333, 80.0, 3.333], abs=0.01)
    assert rec["mean_of_seed_recoveries_pct"] == pytest.approx((33.333 + 80.0 + 3.333) / 3, abs=0.01)
    # 평균의 회복률: 중앙 0.50 · 연합 0.32 · 로컬평균 0.21667 → (0.32−0.21667)/(0.28333) = 36.47 %
    assert rec["recovery_of_seed_means_pct"] == pytest.approx(36.47, abs=0.02)
    assert rec["mean_of_seed_recoveries_pct"] != pytest.approx(rec["recovery_of_seed_means_pct"], abs=1.0)


@pytest.mark.parametrize("fed_better", [True, False])
def test_m2_놓침은_낮을수록_좋다_연합이_로컬평균보다_좋은_경우와_나쁜_경우(fed_better):
    """miss_rate 는 부호를 뒤집어 잰다 — 연합이 로컬평균보다 **덜** 놓치면 회복률이 양수다."""
    op = {"sep_central": 0.60, "sep_local_C1": 0.40, "sep_local_C2": 0.40, "sep_local_C3": 0.40,
          "sep_fed": 0.55 if fed_better else 0.35}
    # _payload 는 miss_rate = 1 − op 로 만든다 → 중앙 0.40 · 로컬 0.60 · 연합 0.45 / 0.65
    p = {n: _payload(n, S1, op=op) for n in (1, 2, 3)}
    rec = recovery_table(p, {})["miss_rate"]
    r = rec["by_seed"][0]
    assert r["central"] == pytest.approx(0.40) and r["local_mean"] == pytest.approx(0.60)
    assert r["denominator"] == pytest.approx(0.20)          # 부호 반전 후 양수
    expected = ((0.60 - 0.45) / 0.20 * 100) if fed_better else ((0.60 - 0.65) / 0.20 * 100)
    assert r["recovery_pct"] == pytest.approx(expected, abs=1e-9)
    assert (r["recovery_pct"] > 0) is fed_better


def test_m4_풀링_sd_와_카이제곱_CI_를_수치로_대조():
    """풀링 sd = sqrt(Σ(n_i−1)s_i² / Σ(n_i−1)). 손계산과 맞아야 한다 — ×2 변이는 여기서 죽는다."""
    import statistics as st

    from scipy import stats as sps

    p = nonprop()
    g = recovery_table(p, {})["map_50"]["denominator_gate"]
    vals = {t: [NP[n][t] for n in (1, 2, 3)] for t in DET_TAGS}
    ss = sum(st.variance(v) * 2 for v in vals.values())
    pooled = math.sqrt(ss / 10)
    assert g["seed_sd_pooled"]["sd"] == pytest.approx(pooled, rel=1e-12)
    assert g["seed_sd_pooled"]["df"] == 10
    lo = pooled * math.sqrt(10 / sps.chi2.ppf(0.975, 10))
    hi = pooled * math.sqrt(10 / sps.chi2.ppf(0.025, 10))
    assert g["seed_sd_pooled"]["ci"] == pytest.approx([lo, hi], rel=1e-12)
    for t in DET_TAGS:
        assert g["seed_sd_by_cell"][t]["sd"] == pytest.approx(st.stdev(vals[t]), rel=1e-12)
    assert g["tripwire_by_definition"]["max_per_cell_df2"]["seed_sd"] == pytest.approx(
        max(st.stdev(v) for v in vals.values()), rel=1e-12)


def test_m3_근접_픽스처에서_sd_가_실제로_막는다():
    """0 < D < 3·sd 구간 — sd 인자를 0 으로 바꾸는 변이는 여기서 죽는다."""
    base = {"sep_central": 0.310, "sep_local_C1": 0.300, "sep_local_C2": 0.300,
            "sep_local_C3": 0.300, "sep_fed": 0.305}         # D = 0.010
    p = {1: _payload(1, base),
         2: _payload(2, {t: v + 0.006 for t, v in base.items()}),
         3: _payload(3, {t: v - 0.006 for t, v in base.items()})}   # 칸별 sd 0.006 → 3·sd 0.018 > D
    g = recovery_table(p, {})["map_50"]["denominator_gate"]
    assert g["denominator"] == pytest.approx(0.010)
    tw = g["tripwire_by_definition"]
    assert tw["max_per_cell_df2"]["pass"] is False
    assert tw["pooled_df10"]["pass"] is False
    # 세 시드가 같은 방향으로 움직이면 분모 D 자체의 시드 sd 는 0 이라 그 정의만 통과한다 —
    # 정의가 갈리는 상태를 "갈림" 으로 드러내는 것이 설계다. sd 를 0 으로 바꾸는 변이는
    # 위 두 단언에서 죽는다.
    assert tw["denominator_sd_df2"]["tripwire_3sigma_pass"] is True
    assert g["verdict"]["1_tripwire_3sigma"] == "정의에 따라 갈림"


@pytest.mark.parametrize("axis", ["classification_axis", "position_axis"])
def test_m5_대조선은_두_축_각각_흔들어도_멈춘다(axis):
    p = three_seeds()
    changed = dict(BASE[axis])
    k = next(iter(changed))
    changed[k] = changed[k] + 1e-6
    p[3]["content_free_baseline"] = {**BASE, axis: changed}
    with pytest.raises(SystemExit, match="동결본"):
        check_baselines_identical(p)


# ======================================================================================
# Minor 8 — 시드 2개 미만이면 미판정
# ======================================================================================

def test_시드가_하나면_트립와이어는_미판정():
    g = recovery_table({1: _payload(1, S1)}, {})["map_50"]["denominator_gate"]
    assert g["verdict"]["1_tripwire_3sigma"].startswith("미판정")
    assert g["tripwire_by_definition"]["pooled_df10"]["pass"] is None
    assert g["tripwire_by_definition"]["denominator_sd_df2"]["recovery_reportable"] is None


# ======================================================================================
# T5 — 회복률 CI 배선 정책: CI 없음 → null · 통과 → true · 미달 → false
# ======================================================================================

def _ci_payload(half_width: float, d_lo: float = 0.2, undefined: int = 0) -> dict:
    return {
        "point": {"R_bar": 0.25},
        "statistic": {"n_resamples": 2000, "n_groups": 1245},
        "ci": {"R_bar": {"ci_lo": 0.25 - half_width, "ci_hi": 0.25 + half_width,
                         "half_width": half_width, "n_undefined": undefined},
               "D_by_seed": {"1": {"ci_lo": d_lo}, "2": {"ci_lo": d_lo}, "3": {"ci_lo": d_lo}}},
        "verdict": {
            "2_half_width": {"pass": half_width <= 0.25},
            "3_denominator_ci_excludes_zero": {"pass": d_lo > 0},
            "4_undefined_fraction": {"value": undefined / 2000, "pass": undefined / 2000 <= 0.10},
            "ci_pass": (half_width <= 0.25) and (d_lo > 0) and (undefined / 2000 <= 0.10),
        },
        "registration": {"status": "픽스처"}, "seed_caveat": "픽스처",
        "inputs": {"artifact_name": "score_cells_v1.json"},
    }


def test_CI_없으면_recovery_reportable_은_null():
    rec = apply_recovery_ci(recovery_table(three_seeds(), {}), None)
    g = rec["map_50"]["denominator_gate"]
    assert g["recovery_reportable"] is None
    assert "미산출" in g["verdict"]["2_recovery_ci"]


def test_CI_통과면_true():
    rec = apply_recovery_ci(recovery_table(three_seeds(), {}), _ci_payload(0.10))
    g = rec["map_50"]["denominator_gate"]
    assert g["recovery_reportable"] is True
    assert "통과" in g["verdict"]["2_recovery_ci"]
    assert g["recovery_ci"]["interval"]["ci_lo"] == pytest.approx(0.15)


@pytest.mark.parametrize("bad", [
    {"half_width": 0.30},                       # ② 반폭 초과
    {"half_width": 0.10, "d_lo": -0.01},        # ③ 분모 CI 가 0 을 품음
    {"half_width": 0.10, "undefined": 300},     # ④ 미정의 15 %
])
def test_규칙_하나라도_미달이면_false(bad):
    rec = apply_recovery_ci(recovery_table(three_seeds(), {}), _ci_payload(**bad))
    assert rec["map_50"]["denominator_gate"]["recovery_reportable"] is False


def test_트립와이어가_불충족이면_CI_가_통과해도_false():
    near = {"sep_central": 0.30, "sep_local_C1": 0.29, "sep_local_C2": 0.30, "sep_local_C3": 0.31,
            "sep_fed": 0.30}
    p = {1: _payload(1, near), 2: _payload(2, {t: v + 0.03 for t, v in near.items()}),
         3: _payload(3, {t: v - 0.03 for t, v in near.items()})}
    rec = apply_recovery_ci(recovery_table(p, {}), _ci_payload(0.10))
    assert rec["map_50"]["denominator_gate"]["recovery_reportable"] is False


# ======================================================================================
# Minor 6·Important 4 — main() 을 실제로 돈다 (D ≤ 0 포함). 집계본이 먼저 써져야 한다
# ======================================================================================

def _write_seed_dir(root, n, floor, **kw):
    d = root / f"seed{n}"
    d.mkdir(parents=True, exist_ok=True)
    p = _payload(n, floor, **kw)
    p.update({
        "curve": {"grid": [0.01, 0.25, 0.5]}, "exit_code": 0,
        "gates_evaluated": {"blocking_failures": [], "ok": True, "n_evaluated": 11},
        "discrimination": {"results": [
            {"cell": t.split("_")[0] + ("_" + "_".join(t.split("_")[1:]) if False else ""),
             "client": None, "delta": {"point": 0.3, "ci_lo": 0.2, "ci_hi": 0.4}}
            for t in DET_TAGS]},
        "discrimination_threshold_free": {"by_cell": {t: {"point": 0.5, "ci_lo": 0.4, "ci_hi": 0.6}
                                                      for t in DET_TAGS},
                                          "contrasts": {}, "n_defect": 7086, "n_normal": 654},
        "stratified": {"default_k": 64, "by_k": {"64": {**{t: {"stratified_lift": -0.01} for t in DET_TAGS},
                                                          "__shortcut__": {"stratified_lift": 0.0}}}},
        "p9": {"results": [], "all_equivalent": False},
    })
    # discrimination.results 의 cell/client 를 실제 규약대로
    p["discrimination"]["results"] = [
        {"cell": "sep_local" if t.startswith("sep_local") else t,
         "client": t.split("_")[-1] if t.startswith("sep_local") else None,
         "delta": {"point": 0.3, "ci_lo": 0.2, "ci_hi": 0.4}} for t in DET_TAGS]
    (d / "score_cells_v1.json").write_text(json.dumps(p, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("degenerate", [False, True])
def test_main_이_실제로_돌고_집계본이_먼저_써진다(tmp_path, degenerate):
    """D ≤ 0 인 시드가 있어도 집계본은 써진다(C 34번 Minor 6) — 회복률 무관 표까지 잃지 않는다."""
    import os

    root = tmp_path / "main_d"
    for n in (1, 2, 3):
        floor = dict(S1)
        if degenerate and n == 2:
            floor["sep_central"] = 0.10          # 로컬평균 0.196 > 중앙 → D < 0
        _write_seed_dir(root, n, floor)
    dest = tmp_path / "seed3set"
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/probe/aggregate_seeds.py"), "--root", str(root),
         "--seeds", "1,2,3", "--dest", str(dest)],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=REPO, check=False,
    )
    assert (dest / "aggregate_v1.json").exists(), proc.stderr[-800:]
    assert proc.returncode == 0, proc.stderr[-800:]
    agg = json.loads((dest / "aggregate_v1.json").read_text(encoding="utf-8"))
    rec = agg["recovery"]["map_50"]
    if degenerate:
        assert rec["by_seed"][1]["recovery_pct"] is None
        assert rec["n_seeds_recovery_defined"] == 2
        assert "미정의" in proc.stdout
    assert agg["recovery"]["map_50"]["denominator_gate"]["recovery_reportable"] is None
    assert agg["local_mean_definition"].startswith("로컬 평균")
