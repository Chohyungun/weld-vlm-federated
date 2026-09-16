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
7. **CI 를 입력에 묶는다(codex_reply §18)** — 시드 집합·시드별 산출물 sha256·채점기 지문이 하나라도
   다르면 CI 를 쓰지 않는다(null). 기록된 `ci_pass` 는 믿지 않고 집계기가 다시 판정한다.
8. **C 42번 후속** — 신뢰수준·구간 방식도 등록 조건이다(I-1). 집계본·CI 이름은 입력 산출물의 판에서
   뽑고 옛 판을 덮지 않는다(I-2). CI 픽스처의 점추정은 검사 대상 함수가 아니라 생성기와 같은 계산으로
   만든다(m-4).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.recovery_ci import make_binding, recovery_from
from scripts.probe.aggregate_seeds import (
    DET_TAGS,
    PUBLIC_STATUS,
    apply_recovery_ci,
    check_baselines_identical,
    client_improvement,
    observed_points,
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


LOCAL_TAGS = ("sep_local_C1", "sep_local_C2", "sep_local_C3")


def _records(seed: int) -> dict[str, str]:
    """채점 산출물 `input_records` 모양 — 운용점 레코드 5개 + 원시(하한) 레코드 5개."""
    out = {}
    for t in DET_TAGS:
        for kind, path in (("op", f"outputs/main_d/seed{seed}/{t}_s{seed}.jsonl"),
                           ("raw", f"outputs/main_d/seed{seed}/sweep/{t}_raw_s{seed}.jsonl")):
            out[path] = hashlib.sha256(f"{kind}{seed}{t}".encode()).hexdigest()
    return out


def _payload(seed: int, floor: dict[str, float], op: dict[str, float] | None = None,
             al: dict[str, float] | None = None) -> dict:
    op = op or {t: 0.5 for t in DET_TAGS}
    # 재질 분해 — AL 은 C3 단독, ST 는 C1∪C2 라 분리 불가
    al = al or {t: floor[t] for t in DET_TAGS}
    return {
        "params": {"seed": seed, "profile": "main", "conf_floor": 0.01},
        "scorer": "evaluation.score.score_records (단일)",
        "input_records": _records(seed),
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
    # 게재 가부는 이 블록에 없다(49번 §5) — 최상위 한 곳에만 둔다
    for k in ("ci_pass", "recovery_reportable", "recovery_reportable_policy"):
        assert k not in dsd, k
    assert "최상위" in dsd["field_caveat"] and "지금은" not in dsd["field_caveat"]
    assert "게재 가능이 아니다" in dsd["field_caveat"]      # 트립와이어 단독 결과를 게재 가능으로 읽지 않게


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
    dsd = g["tripwire_by_definition"]["denominator_sd_df2"]
    assert dsd["tripwire_3sigma_pass"] is None and dsd["tripwire_only_reportable"] is None
    assert "recovery_reportable" not in dsd


# ======================================================================================
# T5 — 회복률 CI 배선 정책 + 입력 결속 (codex_reply §18)
#   CI 없음 → null · 결속 성립 + 재판정 통과 → true · 재판정 미달 → false
#   결속 불일치(한 바이트·시드 부분집합·지문) · 기록 자기모순 → null
#   기록된 ci_pass 는 믿지 않는다
# ======================================================================================

CODE = {"combined": "c" * 64, "rule": "규칙-픽스처", "newline_normalized": True, "stable": True}
ART = "score_cells_v2.json"
PASSED = "CI 산출 완료, 코드·출처 최종 검수 및 대표 채택 대기"


def _coded(payloads: dict[int, dict], code: dict = CODE) -> dict[int, dict]:
    for p in payloads.values():
        p["scorer_code"] = dict(code)
    return payloads


def _entry(n: int, payload: dict, sha: str | None = None) -> dict:
    """`read_artifact` 가 내는 결속 항목과 같은 모양 — 해시만 지어 넣는다(단위 시험)."""
    sc = payload.get("scorer_code") or {}
    return {"path": f"outputs/main_d/seed{n}/{ART}", "sha256": sha or f"{n:064d}",
            "seed_value": payload["params"]["seed"],
            "scorer_code_combined": sc.get("combined"), "scorer_code_rule": sc.get("rule"),
            "newline_normalized": bool(sc.get("newline_normalized", False)),
            "scorer_code_stable": sc.get("stable"),
            "input_records": payload.get("input_records")}


def _observed(payloads: dict[int, dict], **sha_by_seed) -> dict:
    return make_binding(ART, {n: _entry(n, p, sha_by_seed.get(f"s{n}")) for n, p in payloads.items()})


def _generator_points(payloads) -> dict:
    """**CI 생성기처럼** 채점 파일 값에서 바로 계산한 점추정(C 42번 m-4).

    검사 대상인 `observed_points` 로 기대값을 만들면 그 함수의 결함이 CI 쪽에도 같이 실려 시험이
    못 잡는다. 생성기와 같게 `recovery_from`(np.mean) 을 쓰고, 미정의 시드는 nan, R̄ 는 nan 이
    하나라도 섞이면 nan 이다.
    """
    tpi = {n: p["threshold_independent"]["per_tag"] for n, p in payloads.items()}
    out: dict = {"map": {}, "basis": {}, "R": {}, "D": {}}
    for n in sorted(payloads):
        m = {t: tpi[n][t]["map_50"] for t in DET_TAGS}
        out["map"][n] = m
        out["basis"][n] = {t: (tpi[n][t]["map_50"], tpi[n][t]["map_50_95"]) for t in DET_TAGS}
        out["R"][n], out["D"][n] = recovery_from(m["sep_central"], m["sep_fed"], [m[t] for t in LOCAL_TAGS])
    rs = [out["R"][n] for n in sorted(payloads)]
    out["R_bar"] = float(np.mean(rs)) if all(np.isfinite(rs)) else float("nan")
    return out


def _identity(gp) -> dict:
    """항등 검사 기록 — 칸 × 시드 전부 통과·차 0, 기준값은 채점 파일 값."""
    return {str(s): {t: {"passed": True, "abs_diff_map_50": 0.0, "abs_diff_map_50_95": 0.0,
                         "map_50": m50, "expected_map_50": m50,
                         "map_50_95": m5095, "expected_map_50_95": m5095}
                     for t, (m50, m5095) in per.items()}
            for s, per in gp["basis"].items()}


def _raw_records(payload) -> dict:
    """생성기가 캐시를 만들며 읽는 원시(하한) 레코드의 해시 — 산출물 기록의 부분집합."""
    return {k: v for k, v in (payload.get("input_records") or {}).items() if "_raw_" in k}


def _bound_ci(payloads, observed, *, half_width=0.10, d_lo=0.2, undefined=0, n_res=2000,
              rng=20260825, ci_pass=None, half_width_field=None, code=CODE,
              alpha=0.05, interval="백분위", n_defined=None) -> dict:
    """결속·항등·점추정이 **일관된** CI 산출물. 시험이 한 군데씩 어긋나게 만든다."""
    gp = _generator_points(payloads)
    rbar = gp["R_bar"]
    center = rbar if math.isfinite(rbar) else 0.2
    lo, hi = center - half_width, center + half_width
    real_pass = half_width <= 0.25 and d_lo > 0 and undefined / n_res <= 0.10
    return {
        "binding": json.loads(json.dumps(observed)),
        "scorer_code": dict(code),
        "identity_check": _identity(gp),
        "point": {"R_bar": rbar,
                  "R_by_seed": {str(s): v for s, v in gp["R"].items()},
                  "D_by_seed": {str(s): v for s, v in gp["D"].items()},
                  "map_50_by_seed": {str(s): dict(v) for s, v in gp["map"].items()}},
        "statistic": {"n_resamples": n_res, "n_groups": 1245, "rng_seed": rng,
                      "alpha": alpha, "interval": interval},
        "ci": {"R_bar": {"ci_lo": lo, "ci_hi": hi,
                         "half_width": half_width if half_width_field is None else half_width_field,
                         "n_defined": n_res - undefined if n_defined is None else n_defined,
                         "n_undefined": undefined},
               "D_by_seed": {str(s): {"ci_lo": d_lo} for s in observed["seeds"]}},
        "verdict": {"ci_pass": real_pass if ci_pass is None else ci_pass},
        "registration": {"status": "픽스처"}, "seed_caveat": "픽스처",
        "inputs": {"artifact_name": observed["artifact_name"],
                   "records_by_seed": {str(n): _raw_records(p) for n, p in payloads.items()}},
    }


def _judge(payloads, ci, observed=None):
    rec = recovery_table(payloads, {})
    obs = observed if observed is not None else _observed(payloads)
    rec = apply_recovery_ci(rec, ci, obs, observed_points(rec, payloads))
    return rec["map_50"]["denominator_gate"]


def test_CI_없으면_recovery_reportable_은_null():
    p = _coded(three_seeds())
    g = _judge(p, None)
    assert g["recovery_reportable"] is None
    assert "미산출" in g["verdict"]["2_recovery_ci"]
    assert g["public_status"].startswith("CI 미산출")


def test_결속되고_규칙을_통과하면_true_와_공개_문구():
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs), obs)
    assert g["recovery_reportable"] is True
    assert g["public_status"] == PASSED
    assert "집계기 재판정" in g["verdict"]["2_recovery_ci"]
    assert g["recovery_ci"]["rules_rejudged"]["recorded_agrees"] is True
    assert g["recovery_ci_rejected"] is None


@pytest.mark.parametrize("bad", [
    {"half_width": 0.30},                       # ② 반폭 초과
    {"half_width": 0.10, "d_lo": -0.01},        # ③ 분모 CI 가 0 을 품음
    {"half_width": 0.10, "undefined": 300},     # ④ 미정의 15 %
])
def test_규칙_하나라도_미달이면_false(bad):
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, **bad), obs)
    assert g["recovery_reportable"] is False
    assert "규칙 미달" in g["public_status"]


def test_트립와이어가_불충족이면_CI_가_통과해도_false():
    """분모는 양수(0.010)지만 시드 잡음 3·sd 0.018 에 못 미친다 — CI 가 좁아도 게재 불가."""
    base = {"sep_central": 0.310, "sep_local_C1": 0.300, "sep_local_C2": 0.300,
            "sep_local_C3": 0.300, "sep_fed": 0.305}
    p = _coded({1: _payload(1, base),
                2: _payload(2, {t: v + 0.006 for t, v in base.items()}),
                3: _payload(3, {t: v - 0.006 for t, v in base.items()})})
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs), obs)
    assert g["verdict"]["1_tripwire_3sigma"] != "충족"
    assert g["recovery_reportable"] is False


# ---- 요구 3 — 네 경우 각각에서 true 가 나오지 않는다 --------------------------------

def test_요구3a_채점_파일_한_바이트_변경이면_null():
    """CI 는 시드 2 산출물의 옛 해시로 결속됐고, 지금 읽은 파일은 해시가 다르다."""
    p = _coded(three_seeds())
    ci = _bound_ci(p, _observed(p))
    now = _observed(p, s2="f" * 64)                 # 한 바이트라도 바뀌면 sha256 이 다르다
    g = _judge(p, ci, now)
    assert g["recovery_reportable"] is None
    assert any("시드 2 의 sha256" in r for r in g["recovery_ci_rejected"]["reasons"])
    assert "결속되지 않음" in g["public_status"]


def test_요구3b_시드_부분집합이면_null():
    """CI 는 1·2·3 으로 냈는데 1·2 만 집계한다 — 시드 조합이 다른 CI 를 쓰지 않는다."""
    full = _coded(three_seeds())
    ci = _bound_ci(full, _observed(full))
    sub = {n: full[n] for n in (1, 2)}
    g = _judge(sub, ci, _observed(sub))
    assert g["recovery_reportable"] is None
    assert any("시드 집합" in r for r in g["recovery_ci_rejected"]["reasons"])


@pytest.mark.parametrize("where", ["ci_run_code", "artifact_code", "rule", "newline", "unstable"])
def test_요구3c_지문_불일치면_null(where):
    p = _coded(three_seeds())
    obs = _observed(p)
    code = dict(CODE)
    if where == "ci_run_code":                       # CI 를 계산한 코드가 채점 코드와 다르다
        code["combined"] = "d" * 64
    elif where == "rule":
        code["rule"] = "다른 규칙"
    elif where == "newline":
        code["newline_normalized"] = False
    elif where == "unstable":
        code["stable"] = False
    ci = _bound_ci(p, obs, code=code)
    if where == "artifact_code":                     # 산출물 지문이 CI 기록 이후 바뀌었다
        p = _coded(three_seeds(), {**CODE, "combined": "e" * 64})
        obs = _observed(p)
    g = _judge(p, ci, obs)
    assert g["recovery_reportable"] is None, where
    assert g["recovery_ci_rejected"]["reasons"]


def test_요구3d_ci_pass_위조는_믿지_않는다():
    """기록은 ci_pass=true 인데 구간 반폭이 0.30 — 집계기가 다시 판정해 false 를 낸다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, half_width=0.30, ci_pass=True), obs)
    assert g["recovery_reportable"] is False
    rj = g["recovery_ci"]["rules_rejudged"]
    assert rj["recorded_ci_pass"] is True and rj["ci_pass"] is False
    assert rj["recorded_agrees"] is False


# ---- 보조 방어 ---------------------------------------------------------------------

def test_반폭_필드를_구간과_다르게_위조하면_null():
    """구간은 넓은데 반폭 필드만 좁게 적어도 통과하지 않는다 — 반폭은 구간에서 다시 계산한다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, half_width=0.30, half_width_field=0.05, ci_pass=True), obs)
    assert g["recovery_reportable"] is None
    assert any("자기모순" in r for r in g["recovery_ci_rejected"]["reasons"])


@pytest.mark.parametrize("kw", [{"n_res": 100}, {"rng": 1}, {"undefined": 2500}])
def test_등록되지_않은_재표집_조건이나_불가능한_미정의_수면_null(kw):
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, **kw), obs)
    assert g["recovery_reportable"] is None


def test_결속_블록이_없는_구판_CI_는_쓰지_않는다():
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    del ci["binding"]
    g = _judge(p, ci, obs)
    assert g["recovery_reportable"] is None
    assert "binding" in g["recovery_ci_rejected"]["reasons"][0]


# ---- 요구 6 — 지금 읽은 채점 파일로 계산한 점추정과 CI 기록이 다르면 null ----------

@pytest.mark.parametrize("field,seed,label", [
    ("R_by_seed", "2", "시드 2 회복률"),
    ("D_by_seed", "3", "시드 3 분모"),
])
def test_요구6_시드별_점추정이_다르면_null(field, seed, label):
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    ci["point"][field][seed] += 1e-6           # 검출 한 건(≈1e-5)보다 작은 변화도 걸린다
    g = _judge(p, ci, obs)
    assert g["recovery_reportable"] is None
    assert any(label in r for r in g["recovery_ci_rejected"]["reasons"])


def test_요구6_해시가_같아도_점추정은_따로_본다():
    """결속(해시·시드·지문)이 전부 맞아도 점추정이 어긋나면 쓰지 않는다 — 서로 대신하지 않는다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    ci["point"]["map_50_by_seed"]["1"]["sep_fed"] += 1e-9
    g = _judge(p, ci, obs)
    reasons = g["recovery_ci_rejected"]["reasons"]
    assert g["recovery_reportable"] is None
    assert not any("sha256" in r or "시드 집합" in r or "지문" in r for r in reasons)
    assert any("sep_fed map_50" in r for r in reasons)


def test_요구6_양쪽_다_미정의인_시드는_일치로_보고_판정은_false():
    """시드 2 의 분모가 음수 — CI 는 nan, 집계기는 None. 둘이 같은 사실을 적었으니 결속은 성립하고,
    트립와이어·미정의 비율에서 떨어져 false 다(결속 실패의 null 로 흐리지 않는다)."""
    p = three_seeds()
    p[2]["threshold_independent"]["per_tag"]["sep_central"]["map_50"] = 0.10
    p = _coded(p)
    obs = _observed(p)
    ci = _bound_ci(p, obs, undefined=1500, d_lo=-0.05)
    g = _judge(p, ci, obs)
    assert g["recovery_ci_rejected"] is None
    assert g["recovery_reportable"] is False


def test_요구6_한쪽만_미정의면_null():
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    ci["point"]["R_by_seed"]["3"] = float("nan")
    g = _judge(p, ci, obs)
    assert g["recovery_reportable"] is None
    assert any("시드 3 회복률 점추정을 비교할 수 없다" in r for r in g["recovery_ci_rejected"]["reasons"])


@pytest.mark.parametrize("delta,ok", [(5e-17, True), (5e-13, True), (5e-12, False)])
def test_요구6_허용_오차_경계(delta, ok):
    """실측 잡음(≤ 6e-17)은 통과하고 허용 오차 1e-12 를 넘으면 걸린다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    ci["point"]["R_by_seed"]["1"] += delta
    g = _judge(p, ci, obs)
    assert (g["recovery_reportable"] is True) is ok, g.get("recovery_ci_rejected")


# ---- 요구 5 — T1 항등 결과가 없거나 실패면 null --------------------------------------

@pytest.mark.parametrize("breakage", [
    "missing", "one_failed", "diff_nonzero", "diff_95_nonzero", "seed_missing",
    "cell_missing", "basis_mismatch",
])
def test_요구5_항등_검사_결과가_없거나_실패면_null(breakage):
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    ic = ci["identity_check"]
    if breakage == "missing":
        del ci["identity_check"]
    elif breakage == "one_failed":
        ic["2"]["sep_fed"]["passed"] = False
    elif breakage == "diff_nonzero":
        ic["3"]["sep_central"]["abs_diff_map_50"] = 1e-17
    elif breakage == "diff_95_nonzero":
        ic["1"]["sep_local_C2"]["abs_diff_map_50_95"] = 2e-16
    elif breakage == "seed_missing":
        del ic["2"]
    elif breakage == "cell_missing":
        del ic["1"]["sep_local_C3"]              # 14/15
    elif breakage == "basis_mismatch":            # 항등 검사가 다른 채점 파일을 기준으로 돌았다
        ic["1"]["sep_fed"]["expected_map_50"] += 1e-3
    g = _judge(p, ci, obs)
    assert g["recovery_reportable"] is None, breakage
    assert any("항등" in r for r in g["recovery_ci_rejected"]["reasons"]), breakage
    assert g["public_status"] == PUBLIC_STATUS["ci_invalid"], breakage      # m-11


def _walk_keys(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield f"{path}/{k}", k
            yield from _walk_keys(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_keys(v, f"{path}[{i}]")


def test_49_5_게재_가부는_게이트_최상위_한_곳에만_있다():
    """v3 집계본에서 최상위는 true 인데 트립와이어 하위 블록에 null·"지금은 미판정" 이 남아 있었다.
    판정이 true 로 채워진 뒤에도 게이트 안에 `recovery_reportable` 이 한 곳에만 있어야 한다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs), obs)
    assert g["recovery_reportable"] is True
    paths = [path for path, k in _walk_keys(g) if k == "recovery_reportable"]
    assert paths == ["/recovery_reportable"], paths
    tw = g["tripwire_by_definition"]
    assert not [path for path, k in _walk_keys(tw) if k == "ci_pass"]   # CI 판정은 recovery_ci 에만
    assert "지금은" not in json.dumps(tw, ensure_ascii=False)


def test_정상_조합은_항등_점추정_결속을_모두_지나_true():
    """성공 시험 — 불일치 시험만 있으면 '늘 null' 인 구현도 통과한다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs), obs)
    assert g["recovery_reportable"] is True
    assert g["recovery_ci_rejected"] is None
    assert g["recovery_ci_warnings"] == []
    assert "T1 항등" in g["recovery_ci"]["binding"]


def test_지문_없는_산출물은_결속할_수_없다():
    """v1 처럼 scorer_code 가 없는 산출물 — CI 기록과 같아도 결속 불가다."""
    p = three_seeds()                                   # 지문 없음
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs), obs)
    assert g["recovery_reportable"] is None
    assert any("지문이 없다" in r for r in g["recovery_ci_rejected"]["reasons"])


# ======================================================================================
# C 42번 후속 — I-1 신뢰수준·구간 방식 · m-1 · m-2 · m-3 · m-10 · m-11
# ======================================================================================

@pytest.mark.parametrize("kw,label", [
    ({"alpha": 0.10}, "신뢰수준"),                 # 90 % 구간 — 더 좁아 ② 를 쉽게 통과한다(§4-2 F)
    ({"alpha": 0.10, "half_width": 0.20}, "신뢰수준"),
    ({"alpha": "0.05"}, "신뢰수준"),               # 문자열로 적힌 값
    ({"alpha": None}, "신뢰수준"),                 # 기록 없음
    ({"interval": "BCa"}, "구간 방식"),
    ({"interval": None}, "구간 방식"),
])
def test_I1_신뢰수준이나_구간_방식이_등록값과_다르면_null(kw, label):
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, **kw), obs)
    assert g["recovery_reportable"] is None
    assert any(label in r for r in g["recovery_ci_rejected"]["reasons"])
    assert g["public_status"] == PUBLIC_STATUS["ci_invalid"]


def test_I1_생성기와_재판정이_같은_등록_상수를_읽는다():
    from evaluation import prereg
    from scripts.probe import recovery_bootstrap as rb

    draws = np.linspace(0.0, 1.0, 2001)
    got = rb._percentile_ci(draws)
    lo, hi = np.percentile(draws, [100 * prereg.RECOVERY_CI_ALPHA / 2,
                                   100 * (1 - prereg.RECOVERY_CI_ALPHA / 2)])
    assert (got["ci_lo"], got["ci_hi"]) == (float(lo), float(hi))
    assert prereg.RECOVERY_CI_ALPHA == 0.05 and prereg.RECOVERY_CI_INTERVAL == "백분위"
    assert prereg.RECOVERY_CI_REGISTRATION["interval"]["alpha"] == prereg.RECOVERY_CI_ALPHA
    assert not hasattr(rb, "ALPHA")          # 생성기 안에 따로 둔 상수가 없다


def test_m1_정의된_수와_미정의_수의_합이_재표집_수와_다르면_null():
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, n_defined=10), obs)
    assert g["recovery_reportable"] is None
    assert any("정의된 수" in r for r in g["recovery_ci_rejected"]["reasons"])


@pytest.mark.parametrize("breakage", [
    "hash_changed", "ci_missing", "ci_empty", "key_not_in_artifact", "ci_unread", "artifact_missing",
])
def test_m2_CI_가_잰_원시_레코드_해시가_채점_기록과_다르면_null(breakage):
    """sha256 결속은 산출물 파일만 묶는다 — 채점 뒤에 레코드가 바뀌면 산출물은 그대로여도 CI 는
    다른 입력으로 계산된다. 결속·항등·점추정이 모두 맞는 상태에서 레코드 해시만 어긋나게 한다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    recs = ci["inputs"]["records_by_seed"]
    k = min(recs["2"])
    if breakage == "hash_changed":
        recs["2"][k] = "0" * 64
    elif breakage == "ci_missing":
        del recs["2"]
    elif breakage == "ci_empty":
        recs["2"] = {}
    elif breakage == "key_not_in_artifact":
        recs["2"]["outputs/main_d/seed2/sweep/other_raw_s2.jsonl"] = "a" * 64
    elif breakage == "ci_unread":
        recs["2"][k] = None
    elif breakage == "artifact_missing":             # 산출물에 input_records 가 없다(v1 형식)
        obs["artifacts"]["2"]["input_records"] = None
    g = _judge(p, ci, obs)
    reasons = g["recovery_ci_rejected"]["reasons"]
    assert g["recovery_reportable"] is None, breakage
    assert any("시드 2" in r and "레코드" in r for r in reasons), (breakage, reasons)
    assert g["public_status"] == PUBLIC_STATUS["ci_unbound"]


def test_m3_결속_판_번호가_다르면_null():
    p = _coded(three_seeds())
    obs = _observed(p)
    ci = _bound_ci(p, obs)
    ci["binding"]["version"] = 2
    g = _judge(p, ci, obs)
    assert g["recovery_reportable"] is None
    assert any("결속 판" in r for r in g["recovery_ci_rejected"]["reasons"])


def test_m3_뒤집힌_구간은_null():
    """하한 > 상한. 반폭 필드도 같이 음수로 적어 자기모순 검사는 지나가게 하고 뒤집힘만 남긴다."""
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, half_width=-0.05), obs)
    reasons = g["recovery_ci_rejected"]["reasons"]
    assert g["recovery_reportable"] is None
    assert any("뒤집혀" in r for r in reasons)
    assert not any("자기모순" in r for r in reasons)


def test_m10_기록_ci_pass_가_재판정과_갈리면_판정은_재판정_경고는_남긴다():
    p = _coded(three_seeds())
    obs = _observed(p)
    g = _judge(p, _bound_ci(p, obs, ci_pass=False), obs)       # 규칙은 실제로 통과
    assert g["recovery_reportable"] is True
    assert len(g["recovery_ci_warnings"]) == 1 and "갈렸다" in g["recovery_ci_warnings"][0]
    assert "경고" in g["verdict"]["2_recovery_ci"]
    assert g["recovery_ci"]["rules_rejudged"]["recorded_agrees"] is False


def test_m11_공개_상태는_결속_실패와_기록_검사_실패를_가른다():
    p = _coded(three_seeds())
    obs = _observed(p)
    bad_bind = _bound_ci(p, obs)
    bad_bind["binding"]["seeds"] = [1, 2]
    contradiction = _bound_ci(p, obs, half_width=0.30, half_width_field=0.05)
    assert _judge(p, bad_bind, obs)["public_status"] == PUBLIC_STATUS["ci_unbound"]
    assert _judge(p, contradiction, obs)["public_status"] == PUBLIC_STATUS["ci_invalid"]
    assert PUBLIC_STATUS["ci_unbound"] != PUBLIC_STATUS["ci_invalid"]


# ======================================================================================
# Minor 6·Important 4 — main() 을 실제로 돈다 (D ≤ 0 포함). 집계본이 먼저 써져야 한다
# + 요구 3 을 **실제 파일**로 — 한 바이트 변경 · 시드 부분집합이 main 에서도 true 를 막는다
# ======================================================================================

def _write_seed_dir(root, n, floor, artifact="score_cells_v1.json", code=None, **kw):
    d = root / f"seed{n}"
    d.mkdir(parents=True, exist_ok=True)
    p = _payload(n, floor, **kw)
    p.update({
        "curve": {"grid": [0.01, 0.25, 0.5]}, "exit_code": 0,
        "gates_evaluated": {"blocking_failures": [], "ok": True, "n_evaluated": 11},
        "discrimination_threshold_free": {"by_cell": {t: {"point": 0.5, "ci_lo": 0.4, "ci_hi": 0.6}
                                                      for t in DET_TAGS},
                                          "contrasts": {}, "n_defect": 7086, "n_normal": 654},
        "stratified": {"default_k": 64, "by_k": {"64": {**{t: {"stratified_lift": -0.01} for t in DET_TAGS},
                                                          "__shortcut__": {"stratified_lift": 0.0}}}},
        "p9": {"results": [], "all_equivalent": False},
    })
    p["discrimination"] = {"results": [
        {"cell": "sep_local" if t.startswith("sep_local") else t,
         "client": t.split("_")[-1] if t.startswith("sep_local") else None,
         "delta": {"point": 0.3, "ci_lo": 0.2, "ci_hi": 0.4}} for t in DET_TAGS]}
    if code is not None:
        p["scorer_code"] = dict(code)
    # 바이트로 쓴다 — write_text 는 윈도우에서 개행을 바꾼다
    (d / artifact).write_bytes(json.dumps(p, ensure_ascii=False).encode("utf-8"))


def _run_main(root, dest, *extra):
    import os

    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/probe/aggregate_seeds.py"), "--root", str(root),
         "--dest", str(dest), *extra],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=REPO, check=False,
    )


@pytest.mark.parametrize("degenerate", [False, True])
def test_main_이_실제로_돌고_집계본이_먼저_써진다(tmp_path, degenerate):
    """D ≤ 0 인 시드가 있어도 집계본은 써진다(C 34번 Minor 6) — 회복률 무관 표까지 잃지 않는다."""
    root = tmp_path / "main_d"
    for n in (1, 2, 3):
        floor = dict(S1)
        if degenerate and n == 2:
            floor["sep_central"] = 0.10          # 로컬평균 0.196 > 중앙 → D < 0
        _write_seed_dir(root, n, floor)
    dest = tmp_path / "seed3set"
    proc = _run_main(root, dest, "--seeds", "1,2,3")
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


def _real_ci(root, seeds, ci_path, artifact=ART):
    """**실제 파일**을 `read_artifact` 로 읽어 결속한 CI — 생성기와 같은 함수로 만든다."""
    from scripts.probe.aggregate_seeds import load_seed_entry

    loaded = {n: load_seed_entry(root, n, artifact) for n in seeds}
    payloads = {n: v[0] for n, v in loaded.items()}
    observed = make_binding(artifact, {n: v[1] for n, v in loaded.items()})
    ci = _bound_ci(payloads, observed)
    ci_path.write_bytes(json.dumps(ci, ensure_ascii=False).encode("utf-8"))


def test_main_실제_파일에서_결속_성립_한_바이트_변경_시드_부분집합(tmp_path):
    root, dest = tmp_path / "main_d", tmp_path / "seed3set"
    for n, scale in ((1, 1.0), (2, 0.98), (3, 1.01)):
        _write_seed_dir(root, n, {t: v * scale for t, v in S1.items()}, artifact=ART, code=CODE)
    ci_path = tmp_path / "recovery_ci_v1.json"
    _real_ci(root, (1, 2, 3), ci_path)

    runs = iter(range(10))

    def reportable(*extra):
        d = dest / f"run{next(runs)}"            # 집계본은 덮지 않는다 — 호출마다 새 dest
        proc = _run_main(root, d, "--artifact", ART, "--recovery-ci", str(ci_path), *extra)
        assert proc.returncode == 0, proc.stderr[-800:]
        agg = json.loads((d / "aggregate_v2.json").read_text(encoding="utf-8"))
        return agg["recovery"]["map_50"]["denominator_gate"], agg

    # 1) 결속 성립 → true, 공개 문구, 입력 결속이 집계본에 실린다
    g, agg = reportable("--seeds", "1,2,3")
    assert g["recovery_reportable"] is True
    assert agg["public_status"] == PASSED
    assert agg["input_binding"]["seeds"] == [1, 2, 3]

    # 2) 시드 부분집합 → null
    g, _ = reportable("--seeds", "1,2")
    assert g["recovery_reportable"] is None
    assert any("시드 집합" in r for r in g["recovery_ci_rejected"]["reasons"])

    # 3) 시드 2 산출물 한 바이트 변경(e → E) → null
    f = root / "seed2" / ART
    raw = f.read_bytes()
    assert raw.count(b'"scorer": "evaluation') == 1
    f.write_bytes(raw.replace(b'"scorer": "evaluation', b'"scorer": "Evaluation'))
    assert sum(a != b for a, b in zip(raw, f.read_bytes(), strict=True)) == 1
    g, _ = reportable("--seeds", "1,2,3")
    assert g["recovery_reportable"] is None
    assert any("시드 2 의 sha256" in r for r in g["recovery_ci_rejected"]["reasons"])


# ======================================================================================
# C 42번 I-2 — 판 번호로 이름을 짓고, 옛 판을 덮지 않는다
# ======================================================================================

ART3 = "score_cells_v3.json"


def test_I2_v3_입력은_v3_이름으로_쓰고_옛_판은_건드리지_않는다(tmp_path):
    """C 42번 §6 I-2 의 실증 경로 — v3 입력이 `aggregate_v1.json` 에 쓰였다. 이제 v3 이름으로 쓰고,
    기본 CI 경로도 `recovery_ci_v3.json` 이다(v2 로 만든 옛 `recovery_ci_v1.json` 을 읽지 않는다)."""
    root, dest = tmp_path / "main_d", tmp_path / "seed3set"
    for n, scale in ((1, 1.0), (2, 0.98), (3, 1.01)):
        _write_seed_dir(root, n, {t: v * scale for t, v in S1.items()}, artifact=ART3, code=CODE)
    dest.mkdir()
    old = {name: f"옛 판 {name}\n".encode() for name in
           ("aggregate_v1.json", "aggregate_v2.json", "recovery_ci_v1.json")}
    for name, raw in old.items():
        (dest / name).write_bytes(raw)
    _real_ci(root, (1, 2, 3), dest / "recovery_ci_v3.json", artifact=ART3)

    proc = _run_main(root, dest, "--artifact", ART3, "--seeds", "1,2,3")
    assert proc.returncode == 0, proc.stderr[-800:]
    agg = json.loads((dest / "aggregate_v3.json").read_text(encoding="utf-8"))
    g = agg["recovery"]["map_50"]["denominator_gate"]
    assert agg["artifact"] == ART3
    assert g["recovery_reportable"] is True, g.get("recovery_ci_rejected")
    assert g["recovery_ci"]["source"] == "recovery_ci_v3.json"
    assert agg["aggregator_code"]["tree_matches_scoring"] is False      # 픽스처 지문은 트리와 다르다
    for name, raw in old.items():
        assert (dest / name).read_bytes() == raw, name


@pytest.mark.parametrize("artifact,existing", [
    ("score_cells_v1.json", "aggregate_v1.json"),
    ("score_cells_v2.json", "aggregate_v2.json"),
    (ART3, "aggregate_v3.json"),
])
def test_I2_집계본이_이미_있으면_계산_전에_멈추고_덮지_않는다(tmp_path, artifact, existing):
    """**계산 전에** 멈춘다 — 시드 산출물이 아예 없는 루트를 준다. 확인이 적재보다 늦으면 파일 없음
    오류로 죽고, 쓰기 시점에만 막으면 "집계 도중 생겼다" 로 멈춘다. 둘 다 이 시험에서 떨어진다."""
    root, dest = tmp_path / "empty_root", tmp_path / "seed3set"
    dest.mkdir()
    (dest / existing).write_bytes(b"old\n")
    proc = _run_main(root, dest, "--artifact", artifact)
    assert proc.returncode != 0
    assert f"{existing} 이 이미 있다" in proc.stderr, proc.stderr[-600:]
    assert "Traceback" not in proc.stderr
    assert (dest / existing).read_bytes() == b"old\n"
    assert not root.exists()


@pytest.mark.parametrize("artifact", ["score_cells.json", "score_cells_latest.json", "score_cells_v2.jsonl"])
def test_I2_판_번호를_읽을_수_없는_이름은_거부(tmp_path, artifact):
    proc = _run_main(tmp_path, tmp_path / "d", "--artifact", artifact)
    assert proc.returncode != 0
    assert "판 번호" in proc.stderr
    assert not (tmp_path / "d").exists()


def _run_bootstrap(root, dest, *extra):
    import os

    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, str(REPO / "scripts/probe/recovery_bootstrap.py"), "--root", str(root),
         "--dest", str(dest), *extra],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=REPO, check=False,
    )


@pytest.mark.parametrize("artifact,existing,stops_for_overwrite", [
    (ART3, "recovery_ci_v3.json", True),
    ("score_cells_v2.json", "recovery_ci_v2.json", True),
    (ART3, "recovery_ci_v1.json", False),        # 옛 이름(v2 로 만든 CI)과는 겹치지 않는다
])
def test_I2_CI_생성기는_같은_판_CI_가_있으면_계산_전에_멈춘다(tmp_path, artifact, existing,
                                                   stops_for_overwrite):
    dest = tmp_path / "seed3set"
    dest.mkdir()
    (dest / existing).write_bytes(b"old\n")
    proc = _run_bootstrap(tmp_path / "no_root", dest, "--artifact", artifact)
    assert proc.returncode != 0                      # 루트가 없으니 어느 쪽이든 끝까지 못 간다
    assert ("덮지 않는다" in proc.stderr) is stops_for_overwrite, proc.stderr[-600:]
    assert (dest / existing).read_bytes() == b"old\n"
    assert sorted(x.name for x in dest.iterdir()) == [existing]


# ======================================================================================
# C 42번 §9-6 n-3 · n-5 — CI 생성기 항등 전용 사전 멈춤, 본문 판 = 파일명
# ======================================================================================

def test_n3_CI_생성기_항등_전용도_같은_결과가_있으면_계산_전에_멈춘다(tmp_path):
    dest = tmp_path / "seed3set"
    dest.mkdir()
    existing = dest / "identity_check_score_cells_v3.json"
    existing.write_bytes(b"old\n")
    proc = _run_bootstrap(tmp_path / "no_root", dest, "--artifact", ART3, "--identity-only")
    assert proc.returncode != 0
    assert "identity_check_score_cells_v3.json 이 이미 있다" in proc.stderr, proc.stderr[-600:]
    assert "Traceback" not in proc.stderr
    assert existing.read_bytes() == b"old\n"
    assert sorted(x.name for x in dest.iterdir()) == [existing.name]


def _rename_artifact(root, n, declared):
    """시드 n 의 v3 파일 본문에 다른 판을 적는다 — v2 파일을 v3 이름으로 옮긴 상황."""
    f = root / f"seed{n}" / ART3
    body = json.loads(f.read_bytes().decode("utf-8"))
    body["artifact_version"] = declared
    f.write_bytes(json.dumps(body, ensure_ascii=False).encode("utf-8"))


def test_n5_본문_판이_파일명과_다르면_읽지_않는다(tmp_path):
    from evaluation.recovery_ci import ArtifactNameMismatch, read_artifact

    root = tmp_path / "main_d"
    _write_seed_dir(root, 1, dict(S1), artifact=ART3)
    payload, _ = read_artifact(root / "seed1" / ART3)          # 필드 없음(v1 형식) — 대조하지 않는다
    assert "artifact_version" not in payload
    _rename_artifact(root, 1, ART3)                            # 같으면 통과
    assert read_artifact(root / "seed1" / ART3)[0]["artifact_version"] == ART3
    _rename_artifact(root, 1, "score_cells_v2.json")
    with pytest.raises(ArtifactNameMismatch, match="score_cells_v2.json"):
        read_artifact(root / "seed1" / ART3)


def test_n5_집계기와_CI_생성기_둘_다_이름을_바꾼_산출물에서_멈춘다(tmp_path):
    root, dest = tmp_path / "main_d", tmp_path / "seed3set"
    for n in (1, 2, 3):
        _write_seed_dir(root, n, dict(S1), artifact=ART3, code=CODE)
    _rename_artifact(root, 2, "score_cells_v2.json")

    agg = _run_main(root, dest, "--artifact", ART3)
    assert agg.returncode != 0
    assert "시드 2" in agg.stderr and "파일명과 판이 다른" in agg.stderr, agg.stderr[-600:]
    assert "Traceback" not in agg.stderr
    assert not (dest / "aggregate_v3.json").exists()

    ci_dest = tmp_path / "ci"
    ci = _run_bootstrap(root, ci_dest, "--artifact", ART3, "--seeds", "2")
    assert ci.returncode != 0
    assert "시드 2" in ci.stderr and "파일명과 판이 다른" in ci.stderr, ci.stderr[-600:]
    assert "Traceback" not in ci.stderr
    assert not (ci_dest / "recovery_ci_v3.json").exists()
