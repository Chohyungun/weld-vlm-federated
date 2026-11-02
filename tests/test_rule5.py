"""규칙 ⑤ 등록 판정(`evaluation.rule5`) — 07번 미니스펙 §22-2 · §22-3 · §22-4 · §23-1 · §23-2 · §24-2 · §24-5 · §25-1 의 시험 문안.

합성 근거만 쓴다. 숫자 문턱(`min_defined_draws`)은 시험의 보기 값이다 — 등록값이 아니다.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from evaluation import percentile_rule as PR
from evaluation import rule5 as R5

N = 100


_DEFAULT = object()


def reg(*, seeds=_DEFAULT, n=N, m=90, stat="seed_mean_m1_minus_baseline_paired", planned=None):
    ci = SimpleNamespace(n_resamples=n, min_defined_draws=m, rule5_statistic=stat)
    return SimpleNamespace(seeds={"1": 20260828} if seeds is _DEFAULT else seeds, planned_seed_count=planned, ci=ci)


def diff(lo=0.05, hi=0.15, n_def=N, n_undef=0, reasons=None) -> R5.DiffCI:
    return R5.DiffCI(PR.bound_of(lo) if n_def else PR.ABSENT, PR.bound_of(hi) if n_def else PR.ABSENT, n_def, n_undef,
                     reasons if reasons is not None else ({"support": n_undef} if n_undef else {}))


def ev(tag, m1=0.40, d=None, seeds=(1,)) -> R5.ModelEvidence:
    return R5.ModelEvidence(tag, {s: m1 for s in seeds}, d or diff())


def judge(r=None, *, baseline=0.16, evidence=None):
    r = r or R5.registered_inputs(reg())
    evidence = evidence or {"uni_central": ev("uni_central"), "uni_fed": ev("uni_fed")}
    return R5.judge_rule5_registered(r, baseline_m1=baseline, evidence=evidence)


# ---------------------------------------------------------------- §22-2 · §23-2 등록 시드 · 추첨 칸

@pytest.mark.parametrize("seeds", [None, {}, {"1": 5, "2": 5}, {"01": 5}, {"a": 5}, {"0": 5}, {"1": True}, {"1": -1}])
def test_등록_시드의_꼴이_틀리면_ValueError(seeds) -> None:
    with pytest.raises(ValueError):
        R5.registered_inputs(reg(seeds=seeds))


@pytest.mark.parametrize(("kw", "value"), [("n", None), ("n", 0), ("n", True), ("m", None), ("m", 0), ("m", N + 1),
                                           ("m", True), ("stat", ""), ("stat", None), ("planned", 2)])
def test_등록_추첨_칸이_없거나_틀리면_ValueError(kw, value) -> None:
    with pytest.raises(ValueError):
        R5.registered_inputs(reg(**{kw: value}))


def test_시드_번호는_한_번만_정수로_정규화한다() -> None:
    r = R5.registered_inputs(reg(seeds={"2": 20260829, "1": 20260828}, planned=2))
    assert r.seeds == {1: 20260828, 2: 20260829} and list(r.seeds) == [1, 2]


def test_번호_자리에_시드_값이_오면_ValueError() -> None:
    bad = R5.ModelEvidence("uni_central", {20260828: 0.4}, diff())
    with pytest.raises(ValueError, match="시드 값"):
        judge(evidence={"uni_central": bad, "uni_fed": ev("uni_fed")})


def test_번호_집합이_등록과_다르면_rule5_1_은_None_이고_이름을_남긴다() -> None:
    r = R5.registered_inputs(reg(seeds={"1": 5, "2": 6, "3": 7}, planned=3))
    short = {"uni_central": ev("uni_central", seeds=(1, 2)), "uni_fed": ev("uni_fed", seeds=(1, 2))}
    v = R5.judge_rule5_registered(r, baseline_m1=0.16, evidence=short)
    assert v.per_model["uni_central"].rule5_1 is None and not v.rule5_passed
    assert any("빠짐 [3]" in x for x in v.per_model["uni_central"].reasons)


def test_모델끼리_시드_집합이_다르면_ValueError() -> None:
    r = R5.registered_inputs(reg(seeds={"1": 5, "2": 6}, planned=2))
    with pytest.raises(ValueError, match="짝"):
        R5.judge_rule5_registered(r, baseline_m1=0.16, evidence={"uni_central": ev("uni_central", seeds=(1, 2)),
                                                                 "uni_fed": ev("uni_fed", seeds=(1,))})


# ---------------------------------------------------------------- ⑤-① 엄격 초과

@pytest.mark.parametrize(("m1", "baseline", "want"), [(0.40, 0.16, True), (0.16, 0.16, False), (0.10, 0.16, False),
                                                      (None, 0.16, None), (1.2, 0.16, None), (0.4, None, None),
                                                      (float("nan"), 0.16, None)])
def test_rule5_1_은_전_시드에서_엄격히_넘어야_참이다(m1, baseline, want) -> None:
    v = judge(baseline=baseline, evidence={"uni_central": ev("uni_central", m1=m1), "uni_fed": ev("uni_fed")})
    assert v.per_model["uni_central"].rule5_1 is want


# ---------------------------------------------------------------- §22-3 ⑤-② 회계 · 하한

def test_유효와_미정의의_합이_등록_재표집_수가_아니면_ValueError() -> None:
    with pytest.raises(ValueError, match="재표집 수"):
        judge(evidence={"uni_central": ev("uni_central", d=diff(n_def=50, n_undef=10)), "uni_fed": ev("uni_fed")})


def test_유효_추첨이_하한보다_적으면_방향이_위여도_rule5_2_는_None() -> None:
    d = diff(lo=0.05, hi=0.2, n_def=80, n_undef=20)
    v = judge(evidence={"uni_central": ev("uni_central", d=d), "uni_fed": ev("uni_fed")})
    pm = v.per_model["uni_central"]
    assert pm.rule5_2 is None and pm.direction is None and not v.rule5_passed
    assert any("하한" in x for x in pm.reasons)


def test_유효_추첨이_0_이면_rule5_2_는_None() -> None:
    r = R5.registered_inputs(reg(m=1))
    d = diff(n_def=0, n_undef=N)
    v = R5.judge_rule5_registered(r, baseline_m1=0.16, evidence={"uni_central": ev("uni_central", d=d),
                                                                 "uni_fed": ev("uni_fed")})
    assert v.per_model["uni_central"].rule5_2 is None


@pytest.mark.parametrize(("lo", "hi", "want", "direction"), [(0.01, 0.1, True, "above"), (-0.01, 0.1, False, "indeterminate"),
                                                            (-0.2, -0.1, False, "below"), (0.0, 0.1, False, "indeterminate")])
def test_rule5_2_는_구간_하한이_0_보다_커야_참이다(lo, hi, want, direction) -> None:
    v = judge(evidence={"uni_central": ev("uni_central", d=diff(lo, hi)), "uni_fed": ev("uni_fed")})
    assert (v.per_model["uni_central"].rule5_2, v.per_model["uni_central"].direction) == (want, direction)


def test_대응_차의_미정의_사유_합은_미정의_수와_같아야_한다() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="사유"):
        R5.DiffCI.from_draws(np.array([0.1, np.nan, 0.2]), {"support": 2}, alpha=0.05)
    d = R5.DiffCI.from_draws(np.array([0.1, np.nan, 0.2]), {"support": 1}, alpha=0.05)
    assert (d.n_defined, d.n_undefined, d.lo.state) == (2, 1, "finite")


# ---------------------------------------------------------------- §22-4 · §23-2 · §24-5 집합

def test_근거의_자리와_신원이_다르면_ValueError() -> None:
    with pytest.raises(ValueError, match="신원"):
        judge(evidence={"uni_central": ev("uni_fed"), "uni_fed": ev("uni_fed")})


def test_등록_조건_모델의_근거가_없으면_ValueError() -> None:
    with pytest.raises(ValueError, match="근거가 없다"):
        judge(evidence={"uni_central": ev("uni_central")})


def test_진단_모델의_실패는_공식_판정을_바꾸지_않는다() -> None:
    evidence = {"uni_central": ev("uni_central"), "uni_fed": ev("uni_fed"),
                "uni_local_C1": ev("uni_local_C1", m1=0.01, d=diff(-0.3, -0.1))}
    d = judge(evidence=evidence).as_dict()
    assert d["rule5_passed"] is True
    assert d["diagnostic_models"] == ["uni_local_C1"] and d["condition_models_registered"] == ["uni_central", "uni_fed"]
    assert d["per_model"]["uni_local_C1"]["passed"] is False
    assert sorted(d["per_model"]) == d["condition_models_checked"]


def test_판정은_근거와_최종_채택_보류를_싣는다() -> None:
    d = judge().as_dict()
    assert d["verdict_authority"] == "registered" and d["final_adoption"] is None and d["final_adoption_reason"]
    e = d["per_model"]["uni_fed"]["evidence"]
    assert e["per_seed_m1"] == {"1": 0.40} and e["baseline_m1"] == 0.16
    assert (e["diff"]["n_defined"], e["diff"]["n_undefined"], e["diff"]["ci_lo_state"]) == (N, 0, "finite")


# ---------------------------------------------------------------- §24-2 · §25-1 근거에서 다시 판정

def stored():
    return json.loads(json.dumps(judge().as_dict()))


def rj(s, r, **kw):
    return R5.rejudge(s, r, **({"scoring_sha256": None, "stored_scoring_sha256": None} | kw))


def test_저장된_판정은_근거에서_다시_판정해_같으면_지난다() -> None:
    rj(stored(), R5.registered_inputs(reg()))


def test_근거는_불통과인데_불리언만_참이면_거부한다() -> None:
    """§25-1 벡터 1 — 앞 계약(불리언 AND)이 통과시켰을 입력."""
    s = stored()
    s["per_model"]["uni_fed"]["evidence"]["diff"]["ci_lo"] = -0.01           # 근거는 ⑤-② 불통과
    s["per_model"]["uni_fed"]["rule5_2"] = True                             # 불리언은 그대로 참
    with pytest.raises(ValueError, match="다시 판정"):
        rj(s, R5.registered_inputs(reg()))


def test_근거가_빠지면_거부한다() -> None:
    s = stored()
    del s["per_model"]["uni_central"]["evidence"]["diff"]
    with pytest.raises(ValueError, match="근거"):
        rj(s, R5.registered_inputs(reg()))


def test_유효_추첨이_하한_미달인데_참으로_적힌_판정은_거부한다() -> None:
    s = stored()
    s["per_model"]["uni_central"]["evidence"]["diff"].update({"n_defined": 80, "n_undefined": 20,
                                                              "undefined_reasons": {"support": 20}})
    with pytest.raises(ValueError, match="다시 판정"):
        rj(s, R5.registered_inputs(reg()))


@pytest.mark.parametrize("edit", ["seeds", "models", "authority"])
def test_시드_모델_집합이나_권한이_등록과_다르면_거부한다(edit) -> None:
    s = stored()
    r = R5.registered_inputs(reg())
    if edit == "seeds":
        r = R5.registered_inputs(reg(seeds={"1": 20260828, "2": 20260829}, planned=2))
    elif edit == "models":
        s = copy.deepcopy(s)
        s["condition_models_registered"] = ["uni_central"]
    else:
        s["verdict_authority"] = "diagnostic"
    with pytest.raises(ValueError):
        rj(s, r)


def test_채점_지문이_다르면_거부한다() -> None:
    with pytest.raises(ValueError, match="채점 지문"):
        rj(stored(), R5.registered_inputs(reg()), scoring_sha256="a" * 64, stored_scoring_sha256="b" * 64)


def test_채점_지문은_부르는_쪽이_반드시_준다() -> None:
    with pytest.raises(TypeError):
        R5.rejudge(stored(), R5.registered_inputs(reg()))                    # 빠뜨리면 조용히 넘어가지 않는다


@pytest.mark.parametrize("kw", [{"seeds": {}}, {"seeds": {1: 5, 2: 5}}, {"seeds": {0: 5}}, {"n_resamples": 0},
                                {"min_defined_draws": 0}, {"statistic": " "}, {"condition_models": ()}])
def test_등록_입력을_직접_만들어도_같은_규칙으로_본다(kw) -> None:
    """빈 시드 사상이면 ⑤-① 의 all([]) 가 참이 되어 검사 없이 통과한다 — 직접 만든 등록 입력도 막는다."""
    args = {"seeds": {1: 20260828}, "n_resamples": N, "min_defined_draws": 90, "statistic": "x"} | kw
    with pytest.raises(ValueError):
        R5.RegisteredRule5(**args)


@pytest.mark.parametrize(("key", "value"), [("final_adoption", True), ("min_defined_draws", 1), ("n_resamples", N + 1),
                                            ("statistic", "other")])
def test_등록_칸이나_최종_채택을_고친_판정은_거부한다(key, value) -> None:
    s = stored()
    s[key] = value
    with pytest.raises(ValueError):
        rj(s, R5.registered_inputs(reg()))


def test_등록_모델의_규약_id_가_다르면_거부한다() -> None:
    s = stored()
    s["per_model"]["uni_fed"]["evidence"]["diff"]["convention_id"] = "other-convention"
    with pytest.raises(ValueError, match="근거"):
        rj(s, R5.registered_inputs(reg()))


def test_진단_모델의_근거가_빠져도_공식_판정은_막지_않는다() -> None:
    """§25-1 금지 5 — 등록 외 모델의 근거가 없다고 공식 판정을 막지 않는다."""
    evidence = {"uni_central": ev("uni_central"), "uni_fed": ev("uni_fed"), "uni_local_C1": ev("uni_local_C1", m1=0.1)}
    s = json.loads(json.dumps(judge(evidence=evidence).as_dict()))
    del s["per_model"]["uni_local_C1"]["evidence"]
    rj(s, R5.registered_inputs(reg()))
