"""본채점 표본 모드의 지표 단계(`evaluation.strict_main`) — 합성 계수표만. 묶음 · 정답 파일 · 실물 목록을 열지 않는다.

지키는 것.

1. 회복률 · 참여자 차 · 규칙 ⑤ 의 점추정이 설계 가중 M1 에서 그대로 나온다(시험이 따로 다시 계산해 맞댄다).
2. 한 추첨을 모두가 쓴다 — 시험이 같은 등록 값으로 추첨을 다시 내어 규칙 ⑤ 의 구간 끝점을 재현한다.
3. 거부된 칸은 값 없이 이름으로, 그 칸이 필요한 파생값은 `산출 불가` 다 — 분모를 줄이지 않는다.
4. 회복률의 분모 실패는 규칙 ⑤ 의 유효 추첨을 줄이지 않는다(07번 §23-1 의 회귀 벡터).
5. 산출은 표준 JSON 이다(`NaN` · `Infinity` 없음). 대조선 예측 파일의 꼴을 본다.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation import percentile_rule as PR
from evaluation import strict_main as SM
from evaluation.metrics.box_f1 import SENSITIVITY_REGISTERED, CountTable, score_m1, sensitivity_counts
from evaluation.recovery_interval import REGISTERED_CI
from evaluation.sample_bootstrap import design_weights, draw_weights, m1_draws
from evaluation.sample_weights import SideRow
from evaluation.schema_v14 import UNI_MAIN_TAGS

CLASSES = ("100", "2011", "301", "401")
B = 300


def design():
    """층 둘(S0: 뽑힌 묶음 4 · 전체 12, S1: 3 · 6), 묶음마다 이미지 둘. 401 은 한 묶음에만 있다 — 추첨에서 사라질 수 있다."""
    ids, group_of, side = [], {}, {}
    for h, (n_h, G_h) in enumerate(((4, 12), (3, 6))):
        for g in range(n_h):
            for k in range(2):
                i = f"e{h}{g}{k}"
                ids.append(i)
                group_of[i] = f"g{h}{g}"
                side[i] = SideRow(("ST|1", "AL|2-3")[h], G_h, n_h)
    ids = sorted(ids)
    gold = np.zeros((len(ids), 4), dtype=np.int32)
    rng = np.random.default_rng(5)
    gold[:, :3] = rng.integers(0, 3, (len(ids), 3))
    gold[ids.index("e000"), 3] = 2                      # 401 은 묶음 g00 에만
    side = {i: side[i] for i in ids}
    return ids, group_of, side, gold


def table_of(ids, gold, skill: float, *, fp_rate: float, seed: int) -> CountTable:
    rng = np.random.default_rng(seed)
    tp = np.minimum(gold, np.floor(gold * skill + rng.random(gold.shape)).astype(np.int32))
    fn = gold - tp
    fp = (rng.random(gold.shape) < fp_rate).astype(np.int32)
    return CountTable(CLASSES, tuple(ids), tp.astype(np.int32), fp, fn.astype(np.int32))


def score_of(tag, ids, gold, skill, *, fp_rate=0.3, seed=0, failed=()):
    t = table_of(ids, gold, skill, fp_rate=fp_rate, seed=seed)
    gold_codes = {i: frozenset(CLASSES[k] for k in range(4) if gold[r, k]) for r, i in enumerate(ids)}
    sens = sensitivity_counts(t, gold_codes=gold_codes, failed_image_ids=list(failed), variant=SENSITIVITY_REGISTERED)
    return SimpleNamespace(tag=tag, seed_index=1, table=t, sens=sens, gold_codes=gold_codes, failed=tuple(failed),
                           failures={"n_records": len(ids), "n_record_failures": len(failed)})


SKILL = {"uni_central": 0.9, "uni_fed": 0.75, "uni_local_C1": 0.5, "uni_local_C2": 0.55, "uni_local_C3": 0.45}


def scores_for(ids, gold, skill=SKILL, drop=()):
    return {(t, 1): score_of(t, ids, gold, s, seed=k, failed=(ids[k],) if t == "uni_fed" else ())
            for k, (t, s) in enumerate(skill.items()) if t not in drop}


def registration(**ci_over):
    ci = SimpleNamespace(**REGISTERED_CI, alpha=0.05, n_resamples=B, rng_seed=20261014, min_defined_draws=200,
                         rule5_statistic="seed_mean_m1_minus_baseline_paired", rule2_max_half_width=0.25,
                         rule4_max_undefined_fraction=0.05)
    for k, v in ci_over.items():
        setattr(ci, k, v)
    return SimpleNamespace(seeds={"1": 20260828}, planned_seed_count=1, ci=ci)


EVAL_DOC = {"population": {"sample": {"full_addendum_d_min": 0.115,
                                      "not_judged_on_sample": ["al_subset_macro_m1", "p9_equivalence"]}},
            "auxiliary": ["class_jaccard", "image_presence_f1", "m1_micro", "per_class_f1", "latency_s_per_image",
                          "exchange_bytes"]}


def run(*, skill=SKILL, drop=(), rejected=None, reg=None):
    ids, group_of, side, gold = design()
    base = table_of(ids, gold, 0.2, fp_rate=0.8, seed=99)
    out = SM.compute(scores=scores_for(ids, gold, skill, drop), baseline=base, side=side, group_of=group_of,
                     reg=reg or registration(), eval_doc=EVAL_DOC, rejected=rejected or {})
    return out, (ids, group_of, side, gold, base)


def test_태그는_본실험_통합형_다섯이다() -> None:
    assert set(SM.TAGS) == set(UNI_MAIN_TAGS) and len(SM.TAGS) == 5


def test_점추정은_설계_가중_M1_에서_그대로_나온다() -> None:
    out, (ids, group_of, side, gold, base) = run()
    w = design_weights(ids, side)
    m = {t: score_m1(score_of(t, ids, gold, s, seed=k, failed=(ids[k],) if t == "uni_fed" else ()).table, w).macro_f1
         for k, (t, s) in enumerate(SKILL.items())}
    lbar = sum(m[t] for t in SM.LOCALS) / 3
    r = out["rq1_recovery"]["main"]
    assert r["status"] == "computed" and r["point"]["state"] == "finite"
    assert r["point"]["R_bar"] == pytest.approx((m["uni_fed"] - lbar) / (m["uni_central"] - lbar), abs=1e-12)
    rows = out["rq2_participants"]["main"]["rows"]
    for t in SM.LOCALS:
        assert rows[t]["diff_F_minus_L_k"] == pytest.approx(m["uni_fed"] - m[t], abs=1e-12)
    assert not {"average", "mean"} & set(out["rq2_participants"]["main"]) and out["rq2_participants"]["main"]["worst"]
    assert out["full_addendum"]["D_point"] == pytest.approx(m["uni_central"] - lbar, abs=1e-12)
    assert out["full_addendum"]["full_addendum_required"] is bool(m["uni_central"] - lbar < 0.115)
    assert out["baseline"]["main"]["m1_macro_f1"] == pytest.approx(score_m1(base, w).macro_f1, abs=1e-15)
    assert out["models"]["uni_fed_s1"]["representative"]["weighting"] == "stratified_g_over_n"


def test_한_추첨을_모두가_쓴다_규칙_5_의_구간을_재현한다() -> None:
    out, (ids, group_of, side, gold, base) = run()
    W = draw_weights(ids, group_of, side, n_resamples=B, rng_seed=20261014)
    classes = out["classes_in_macro"]
    fed = score_of("uni_fed", ids, gold, SKILL["uni_fed"], seed=1, failed=(ids[1],))
    f, ok = m1_draws(fed.table, W, classes)
    b, ok_b = m1_draws(base, W, classes)
    assert np.array_equal(ok, ok_b) and (~ok).sum() == out["bootstrap"]["n_excluded_support"] > 0
    iv, n_def, n_undef = PR.defined_interval(np.where(ok, f - b, np.nan), 0.05)
    d = out["rule5"]["per_model"]["uni_fed"]["evidence"]["diff"]
    assert (d["ci_lo"], d["ci_hi"], d["n_defined"], d["n_undefined"]) == (iv.lo.value, iv.hi.value, n_def, n_undef)
    assert d["undefined_reasons"] == {"support": n_undef}


def test_회복률의_추첨_회계는_등록_재표집_수와_같다() -> None:
    out, _ = run()
    for v in ("main", "sensitivity"):
        iv = out["rq1_recovery"][v]["interval"]
        total = sum(iv[k] for k in ("n_finite", "n_infinite_pos", "n_infinite_neg", "n_excluded_support",
                                    "n_indeterminate"))
        assert total == B and iv["n_excluded_support"] == out["bootstrap"]["n_excluded_support"]
        assert iv["convention_id"] == PR.CONVENTION_ID
    assert out["rq1_recovery"]["main"]["rules"]["R1"].startswith("미판정")
    assert out["rq1_recovery"]["main"]["adoption"].startswith("미판정")


def test_규칙_5_는_등록_판정이고_근거를_싣는다() -> None:
    out, _ = run()
    r5 = out["rule5"]
    assert r5["status"] == "computed" and r5["verdict_authority"] == "registered" and r5["final_adoption"] is None
    assert r5["condition_models_registered"] == ["uni_central", "uni_fed"]
    assert r5["per_model"]["uni_central"]["evidence"]["per_seed_m1"]["1"] == \
        out["models"]["uni_central_s1"]["representative"]["main"]["m1_macro_f1"]


def test_거부된_칸은_이름으로_남고_파생값은_산출_불가다() -> None:
    out, _ = run(drop=("uni_fed",), rejected={"uni_fed_s1": ["sidecar_mismatch"]})
    assert out["models"]["uni_fed_s1"] == {"status": "rejected", "reasons": ["sidecar_mismatch"]}
    for v in ("main", "sensitivity"):
        assert out["rq1_recovery"][v] == {"status": "산출 불가", "missing_cells": ["uni_fed_s1"]}
        assert all(r["status"] == "산출 불가" for r in out["rq2_participants"][v]["rows"].values())
    assert out["rule5"]["status"] == "산출 불가" and out["rule5"]["missing_cells"] == ["uni_fed_s1"]
    assert out["main_vs_sensitivity"]["status"] == "산출 불가"
    assert out["rq2_participants"]["main"]["worst"]["status"] == "산출 불가"            # 빠진 참여자가 있으면 고르지 않는다


def test_회복률의_분모_실패는_규칙_5_의_유효_추첨을_줄이지_않는다() -> None:
    """07번 §23-1 — 중앙이 로컬 평균보다 낮으면 분모 ≤ 0 이다. 회복률은 무한 · 판정 불가로 가고, 대응 차는 정의된다."""
    out, _ = run(skill=dict(SKILL, uni_central=0.02, uni_local_C1=0.7, uni_local_C2=0.7, uni_local_C3=0.7))
    iv = out["rq1_recovery"]["main"]["interval"]
    excluded = out["bootstrap"]["n_excluded_support"]
    assert iv["n_finite"] == 0 and iv["n_infinite_pos"] + iv["n_infinite_neg"] + iv["n_indeterminate"] == B - excluded
    assert out["rq1_recovery"]["main"]["point"]["state"] == "absent"
    assert out["rq1_recovery"]["main"]["rules"]["R2"]["half_width_within"] is None      # 수가 아닌 반폭은 판정하지 않는다
    for t in ("uni_central", "uni_fed"):
        assert out["rule5"]["per_model"][t]["evidence"]["diff"]["n_defined"] == B - excluded


def test_환경이_다르면_모델을_맞대는_값은_산출_불가다() -> None:
    ids, group_of, side, gold = design()
    out = SM.compute(scores=scores_for(ids, gold), baseline=table_of(ids, gold, 0.2, fp_rate=0.8, seed=99), side=side,
                     group_of=group_of, reg=registration(), eval_doc=EVAL_DOC, rejected={"env:model": ["env_mismatch"]},
                     invalid_comparison=["env:model"])
    for key in ("bootstrap", "rq1_recovery", "rq2_participants", "rule5", "main_vs_sensitivity", "full_addendum"):
        assert out[key] == {"status": "산출 불가", "reason": "env_mismatch", "detail": ["env:model"],
                            "note": out[key]["note"]}, key
    assert all(m["status"] == "scored" for m in out["models"].values())
    json.dumps(out, allow_nan=False)


def test_산출은_표준_JSON_이다() -> None:
    out, _ = run()
    json.dumps(out, allow_nan=False)


def test_항등_검사가_어긋나면_멈춘다(monkeypatch) -> None:
    monkeypatch.setattr(SM, "identity_problems", lambda *a: ["어긋남"])
    with pytest.raises(ValueError, match="재표집 검사"):
        run()


def test_등록_구간_규약이_다르면_멈춘다() -> None:
    with pytest.raises(ValueError, match="구간 규약"):
        run(reg=registration(percentile_method="linear"))


def test_모델끼리_모집단이_다르면_멈춘다() -> None:
    ids, group_of, side, gold = design()
    scores = scores_for(ids, gold)
    other = gold.copy()
    other[0, 0] += 1                                     # 정답이 다른 표
    scores[("uni_fed", 1)] = score_of("uni_fed", ids, other, 0.75)
    with pytest.raises(ValueError):
        SM.compute(scores=scores, baseline=table_of(ids, gold, 0.2, fp_rate=0.8, seed=99), side=side, group_of=group_of,
                   reg=registration(), eval_doc=EVAL_DOC, rejected={})


def test_표본에서_판정하지_않는_것과_내지_않는_것은_키가_남는다() -> None:
    out, _ = run()
    assert set(out["not_judged_on_sample"]) == {"al_subset_macro_m1", "p9_equivalence"}
    assert {"class_jaccard", "image_presence_f1", "latency_s_per_image", "exchange_bytes", "source_axis"} <= \
        set(out["not_computed"])


# ---------------------------------------------------------------- 대조선 예측 파일

POP = ["e000", "e001"]
EVAL = {"e000", "e001", "e002"}


def lines(*rows) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def box(code="2011", b=(1, 2, 3, 4)):
    return {"iso_code": code, "bbox_px": list(b)}


def read(raw):
    return SM.read_baseline_predictions(raw, population=POP, allowed_ids=EVAL, known_codes={"100", "2011", "301", "401", "402"},
                                        classes=CLASSES)


def test_대조선_예측은_모집단을_덮고_밖의_코드는_세고_버린다() -> None:
    pred, info = read(lines({"image_id": "e000", "defects": [box(), box("402")]}, {"image_id": "e001", "defects": []},
                            {"image_id": "e002", "defects": [box()]}))
    assert pred == {"e000": [("2011", [1.0, 2.0, 3.0, 4.0])], "e001": []}
    assert info == {"n_lines": 3, "n_outside_population": 1, "n_out_of_scope_dropped": 1}


@pytest.mark.parametrize("raw", [
    lines({"image_id": "e000", "defects": []}),                                              # 모집단을 덮지 않는다
    lines({"image_id": "e000", "defects": []}, {"image_id": "e000", "defects": []}),          # 겹침
    lines({"image_id": "e000", "defects": []}, {"image_id": "x9", "defects": []}),            # 평가 분할 밖
    lines({"image_id": "e000", "defects": [box("999")]}, {"image_id": "e001", "defects": []}),  # 모르는 코드
    lines({"image_id": "e000", "defects": [box(b=(3, 2, 1, 4))]}, {"image_id": "e001", "defects": []}),  # 퇴화 상자
    lines({"image_id": "e000", "defects": [], "x": 1}, {"image_id": "e001", "defects": []}),  # 여분 키
    lines({"image_id": "e000", "defects": []}, {"image_id": "e001", "defects": []}).replace(b"\n", b"\r\n"),
    b"not json\n",
])
def test_대조선_예측의_꼴이_틀리면_거부한다(raw) -> None:
    with pytest.raises(SM.BaselineFileError):
        read(raw)


def test_재표집은_층마다_n_h_와_중복도_지문_NumPy_판을_싣고_코드가_정한_것을_밝힌다() -> None:
    out, _ = run()
    b = out["bootstrap"]
    assert b["identity_check"] == b["multiplicity_check"] == "pass" and len(b["multiplicity_sha256"]) == 64
    assert b["numpy_version"] == np.__version__
    assert b["strata"] == {"AL|2-3": {"n_h": 3, "G_h": 6, "variance_factor_without_correction": 2 / 3},
                           "ST|1": {"n_h": 4, "G_h": 12, "variance_factor_without_correction": 3 / 4}}
    assert out["unregistered_choices"] and "Rao–Wu" in out["unregistered_choices"][0]["what"]
    again, _ = run()
    assert again["bootstrap"]["multiplicity_sha256"] == b["multiplicity_sha256"]


def test_참여자_방향은_등록_유효_하한을_따른다() -> None:
    strict, _ = run(reg=registration(min_defined_draws=B))
    loose, _ = run(reg=registration(min_defined_draws=1))
    assert strict["bootstrap"]["n_excluded_support"] > 0
    for t in SM.LOCALS:
        row = strict["rq2_participants"]["main"]["rows"][t]
        assert row["direction"] is None and "하한" in row["direction_reason"]
        assert loose["rq2_participants"]["main"]["rows"][t]["direction"] in ("위", "아래", "구분 불가")
    w = loose["rq2_participants"]["main"]["worst"]
    assert w["participant"] in SM.LOCALS and w["interval"]["n_defined"] > 0


def test_로컬과_로컬_평균은_규칙_5_의_진단으로_병기되고_공식_판정을_바꾸지_않는다() -> None:
    out, _ = run()
    r5 = out["rule5"]
    assert r5["diagnostic_models"] == sorted([*SM.LOCALS, SM.LOCAL_MEAN])
    assert r5["condition_models_registered"] == ["uni_central", "uni_fed"]
    lm = r5["per_model"][SM.LOCAL_MEAN]["evidence"]
    locals_ = [out["models"][f"{t}_s1"]["representative"]["main"]["m1_macro_f1"] for t in SM.LOCALS]
    assert lm["per_seed_m1"]["1"] == pytest.approx(sum(locals_) / 3, abs=1e-12)
    official = all(r5["per_model"][m]["passed"] for m in r5["condition_models_registered"])
    assert r5["rule5_passed"] is official


def test_AL_부분집합은_판정_불가로_점추정과_구간만_싣는다() -> None:
    out, (ids, group_of, side, gold, base) = run()
    al = out["not_judged_on_sample"]["al_subset_macro_m1"]
    assert al["status"].startswith("판정 불가") and al["n_images"] == sum(side[i].stratum.startswith("AL|") for i in ids)
    assert set(al["per_model"]) == {f"{t}_s1" for t in SM.TAGS}
    one = al["per_model"]["uni_fed_s1"]
    assert "interval" in one and "point" in one
    assert out["not_judged_on_sample"]["p9_equivalence"]["computed"] is False
