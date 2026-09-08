"""게이트 레지스트리 — 체크리스트 18 (80번 D2·D10·D11·D12).

80번 §8 의 판정은 "결함이 있는데 시험이 초록"이 아니라 **"검사를 만들어 두고 부르지
않았다"** 였다. 그래서 여기 시험의 절반은 게이트의 **판정 내용**이 아니라
**호출된다는 사실**을 본다.

각 게이트에는 **이빨 시험**이 붙는다 — 실패해야 하는 입력에서 실제로 실패하는가.
그것 없이는 "전부 통과"가 "전부 죽어 있음"과 구별되지 않는다.
"""

from __future__ import annotations

import pytest

from evaluation.gates import REGISTRY, GateContext, run_scoring_gates

# --------------------------------------------------------------------------------------
# 레지스트리 자체가 무이빨이 되지 않게
# --------------------------------------------------------------------------------------

def test_every_registered_gate_is_evaluated() -> None:
    """등록된 게이트가 **전부** 돈다. 골라 부르면 여기서 깨진다."""
    out = run_scoring_gates(GateContext())
    assert out["n_registered"] == len(REGISTRY)
    assert out["n_evaluated"] == len(REGISTRY)
    assert {r["name"] for r in out["results"]} == set(REGISTRY)


def test_registry_is_not_empty() -> None:
    """비어 있으면 모든 게이트 시험이 조용히 통과한다."""
    assert len(REGISTRY) >= 6


def test_expected_gates_are_registered() -> None:
    """체크리스트 18 이 이름으로 지목한 것들이 실제로 등록돼 있는가."""
    assert {
        "prereg_constants_reproduced",
        "recovery_denominator",
        "no_cloud_logging",
        "required_tags",
        "coord_space_contract",
        "scoring_population",
        "stratified_scoring",          # 13번 D-1 — 판정 6 이행 담보
        "sweep_curve_recorded",        # 22번 §1-2 — 총괄 판정 1 이행 담보
        "p9_source_separation",        # 22번 §2-2-2 — 총괄 판정 2 이행 담보
    } <= set(REGISTRY)


def test_skipped_is_distinguished_from_passed() -> None:
    """입력이 없어 못 잰 것과 재서 통과한 것을 섞지 않는다."""
    out = run_scoring_gates(GateContext())
    assert out["n_skipped"] > 0
    for r in out["results"]:
        assert "skipped" in r


# --------------------------------------------------------------------------------------
# 이빨 시험 — 실패해야 하는 입력에서 실패하는가
# --------------------------------------------------------------------------------------

def test_cloud_logging_gate_has_teeth_for_mlflow() -> None:
    """**`MLFLOW_` 가 빠져 있던 것이 D12 다.** 이 프로젝트의 추적기가 그것이다."""
    out = run_scoring_gates(GateContext(env={"MLFLOW_TRACKING_URI": "https://x/y"}))
    g = next(r for r in out["results"] if r["name"] == "no_cloud_logging")
    assert g["passed"] is False
    assert "MLFLOW_TRACKING_URI" in g["detail"]
    assert "no_cloud_logging" in out["blocking_failures"]


def test_cloud_logging_gate_passes_when_clean() -> None:
    out = run_scoring_gates(GateContext(env={"PATH": "/usr/bin"}))
    g = next(r for r in out["results"] if r["name"] == "no_cloud_logging")
    assert g["passed"] is True


def test_required_tags_gate_has_teeth() -> None:
    from tracking.mlflow_local import REQUIRED_TAGS

    full = {t: "x" for t in REQUIRED_TAGS}
    ok = run_scoring_gates(GateContext(env={}, tags=full))
    assert next(r for r in ok["results"] if r["name"] == "required_tags")["passed"]

    missing = dict(full)
    missing[REQUIRED_TAGS[0]] = ""
    bad = run_scoring_gates(GateContext(env={}, tags=missing))
    g = next(r for r in bad["results"] if r["name"] == "required_tags")
    assert g["passed"] is False
    assert REQUIRED_TAGS[0] in g["detail"]


def test_coord_space_gate_has_teeth() -> None:
    class R:
        def __init__(self, cs):
            self.coord_space = cs

    ok = run_scoring_gates(GateContext(
        env={}, expected_coord_space="ABS_ORIG",
        records_by_cell={"a": [R("ABS_ORIG")], "b": [R("ABS_ORIG")]}))
    assert next(r for r in ok["results"] if r["name"] == "coord_space_contract")["passed"]

    bad = run_scoring_gates(GateContext(
        env={}, expected_coord_space="ABS_ORIG",
        records_by_cell={"a": [R("ABS_ORIG")], "b": [R("NORM_1000")]}))
    g = next(r for r in bad["results"] if r["name"] == "coord_space_contract")
    assert g["passed"] is False
    assert "NORM_1000" in g["detail"]


def test_population_gate_has_teeth() -> None:
    ok = run_scoring_gates(GateContext(env={}, n_eval=10, n_scored={"a": 10, "b": 10}))
    assert next(r for r in ok["results"] if r["name"] == "scoring_population")["passed"]

    bad = run_scoring_gates(GateContext(env={}, n_eval=10, n_scored={"a": 10, "b": 6}))
    g = next(r for r in bad["results"] if r["name"] == "scoring_population")
    assert g["passed"] is False
    assert "b" in g["detail"]


def test_prereg_gate_has_teeth() -> None:
    from evaluation.prereg import PREREG

    ok = run_scoring_gates(GateContext(env={}, extra={"measured_prereg": {
        "all_positive_macro_f1": PREREG.all_positive_macro_f1,
        "spec_only_macro_f1": PREREG.spec_only_macro_f1}}))
    assert next(r for r in ok["results"]
                if r["name"] == "prereg_constants_reproduced")["passed"]

    bad = run_scoring_gates(GateContext(env={}, extra={"measured_prereg": {
        "all_positive_macro_f1": 0.5, "spec_only_macro_f1": 0.5}}))
    g = next(r for r in bad["results"] if r["name"] == "prereg_constants_reproduced")
    assert g["passed"] is False


def test_recovery_gate_refuses_headline_on_single_seed() -> None:
    """시드 1세트면 분모 규칙을 판정할 수 없다 — 통과로 넘기지 않는다(80번 D10)."""
    out = run_scoring_gates(GateContext(
        env={}, recovery={"separated": {"central": 0.19, "local_mean": 0.09,
                                        "denominator": 0.099}}))
    g = next(r for r in out["results"] if r["name"] == "recovery_denominator")
    assert g["passed"] is False
    assert g["skipped"] is False, "판정을 안 한 것이 아니라 판정한 결과다"


def test_recovery_gate_has_teeth_with_seed_sd() -> None:
    narrow = run_scoring_gates(GateContext(
        env={}, seed_sd=0.05,
        recovery={"separated": {"central": 0.19, "local_mean": 0.09}}))
    g = next(r for r in narrow["results"] if r["name"] == "recovery_denominator")
    assert g["passed"] is False, "D=0.0992 < 3·0.05=0.15 인데 통과했다"

    wide = run_scoring_gates(GateContext(
        env={}, seed_sd=0.01,
        recovery={"separated": {"central": 0.19, "local_mean": 0.09}}))
    g2 = next(r for r in wide["results"] if r["name"] == "recovery_denominator")
    assert g2["passed"] is True


# --------------------------------------------------------------------------------------
# 차단과 기록의 구분
# --------------------------------------------------------------------------------------

def test_non_blocking_failure_does_not_block() -> None:
    """`gate_status: 판정_대기` 인 게이트는 재고 기록하되 차단하지 않는다."""
    out = run_scoring_gates(GateContext(
        env={}, metrics={"a": {"macro_f1": 0.2}}, population_bound=0.216,
        gate_status="판정_대기", extra={"gate_pass_line": 0.9199}))
    g = next(r for r in out["results"] if r["name"] == "content_free_gate")
    assert g["passed"] is False
    assert g["blocking"] is False
    assert "content_free_gate" not in out["blocking_failures"]
    assert out["ok"] is True


def test_status_applied_makes_it_blocking() -> None:
    out = run_scoring_gates(GateContext(
        env={}, metrics={"a": {"macro_f1": 0.2}}, population_bound=0.216,
        gate_status="적용", extra={"gate_pass_line": 0.9199}))
    g = next(r for r in out["results"] if r["name"] == "content_free_gate")
    assert g["blocking"] is True
    assert "content_free_gate" in out["blocking_failures"]
    assert out["ok"] is False


def test_duplicate_registration_is_refused() -> None:
    from evaluation.gates import register

    with pytest.raises(ValueError):
        register("no_cloud_logging")(lambda ctx: None)


# --------------------------------------------------------------------------------------
# stratified_scoring — 판정 6 이행 담보 (13번 D-1). 층화 블록이 같은 산출물 안에 있는가
# --------------------------------------------------------------------------------------

def _strata(lift: float = 0.0, lift_w: float = 0.0, *, cells=("a", "b"),
            with_shortcut: bool = True, k: int = 64) -> dict:
    from evaluation.strata import SHORTCUT_TAG

    rows = {c: {"stratified_lift": -0.2, "stratified_macro_f1": 0.3} for c in cells}
    if with_shortcut:
        rows[SHORTCUT_TAG] = {
            "stratified_lift": lift, "stratified_lift_weighted": lift_w,
            "global_macro_f1": 0.8957, "stratified_macro_f1": 0.8614,
            "n_strata_impure_scored": 13,
        }
    return {"default_k": k, "by_k": {str(k): rows}}


def _result(out: dict, name: str) -> dict:
    return next(r for r in out["results"] if r["name"] == name)


def test_stratified_gate_skips_without_block() -> None:
    """단위 문맥(블록 미제공)에서는 skipped — passed 와 구분된다."""
    r = _result(run_scoring_gates(GateContext(env={})), "stratified_scoring")
    assert r["skipped"] and r["passed"]


def test_stratified_gate_passes_when_shortcut_lift_is_zero() -> None:
    metrics = {"a": {"macro_f1": 0.2}, "b": {"macro_f1": 0.3}}
    out = run_scoring_gates(GateContext(env={}, metrics=metrics,
                                        extra={"stratified": _strata()}))
    r = _result(out, "stratified_scoring")
    assert r["passed"] and not r["skipped"] and r["blocking"]
    assert "stratified_scoring" not in out["blocking_failures"]
    assert r["value"]["n_cells"] == 2


def test_stratified_gate_has_teeth() -> None:
    """실패해야 하는 입력 넷 — 전부 **차단** 실패여야 한다."""
    metrics = {"a": {"macro_f1": 0.2}, "b": {"macro_f1": 0.3}}

    def blocked(extra: dict) -> bool:
        out = run_scoring_gates(GateContext(env={}, metrics=metrics, extra=extra))
        return "stratified_scoring" in out["blocking_failures"]

    # (1) 기본 K 의 표가 없다 — 병기 미이행
    assert blocked({"stratified": {"default_k": 64, "by_k": {}}})
    # (2) 채점된 칸이 층화 표에 없다
    assert blocked({"stratified": _strata(cells=("a",))})
    # (3) 지름길 규칙 행이 없다 — 판별력 계측 부재
    assert blocked({"stratified": _strata(with_shortcut=False)})
    # (4) 지름길 규칙의 lift 가 0 이 아니다 — 층 정의 또는 기준선 고장
    assert blocked({"stratified": _strata(lift=1e-6)})
    assert blocked({"stratified": _strata(lift_w=-1e-6)})


# --------------------------------------------------------------------------------------
# sweep_curve_recorded — 곡선 병기 담보 (총괄 판정 1 · 22번 §1-2)
# --------------------------------------------------------------------------------------

def _curve(grid=(0.01, 0.25, 0.5), tags=("sep_central", "sep_fed"), missing=(),
           drop_point=False) -> dict:
    keys = [f"{t:.2f}" for t in grid]
    by_tag = {}
    for t in tags:
        pts = {k: {"n_boxes": 10, "macro_f1": 0.5, "miss_rate": 0.4,
                   "defect_recall": 0.6, "class_jaccard": 0.5} for k in keys}
        if drop_point and t == tags[-1]:
            pts.pop(keys[-1])
        by_tag[t] = {"n_boxes_floor": 100, "by_threshold": pts}
    return {"grid": list(grid), "grid_source": "configs/base.yaml:x",
            "by_tag": by_tag, "missing_tags": list(missing),
            "threshold_dependence": {"macro_f1": {"fed_vs_local_mean_flips": True,
                                                  "fed_vs_central_flips": False}}}


def _indep(primary="map_50", with_recovery=True, pct=29.0) -> dict:
    return {"primary": primary,
            "recovery": {primary: {"recovery_pct": pct}} if with_recovery else {}}


def test_curve_gate_skips_without_block() -> None:
    r = _result(run_scoring_gates(GateContext(env={})), "sweep_curve_recorded")
    assert r["skipped"] and r["passed"]


def test_curve_gate_passes_with_full_grid() -> None:
    out = run_scoring_gates(GateContext(
        env={}, extra={"curve": _curve(), "threshold_independent": _indep()}))
    r = _result(out, "sweep_curve_recorded")
    assert r["passed"] and not r["skipped"] and r["blocking"]
    assert r["value"]["flipping_axes"] == ["macro_f1"]


def test_curve_gate_has_teeth() -> None:
    """실패해야 하는 입력 넷 — 전부 **차단** 실패여야 한다."""
    def blocked(extra: dict) -> bool:
        out = run_scoring_gates(GateContext(env={}, extra=extra))
        return "sweep_curve_recorded" in out["blocking_failures"]

    ind = _indep()
    # (1) 격자가 비어 있다
    assert blocked({"curve": {**_curve(), "grid": []}, "threshold_independent": ind})
    # (2) 하한 레코드가 없어 칸이 빠졌다 — 곡선이 조용히 줄어드는 경로
    assert blocked({"curve": _curve(missing=("sep_fed",)), "threshold_independent": ind})
    # (3) 격자의 한 점이 어느 칸에서 빠졌다 — 시드·칸이 다른 격자를 쓰면 비교가 깨진다
    assert blocked({"curve": _curve(drop_point=True), "threshold_independent": ind})
    # (4) 임계 독립 헤드라인 지표의 회복률이 없다
    assert blocked({"curve": _curve(), "threshold_independent": _indep(with_recovery=False)})
    # (5) 키는 있는데 값이 None — 칸이 모자라 산출되지 않은 상태. 키만 세면 통과한다
    assert blocked({"curve": _curve(), "threshold_independent": _indep(pct=None)})


# --------------------------------------------------------------------------------------
# p9_source_separation — 출처별 분리 보고 필수화 (총괄 판정 2 · 22번 §2-2-2)
# --------------------------------------------------------------------------------------

def _p9(equivalent=False, client="C1", drop=None) -> dict:
    row = {
        "cell": "sep_local", "client": client, "tag": "sep_local_C1", "seed": 1,
        "fp_rate_crop": {"point": 0.49}, "fp_rate_tile": {"point": 0.002},
        "fp_rate_diff": {"point": -0.49, "ci_lo": -0.53, "ci_hi": -0.45},
        "tost": {"equivalent": equivalent},
        "fp_breakdown": {"N-crop": {"fp_rate": 0.49}, "N-tile": {"fp_rate": 0.002}},
    }
    if drop:
        row.pop(drop)
    return {"results": [row]}


def _note(lines=("함정 #11 …",)) -> dict:
    return {"lines": list(lines)}


def test_p9_gate_skips_without_block() -> None:
    r = _result(run_scoring_gates(GateContext(env={})), "p9_source_separation")
    assert r["skipped"] and r["passed"]


def test_p9_gate_passes_with_separation_even_when_not_equivalent() -> None:
    """**동등하지 않다는 사실은 차단 사유가 아니다** — 총괄이 알고 보류를 정했다.
    차단하는 것은 분리 표가 빠지는 것이다."""
    out = run_scoring_gates(GateContext(
        env={}, extra={"p9": _p9(equivalent=False), "shortcut_footnote": _note()}))
    r = _result(out, "p9_source_separation")
    assert r["passed"] and r["blocking"] and r["value"]["n_not_equivalent"] == 1
    assert "p9_source_separation" not in out["blocking_failures"]


def test_p9_gate_has_teeth() -> None:
    def blocked(extra: dict) -> bool:
        out = run_scoring_gates(GateContext(env={}, extra=extra))
        return "p9_source_separation" in out["blocking_failures"]

    # (1) 결과가 비어 있다
    assert blocked({"p9": {"results": []}, "shortcut_footnote": _note()})
    # (2) 출처별 항목이 빠졌다 — 오탐률·차·CI·TOST 병기가 필수다
    for key in ("fp_rate_crop", "fp_rate_tile", "fp_rate_diff", "tost", "fp_breakdown"):
        assert blocked({"p9": _p9(drop=key), "shortcut_footnote": _note()}), key
    # (3) sep_local 행에 client 가 없다 — 3모델이 이름으로 구분되지 않는다
    assert blocked({"p9": _p9(client=None), "shortcut_footnote": _note()})
    # (4) 전역 표의 규격 지름길 각주가 비어 있다
    assert blocked({"p9": _p9(), "shortcut_footnote": _note(lines=())})
