"""게이트 레지스트리 — **만들어 두고 부르지 않는 검사를 없앤다.** 체크리스트 18.

80번 §8 이 이번 감사의 형태를 한 줄로 정리했다.

> 반복해서 나온 형태는 "결함이 있는데 시험이 초록"이 아니라 **"검사를 만들어 두고
> 부르지 않았다"** 였다. 정본 검증기 두 개, 게이트 함수 여섯 개, 증빙 필드 넷, 집계 규약
> 세 항목이 전부 같은 방식으로 죽어 있었다.

`verify_against_prereg`·`recovery_denominator_ok`·`guard_no_cloud_logging`·`require_tags`
는 전부 자기 시험 외 호출처가 0건이었다. 함수가 존재한다는 사실이 규칙이 지켜진다는
증거가 아니다.

**이 파일의 설계 제약은 그 하나다.** 게이트는 레지스트리에 등록되고, 채점이 끝나면
`run_scoring_gates()` 가 등록된 것을 **전부** 돌리며, 산출물에 `gates_evaluated` 로
이름과 결과가 남는다. 등록됐는데 안 불린 게이트가 있으면 시험이 깨진다
(`tests/test_gate_registry.py`) — 레지스트리 자체가 무이빨이 되는 것을 막는 장치다.

블로킹 여부는 게이트마다 다르다. 지금 `gate_status: 판정_대기` 인 항목은 **기록하되
차단하지 않는다**(76번 §1-4, A 가 박아 둔 스위치). 차단하지 않는 것과 재지 않는 것은
다르다 — 재고, 기록하고, 차단만 보류한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

GateFn = Callable[..., "GateResult"]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str
    blocking: bool = True
    """False 면 결과만 남기고 진행을 막지 않는다. **재지 않는다는 뜻이 아니다.**"""
    skipped: bool = False
    """입력이 없어 판정 자체를 못 한 경우. `passed=True` 와 구분해서 센다 —
    섞으면 "안 돌았는데 통과"가 만들어진다."""
    value: Any = None

    def as_dict(self) -> dict:
        return {
            "name": self.name, "passed": self.passed, "skipped": self.skipped,
            "blocking": self.blocking, "detail": self.detail, "value": self.value,
        }


REGISTRY: dict[str, GateFn] = {}


def register(name: str) -> Callable[[GateFn], GateFn]:
    def deco(fn: GateFn) -> GateFn:
        if name in REGISTRY:
            raise ValueError(f"게이트 이름 중복: {name}")
        REGISTRY[name] = fn
        return fn
    return deco


# --------------------------------------------------------------------------------------
# 게이트 구현 — 전부 `ctx` 하나만 받는다. 시그니처가 같아야 레지스트리가 전수 호출할 수 있다.
# --------------------------------------------------------------------------------------

@dataclass
class GateContext:
    """게이트가 보는 것 전부. 없는 항목은 `None` 이고 그 게이트는 `skipped` 가 된다."""

    metrics: Mapping[str, Mapping[str, Any]] | None = None
    records_by_cell: Mapping[str, Sequence[Any]] | None = None
    expected_coord_space: str | None = None
    population_bound: float | None = None
    n_eval: int | None = None
    n_scored: Mapping[str, int] | None = None
    recovery: Mapping[str, Any] | None = None
    seed_sd: float | None = None
    env: Mapping[str, str] | None = None
    tags: Mapping[str, str] | None = None
    gate_status: str = "미등록"
    extra: dict = field(default_factory=dict)


@register("prereg_constants_reproduced")
def _gate_prereg(ctx: GateContext) -> GateResult:
    """재산출 상수가 등록 상수를 재현하는가 (`verify_against_prereg` 실호출)."""
    from evaluation.prereg import PREREG, verify_against_prereg

    measured = ctx.extra.get("measured_prereg")
    if not measured:
        return GateResult("prereg_constants_reproduced", True,
                          "재산출값 미제공 — 판정 안 함", skipped=True)
    ok, msg = verify_against_prereg(
        measured["all_positive_macro_f1"], measured["spec_only_macro_f1"])
    return GateResult("prereg_constants_reproduced", ok, msg,
                      value={"measured": measured,
                             "registered": PREREG.all_positive_macro_f1})


@register("recovery_denominator")
def _gate_recovery(ctx: GateContext) -> GateResult:
    """회복률 분모 규칙 (`recovery_denominator_ok` 실호출).

    산출부가 `if sep_denom > 0` 한 줄만 보고 회복률을 냈다 — 시드 1세트에서 분모
    0.09918 위에 `-29.16%` 가 실렸다(80번 D10). 규칙은 D ≥ 3·시드 sd 다.
    """
    from evaluation.prereg import recovery_denominator_ok

    rec = (ctx.recovery or {}).get("separated") if ctx.recovery else None
    if not rec:
        return GateResult("recovery_denominator", True, "회복률 미산출 — 판정 안 함",
                          skipped=True)
    if ctx.seed_sd is None:
        return GateResult(
            "recovery_denominator", False,
            "시드 sd 가 없다. 시드 1세트에서는 분모 규칙을 판정할 수 없으므로 "
            "**회복률을 헤드라인으로 쓰지 않는다**(사전등록 §5-4)",
            blocking=False, value={"denominator": rec.get("denominator")})
    ok, d, msg = recovery_denominator_ok(
        float(rec["central"]), float(rec["local_mean"]), float(ctx.seed_sd))
    return GateResult("recovery_denominator", ok, msg, value={"denominator": d})


@register("no_cloud_logging")
def _gate_cloud(ctx: GateContext) -> GateResult:
    """외부 클라우드 로깅 환경변수 (`guard_no_cloud_logging` 실호출, `MLFLOW_` 포함)."""
    from tracking.mlflow_local import (
        CLOUD_ENV_PREFIXES,
        CloudLoggingBlocked,
        guard_no_cloud_logging,
    )

    try:
        guard_no_cloud_logging(ctx.env)
    except CloudLoggingBlocked as e:
        return GateResult("no_cloud_logging", False, str(e))
    return GateResult("no_cloud_logging", True,
                      f"차단 접두 {len(CLOUD_ENV_PREFIXES)}종 모두 미검출",
                      value={"prefixes": list(CLOUD_ENV_PREFIXES)})


@register("required_tags")
def _gate_tags(ctx: GateContext) -> GateResult:
    """필수 로깅 태그 11종 (`require_tags` 실호출)."""
    from tracking.mlflow_local import (
        REQUIRED_TAGS,
        MissingRunMetadata,
        require_tags,
    )

    if ctx.tags is None:
        return GateResult(
            "required_tags", False,
            f"태그가 제공되지 않았다. 필수 {len(REQUIRED_TAGS)}종이 run 종료 시점에 "
            "채워져 있어야 한다(§8-2). 채점 단계에서는 차단하지 않는다",
            blocking=False, skipped=True, value={"required": list(REQUIRED_TAGS)})
    try:
        require_tags(ctx.tags)
    except MissingRunMetadata as e:
        return GateResult("required_tags", False, str(e))
    return GateResult("required_tags", True, f"필수 태그 {len(REQUIRED_TAGS)}종 전부 존재")


@register("coord_space_contract")
def _gate_coord(ctx: GateContext) -> GateResult:
    """다섯 칸이 **같은 좌표 규약**을 선언하는가 (총괄 판정 1, main 47c4dbc).

    D 는 좌표를 변환하지 않는다. 그래서 이 게이트가 좌표 축의 유일한 방어선이다 —
    규약이 갈린 레코드가 같은 표에 실리는 것이 함정 #4 다.
    """
    if not ctx.records_by_cell or not ctx.expected_coord_space:
        return GateResult("coord_space_contract", True, "레코드 미제공 — 판정 안 함",
                          skipped=True)
    seen: dict[str, dict[str, int]] = {}
    for tag, recs in ctx.records_by_cell.items():
        per: dict[str, int] = {}
        for r in recs:
            k = str(getattr(r, "coord_space", None))
            per[k] = per.get(k, 0) + 1
        seen[tag] = per
    bad = {t: p for t, p in seen.items()
           if set(p) != {ctx.expected_coord_space}}
    if not bad:
        return GateResult("coord_space_contract", True,
                          f"전 칸 {ctx.expected_coord_space} 선언", value=seen)
    return GateResult(
        "coord_space_contract", False,
        f"규약이 기대값({ctx.expected_coord_space})과 다른 칸이 있다: "
        f"{ {t: sorted(p) for t, p in bad.items()} }. "
        "파일럿 통합형 산출물은 NORM_1000 시절의 것이라 재실행 대상이다(80번 E3·인용 금지)",
        blocking=False, value=seen)


@register("scoring_population")
def _gate_population(ctx: GateContext) -> GateResult:
    """다섯 칸이 **같은 모집단 전량**으로 채점됐는가.

    정상 이미지가 위치·mAP 축에서 빠져 있던 것이 80번 D9 다. 여기서 매 채점마다 센다.
    """
    if ctx.n_eval is None or not ctx.n_scored:
        return GateResult("scoring_population", True, "모집단 정보 미제공 — 판정 안 함",
                          skipped=True)
    bad = {t: n for t, n in ctx.n_scored.items() if n != ctx.n_eval}
    if bad:
        return GateResult("scoring_population", False,
                          f"평가셋 {ctx.n_eval}장과 다른 칸: {bad}", value=dict(ctx.n_scored))
    return GateResult("scoring_population", True,
                      f"전 칸 {ctx.n_eval}장 동일", value=dict(ctx.n_scored))


@register("content_free_gate")
def _gate_content_free(ctx: GateContext) -> GateResult:
    """content-free 천장 대조. `gate_status` 에 따라 차단 여부가 갈린다.

    `판정_대기` 면 **재고 기록하되 차단하지 않는다** — A 가 76번 §1-4 에서 "값은
    등록하되 자동 차단에는 쓰지 않는다"로 박아 둔 스위치다. 총괄 판정 6(을안)이
    층화 채점 병기를 정했으므로, 전역 Macro-F1 에 대한 이 선은 **한계 서술용**이다.
    """
    if not ctx.metrics or ctx.population_bound is None:
        return GateResult("content_free_gate", True, "지표 미제공 — 판정 안 함",
                          skipped=True)
    line = float(ctx.extra.get("gate_pass_line", ctx.population_bound))
    above = {t: m["macro_f1"] for t, m in ctx.metrics.items()
             if float(m["macro_f1"]) > line}
    blocking = ctx.gate_status == "적용"
    ok = bool(above)
    return GateResult(
        "content_free_gate", ok,
        f"통과선 {line} · {len(above)}/{len(ctx.metrics)} 통과 "
        f"(gate_status={ctx.gate_status}"
        f"{'' if blocking else ' — 기록만, 차단 안 함'})",
        blocking=blocking, value={"pass_line": line, "above": above})


@register("stratified_scoring")
def _gate_stratified(ctx: GateContext) -> GateResult:
    """총괄 판정 6 이행 — 층화 블록이 **같은 산출물 안에** 있고 계측기가 작동하는가.

    13번 D-1: 본채점 진입점이 층화 블록을 산출하지 않아 병기가 사람 손 절차(별도
    스크립트 실행)에 걸려 있었다. 이제 `score` 가 블록을 만들고 이 게이트가 매 채점마다
    본다 — (1) 기본 K 의 표가 있고, (2) 채점된 칸 전부에 행이 있으며, (3) 지름길 규칙
    행의 lift 가 정확히 0 이다. 0 이 아니면 층 정의나 기준선이 틀린 것이다(계측기 고장).
    """
    from evaluation.strata import SHORTCUT_TAG

    s = ctx.extra.get("stratified")
    if s is None:
        return GateResult("stratified_scoring", True, "층화 블록 미제공 — 판정 안 함",
                          skipped=True)
    k = str(s.get("default_k", ""))
    rows = (s.get("by_k") or {}).get(k)
    if not rows:
        return GateResult("stratified_scoring", False,
                          f"기본 K={k or '?'} 의 층화 표가 없다 — 판정 6(병기) 미이행")
    missing = sorted(set(ctx.metrics or {}) - set(rows))
    if missing:
        return GateResult("stratified_scoring", False,
                          f"층화 표에 없는 칸: {missing}", value={"k": k})
    sc = rows.get(SHORTCUT_TAG)
    if not sc:
        return GateResult("stratified_scoring", False,
                          "지름길 규칙 행이 없다 — 판별력 계측 부재", value={"k": k})
    lift = float(sc.get("stratified_lift", float("nan")))
    lift_w = float(sc.get("stratified_lift_weighted", float("nan")))
    summary = {
        "k": k,
        "n_cells": len(rows) - 1,
        "shortcut_global_macro_f1": sc.get("global_macro_f1"),
        "shortcut_stratified_macro_f1": sc.get("stratified_macro_f1"),
        "shortcut_lift": lift,
        "n_strata_impure_scored": sc.get("n_strata_impure_scored"),
    }
    if not (abs(lift) <= 1e-9 and abs(lift_w) <= 1e-9):
        return GateResult(
            "stratified_scoring", False,
            f"지름길 규칙의 lift 가 0 이 아니다 ({lift:+.3e} / 가중 {lift_w:+.3e}) — "
            "층 정의 또는 기준선 정의가 틀렸다", value=summary)
    return GateResult(
        "stratified_scoring", True,
        f"K={k} · 칸 {len(rows) - 1} + 지름길 · 지름길 lift {lift:+.1e} · "
        f"비순수 구간 {sc.get('n_strata_impure_scored')}", value=summary)


def run_scoring_gates(ctx: GateContext) -> dict:
    """등록된 게이트를 **전부** 돌린다. 골라 부르지 않는다.

    Returns:
        `gates_evaluated` 로 산출물에 실을 dict. 이름·통과·차단여부·사유가 전부 남는다.
    """
    results = [REGISTRY[name](ctx) for name in sorted(REGISTRY)]
    blocking_failures = [r.name for r in results
                         if r.blocking and not r.passed and not r.skipped]
    return {
        "n_registered": len(REGISTRY),
        "n_evaluated": len(results),
        "registered": sorted(REGISTRY),
        "results": [r.as_dict() for r in results],
        "n_skipped": sum(1 for r in results if r.skipped),
        "blocking_failures": blocking_failures,
        "ok": not blocking_failures,
    }
