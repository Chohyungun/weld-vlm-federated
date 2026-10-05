"""통합형 회복률의 추첨 분류와 구간 — 2026-10-01 채택한 구간 규약(의사결정로그 10-01 · 결정표 11행).

동결 모듈 `evaluation/recovery_ci.py` 는 고치지 않는다. 그 모듈은 분모 실패 추첨을 **버리고** 라이브러리 기본 보간으로
구간을 내며, 사전실험의 값은 그 계약의 값이다. 여기는 새 계약이다 — 07번 미니스펙 §15-6 · §24-3 · §25-2 와 §33.

`R = (F − L̄) / (C − L̄)` — 분자 `N = F − L̄`, 분모 `D = C − L̄`, `L̄` 은 세 로컬의 평균이다(§24-4).
`L̄` 은 `로컬의 합 ÷ 로컬 수` 로 계산한다 — 점추정과 추첨이 같은 식을 쓴다.

## 시드 하나의 `R_s`

| 조건 | `R_s` |
|---|---|
| `D > 0` | `N / D` (유한) |
| `D ≤ 0` 이고 `N > 0` | `+∞` |
| `D ≤ 0` 이고 `N < 0` | `−∞` |
| `D ≤ 0` 이고 `N == 0` | **판정 불가**(`numerator_zero`) — 끝없이 좋은 회복으로 세지 않는다 |

`N == 0` 은 **부동소수 정확 비교**다(`−0.0` 도 0 이다). 허용 오차를 두지 않는다.
`±무한` 의 부호는 비율의 값이 아니라 **약속**이다 — `D < 0` 에서 대수적 비율의 부호는 분자와 반대다.

## 추첨 하나의 분류 — 정확히 한 범주

순서대로 본다. 앞에서 걸리면 뒤를 보지 않는다.

| 순서 | 조건 | 범주 · 사유 | 백분위 집계 |
|---|---|---|---|
| 1 | 평균 대상 클래스의 정답 지지량이 0 | `n_excluded_support` · `support` | 안 들어간다 |
| 2 | 어느 시드가 `numerator_zero` | `n_indeterminate` · `numerator_zero` | 안 들어간다 |
| 3 | 시드들에 `+∞` 와 `−∞` 가 함께 | `n_indeterminate` · `sign_conflict` | 안 들어간다 |
| 4 | `+∞` 가 있다 | `n_infinite_pos` · `denominator_pos` | 들어간다(위 꼬리) |
| 5 | `−∞` 가 있다 | `n_infinite_neg` · `denominator_neg` | 들어간다(아래 꼬리) |
| 6 | 그 밖 | `n_finite` — `R̄` 는 `R_s` 의 평균 | 들어간다 |

회계: 다섯 카운터의 합 == 재표집 수. 백분위의 `n` 은 `n_finite + n_infinite_pos + n_infinite_neg` 다.
규칙 ④ 의 분자는 `n_excluded_support + n_indeterminate` 이고 **분모 실패는 들지 않는다** — 이미 구간 안에 있다.

## 점추정

어느 시드든 `D_s ≤ 0` 이면 그 시드의 `R_s` 와 `R̄` 를 **내지 않는다**(상태 `absent` 와 사유). 점추정에 무한을 적지 않는다 —
추첨의 ±무한은 순서를 주려는 약속이고 점추정에는 세울 순서가 없다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation import percentile_rule as PR

SUPPORT = "support"
NUMERATOR_ZERO = "numerator_zero"
SIGN_CONFLICT = "sign_conflict"
DENOMINATOR_POS = "denominator_pos"
DENOMINATOR_NEG = "denominator_neg"
INDETERMINATE_REASONS = (NUMERATOR_ZERO, SIGN_CONFLICT)
COUNTERS = ("n_finite", "n_infinite_pos", "n_infinite_neg", "n_excluded_support", "n_indeterminate")

NUMERATOR_ZERO_RULE = "indeterminate"
"""등록 칸 `ci.numerator_zero_rule` — 분모 실패 추첨에서 분자가 0 이면 판정 불가로 센다."""
NUMERATOR_ZERO_TEST = "exact_float_equality"
"""등록 칸 `ci.numerator_zero_test` — `N == 0.0`(IEEE 배정밀도, `−0.0` 포함). 허용 오차 없음."""
PERCENTILE_N = "n_finite + n_infinite_pos + n_infinite_neg"
"""등록 칸 `ci.percentile_n_definition` — 백분위의 `n` 은 집계에 들어가는 추첨 수."""
POINT_DENOMINATOR_FAILURE = "absent"
"""등록 칸 `ci.point_denominator_failure` — 점추정의 분모 실패는 내지 않는다."""
RULE4_COUNTS = (SUPPORT, "indeterminate")
"""등록 칸 `ci.rule4_counts` — 규칙 ④ 가 세는 범주. 분모 실패는 들지 않는다."""
INTERVAL_SCOPE = "all_percentile_intervals_in_unified_outputs"
"""등록 칸 `ci.interval_scope` — 통합형 산출물에 실리는 모든 백분위 구간이 한 규칙을 쓴다."""

_CODE_FINITE, _CODE_POS, _CODE_NEG, _CODE_SUPPORT, _CODE_ZERO, _CODE_CONFLICT = range(6)


def _local_mean(locals_: np.ndarray) -> np.ndarray:
    return locals_.sum(axis=-1) / locals_.shape[-1]


def _check_inputs(fed, central, locals_, *, draws: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = np.asarray(fed, dtype=float)
    c = np.asarray(central, dtype=float)
    l = np.asarray(locals_, dtype=float)
    want = 2 if draws else 1
    if f.ndim != want or c.shape != f.shape or l.ndim != want + 1 or l.shape[:-1] != f.shape or l.shape[-1] < 1:
        axes = "(추첨, 시드)" if draws else "(시드,)"
        raise ValueError(f"모양이 어긋난다 — 연합·중앙 {axes}, 로컬 {axes[:-1]}, 로컬 수): "
                         f"{f.shape} · {c.shape} · {l.shape}")
    if f.shape[-1] < 1:
        raise ValueError("시드가 없다")
    for name, a in (("연합", f), ("중앙", c), ("로컬", l)):
        if not np.isfinite(a).all():
            raise ValueError(f"{name} 값에 유한하지 않은 수가 있다 — 회복률의 입력은 유효한 지표값이다(§24-4)")
    return f, c, l


def _seed_ratios(f: np.ndarray, c: np.ndarray, l: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(R_s, 분모 실패, 분자 0)`. 분모 실패 자리의 `R_s` 는 분자 부호의 ±무한이고, 분자 0 이면 `NaN` 이다."""
    lbar = _local_mean(l)
    num = f - lbar
    den = c - lbar
    fail = den <= 0
    zero = fail & (num == 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(fail, np.where(num > 0, np.inf, np.where(num < 0, -np.inf, np.nan)), num / np.where(fail, 1.0, den))
    return r, fail, zero


def classify_draws(fed, central, locals_, support_ok) -> tuple[np.ndarray, np.ndarray]:
    """추첨마다 `(R̄ 값, 범주 부호)`. 집계에 들어가지 않는 추첨의 값은 `NaN` 이다(집계 전에 빠진다).

    입력 모양 — 연합·중앙 `(추첨, 시드)`, 로컬 `(추첨, 시드, 로컬 수)`, 지지량 `(추첨,)` 참·거짓.
    """
    f, c, l = _check_inputs(fed, central, locals_, draws=True)
    s = np.asarray(support_ok)
    if s.dtype != bool or s.shape != f.shape[:1]:
        raise ValueError(f"지지량 표시는 추첨마다 참·거짓 하나다: {s.dtype} {s.shape}")
    r, _fail, zero = _seed_ratios(f, c, l)
    has_zero = zero.any(axis=1)
    has_pos = (r == np.inf).any(axis=1)
    has_neg = (r == -np.inf).any(axis=1)
    code = np.full(f.shape[0], _CODE_FINITE)
    code[has_neg] = _CODE_NEG
    code[has_pos] = _CODE_POS
    code[has_pos & has_neg] = _CODE_CONFLICT
    code[has_zero] = _CODE_ZERO
    code[~s] = _CODE_SUPPORT
    value = np.full(f.shape[0], np.nan)
    fin = code == _CODE_FINITE
    value[fin] = r[fin].sum(axis=1) / r.shape[1]
    value[code == _CODE_POS] = np.inf
    value[code == _CODE_NEG] = -np.inf
    return value, code


@dataclass(frozen=True)
class RecoveryInterval:
    interval: PR.RuleInterval
    counters: dict[str, int]
    indeterminate_reasons: dict[str, int]
    n_resamples: int
    rule4_fraction: float
    """규칙 ④ 의 분자 비율 — `(n_excluded_support + n_indeterminate) / n_resamples`."""

    def as_dict(self) -> dict:
        return {**self.interval.as_dict(), **self.counters, "indeterminate_reasons": dict(self.indeterminate_reasons),
                "n_resamples": self.n_resamples, "rule4_fraction": self.rule4_fraction,
                "rule4_counts": list(RULE4_COUNTS)}


def recovery_interval(fed, central, locals_, support_ok, *, alpha: float, n_resamples: int,
                      method: str = PR.PERCENTILE_RULE) -> RecoveryInterval:
    """추첨을 분류하고 집계에 들어가는 것만으로 구간을 낸다. 추첨 수가 `n_resamples` 와 다르면 거부한다(회계)."""
    value, code = classify_draws(fed, central, locals_, support_ok)
    if type(n_resamples) is not int or value.shape[0] != n_resamples:
        raise ValueError(f"추첨 수 {value.shape[0]} 가 등록한 재표집 수 {n_resamples!r} 와 다르다")
    counts = np.bincount(code, minlength=6)
    counters = {"n_finite": int(counts[_CODE_FINITE]), "n_infinite_pos": int(counts[_CODE_POS]),
                "n_infinite_neg": int(counts[_CODE_NEG]), "n_excluded_support": int(counts[_CODE_SUPPORT]),
                "n_indeterminate": int(counts[_CODE_ZERO] + counts[_CODE_CONFLICT])}
    reasons = {NUMERATOR_ZERO: int(counts[_CODE_ZERO]), SIGN_CONFLICT: int(counts[_CODE_CONFLICT])}
    if sum(counters.values()) != n_resamples:
        raise AssertionError("회계가 맞지 않는다 — 범주가 겹치거나 빠졌다")
    kept = value[code <= _CODE_NEG]
    iv = PR.interval(kept, alpha, method=method)
    if iv.n != counters["n_finite"] + counters["n_infinite_pos"] + counters["n_infinite_neg"]:
        raise AssertionError("백분위의 n 이 정의와 다르다")
    r4 = (counters["n_excluded_support"] + counters["n_indeterminate"]) / n_resamples
    return RecoveryInterval(iv, counters, reasons, n_resamples, r4)


def point_recovery(fed, central, locals_) -> dict:
    """시드별 `R_s` 와 `R̄`. 분모 실패 시드가 하나라도 있으면 `R̄` 를 내지 않는다 — 상태 `absent` 와 사유."""
    f, c, l = _check_inputs(fed, central, locals_, draws=False)
    lbar = _local_mean(l)
    num, den = f - lbar, c - lbar
    seeds = []
    for i in range(f.shape[0]):
        if den[i] > 0:
            seeds.append({"R": float(num[i] / den[i]), "state": "finite", "reason": None, "D": float(den[i])})
        else:
            seeds.append({"R": None, "state": "absent", "reason": "denominator", "D": float(den[i])})
    ok = all(s["state"] == "finite" for s in seeds)
    rbar = {"R_bar": float(sum(s["R"] for s in seeds) / len(seeds)), "state": "finite", "reason": None} if ok else \
        {"R_bar": None, "state": "absent", "reason": "denominator"}
    return {"seeds": seeds, **rbar, "convention_id": PR.CONVENTION_ID}


REGISTERED_CI = {
    "undefined_denominator_policy": PR.UNDEFINED_DENOMINATOR,
    "infinity_handling": PR.INFINITY_HANDLING,
    "percentile_method": PR.PERCENTILE_RULE,
    "interval_absence_representation": PR.ABSENCE_REPRESENTATION,
    "endpoint_states": PR.ENDPOINT_STATES,
    "half_width_states": PR.HALF_WIDTH_STATES,
    "numerator_zero_rule": NUMERATOR_ZERO_RULE,
    "numerator_zero_test": NUMERATOR_ZERO_TEST,
    "percentile_n_definition": PERCENTILE_N,
    "point_denominator_failure": POINT_DENOMINATOR_FAILURE,
    "rule4_counts": RULE4_COUNTS,
    "interval_scope": INTERVAL_SCOPE,
    "convention_id": PR.CONVENTION_ID,
}
"""등록 블록 `ci` 의 규약 칸 → 이 모듈이 구현한 값. 등록이 다른 값을 적었으면 이 모듈로 채점하지 않는다."""


def registered_mismatches(ci) -> list[str]:
    """등록의 `ci` 가 이 모듈이 구현한 규약과 어긋나는 칸. 비어 있는 칸도 어긋남이다."""
    out = []
    for key, want in REGISTERED_CI.items():
        got = getattr(ci, key, None)
        if (tuple(got) if isinstance(got, list) else got) != want:
            out.append(f"ci.{key}: 등록 {got!r} · 구현 {want!r}")
    if getattr(ci, "alpha", None) != 0.05:
        out.append(f"ci.alpha: 순위 식의 25/1000 은 α = 0.05 의 α/2 다 — 등록 {getattr(ci, 'alpha', None)!r}")
    return out


def recovery_interval_registered(ci, fed, central, locals_, support_ok) -> RecoveryInterval:
    """등록한 규약으로만 낸다. 등록의 규약 칸 · α · 재표집 수를 읽고, 규약이 어긋나면 거부한다."""
    bad = registered_mismatches(ci)
    if bad:
        raise ValueError("등록한 구간 규약이 이 구현과 다르다: " + " · ".join(bad))
    return recovery_interval(fed, central, locals_, support_ok, alpha=ci.alpha, n_resamples=ci.n_resamples)
