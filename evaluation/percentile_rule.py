"""백분위 구간의 규칙 — 2026-10-01 채택한 구간 규약(의사결정로그 10-01 · 결정표 11행)의 계산 쪽.

회복률만의 규칙이 아니다. 통합형 산출물에 실리는 **모든 백분위 구간**(회복률 · 분모 · 대응 차 · 참여자 보고 ·
출처 진단)이 이 한 규칙을 쓴다. 회복률의 추첨 분류(분모 실패 · 동률 · 부호 혼재 · 지지량)는
`evaluation.recovery_interval` 이 하고, 여기는 **이미 분류된 추첨값**에서 끝점을 낸다.

## 규칙 — 식으로 적는다

`n` 은 집계에 들어가는 추첨 수, `x` 는 그 추첨값의 오름차순(0부터 센다)이다. ±무한은 순서의 양 끝에 그대로 놓는다.

```
하한 자리 k = ((n − 1) × 25) // 1000        # 정수 나눗셈 — α = 0.05 일 때. 일반형은 ⌊(n − 1)·α/2⌋ 를 정확한 유리수로
상한 자리   = (n − 1) − k
끝점        = 그 자리의 추첨값 그대로. 보간하지 않는다
```

이 식이 라이브러리의 이름 있는 방식과 같은 값을 내는지는 시험이 맞댄다. **이름으로 등록하지 않는다** — 판이 바뀌어도 식은 같다.

## 상태 — 수 칸은 유한일 때만 수다

| 칸 | 상태 |
|---|---|
| 끝점 | `finite` · `pos_inf` · `neg_inf` · `absent`(집계 추첨 0건) |
| 반폭 | `finite` · `infinite`(한쪽이라도 무한, 또는 −∞~+∞) · `undefined`(두 끝점이 같은 부호의 무한) · `absent` |

수 칸은 상태가 `finite` 일 때만 수이고 아니면 `None` 이다 — 표준 JSON 이 `NaN` · `Infinity` 를 담지 못한다.

**방식은 인자로 받는다.** 등록 규칙이 아닌 방식(라이브러리의 방식 이름)을 주면 구간의 규약 id 가 달라지고,
보간하는 방식은 무한 꼬리에서 끝점이 정해지지 않을 수 있어 끝점 상태 `undefined` 가 하나 더 생긴다.
그 상태는 **등록 규칙에서는 나오지 않는다.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

CONVENTION_ID = "interval-20261001-1"
"""채택한 구간 규약의 id — 분모 실패는 분자 부호의 ±무한 · 동률은 판정 불가 · 무한은 끝값으로 포함 ·
아래 순위 식 · 세 상태 표현. **구간마다 싣는다.** 사전실험 구간은 다른 규약이다 — `PRE_EXPERIMENT_CONVENTION`."""

PRE_EXPERIMENT_CONVENTION = "pre-drop-undefined-numpy-default"
"""사전실험(분리형) 회복률 구간의 규약 — 분모 실패 추첨을 버리고 라이브러리 기본 보간으로 낸다
(`evaluation/recovery_ci.py`, 동결). 두 구간을 한 표에 놓을 때 규약이 다르다고 적는 데 쓴다. 소급 계산은 하지 않는다."""

PERCENTILE_RULE = ("k = ((n-1)*25)//1000; lo = x[k]; hi = x[(n-1)-k]; "
                   "x ascending, 0-based, +-inf at the ends; no interpolation")
"""등록 칸 `ci.percentile_method` 의 값 — **식**이다. 25/1000 은 α/2 이므로 등록의 α 가 0.05 여야 한다."""

INFINITY_HANDLING = "include_as_extreme"
"""등록 칸 `ci.infinity_handling` 의 값 — ±무한을 버리지 않고 순서의 끝값으로 포함한다."""

UNDEFINED_DENOMINATOR = "signed_infinity"
"""등록 칸 `ci.undefined_denominator_policy` 의 값 — 분모 `D ≤ 0` 인 추첨의 `R` 을 분자 부호의 ±무한으로 둔다."""

ABSENCE_REPRESENTATION = "state_field"
"""등록 칸 `ci.interval_absence_representation` 의 값 — 수 칸 옆에 상태 칸을 둔다."""

ENDPOINT_STATES = ("finite", "pos_inf", "neg_inf", "absent")
HALF_WIDTH_STATES = ("finite", "infinite", "undefined", "absent")

UNDEFINED = "undefined"
"""끝점의 넷째 상태 — **등록 규칙이 아닌 보간 방식**에서만 나온다(두 이웃이 무한이면 보간값이 정해지지 않는다)."""

NUMPY_METHODS = ("linear", "lower", "higher", "nearest", "midpoint", "inverted_cdf", "averaged_inverted_cdf",
                 "closest_observation", "interpolated_inverted_cdf", "hazen", "weibull", "median_unbiased",
                 "normal_unbiased")
"""등록 규칙 밖에서 받는 방식 — 라이브러리의 방식 이름. 쓰면 규약 id 에 이름과 라이브러리 판이 붙는다."""


REGISTERED_ALPHA = 0.05
"""등록 규칙의 α — 순위 식의 25/1000 이 이 값의 α/2 다(07번 §33-4). 다른 α 의 구간은 채택 규약이 아니다."""


def convention_id(method: str = PERCENTILE_RULE, alpha: float = REGISTERED_ALPHA) -> str:
    """구간의 규약 id. 등록 규칙이고 α = 0.05 이면 `CONVENTION_ID` 다.

    방식이 다르면 방식 이름과 라이브러리 판을, α 가 다르면 α 를 붙인다 — 같은 순위 식을 다른 α 로 계산한 구간이
    채택 규약의 id 를 달지 않게(07번 §33-4 · §33-6).
    """
    _check_method(method)
    cid = CONVENTION_ID if method == PERCENTILE_RULE else f"{CONVENTION_ID}~method={method}@numpy-{np.__version__}"
    if alpha != REGISTERED_ALPHA:
        _half_alpha(alpha)
        cid += f"~alpha={float(alpha)!r}"
    return cid


def _check_method(method: str) -> None:
    if method != PERCENTILE_RULE and method not in NUMPY_METHODS:
        raise ValueError(f"백분위 방식이 등록 규칙도 라이브러리 방식 이름도 아니다: {method!r}")


def _half_alpha(alpha: float) -> Fraction:
    """α/2 를 **정확한 유리수**로. 부동소수 0.05 를 그대로 유리수로 바꾸면 1/20 이 아니다 — 십진 표기에서 읽는다."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError(f"α 는 (0, 1) 의 수다: {alpha!r}")
    return Fraction(repr(float(alpha))) / 2


def rank_positions(n: int, alpha: float = 0.05) -> tuple[int, int]:
    """등록 규칙의 두 자리 `(k, (n − 1) − k)`. `k = ⌊(n − 1)·α/2⌋` 를 정수로 — α = 0.05 이면 `((n − 1)·25)//1000`."""
    if type(n) is not int or n < 1:
        raise ValueError(f"집계 추첨 수는 1 이상의 정수다: {n!r}")
    h = _half_alpha(alpha)
    k = ((n - 1) * h.numerator) // h.denominator
    return k, (n - 1) - k


@dataclass(frozen=True)
class Bound:
    """수 하나와 그 상태. `value` 는 상태가 `finite` 일 때만 수다."""

    value: float | None
    state: str

    def __post_init__(self) -> None:
        if (self.state == "finite") != (self.value is not None):
            raise ValueError(f"수 칸과 상태가 어긋난다: {self.value!r} · {self.state!r}")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"유한 상태의 수가 유한하지 않다: {self.value!r}")


def bound_of(x: float) -> Bound:
    """계산이 낸 수 하나를 상태로 옮긴다. `NaN` 은 `undefined` 다 — 0 이나 유한값으로 채우지 않는다."""
    x = float(x)
    if math.isnan(x):
        return Bound(None, UNDEFINED)
    if math.isinf(x):
        return Bound(None, "pos_inf" if x > 0 else "neg_inf")
    return Bound(x, "finite")


ABSENT = Bound(None, "absent")


def half_width_of(lo: Bound, hi: Bound) -> Bound:
    """반폭. 두 끝점이 같은 부호의 무한이면 `무한 − 무한` 이라 `undefined`(§25-2 의 가5)."""
    if lo.state == "absent" or hi.state == "absent":
        return ABSENT
    if UNDEFINED in (lo.state, hi.state):
        return Bound(None, UNDEFINED)
    if lo.state == "finite" and hi.state == "finite":
        return Bound((hi.value - lo.value) / 2, "finite")
    if lo.state == hi.state:
        return Bound(None, UNDEFINED)
    return Bound(None, "infinite")


@dataclass(frozen=True)
class RuleInterval:
    """백분위 구간 하나. **규약 id 를 늘 싣는다.**"""

    lo: Bound
    hi: Bound
    half_width: Bound
    n: int
    """집계에 들어간 추첨 수 — 유한과 ±무한 둘 다."""
    convention_id: str

    def as_dict(self) -> dict:
        """표준 JSON 으로 쓸 수 있는 꼴. 수 칸은 유한일 때만 수다."""
        return {"ci_lo": self.lo.value, "ci_lo_state": self.lo.state,
                "ci_hi": self.hi.value, "ci_hi_state": self.hi.state,
                "half_width": self.half_width.value, "half_width_state": self.half_width.state,
                "n": self.n, "convention_id": self.convention_id}


def quantile_bound(values, q: float, method: str) -> Bound:
    """라이브러리 방식 하나로 백분위 `q`(0~100)의 끝점. 무한이 끼면 보간 방식은 `NaN` 을 낼 수 있다 — 상태 `undefined`."""
    if method not in NUMPY_METHODS:
        raise ValueError(f"라이브러리 방식 이름이 아니다: {method!r}")
    arr = _as_draws(values)
    if arr.size == 0:
        return ABSENT
    with np.errstate(invalid="ignore"):
        return bound_of(np.percentile(arr, q, method=method))


def _as_draws(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if np.isnan(arr).any():
        raise ValueError("추첨값에 NaN 이 있다 — 집계 전에 제외·판정 불가로 분류해야 한다(버리지 않는다)")
    return arr


def interval(values, alpha: float = 0.05, *, method: str = PERCENTILE_RULE) -> RuleInterval:
    """집계에 들어가는 추첨값(유한과 ±무한)에서 구간을 낸다. `NaN` 이 있으면 거부한다.

    추첨이 0건이면 두 끝점과 반폭이 `absent` 다(§25-2 의 가4).
    """
    _check_method(method)
    arr = _as_draws(values)
    cid = convention_id(method, alpha)
    if arr.size == 0:
        _half_alpha(alpha)
        return RuleInterval(ABSENT, ABSENT, ABSENT, 0, cid)
    if method == PERCENTILE_RULE:
        k_lo, k_hi = rank_positions(int(arr.size), alpha)
        x = np.sort(arr)
        lo, hi = bound_of(x[k_lo]), bound_of(x[k_hi])
    else:
        h = float(_half_alpha(alpha))
        lo = quantile_bound(arr, 100 * h, method)
        hi = quantile_bound(arr, 100 * (1 - h), method)
    return RuleInterval(lo, hi, half_width_of(lo, hi), int(arr.size), cid)


def defined_interval(values, alpha: float = 0.05, *, method: str = PERCENTILE_RULE) -> tuple[RuleInterval, int, int]:
    """**미정의를 빼는** 통계량(규칙 ⑤ 의 대응 차 · 출처 진단)의 구간. `(구간, n_defined, n_undefined)`.

    `NaN` 은 미정의로 세고 뺀다. 이런 통계량에는 무한이 생길 자리가 없으므로 ±무한이 있으면 거부한다 —
    회복률(무한을 넣는다)과 어휘를 섞지 않는다(§24-3).
    """
    arr = np.asarray(values, dtype=float).ravel()
    if np.isinf(arr).any():
        raise ValueError("미정의를 빼는 통계량에 ±무한이 있다 — 회복률의 규약과 섞였다")
    ok = ~np.isnan(arr)
    return interval(arr[ok], alpha, method=method), int(ok.sum()), int((~ok).sum())
