"""묶음 단위 클러스터 부트스트랩 · TOST 등가성 검정 · 시드 분산 진단. §5-3·§5-4·D-4.

**모든 CI 는 묶음(pHash 중복 묶음) 단위로 재표집한다.** 이미지 단위로 내면 같은 용접부
연속 촬영이 독립 표본처럼 세어져 **CI 가 3배 좁게** 나온다. 좁은 CI 는 없는 유의성을
만들어낸다.

시드 3세트에서 표준편차를 추정하는 것 자체가 불안정하다는 점도 여기서 계량한다 —
자유도 2 의 표본 sd 는 변동계수가 0.52 다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import gamma, sqrt

import numpy as np

BOOTSTRAP_N = 2000
"""§5-3 고정값. 재현을 위해 시드도 함께 고정한다."""
BOOTSTRAP_SEED = 20260825


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n_resamples: int
    n_clusters: int
    n_undefined: int = 0
    """통계량이 정의되지 않은 재표집 수(`drop_undefined=True` 일 때만 0 이 아니다).

    **버린 사실을 숨기지 않는다** — 이 값이 크면 CI 가 좁아 보이는 것이 표본 성질이지
    정밀도가 아니다.
    """

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2

    def as_dict(self) -> dict:
        return {
            "point": self.point, "ci_lo": self.lo, "ci_hi": self.hi,
            "half_width": self.half_width,
            "n_resamples": self.n_resamples, "n_clusters": self.n_clusters,
            "n_undefined_resamples": self.n_undefined,
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [95% CI {self.lo:.4f}, {self.hi:.4f}]"


def cluster_bootstrap(
    units: Sequence[str],
    statistic: Callable[[Sequence[str]], float],
    *,
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
    drop_undefined: bool = False,
) -> Interval:
    """묶음 단위 재표집으로 통계량의 CI 를 낸다.

    Args:
        units: **묶음 ID 목록**. 이미지 ID 를 넘기면 CI 가 좁아진다 — 그것이 이 함수를
            따로 두는 이유다.
        statistic: 재표집된 묶음 ID 목록을 받아 통계량을 내는 함수. 같은 묶음이 여러 번
            뽑히면 그 묶음의 이미지가 그만큼 중복 계산돼야 한다.
        drop_undefined: 통계량이 `nan` 을 돌려준 재표집을 백분위 계산에서 뺀다.
            **두 부분모집단의 차** 같은 통계량은 한쪽이 통째로 안 뽑히면 정의되지 않는데,
            그때 0 으로 대치하면 CI 가 0 쪽으로 끌려가 판정이 뒤집힌다. 기본값은 False 라
            기존 호출부(P9)의 동작은 바뀌지 않는다.

    시드를 고정하므로 같은 입력에 같은 CI 가 나온다 — `check-scorer` 의 비트 단위 일치
    요구와 어긋나지 않는다.
    """
    arr = list(units)
    if not arr:
        return Interval(0.0, 0.0, 0.0, 0, 0)
    rng = np.random.default_rng(seed)
    point = statistic(arr)
    n = len(arr)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = statistic([arr[j] for j in idx])
    n_undefined = 0
    if drop_undefined:
        ok = np.isfinite(draws)
        n_undefined = int((~ok).sum())
        draws = draws[ok]
        if draws.size == 0:
            return Interval(float(point), float("nan"), float("nan"),
                            n_resamples, n, n_undefined)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(point), float(lo), float(hi), n_resamples, n, n_undefined)


@dataclass(frozen=True)
class TostResult:
    """TOST 등가성 검정 결과. §5-2 P9."""

    diff: float
    ci_lo: float
    ci_hi: float
    margin: float
    equivalent: bool
    verdict: str
    n_clusters_a: int
    n_clusters_b: int

    def as_dict(self) -> dict:
        return {
            "diff": self.diff, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi,
            "margin": self.margin, "equivalent": self.equivalent,
            "verdict": self.verdict,
            "n_clusters_a": self.n_clusters_a, "n_clusters_b": self.n_clusters_b,
        }


def tost_equivalence(
    diff_interval: Interval, margin: float, *, min_clusters: int = 20
) -> TostResult:
    """두 값의 차가 ±margin 안에 있는지 판정한다.

    **CI 겹침이 아니라 TOST 를 쓰는 이유**: 검정력이 낮을수록 CI 가 넓어져 "차이 없음"
    주장이 쉬워지는 역인센티브를 없애기 위해서다. TOST 는 CI 가 넓으면 등가성을 주장할 수
    없다.

    검정력이 부족하면 **"지름길 없음"이 아니라 "판정 불가"** 로 보고한다(§5-3).
    """
    n_a = diff_interval.n_clusters
    under_powered = n_a < min_clusters
    equivalent = (
        not under_powered
        and diff_interval.lo > -margin
        and diff_interval.hi < margin
    )
    if under_powered:
        verdict = (
            f"판정 불가 — 묶음 {n_a}개는 검정력 부족(최소 {min_clusters}). "
            "'지름길 없음'으로 쓰지 않는다"
        )
    elif equivalent:
        verdict = f"등가 — 차 CI [{diff_interval.lo:.4f}, {diff_interval.hi:.4f}] ⊂ ±{margin}"
    else:
        verdict = (
            f"등가 아님 — 차 CI [{diff_interval.lo:.4f}, {diff_interval.hi:.4f}] 가 "
            f"±{margin} 를 벗어난다"
        )
    return TostResult(
        diff=diff_interval.point, ci_lo=diff_interval.lo, ci_hi=diff_interval.hi,
        margin=margin, equivalent=equivalent, verdict=verdict,
        n_clusters_a=n_a, n_clusters_b=n_a,
    )


# --- 시드 분산 진단 ---------------------------------------------------------------

def _c4(n: int) -> float:
    """`E[s] = c4·σ`. n=3 에서 0.8862 — 평균적으로 σ 를 11.4% 과소추정한다."""
    return sqrt(2 / (n - 1)) * gamma(n / 2) / gamma((n - 1) / 2)


@dataclass(frozen=True)
class SeedSdDiagnostic:
    """시드 표준편차 추정의 불안정성. **3시드에서는 sd 자체가 큰 오차를 갖는다.**"""

    s: float
    n_seeds: int
    bias_corrected: float
    ci_lo: float
    ci_hi: float
    cv: float
    reliable: bool
    note: str

    def as_dict(self) -> dict:
        return {
            "seed_sd": self.s, "n_seeds": self.n_seeds,
            "seed_sd_unbiased": self.bias_corrected,
            "seed_sd_ci": [self.ci_lo, self.ci_hi],
            "seed_sd_cv": self.cv, "reliable": self.reliable, "note": self.note,
        }


def seed_sd_diagnostic(values: Sequence[float], *, alpha: float = 0.05) -> SeedSdDiagnostic:
    """시드별 지표에서 sd 를 내되 **그 sd 의 불확실성까지 함께 낸다.**

    카이제곱 분포로 σ 의 신뢰구간을 구한다. n=3 이면 구간이 매우 넓다 — 그 사실을
    숨기지 않는 것이 이 함수의 목적이다.
    """
    from scipy import stats

    n = len(values)
    if n < 2:
        return SeedSdDiagnostic(0.0, n, 0.0, 0.0, 0.0, float("inf"), False,
                                "시드 2개 미만이면 분산을 추정할 수 없다")
    s = float(np.std(values, ddof=1))
    df = n - 1
    c4 = _c4(n)
    lo = s * sqrt(df / stats.chi2.ppf(1 - alpha / 2, df))
    hi = s * sqrt(df / stats.chi2.ppf(alpha / 2, df))
    cv = sqrt(1 - c4**2) / c4
    reliable = n >= 10
    note = (
        f"시드 {n}개의 sd 는 변동계수 {cv:.2f} 로 불안정하다. "
        f"σ 의 95% CI [{lo:.4f}, {hi:.4f}] 를 함께 보고한다"
        if not reliable
        else f"시드 {n}개 — sd 추정이 비교적 안정적이다"
    )
    return SeedSdDiagnostic(s, n, s / c4, lo, hi, cv, reliable, note)


@dataclass(frozen=True)
class DenominatorVerdict:
    """회복률 분모 판정. 사전등록 3σ 규칙과 CI 기반 보조 판정을 **함께** 낸다."""

    d: float
    seed_sd: float
    tripwire_pass: bool
    ci_pass: bool | None
    reportable: bool
    detail: str
    diagnostic: SeedSdDiagnostic | None = None

    def as_dict(self) -> dict:
        out = {
            "denominator_d": self.d, "seed_sd": self.seed_sd,
            "tripwire_3sigma_pass": self.tripwire_pass,
            "ci_pass": self.ci_pass, "recovery_reportable": self.reportable,
            "detail": self.detail,
        }
        if self.diagnostic:
            out["seed_sd_diagnostic"] = self.diagnostic.as_dict()
        return out


MAX_RECOVERY_HALF_WIDTH = 0.25
"""회복률 95% CI 반폭 상한(비율 단위 = 25pp). 이보다 넓으면 헤드라인으로 싣지 않는다.

§5-4 가 델타법으로 든 예(D=0.08 → 반폭 31pp, D=0.05 → 50pp)가 곧 "못 쓰는 구간"이다.
"""


def recovery_denominator_verdict(
    central: float,
    local_values: Sequence[float],
    seed_values: Sequence[float] | None = None,
    recovery_ci: Interval | None = None,
) -> DenominatorVerdict:
    """분모 판정 — 사전등록 3σ 트립와이어 + CI 기반 보조.

    **3σ 규칙 단독으로는 검정력이 낮다.** 시드 3개의 sd 는 자유도 2 라 22%의 경우
    참 σ 의 절반 미만으로 나오고, 그때 통과선이 절반으로 내려가 막아야 할 분모가 통과한다
    (몬테카를로 20만 회: 참 D/σ=2 에서 통과율 36%, D/σ=5 에서 94%).

    그래서 판정을 **둘로 나눠 병기**한다.
      - 트립와이어: 사전등록된 `D ≥ 3·시드sd`. 값을 바꾸지 않는다.
      - 보조: 회복률 자체의 묶음 클러스터 부트스트랩 CI 반폭. 평가셋이 크므로 이쪽은
        잘 추정된다.

    둘 중 **하나라도 불합격이면 회복률을 헤드라인으로 싣지 않는다.**
    """
    local_mean = float(np.mean(local_values)) if len(local_values) else 0.0
    d = central - local_mean
    diag = seed_sd_diagnostic(seed_values) if seed_values and len(seed_values) >= 2 else None
    sd = diag.s if diag else 0.0
    tripwire = d > 0 and d >= 3 * sd

    ci_pass: bool | None = None
    if recovery_ci is not None:
        ci_pass = recovery_ci.half_width <= MAX_RECOVERY_HALF_WIDTH

    reportable = tripwire and (ci_pass is not False)
    parts = [f"분모 D={d:.4f}", f"시드sd={sd:.4f}", f"3·sd={3 * sd:.4f}"]
    parts.append("트립와이어 통과" if tripwire else "트립와이어 불합격")
    if ci_pass is not None:
        parts.append(
            f"회복률 CI 반폭 {recovery_ci.half_width:.3f} "
            f"({'≤' if ci_pass else '>'} {MAX_RECOVERY_HALF_WIDTH})"
        )
    if not reportable:
        parts.append("→ 회복률을 산출하지 않고 세 절대값만 CI 와 함께 보고한다")
    return DenominatorVerdict(
        d=d, seed_sd=sd, tripwire_pass=tripwire, ci_pass=ci_pass,
        reportable=reportable, detail=" · ".join(parts), diagnostic=diag,
    )


def rate_by_group(
    per_image: Mapping[str, float], image_to_group: Mapping[str, str]
) -> Callable[[Sequence[str]], float]:
    """묶음 목록 → 비율을 내는 통계량 함수를 만든다. `cluster_bootstrap` 에 넘긴다.

    같은 묶음이 재표집에서 두 번 뽑히면 그 묶음의 이미지가 두 번 세어진다 — 그것이
    클러스터 부트스트랩의 정의다.
    """
    members: dict[str, list[float]] = {}
    for img, v in per_image.items():
        g = image_to_group.get(img)
        if g is not None:
            members.setdefault(g, []).append(v)

    def statistic(groups: Sequence[str]) -> float:
        total = 0.0
        count = 0
        for g in groups:
            for v in members.get(g, ()):
                total += v
                count += 1
        return total / count if count else 0.0

    return statistic
