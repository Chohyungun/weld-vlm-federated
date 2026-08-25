"""3시드에서 D >= 3*sd 규칙이 검정력을 갖는가 — 몬테카를로.

문제: n=3 에서 표본 표준편차 s 는 자유도 2 라 매우 불안정하다.
s 가 작게 나오면 통과하면 안 될 때 통과하고, 크게 나오면 통과해야 할 때 막힌다.
"""
import numpy as np

rng = np.random.default_rng(20260825)
TRIALS = 200_000
N_SEEDS = 3

# c4 보정계수: E[s] = c4 * sigma
from math import gamma, sqrt

c4 = sqrt(2/(N_SEEDS-1)) * gamma(N_SEEDS/2) / gamma((N_SEEDS-1)/2)
s = rng.standard_normal((TRIALS, N_SEEDS)).std(axis=1, ddof=1)
print(f"n={N_SEEDS} 표본 sd 의 성질 (sigma=1 기준)")
print(f"  이론 c4 = {c4:.4f}  (평균적으로 sigma 를 {100*(1-c4):.1f}% 과소추정)")
print(f"  실측 평균 s = {s.mean():.4f}, 변동계수 CV = {s.std()/s.mean():.3f}")
print(f"  s 백분위: p05={np.percentile(s,5):.3f}  p50={np.percentile(s,50):.3f}  p95={np.percentile(s,95):.3f}")
print(f"  P(s < 0.5*sigma) = {(s<0.5).mean():.3f}   <- 이만큼은 통과선이 절반 이하로 내려간다")
print()

print("규칙 'D >= 3*s' 의 통과확률 (D 는 참값으로 고정, sigma=1)")
print(f"{'참 D/sigma':>10} {'P(통과)':>9}  해석")
for ratio in [1, 2, 3, 4, 5, 6, 8, 10]:
    p = (ratio >= 3*s).mean()
    if ratio < 3:
        note = "막아야 정상 — 통과하면 오통과"
    elif ratio < 5:
        note = "경계"
    else:
        note = "통과해야 정상 — 막히면 오차단"
    print(f"{ratio:>10} {p:>9.3f}  {note}")
print()

# D 도 추정량인 경우 (중앙·로컬 각각 3시드)
print("D 도 3시드 추정량일 때 (중앙 3 + 로컬 3, 동일 sigma)")
print(f"{'참 D/sigma':>10} {'P(통과)':>9}")
for ratio in [1, 2, 3, 4, 5, 6, 8, 10]:
    c = rng.standard_normal((TRIALS, N_SEEDS)) + ratio
    l = rng.standard_normal((TRIALS, N_SEEDS))
    d_hat = c.mean(axis=1) - l.mean(axis=1)
    # 규칙이 쓰는 sd 를 '시드 sd'로 해석 — 두 칸 pooled
    s_hat = np.sqrt((c.var(axis=1, ddof=1) + l.var(axis=1, ddof=1)) / 2)
    print(f"{ratio:>10} {(d_hat >= 3*s_hat).mean():>9.3f}")
print()

print("대안: D 의 95% 신뢰구간 하한 > 0 (t 기반, 이표본)")
from scipy import stats

print(f"{'참 D/sigma':>10} {'P(통과)':>9}")
for ratio in [1, 2, 3, 4, 5, 6, 8, 10]:
    c = rng.standard_normal((TRIALS, N_SEEDS)) + ratio
    l = rng.standard_normal((TRIALS, N_SEEDS))
    d_hat = c.mean(axis=1) - l.mean(axis=1)
    sp2 = (c.var(axis=1, ddof=1) + l.var(axis=1, ddof=1)) / 2
    se = np.sqrt(sp2 * (2/N_SEEDS))
    tcrit = stats.t.ppf(0.975, df=2*(N_SEEDS-1))
    print(f"{ratio:>10} {((d_hat - tcrit*se) > 0).mean():>9.3f}")
