"""백분위 구간 규칙 — 2026-10-01 채택한 구간 규약의 계산 쪽. **합성 벡터만** 쓴다(결과 파일을 열지 않는다).

사례의 출처 둘.

1. 28번 측정의 대조표 **45칸**(`docs/dev_log/2026-09-24-후속정리/28_27번응답_벡터측정_D.md` 4-2) — 손 도출과 실행이
   45칸 같았던 표를 **이 모듈의 상태 어휘로** 다시 적었다. `NaN` 끝점은 `undefined` 다.
2. 27번 검토 4-2 의 **순위 이동 식** — 버림(유한만)에서 넣음(±무한을 끝값으로)으로 갈 때 끝점이 옮겨 가는 순위.

유한값은 **자기 자리 번호**다 — 끝점으로 나온 유한값이 곧 집힌 순위다(28번 4-1 과 같은 꼴).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from evaluation import percentile_rule as PR

INF = math.inf
RNG = np.random.default_rng(20261001)


def ranked(n: int, *, pos: int = 0, neg: int = 0) -> np.ndarray:
    """길이 `n`. 아래 `neg` 자리는 −∞, 위 `pos` 자리는 +∞, 나머지는 자기 자리 번호. 섞어서 준다."""
    x = np.arange(n, dtype=float)
    x[:neg] = -INF
    if pos:
        x[n - pos:] = INF
    return RNG.permutation(x)


def st(b: PR.Bound):
    return b.value if b.state == "finite" else b.state


# ================================================================ 1. 45칸 — 28번 4-2

# 1~15: 두 원소 [a, b] 에 q = 100·t. 아래 이웃 a, 위 이웃 b, 가중 t. 유한은 0 · 1.
PAIR = {"finite·finite": (0.0, 1.0), "finite·pos_inf": (0.0, INF), "neg_inf·finite": (-INF, 1.0),
        "pos_inf·pos_inf": (INF, INF), "neg_inf·neg_inf": (-INF, -INF)}
T_GROUPS = {"t=0": (0.0,), "0<t<0.5": (0.025, 0.25), "t>=0.5": (0.5, 0.75, 0.975)}
PAIR_CELLS = [
    (1, "finite·finite", "t=0", "finite"), (2, "finite·finite", "0<t<0.5", "finite"),
    (3, "finite·finite", "t>=0.5", "finite"),
    (4, "finite·pos_inf", "t=0", "undefined"), (5, "finite·pos_inf", "0<t<0.5", "pos_inf"),
    (6, "finite·pos_inf", "t>=0.5", "undefined"),
    (7, "neg_inf·finite", "t=0", "undefined"), (8, "neg_inf·finite", "0<t<0.5", "undefined"),
    (9, "neg_inf·finite", "t>=0.5", "neg_inf"),
    (10, "pos_inf·pos_inf", "t=0", "undefined"), (11, "pos_inf·pos_inf", "0<t<0.5", "undefined"),
    (12, "pos_inf·pos_inf", "t>=0.5", "undefined"),
    (13, "neg_inf·neg_inf", "t=0", "undefined"), (14, "neg_inf·neg_inf", "0<t<0.5", "undefined"),
    (15, "neg_inf·neg_inf", "t>=0.5", "undefined"),
]


@pytest.mark.parametrize(("cell", "pair", "group", "want"), PAIR_CELLS)
def test_45칸_1_15_보간_식의_칸(cell: int, pair: str, group: str, want: str) -> None:
    for t in T_GROUPS[group]:
        b = PR.quantile_bound(list(PAIR[pair]), 100 * t, "linear")
        assert b.state == want, (cell, t)
        if want == "finite":
            assert b.value == pytest.approx(t)


# 16~21: 선형 · 길이 2,000 · 한쪽 꼬리의 건수
LINEAR_TAIL = [(16, "pos", (0, 48, 49), "finite"), (17, "pos", (50,), "pos_inf"), (18, "pos", (51, 200), "undefined"),
               (19, "neg", (0, 48, 49), "finite"), (20, "neg", (50,), "neg_inf"), (21, "neg", (51, 200), "undefined")]


@pytest.mark.parametrize(("cell", "side", "counts", "want"), LINEAR_TAIL)
def test_45칸_16_21_선형은_무한_꼬리_안쪽에서_끝점이_정해지지_않는다(cell, side, counts, want) -> None:
    for c in counts:
        iv = PR.interval(ranked(2000, **{side: c}), 0.05, method="linear")
        end = iv.hi if side == "pos" else iv.lo
        assert end.state == want, (cell, c)


# 22~45: 방식마다 길이 2,000 의 두 자리와, 끝점이 무한이 되기 시작하는 꼬리 건수
POSITIONS = {  # 방식: (하한 자리, 상한 자리, −∞ 하한이 되는 최소 건수, +∞ 상한이 되는 최소 건수)
    "lower": (49, 1949, 50, 51),
    "higher": (50, 1950, 51, 50),
    "nearest": (50, 1949, 51, 51),
    "inverted_cdf": (49, 1949, 50, 51),
    PR.PERCENTILE_RULE: (49, 1950, 50, 50),
}
COUNTS = (0, 48, 49, 50, 51, 200)


@pytest.mark.parametrize("method", list(POSITIONS))
def test_45칸_22_37_42_45_자리와_무한이_되는_건수(method: str) -> None:
    lo_at, hi_at, neg_from, pos_from = POSITIONS[method]
    iv = PR.interval(ranked(2000), 0.05, method=method)
    assert (iv.lo.value, iv.hi.value) == (lo_at, hi_at)
    for c in COUNTS:
        lo = PR.interval(ranked(2000, neg=c), 0.05, method=method).lo
        hi = PR.interval(ranked(2000, pos=c), 0.05, method=method).hi
        assert lo.state == ("neg_inf" if c >= neg_from else "finite"), ("하한", c)
        assert hi.state == ("pos_inf" if c >= pos_from else "finite"), ("상한", c)
        if lo.state == "finite":
            assert lo.value == lo_at, ("하한 자리", c)
        if hi.state == "finite":
            assert hi.value == hi_at, ("상한 자리", c)


def test_45칸_38_41_선형의_자리는_두_이웃_사이다() -> None:
    iv = PR.interval(ranked(2000), 0.05, method="linear")
    assert iv.lo.value == pytest.approx(49.975) and iv.hi.value == pytest.approx(1949.025)
    for c, want in ((49, "finite"), (50, "neg_inf"), (51, "undefined"), (200, "undefined")):
        assert PR.interval(ranked(2000, neg=c), 0.05, method="linear").lo.state == want
    for c, want in ((49, "finite"), (50, "pos_inf"), (51, "undefined"), (200, "undefined")):
        assert PR.interval(ranked(2000, pos=c), 0.05, method="linear").hi.state == want


def test_45칸_밖_길이_1937_의_등록_규칙() -> None:
    """28번 4-4 다 의 마지막 열 — 하한 48 · 상한 1888, 꼬리 49건부터 무한."""
    assert st(PR.interval(ranked(1937), 0.05).lo) == 48 and st(PR.interval(ranked(1937), 0.05).hi) == 1888
    assert st(PR.interval(ranked(1937, neg=48), 0.05).lo) == 48
    assert st(PR.interval(ranked(1937, neg=49), 0.05).lo) == "neg_inf"
    assert st(PR.interval(ranked(1937, pos=48), 0.05).hi) == 1888
    assert st(PR.interval(ranked(1937, pos=49), 0.05).hi) == "pos_inf"


# ================================================================ 2. 순위 식 — 모든 길이

def test_순위_식은_등록한_글자_그대로다() -> None:
    for n in range(1, 5001):
        k, top = PR.rank_positions(n, 0.05)
        assert k == ((n - 1) * 25) // 1000 and top == (n - 1) - k


def test_유한만일_때_라이브러리의_하한_내림_상한_올림과_같다() -> None:
    """이름으로 등록하지 않는다 — 식과 라이브러리가 같은 값을 내는지를 시험이 맞댄다(28번: 길이 2~2,500 에서 0개)."""
    for n in range(1, 2501):
        x = RNG.permutation(np.arange(n, dtype=float))
        iv = PR.interval(x, 0.05)
        assert iv.lo.value == np.percentile(x, 2.5, method="lower")
        assert iv.hi.value == np.percentile(x, 97.5, method="higher")


def test_꼬리가_n_α_2_의_올림_이상이면_그쪽_끝점이_무한이다_양쪽_모든_길이() -> None:
    for n in range(1, 2501):
        c = math.ceil(n * 0.025)
        assert PR.interval(ranked(n, pos=c), 0.05).hi.state == "pos_inf", n
        assert PR.interval(ranked(n, neg=c), 0.05).lo.state == "neg_inf", n
        if c > 1:
            assert PR.interval(ranked(n, pos=c - 1), 0.05).hi.state == "finite", n
            assert PR.interval(ranked(n, neg=c - 1), 0.05).lo.state == "finite", n


def test_구간은_집계_추첨의_1_α_이상을_언제나_품는다() -> None:
    for n in range(1, 5001):
        k, top = PR.rank_positions(n, 0.05)
        assert top - k + 1 >= n * 0.95, n


# ================================================================ 3. 27번 4-2 — 버림에서 넣음으로

A = 0.025
GRID = [(nf, p, m) for nf in (1000, 2000) for p in (0, 1, 3, 7, 15) for m in (0, 1, 3, 7, 15)]


@pytest.mark.parametrize(("nf", "npos", "nneg"), GRID)
def test_순위_이동_식_가상_순위(nf: int, npos: int, nneg: int) -> None:
    """선형(가상 순위 `(n−1)·q`)으로 잰다 — 두 이웃이 유한한 격자다. 넣음 − 버림 == 27번 4-2 의 식."""
    drop = PR.interval(np.arange(nf, dtype=float), 0.05, method="linear")
    inc = PR.interval(ranked(nf + npos + nneg, pos=npos, neg=nneg), 0.05, method="linear")
    assert inc.lo.state == inc.hi.state == "finite"
    # 넣은 벡터의 유한값은 전체 자리 번호다 — 유한 추첨의 순위로 바꾸려면 아래 −∞ 수를 뺀다
    assert (inc.lo.value - nneg) - drop.lo.value == pytest.approx(A * npos - (1 - A) * nneg, abs=1e-9)
    assert (inc.hi.value - nneg) - drop.hi.value == pytest.approx((1 - A) * npos - A * nneg, abs=1e-9)


@pytest.mark.parametrize(("nf", "npos", "nneg"), GRID)
def test_순위_이동_식_등록_규칙(nf: int, npos: int, nneg: int) -> None:
    """등록 규칙의 자리는 가상 순위의 내림(하한) · 올림(상한)이다 — 이동은 식에서 한 칸 안쪽이고, 포함 영역이면 넣은 구간이 버린 구간을 품는다."""
    drop = PR.interval(np.arange(nf, dtype=float), 0.05)
    inc = PR.interval(ranked(nf + npos + nneg, pos=npos, neg=nneg), 0.05)
    n = nf + npos + nneg
    assert inc.lo.value - nneg == math.floor((n - 1) * A) - nneg == math.floor((n - 1) * A - nneg)
    assert inc.hi.value - nneg == math.ceil((n - 1) * (1 - A) - nneg)
    d_lo = (inc.lo.value - nneg) - drop.lo.value
    d_hi = (inc.hi.value - nneg) - drop.hi.value
    assert abs(d_lo - (A * npos - (1 - A) * nneg)) < 1 and abs(d_hi - ((1 - A) * npos - A * nneg)) < 1
    if npos and nneg and npos / 39 <= nneg <= 39 * npos:
        assert inc.lo.value - nneg <= drop.lo.value and inc.hi.value - nneg >= drop.hi.value, "넣은 구간이 품는다"


# ================================================================ 4. 상태 — §25-2 의 표 가

def test_추첨이_0건이면_세_칸이_없음이다() -> None:
    iv = PR.interval([], 0.05)
    assert (iv.lo.state, iv.hi.state, iv.half_width.state, iv.n) == ("absent", "absent", "absent", 0)


@pytest.mark.parametrize(("values", "lo", "hi", "hw"), [
    ([1.0, 2.0, 3.0], "finite", "finite", "finite"),
    ([INF, INF, INF], "pos_inf", "pos_inf", "undefined"),
    ([-INF, -INF], "neg_inf", "neg_inf", "undefined"),
    ([-INF, INF], "neg_inf", "pos_inf", "infinite"),
    ([1.0, INF], "finite", "pos_inf", "infinite"),
    ([-INF, 1.0], "neg_inf", "finite", "infinite"),
])
def test_끝점과_반폭의_상태(values, lo, hi, hw) -> None:
    iv = PR.interval(values, 0.05)
    assert (iv.lo.state, iv.hi.state, iv.half_width.state) == (lo, hi, hw)


def test_수_칸은_유한일_때만_수이고_표준_JSON_으로_쓴다() -> None:
    for values in ([1.0, 5.0], [INF], [-INF, INF], []):
        d = PR.interval(values, 0.05).as_dict()
        json.dumps(d, allow_nan=False)
        for key in ("ci_lo", "ci_hi", "half_width"):
            assert (d[key] is None) == (d[f"{key}_state"] != "finite")
        assert d["convention_id"] == PR.CONVENTION_ID


def test_등록_규칙은_끝점_상태_undefined_를_내지_않는다() -> None:
    """보간하지 않으므로 끝점은 늘 추첨값 하나다 — 넷째 상태는 라이브러리 보간 방식에서만."""
    for n in (1, 2, 40, 41, 2000):
        for p in range(0, n + 1, max(1, n // 7)):
            for m in range(0, n - p + 1, max(1, n // 5)):
                iv = PR.interval(ranked(n, pos=p, neg=m), 0.05)
                assert {iv.lo.state, iv.hi.state} <= set(PR.ENDPOINT_STATES)


def test_NaN_은_추첨값이_아니다() -> None:
    with pytest.raises(ValueError, match="NaN"):
        PR.interval([1.0, float("nan")], 0.05)


def test_상태와_수가_어긋난_끝점은_만들지_못한다() -> None:
    with pytest.raises(ValueError):
        PR.Bound(1.0, "absent")
    with pytest.raises(ValueError):
        PR.Bound(None, "finite")
    with pytest.raises(ValueError):
        PR.Bound(INF, "finite")


def test_규약_id_는_방식을_따른다() -> None:
    assert PR.interval([1.0], 0.05).convention_id == PR.CONVENTION_ID
    other = PR.interval([1.0], 0.05, method="linear").convention_id
    assert other != PR.CONVENTION_ID and "linear" in other and np.__version__ in other
    with pytest.raises(ValueError, match="방식"):
        PR.interval([1.0], 0.05, method="midpoint_of_something")


def test_α_는_십진_표기의_유리수로_읽는다() -> None:
    assert PR.rank_positions(2001, 0.05) == (50, 1950)
    assert PR.rank_positions(11, 0.1) == (0, 10)
    assert PR.rank_positions(21, 0.1) == (1, 19)
    for bad in (0, 1, -0.1, True, "0.05"):
        with pytest.raises(ValueError):
            PR.rank_positions(10, bad)


# ================================================================ 5. 미정의를 빼는 통계량

def test_빼는_통계량은_NaN_을_세고_뺀다() -> None:
    iv, n_def, n_undef = PR.defined_interval([1.0, float("nan"), 2.0, float("nan"), 3.0], 0.05)
    assert (n_def, n_undef, iv.n) == (3, 2, 3)
    assert iv.convention_id == PR.CONVENTION_ID


def test_빼는_통계량에_무한이_오면_거부한다() -> None:
    with pytest.raises(ValueError, match="무한"):
        PR.defined_interval([1.0, INF], 0.05)


# ---------------------------------------------------------------- α 가 다르면 규약 id 가 다르다 (07번 §33-4)

@pytest.mark.parametrize("alpha", [0.10, 0.01, 0.2])
def test_α_가_0_05_가_아니면_채택_규약_id_를_달지_않는다(alpha: float) -> None:
    """순위 식의 25/1000 은 α = 0.05 의 α/2 다. 같은 식을 다른 α 로 계산한 구간이 채택 id 를 달면 등록이 무효인 구간이 채택처럼 읽힌다."""
    iv = PR.interval(list(range(100)), alpha)
    assert iv.convention_id != PR.CONVENTION_ID and iv.convention_id.startswith(PR.CONVENTION_ID)
    assert f"alpha={alpha!r}" in iv.convention_id
    assert PR.interval([], alpha).convention_id == iv.convention_id, "구간이 없어도 같은 id"
    assert PR.convention_id(PR.PERCENTILE_RULE, alpha) == iv.convention_id


def test_α_가_0_05_면_채택_id_이고_방식과_α_는_둘_다_붙는다() -> None:
    assert PR.interval(list(range(100)), 0.05).convention_id == PR.CONVENTION_ID
    both = PR.convention_id("linear", 0.10)
    assert "method=linear" in both and "alpha=0.1" in both
    with pytest.raises(ValueError, match="α"):
        PR.convention_id(PR.PERCENTILE_RULE, 1.5)
