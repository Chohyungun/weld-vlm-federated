"""통합형 회복률의 추첨 분류와 구간 — 2026-10-01 채택한 구간 규약. **합성 벡터만** 쓴다.

지키는 것 — 분모 실패는 분자 부호의 ±무한(버리지 않는다) · 동률(분자 0)은 판정 불가 · 부호 혼재는 판정 불가 ·
지지량 0 은 제외 · 회계 · 백분위의 `n` · 규칙 ④ 의 분자 · 점추정의 분모 실패는 `absent` · 등록과의 결속 ·
그리고 **구간 규약의 적용 범위**(기존 `cluster_bootstrap` 의 기본값은 그대로, 방식은 인자로).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from evaluation import percentile_rule as PR
from evaluation import recovery_interval as RI
from evaluation.stats import cluster_bootstrap
from tests.test_prereg_unified import filled

INF = math.inf


def one_draw(seeds: list[tuple[float, float, float]]):
    """추첨 하나 — 시드마다 `(연합, 중앙, 로컬 평균)`. 로컬 셋을 같은 값으로 둔다.

    값은 **이진 유한소수**(0.125 의 배수)로 쓴다 — 셋의 합을 3 으로 나눈 값이 그 값과 정확히 같아야 "분자 0" 사례가 선다.
    0.4 셋의 평균은 부동소수로 0.4 가 아니다(아래 시험).
    """
    f = np.array([[s[0] for s in seeds]])
    c = np.array([[s[1] for s in seeds]])
    l = np.array([[[s[2]] * 3 for s in seeds]])
    return f, c, l


def classify(seeds, support=True):
    f, c, l = one_draw(seeds)
    v, code = RI.classify_draws(f, c, l, np.array([support]))
    return v[0], int(code[0])


# ================================================================ 1. 시드 하나의 R_s — §15-6 과 §33

@pytest.mark.parametrize(("seed", "want"), [
    ((0.375, 0.5, 0.25), 0.5),               # D > 0 — N/D = 0.125/0.25
    ((0.25, 0.5, 0.25), 0.0),                # D > 0 · N = 0 — 회복률 0, 유한
    ((0.5, 0.25, 0.25), INF),                # D = 0 · N > 0 — +∞
    ((0.5, 0.125, 0.25), INF),               # D < 0 · N > 0 — +∞ (약속이다 — 대수적 비율은 음수)
    ((0.125, 0.25, 0.25), -INF),             # D = 0 · N < 0 — −∞
    ((0.125, 0.125, 0.25), -INF),            # D < 0 · N < 0 — −∞ (대수적 비율은 양수)
])
def test_시드_하나의_회복률(seed, want) -> None:
    v, _ = classify([seed])
    assert v == pytest.approx(want) if math.isfinite(want) else v == want


def test_분모_실패에서_분자가_0_이면_판정_불가다_끝없이_좋은_회복이_아니다() -> None:
    """§15-6 의 글자(`F − L̄ ≥ 0` 이면 +∞)를 §33 이 대신한다 — 세 모델이 같은 값인 추첨은 +∞ 가 아니다."""
    v, code = classify([(0.0, 0.0, 0.0)])
    assert math.isnan(v) and code == RI._CODE_ZERO


def test_같아야_할_두_수의_차가_부동소수로_0_이_아니면_무한으로_간다() -> None:
    """등록한 판정은 정확 비교다. 0.4 셋의 평균은 0.4000000000000001 이라 분자·분모가 0 이 아니다 —
    이 추첨은 판정 불가가 아니라 −∞ 다. 2판 4절이 적은 대로 오차가 판정 불가와 무한을 가르는 자리이고, 등록이 그것을 받았다."""
    assert (0.4 + 0.4 + 0.4) / 3 != 0.4
    v, code = classify([(0.4, 0.4, 0.4)])
    assert v == -INF and code == RI._CODE_NEG


def test_분자_0_의_판정은_정확_비교다() -> None:
    """허용 오차가 없다 — 아주 작은 양수는 0 이 아니라 +∞ 로 간다. −0.0 은 0 이다."""
    tiny = 5e-324
    assert classify([(tiny, 0.0, 0.0)])[0] == INF
    assert classify([(-tiny, 0.0, 0.0)])[0] == -INF
    f, c, l = one_draw([(0.0, 0.0, 0.0)])
    f[0, 0] = -0.0
    assert int(RI.classify_draws(f, c, l, np.array([True]))[1][0]) == RI._CODE_ZERO


def test_로컬_평균은_합을_로컬_수로_나눈다_점추정과_추첨이_같은_식() -> None:
    f = np.array([[0.7]])
    c = np.array([[0.9]])
    l = np.array([[[0.1, 0.2, 0.4]]])
    v, _ = RI.classify_draws(f, c, l, np.array([True]))
    p = RI.point_recovery(f[0], c[0], l[0])
    lbar = (0.1 + 0.2 + 0.4) / 3
    assert v[0] == (0.7 - lbar) / (0.9 - lbar) == p["seeds"][0]["R"] == p["R_bar"]


# ================================================================ 2. 추첨 하나의 범주 — 순서

@pytest.mark.parametrize(("seeds", "support", "code", "value"), [
    ([(0.375, 0.5, 0.25), (0.5, 0.5, 0.25)], True, RI._CODE_FINITE, 0.75),
    ([(0.375, 0.5, 0.25), (0.5, 0.125, 0.25)], True, RI._CODE_POS, INF),       # 한 시드만 +∞ 여도 R̄ = +∞
    ([(0.375, 0.5, 0.25), (0.125, 0.125, 0.25)], True, RI._CODE_NEG, -INF),
    ([(0.5, 0.25, 0.25), (0.5, 0.125, 0.25)], True, RI._CODE_POS, INF),        # 같은 부호끼리
    ([(0.5, 0.25, 0.25), (0.125, 0.25, 0.25)], True, RI._CODE_CONFLICT, None),  # 부호 혼재
    ([(0.25, 0.25, 0.25), (0.5, 0.25, 0.25), (0.125, 0.25, 0.25)], True, RI._CODE_ZERO, None),  # 동률이 혼재보다 앞
    ([(0.25, 0.25, 0.25), (0.125, 0.25, 0.25)], False, RI._CODE_SUPPORT, None),  # 지지량이 가장 앞
    ([(0.375, 0.5, 0.25)], False, RI._CODE_SUPPORT, None),
])
def test_추첨의_범주는_정확히_하나이고_순서대로_본다(seeds, support, code, value) -> None:
    v, got = classify(seeds, support)
    assert got == code
    if value is None:
        assert math.isnan(v), "집계에 들어가지 않는 추첨은 값이 없다"
    elif math.isfinite(value):
        assert v == pytest.approx(value)
    else:
        assert v == value


# ================================================================ 3. 회계 · n · 규칙 ④

def draws_with(n_fin=0, n_pos=0, n_neg=0, n_sup=0, n_zero=0, n_conf=0, *, seeds=3):
    """범주별 건수로 추첨을 짓는다. 유한 추첨의 R 은 자기 자리 번호에 비례한다."""
    rows_f, rows_c, rows_l, sup = [], [], [], []

    def add(per_seed, support=True):
        rows_f.append([s[0] for s in per_seed])
        rows_c.append([s[1] for s in per_seed])
        rows_l.append([[s[2]] * 3 for s in per_seed])
        sup.append(support)

    ok = (0.375, 0.5, 0.25)
    for i in range(n_fin):
        add([(0.25 + 0.25 * (i + 1) / (n_fin + 1), 0.5, 0.25)] * seeds)
    for _ in range(n_pos):
        add([(0.5, 0.125, 0.25)] + [ok] * (seeds - 1))
    for _ in range(n_neg):
        add([(0.125, 0.125, 0.25)] + [ok] * (seeds - 1))
    for _ in range(n_sup):
        add([ok] * seeds, support=False)
    for _ in range(n_zero):
        add([(0.25, 0.25, 0.25)] + [ok] * (seeds - 1))
    for _ in range(n_conf):
        add([(0.5, 0.125, 0.25), (0.125, 0.125, 0.25)] + [ok] * (seeds - 2))
    order = np.random.default_rng(7).permutation(len(sup))
    return (np.array(rows_f)[order], np.array(rows_c)[order], np.array(rows_l)[order], np.array(sup)[order])


def test_회계와_n_과_규칙_4_의_분자() -> None:
    f, c, l, s = draws_with(n_fin=1900, n_pos=30, n_neg=20, n_sup=25, n_zero=15, n_conf=10)
    ri = RI.recovery_interval(f, c, l, s, alpha=0.05, n_resamples=2000)
    assert ri.counters == {"n_finite": 1900, "n_infinite_pos": 30, "n_infinite_neg": 20,
                           "n_excluded_support": 25, "n_indeterminate": 25}
    assert ri.indeterminate_reasons == {"numerator_zero": 15, "sign_conflict": 10}
    assert ri.interval.n == 1950, "백분위의 n 은 유한과 ±무한"
    assert ri.rule4_fraction == (25 + 25) / 2000, "분모 실패는 규칙 ④ 의 분자에 들지 않는다"
    d = ri.as_dict()
    json.dumps(d, allow_nan=False)
    assert d["convention_id"] == PR.CONVENTION_ID and d["rule4_counts"] == ["support", "indeterminate"]


def test_꼬리가_임계에_닿으면_그쪽_끝점이_무한이다_끝값으로_포함() -> None:
    """n = 2,000 이면 ⌈n·α/2⌉ = 50. 분모 실패 +∞ 가 49건이면 상한이 유한하고 50건이면 +∞ 다."""
    for n_pos, want in ((49, "finite"), (50, "pos_inf")):
        f, c, l, s = draws_with(n_fin=2000 - n_pos, n_pos=n_pos)
        ri = RI.recovery_interval(f, c, l, s, alpha=0.05, n_resamples=2000)
        assert ri.interval.hi.state == want and ri.interval.lo.state == "finite"
    f, c, l, s = draws_with(n_fin=1950, n_neg=50)
    assert RI.recovery_interval(f, c, l, s, alpha=0.05, n_resamples=2000).interval.lo.state == "neg_inf"


def test_무한이_꼬리에_닿지_않아도_유한_추첨을_밀어낸다() -> None:
    """§25-2 의 09-25 정정 — 끝점이 유한한 채 옮겨 간다. 버렸다면 그 자리가 아니다."""
    f, c, l, s = draws_with(n_fin=1960, n_pos=40)
    ri = RI.recovery_interval(f, c, l, s, alpha=0.05, n_resamples=2000)
    kept = np.sort(RI.classify_draws(f, c, l, s)[0])
    fin = kept[np.isfinite(kept)]
    assert ri.interval.hi.state == "finite"
    assert ri.interval.hi.value == fin[1950], "넣으면 상한은 1,950 번째 자리의 추첨"
    assert ri.interval.hi.value != PR.interval(fin, 0.05).hi.value, "버렸다면 다른 자리다"


def test_같은_부호의_무한만_남으면_반폭이_정해지지_않는다() -> None:
    f, c, l, s = draws_with(n_pos=10, n_sup=5)
    ri = RI.recovery_interval(f, c, l, s, alpha=0.05, n_resamples=15)
    assert (ri.interval.lo.state, ri.interval.hi.state, ri.interval.half_width.state) == ("pos_inf", "pos_inf", "undefined")


def test_집계_추첨이_0건이면_구간이_없다() -> None:
    f, c, l, s = draws_with(n_sup=4, n_conf=3)
    ri = RI.recovery_interval(f, c, l, s, alpha=0.05, n_resamples=7)
    assert ri.interval.n == 0 and ri.interval.lo.state == ri.interval.half_width.state == "absent"


def test_추첨_수가_재표집_수와_다르면_거부한다() -> None:
    f, c, l, s = draws_with(n_fin=10)
    with pytest.raises(ValueError, match="재표집 수"):
        RI.recovery_interval(f, c, l, s, alpha=0.05, n_resamples=11)


@pytest.mark.parametrize("where", ["fed", "central", "locals"])
def test_입력에_유한하지_않은_값이_있으면_거부한다(where: str) -> None:
    f, c, l, s = draws_with(n_fin=5)
    {"fed": f, "central": c, "locals": l}[where].flat[0] = float("nan")
    with pytest.raises(ValueError, match="유한"):
        RI.classify_draws(f, c, l, s)


def test_모양이_어긋나면_거부한다() -> None:
    f, c, l, s = draws_with(n_fin=5)
    with pytest.raises(ValueError, match="모양"):
        RI.classify_draws(f, c[:, :2], l, s)
    with pytest.raises(ValueError, match="지지량"):
        RI.classify_draws(f, c, l, s.astype(int))


# ================================================================ 4. 점추정 — P-가

def test_점추정은_분모_실패_시드가_있으면_내지_않는다_무한을_적지_않는다() -> None:
    f = np.array([0.5, 0.5, 0.5])
    c = np.array([0.6, 0.3, 0.6])
    l = np.array([[0.4] * 3] * 3)
    p = RI.point_recovery(f, c, l)
    assert [s["state"] for s in p["seeds"]] == ["finite", "absent", "finite"]
    assert p["seeds"][1] == {"R": None, "state": "absent", "reason": "denominator", "D": pytest.approx(-0.1)}
    assert p["R_bar"] is None and p["state"] == "absent" and p["reason"] == "denominator"
    json.dumps(p, allow_nan=False)


def test_점추정은_모두_양수_분모면_시드_평균이다() -> None:
    p = RI.point_recovery(np.array([0.5, 0.6]), np.array([0.6, 0.6]), np.array([[0.4] * 3, [0.4] * 3]))
    assert p["state"] == "finite" and p["R_bar"] == pytest.approx((0.5 + 1.0) / 2)


# ================================================================ 5. 등록과의 결속

def test_채택한_등록이면_어긋남이_없다() -> None:
    assert RI.registered_mismatches(filled().ci) == []


@pytest.mark.parametrize(("key", "value"), [
    ("percentile_method", "linear"), ("infinity_handling", "drop_nonfinite"),
    ("undefined_denominator_policy", "drop_undefined"), ("numerator_zero_rule", "pos_inf"),
    ("convention_id", None), ("rule4_counts", ("support", "indeterminate", "denominator")), ("alpha", 0.1),
])
def test_등록이_다른_규약이면_이_구현으로_채점하지_않는다(key, value) -> None:
    ci = filled().ci
    setattr(ci, key, value)
    f, c, l, s = draws_with(n_fin=ci.n_resamples)
    with pytest.raises(ValueError, match=f"ci.{key}"):
        RI.recovery_interval_registered(ci, f, c, l, s)


def test_등록의_α_와_재표집_수를_읽는다() -> None:
    ci = filled().ci
    f, c, l, s = draws_with(n_fin=ci.n_resamples - 3, n_pos=3)
    ri = RI.recovery_interval_registered(ci, f, c, l, s)
    assert ri.n_resamples == ci.n_resamples and ri.interval.convention_id == PR.CONVENTION_ID


# ================================================================ 6. 적용 범위 — 기존 함수의 기본값은 그대로

def _old_cluster_bootstrap(units, statistic, *, n_resamples, seed, alpha, drop_undefined):
    """바꾸기 전 `cluster_bootstrap` 의 백분위 부분 그대로 — 기본값이 말없이 달라지지 않았는지 맞댄다."""
    arr = list(units)
    rng = np.random.default_rng(seed)
    point = statistic(arr)
    n = len(arr)
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[i] = statistic([arr[j] for j in idx])
    if drop_undefined:
        draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


UNITS = [f"g{i:03d}" for i in range(137)]
VAL = {u: (i * 37 % 101) / 101 for i, u in enumerate(UNITS)}


def mean_stat(gs):
    return float(np.mean([VAL[g] for g in gs]))


def sometimes_nan(gs):
    xs = [VAL[g] for g in gs if VAL[g] > 0.2]
    return float("nan") if len(xs) < 110 else float(np.mean(xs))


@pytest.mark.parametrize(("stat", "drop"), [(mean_stat, False), (sometimes_nan, True)])
@pytest.mark.parametrize("seed", [20260825, 7])
def test_기본값의_구간은_바꾸기_전과_같다(stat, drop, seed) -> None:
    got = cluster_bootstrap(UNITS, stat, n_resamples=400, seed=seed, drop_undefined=drop)
    point, lo, hi = _old_cluster_bootstrap(UNITS, stat, n_resamples=400, seed=seed, alpha=0.05, drop_undefined=drop)
    np.testing.assert_array_equal([got.point, got.lo, got.hi], [point, lo, hi])  # NaN 끼리 같다고 본다
    assert got.convention_id is None
    assert set(got.as_dict()) == {"point", "ci_lo", "ci_hi", "half_width", "n_resamples", "n_clusters",
                                  "n_undefined_resamples"}, "기본 호출의 산출 꼴에 키가 늘지 않는다"


def test_방식을_주면_등록_규칙의_자리이고_규약_id_를_싣는다() -> None:
    got = cluster_bootstrap(UNITS, mean_stat, n_resamples=400, seed=11, method=PR.PERCENTILE_RULE)
    rng = np.random.default_rng(11)
    draws = np.sort([mean_stat([UNITS[j] for j in rng.integers(0, len(UNITS), size=len(UNITS))]) for _ in range(400)])
    k, top = PR.rank_positions(400, 0.05)
    assert (got.lo, got.hi) == (draws[k], draws[top])
    assert got.as_dict()["convention_id"] == PR.CONVENTION_ID


def test_방식을_주고_미정의를_빼지_않으면_거부한다() -> None:
    with pytest.raises(ValueError, match="NaN"):
        cluster_bootstrap(UNITS, sometimes_nan, n_resamples=200, seed=3, method=PR.PERCENTILE_RULE)
    got = cluster_bootstrap(UNITS, sometimes_nan, n_resamples=200, seed=3, drop_undefined=True,
                            method=PR.PERCENTILE_RULE)
    assert got.n_undefined > 0 and got.convention_id == PR.CONVENTION_ID


def test_동결_모듈은_이_규약을_쓰지_않는다() -> None:
    """`evaluation/recovery_ci.py` 는 고치지 않는다 — 사전실험의 값은 그 계약의 값이다. 새 모듈은 그것을 부르지 않는다."""
    import ast
    from pathlib import Path
    for mod in (RI, PR):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert "evaluation.recovery_ci" not in mods and not names & {"recovery_from", "percentile_ci"}


# ---------------------------------------------------------------- 방식 경로 — 상태 · 무한 · 빈 입력 (검토 10-02 병합 뒤 1)

def sometimes_inf(gs):
    """첫 묶음의 값이 크면 +∞ — 추첨의 약 10 % 에서. 전체(점추정)의 첫 묶음은 0 이라 유한하다."""
    return float("inf") if VAL[gs[0]] > 0.9 else float(np.mean([VAL[g] for g in gs]))


@pytest.mark.parametrize("drop", [True, False])
def test_방식_경로는_무한을_거부한다(drop: bool) -> None:
    """미정의를 빼는 통계량에 ±무한이 오면 거부한다(07번 §33-6) — 기본 경로처럼 `isfinite` 로 버리지 않는다."""
    with pytest.raises(ValueError, match="무한"):
        cluster_bootstrap(UNITS, sometimes_inf, n_resamples=200, seed=3, drop_undefined=drop, method=PR.PERCENTILE_RULE)


def test_방식_경로의_산출은_상태를_싣고_표준_JSON_이다() -> None:
    got = cluster_bootstrap(UNITS, mean_stat, n_resamples=200, seed=5, method=PR.PERCENTILE_RULE)
    d = got.as_dict()
    assert {"ci_lo_state", "ci_hi_state", "half_width_state", "convention_id", "point_state"} <= set(d)
    assert d["ci_lo_state"] == d["ci_hi_state"] == d["half_width_state"] == "finite"
    assert d["half_width"] == pytest.approx(got.half_width)
    json.dumps(d, allow_nan=False)


def test_방식_경로에서_추첨이_모두_미정의면_구간이_없다() -> None:
    def always_nan(gs):
        return float("nan")
    got = cluster_bootstrap(UNITS, always_nan, n_resamples=50, seed=1, drop_undefined=True, method=PR.PERCENTILE_RULE)
    d = got.as_dict()
    assert (d["ci_lo_state"], d["ci_hi_state"], d["half_width_state"]) == ("absent", "absent", "absent")
    assert d["ci_lo"] is None and d["point"] is None and d["point_state"] == "undefined"
    assert d["n_undefined_resamples"] == 50 and d["convention_id"] == PR.CONVENTION_ID
    json.dumps(d, allow_nan=False)


def test_방식_경로의_빈_입력은_규약_id_를_실은_없음이다() -> None:
    got = cluster_bootstrap([], mean_stat, method=PR.PERCENTILE_RULE)
    d = got.as_dict()
    assert d["ci_lo_state"] == "absent" and d["convention_id"] == PR.CONVENTION_ID and d["n_resamples"] == 0
    json.dumps(d, allow_nan=False)
    old = cluster_bootstrap([], mean_stat)
    assert (old.point, old.lo, old.hi, old.convention_id) == (0.0, 0.0, 0.0, None), "기본 경로의 빈 입력은 그대로다"


@pytest.mark.parametrize("alpha", [0.10, 0.01])
def test_α_가_다르면_회복률_구간과_묶음_구간도_채택_id_가_아니다(alpha: float) -> None:
    """검토(10-04 대조) — α = 0.10 으로 불러도 채택 id 가 붙던 자리. 등록 경로는 α 를 따로 거부하고, 여기는 id 가 다르다."""
    f, c, l, s = draws_with(n_fin=200)
    ri = RI.recovery_interval(f, c, l, s, alpha=alpha, n_resamples=200)
    assert ri.as_dict()["convention_id"] != PR.CONVENTION_ID and f"alpha={alpha!r}" in ri.as_dict()["convention_id"]
    got = cluster_bootstrap(UNITS, mean_stat, n_resamples=200, seed=5, alpha=alpha, method=PR.PERCENTILE_RULE)
    assert got.as_dict()["convention_id"] == PR.convention_id(PR.PERCENTILE_RULE, alpha) != PR.CONVENTION_ID
    assert cluster_bootstrap([], mean_stat, alpha=alpha, method=PR.PERCENTILE_RULE).as_dict()["convention_id"] \
        == PR.convention_id(PR.PERCENTILE_RULE, alpha)
