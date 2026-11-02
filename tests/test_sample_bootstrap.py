"""층 안 묶음 재표집(`evaluation.sample_bootstrap`) — 합성 표본만. 실물 목록 · 곁 파일을 읽지 않는다.

지키는 것.

1. 층은 곁 파일에서만 오고 묶음 단위다 — 두 층에 걸친 묶음 · 층의 묶음 수와 `n_h` 의 불일치는 거부한다.
2. 추첨은 등록 시드로 결정적이고, 추첨마다 층의 묶음 중복도 합이 `n_h` 이며 묶음의 이미지는 같은 중복도를 받는다.
3. 행렬 M1 은 계수표 위에서 고정 대상 `score_m1` 과 같다 — 독립 구현(이미지 고리)과도 맞댄다. 잴 수 없는 추첨은 `NaN` 과 `support`.
4. 항등 추첨은 점추정이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics.box_f1 import CountTable, score_m1
from evaluation.sample_bootstrap import (IDENTITY_TOL, SampleDesignError, design_weights, draw_multiplicities,
                                         draw_weights, identity_problems, m1_draws, multiplicity_problems,
                                         multiplicity_sha256, strata_groups, weights_from)
from evaluation.sample_weights import SideRow

CLASSES = ("100", "2011", "301", "401")


def world(*, groups_per_stratum=(3, 2), imgs_per_group=2, G=(9, 4)):
    """층 둘 · 층마다 뽑힌 묶음 · 묶음마다 이미지. 곁 파일은 `G_h` · `n_h` 를 싣는다."""
    ids, group_of, side = [], {}, {}
    for h, (n_h, G_h) in enumerate(zip(groups_per_stratum, G, strict=True)):
        for g in range(n_h):
            for k in range(imgs_per_group):
                i = f"img{h}{g}{k}"
                ids.append(i)
                group_of[i] = f"g{h}{g}"
                side[i] = SideRow(f"S{h}", G_h, n_h)
    return ids, group_of, side


def random_table(ids, seed=0, *, zero_class=None) -> CountTable:
    rng = np.random.default_rng(seed)
    n, k = len(ids), len(CLASSES)
    tp = rng.integers(0, 3, (n, k))
    fp = rng.integers(0, 2, (n, k))
    fn = rng.integers(0, 2, (n, k))
    if zero_class is not None:          # 한 클래스의 지지량을 몇 이미지에만 둔다 — 추첨에서 사라질 수 있게
        c = CLASSES.index(zero_class)
        tp[:, c] = fn[:, c] = 0
        tp[0, c] = 1
    return CountTable(CLASSES, tuple(sorted(ids)), tp.astype(np.int32), fp.astype(np.int32), fn.astype(np.int32))


def test_층은_곁_파일에서_오고_묶음_단위다() -> None:
    ids, group_of, side = world()
    got = strata_groups(ids, group_of, side)
    assert got == {"S0": ["g00", "g01", "g02"], "S1": ["g10", "g11"]}


@pytest.mark.parametrize("break_", ["cross", "n_h", "no_side", "no_group"])
def test_곁_파일과_묶음이_어긋나면_거부한다(break_) -> None:
    ids, group_of, side = world()
    if break_ == "cross":
        side["img001"] = SideRow("S1", 4, 2)            # 묶음 g00 의 둘째 이미지만 다른 층
    elif break_ == "n_h":
        for i in ids:
            if side[i].stratum == "S1":
                side[i] = SideRow("S1", 4, 3)           # 뽑힌 묶음은 둘인데 n_h = 3
    elif break_ == "no_side":
        del side[ids[0]]
    else:
        del group_of[ids[0]]
    with pytest.raises(SampleDesignError):
        strata_groups(ids, group_of, side)


def test_추첨은_등록_시드로_결정적이다() -> None:
    ids, group_of, side = world()
    a = draw_weights(ids, group_of, side, n_resamples=50, rng_seed=7)
    b = draw_weights(ids, group_of, side, n_resamples=50, rng_seed=7)
    c = draw_weights(ids, group_of, side, n_resamples=50, rng_seed=8)
    assert a.shape == (50, len(ids)) and np.array_equal(a, b) and not np.array_equal(a, c)


def test_층마다_묶음_중복도의_합은_n_h_이고_묶음의_이미지는_같은_중복도다() -> None:
    ids, group_of, side = world()
    W = draw_weights(ids, group_of, side, n_resamples=200, rng_seed=3)
    base = design_weights(ids, side)
    m = W / base                                        # 이미지마다 중복도
    assert np.allclose(m, np.round(m)) and (m >= 0).all()
    for h, gs in strata_groups(ids, group_of, side).items():
        n_h = len(gs)
        per_group = []
        for g in gs:
            cols = [k for k, i in enumerate(ids) if group_of[i] == g]
            assert (m[:, cols] == m[:, cols[:1]]).all()     # 묶음 안은 같은 중복도
            per_group.append(m[:, cols[0]])
        assert (np.sum(per_group, axis=0) == n_h).all(), h
    assert (m.mean(axis=0) > 0.5).all()                 # 어느 묶음도 늘 빠지지 않는다(추첨이 실제로 섞는다)


@pytest.mark.parametrize(("kw", "bad"), [("n_resamples", 0), ("n_resamples", True), ("rng_seed", -1), ("rng_seed", 1.0)])
def test_추첨_수와_시드는_등록의_정수만_받는다(kw, bad) -> None:
    ids, group_of, side = world()
    args = {"n_resamples": 5, "rng_seed": 1} | {kw: bad}
    with pytest.raises(ValueError):
        draw_weights(ids, group_of, side, **args)


def _naive_macro(table: CountTable, w: np.ndarray, classes) -> float | None:
    """독립 구현 — 이미지를 하나씩 돌며 클래스별 가중 합을 낸다(행렬을 쓰지 않는다)."""
    sums = {c: [0.0, 0.0, 0.0] for c in classes}
    for r in range(len(table.image_ids)):
        for c in classes:
            k = table.classes.index(c)
            sums[c][0] += w[r] * float(table.tp[r, k])
            sums[c][1] += w[r] * float(table.fp[r, k])
            sums[c][2] += w[r] * float(table.fn[r, k])
    f1 = []
    for c in classes:
        tp, fp, fn = sums[c]
        if tp + fn <= 0:
            return None
        f1.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1) / len(f1)


def test_행렬_M1_은_고정_대상_score_m1_과_독립_구현과_같다() -> None:
    ids, group_of, side = world(groups_per_stratum=(4, 3), imgs_per_group=3, G=(20, 9))
    table = random_table(ids, seed=1, zero_class="401")
    W = draw_weights(table.image_ids, group_of, side, n_resamples=120, rng_seed=11)
    macro, ok = m1_draws(table, W, CLASSES)
    assert (~ok).any() and ok.any()                    # 지지량이 사라지는 추첨과 아닌 추첨이 다 있다
    for b in range(W.shape[0]):
        rep = score_m1(table, W[b], fixed_classes=CLASSES)
        naive = _naive_macro(table, W[b], CLASSES)
        if ok[b]:
            assert rep.undefined_reason is None
            assert abs(macro[b] - rep.macro_f1) < IDENTITY_TOL and abs(macro[b] - naive) < IDENTITY_TOL
        else:
            assert np.isnan(macro[b]) and rep.undefined_reason == "support" and naive is None


def test_항등_추첨은_점추정이다() -> None:
    ids, group_of, side = world()
    table = random_table(ids, seed=2)
    classes = score_m1(table, design_weights(table.image_ids, side)).classes_in_macro
    assert identity_problems(table, side, group_of, classes) == []


def test_항등_추첨의_어긋남을_잡는다(monkeypatch) -> None:
    ids, group_of, side = world()
    table = random_table(ids, seed=2)
    classes = score_m1(table).classes_in_macro
    import evaluation.sample_bootstrap as SB

    monkeypatch.setattr(SB, "m1_draws", lambda t, W, c: (np.array([0.123]), np.array([True])))
    assert SB.identity_problems(table, side, group_of, classes)


@pytest.mark.parametrize("W", [np.full((2, 3), -1.0), np.full((2, 3), np.nan), np.ones((2, 2))])
def test_행렬_M1_은_잘못된_가중을_받지_않는다(W) -> None:
    table = random_table(["a", "b", "c"], seed=0)
    with pytest.raises(ValueError):
        m1_draws(table, W, CLASSES)


def test_중복도는_정수이고_층마다_합이_n_h_다() -> None:
    ids, group_of, side = world()
    groups = strata_groups(ids, group_of, side)
    M = draw_multiplicities(groups, n_resamples=100, rng_seed=5)
    assert M.dtype == np.int64 and M.shape == (100, 5) and multiplicity_problems(M, groups) == []
    bad = M.copy()
    bad[0, 0] += 1                                       # 층 S0 의 첫 추첨 합이 n_h + 1
    assert multiplicity_problems(bad, groups)
    assert multiplicity_sha256(M) == multiplicity_sha256(M.copy()) != multiplicity_sha256(bad)


def test_항등_중복도의_가중은_설계_가중이다() -> None:
    ids, group_of, side = world()
    groups = strata_groups(ids, group_of, side)
    ones = np.ones((1, sum(len(g) for g in groups.values())), dtype=np.int64)
    assert np.array_equal(weights_from(ones, ids, group_of, side, groups)[0], design_weights(ids, side))


def test_항등_검사는_재표집의_가중_길이_틀리면_잡는다(monkeypatch) -> None:
    """중복도 → 열 대응이나 G_h/n_h 곱이 틀리면 — 항등 추첨이 설계 가중과 달라진다(외부 통계 검토 F1)."""
    import evaluation.sample_bootstrap as SBM

    ids, group_of, side = world()
    table = random_table(ids, seed=3)
    classes = score_m1(table).classes_in_macro
    real = SBM.weights_from
    monkeypatch.setattr(SBM, "weights_from", lambda M, *a: real(M, *a) / 2.0)        # 설계 가중을 빠뜨린 꼴
    assert any("설계 가중" in p for p in SBM.identity_problems(table, side, group_of, classes))


def test_draw_weights_는_중복도_길과_같다() -> None:
    ids, group_of, side = world()
    groups = strata_groups(ids, group_of, side)
    M = draw_multiplicities(groups, n_resamples=30, rng_seed=9)
    assert np.array_equal(draw_weights(ids, group_of, side, n_resamples=30, rng_seed=9),
                          weights_from(M, ids, group_of, side, groups))
