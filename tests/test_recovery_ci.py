"""회복률 CI 의 집계 재구현 — pycocotools 와 같은 수를 내는가 (36번 §5 T2·T3·T4).

지키는 것:
1. **항등** — 중복도 전부 1 이면 `coco_map` 과 같다(합성 데이터, 마지막 자리까지).
2. **중복 = 블록 반복** — 이미지를 **인접한 id 로 물리적으로 복제**한 데이터를 pycocotools 로
   직접 평가한 값 = 원본 + 중복도로 낸 집계값. 동점 검출·빈 이미지·GT 0 카테고리·검출 0
   카테고리를 섞는다. 이것이 부트스트랩이 "같은 이미지가 두 번 뽑힌 것" 을 정확히 재는 근거다.
3. **결정론** — 같은 시드면 같은 중복도.
4. **미정의** — 분모 ≤ 0 이면 nan.
5. 항등 실패는 예외다 — 근사값으로 판정하지 않는다.
6. **절단과 동점 순서**(C 42번 m-5) — 이미지·카테고리당 100건 절단을 타는 데이터와, 같은 이미지 안의
   동점 tp·fp 를 손계산 값으로 고정한다. 우연한 동점에 기대지 않는다.
7. **짝지은 부트스트랩**(C 42번 m-6) — 추첨마다 중복도 하나를 시드 × 칸 전부에 쓴다. 미정의는 버리고 센다.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from evaluation.metrics.localization import coco_map
from evaluation.recovery_ci import (
    MAX_DETS,
    IdentityCheckFailed,
    build_cache,
    group_weights,
    identity_check,
    paired_bootstrap,
    percentile_ci,
    recovery_from,
    weighted_map,
)

CLASSES = ["100", "2011", "401"]


def _synthetic(seed: int, n_img: int = 40):
    """동점·빈 이미지·GT 없는 카테고리를 일부러 섞은 작은 평가셋."""
    rng = random.Random(seed)
    gold, pred = {}, {}
    for i in range(n_img):
        iid = f"img{i:03d}"
        g, p = [], []
        for _ in range(rng.randint(0, 3)):
            x, y = rng.uniform(0, 500), rng.uniform(0, 300)
            w, h = rng.uniform(20, 120), rng.uniform(20, 120)
            code = rng.choice(CLASSES[:2])            # '401' 은 GT 가 없다
            g.append((code, (x, y, x + w, y + h)))
            if rng.random() < 0.7:                   # 맞힌 검출 — 살짝 흔든 박스
                dx, dy = rng.uniform(-8, 8), rng.uniform(-8, 8)
                score = rng.choice([0.9, 0.9, 0.7, 0.5, 0.3])   # 동점 다수
                p.append((code, (x + dx, y + dy, x + w + dx, y + h + dy), score))
        for _ in range(rng.randint(0, 2)):           # 오탐
            x, y = rng.uniform(0, 500), rng.uniform(0, 300)
            p.append((rng.choice(CLASSES), (x, y, x + 40, y + 40), rng.choice([0.9, 0.4, 0.1])))
        gold[iid] = g
        pred[iid] = p
    return pred, gold


def test_항등_중복도_전부_1이면_coco_map_과_같다():
    pred, gold = _synthetic(1)
    ref = coco_map(pred, gold, CLASSES)
    cache = build_cache(pred, gold, CLASSES)
    m50, m5095 = weighted_map(cache, np.ones(cache.n_images, dtype=int))
    assert m50 == ref["map_50"]
    assert m5095 == ref["map_50_95"]
    assert identity_check(cache, ref["map_50"], ref["map_50_95"])["passed"]


@pytest.mark.parametrize("seed", range(6))
def test_중복은_인접_id_물리_복제와_같다(seed):
    """이미지 i 를 w_i 번 복제(id 를 인접하게 배치)한 데이터를 pycocotools 로 직접 평가한 값과
    원본 + 중복도로 낸 집계값이 같아야 한다 — 마지막 자리까지."""
    pred, gold = _synthetic(100 + seed, n_img=30)
    ids = sorted(gold)
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 4, size=len(ids))            # 0 도 섞는다(안 뽑힌 묶음)
    if w.sum() == 0:
        w[0] = 1
    # 물리 복제: 원본 순서를 지키며 i 의 사본을 인접 id 로 — 이어 붙이는 순서가 블록 반복과 같다
    dup_gold, dup_pred = {}, {}
    for i, iid in enumerate(ids):
        for r in range(int(w[i])):
            nid = f"{iid}_r{r}"
            dup_gold[nid] = list(gold[iid])
            dup_pred[nid] = list(pred[iid])
    ref = coco_map(dup_pred, dup_gold, CLASSES)
    cache = build_cache(pred, gold, CLASSES)
    m50, m5095 = weighted_map(cache, w)
    assert m50 == pytest.approx(ref["map_50"], abs=1e-12)
    assert m5095 == pytest.approx(ref["map_50_95"], abs=1e-12)


def test_항등_실패는_예외다():
    pred, gold = _synthetic(3)
    cache = build_cache(pred, gold, CLASSES)
    with pytest.raises(IdentityCheckFailed, match="항등 검사 실패"):
        identity_check(cache, 0.123456)


def test_중복도_길이가_틀리면_거부():
    pred, gold = _synthetic(4)
    cache = build_cache(pred, gold, CLASSES)
    with pytest.raises(ValueError):
        weighted_map(cache, np.ones(cache.n_images + 1, dtype=int))


def test_묶음_추첨은_결정론적이고_묶음_단위다():
    g = np.array([0, 0, 1, 1, 1, 2])              # 이미지 6장 → 묶음 3개
    a = group_weights(g, 3, np.random.default_rng(7))
    b = group_weights(g, 3, np.random.default_rng(7))
    assert np.array_equal(a, b)
    # 같은 묶음의 이미지는 같은 중복도를 받는다
    assert a[0] == a[1] and a[2] == a[3] == a[4]


def test_분모가_0_이하면_회복률은_nan():
    r, d = recovery_from(central=0.30, fed=0.35, locals_=[0.30, 0.31, 0.29])
    assert d == pytest.approx(0.0) and np.isnan(r)
    r2, d2 = recovery_from(central=0.49, fed=0.28, locals_=[0.36, 0.13, 0.09])
    assert d2 > 0 and r2 == pytest.approx((0.28 - (0.36 + 0.13 + 0.09) / 3) / d2)


def test_GT_없는_카테고리는_평균에서_빠진다():
    """'401' 은 GT 가 없다 — pycocotools 는 그 카테고리를 -1 로 두고 평균에서 뺀다."""
    pred, gold = _synthetic(5)
    ref = coco_map(pred, gold, CLASSES)
    cache = build_cache(pred, gold, CLASSES)
    m50, _ = weighted_map(cache, np.ones(cache.n_images, dtype=int))
    assert m50 == ref["map_50"]
    assert len(cache.per_class) == 3 and cache.per_class[2].npig.sum() == 0


def test_반폭_상한은_prereg_와_stats_가_같다():
    """`prereg.RECOVERY_CI_MAX_HALF_WIDTH` docstring 이 약속한 동일성(C 42번 m-8)."""
    from evaluation.prereg import RECOVERY_CI_MAX_HALF_WIDTH
    from evaluation.stats import MAX_RECOVERY_HALF_WIDTH

    assert RECOVERY_CI_MAX_HALF_WIDTH == MAX_RECOVERY_HALF_WIDTH


# --------------------------------------------------------------------------------------
# C 42번 m-5 — 절단 · 동점 순서
# --------------------------------------------------------------------------------------

def _physical_dup(pred, gold, w):
    ids = sorted(gold)
    dup_gold, dup_pred = {}, {}
    for i, iid in enumerate(ids):
        for r in range(int(w[i])):
            dup_gold[f"{iid}_r{r}"] = list(gold[iid])
            dup_pred[f"{iid}_r{r}"] = list(pred.get(iid, []))
    return dup_pred, dup_gold


def test_이미지당_검출_100건_초과도_coco_map_과_같다():
    """pycocotools 는 이미지·카테고리당 점수 상위 100건만 본다. 12위 참검출(10건으로 자르면 사라진다)과
    111위 참검출(100건 절단이 버린다)을 함께 넣어, 절단 위치가 틀리면 값이 바뀌게 한다."""
    gold, pred = {}, {}
    for i in range(3):
        iid = f"big{i}"
        gts = [("100", (10.0 + 60 * j, 10.0, 60.0 + 60 * j, 60.0)) for j in range(3)]
        dets = [("100", (1000.0 + k, 1000.0, 1040.0 + k, 1040.0), 0.99 - 0.001 * k) for k in range(11)]
        dets.append(("100", gts[0][1], 0.50))                                   # 12위 — 참
        dets += [("100", (2000.0 + k, 2000.0, 2040.0 + k, 2040.0), 0.40 - 0.001 * k) for k in range(98)]
        dets.append(("100", gts[1][1], 0.01))                                   # 111위 — 참, 버려진다
        gold[iid], pred[iid] = gts, dets
    gold["small"] = [("2011", (5.0, 5.0, 50.0, 50.0))]
    pred["small"] = [("2011", (6.0, 5.0, 51.0, 50.0), 0.8)]

    ref = coco_map(pred, gold, CLASSES)
    cache = build_cache(pred, gold, CLASSES)
    assert int(cache.per_class[0].length.max()) == MAX_DETS == 100          # 절단이 실제로 일어났다
    m50, m5095 = weighted_map(cache, np.ones(cache.n_images, dtype=int))
    assert (m50, m5095) == (ref["map_50"], ref["map_50_95"])
    assert 0.0 < ref["map_50"] < 1.0

    w = np.array([2, 0, 1, 3])                          # sorted: big0 big1 big2 small
    dup = coco_map(*_physical_dup(pred, gold, w), CLASSES)
    d50, d5095 = weighted_map(cache, w)
    assert d50 == pytest.approx(dup["map_50"], abs=1e-12)
    assert d5095 == pytest.approx(dup["map_50_95"], abs=1e-12)


def test_동점_참검출과_오탐은_블록_순서로_반복된다():
    """같은 이미지 안 동점(0.9) 참검출 d1 · 오탐 d2 를 두 번 뽑는다.

    블록 반복 d1 d2 d1 d2 → 재현율 0.5 이하에서 정밀도 1, 그 위에서 2/3.
    원소 반복 d1 d1 d2 d2 → 재현율 1 까지 정밀도 1 → AP 1.0. 둘을 손계산 값으로 가른다.
    """
    box = (10.0, 10.0, 60.0, 60.0)
    gold = {"a": [("100", box)]}
    pred = {"a": [("100", box, 0.9), ("100", (300.0, 300.0, 350.0, 350.0), 0.9)]}
    cache = build_cache(pred, gold, CLASSES)
    m50, _ = weighted_map(cache, np.array([2]))

    dup = coco_map(*_physical_dup(pred, gold, [2]), CLASSES)
    assert m50 == pytest.approx(dup["map_50"], abs=1e-12)

    n_low = int((cache.rec_thrs <= 0.5).sum())         # 재현율 격자에서 0.5 이하인 점 수
    block = (n_low * 1.0 + (cache.rec_thrs.size - n_low) * (2 / 3)) / cache.rec_thrs.size
    element = 1.0
    assert m50 == pytest.approx(block, abs=1e-9)
    assert abs(m50 - element) > 0.1


# --------------------------------------------------------------------------------------
# C 42번 m-6 — 짝지은 부트스트랩 · 백분위 구간
# --------------------------------------------------------------------------------------

def _thinned(pred, keep, seed):
    rng = random.Random(seed)
    return {i: [d for d in ds if rng.random() < keep] for i, ds in pred.items()}


def _boot_caches(central_keep=1.0):
    pred, gold = _synthetic(11, n_img=24)
    variants = {"c": _thinned(pred, central_keep, 0), "l1": _thinned(pred, 0.4, 1),
                "l2": _thinned(pred, 0.4, 2), "l3": _thinned(pred, 0.4, 3), "f": _thinned(pred, 0.7, 4)}
    caches = {t: build_cache(p, gold, CLASSES) for t, p in variants.items()}
    groups = np.repeat(np.arange(8), 3)                  # 이미지 24장 → 묶음 8개
    return caches, groups


def _boot(per_seed, groups, n_res, seed):
    return paired_bootstrap(per_seed, groups, 8, seeds=sorted(per_seed), central="c",
                            locals_=("l1", "l2", "l3"), fed="f", n_resamples=n_res,
                            rng=np.random.default_rng(seed))


def test_짝지은_부트스트랩은_추첨마다_중복도_하나를_시드와_칸_전부에_쓴다():
    caches, groups = _boot_caches()
    n_res = 25
    out = _boot({1: caches, 2: caches}, groups, n_res, seed=3)

    # 칸들이 실제로 다르다 — 아래 대조가 빈 대조가 아니다
    assert not np.array_equal(out["map"][1]["c"], out["map"][1]["l1"])
    # 시드 2 는 시드 1 과 같은 캐시다. 같은 추첨을 받았으면 비트가 같고, 시드마다 따로 뽑았으면 갈린다
    for t in caches:
        assert np.array_equal(out["map"][1][t], out["map"][2][t]), t
    assert np.array_equal(out["R"][1], out["R"][2], equal_nan=True)

    # 재생 — 같은 난수로 추첨마다 **한 번** 뽑은 중복도로 다시 계산하면 같다(칸마다 따로 뽑지 않는다)
    rng = np.random.default_rng(3)
    for b in range(n_res):
        w = group_weights(groups, 8, rng)
        m = {t: weighted_map(c, w)[0] for t, c in caches.items()}
        for t in caches:
            assert out["map"][1][t][b] == m[t]
        r, d = recovery_from(m["c"], m["f"], [m["l1"], m["l2"], m["l3"]])
        assert out["D"][1][b] == d
        assert (np.isnan(r) and np.isnan(out["R"][1][b])) or out["R"][1][b] == r
        rbar = out["R_bar"][b]
        assert (np.isnan(r) and np.isnan(rbar)) or rbar == float(np.mean([r, r]))


def test_분모가_0_이하인_재표집은_nan_이고_한_시드라도_nan_이면_R̄_도_nan():
    good, groups = _boot_caches()
    bad, _ = _boot_caches(central_keep=0.0)             # 중앙 칸 검출 0 → 분모 ≤ 0
    n_res = 20
    only_bad = _boot({1: bad}, groups, n_res, seed=5)
    assert np.isnan(only_bad["R"][1]).all() and (only_bad["D"][1] <= 0).all()
    ci = percentile_ci(only_bad["R_bar"], 0.05)
    assert ci == {"ci_lo": None, "ci_hi": None, "half_width": None, "n_defined": 0, "n_undefined": n_res}

    mixed = _boot({1: good, 2: bad}, groups, n_res, seed=5)
    assert np.isfinite(mixed["R"][1]).any()
    assert np.isnan(mixed["R_bar"]).all()


def test_백분위_구간은_유한값만으로_내고_수를_센다():
    core = np.linspace(0.0, 1.0, 101)
    draws = np.array([np.nan, *core, np.nan])
    ci = percentile_ci(draws, 0.05)
    lo, hi = np.percentile(core, [2.5, 97.5])
    assert (ci["ci_lo"], ci["ci_hi"]) == (float(lo), float(hi))
    assert ci["half_width"] == float((hi - lo) / 2)
    assert (ci["n_defined"], ci["n_undefined"]) == (101, 2)
    assert percentile_ci(draws, 0.10)["half_width"] < ci["half_width"]   # 90 % 구간은 더 좁다
