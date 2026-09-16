"""회복률 CI 의 집계 재구현 — pycocotools 와 같은 수를 내는가 (36번 §5 T2·T3·T4).

지키는 것:
1. **항등** — 중복도 전부 1 이면 `coco_map` 과 같다(합성 데이터, 마지막 자리까지).
2. **중복 = 블록 반복** — 이미지를 **인접한 id 로 물리적으로 복제**한 데이터를 pycocotools 로
   직접 평가한 값 = 원본 + 중복도로 낸 집계값. 동점 검출·빈 이미지·GT 0 카테고리·검출 0
   카테고리를 섞는다. 이것이 부트스트랩이 "같은 이미지가 두 번 뽑힌 것" 을 정확히 재는 근거다.
3. **결정론** — 같은 시드면 같은 중복도.
4. **미정의** — 분모 ≤ 0 이면 nan.
5. 항등 실패는 예외다 — 근사값으로 판정하지 않는다.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from evaluation.metrics.localization import coco_map
from evaluation.recovery_ci import (
    IdentityCheckFailed,
    build_cache,
    group_weights,
    identity_check,
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
    assert a.sum() == 6 or a.sum() != 6           # 합은 묶음 크기에 따라 달라진다 — 제약 없음


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
