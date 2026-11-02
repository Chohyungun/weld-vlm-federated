"""평가 층화 표본의 설계 규칙(25번 §2) — 두 구현이 맞춰야 할 의미를 합성 묶음으로 고정한다.

실물 매니페스트를 읽지 않는다. 공식 시드로 실물 표본을 뽑지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe import eval_sample_ci_estimate as E  # noqa: E402


@pytest.mark.parametrize(("n_def", "n_img", "want"), [
    (0, 5, "0"), (1, 10, "1"), (2, 2, "1"),          # m = 0.1 · 1.0
    (3, 2, "2-3"), (14, 10, "1"), (15, 10, "2-3"),   # 1.5 경계는 2-3 쪽
    (34, 10, "2-3"), (35, 10, "4+"), (7, 2, "4+"),   # 3.5 경계는 4+ 쪽
])
def test_묶음_구간은_정수_비교로(n_def, n_img, want) -> None:
    assert E.group_bucket(n_def, n_img) == want


def test_결함이_있으면_0_구간이_아니다() -> None:
    assert all(E.group_bucket(k, 50) != "0" for k in range(1, 200))


@pytest.mark.parametrize(("G_h", "n", "want"), [
    (400, 3000, 96), (9, 3000, 2), (1, 3000, 1), (2, 2000, 2), (81, 2000, 13), (213, 4000, 68),
    (400, 12461, 400), (400, 20000, 400),
])
def test_배분_식(G_h, n, want) -> None:
    assert E.allocation(G_h, n) == want


def test_배분은_크기에_단조() -> None:
    for G_h in range(1, 500, 7):
        vals = [E.allocation(G_h, n) for n in (2000, 3000, 4000, 12461)]
        assert vals == sorted(vals) and vals[-1] == G_h


def _groups(k: int) -> list[dict]:
    out = []
    for i in range(k):
        out.append({"group_id": f"grp_{i:04d}", "material": "ST" if i % 5 else "AL", "n_def": (i * 7) % 23,
                    "n_img": 1 + i % 9, "rare": i % 11 == 0})
    return out


def test_같은_시드에서_작은_표본은_큰_표본의_부분집합() -> None:
    gs = _groups(600)
    seed = "test_seed|synthetic"
    picks = {}
    for n in (2000, 3000, 4000, 12461):
        sel = E.draw(gs, "P", n, seed)
        picks[n] = {h: c for h, (c, _) in sel.items()}
    for small, big in ((2000, 3000), (3000, 4000), (4000, 12461)):
        for h, c in picks[small].items():
            assert c == picks[big][h][:len(c)], (small, big, h)   # 접두


def test_순서는_시드와_묶음_id_의_sha256() -> None:
    gs = _groups(60)
    sel = E.draw(gs, "P", 12461, "s")
    for h, (chosen, G_h) in sel.items():
        keys = [hashlib.sha256(f"s|{gs[i]['group_id']}".encode("utf-8")).hexdigest() for i in chosen]
        assert keys == sorted(keys) and len(chosen) == G_h


def test_시드가_다르면_다른_표본() -> None:
    gs = _groups(600)
    a = E.draw(gs, "P", 3000, "a")
    b = E.draw(gs, "P", 3000, "b")
    assert any(a[h][0] != b[h][0] for h in a)
    assert {h: len(c) for h, (c, _) in a.items()} == {h: len(c) for h, (c, _) in b.items()}   # 층별 묶음 수는 시드와 무관


def test_복제_시드는_공식_시드와_다른_꼴() -> None:
    assert not E.OFFICIAL_SEED.startswith("ci_estimate|")
    assert E.OFFICIAL_SEED.startswith("eval_sample_v1|")


# ---------------------------------------------------------------- 구간 규약 — 본채점과 같은 분류(외부 검토 137-2 M-03)

def _one(fed, cen, loc):
    """추첨 하나 · 시드 여럿 — 연합 · 중앙 `(1, 시드)`, 로컬 `(1, 시드, 로컬 수)`."""
    return np.array([fed], dtype=float), np.array([cen], dtype=float), np.array([loc], dtype=float)


def _reference_class(fed, cen, loc) -> str:
    """`recovery_interval` 문서의 표를 그대로 옮긴 독립 판정 — 추첨 하나(시드 축)의 범주."""
    if not (np.isfinite(fed).all() and np.isfinite(cen).all() and np.isfinite(loc).all()):
        return "n_excluded_support"
    lbar = loc.sum(axis=-1) / loc.shape[-1]
    num, den = fed - lbar, cen - lbar
    fail = den <= 0
    if (fail & (num == 0.0)).any():
        return "n_indeterminate"
    pos, neg = (fail & (num > 0)).any(), (fail & (num < 0)).any()
    if pos and neg:
        return "n_indeterminate"
    return "n_infinite_pos" if pos else "n_infinite_neg" if neg else "n_finite"


def test_분모_실패에서_분자가_0_이면_판정_불가다() -> None:
    """앞 판은 분자 0 을 +무한으로 셌다(위 꼬리) — 본채점 규약은 판정 불가이고 백분위에 들지 않는다."""
    iv = E.recovery_draws(*_one([0.5], [0.4], [[0.5, 0.5, 0.5]]))
    assert (iv["n_indeterminate"], iv["n_infinite_pos"], iv["n"]) == (1, 0, 0)
    assert iv["indeterminate_reasons"] == {"numerator_zero": 1, "sign_conflict": 0}
    assert (iv["ci_lo_state"], iv["ci_hi_state"], iv["half_width_state"]) == ("absent", "absent", "absent")
    assert iv["ci_lo"] is None and iv["half_width"] is None


def test_분모가_0_이어도_분자가_0_이면_판정_불가다() -> None:
    iv = E.recovery_draws(*_one([0.5], [0.5], [[0.5, 0.5, 0.5]]))
    assert iv["indeterminate_reasons"]["numerator_zero"] == 1 and iv["n_infinite_pos"] == 0


@pytest.mark.parametrize(("fed", "key", "state"), [(0.6, "n_infinite_pos", "pos_inf"), (0.4, "n_infinite_neg", "neg_inf")])
def test_분모_실패의_부호는_분자를_따르고_백분위에_든다(fed, key, state) -> None:
    iv = E.recovery_draws(*_one([fed], [0.45], [[0.5, 0.5, 0.5]]))
    assert iv[key] == 1 and iv["n"] == 1
    assert iv["ci_lo_state"] == iv["ci_hi_state"] == state and iv["half_width_state"] == "undefined"


def test_시드끼리_무한의_부호가_갈리면_판정_불가다() -> None:
    """앞 판은 시드 평균(+무한 + −무한 = NaN)을 지지량 미정의와 한 칸으로 셌다."""
    iv = E.recovery_draws(*_one([0.6, 0.4], [0.45, 0.45], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))
    assert iv["indeterminate_reasons"] == {"numerator_zero": 0, "sign_conflict": 1}
    assert iv["n_excluded_support"] == 0


def test_수가_아닌_항이_있는_추첨은_지지량_제외다() -> None:
    iv = E.recovery_draws(*_one([np.nan], [0.7], [[0.5, 0.5, 0.5]]))
    assert iv["n_excluded_support"] == 1 and iv["n_indeterminate"] == 0 and iv["n"] == 0


def test_추첨_분류는_규약의_표와_같다() -> None:
    """경계를 일부러 섞은 합성 추첨 — 분모 0 · 음수, 분자 0, 시드 사이 부호, 수가 아닌 항."""
    rng = np.random.default_rng(20261014)
    B, S = 600, 3
    loc = rng.choice([0.4, 0.5, 0.6], size=(B, S, 3))
    lbar = loc.sum(axis=-1) / 3
    fed = lbar + rng.choice([-0.1, 0.0, 0.05, 0.1], size=(B, S))
    cen = lbar + rng.choice([-0.1, 0.0, 0.2], size=(B, S))
    fed[rng.random((B, S)) < 0.02] = np.nan
    want: dict[str, int] = {}
    for b in range(B):
        k = _reference_class(fed[b], cen[b], loc[b])
        want[k] = want.get(k, 0) + 1
    iv = E.recovery_draws(fed, cen, loc)
    for key in ("n_finite", "n_infinite_pos", "n_infinite_neg", "n_excluded_support", "n_indeterminate"):
        assert iv[key] == want.get(key, 0), key
    assert iv["n"] == iv["n_finite"] + iv["n_infinite_pos"] + iv["n_infinite_neg"]
    assert min(want.get(k, 0) for k in ("n_finite", "n_infinite_pos", "n_infinite_neg", "n_indeterminate")) > 0


def test_점추정은_분모_실패와_지지량_0_에서_수를_내지_않는다() -> None:
    ok = E.recovery_point(np.array([0.6, 0.7]), np.array([0.8, 0.9]), np.array([[0.4, 0.4, 0.4], [0.5, 0.5, 0.5]]))
    assert ok["state"] == "finite" and ok["R_bar"] == pytest.approx((0.2 / 0.4 + 0.2 / 0.4) / 2)
    den = E.recovery_point(np.array([0.6, 0.7]), np.array([0.4, 0.9]), np.array([[0.4, 0.4, 0.4], [0.5, 0.5, 0.5]]))
    assert (den["R_bar"], den["state"], den["reason"]) == (None, "absent", "denominator")
    sup = E.recovery_point(np.array([np.nan, 0.7]), np.array([0.8, 0.9]), np.array([[0.4, 0.4, 0.4], [0.5, 0.5, 0.5]]))
    assert (sup["R_bar"], sup["state"], sup["reason"]) == (None, "absent", "support")


def test_수가_아닌_값은_상태_칸으로_표준_JSON_에_실린다() -> None:
    assert E.value_state(float("nan")) == {"value": None, "state": "undefined"}
    assert E.value_state(float("inf")) == {"value": None, "state": "pos_inf"}
    assert E.value_state(0.25) == {"value": 0.25, "state": "finite"}
    empty = E.metric_interval(np.array([np.nan, np.nan]))
    assert (empty["n_defined"], empty["n_undefined"], empty["half_width_state"]) == (0, 2, "absent")
    cells = [
        {"point": E.recovery_point(np.array([0.6]), np.array([0.8]), np.array([[0.4, 0.4, 0.4]])),
         "interval": E.recovery_draws(*_one([0.6], [0.8], [[0.4, 0.4, 0.4]]))},
        {"point": E.recovery_point(np.array([0.6]), np.array([0.4]), np.array([[0.5, 0.5, 0.5]])),
         "interval": E.recovery_draws(*_one([0.6], [0.45], [[0.5, 0.5, 0.5]]))},
        {"point": E.recovery_point(np.array([0.5]), np.array([0.4]), np.array([[0.5, 0.5, 0.5]])),
         "interval": E.recovery_draws(*_one([0.5], [0.4], [[0.5, 0.5, 0.5]]))},
    ]
    agg = E.aggregate(cells)
    assert agg["half_states"] == {"absent": 1, "finite": 1, "undefined": 1}
    assert agg["point_states"] == {"absent": 2, "finite": 1}
    assert agg["half_median"] == 0.0 and agg["point_mean"] == pytest.approx(0.5)
    metric = E.aggregate([{"point": E.value_state(float("nan")), "interval": empty}])
    assert metric["half_median"] is None and metric["point_mean"] is None and metric["undefined_share_mean"] == 1.0
    json.dumps({"cells": cells, "agg": agg, "metric": metric}, allow_nan=False)     # NaN · Infinity 가 새면 여기서 깨진다
