"""fl/aggregate.py 테스트 (지정 함정 구간 #2 — BatchNorm 연합 평균)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from detection import serialize
from fl.aggregate import bn_divergence, weighted_fedavg


def _ref() -> dict[str, torch.Tensor]:
    return {
        "conv.weight": torch.zeros(2, 1, 1, 1),
        "bn.running_mean": torch.zeros(2),
        "bn.running_var": torch.ones(2),
        "bn.num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }


def _client(w: float, mean: float, var: float, tracked: int) -> list[np.ndarray]:
    return [
        np.full((2, 1, 1, 1), w, dtype=np.float32),
        np.full((2,), mean, dtype=np.float32),
        np.full((2,), var, dtype=np.float32),
        np.array(tracked, dtype=np.int64),
    ]


def test_표본수_가중_평균이_수식대로다():
    ref, keys = _ref(), list(_ref().keys())
    a = _client(1.0, 10.0, 1.0, tracked=100)
    b = _client(4.0, 20.0, 2.0, tracked=50)
    # 가중치 2:1 → (1*2 + 4*1)/3 = 2.0
    r = weighted_fedavg([a, b], [200, 100], keys, ref)
    assert r.ndarrays[0].flatten()[0] == pytest.approx(2.0)
    assert r.ndarrays[1].flatten()[0] == pytest.approx((10 * 2 + 20 * 1) / 3)
    assert r.total_examples == 300


def test_num_batches_tracked는_평균이_아니라_최댓값이다():
    """정수 카운터를 평균하면 반올림이 끼어들고 어느 클라이언트의 값도 아니게 된다."""
    ref, keys = _ref(), list(_ref().keys())
    a = _client(1.0, 0.0, 1.0, tracked=100)
    b = _client(1.0, 0.0, 1.0, tracked=51)
    r = weighted_fedavg([a, b], [1, 1], keys, ref)
    tracked = r.ndarrays[3]
    assert tracked.dtype == np.int64, "정수 dtype 이 보존돼야 한다"
    assert int(tracked) == 100, "가중 평균(75.5)이 아니라 최댓값이어야 한다"


def test_0차원_버퍼의_shape이_보존된다():
    ref, keys = _ref(), list(_ref().keys())
    r = weighted_fedavg([_client(1.0, 0.0, 1.0, 7)], [10], keys, ref)
    assert r.ndarrays[3].shape == (), "num_batches_tracked 는 0차원이다"
    # 집계 결과를 그대로 모델에 되돌릴 수 있어야 한다
    sd = serialize.ndarrays_to_state_dict(r.ndarrays, keys, ref)
    assert sd["bn.num_batches_tracked"].shape == ref["bn.num_batches_tracked"].shape


def test_집계_전후_norm이_기록된다():
    ref, keys = _ref(), list(_ref().keys())
    a, b = _client(1.0, 0.0, 1.0, 1), _client(3.0, 0.0, 1.0, 1)
    r = weighted_fedavg([a, b], [1, 1], keys, ref)
    assert len(r.client_norms) == 2
    assert r.client_norms[0] < r.client_norms[1]
    assert r.global_norm > 0


def test_어긋난_입력은_집계하지_않고_실패한다():
    ref, keys = _ref(), list(_ref().keys())
    good = _client(1.0, 0.0, 1.0, 1)
    bad = list(good)
    bad[1] = np.zeros((5,), dtype=np.float32)  # shape 불일치
    with pytest.raises(serialize.SerializeError):
        weighted_fedavg([good, bad], [1, 1], keys, ref)


def test_클라이언트가_없거나_표본수가_0이면_실패한다():
    ref, keys = _ref(), list(_ref().keys())
    with pytest.raises(ValueError, match="클라이언트가 없다"):
        weighted_fedavg([], [], keys, ref)
    with pytest.raises(ValueError, match="가중치를 만들 수 없다"):
        weighted_fedavg([_client(1.0, 0.0, 1.0, 1)], [0], keys, ref)


def test_bn_거리는_분포_차이에_반응한다():
    keys = list(_ref().keys())
    같음 = [_client(1.0, 5.0, 1.0, 1), _client(1.0, 5.0, 1.0, 1)]
    다름 = [_client(1.0, 5.0, 1.0, 1), _client(1.0, 50.0, 9.0, 1)]
    assert bn_divergence(같음, keys) == pytest.approx(0.0)
    assert bn_divergence(다름, keys) > 10.0


def test_누락_분산비는_평균이_같으면_0이다():
    """분산의 가중 평균이 놓치는 것은 클라이언트 간 '평균 차이' 항이다."""
    ref, keys = _ref(), list(_ref().keys())
    동일평균 = weighted_fedavg(
        [_client(1.0, 5.0, 1.0, 1), _client(1.0, 5.0, 4.0, 1)], [1, 1], keys, ref
    )
    assert 동일평균.missing_variance_ratio == pytest.approx(0.0, abs=1e-12)

    갈린평균 = weighted_fedavg(
        [_client(1.0, 0.0, 1.0, 1), _client(1.0, 100.0, 1.0, 1)], [1, 1], keys, ref
    )
    assert 갈린평균.missing_variance_ratio > 0.9, "평균이 크게 갈리면 놓치는 성분이 지배적이다"


def test_진단은_집계_결과를_바꾸지_않는다():
    """진단은 관측 전용이다 — 값이 커도 집계 산출물은 동일해야 한다."""
    ref, keys = _ref(), list(_ref().keys())
    clients = [_client(1.0, 0.0, 1.0, 1), _client(3.0, 100.0, 1.0, 1)]
    r = weighted_fedavg(clients, [1, 1], keys, ref)
    assert r.ndarrays[0].flatten()[0] == pytest.approx(2.0)
    assert r.ndarrays[1].flatten()[0] == pytest.approx(50.0)
    assert r.missing_variance_ratio > 0  # 진단은 켜져 있고
    # 그럼에도 산출물은 순수한 가중 평균 그대로다
