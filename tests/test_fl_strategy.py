"""fl/strategy.py 테스트 — **fl extra(flwr) 필요**.

핵심은 하나다: **실패한 라운드가 조용히 성립하지 않는가.** Message API 기본 동작은
에러 응답을 버리고 남은 것으로 재정규화 집계하므로, 그 경로를 막았는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("flwr", reason="fl extra 미설치 — uv sync --extra fl")

from flwr.common import ArrayRecord, ConfigRecord, MetricRecord, RecordDict  # noqa: E402

from detection import serialize  # noqa: E402
from fl.strategy import (ARRAYS_KEY, CONFIG_KEY, METRICS_KEY, WEIGHT_KEY,  # noqa: E402
                         RoundFailure, WeldFedAvg)


def _ref() -> dict[str, torch.Tensor]:
    return {
        "conv.weight": torch.zeros(2, 1),
        "bn.running_mean": torch.zeros(2),
        "bn.num_batches_tracked": torch.tensor(0, dtype=torch.int64),
    }


KEYS = list(_ref().keys())


def _arrays(val: float, tracked: int = 5) -> list[np.ndarray]:
    return [
        np.full((2, 1), val, dtype=np.float32),
        np.full((2,), val, dtype=np.float32),
        np.array(tracked, dtype=np.int64),
    ]


class _FakeMeta:
    def __init__(self, node_id: int) -> None:
        self.src_node_id = node_id


class _FakeError:
    def __init__(self, reason: str) -> None:
        self.reason = reason


class _FakeMsg:
    """Message 의 최소 계약만 흉내낸다 — has_error / has_content / content / metadata."""

    def __init__(self, node_id: int, *, arrays=None, num_examples=0, error=None, digest=None):
        self.metadata = _FakeMeta(node_id)
        self.error = _FakeError(error) if error else None
        self._content = None
        if arrays is not None:
            metrics = {
                WEIGHT_KEY: float(num_examples),
                "client-idx": float(node_id),
                "epochs-ran": 2.0,
            }
            # 문자열은 ConfigRecord 로만 나를 수 있다 — MetricRecord 는 str 을 거부한다
            strings = {
                "keys-digest": digest if digest is not None else serialize.keys_digest(KEYS),
                "optimizer": "SGD",
                "arg-optimizer": "SGD",
            }
            self._content = RecordDict(
                {
                    ARRAYS_KEY: ArrayRecord(arrays),
                    METRICS_KEY: MetricRecord(metrics),
                    CONFIG_KEY: ConfigRecord(strings),
                }
            )

    def has_error(self) -> bool:
        return self.error is not None

    def has_content(self) -> bool:
        return self._content is not None

    @property
    def content(self):
        return self._content


def _strategy(expected=3, on_round_end=None) -> WeldFedAvg:
    return WeldFedAvg(
        expected_nodes=expected,
        canonical_keys=KEYS,
        reference_state_dict=_ref(),
        on_round_end=on_round_end,
    )


# --------------------------------------------------------------------------
# 실패 차단 — 이 파일의 존재 이유
# --------------------------------------------------------------------------

def test_에러_응답이_있으면_라운드가_중단된다():
    """상위 구현은 에러를 로그만 남기고 버린 뒤 남은 것으로 집계한다."""
    replies = [
        _FakeMsg(0, arrays=_arrays(1.0), num_examples=32000),
        _FakeMsg(1, arrays=_arrays(2.0), num_examples=16000),
        _FakeMsg(2, error="CUDA out of memory"),
    ]
    with pytest.raises(RoundFailure, match="실패했다"):
        _strategy().aggregate_train(30, replies)


def test_응답_수가_모자라면_라운드가_중단된다():
    """응답하지 않은 노드는 에러 응답도 남기지 않는다 — 개수가 유일한 탐지 수단이다."""
    replies = [
        _FakeMsg(0, arrays=_arrays(1.0), num_examples=32000),
        _FakeMsg(1, arrays=_arrays(2.0), num_examples=16000),
    ]
    with pytest.raises(RoundFailure, match="유효 응답 2개"):
        _strategy(expected=3).aggregate_train(30, replies)


def test_재정규화_집계가_일어나지_않는다():
    """2/3만 응답한 상태에서 집계가 성공하면 그것이 곧 결함이다."""
    replies = [
        _FakeMsg(0, arrays=_arrays(1.0), num_examples=32000),
        _FakeMsg(1, arrays=_arrays(4.0), num_examples=16000),
    ]
    strat = _strategy(expected=3)
    with pytest.raises(RoundFailure):
        strat.aggregate_train(1, replies)


def test_키_다이제스트가_다르면_중단된다():
    replies = [
        _FakeMsg(0, arrays=_arrays(1.0), num_examples=1, digest="deadbeef"),
        _FakeMsg(1, arrays=_arrays(1.0), num_examples=1),
        _FakeMsg(2, arrays=_arrays(1.0), num_examples=1),
    ]
    with pytest.raises(RoundFailure, match="다이제스트 불일치"):
        _strategy().aggregate_train(1, replies)


def test_어긋난_shape은_집계_전에_잡힌다():
    bad = _arrays(1.0)
    bad[1] = np.zeros((7,), dtype=np.float32)
    replies = [
        _FakeMsg(0, arrays=_arrays(1.0), num_examples=1),
        _FakeMsg(1, arrays=bad, num_examples=1),
        _FakeMsg(2, arrays=_arrays(1.0), num_examples=1),
    ]
    with pytest.raises(serialize.SerializeError):
        _strategy().aggregate_train(1, replies)


# --------------------------------------------------------------------------
# 정상 경로
# --------------------------------------------------------------------------

def test_전원_응답하면_우리_규약으로_집계된다():
    seen: list = []
    strat = _strategy(on_round_end=lambda r, cells, agg: seen.append((r, cells, agg)))
    replies = [
        _FakeMsg(0, arrays=_arrays(1.0, tracked=100), num_examples=200),
        _FakeMsg(1, arrays=_arrays(4.0, tracked=50), num_examples=100),
        _FakeMsg(2, arrays=_arrays(4.0, tracked=10), num_examples=100),
    ]
    arrays_out, metrics = strat.aggregate_train(1, replies)
    back = arrays_out.to_numpy_ndarrays()

    # 가중 평균: (1*200 + 4*100 + 4*100) / 400 = 1000/400 = 2.5
    assert back[0].flatten()[0] == pytest.approx(2.5)
    # num_batches_tracked 는 평균이 아니라 최댓값이어야 한다 (가중 평균이면 65)
    assert back[2].dtype == np.int64 and int(back[2]) == 100
    assert back[2].shape == (), "0차원이 보존돼야 한다"
    assert metrics["total-examples"] == 400.0
    assert len(seen) == 1 and len(seen[0][1]) == 3


def test_평가_라운드는_열리지_않는다():
    strat = _strategy()
    assert strat.configure_evaluate(1, None, None, None) == []
    assert strat.aggregate_evaluate(1, []) is None


def test_평가_응답이_오면_실패한다():
    """열리지 않는 경로라도 열렸다면 조용히 통과시키지 않는다."""
    with pytest.raises(RoundFailure, match="평가 라운드는 쓰지 않는데"):
        _strategy().aggregate_evaluate(1, [_FakeMsg(0, arrays=_arrays(1.0), num_examples=1)])


def test_생성자가_전수_참여를_고정한다():
    """상위 기본값 min_train_nodes=2 를 그대로 두면 한 클라이언트가 빠져도 라운드가 성립한다."""
    strat = _strategy(expected=3)
    assert strat.min_train_nodes == 3
    assert strat.min_available_nodes == 3
    assert strat.fraction_evaluate == 0.0
    assert strat.min_evaluate_nodes == 0
    assert strat.weighted_by_key == WEIGHT_KEY


# --------------------------------------------------------------------------
# VLM 교환 폐포 (G2-3)
# --------------------------------------------------------------------------

def test_어댑터_교환은_집합_등식이어야_한다():
    from fl.client_vlm import adapter_exchange_contract

    ok, fails = adapter_exchange_contract(["a.lora_A", "b.lora_B"], ["a.lora_A", "b.lora_B"])
    assert ok and not fails

    ok, fails = adapter_exchange_contract(["a.lora_A", "b.lora_B", "c.mtp"], ["a.lora_A", "b.lora_B"])
    assert not ok and any("학습되는데 교환되지 않는" in f for f in fails)

    ok, fails = adapter_exchange_contract(["a.lora_A"], ["a.lora_A", "z.frozen"])
    assert not ok and any("교환되는데 학습되지 않는" in f for f in fails)
