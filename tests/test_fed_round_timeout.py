"""연합 라운드 응답 대기 상한과 집계 순서(2026-10-17 본실험 연합 사망, 06 §8) — **fl extra(flwr) 필요**, 모델 · GPU 없이.

본다:
- 서버가 flwr `Strategy.start` 기본 `timeout=3600` 에 기대지 않는다 — 실행 설정 `round-timeout`(비면 메시지 TTL 12 h).
- 실제 `Strategy.start` + `WeldFedAvg` 를 시간을 모사한 grid 로 돌려, 한 GPU 에서 클라이언트 셋이 차례로 도는 길이
  (C1 ≈ 2.1 h)에 3600 초면 응답 0 개로 죽고(사고 재현) 기본 상한이면 서는지.
- 응답 도착 순서가 바뀌어도(GPU 여럿에 동시) 가중 평균이 같은 바이트인지 — float64 로 더하는 순서가 float32 결과를 바꾸는 사례로.
"""

from __future__ import annotations

import ast
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flwr", reason="fl extra 미설치")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flwr.app import DEFAULT_TTL
from flwr.common import ArrayRecord

from fl import server_app as S
from fl.strategy import RoundFailure, WeldFedAvg
from tests.test_fl_strategy import KEYS, _arrays, _FakeMsg, _ref

#: 본실험 N = 1/8 의 클라이언트 학습 시간(초, 01 §9 의 C1 2.13 · C2 1.33 · C3 0.60 h) — 한 actor 가 차례로 돈다.
SEQUENTIAL_S = (7_668.0, 4_788.0, 2_160.0)


class _ClockGrid:
    """flwr `InMemoryGrid.send_and_receive` 의 꼴 — `timeout` 초까지 온 응답만 돌려준다(시계는 모사, 잠들지 않는다).

    `concurrent=False` 면 클라이언트가 차례로 돌아 i 번째 응답이 앞 학습 시간의 누적 합에 온다. `order` 는 도착 순서다."""

    def __init__(self, durations, *, concurrent: bool = False, order=None, values=(1.0, 4.0, 4.0), weights=(200, 100, 100)):
        self.durations, self.concurrent = list(durations), concurrent
        self.order = list(order) if order is not None else list(range(len(self.durations)))
        self.values, self.weights = values, weights
        self.timeouts: list = []

    def get_node_ids(self):
        return list(range(len(self.durations)))

    def send_and_receive(self, messages, *, timeout=None):
        self.timeouts.append(timeout)
        messages = list(messages)
        if not messages:
            return []
        t, out = 0.0, []
        for i in self.order:
            t = self.durations[i] if self.concurrent else t + self.durations[i]
            if timeout is None or t <= timeout:
                out.append(_FakeMsg(i, arrays=_arrays(self.values[i]), num_examples=self.weights[i]))
        return out


def _strategy(seen=None) -> WeldFedAvg:
    return WeldFedAvg(expected_nodes=3, canonical_keys=KEYS, reference_state_dict=_ref(),
                      on_round_end=None if seen is None else (lambda r, cells, agg: seen.append(cells)))


def _start(grid, timeout):
    return S.start_rounds(_strategy(), grid=grid, initial_arrays=ArrayRecord(_arrays(0.0)), num_rounds=1,
                          train_config=None, timeout=timeout)


def test_round_timeout_은_비면_메시지_TTL_이고_틀린_값을_거부한다():
    assert S.ROUND_TIMEOUT_MAX == float(DEFAULT_TTL) == 43_200.0
    for cfg in ({}, {"round-timeout": ""}, {"round-timeout": "  "}):
        assert S.round_timeout(cfg) == 43_200.0
    assert S.round_timeout({"round-timeout": "21600"}) == S.round_timeout({"round-timeout": 21600}) == 21_600.0
    for bad in ("x", 0, -1, "43200.5", 50_000, True):
        with pytest.raises(S.RoundTimeoutRejected) as exc:
            S.round_timeout({"round-timeout": bad})
        assert exc.value.code == "round_timeout" and isinstance(exc.value, ValueError)


def test_한_GPU_에서_차례로_도는_길이에_3600_초면_응답_0_으로_죽고_기본_상한이면_선다():
    with pytest.raises(RoundFailure, match="유효 응답 0개"):                              # 10-17 사고 그대로
        _start(_ClockGrid(SEQUENTIAL_S), timeout=3_600)
    grid = _ClockGrid(SEQUENTIAL_S)
    res = _start(grid, timeout=S.round_timeout({}))
    assert grid.timeouts[0] == 43_200.0
    assert res.arrays.to_numpy_ndarrays()[0].flatten()[0] == pytest.approx(2.5)
    with pytest.raises(RoundFailure, match="유효 응답 2개"):                              # 상한이 셋째 응답 앞이면 여전히 죽는다
        _start(_ClockGrid(SEQUENTIAL_S), timeout=sum(SEQUENTIAL_S[:2]))


def test_서버_앱은_round_timeout_을_읽어_start_rounds_로만_라운드를_돈다():
    tree = ast.parse((ROOT / "fl" / "server_app.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    names = [getattr(c.func, "id", None) or getattr(c.func, "attr", None) for c in calls]
    assert "start" not in names, "main 이 strategy.start 를 바로 부른다 — 대기 상한이 flwr 기본 3600 초로 돌아간다"
    start = [c for c in calls if getattr(c.func, "id", None) == "start_rounds"]
    assert len(start) == 1 and {k.arg for k in start[0].keywords} >= {"timeout"}
    rt = [c for c in calls if getattr(c.func, "id", None) == "round_timeout"]
    assert len(rt) == 1 and [getattr(a, "id", None) for a in rt[0].args] == ["cfg"]


def test_선언된_실행_설정에_round_timeout_이_있고_빈_값이다():
    import tomllib

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    assert declared["round-timeout"] == ""


def test_도착_순서가_바뀌어도_가중_평균은_같은_바이트이고_회계는_참여자_순이다():
    # float64 로 0.25·2^30 + 0.5·2^-30 을 먼저 더하면 2^-31 이 사라진다 — 순서가 float32 결과를 바꾸는 사례.
    vals, wts = (2.0 ** 30, -(2.0 ** 30), 2.0 ** -30), (1, 1, 2)
    outs, cells_seen = set(), []
    for order in itertools.permutations(range(3)):
        seen: list = []
        grid = _ClockGrid((1.0, 1.0, 1.0), concurrent=True, order=order, values=vals, weights=wts)
        res = S.start_rounds(_strategy(seen), grid=grid, initial_arrays=ArrayRecord(_arrays(0.0)), num_rounds=1,
                             train_config=None, timeout=10.0)
        outs.add(b"".join(a.tobytes() for a in res.arrays.to_numpy_ndarrays()))
        cells_seen.append([int(c["client-idx"]) for c in seen[0]])
    assert len(outs) == 1
    assert np.frombuffer(next(iter(outs))[:4], dtype=np.float32)[0] == np.float32(2.0 ** -31)
    assert all(c == [0, 1, 2] for c in cells_seen)
