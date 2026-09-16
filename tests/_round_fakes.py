"""`detection.round_runner.train_round` 를 GPU·모델 다운로드 없이 통과시키는 가짜 트레이너.

26번 §8 후속(8-1·8-2·8-3)의 시험이 전부 이 표면을 쓴다. **한 항목의 시험 파일에 두면
다른 항목이 그 파일에 의존하게 되고**, 항목별로 게이트를 여는 지금 구조에서 8-2 를 되돌리면
8-3 시험이 8-3 과 무관한 이유로 깨진다. 그래서 어느 항목의 생산 코드에도 의존하지 않는
중립 헬퍼로 분리했다.

시험 파일이 아니므로 pytest 가 수집하지 않는다(`_` 접두사).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


class Net(torch.nn.Module):
    """`export_weights`·`state_dict` 왕복에 필요한 최소 모델."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)


class RoundTrainer:
    """`train_round` 가 만지는 표면만 가진 트레이너. **학습하지 않는다.**

    `FedDetectionTrainer` 자리에 monkeypatch 로 끼운다. 생성자 인자는 `train_round` 가
    넘기는 그대로 받아 `overrides`·`kw` 에 보관하므로 시험이 무엇이 넘어왔는지 볼 수 있다.
    """

    def __init__(self, overrides=None, **kw) -> None:
        self.overrides = overrides
        self.kw = kw
        self.model = Net()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.args = SimpleNamespace(optimizer="SGD", lr0=0.01, momentum=0.9)
        self.n_optimizer_updates = 0
        self.n_optimizer_updates_applied = 0
        self.budget = None
        self.stopper = SimpleNamespace(calls=[], true_count=None)
        self.injection_digest = []
        self.loader_reseed = None
        self.val_loader_workers = 0
        self._canonical_keys = list(kw.get("canonical_keys") or []) or None
        self.callbacks: list[tuple[str, object]] = []

    def add_callback(self, event, fn) -> None:
        self.callbacks.append((event, fn))

    def train(self) -> None:
        pass

    def _unwrapped_model(self):
        return self.model

    def export_weights(self):
        from detection import serialize

        return serialize.state_dict_to_ndarrays(self.model.state_dict(), self._canonical_keys)

    def effective_optimizer(self) -> dict:
        return {"optimizer": "SGD", "lr": 0.01, "momentum": 0.9, "arg_optimizer": "SGD",
                "arg_lr0": 0.01, "arg_momentum": 0.9}
