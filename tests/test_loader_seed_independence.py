"""Ultralytics 학습 로더의 셔플 순열이 `args.seed` 와 무관함을 실측으로 못 박는다.

## 무엇이 걸려 있는가

`ultralytics/data/build.py::build_dataloader` 는 로더 생성기를 **상수**로 시드한다.

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)

`shuffle=True`·`sampler=None` 이면 PyTorch 가 이 생성기로 `RandomSampler` 를 돌리므로,
**epoch 별 셔플 순열도 워커 시드도 우리가 준 `seed` 를 타지 않는다.** 워커 시드는
`seed_worker` 가 `torch.initial_seed()` 에서 끌어오는데 그 값 역시 이 생성기에서 나온다.

그러면 이렇게 된다.

- `derive_seed(base_seed, round_idx, client_idx)` 가 라운드·클라이언트마다 다른 값을 내도
  **데이터 순서는 같다.** 시드 파생의 원래 목적(라운드마다 같은 순서를 반복하지 않게)이
  검출 쪽에서는 달성되지 않는다.
- **시드 3세트의 독립성이 검출에서는 약해진다.** 세 런이 같은 순서·같은 증강 난수열을
  본다. 남는 차이는 헤드 초기화(`build_initial_weights(seed)`)와 비결정 연산뿐이다.

선언(`seed`·`deterministic=True`)과 프레임워크의 실제 동작이 어긋나는 형태이고,
그 어긋남이 지표에 아무 흔적을 남기지 않는다 — 숨은 기본값 레지스트리의 전형이다.

이 시험은 **현상을 고정한다.** 상류가 동작을 바꾸면 여기서 깨지고, 그때 레지스트리와
시드 정책을 다시 본다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("torch")

import torch


class _Toy(torch.utils.data.Dataset):
    """인덱스를 그대로 돌려주는 최소 데이터셋. 순열만 관찰하면 된다."""

    def __init__(self, n: int = 64) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


def _order(seed: int, epochs: int = 2) -> list[list[int]]:
    from ultralytics.data.build import build_dataloader
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(seed, deterministic=True)
    loader = build_dataloader(_Toy(), batch=8, workers=0, shuffle=True, device="cpu")
    return [[int(v) for b in loader for v in b] for _ in range(epochs)]


def test_shuffle_order_does_not_depend_on_our_seed():
    a = _order(0)
    b = _order(20115)          # derive_seed(0, 2, 1) 급으로 멀리 떨어진 값
    assert a == b, (
        "로더 셔플이 시드를 타기 시작했다. 숨은 기본값 레지스트리 #9 의 전제가 바뀌었으니 "
        "시드 정책(시드 3세트의 독립성)을 다시 판단해야 한다."
    )


def test_epochs_differ_from_each_other():
    """epoch 간에는 순열이 달라진다 — '순서가 아예 고정'인 것과는 다른 이야기다."""
    a = _order(0, epochs=2)
    assert a[0] != a[1]


def test_loader_generator_is_seeded_with_the_hardcoded_constant():
    """상수 자체를 못 박는다. 상류가 값을 바꾸면 재개 경로의 전제도 함께 바뀐다.

    순열을 직접 재현하지는 않는다 — PyTorch 의 `DataLoader` 반복자가 순열보다 **먼저**
    워커용 base seed 를 한 번 뽑기 때문에, 순열 재현은 torch 내부 순서에 묶인다.
    `initial_seed()` 는 상태가 얼마나 진행됐든 시드값을 그대로 돌려주므로 버전에 둔감하다.

    두 번째 단언이 재개 경로의 전제다: 새로 뜬 프로세스의 로더는 **같은 상태에서 시작**한다.
    그래서 재개할 때 생성기 상태를 되돌리지 않으면 epoch 0 의 순열을 다시 보게 된다.
    """
    from ultralytics.data.build import build_dataloader
    from ultralytics.utils import RANK

    # 비분산 실행에서 `RANK` 는 -1 이다. 그래서 실제 시드는 상수보다 1 작다 —
    # 상수를 그대로 기대하면 여기서 어긋난다.
    a = build_dataloader(_Toy(64), batch=8, workers=0, shuffle=True, device="cpu")
    assert a.generator.initial_seed() == 6148914691236517205 + RANK

    b = build_dataloader(_Toy(64), batch=8, workers=0, shuffle=True, device="cpu")
    assert torch.equal(a.generator.get_state(), b.generator.get_state())
