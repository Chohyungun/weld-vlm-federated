"""학습 경로 공통 시드 진입점 — 난수 초기화가 갈라지는 것을 한 곳에서 막는다.

## 왜 모듈 하나로 모으는가

⑦ 통합·연합 r0 에서 세 클라이언트가 **각자 난수 LoRA A** 로 출발했다. 가중 평균이
"같은 기저의 평균"이 아니라 "독립 난수의 상쇄"가 되어, r0 로컬 학습 144스텝(전체 432의
33%)이 사실상 폐기됐다. 지정 함정 구간 #3 그 자체다.

실측이 그것을 그대로 보여 준다.

    r0 클라이언트 param_l2   31.7405 / 31.6717 / 31.6238
    r0 서버 global_l2        20.5755
    독립 난수 가정 예측       31.68 x sqrt(sum w^2) = 31.68 x 0.64852 = 20.55   <- 일치
    공유 초기값 가정 예측      ~= 31.7                                          <- 불일치

원인은 산술이 아니라 **배치**였다. 시드 고정이 검출(`detection/init_weights.py`)에만
있었고 VLM 에는 없었으며, 프로브는 또 각자 시드를 다뤘다. 고정 지점이 흩어져 있으면
빠진 곳이 눈에 띄지 않는다. 그래서 지점을 하나로 모은다 — 새 학습 경로를 붙이는 사람이
이 모듈을 지나가지 않으면 어색하도록.

## 두 종류의 시드를 구분한다

- **공유 시드**(`shared_init_seed`): 모든 클라이언트·모든 칸이 **같은 값**을 써야 하는
  자리. 초기 가중치·초기 어댑터가 여기다. 갈라지면 "같은 출발점" 주장이 깨진다.
- **파생 시드**(`detection.round_runner.derive_seed`): 라운드·클라이언트마다 **달라야**
  하는 자리. 데이터 셔플이 여기다. 같아지면 50라운드가 같은 순서를 반복한다.

둘을 한 이름으로 부르면 반드시 한쪽이 틀린다. 호출부에서 어느 쪽인지 보이게 둔다.

## 전역 상태를 되돌린다

`seeded()` 는 컨텍스트 매니저다. 블록 안에서만 전역 난수를 고정하고 나갈 때 원래 상태로
되돌린다. 초기화 한 번 때문에 그 뒤 학습 전체의 난수 흐름이 호출 순서에 묶이면, 이번과
반대 방향의 재현성 사고가 난다.
"""

from __future__ import annotations

import contextlib
import random
from typing import Iterator

__all__ = ["seed_all", "seeded", "shared_init_seed", "deterministic_torch"]


def shared_init_seed(base_seed: int) -> int:
    """모든 칸·모든 클라이언트가 공유해야 하는 초기화 시드.

    `base_seed` 를 그대로 쓴다. 검출 칸의 `build_initial_weights(seed=BASE_SEED)` 와
    같은 값이 되도록 일부러 항등으로 뒀다 — 두 칸의 "동일 출발"이 같은 상수에서 나오는
    편이 사후 대조가 쉽다. 라운드·클라이언트 인자를 **받지 않는 것이 이 함수의 요점**이다.
    """
    return int(base_seed)


def seed_all(seed: int, *, deterministic: bool = False) -> None:
    """python·numpy·torch 전역 난수를 한 번에 고정한다.

    torch·numpy 가 없는 환경에서도 죽지 않는다 — 좌표·직렬화 시험처럼 가벼운 경로가
    이 모듈을 스치더라도 무거운 의존을 강제하지 않기 위해서다.
    """
    seed = int(seed)
    random.seed(seed)
    # `PYTHONHASHSEED` 는 **여기서 건드리지 않는다.** 인터프리터 시작 전에만 효력이 있어
    # 현재 프로세스에는 아무 영향이 없는데, `seeded()` 안에서 설정하면 그 뒤 spawn 되는
    # 로더 워커가 초기화용 임시 시드를 물려받는다. 이득 없이 프로세스 간 결합만 생긴다.
    # 필요하면 런처(pane 기동 명령)에서 준다.
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return
    if deterministic:
        deterministic_torch()


def deterministic_torch() -> None:
    """cudnn 결정성 고정 (개발규약 스택 절). 성능 대가가 있으므로 명시 호출로 둔다."""
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def seeded(seed: int) -> Iterator[None]:
    """블록 안에서만 전역 난수를 고정하고, 빠져나올 때 원래 상태로 되돌린다.

    `get_peft_model` 처럼 **호출 한 번의 난수 소비**를 고정하고 싶을 때 쓴다. 전역
    `seed_all` 로 대신하면 그 뒤의 모든 난수가 호출 위치에 묶여, 라운드마다 다른 셔플을
    쓰는 설계와 충돌한다.
    """
    py_state = random.getstate()
    np_state = None
    torch_state = None
    cuda_state = None
    try:
        import numpy as np

        np_state = np.random.get_state()
    except ImportError:
        np = None  # noqa: F841
    try:
        import torch

        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
    except ImportError:
        torch = None  # noqa: F841

    try:
        seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        if np_state is not None:
            import numpy as np

            np.random.set_state(np_state)
        if torch_state is not None:
            import torch

            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
