"""연합 전략 — 실패를 직접 검사하는 FedAvg 서브클래스.

## 왜 상위 구현을 그대로 쓰지 않는가

`flwr 1.33.0`의 Message API `FedAvg`에는 **`accept_failures` 파라미터가 없다**(레거시에는
있다). `aggregate_train`은 `_check_and_log_replies`로 에러 응답을 갈라 **로그만 남기고
버린 뒤** 살아남은 것으로 집계하고, `aggregate_arrayrecords`가 `total_weight`로 가중치를
**다시 1로 정규화**한다.

그래서 C3가 라운드 30에서 죽어도 C1:C2 가중치가 32:16에서 2:1로 맞춰져 집계가 성립하고,
라운드는 정상 종료하며, 지표는 초록이다. **`R × E = N`만 조용히 깨진다.** 게이트 #6 결정 A가
막으려던 경로이고, 파라미터로는 막을 수단이 없어 전략이 직접 검사한다.

집계 자체도 상위 구현에 맡기지 않는다. `aggregate_arrayrecords`는 BatchNorm 버퍼 정책을
모르고 전부 가중 평균하는데, `num_batches_tracked`는 정수 카운터라 평균하면 반올림이
끼어들어 어느 클라이언트의 값도 아니게 된다. 집계는 `fl.aggregate.weighted_fedavg`가 한다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from flwr.common import ArrayRecord, MetricRecord
    from flwr.serverapp.strategy import FedAvg
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "fl extra 가 설치되지 않았다. `uv sync --extra fl` 로 flwr 를 설치해야 한다. "
        "집계 산술만 쓰려면 `fl.aggregate` 를 직접 import 하면 된다 — 그쪽은 flwr 에 의존하지 않는다."
    ) from exc

from detection import serialize
from fl.aggregate import weighted_fedavg

__all__ = ["RoundFailure", "WeldFedAvg", "ARRAYS_KEY", "METRICS_KEY", "WEIGHT_KEY"]

#: Message FedAvg 의 기본 키 이름(실측). 클라이언트와 서버가 같은 이름을 써야 한다.
ARRAYS_KEY = "arrays"
METRICS_KEY = "metrics"
CONFIG_KEY = "config"
#: `weighted_by_key` 기본값. 클라이언트가 이 이름으로 표본 수를 올린다.
WEIGHT_KEY = "num-examples"

#: `MetricRecord` 는 `int | float | list` 만 받는다(실측 — str·bool 거부).
#: 그래서 문자열 필드(정본 키 다이제스트, 실사용 optimizer 이름)는 `ConfigRecord` 로 나른다.
#: 이 분리는 취향이 아니라 자료형 제약이며, 한쪽에 몰아넣으면 라운드 1에서 TypeError 로 죽는다.
STR_FIELDS = ("keys-digest", "optimizer", "arg-optimizer")


class RoundFailure(RuntimeError):
    """라운드가 성립하지 않았다. 재정규화로 넘어가지 않고 즉시 중단한다."""


class WeldFedAvg(FedAvg):
    """실패를 라운드 중단으로 만들고, 집계를 우리 규약으로 수행하는 전략.

    Args:
        expected_nodes: 매 라운드 참여해야 하는 노드 수. 우리 설계는 3클라이언트 전수 참여가
            전제다. 상위 기본값(`min_train_nodes=2`)을 그대로 두면 한 클라이언트가 빠져도
            라운드가 성립하므로 생성자에서 셋을 모두 이 값으로 고정한다.
        canonical_keys: 정본 키 리스트(순서 포함). `ArrayRecord` 는 리스트 경로에서 키를
            인덱스 문자열로 바꾸므로 이름은 이쪽이 관리한다.
        reference_state_dict: dtype·shape 검사 기준.
        on_round_end: 라운드별 회계 적재 콜백. `(server_round, cells, agg)` 를 받는다.
    """

    def __init__(
        self,
        *,
        expected_nodes: int,
        canonical_keys: Sequence[str],
        reference_state_dict: Mapping[str, Any],
        on_round_end: Any = None,
        **kwargs: Any,
    ) -> None:
        # 평가 라운드를 쓰지 않는다. 라운드별 곡선은 서버가 val 로 직접 산출하며
        # (게이트 #6 결정 B), 클라이언트 평가 경로를 열어 두면 평가셋 접근 경로가 하나 더 생긴다.
        kwargs.setdefault("fraction_evaluate", 0.0)
        kwargs["min_evaluate_nodes"] = 0
        kwargs["fraction_train"] = 1.0
        kwargs["min_train_nodes"] = int(expected_nodes)
        kwargs["min_available_nodes"] = int(expected_nodes)
        kwargs.setdefault("weighted_by_key", WEIGHT_KEY)
        super().__init__(**kwargs)

        if expected_nodes <= 0:
            raise ValueError(f"expected_nodes 는 양수여야 한다: {expected_nodes}")
        self.expected_nodes = int(expected_nodes)
        self.canonical_keys = list(canonical_keys)
        self.reference_state_dict = reference_state_dict
        self.on_round_end = on_round_end
        self.keys_digest = serialize.keys_digest(self.canonical_keys)

    # -- 학습 라운드 --------------------------------------------------------
    def aggregate_train(
        self, server_round: int, replies: Iterable[Any]
    ) -> tuple[Any | None, Any | None]:
        """실패를 먼저 검사하고, 통과한 경우에만 우리 규약으로 집계한다."""
        replies = list(replies)

        # ① 에러 응답이 하나라도 있으면 라운드가 성립하지 않는다.
        errored = [m for m in replies if m.has_error()]
        if errored:
            reasons = "; ".join(
                f"node={m.metadata.src_node_id} reason={m.error.reason}" for m in errored
            )
            raise RoundFailure(
                f"라운드 {server_round}: 클라이언트 {len(errored)}개가 실패했다 ({reasons}). "
                "남은 응답으로 집계하면 가중치가 재정규화되어 R×E=N 이 흔적 없이 깨진다."
            )

        # ② 응답이 아예 오지 않은 노드는 에러 메시지조차 없다. 개수가 유일한 탐지 수단이다.
        valid = [m for m in replies if m.has_content()]
        if len(valid) != self.expected_nodes:
            raise RoundFailure(
                f"라운드 {server_round}: 유효 응답 {len(valid)}개 != 기대 {self.expected_nodes}개. "
                "응답하지 않은 노드는 에러 응답도 남기지 않으므로 개수로만 잡힌다."
            )

        # ③ 교환 규약 검사 — 키·shape·dtype 이 어긋난 채로 평균하면 조용히 오염된다.
        client_arrays: list[list[np.ndarray]] = []
        num_examples: list[int] = []
        cells: list[dict[str, Any]] = []
        for msg in valid:
            content = msg.content
            arrays = content[ARRAYS_KEY].to_numpy_ndarrays()
            metrics = content[METRICS_KEY]
            # 문자열은 ConfigRecord 에 실려 온다(MetricRecord 가 str 을 거부한다).
            strings = dict(content[CONFIG_KEY]) if CONFIG_KEY in content else {}
            serialize.assert_compatible(arrays, self.canonical_keys, self.reference_state_dict)

            got_digest = str(strings.get("keys-digest", self.keys_digest))
            if got_digest != self.keys_digest:
                raise RoundFailure(
                    f"라운드 {server_round}: 정본 키 다이제스트 불일치 "
                    f"(수신 {got_digest[:12]}… != 기대 {self.keys_digest[:12]}…). "
                    "클라이언트가 다른 키 순서로 직렬화했다."
                )

            client_arrays.append(arrays)
            num_examples.append(int(metrics[WEIGHT_KEY]))
            cells.append({"node_id": msg.metadata.src_node_id, **dict(metrics), **strings})

        # ④ 집계는 우리 규약으로. 상위 구현은 num_batches_tracked 까지 평균한다.
        agg = weighted_fedavg(
            client_arrays, num_examples, self.canonical_keys, self.reference_state_dict
        )

        # ⑤ 회계 적재는 서버가 맡는다. 전략은 값을 넘기기만 한다.
        if self.on_round_end is not None:
            self.on_round_end(server_round, cells, agg)

        out_metrics = MetricRecord(
            {
                "total-examples": float(agg.total_examples),
                "global-l2": float(agg.global_norm),
                "bn-divergence": float(agg.bn_buffer_divergence),
                "missing-variance-ratio": float(agg.missing_variance_ratio),
                "clients": float(len(valid)),
            }
        )
        return ArrayRecord(agg.ndarrays), out_metrics

    # -- 평가 라운드 (열리지 않지만 대칭으로 막는다) ------------------------
    def configure_evaluate(self, *args: Any, **kwargs: Any) -> list:
        """평가 라운드를 만들지 않는다."""
        return []

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Any]) -> Any | None:
        """열리지 않는 경로라도 열렸다면 조용히 통과시키지 않는다.

        반환은 상위 규약대로 **MetricRecord 하나(또는 None)** 다. aggregate_train 처럼
        튜플을 돌려주면 Strategy.start 가 그 튜플을 라운드 결과 dict 에 그대로 넣고,
        실행 끝의 결과 요약 출력에서 `.items()` 를 불러 죽는다(파일럿 실측 — 학습은
        완주됐는데 요약에서 크래시해 뒤의 회계 마감이 날아갔다).
        """
        replies = list(replies)
        if replies:
            raise RoundFailure(
                f"라운드 {server_round}: 평가 라운드는 쓰지 않는데 응답 {len(replies)}개가 왔다. "
                "클라이언트 평가 경로가 열려 있다면 평가셋 접근 경로가 하나 더 생긴 것이다."
            )
        return None
