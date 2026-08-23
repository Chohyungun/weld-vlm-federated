"""통합형(VLM) 클라이언트 — 골격.

RQ 우선순위 변경으로 분리형 세 칸이 논문 본체가 됐고 통합형 학습 본체는 그 완주 뒤로
미뤄졌다. 이 파일은 **교환 단위가 검출과 다르다는 사실이 인터페이스에 드러나게** 자리만
잡아 둔다.

## 검출과 무엇이 다른가

검출은 `state_dict` **전체**를 교환하므로 서버가 받은 것을 전부 평균하면 된다. 통합형은
**LoRA 어댑터만** 교환한다 — 즉 교환 단위가 `state_dict` 의 진부분집합이다. 그래서 위험이
정반대다.

- 검출: "평균하면 안 될 통계(BN running stats)를 조용히 평균했다"
- 통합형: **"클라이언트에 남은 상태를 아무도 집계하지 않았고 지표에도 흔적이 없다"**

후자는 diff 로 잡히지 않고 **집합 등식**으로만 잡힌다. 30번 명세의 G2(교환 폐포 감사)가
`{n for n, p in model.named_parameters() if p.requires_grad}` 와 어댑터 페이로드 키 집합이
**완전 일치**(부분집합이 아니다)함을 요구하는 이유다.

**G2 통과 전에는 본체를 채우지 않는다.** 학습되는데 교환되지 않는 파라미터가 하나라도 있으면
6라운드 뒤 "글로벌 모델"이 어느 클라이언트의 모델과도 다른 물건이 된다.
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    from flwr.clientapp import ClientApp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "fl extra 가 설치되지 않았다. `uv sync --extra fl` 로 flwr 를 설치해야 한다."
    ) from exc

__all__ = ["app", "adapter_exchange_contract"]

app = ClientApp()


def adapter_exchange_contract(
    trainable_names: Sequence[str], payload_keys: Sequence[str]
) -> tuple[bool, list[str]]:
    """G2-3 의 집합 등식을 검사한다 — 학습되는 것과 교환되는 것이 정확히 같은가.

    부분집합으로는 부족하다. 학습되는데 교환되지 않는 파라미터가 있으면 그 갱신은
    클라이언트 로컬에만 남고, 교환되는데 학습되지 않는 것이 있으면 불필요한 통신량이다.

    Returns:
        (통과 여부, 위반 사유 목록)
    """
    trainable = set(trainable_names)
    payload = set(payload_keys)
    failures: list[str] = []

    only_trained = sorted(trainable - payload)
    if only_trained:
        failures.append(
            f"학습되는데 교환되지 않는 파라미터 {len(only_trained)}개: {only_trained[:5]} — "
            "이 갱신은 클라이언트에만 남고 글로벌 모델에 반영되지 않는다"
        )
    only_sent = sorted(payload - trainable)
    if only_sent:
        failures.append(
            f"교환되는데 학습되지 않는 파라미터 {len(only_sent)}개: {only_sent[:5]} — "
            "통신량만 늘리거나, 동결됐어야 할 것이 실려 있다"
        )
    return (not failures), failures


@app.train()
def train(msg: Any, context: Any) -> Any:  # pragma: no cover - 본체 미구현
    raise NotImplementedError(
        "통합형 연합 학습 본체는 분리형 세 칸 완주 후에 채운다(RQ 우선순위 변경). "
        "착수 전 30번 명세 G2(교환 폐포 감사) 통과가 선행 조건이다 — "
        "adapter_exchange_contract 로 G2-3 집합 등식을 확인한다."
    )
