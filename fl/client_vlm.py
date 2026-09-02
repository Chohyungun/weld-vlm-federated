"""통합형(VLM) 클라이언트 — `train_rounds` 를 부르는 얇은 배선.

⑦ 통합·연합은 파일럿에서 **인프로세스 순차 루프**로 돌았다(`scripts/pilot_c.cmd_cell7`).
전송 계층을 건너뛰었으므로 전략의 실패 검사(에러 응답·응답 수 대조·키 다이제스트)가
한 번도 발화하지 않았고, 회계 실패 시 예외를 올리지도 `audit.json` 을 쓰지도 않았다
(80번 F6). 80번 체크리스트 15항이 그 우회를 닫는다 — 이제 두 연합 칸이 같은
`WeldFedAvg` 경로를 탄다.

**검출과 같은 모양으로 짰다.** `run_client_round` 가 Flower 자료형과 무관한 순수 함수이고
`@app.train()` 은 자료형 변환만 한다. 두 칸의 클라이언트가 다른 모양이면 "구조 내 동일"
주장이 문면으로만 남는다.

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
`train_rounds` 가 라운드 1 에서 `adapter_exchange_contract` 로 그 등식을 검사한다.

## FedAvg 가중은 감독 토큰 총합이다 (총괄 판정 2, 2026-09-02)

페어 수가 아니다. 손실 정규화가 감독 토큰 총합 분모(`vlm/loss_norm.py`)이므로 **같은
단위로 가중해야** 연합 목적함수가 중앙(합동 학습)의 목적함수와 일치한다 — R×E=N 등가
주장의 실질이 그것이다. 페어 수로 가중하면 토큰 밀도 차(실측 c0 192.1 / c1 105.3 /
c2 100.8 tok/페어, c0 이 1.9배)가 목적함수를 비튼다.

회계에는 **둘 다** 남긴다. 가중이 바뀌면 C3 비중이 1.52배 움직이고 RQ3 해석이 그
위에 서므로, 어느 단위로 잰 값인지 산출물이 말해야 한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    from flwr.clientapp import ClientApp
    from flwr.common import (ArrayRecord, ConfigRecord, Context, Message,
                             MetricRecord, RecordDict)
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "fl extra 가 설치되지 않았다. `uv sync --extra fl` 로 flwr 를 설치해야 한다."
    ) from exc

from detection import serialize
from fl.round_wiring import CANONICAL_KEYS_KEY, SERVER_ROUND_KEY
from fl.strategy import ARRAYS_KEY, CONFIG_KEY, METRICS_KEY, WEIGHT_KEY

__all__ = ["app", "adapter_exchange_contract", "run_client_round"]

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


def run_client_round(
    *,
    adapter_in: list,
    canonical_keys: list[str],
    round_idx: int,
    client_idx: int,
    cfg: dict[str, Any],
) -> tuple[list, dict[str, Any], dict[str, str]]:
    """어댑터 학습 1라운드. Flower 자료형과 무관해 프레임워크 없이 시험이 돈다.

    Returns:
        (어댑터 ndarray 리스트, 수치 메트릭, 문자열 필드). 검출 클라이언트와 같은
        3튜플이다 — `MetricRecord` 가 문자열을 거부하므로 나눠 돌려준다.
    """
    from vlm.init_adapter import assert_injected_matches
    from vlm.pilot_vlm import load_pairs, train_rounds

    rows = load_pairs("train", client=str(cfg["client_tag"]))
    arrays, keys, m, _ref = train_rounds(
        rows=rows,
        epochs=int(cfg["local_epochs"]),
        round_idx=round_idx,
        client_idx=client_idx,
        base_seed=int(cfg["base_seed"]),
        adapter_in=adapter_in,
        adapter_keys=canonical_keys,
        resume_dir=str(Path(cfg["resume_root"]) / f"r{round_idx:03d}_c{client_idx}")
        if cfg.get("resume_root") else None,
        run_id=str(cfg.get("run_id", "")),
        num_rounds=int(cfg["num_rounds"]),
    )

    # G2-5 — 주입이 **서버가 보낸 것과** 같은지 대조한다. 클라이언트끼리 비교하면
    # 셋 모두 no-op 일 때 통과한다.
    if adapter_in is not None and m.get("injected_proof") is not None:
        assert_injected_matches(list(adapter_in), list(canonical_keys),
                                m["injected_proof"], who=f"c{client_idx} r{round_idx}")

    # 판정 2 — 가중은 감독 토큰 총합. 페어 수는 회계용으로 함께 싣는다.
    metrics: dict[str, Any] = {
        WEIGHT_KEY: float(m["supervised_tokens"]),
        "num-examples": float(len(rows)),
        "supervised-tokens": float(m["supervised_tokens"]),
        "epochs-ran": float(m["epochs_ran"]),
        "optimizer-steps": float(m["optimizer_steps"]),
        "optimizer-updates": float(m["optimizer_steps"]),   # micro=1 — 배치 수 = 갱신 수
        "resumed-from-epoch": float(
            m["resumed_from_epoch"] if m.get("resumed_from_epoch") is not None else -1),
        "param-l2": float(m["param_l2"]),
        "payload-bytes": float(m["payload_bytes"]),
        "seed": float(m["seed"]),
        "client-idx": float(client_idx),
        "lr": float(m["lr"]),
        "momentum": float("nan"),
        "budget-fired-at": -1.0,
        "peak-vram-gb": float(m["peak_vram_gb"]),
        # 통합형에는 Ultralytics stopper 가 없다. -1 은 "계측 없음"이며 0 과 다르다.
        "stopper-true-count": -1.0,
        "init-l2": float(m["init_proof"]["l2"]),
    }
    strings: dict[str, str] = {
        "keys-digest": serialize.keys_digest(canonical_keys),
        "optimizer": str(m["optimizer"]),
        "arg-optimizer": "AdamW",
        "stopper-class": "",
        "weight-unit": "supervised_tokens",
    }
    return arrays, metrics, strings


@app.train()
def train(msg: "Message", context: "Context") -> "Message":
    """서버가 보낸 어댑터로 로컬 학습을 돌리고 어댑터를 돌려준다."""
    in_cfg = msg.content[CONFIG_KEY] if CONFIG_KEY in msg.content else {}
    if CANONICAL_KEYS_KEY not in in_cfg:
        raise RuntimeError(
            "서버가 정본 키 리스트를 보내지 않았다. ArrayRecord 는 리스트 경로에서 키를 "
            "인덱스 문자열로 바꾸므로 이름은 별도로 전달돼야 한다."
        )
    node_cfg = dict(context.node_config or {})
    if "partition-id" not in node_cfg:
        raise RuntimeError(
            "시뮬레이션 백엔드가 partition-id 를 넣지 않았다. 클라이언트를 식별할 수 없으므로 "
            "추측하지 않고 멈춘다 — 세 클라이언트가 같은 데이터를 돌면 지표는 초록인 채 "
            "연합이 무의미해진다."
        )
    client_idx = int(node_cfg["partition-id"])
    round_idx = int(in_cfg[SERVER_ROUND_KEY]) - 1     # flwr 는 1부터 센다

    arrays_out, metrics, strings = run_client_round(
        adapter_in=msg.content[ARRAYS_KEY].to_numpy_ndarrays(),
        canonical_keys=list(in_cfg[CANONICAL_KEYS_KEY]),
        round_idx=round_idx,
        client_idx=client_idx,
        cfg={
            "client_tag": str(in_cfg[f"client-tag-{client_idx}"]),
            "local_epochs": int(in_cfg["local-epochs"]),
            "num_rounds": int(in_cfg["num-rounds"]),
            "base_seed": int(in_cfg["base-seed"]),
            "resume_root": str(in_cfg["resume-root"]) if in_cfg.get("resume-root") else None,
            "run_id": str(in_cfg.get("run-stamp", "")),
        },
    )
    return Message(
        content=RecordDict({
            ARRAYS_KEY: ArrayRecord(arrays_out),
            METRICS_KEY: MetricRecord(metrics),
            CONFIG_KEY: ConfigRecord(strings),
        }),
        reply_to=msg,
    )
