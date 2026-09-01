"""검출 클라이언트 — `train_round` 를 부르는 얇은 배선.

이 파일은 얇아야 한다. 학습 자체는 `detection/round_runner.train_round` 가 하고, 그 함수는
로컬·중앙·연합 세 칸이 **모두 통과하는 유일한 진입점**이다. 클라이언트가 자기 학습 경로를
만들면 연합 칸만 다른 코드를 타게 되어 "구조 내 동일" 원칙이 문면으로만 남는다.

클라이언트는 무상태다. 라운드마다 트레이너를 새로 만들고 상태를 액터 수명에 걸지 않는다.
Ray 액터가 warm 재사용될 수 있으므로, 상태를 들고 있는 설계는 시뮬레이션에서만 우연히 동작한다.

예외를 삼키지 않는다. 학습이 실패하면 그대로 올려 Flower 가 에러 응답을 만들게 하고,
전략이 그 라운드를 중단시킨다. 여기서 잡아 빈 가중치를 돌려주면 서버가 재정규화로
집계를 성립시켜 버린다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from flwr.clientapp import ClientApp
    from flwr.common import (ArrayRecord, ConfigRecord, Context, Message,
                             MetricRecord, RecordDict)
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "fl extra 가 설치되지 않았다. `uv sync --extra fl` 로 flwr 를 설치해야 한다."
    ) from exc

from detection import serialize
from detection.round_runner import train_round
from fl.strategy import ARRAYS_KEY, CONFIG_KEY, METRICS_KEY, WEIGHT_KEY

__all__ = ["app", "run_client_round"]

app = ClientApp()


def run_client_round(
    *,
    weights_in: list,
    canonical_keys: list[str],
    round_idx: int,
    client_idx: int,
    cfg: dict[str, Any],
    profile: str = "main",
) -> tuple[list, dict[str, Any], dict[str, str]]:
    """학습 1라운드. Flower 자료형과 무관한 순수 경로라 테스트가 프레임워크 없이 돈다.

    Returns:
        (가중치 ndarray 리스트, 수치 메트릭, 문자열 필드)

        수치와 문자열을 나눠 돌려주는 이유는 `MetricRecord` 가 `int | float | list` 만
        받기 때문이다(실측). 문자열은 `ConfigRecord` 로 나른다.
    """
    result = train_round(
        data_yaml=cfg["data_yaml"],
        model=cfg["model"],
        total_epochs=int(cfg["total_epochs"]),
        local_epochs=int(cfg["local_epochs"]),
        round_idx=round_idx,
        client_idx=client_idx,
        base_seed=int(cfg["base_seed"]),
        num_examples=int(cfg["num_examples"]),
        weights_in=weights_in,
        canonical_keys=canonical_keys,
        project=Path(cfg["project"]).resolve(),
        profile=profile,
        # 재개 전용 체크포인트. 라운드 안에서 죽으면 그 라운드를 0부터 다시 도는 대신
        # epoch 경계에서 이어 간다. 신원(라운드·클라이언트·시드)이 다르면 거부되므로
        # 옆 라운드의 상태를 잘못 물려받는 경로는 없다. 채점 대상이 아니다.
        resume_dir=(Path(cfg["resume_root"]).resolve() / f"r{round_idx:03d}_c{client_idx}"
                    if cfg.get("resume_root") else None),
        run_id=str(cfg.get("run_id", "")),
    )
    eff = result.effective_optimizer
    metrics: dict[str, Any] = {
        # 상위 전략의 weighted_by_key 기본값과 같은 이름이어야 가중 평균이 성립한다
        WEIGHT_KEY: float(result.num_examples),
        "epochs-ran": float(result.epochs_ran),
        "optimizer-steps": float(result.optimizer_steps),
        # 배치 수와 실제 갱신 횟수는 다르다(숨은 기본값 #10). 논문의 "총 갱신 횟수"는 아래다.
        "optimizer-updates": float(getattr(result, "optimizer_updates", 0) or 0),
        # 재개해서 이어 간 라운드인가. -1 은 재개 아님. 이어 간 런은 궤적이 다르다.
        "resumed-from-epoch": float(
            result.resumed_from_epoch if getattr(result, "resumed_from_epoch", None) is not None else -1
        ),
        "param-l2": float(result.param_l2_norm),
        "payload-bytes": float(result.payload_bytes),
        "seed": float(result.seed),
        "client-idx": float(client_idx),
        # 설정에 무엇을 적었는지가 아니라 무엇이 실제로 돌았는지를 올린다.
        # optimizer='auto' 는 명시한 lr0·momentum 을 버리고 AdamW 로 갈아치운다.
        "lr": float(eff.get("lr", float("nan"))),
        "momentum": float(eff.get("momentum", float("nan"))),
        "budget-fired-at": float(result.budget_fired_at if result.budget_fired_at is not None else -1),
        "peak-vram-gb": float(result.peak_vram_gb),
    }
    strings: dict[str, str] = {
        "keys-digest": serialize.keys_digest(canonical_keys),
        "optimizer": str(eff.get("optimizer", "")),
        "arg-optimizer": str(eff.get("arg_optimizer", "")),
    }
    return result.ndarrays, metrics, strings


@app.train()
def train(msg: "Message", context: "Context") -> "Message":
    """서버가 보낸 가중치로 로컬 학습을 돌리고 raw 가중치를 돌려준다."""
    run_cfg = context.run_config
    node_cfg = context.node_config

    # 서버가 보낸 설정은 ConfigRecord 에 있다. 정본 키 리스트가 여기 실리는 이유는
    # MetricRecord 가 문자열 리스트를 받지 않기 때문이다(실측).
    in_cfg = msg.content[CONFIG_KEY] if CONFIG_KEY in msg.content else {}
    if "canonical-keys" not in in_cfg:
        raise RuntimeError(
            "서버가 정본 키 리스트를 보내지 않았다. ArrayRecord 는 리스트 경로에서 키를 "
            "인덱스 문자열로 바꾸므로 이름은 별도로 전달돼야 한다."
        )
    canonical_keys = list(in_cfg["canonical-keys"])

    weights_in = msg.content[ARRAYS_KEY].to_numpy_ndarrays()
    round_idx = int(in_cfg["round"])
    client_idx = int(node_cfg.get("partition-id", 0))

    cfg = {
        "data_yaml": run_cfg["data-yaml"],
        "model": run_cfg["model"],
        "total_epochs": run_cfg["total-epochs"],
        "local_epochs": run_cfg["local-epochs"],
        "base_seed": run_cfg["base-seed"],
        "num_examples": node_cfg["num-examples"],
        "project": run_cfg["project"],
    }

    arrays_out, metrics, strings = run_client_round(
        weights_in=weights_in,
        canonical_keys=canonical_keys,
        round_idx=round_idx,
        client_idx=client_idx,
        cfg=cfg,
    )
    content = RecordDict(
        {
            ARRAYS_KEY: ArrayRecord(arrays_out),
            METRICS_KEY: MetricRecord(metrics),
            CONFIG_KEY: ConfigRecord(strings),
        }
    )
    return Message(content=content, reply_to=msg)
