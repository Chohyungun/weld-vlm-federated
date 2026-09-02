"""검출 클라이언트 — `train_round` 를 부르는 순수 함수.

**이 파일에는 Flower 핸들러가 없다.** 이전의 `@app.train()` 은 죽어 있었다 —
구판 키 이름(round)을 읽는데 서버는 SERVER_ROUND_KEY 값을 보내고, 서버가 보내지 않는 키
(`data-yaml`·`num-examples`)를 읽고, 실제로 오는 키는 무시했다. pyproject 가 등록한
유일한 앱이 그 상태였다(85번 ⑤). 핸들러는 `fl/client_app.py` 하나로 모았고, 이 파일은
자료형과 무관한 학습 경로만 판다 — 시험이 프레임워크 없이 도는 이유이기도 하다.

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

from detection import serialize
from detection.round_runner import train_round
from fl.strategy import WEIGHT_KEY

__all__ = ["run_client_round"]


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
        # 가중 키와 의미 키는 **다른 키**다(85번 ①). 검출은 가중 == 표본 수라 값이 같지만
        # 키를 겹쳐 쓰면 통합형에서처럼 dict 리터럴 충돌로 가중이 조용히 바뀐다.
        WEIGHT_KEY: float(result.num_examples),
        "num-examples": float(result.num_examples),
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
        # 조기 종료 계측. 서버 회계가 이 값을 읽는다 — 리터럴 0 을 박아 두면 검사가
        # 공허해진다(74번 감사 P9). -1 은 "계측 없음"이며 0 과 다르다.
        "stopper-true-count": float(
            result.stopper_true_count if result.stopper_true_count is not None else -1
        ),
        "stopper-calls": float(len(result.stopper_calls)),
    }
    strings: dict[str, str] = {
        "keys-digest": serialize.keys_digest(canonical_keys),
        "optimizer": str(eff.get("optimizer", "")),
        "arg-optimizer": str(eff.get("arg_optimizer", "")),
        # 스텁 교체가 실패하면 여기가 달라지고 서버 회계가 실패한다.
        "stopper-class": str(result.stopper_class),
        # 가중 단위를 클라이언트가 스스로 밝힌다 — 회계의 단위 일치 감사가 이걸 대조한다.
        "weight-unit": "num_examples",
    }
    return result.ndarrays, metrics, strings
