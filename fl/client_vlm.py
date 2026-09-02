"""통합형(VLM) 클라이언트 — `train_rounds` 를 부르는 얇은 배선.

⑦ 통합·연합은 파일럿에서 **인프로세스 순차 루프**로 돌았다(`scripts/pilot_c.cmd_cell7`).
전송 계층을 건너뛰었으므로 전략의 실패 검사(에러 응답·응답 수 대조·키 다이제스트)가
한 번도 발화하지 않았고, 회계 실패 시 예외를 올리지도 `audit.json` 을 쓰지도 않았다
(80번 F6). 80번 체크리스트 15항이 그 우회를 닫는다 — 이제 두 연합 칸이 같은
`WeldFedAvg` 경로를 탄다.

**검출과 같은 모양으로 짰다.** `run_client_round` 가 Flower 자료형과 무관한 순수 함수다.
Flower 핸들러는 이 파일에 없다 — 등록 핸들러는 `fl/client_app.py` 하나이고 거기서 칸을
분기한다(85번 ⑥: 이 모듈의 자체 앱은 어디에도 등록돼 있지 않아 죽은 진입점이었다).

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

from detection import serialize
from fl.strategy import WEIGHT_KEY

__all__ = ["adapter_exchange_contract", "run_client_round", "payload_metrics"]


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

    metrics, strings = payload_metrics(m, n_pairs=len(rows),
                                       canonical_keys=canonical_keys,
                                       client_idx=client_idx)
    return arrays, metrics, strings


def payload_metrics(m: dict[str, Any], *, n_pairs: int, canonical_keys: list[str],
                    client_idx: int) -> tuple[dict[str, Any], dict[str, str]]:
    """전송 페이로드의 (수치, 문자열) 두 dict. **순수 함수라 시험이 실구성을 검사한다.**

    85번 ① 이 잡은 사고가 정확히 여기서 났다: `WEIGHT_KEY` 가 "num-examples" 이던 시절
    dict 리터럴이 같은 키를 두 번 써서 뒤(페어 수)가 이겼고, 실제 전송 가중이 페어 수인데
    `weight-unit` 은 supervised_tokens 로 남아 **회계가 단위를 거짓말했다.** 그때의 판정 2
    고정 시험은 소스 문자열 포함 검사라 깨진 코드에서 통과했다.

    그래서 지금은 셋으로 막는다.
    1. 가중 키가 의미 키와 분리됐다(`fl/strategy.WEIGHT_KEY == "fedavg-weight"`).
    2. 구성 직후 **단위 일치를 단언**한다 — 전송될 가중이 감독 토큰 총합과 다르면
       여기서 죽는다(아래). dict 가 어떤 경로로 만들어졌든 마지막에 잡힌다.
    3. 시험이 문자열이 아니라 이 함수를 **실행해** 반환 dict 를 검사한다.
    """
    tokens = float(m["supervised_tokens"])
    metrics: dict[str, Any] = {
        # 판정 2 — 가중은 감독 토큰 총합. 페어 수는 의미 키로 따로 싣는다.
        WEIGHT_KEY: tokens,
        "num-examples": float(n_pairs),
        "supervised-tokens": tokens,
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
    # **단위 일치 단언.** 키 충돌·키 개명·구성 순서 어느 사고로든 전송 가중이 감독 토큰
    # 총합과 어긋나면 라운드가 여기서 죽는다 — 회계가 거짓 단위를 싣는 것보다 낫다.
    if metrics[WEIGHT_KEY] != tokens or metrics["num-examples"] != float(n_pairs):
        raise AssertionError(
            f"판정 2 위반: 전송 가중 {metrics[WEIGHT_KEY]} != 감독 토큰 {tokens} "
            f"또는 페어 수 {metrics['num-examples']} != {n_pairs} — 키 충돌이 재발했다"
        )
    strings: dict[str, str] = {
        "keys-digest": serialize.keys_digest(canonical_keys),
        "optimizer": str(m["optimizer"]),
        "arg-optimizer": "AdamW",
        "stopper-class": "",
        "weight-unit": "supervised_tokens",
    }
    return metrics, strings
