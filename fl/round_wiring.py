"""연합 배선 공통부 — 두 진입점이 같은 코드를 타게 한다 (80번 G10-1·G10-2).

## 왜 이 모듈이 생겼나

배선이 두 벌이었다. `fl/server_app.py`(`flwr run` 용)와 `fl/pilot_sim.py`
(`run_simulation` 용)가 라운드 종료 기록·회계 마감을 **각자** 구현했고, 그 결과
한쪽에만 있는 버그가 생겼다.

- `server_app` 의 회계 마감이 `finally` 가 아니라 평문 호출이었다 — 학습 밖 단계가
  죽으면 회계가 통째로 유실된다. 파일럿에서 실제로 라운드를 날린 그 고장이다.
- `server_app` 은 `audit.json` 을 쓰지 않았다. `pilot_sim` 은 썼다.
- 라운드 번호 키가 한쪽은 `"round"`, 다른 쪽은 `"server-round"` 라 `flwr run` 경로가
  라운드 1 에서 `KeyError` 로 죽었다.

셋 다 "두 벌이라서" 난 고장이다. 공통부를 여기 한 곳에 내리고 두 진입점이 이것만 부른다.

## 라운드 번호 키는 상수로만 참조한다

`SERVER_ROUND_KEY` 하나만 쓴다. 문자열 리터럴을 직접 쓰면 같은 사고가 반복되므로
시험(`test_fl_round_wiring.py`)이 리터럴 사용을 금지한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "SERVER_ROUND_KEY",
    "CANONICAL_KEYS_KEY",
    "make_round_recorder",
    "finalize_accounting",
]

#: 서버가 클라이언트에 내려보내는 라운드 번호 키. **flwr 는 1부터 센다.**
#: 리터럴로 쓰지 마라 — 두 배선이 다른 이름을 쓰다가 `flwr run` 경로가 죽었다(F1).
SERVER_ROUND_KEY = "server-round"

#: 정본 키 리스트 전달 키. `ArrayRecord` 가 리스트 경로에서 이름을 인덱스로 바꾸므로
#: 키 이름은 반드시 따로 실려야 한다.
CANONICAL_KEYS_KEY = "canonical-keys"


def make_round_recorder(
    *,
    accounting: Any,
    atomic: Any,
    timer: Any,
    cell_from_metrics: Callable[[int, dict[str, Any]], Any],
    on_save: Callable[[int, Any], None] | None = None,
) -> Callable[[int, list[dict[str, Any]], Any], None]:
    """라운드 종료 콜백 하나를 만든다. 두 진입점이 이것을 그대로 쓴다.

    기록하는 것은 **학습 과정의 실측만**이다. 성능 지표는 학습이 전부 끝난 뒤 단일
    채점기가 낸다 — 학습 중에 지표를 보면 조기 종료 유혹이 생긴다.
    """

    def on_round_end(server_round: int, cells: list[dict[str, Any]], agg: Any) -> None:
        elapsed = timer.lap()
        round_idx = server_round - 1
        for m in cells:
            accounting.record(cell_from_metrics(round_idx, m))
            up = int(m.get("payload-bytes", 0))
            atomic.log_round(
                round_idx=round_idx,
                client_id=int(m.get("client-idx", -1)),
                n_train_samples=int(m.get("num-examples", 0)),
                metrics={
                    # F9 — ⑦ 원자 로그에 epochs_ran·lr 이 없어 R×E=N 을 로그에서
                    # 복원할 수 없었다. 두 칸 모두 같은 지표 집합을 남긴다.
                    "epochs_ran": float(m.get("epochs-ran", 0)),
                    "optimizer_steps": float(m.get("optimizer-steps", 0)),
                    "optimizer_updates": float(m.get("optimizer-updates", 0)),
                    "param_l2": float(m.get("param-l2", 0.0)),
                    "lr": float(m.get("lr", float("nan"))),
                    "peak_vram_gb": float(m.get("peak-vram-gb", 0.0)),
                    # 판정 2 — 가중 단위를 산출물이 말하게 한다(RQ3 해석 재료).
                    "supervised_tokens": float(m.get("supervised-tokens", 0.0)),
                    "fedavg_weight": float(m.get(WEIGHT_METRIC, 0.0)),
                },
                bytes_up=up,
                bytes_down=int(getattr(agg, "payload_bytes_down", 0) or up),
                wall_time=elapsed,
            )
        atomic.log_round(
            round_idx=round_idx,
            client_id="server",
            n_train_samples=int(getattr(agg, "total_examples", 0)),
            metrics={
                "global_l2": float(getattr(agg, "global_norm", 0.0)),
                "bn_divergence": float(getattr(agg, "bn_buffer_divergence", 0.0)),
                "missing_variance_ratio": float(getattr(agg, "missing_variance_ratio", 0.0)),
            },
            wall_time=elapsed,
        )
        if on_save is not None:
            on_save(server_round, agg)

    return on_round_end


#: `fl.strategy.WEIGHT_KEY` 와 같은 값. 순환 import 를 피하려고 여기서 다시 적는다.
WEIGHT_METRIC = "num-examples"


def finalize_accounting(
    *,
    accounting: Any,
    atomic: Any,
    out_dir: Path,
    num_rounds: int,
    client_ids: Iterable[Any],
    raise_on_failure: bool = True,
) -> Any:
    """회계 마감. **반드시 `finally` 에서 부른다.**

    학습이 끝난 뒤의 요약 출력 같은 단계가 죽어도 회계는 디스크에 남아야 한다.
    파일럿에서 정확히 그 순서로 라운드를 날렸다 — 학습은 완주됐는데 요약에서 크래시해
    뒤의 회계 마감이 통째로 사라졌고, 어디서 끊겼는지를 잃었다.

    `audit.json` 은 통과 여부와 무관하게 쓴다. 실패했다는 사실도 산출물이다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    accounting.to_csv(out_dir / "accounting.csv")
    accounting.to_json(out_dir / "accounting.json")
    report = accounting.audit()
    gaps = atomic.audit_rounds(num_rounds, list(client_ids) + ["server"])
    if gaps:
        report.failures.extend(gaps)
        report.ok = False
    (out_dir / "audit.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if raise_on_failure and not report.ok:
        from fl.strategy import RoundFailure

        raise RoundFailure(
            "회계 감사 실패 — run 을 무효로 처리한다. 채점하지 않는다.\n  - "
            + "\n  - ".join(report.failures)
        )
    return report
