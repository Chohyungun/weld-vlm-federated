"""연합 서버 — 라운드 루프 배선과 회계 마감.

`cell` 로 검출/VLM 을 가르되 **라운드 루프는 두 칸이 같은 코드를 탄다.** 칸마다 루프가
갈라지면 그 차이가 결과에 섞이고, 무엇이 학습 방식의 차이이고 무엇이 코드 경로의 차이인지
사후에 구분할 수 없다. 분기는 초기 파라미터·정본 키·클라이언트 앱 선택에만 걸린다.

라운드가 끝나면 회계 매트릭스를 감사한다. 빈 셀이 하나라도 있으면 **run 을 무효로 만든다** —
채점으로 넘어가지 않는다(게이트 #6 결정 A, 머지 차단 조건).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
    from flwr.serverapp import Grid, ServerApp
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "fl extra 가 설치되지 않았다. `uv sync --extra fl` 로 flwr 를 설치해야 한다."
    ) from exc

from detection.budget_audit import AccountingCell, AccountingMatrix
from fl.atomic_log import AtomicLog, RoundTimer, new_run_id
from fl.strategy import METRICS_KEY, RoundFailure, WeldFedAvg

__all__ = ["app", "build_accounting", "cell_to_client_ids"]

app = ServerApp()

#: 지원하는 칸. 라운드 루프는 공통이고 분기는 준비 단계에만 있다.
FED_CELLS = ("sep_fed", "uni_fed")


def cell_to_client_ids(cell: str, num_clients: int = 3) -> tuple[int, ...]:
    if cell not in FED_CELLS:
        raise ValueError(f"연합 칸이 아니다: {cell!r}. 허용: {FED_CELLS}")
    return tuple(range(num_clients))


def build_accounting(*, num_rounds: int, client_ids, local_epochs: int, total_epochs: int) -> AccountingMatrix:
    return AccountingMatrix(
        num_rounds=num_rounds,
        client_ids=client_ids,
        local_epochs=local_epochs,
        total_epochs=total_epochs,
    )


def _cell_from_metrics(round_idx: int, m: dict[str, Any]) -> AccountingCell:
    """클라이언트가 올린 메트릭을 회계 셀로 옮긴다.

    실사용 optimizer·lr 을 그대로 싣는 것이 요점이다. 설정 파일에 SGD 라고 적혀 있어도
    `optimizer='auto'` 가 남아 있으면 AdamW 가 돌 수 있고, 그 순간 5칸 공통 고정의
    '최적화' 항목이 깨진다.
    """
    return AccountingCell(
        round_idx=round_idx,
        client_idx=int(m.get("client-idx", -1)),
        epochs_ran=int(m.get("epochs-ran", 0)),
        optimizer_steps=int(m.get("optimizer-steps", 0)),
        num_examples=int(m.get("num-examples", 0)),
        seed=int(m.get("seed", 0)),
        param_l2_norm=float(m.get("param-l2", 0.0)),
        payload_bytes=int(m.get("payload-bytes", 0)),
        optimizer=str(m.get("optimizer", "")),
        lr=float(m.get("lr", float("nan"))),
        momentum=float(m.get("momentum", float("nan"))),
        arg_optimizer=str(m.get("arg-optimizer", "")),
        budget_fired_at=(None if float(m.get("budget-fired-at", -1)) < 0 else int(m["budget-fired-at"])),
    )


@app.main()
def main(grid: "Grid", context: "Context") -> None:
    cfg = context.run_config
    cell = str(cfg["cell"])
    num_rounds = int(cfg["num-server-rounds"])
    local_epochs = int(cfg["local-epochs"])
    total_epochs = int(cfg["total-epochs"])
    out_dir = Path(str(cfg["project"])).resolve() / "fl" / cell

    client_ids = cell_to_client_ids(cell, int(cfg.get("num-clients", 3)))
    accounting = build_accounting(
        num_rounds=num_rounds,
        client_ids=client_ids,
        local_epochs=local_epochs,
        total_epochs=total_epochs,
    )

    # 초기 파라미터·정본 키는 칸마다 다르다. 그 외는 공통 경로다.
    initial_arrays, canonical_keys, reference_sd = _load_initial(cell, cfg)

    atomic = AtomicLog(
        out_dir / "atomic_log.csv",
        run_id=new_run_id(cell, int(cfg.get("base-seed", 0)), str(cfg.get("run-stamp", "000000"))),
        seed=int(cfg.get("base-seed", 0)),
        cell=cell,
        split_hash=str(cfg.get("split-hash", "")),
    )
    timer = RoundTimer()

    def on_round_end(server_round: int, cells: list[dict[str, Any]], agg: Any) -> None:
        elapsed = timer.lap()
        for m in cells:
            accounting.record(_cell_from_metrics(server_round - 1, m))
            # 학습 과정의 실측만 남긴다. 성능 지표는 학습이 끝난 뒤 일괄 채점으로 얻는다.
            up = int(m.get("payload-bytes", 0))
            atomic.log_round(
                round_idx=server_round - 1,
                client_id=int(m.get("client-idx", -1)),
                n_train_samples=int(m.get("num-examples", 0)),
                metrics={
                    "epochs_ran": float(m.get("epochs-ran", 0)),
                    "optimizer_steps": float(m.get("optimizer-steps", 0)),
                    "param_l2": float(m.get("param-l2", 0.0)),
                    "lr": float(m.get("lr", float("nan"))),
                },
                bytes_up=up,
                bytes_down=int(getattr(agg, "payload_bytes_down", 0) or up),
                wall_time=elapsed,
            )
        # 서버 시점 집계 진단도 한 줄로 남긴다(client_id = "server")
        atomic.log_round(
            round_idx=server_round - 1,
            client_id="server",
            n_train_samples=int(getattr(agg, "total_examples", 0)),
            metrics={
                "global_l2": float(getattr(agg, "global_norm", 0.0)),
                "bn_divergence": float(getattr(agg, "bn_buffer_divergence", 0.0)),
                "missing_variance_ratio": float(getattr(agg, "missing_variance_ratio", 0.0)),
            },
            wall_time=elapsed,
        )
        _save_round(out_dir, server_round, num_rounds, agg)

    strategy = WeldFedAvg(
        expected_nodes=len(client_ids),
        canonical_keys=canonical_keys,
        reference_state_dict=reference_sd,
        on_round_end=on_round_end,
    )

    train_cfg = ConfigRecord({"total-epochs": total_epochs, "local-epochs": local_epochs})
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=train_cfg,
    )

    # 회계 마감 — 빈 셀이 하나라도 있으면 run 무효다.
    accounting.to_csv(out_dir / "accounting.csv")
    accounting.to_json(out_dir / "accounting.json")
    report = accounting.audit()
    gaps = atomic.audit_rounds(num_rounds, list(client_ids) + ["server"])
    if gaps:
        report.failures.extend(gaps)
        report.ok = False
    if not report.ok:
        raise RoundFailure(
            "회계 감사 실패 — run 을 무효로 처리한다. 채점하지 않는다.\n  - "
            + "\n  - ".join(report.failures)
        )


def _load_initial(cell: str, cfg: Any) -> tuple["ArrayRecord", list[str], dict]:
    """칸별 초기 가중치·정본 키·기준 state_dict.

    구현은 칸 진입 시점에 채운다. 검출은 동결 출발 체크포인트에서 `serialize` 로 뽑고,
    VLM 은 어댑터 키 집합만 싣는다(교환 단위가 state_dict 의 진부분집합이다).
    """
    raise NotImplementedError(
        "초기 파라미터 로더는 칸 진입 시 연결한다. "
        "검출: 동결 출발 체크포인트 → serialize.state_dict_to_ndarrays. "
        "VLM: 30번 명세 G2(교환 폐포 감사) 통과 후 어댑터 키 집합."
    )


def _save_round(out_dir: Path, server_round: int, num_rounds: int, agg: Any) -> None:
    """**매 라운드** 글로벌 모델을 저장한다.

    라운드별 궤적이 RQ3(참여 이득)의 재료이고, 그 궤적은 학습이 전부 끝난 뒤 저장된
    체크포인트를 단일 채점기로 일괄 채점해 얻는다. 간격을 두고 저장하면 어느 라운드에서
    무엇이 일어났는지 사후에 볼 수 없다.

    Flower 의 평가 라운드를 켜지 않는 이유도 같다. 켜면 실패 검사와 `R × E = N` 회계가
    얽히고, 학습 중에 지표를 보게 되어 조기 종료 유혹이 생긴다. 학습 경로와 평가 경로를
    분리하면 전 라운드가 동일한 채점기를 통과하는 이점도 따라온다.

    용량은 검출 fp32 기준 라운드당 약 38MB다. R=50 이면 1.9GB이고, 파일럿 R=3 이면
    무시할 수준이다. 궤적을 잃는 비용이 디스크 비용보다 크다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_arrays(out_dir / f"global_r{server_round:03d}.npz", agg.ndarrays)
    _write_arrays(out_dir / "latest.npz", agg.ndarrays)


def _write_arrays(path: Path, arrays: list) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, *arrays)
