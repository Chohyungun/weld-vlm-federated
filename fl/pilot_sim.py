"""④ 분리·연합 파일럿 — `run_simulation` 배선.

`fl/server_app.py` 는 `flwr run`(pyproject run-config) 전제로 짜여 있는데, 파일럿은
프로그램에서 직접 띄운다(`run_simulation` 은 run_config 를 받지 않는다). 그래서 이 모듈이
같은 부품(WeldFedAvg·train_round·AccountingMatrix·AtomicLog)을 **다른 배선**으로 조립한다.
함정 겨냥 코드(전략·트레이너·집계)는 손대지 않는다 — 배선만 파일럿용이다.

클라이언트 식별은 시뮬레이션 백엔드가 넣어 주는 `context.node_config["partition-id"]` 를
쓴다. 없으면 추측하지 않고 즉시 실패한다 — 잘못된 파티션으로 조용히 돌면 세 클라이언트가
같은 데이터를 학습하고도 지표는 초록이다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from flwr.clientapp import ClientApp
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from detection.budget_audit import AccountingMatrix
from fl.atomic_log import AtomicLog, RoundTimer, new_run_id
from fl.client_det import run_client_round
from fl.strategy import ARRAYS_KEY, CONFIG_KEY, METRICS_KEY, RoundFailure, WeldFedAvg

__all__ = ["run_pilot_fed", "PILOT_CFG"]

#: run_simulation 이 run_config 를 지원하지 않으므로 모듈 전역으로 주입한다.
#: run_pilot_fed 가 채우고, 서버·클라이언트 앱이 읽는다.
PILOT_CFG: dict[str, Any] = {}

server_app = ServerApp()
client_app = ClientApp()


@client_app.train()
def _client_train(msg: Message, context: Context) -> Message:
    cfg = msg.content[CONFIG_KEY]
    node_cfg = dict(context.node_config or {})
    if "partition-id" not in node_cfg:
        raise RuntimeError(
            "시뮬레이션 백엔드가 partition-id 를 넣지 않았다. 클라이언트를 식별할 수 없으므로 "
            "추측하지 않고 멈춘다 — 세 클라이언트가 같은 데이터를 돌면 지표는 초록인 채 "
            "연합이 무의미해진다."
        )
    client_idx = int(node_cfg["partition-id"])
    round_idx = int(cfg["server-round"]) - 1  # Flower 는 1부터 센다

    weights_in = msg.content[ARRAYS_KEY].to_numpy_ndarrays()
    canonical_keys = list(cfg["canonical-keys"])

    run_cfg = {
        "data_yaml": str(Path(str(cfg["views-root"])) / f"client{client_idx}" / "data.yaml"),
        "model": str(cfg["model"]),
        "total_epochs": int(cfg["total-epochs"]),
        "local_epochs": int(cfg["local-epochs"]),
        "base_seed": int(cfg["base-seed"]),
        "num_examples": int(cfg[f"num-examples-{client_idx}"]),
        "project": str(cfg["project"]),
        # 라운드 안에서 죽어도 그 라운드를 0부터 다시 돌지 않는다. 신원(라운드·클라이언트·
        # 시드)이 다르면 거부되므로 옆 라운드 상태를 물려받는 경로는 없다.
        "resume_root": str(cfg["resume-root"]) if cfg.get("resume-root") else None,
        "run_id": str(cfg.get("run-stamp", "")),
    }
    arrays_out, metrics, strings = run_client_round(
        weights_in=weights_in,
        canonical_keys=canonical_keys,
        round_idx=round_idx,
        client_idx=client_idx,
        cfg=run_cfg,
        profile=str(cfg.get("profile", "main")),
    )
    return Message(
        content=RecordDict(
            {
                ARRAYS_KEY: ArrayRecord(arrays_out),
                METRICS_KEY: MetricRecord(metrics),
                CONFIG_KEY: ConfigRecord(strings),
            }
        ),
        reply_to=msg,
    )


@server_app.main()
def _server_main(grid: Grid, context: Context) -> None:
    cfg = PILOT_CFG
    if not cfg:
        raise RuntimeError("PILOT_CFG 가 비어 있다. run_pilot_fed 로만 띄워라.")

    out_dir = Path(cfg["out_dir"]).resolve()
    num_rounds = int(cfg["num_rounds"])
    client_ids = list(range(int(cfg["num_clients"])))
    canonical_keys = list(cfg["canonical_keys"])

    accounting = AccountingMatrix(
        num_rounds=num_rounds,
        client_ids=client_ids,
        local_epochs=int(cfg["local_epochs"]),
        total_epochs=int(cfg["total_epochs"]),
    )
    atomic = AtomicLog(
        out_dir / "atomic_log.csv",
        run_id=new_run_id("sep_fed", int(cfg["base_seed"]), str(cfg["run_stamp"])),
        seed=int(cfg["base_seed"]),
        cell="sep_fed",
        split_hash=str(cfg["split_hash"]),
    )
    timer = RoundTimer()

    from fl.server_app import _cell_from_metrics, _save_round

    def on_round_end(server_round: int, cells: list[dict[str, Any]], agg: Any) -> None:
        elapsed = timer.lap()
        for m in cells:
            accounting.record(_cell_from_metrics(server_round - 1, m))
            up = int(m.get("payload-bytes", 0))
            atomic.log_round(
                round_idx=server_round - 1,
                client_id=int(m.get("client-idx", -1)),
                n_train_samples=int(m.get("num-examples", 0)),
                metrics={
                    "epochs_ran": float(m.get("epochs-ran", 0)),
                    "optimizer_steps": float(m.get("optimizer-steps", 0)),
                    "optimizer_updates": float(m.get("optimizer-updates", 0)),
                    "param_l2": float(m.get("param-l2", 0.0)),
                    "lr": float(m.get("lr", float("nan"))),
                    "peak_vram_gb": float(m.get("peak-vram-gb", 0.0)),
                },
                bytes_up=up,
                bytes_down=up,  # 배포도 같은 페이로드다 (fp32 전체 가중치)
                wall_time=elapsed,
            )
        atomic.log_round(
            round_idx=server_round - 1,
            client_id="server",
            n_train_samples=int(agg.total_examples),
            metrics={
                "global_l2": float(agg.global_norm),
                "bn_divergence": float(agg.bn_buffer_divergence),
                "missing_variance_ratio": float(agg.missing_variance_ratio),
            },
            wall_time=elapsed,
        )
        _save_round(out_dir, server_round, num_rounds, agg)

    reference_sd = {k: v for k, v in cfg["reference_sd"].items()}
    strategy = WeldFedAvg(
        expected_nodes=len(client_ids),
        canonical_keys=canonical_keys,
        reference_state_dict=reference_sd,
        on_round_end=on_round_end,
    )

    train_cfg = ConfigRecord(
        {
            "canonical-keys": canonical_keys,
            "views-root": str(cfg["views_root"]),
            "model": str(cfg["model"]),
            "total-epochs": int(cfg["total_epochs"]),
            "local-epochs": int(cfg["local_epochs"]),
            "base-seed": int(cfg["base_seed"]),
            "project": str(out_dir / "runs"),
            # 재개 파일은 산출물 트리 밖에 둔다 — 채점·내보내기가 훑지 않는 곳이다.
            "resume-root": str(out_dir.parent / "_resume" / "sep_fed"),
            "run-stamp": str(cfg["run_stamp"]),
            "profile": str(cfg.get("profile", "main")),
            **{f"num-examples-{i}": int(n) for i, n in enumerate(cfg["num_examples"])},
        }
    )
    try:
        strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(list(cfg["initial_arrays"])),
            num_rounds=num_rounds,
            train_config=train_cfg,
        )
    finally:
        # 요약 출력 등 학습 밖 단계가 죽어도 회계는 남아야 한다. 파일럿의 산출물은
        # "어디서 끊겼는가"이고, 회계가 함께 사라지면 그 답을 잃는다.
        _finalize(accounting, atomic, out_dir, num_rounds, client_ids)


def _finalize(accounting, atomic, out_dir, num_rounds, client_ids):
    accounting.to_csv(out_dir / "accounting.csv")
    accounting.to_json(out_dir / "accounting.json")
    report = accounting.audit()
    gaps = atomic.audit_rounds(num_rounds, client_ids + ["server"])
    if gaps:
        report.failures.extend(gaps)
        report.ok = False
    (out_dir / "audit.json").write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not report.ok:
        raise RoundFailure("회계 감사 실패:\n  - " + "\n  - ".join(report.failures))


def run_pilot_fed(cfg: dict[str, Any]) -> None:
    """④를 실행한다. `cfg` 키는 `_server_main` 이 읽는 것 전부."""
    from flwr.simulation import run_simulation

    PILOT_CFG.clear()
    PILOT_CFG.update(cfg)
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=int(cfg["num_clients"]),
        backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 1.0}},
    )
