"""연합 칸 `run_simulation` 배선 — ④ 분리·연합과 ⑦ 통합·연합 **둘 다**.

`fl/server_app.py` 는 `flwr run`(pyproject run-config) 전제로 짜여 있는데, 프로그램에서
직접 띄우는 경로(`run_simulation` 은 run_config 를 받지 않는다)가 따로 필요하다. 이 모듈이
같은 부품(WeldFedAvg·AccountingMatrix·AtomicLog·round_wiring)을 **다른 배선**으로 조립한다.
함정 겨냥 코드(전략·트레이너·집계)는 손대지 않는다 — 배선만 여기 것이다.

## ⑦ 도 이 경로로 온다 (80번 체크리스트 15항)

파일럿의 ⑦ 은 `scripts/pilot_c.cmd_cell7` 의 **인프로세스 순차 루프**였다. 전송 계층을
건너뛰었으므로 전략의 실패 검사 셋(에러 응답·응답 수 대조·키 다이제스트)이 한 번도
발화하지 않았고, 회계 실패에 예외도 `audit.json` 도 없었다. 이제 두 칸이 같은
`WeldFedAvg` 를 탄다 — 칸 분기는 **초기 파라미터·클라이언트 함수 선택**에만 있다.

클라이언트 식별은 시뮬레이션 백엔드가 넣어 주는 `context.node_config["partition-id"]` 를
쓴다. 없으면 추측하지 않고 즉시 실패한다 — 잘못된 파티션으로 조용히 돌면 세 클라이언트가
같은 데이터를 학습하고도 지표는 초록이다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flwr.clientapp import ClientApp
from flwr.common import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from detection.budget_audit import AccountingMatrix
from fl.atomic_log import AtomicLog, RoundTimer, new_run_id
from fl.round_wiring import (CANONICAL_KEYS_KEY, SERVER_ROUND_KEY,
                             finalize_accounting, make_round_recorder)
from fl.strategy import ARRAYS_KEY, CONFIG_KEY, METRICS_KEY, WeldFedAvg

__all__ = ["run_pilot_fed", "PILOT_CFG", "FED_CELLS", "SMOKE_CELL",
           "smoke_client_round", "SMOKE_BACKEND", "DEFAULT_BACKEND",
           "ray_startup_env"]

#: run_simulation 이 run_config 를 지원하지 않으므로 모듈 전역으로 주입한다.
#: run_pilot_fed 가 채우고, 서버·클라이언트 앱이 읽는다.
PILOT_CFG: dict[str, Any] = {}

FED_CELLS = ("sep_fed", "uni_fed")

#: 배선 자체를 검사하는 칸. **실험 칸이 아니다.**
#:
#: 진입점이 라운드 1 에서 죽는 상태로 커밋돼 있었는데 아무도 몰랐다(80번 F1). 시험이
#: 없어서가 아니라 **경로를 끝까지 돌리는 시험이** 없어서다. 그런데 실물 칸으로 스모크를
#: 돌리려면 모델·데이터·GPU 가 필요하고, 그러면 CI 에서 돌지 않아 결국 또 안 돌게 된다.
#:
#: 그래서 더미 2텐서를 돌려주는 칸을 배선에 **1급으로** 둔다(80번 G10-2 가 요구한 모양).
#: Ray 액터는 별도 프로세스라 시험 쪽 monkeypatch 가 닿지 않는다 — 스모크 경로가 코드에
#: 있어야만 실제 런타임 위에서 돌릴 수 있다.
#:
#: 오용 방지: 이 칸으로 돈 산출물 디렉터리에는 `DO_NOT_CITE.md` 가 함께 쓰인다.
SMOKE_CELL = "smoke"
ALL_CELLS = FED_CELLS + (SMOKE_CELL,)


def smoke_client_round(round_idx: int, client_idx: int, cfg: Any) -> tuple[list, dict, dict]:
    """더미 2텐서. 학습하지 않고 **배선만** 통과시킨다.

    클라이언트마다 값이 조금씩 다르다 — 전부 같으면 집계가 항등이 되어 평균 산술이
    돌았는지 알 수 없다.
    """
    scale = 1.0 + 0.01 * client_idx
    arrays = [
        (np.arange(12, dtype=np.float32).reshape(4, 3) * scale),
        (np.arange(3, dtype=np.float32) * scale),
    ]
    fail_at = str(cfg.get("smoke-fail-at", ""))
    if fail_at == f"{round_idx},{client_idx}":
        raise RuntimeError(
            f"스모크 의도적 실패: r{round_idx} c{client_idx} — 회계가 남는지 본다"
        )

    from detection import serialize

    tokens = 1000 + client_idx
    metrics = {
        "num-examples": float(tokens),        # 가중 = 감독 토큰 총합(판정 2)
        "supervised-tokens": float(tokens),
        "epochs-ran": float(cfg["local-epochs"]),
        "optimizer-steps": 5.0, "optimizer-updates": 5.0, "resumed-from-epoch": -1.0,
        "param-l2": serialize.params_l2_norm(arrays),
        "payload-bytes": float(serialize.payload_nbytes(arrays)),
        "seed": float(cfg["base-seed"]), "client-idx": float(client_idx),
        "lr": 1e-4, "momentum": float("nan"), "budget-fired-at": -1.0,
        "peak-vram-gb": 0.0, "stopper-true-count": -1.0,
    }
    strings = {
        "keys-digest": serialize.keys_digest(list(cfg[CANONICAL_KEYS_KEY])),
        "optimizer": "AdamW", "arg-optimizer": "AdamW", "stopper-class": "",
        "weight-unit": "supervised_tokens",
    }
    return arrays, metrics, strings

server_app = ServerApp()
client_app = ClientApp()


def _client_idx(context: Context) -> int:
    node_cfg = dict(context.node_config or {})
    if "partition-id" not in node_cfg:
        raise RuntimeError(
            "시뮬레이션 백엔드가 partition-id 를 넣지 않았다. 클라이언트를 식별할 수 없으므로 "
            "추측하지 않고 멈춘다 — 세 클라이언트가 같은 데이터를 돌면 지표는 초록인 채 "
            "연합이 무의미해진다."
        )
    return int(node_cfg["partition-id"])


@client_app.train()
def _client_train(msg: Message, context: Context) -> Message:
    cfg = msg.content[CONFIG_KEY]
    client_idx = _client_idx(context)
    round_idx = int(cfg[SERVER_ROUND_KEY]) - 1        # flwr 는 1부터 센다
    canonical_keys = list(cfg[CANONICAL_KEYS_KEY])
    arrays_in = msg.content[ARRAYS_KEY].to_numpy_ndarrays()
    cell = str(cfg["cell"])

    if cell == "sep_fed":
        from fl.client_det import run_client_round

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
            weights_in=arrays_in, canonical_keys=canonical_keys,
            round_idx=round_idx, client_idx=client_idx, cfg=run_cfg,
            profile=str(cfg.get("profile", "main")),
        )
    elif cell == "uni_fed":
        from fl.client_vlm import run_client_round

        run_cfg = {
            "client_tag": str(cfg[f"client-tag-{client_idx}"]),
            "local_epochs": int(cfg["local-epochs"]),
            "num_rounds": int(cfg["num-rounds"]),
            "base_seed": int(cfg["base-seed"]),
            "resume_root": str(cfg["resume-root"]) if cfg.get("resume-root") else None,
            "run_id": str(cfg.get("run-stamp", "")),
        }
        arrays_out, metrics, strings = run_client_round(
            adapter_in=arrays_in, canonical_keys=canonical_keys,
            round_idx=round_idx, client_idx=client_idx, cfg=run_cfg,
        )
    elif cell == SMOKE_CELL:
        arrays_out, metrics, strings = smoke_client_round(round_idx, client_idx, cfg)
    else:
        raise ValueError(f"연합 칸이 아니다: {cell!r}. 허용: {ALL_CELLS}")

    return Message(
        content=RecordDict({
            ARRAYS_KEY: ArrayRecord(arrays_out),
            METRICS_KEY: MetricRecord(metrics),
            CONFIG_KEY: ConfigRecord(strings),
        }),
        reply_to=msg,
    )


@server_app.main()
def _server_main(grid: Grid, context: Context) -> None:
    cfg = PILOT_CFG
    if not cfg:
        raise RuntimeError("PILOT_CFG 가 비어 있다. run_pilot_fed 로만 띄워라.")

    cell = str(cfg.get("cell", "sep_fed"))
    if cell not in ALL_CELLS:
        raise ValueError(f"연합 칸이 아니다: {cell!r}. 허용: {ALL_CELLS}")

    out_dir = Path(cfg["out_dir"]).resolve()
    if cell == SMOKE_CELL:
        # 배선 검사 산출물이 실험 결과로 인용되는 경로를 막는다.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "DO_NOT_CITE.md").write_text(
            "# 인용 금지\n\n배선 스모크(`SMOKE_CELL`) 산출물이다. 더미 2텐서로 돌았고 "
            "학습이 일어나지 않았다. 실험 결과가 아니다.\n", encoding="utf-8")
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
        run_id=new_run_id(cell, int(cfg["base_seed"]), str(cfg["run_stamp"])),
        seed=int(cfg["base_seed"]),
        cell=cell,
        split_hash=str(cfg["split_hash"]),
    )

    from fl.server_app import _cell_from_metrics, _save_round

    on_round_end = make_round_recorder(
        accounting=accounting, atomic=atomic, timer=RoundTimer(),
        cell_from_metrics=_cell_from_metrics,
        on_save=lambda sr, agg: _save_round(out_dir, sr, num_rounds, agg),
    )

    strategy = WeldFedAvg(
        expected_nodes=len(client_ids),
        canonical_keys=canonical_keys,
        reference_state_dict={k: v for k, v in cfg["reference_sd"].items()},
        on_round_end=on_round_end,
    )

    train_cfg = ConfigRecord({
        "cell": cell,
        CANONICAL_KEYS_KEY: canonical_keys,
        "local-epochs": int(cfg["local_epochs"]),
        "total-epochs": int(cfg["total_epochs"]),
        "num-rounds": num_rounds,
        "base-seed": int(cfg["base_seed"]),
        "run-stamp": str(cfg["run_stamp"]),
        "resume-root": str(cfg.get("resume_root", "")),
        **_cell_train_config(cell, cfg, out_dir),
    })
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
        finalize_accounting(
            accounting=accounting, atomic=atomic, out_dir=out_dir,
            num_rounds=num_rounds, client_ids=client_ids,
        )


def _cell_train_config(cell: str, cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """칸별로만 다른 설정. 라운드 루프는 두 칸이 같은 코드를 탄다."""
    if cell == "sep_fed":
        return {
            "views-root": str(cfg["views_root"]),
            "model": str(cfg["model"]),
            "project": str(out_dir / "runs"),
            "profile": str(cfg.get("profile", "main")),
            **{f"num-examples-{i}": int(n) for i, n in enumerate(cfg["num_examples"])},
        }
    if cell == SMOKE_CELL:
        return {"smoke-fail-at": str(cfg.get("smoke_fail_at", ""))}
    return {
        # 통합형은 페어를 매니페스트가 아니라 `load_pairs(client=...)` 로 고르므로
        # 뷰 경로 대신 클라이언트 태그를 내려보낸다.
        f"client-tag-{i}": str(t) for i, t in enumerate(cfg["client_tags"])
    }


#: 실런 백엔드 — 클라이언트당 GPU 를 통째로 준다. VLM 은 동시성 1 이고, 검출도 batch 32
#: 에서 7.7GB 를 쓰므로 두 클라이언트를 동시에 올릴 수 없다.
DEFAULT_BACKEND = {"client_resources": {"num_cpus": 2, "num_gpus": 1.0}}

#: CI 스모크 백엔드 — **GPU 를 요구하지 않고 Ray 의 발자국도 묶는다.**
#:
#: 배선만 보는 시험이 GPU 자원을 잡으면 (가) CPU 전용 러너에서 아예 못 돌고 (나) 같은
#: 기계에서 도는 학습과 경합해 굶는다. 실제로 게이트 학습과 겹쳐 10분 넘게 굶었다.
#:
#: 더해 **부하 상태에서 Ray 기동 자체가 실패한다.** 머지 게이트에서 raylet 기동이
#: GCS overloaded 로 타임아웃 나 스모크가 죽었다 — 그때 이 기계는 §4-6 파일럿이 GPU 와
#: CPU 를 물고 있었다. 배선을 보는 시험이 부하에 흔들리면 게이트가 불안정해지므로
#: Ray 가 띄우는 워커 수를 묶고(`num_cpus`) 대시보드·드라이버 로깅을 끈다.
#: 기동 타임아웃 상향은 환경변수라 `ray_startup_env()` 가 따로 준다.
SMOKE_BACKEND = {
    "client_resources": {"num_cpus": 1, "num_gpus": 0.0},
    "init_args": {
        # 3 클라이언트에 필요한 최소치. 기본값(호스트 전 코어)이면 부하 시 워커 기동이
        # 서로를 밀어내며 GCS 가 넘친다.
        "num_cpus": 4,
        "include_dashboard": False,
        "log_to_driver": False,
        "configure_logging": False,
    },
}


def ray_startup_env() -> dict[str, str]:
    """부하 상태에서 Ray 기동이 죽지 않도록 올리는 타임아웃. **환경변수로만 먹는다.**

    기본값(raylet 대기 10초)은 한가한 기계 기준이다. 같은 기계에서 학습이 돌면 그 안에
    raylet 이 못 뜨고 `ray.init` 이 GCS overloaded 로 죽는다 — 머지 게이트에서 실제로 났다.
    시험을 건너뛰는 대신 **기다리게** 만든다(건너뛰면 무이빨이 된다).
    """
    return {
        "RAY_raylet_start_wait_time_s": "120",
        "RAY_gcs_server_request_timeout_seconds": "60",
        "RAY_gcs_rpc_server_reconnect_timeout_s": "120",
        "RAY_health_check_initial_delay_ms": "60000",
        "RAY_health_check_timeout_ms": "30000",
        "RAY_health_check_period_ms": "10000",
    }


def run_pilot_fed(cfg: dict[str, Any], *, backend_config: dict | None = None) -> None:
    """연합 칸 하나를 실행한다. `cfg` 키는 `_server_main` 이 읽는 것 전부.

    `cfg["cell"]` 로 ④/⑦ 을 가른다.

    Args:
        backend_config: 생략하면 `DEFAULT_BACKEND`(GPU 1장 전용). 스모크는
            `SMOKE_BACKEND` 를 준다 — 배선을 보는 시험이 GPU 를 잡을 이유가 없다.
    """
    from flwr.simulation import run_simulation

    PILOT_CFG.clear()
    PILOT_CFG.update(cfg)
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=int(cfg["num_clients"]),
        backend_config=backend_config or DEFAULT_BACKEND,
    )
