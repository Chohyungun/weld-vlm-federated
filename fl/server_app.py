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
from fl.round_wiring import (ALL_CELLS, CANONICAL_KEYS_KEY, FED_CELLS,
                             SERVER_ROUND_KEY, SMOKE_CELL, WEIGHT_KEY,
                             finalize_accounting, make_round_recorder)
from fl.strategy import METRICS_KEY, RoundFailure, WeldFedAvg

__all__ = ["app", "build_accounting", "cell_to_client_ids"]

app = ServerApp()

def cell_to_client_ids(cell: str, num_clients: int = 3) -> tuple[int, ...]:
    """칸 검증 + 클라이언트 id. 스모크 칸도 받는다 — `flwr run` 진입점 자체를 더미로
    끝까지 돌리는 시험이 이 경로를 지나야 하기 때문이다(85번 ④: server_app 고유 구간이
    실행 이력 0 인 채로 커밋돼 있었다)."""
    if cell not in ALL_CELLS:
        raise ValueError(f"연합 칸이 아니다: {cell!r}. 허용: {ALL_CELLS}")
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
        optimizer_updates=int(m.get("optimizer-updates", 0)),
        resumed_from_epoch=(None if float(m.get("resumed-from-epoch", -1)) < 0
                            else int(m["resumed-from-epoch"])),
        # 조기 종료 계측을 클라이언트 실측에서 받는다. -1 은 "계측 없음"이라 None 으로
        # 옮긴다 — 0 으로 접으면 재지 않은 셀이 통과한다(74번 감사 P9).
        stopper_class=str(m.get("stopper-class", "")),
        stopper_true_count=(None if float(m.get("stopper-true-count", -1)) < 0
                            else int(m["stopper-true-count"])),
        stopper_calls=(None if m.get("stopper-calls") is None
                       else int(float(m["stopper-calls"]))),
        # 판정 2 — 실제 집계 가중은 **가중 키**에서 읽는다. "num-examples" 는 의미 키
        # (표본/페어 수)라 여기서 읽으면 85번 ① 의 거짓 단위가 재발한다.
        fedavg_weight=(float(m[WEIGHT_KEY]) if m.get(WEIGHT_KEY) is not None else None),
        fedavg_weight_unit=str(m.get("weight-unit", "")),
        supervised_tokens=int(float(m.get("supervised-tokens", 0))),
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
    if cell == SMOKE_CELL:
        # 배선 검사 산출물이 실험 결과로 인용되는 경로를 막는다.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "DO_NOT_CITE.md").write_text(
            "# 인용 금지\n\n배선 스모크(`SMOKE_CELL`) 산출물이다. 더미 2텐서로 돌았고 "
            "학습이 일어나지 않았다. 실험 결과가 아니다.\n", encoding="utf-8")
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
    # 라운드 종료 기록은 `fl/round_wiring` 하나가 만든다. 두 배선이 각자 구현하던 것이
    # 한쪽에만 있는 버그를 낳았다(80번 F1·F9 / G10-2).
    on_round_end = make_round_recorder(
        accounting=accounting, atomic=atomic, timer=RoundTimer(),
        cell_from_metrics=_cell_from_metrics,
        on_save=lambda sr, agg: _save_round(out_dir, sr, num_rounds, agg),
    )

    strategy = WeldFedAvg(
        expected_nodes=len(client_ids),
        canonical_keys=canonical_keys,
        reference_state_dict=reference_sd,
        on_round_end=on_round_end,
    )

    train_cfg = ConfigRecord({
        "cell": cell,
        # 정본 키 이름은 별도로 실린다 — ArrayRecord 가 리스트 경로에서 이름을 인덱스로
        # 바꾸기 때문이다. 이것이 없어서 `flwr run` 경로가 라운드 1 에서 즉사했다(F1).
        CANONICAL_KEYS_KEY: list(canonical_keys),
        "total-epochs": total_epochs,
        "local-epochs": local_epochs,
        "num-rounds": num_rounds,
        "base-seed": int(cfg.get("base-seed", 0)),
        "run-stamp": str(cfg.get("run-stamp", "")),
        "resume-root": str(cfg.get("resume-root", "")),
        **_cell_train_config(cell, cfg, out_dir),
    })
    try:
        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            train_config=train_cfg,
        )
    finally:
        # **`finally` 여야 한다.** 학습 밖 단계(요약 출력 등)가 죽어도 회계는 디스크에
        # 남아야 한다 — 파일럿에서 정확히 그 순서로 라운드를 날렸다(F1).
        finalize_accounting(
            accounting=accounting, atomic=atomic, out_dir=out_dir,
            num_rounds=num_rounds, client_ids=list(client_ids),
        )


def _cell_train_config(cell: str, cfg: Any, out_dir: Path) -> dict[str, Any]:
    """칸별로만 다른 설정. 라운드 루프는 칸이 무엇이든 같은 코드를 탄다."""
    if cell == SMOKE_CELL:
        return {"smoke-fail-at": str(cfg.get("smoke-fail-at", ""))}
    if cell == "sep_fed":
        return {
            "views-root": str(cfg["views-root"]),
            "model": str(cfg["model"]),
            "project": str(out_dir / "runs"),
            "profile": str(cfg.get("profile", "main")),
            **{f"num-examples-{i}": int(n)
               for i, n in enumerate(_as_list(cfg.get("num-examples", [])))},
        }
    return {f"client-tag-{i}": str(t)
            for i, t in enumerate(_as_list(cfg.get("client-tags", ["C1", "C2", "C3"])))}


def _as_list(v: Any) -> list:
    """run_config 값은 문자열로 오기도 한다. 쉼표 구분을 허용한다."""
    if isinstance(v, str):
        return [x for x in (p.strip() for p in v.split(",")) if x]
    return list(v)


def _load_initial(cell: str, cfg: Any) -> tuple["ArrayRecord", list[str], dict]:
    """칸별 초기 가중치·정본 키·기준 state_dict — **동일 출발 증명의 서버 쪽**.

    두 칸 모두 "시드를 박고 1회 만들어 캐시에 떨군 뒤 전 클라이언트가 그것을 주입받는다"
    는 같은 규약을 쓴다. 검출은 `detection/init_weights`, 통합형은 `vlm/init_adapter` 이고
    두 모듈의 시그니처·캐시 규약을 일부러 맞춰 두었다(74번 감사 C-1 의 비대칭 지적).

    이전 판은 `NotImplementedError` 라 `flwr run` 진입점이 아예 못 떴다(80번 F1).
    """
    from detection import serialize

    seed = int(cfg.get("base-seed", 0))
    cache = Path(str(cfg["project"])).resolve() / "fl" / cell / "initial.npz"

    if cell == SMOKE_CELL:
        # 더미 2텐서 — `fl.client_app.smoke_client_round` 가 돌려주는 것과 같은 모양이어야
        # `assert_compatible` 이 성립한다.
        import numpy as np
        import torch

        arrays = [np.arange(12, dtype=np.float32).reshape(4, 3),
                  np.arange(3, dtype=np.float32)]
        keys = ["smoke.w", "smoke.b"]
        ref = {k: torch.as_tensor(a) for k, a in zip(keys, arrays)}
    elif cell == "sep_fed":
        from detection.init_weights import build_initial_weights

        arrays, keys, ref = build_initial_weights(
            pretrained=str(cfg.get("model", "yolo11s.pt")),
            nc=int(cfg.get("num-classes", 4)), seed=seed, cache_path=cache,
        )
    elif cell == "uni_fed":
        from vlm.init_adapter import build_initial_adapter

        arrays, keys, ref = build_initial_adapter(
            model_id=str(cfg["model"]) if cfg.get("model") else None,
            seed=seed, cache_path=cache,
        )
        if not ref:
            # F11 — 캐시 분기가 ref 를 비워 돌려주면 `assert_compatible` 이 모든 키에서
            # "기준 모델에 없는 키"로 죽는다. 검출 쪽은 캐시에서도 ref 를 다시 만든다.
            # 여기서 배열로부터 복원해 같은 계약을 지킨다.
            import torch

            ref = {k: torch.as_tensor(a) for k, a in zip(keys, arrays)}
    else:
        raise ValueError(f"연합 칸이 아니다: {cell!r}. 허용: {ALL_CELLS}")

    serialize.assert_compatible(arrays, keys, ref)
    return ArrayRecord(list(arrays)), list(keys), ref


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
