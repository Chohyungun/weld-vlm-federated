"""분리·로컬 / 분리·중앙 진입점 — 파일럿 순서 ②③.

두 칸은 연합이 아니지만 **같은 `train_round` 를 `R=1, E=N` 으로 통과한다.** 칸마다 다른
학습 루프를 타면 세 칸의 차이가 학습 방식 때문인지 코드 경로 때문인지 구분할 수 없다.

- **② 분리·로컬**: 클라이언트 셋이 각자 자기 데이터로만 학습한다. 모델이 셋 나온다
- **③ 분리·중앙**: 학습 풀 전체로 한 번 학습한다. 모델이 하나 나온다

원자 로그는 연합 칸과 같은 스키마로 남긴다. 라운드 개념이 없으므로 `round=0` 이고,
중앙 칸의 `client_id` 는 `"central"` 이다. 스키마를 칸마다 바꾸면 나중에 한 파일로 못 합친다.

학습 중에 성능을 재지 않는다. 저장된 체크포인트를 학습이 끝난 뒤 단일 채점기로 일괄
채점하는 것이 정본 경로다.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from detection import serialize
from detection.round_runner import RoundResult, train_round
from fl.atomic_log import AtomicLog, RoundTimer, new_run_id

__all__ = ["run_local_cell", "run_central_cell", "save_cell_weights"]


def _log_result(log: AtomicLog, result: RoundResult, client_id: int | str, wall: float) -> None:
    eff = result.effective_optimizer
    log.log_round(
        round_idx=result.round_idx,
        client_id=client_id,
        n_train_samples=result.num_examples,
        metrics={
            "epochs_ran": float(result.epochs_ran),
            "optimizer_steps": float(result.optimizer_steps),
            "optimizer_updates": float(getattr(result, "optimizer_updates", 0) or 0),
            "param_l2": float(result.param_l2_norm),
            "lr": float(eff.get("lr", float("nan"))),
            "peak_vram_gb": float(result.peak_vram_gb),
        },
        bytes_up=0,      # 로컬·중앙 칸은 교환이 없다. 통신량 0 이 정의상 참이다
        bytes_down=0,
        wall_time=wall,
    )


def save_cell_weights(out_dir: Path, tag: str, result: RoundResult) -> Path:
    """최종 가중치와 실행 메타를 남긴다. 채점은 이 산출물을 읽는다."""
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.npz"
    np.savez(path, *result.ndarrays)
    meta = {k: v for k, v in asdict(result).items() if k != "ndarrays"}
    (out_dir / f"{tag}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


def run_local_cell(
    *,
    client_data_yamls: dict[int, str | Path],
    client_num_examples: dict[int, int],
    model: str,
    total_epochs: int,
    base_seed: int,
    out_dir: str | Path,
    split_hash: str,
    run_stamp: str,
    profile: str = "main",
    initial_weights: "Sequence" = None,
    canonical_keys: "Sequence[str]" = None,
    resume_root: str | Path | None = None,
) -> dict[int, RoundResult]:
    """② 분리·로컬. 클라이언트마다 독립 학습하고 결과를 셋 돌려준다.

    RQ3(참여 이득)의 기준선이 여기서 나온다. 연합 칸과 같은 클라이언트 분할·같은 예산으로
    돌아야 비교가 성립하므로, `total_epochs` 는 연합의 `R × E` 와 같은 값을 넣는다.
    """
    out = Path(out_dir).resolve()
    log = AtomicLog(
        out / "atomic_log.csv",
        run_id=new_run_id("sep_local", base_seed, run_stamp),
        seed=base_seed,
        cell="sep_local",
        split_hash=split_hash,
    )
    timer = RoundTimer()
    results: dict[int, RoundResult] = {}
    for client_idx in sorted(client_data_yamls):
        result = train_round(
            data_yaml=client_data_yamls[client_idx],
            model=model,
            total_epochs=total_epochs,
            local_epochs=total_epochs,   # R=1 퇴화 케이스
            round_idx=0,
            client_idx=client_idx,
            base_seed=base_seed,
            num_examples=client_num_examples[client_idx],
            weights_in=initial_weights,
            canonical_keys=canonical_keys,
            project=out,
            profile=profile,
            # ② 는 클라이언트마다 N epoch 단일 런이다. 본실험에서 12시간대이므로
            # 재개 경로를 켠다 — 채점 대상이 아니라 재개용이다(detection/resume.py).
            resume_dir=(Path(resume_root).resolve() / f"sep_local_c{client_idx}"
                        if resume_root else None),
            run_id=run_stamp,
        )
        _log_result(log, result, client_idx, timer.lap())
        save_cell_weights(out, f"sep_local_c{client_idx}", result)
        results[client_idx] = result
    return results


def run_central_cell(
    *,
    data_yaml: str | Path,
    num_examples: int,
    model: str,
    total_epochs: int,
    base_seed: int,
    out_dir: str | Path,
    split_hash: str,
    run_stamp: str,
    profile: str = "main",
    initial_weights: "Sequence" = None,
    canonical_keys: "Sequence[str]" = None,
    resume_root: str | Path | None = None,
) -> RoundResult:
    """③ 분리·중앙. 학습 풀 전체로 한 번 학습한다.

    성능 상한 참조용이다. 현실에서는 데이터를 모을 수 없으므로 결과표에 그 각주를 단다.
    """
    out = Path(out_dir).resolve()
    log = AtomicLog(
        out / "atomic_log.csv",
        run_id=new_run_id("sep_central", base_seed, run_stamp),
        seed=base_seed,
        cell="sep_central",
        split_hash=split_hash,
    )
    timer = RoundTimer()
    result = train_round(
        data_yaml=data_yaml,
        model=model,
        total_epochs=total_epochs,
        local_epochs=total_epochs,
        round_idx=0,
        client_idx=0,
        base_seed=base_seed,
        num_examples=num_examples,
        weights_in=initial_weights,
        canonical_keys=canonical_keys,
        project=out,
        profile=profile,
        # ③ 은 학습 풀 전체로 도는 N epoch **단일 런**이다. 본실험에서 가장 긴 검출 런이고
        # 도중에 죽으면 처음부터다. 재개 경로를 켠다 — 채점 대상이 아니다.
        resume_dir=(Path(resume_root).resolve() / "sep_central" if resume_root else None),
        run_id=run_stamp,
    )
    _log_result(log, result, "central", timer.lap())
    save_cell_weights(out, "sep_central", result)
    return result
