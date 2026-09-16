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
from detection.round_runner import RoundResult, derive_seed, train_round
from fl.atomic_log import (AtomicLog, LedgerIdentityMismatch, RoundTimer, new_run_id,
                           policy_stamp)

__all__ = ["run_local_cell", "run_central_cell", "save_cell_weights",
           "assert_prior_meta_compatible"]


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


def _identity(*, run_id: str, base_seed: int, split_hash: str,
              loader_reseed_per_epoch: bool) -> dict[str, Any]:
    """meta.json 의 `identity` 블록. 다음 실행이 같은 out_dir 에 들어올 때 대조한다(34번 §3-1 8-4)."""
    return {"run_id": str(run_id), "base_seed": int(base_seed), "split_hash": str(split_hash),
            "loader_reseed_per_epoch": bool(loader_reseed_per_epoch)}


def assert_prior_meta_compatible(out_dir: str | Path, tag: str, *, client_idx: int,
                                 base_seed: int, identity: dict[str, Any]) -> None:
    """기존 `<tag>.meta.json` 이 있으면 신원을 대조하고, 다르면 `LedgerIdentityMismatch`.

    원장(`atomic_log.csv`)이 지워진 뒤에도 가중치·meta 는 남을 수 있으므로 원장 검사의
    두 번째 방어선이다. 옛 파일은 고치지 않고 그대로 읽는다(뒤호환):
    - 신판 meta(`identity` 블록): run_id·base_seed·split_hash·loader_reseed_per_epoch 대조.
    - 구판 meta(세 시드 본실험, `identity` 없음): 파생 시드 `derive_seed(base_seed, 0, client_idx)`
      와 `loader_reseed_per_epoch`(키가 없으면 F01 이전 코드 → False)만 대조한다.
    """
    p = Path(out_dir) / f"{tag}.meta.json"
    if not p.exists():
        return
    meta = json.loads(p.read_text(encoding="utf-8"))
    diff: dict[str, tuple[Any, Any]] = {}
    old = meta.get("identity")
    if isinstance(old, dict):
        for k in ("run_id", "base_seed", "split_hash", "loader_reseed_per_epoch"):
            if k in old and old[k] != identity[k]:
                diff[k] = (old[k], identity[k])
    else:
        want_seed = derive_seed(base_seed, 0, client_idx)
        if "seed" in meta and int(meta["seed"]) != want_seed:
            diff["seed"] = (meta["seed"], want_seed)
        old_policy = bool(meta.get("loader_reseed_per_epoch", False))
        if old_policy != bool(identity["loader_reseed_per_epoch"]):
            diff["loader_reseed_per_epoch"] = (old_policy, identity["loader_reseed_per_epoch"])
    if diff:
        raise LedgerIdentityMismatch(
            f"다른 실행의 산출물 위에 쓰지 않는다: {p} — 기존≠새 {diff}. "
            "다른 정책·시드·분할의 실행은 별도 출력 경로로 돌려라."
        )


def save_cell_weights(out_dir: Path, tag: str, result: RoundResult,
                      identity: dict[str, Any] | None = None) -> Path:
    """최종 가중치와 실행 메타를 남긴다. 채점은 이 산출물을 읽는다.

    `identity` 를 주면 meta 에 `identity` 블록(run_id·base_seed·split_hash·정책)을 덧붙인다 —
    키 추가만이라 기존 소비자(채점기)는 영향이 없다.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.npz"
    np.savez(path, *result.ndarrays)
    meta = {k: v for k, v in asdict(result).items() if k != "ndarrays"}
    if identity is not None:
        meta["identity"] = dict(identity)
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
    loader_reseed_per_epoch: bool = False,
) -> dict[int, RoundResult]:
    """② 분리·로컬. 클라이언트마다 독립 학습하고 결과를 셋 돌려준다.

    RQ3(참여 이득)의 기준선이 여기서 나온다. 연합 칸과 같은 클라이언트 분할·같은 예산으로
    돌아야 비교가 성립하므로, `total_epochs` 는 연합의 `R × E` 와 같은 값을 넣는다.
    """
    out = Path(out_dir).resolve()
    # 34번 §3-1 8-4 — 정책 접미를 stamp 에 새기고(꺼져 있으면 그대로), 기존 meta·원장과 신원을
    # 대조한 뒤에야 학습한다. 원장 대조는 `AtomicLog` 생성자가 한다.
    stamp = policy_stamp(run_stamp, loader_reseed_per_epoch)
    run_id = new_run_id("sep_local", base_seed, stamp)
    ident = _identity(run_id=run_id, base_seed=base_seed, split_hash=split_hash,
                      loader_reseed_per_epoch=loader_reseed_per_epoch)
    for client_idx in sorted(client_data_yamls):
        assert_prior_meta_compatible(out, f"sep_local_c{client_idx}", client_idx=client_idx,
                                     base_seed=base_seed, identity=ident)
    log = AtomicLog(
        out / "atomic_log.csv",
        run_id=run_id,
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
            run_id=stamp,
            loader_reseed_per_epoch=loader_reseed_per_epoch,
        )
        _log_result(log, result, client_idx, timer.lap())
        save_cell_weights(out, f"sep_local_c{client_idx}", result, identity=ident)
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
    loader_reseed_per_epoch: bool = False,
) -> RoundResult:
    """③ 분리·중앙. 학습 풀 전체로 한 번 학습한다.

    성능 상한 참조용이다. 현실에서는 데이터를 모을 수 없으므로 결과표에 그 각주를 단다.
    """
    out = Path(out_dir).resolve()
    stamp = policy_stamp(run_stamp, loader_reseed_per_epoch)
    run_id = new_run_id("sep_central", base_seed, stamp)
    ident = _identity(run_id=run_id, base_seed=base_seed, split_hash=split_hash,
                      loader_reseed_per_epoch=loader_reseed_per_epoch)
    assert_prior_meta_compatible(out, "sep_central", client_idx=0, base_seed=base_seed,
                                 identity=ident)
    log = AtomicLog(
        out / "atomic_log.csv",
        run_id=run_id,
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
        run_id=stamp,
        loader_reseed_per_epoch=loader_reseed_per_epoch,
    )
    _log_result(log, result, "central", timer.lap())
    save_cell_weights(out, "sep_central", result, identity=ident)
    return result
