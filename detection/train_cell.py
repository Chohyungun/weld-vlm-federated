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

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from detection import serialize
from detection.resume import clear_resume
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


def _reload_arrays(path: Path) -> list:
    """저장된 NPZ 를 다시 읽는다(재로드 검증용). 파일 핸들을 닫고 돌려준다."""
    import numpy as np

    with np.load(path) as z:
        return [z[name] for name in z.files]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_durable(path: Path, write_fn, verify_fn=None, *, replace: bool = True) -> Path:
    """tmp 에 쓰고 fsync → (선택) tmp 를 검증 → `os.replace`. 쓰기·검증이 실패하면 `.tmp` 만 남는다.

    `replace=False` 면 교체하지 않고 tmp 경로를 돌려준다 — 호출자가 **다른 파일을 먼저 쓴 뒤**
    마지막에 교체하려는 경우다(완료 마커를 맨 나중에 확정하는 순서, 8-2 검토 I3).
    검증에 실패한 tmp 는 **일부러 남긴다**: 38MB 짜리 깨진 저장을 진단할 유일한 물증이고,
    최종 이름이 아니므로 `*.npz` 로 훑는 소비자에 걸리지 않는다.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        write_fn(fh)
        fh.flush()
        os.fsync(fh.fileno())
    if verify_fn is not None:
        verify_fn(tmp)
    if not replace:
        return tmp
    os.replace(tmp, path)
    return path


def _clear_resume_verified(resume_dir: Path, *, why: str) -> None:
    """재개 파일을 지우고 **정말 지워졌는지 확인**한다. 남으면 크게 알린다.

    `clear_resume` 는 `OSError` 를 삼키고 지운 개수만 돌려준다(resume.py). Windows 에서 다른
    핸들이 `.pt` 를 잡고 있으면 조용히 no-op 이 되고, 그러면 완주 잔해가 남아 다음 실행이
    "예산 완료 체크포인트" 로 거부된다. 잔존을 로그에 남겨 사람이 치울 수 있게 한다 —
    다음 실행도 최종 산출물이 있으면 스스로 다시 지운다(`_clear_stale_resume`).
    """
    clear_resume(resume_dir)
    left = sorted(p.name for p in Path(resume_dir).glob("resume_ep*")) if Path(resume_dir).is_dir() else []
    if left:
        print(f"[resume] 정리 실패 — {resume_dir} 에 {left} 가 남았다({why}). "
              "다음 실행 전에 지워라. 남아 있으면 완주 체크포인트로 거부된다.", flush=True)


def _clear_stale_resume(resume_dir: Path | None, final_npz: Path) -> None:
    """최종 산출물이 이미 있으면 그 재개 파일은 **잔해**다 — 학습 전에 치운다.

    8-2 로 `train_round` 가 더 이상 정리하지 않으므로, 저장과 정리 사이에서 죽은 실행은
    **완주 체크포인트**를 남긴다. 그 상태로 다시 띄우면 `validate_detection_resume` 가
    "예산 완료" 로 거부해 사람이 손으로 지워야 재기동된다. 최종 npz 가 있다는 것은 그
    체크포인트가 이미 쓸모를 다했다는 뜻이므로 여기서 지워 하드스톱을 없앤다.
    (신원이 다른 산출물이면 8-4 의 `assert_prior_meta_compatible` 가 이미 막았다.)
    """
    if resume_dir is None or not final_npz.exists():
        return
    if any(Path(resume_dir).glob("resume_ep*")):
        _clear_resume_verified(Path(resume_dir), why=f"{final_npz.name} 이 이미 있는 잔해")


def save_cell_weights(out_dir: Path, tag: str, result: RoundResult,
                      identity: dict[str, Any] | None = None) -> Path:
    """최종 가중치와 실행 메타를 **내구 저장 → 재로드 검증** 순으로 남긴다. 채점은 이 산출물을 읽는다.

    34번 §3-1 8-2: NPZ 는 tmp 에 쓰고 **tmp 를 다시 읽어** 배열 수·shape·dtype·값이 메모리의
    결과와 같음을 확인한다. 검증에 실패하면 최종 `.npz` 가 생기지 않는다.

    **순서가 계약이다** (8-2 검토 I3): `<tag>.npz` 는 `scripts/main_det.py` 의 **칸 건너뜀
    마커**이므로 맨 마지막에 확정한다 — tmp 검증 → sha256(tmp) → `meta.json` 내구 저장 →
    `os.replace(tmp, npz)`. 그래서 "마커가 있으면 meta 도 있다" 가 성립한다(그 반대 상태는
    `scripts/probe/audit_seed1_artifacts.py` 감사 3 이 잡는 고장이고, 건너뜀 경로는 meta 를
    보지 않는다). `os.replace` 는 바이트를 바꾸지 않으므로 tmp 에서 잰 해시·크기가 그대로
    최종 파일의 값이다. meta 개행은 **LF** 다(종전 `write_text` 는 Windows 에서 CRLF).

    이 함수가 정상 반환한 뒤에만 호출자가 재개 파일을 지운다.
    `identity` 를 주면 `identity` 블록(run_id·base_seed·split_hash·정책)도 덧붙인다(8-4).
    두 블록 다 키 추가만이라 기존 소비자(채점기는 NPZ 만 읽는다)는 영향이 없다.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.npz"
    arrays = list(result.ndarrays)

    def _verify(tmp_path: Path) -> None:
        back = _reload_arrays(tmp_path)
        if len(back) != len(arrays):
            raise RuntimeError(
                f"{tmp_path}: 재로드 배열 수 {len(back)} ≠ 메모리 {len(arrays)} — 저장이 깨졌다")
        for i, (a, b) in enumerate(zip(arrays, back)):
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                raise RuntimeError(f"{tmp_path}: 배열 {i} 재로드 불일치(shape/dtype/값) — 저장이 깨졌다")

    tmp = _write_durable(path, lambda fh: np.savez(fh, *arrays), _verify, replace=False)

    meta = {k: v for k, v in asdict(result).items() if k != "ndarrays"}
    meta["weights_file"] = {"name": path.name, "n_arrays": len(arrays),
                            "file_bytes": tmp.stat().st_size, "sha256": _sha256(tmp)}
    if identity is not None:
        meta["identity"] = dict(identity)
    payload = json.dumps(meta, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    _write_durable(out_dir / f"{tag}.meta.json", lambda fh: fh.write(payload))
    os.replace(tmp, path)          # 완료 마커는 맨 마지막
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
        rdir = (Path(resume_root).resolve() / f"sep_local_c{client_idx}" if resume_root else None)
        # 최종 산출물이 이미 있으면 그 재개 파일은 잔해다 — 학습 전에 치운다(8-2 검토 I1).
        _clear_stale_resume(rdir, out / f"sep_local_c{client_idx}.npz")
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
            resume_dir=rdir,
            run_id=stamp,
            loader_reseed_per_epoch=loader_reseed_per_epoch,
        )
        _log_result(log, result, client_idx, timer.lap())
        # 8-2: 내구 저장·재로드 검증이 끝난 뒤에만 재개 파일을 지운다. 저장이 던지면 남는다.
        save_cell_weights(out, f"sep_local_c{client_idx}", result, identity=ident)
        if rdir is not None:
            _clear_resume_verified(rdir, why="정상 완주 인계")
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
    rdir = Path(resume_root).resolve() / "sep_central" if resume_root else None
    _clear_stale_resume(rdir, out / "sep_central.npz")      # 8-2 검토 I1
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
        resume_dir=rdir,
        run_id=stamp,
        loader_reseed_per_epoch=loader_reseed_per_epoch,
    )
    _log_result(log, result, "central", timer.lap())
    # 8-2: 내구 저장·재로드 검증이 끝난 뒤에만 재개 파일을 지운다. 저장이 던지면 남는다.
    save_cell_weights(out, "sep_central", result, identity=ident)
    if rdir is not None:
        _clear_resume_verified(rdir, why="정상 완주 인계")
    return result
