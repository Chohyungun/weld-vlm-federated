"""재개 경로 시험 — 강제 종료 후 이어 간 런이 무중단 런과 같은 가중치에 닿는가.

## 왜 단위시험으로 부족한가

`tests/test_detection_resume.py` 는 저장·복원의 계약을 못 박는다. 하지만 재개가 실제로
쓸모 있으려면 **"이어 하기"가 "다시 하기"와 다르다**는 것이 학습 루프 전체에서 성립해야
한다. 옵티마이저 momentum, 스케줄러 위치, mosaic 종료 시점, 로더 셔플 순열 중 하나만
어긋나도 재개한 런은 다른 궤적을 그린다. 그건 사고 대응이 아니라 오염이다.

## 어떻게 재는가

같은 설정으로 두 번 돌린다.

- **무중단**: `total_epochs=E` 를 한 번에 완주
- **중단·재개**: epoch K 를 마친 직후 `os._exit(9)` 로 **프로세스를 즉시 죽인다**
  (정리 루틴을 타지 않는다 — 파일럿에서 실제로 두 번 일어난 외부 중지와 같은 형태다),
  그다음 같은 명령을 다시 띄워 재개시킨다

두 최종 가중치를 텐서별로 대조한다. 완전 일치가 목표이고, 어긋나면 **어긋난 크기를
그대로 보고한다** — "대체로 같다"로 넘기지 않는다.

    python scripts/verify_resume.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

SNAPSHOT_DIR = "data/processed/aihub71761_rt_v1_pilot3000"
ROOT = Path("outputs/probe_c/resume_verify").resolve()
VIEW = ROOT / "view"
EPOCHS = 4          # close_mosaic=1(파일럿) → epoch 3 에서 mosaic 종료 경로도 함께 탄다
DIE_AFTER = 1       # epoch 1 을 마친 직후 죽인다
FRACTION = 0.022    # 파일럿 스냅샷 train 에서 약 64장 → batch 2 로 32스텝/epoch


def build_view() -> Path:
    from data.manifest_io import load_snapshot
    from detection.dataset_view import build_yolo_view

    if (VIEW / "data.yaml").exists():
        return VIEW / "data.yaml"
    sn = load_snapshot(SNAPSHOT_DIR)
    r = build_yolo_view(sn, out_dir=VIEW, train_client=None)
    print(f"뷰 {r.n_images['train']}장", flush=True)
    return r.data_yaml


def _train(tag: str, resume_dir: Path | None, die_after: int | None,
           reseed: bool = False) -> Path:
    from detection.round_runner import train_round

    data_yaml = build_view()
    out_npz = ROOT / f"{tag}.npz"
    callbacks = {}
    if die_after is not None:
        def kill(trainer):  # noqa: ANN001 - ultralytics 콜백 규약
            # epoch die_after 를 마친 뒤 다음 epoch 이 시작되는 순간 죽는다. 그 시점이면
            # die_after 의 체크포인트는 이미 원자적으로 교체돼 있다.
            if int(trainer.epoch) > die_after:
                print(f"[verify] epoch {trainer.epoch} 진입 — 강제 종료", flush=True)
                sys.stdout.flush()
                os._exit(9)
        callbacks["on_train_epoch_start"] = kill

    res = train_round(
        data_yaml=data_yaml, model="yolo11n.pt",
        total_epochs=EPOCHS, local_epochs=EPOCHS,
        round_idx=0, client_idx=0, base_seed=0,
        project=ROOT / f"run_{tag}", profile="pilot",
        extra_overrides={"fraction": FRACTION},
        callbacks=callbacks or None,
        resume_dir=resume_dir, run_id="verify",
        clear_resume_on_success=False,   # 대조를 위해 남긴다. 본실험 기본값은 True 다
        loader_reseed_per_epoch=reseed,
    )
    np.savez(out_npz, *res.ndarrays)
    (ROOT / f"{tag}.json").write_text(json.dumps({
        "epochs_ran": res.epochs_ran, "optimizer_steps": res.optimizer_steps,
        "param_l2_norm": res.param_l2_norm, "seed": res.seed,
        "budget_fired_at": res.budget_fired_at,
        "effective_optimizer": res.effective_optimizer,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verify] {tag}: epochs {res.epochs_ran} steps {res.optimizer_steps} "
          f"l2 {res.param_l2_norm:.6f}", flush=True)
    return out_npz


def compare(tag_a: str = "baseline", tag_b: str = "resumed") -> dict:
    a = np.load(ROOT / f"{tag_a}.npz")
    b = np.load(ROOT / f"{tag_b}.npz")
    keys = list(a.files)
    assert keys == list(b.files), "키 순서가 다르다"
    worst = ("", 0.0, 0.0)
    n_diff = 0
    for k in keys:
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        d = float(np.max(np.abs(x - y))) if x.size else 0.0
        if d > 0:
            n_diff += 1
        scale = float(np.max(np.abs(x))) if x.size else 0.0
        rel = d / scale if scale else 0.0
        if d > worst[1]:
            worst = (k, d, rel)
    meta_a = json.loads((ROOT / f"{tag_a}.json").read_text(encoding="utf-8"))
    meta_b = json.loads((ROOT / f"{tag_b}.json").read_text(encoding="utf-8"))
    return {
        "pair": f"{tag_a} vs {tag_b}",
        "n_tensors": len(keys),
        "n_tensors_differing": n_diff,
        "identical": n_diff == 0,
        "worst_tensor": worst[0], "worst_abs_diff": worst[1], "worst_rel_diff": worst[2],
        "l2_a": meta_a["param_l2_norm"], "l2_b": meta_b["param_l2_norm"],
        "l2_rel_diff": abs(meta_a["param_l2_norm"] - meta_b["param_l2_norm"])
        / meta_a["param_l2_norm"],
        "meta_a": meta_a, "meta_b": meta_b,
        "epochs_match": meta_a["epochs_ran"] == meta_b["epochs_ran"] == EPOCHS,
        "steps_match": meta_a["optimizer_steps"] == meta_b["optimizer_steps"],
    }


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    ROOT.mkdir(parents=True, exist_ok=True)

    # 재시드 모드는 체크포인트 디렉터리를 따로 쓴다. 신원에 재시드 여부가 들어 있지
    # 않으므로 같은 디렉터리를 쓰면 두 모드가 서로의 상태를 이어받는다.
    if stage.startswith("rs_"):
        rdir, reseed, base = ROOT / "ckpt_rs", True, stage[3:]
    else:
        rdir, reseed, base = ROOT / "ckpt", False, stage
    prefix = "rs_" if reseed else ""

    if base in ("baseline", "baseline2"):
        _train(f"{prefix}{base}", resume_dir=None, die_after=None, reseed=reseed)
        return
    if base == "crash":
        _train(f"{prefix}crashed", resume_dir=rdir, die_after=DIE_AFTER, reseed=reseed)
        return
    if base == "resume":
        _train(f"{prefix}resumed", resume_dir=rdir, die_after=None, reseed=reseed)
        return

    from detection.resume import clear_resume

    py = sys.executable
    rep: dict = {}
    for mode, pre, cdir in (("기본", "", ROOT / "ckpt"), ("epoch별_로더_재시드", "rs_", ROOT / "ckpt_rs")):
        clear_resume(cdir)
        print(f"\n########## 모드: {mode} ##########", flush=True)
        print("=== 1/4 무중단 완주 ===", flush=True)
        subprocess.run([py, __file__, f"{pre}baseline"], check=True)

        # **대조군.** 같은 설정을 그냥 두 번 돌린 차이를 먼저 안다. 이 값을 모르면 재개의
        # 차이가 재개 때문인지 파이프라인 자체의 비결정성 때문인지 판별할 수 없다.
        print("=== 2/4 무중단 완주 재실행 (대조군) ===", flush=True)
        subprocess.run([py, __file__, f"{pre}baseline2"], check=True)

        print("=== 3/4 중단 (강제 종료 기대) ===", flush=True)
        rc = subprocess.run([py, __file__, f"{pre}crash"]).returncode
        left = sorted(p.name for p in cdir.glob("resume_ep*.pt"))
        print(f"[verify] 종료 코드 {rc} · 남은 체크포인트 {left}", flush=True)
        if rc == 0:
            raise SystemExit("중단이 일어나지 않았다 — 시험이 성립하지 않는다")
        if not left:
            raise SystemExit("체크포인트가 없다 — 재개할 것이 없다")

        print("=== 4/4 재개 완주 ===", flush=True)
        subprocess.run([py, __file__, f"{pre}resume"], check=True)

        ctrl = compare(f"{pre}baseline", f"{pre}baseline2")
        res = compare(f"{pre}baseline", f"{pre}resumed")
        rep[mode] = {
            "재실행_대조군": ctrl, "재개": res,
            "판정": {
                "재실행이_결정적인가": ctrl["identical"],
                "재개가_무중단과_동일한가": res["identical"],
                "재개_다른_텐서_수": res["n_tensors_differing"],
                "재개_worst_abs": res["worst_abs_diff"],
                "재개_l2_상대차": res["l2_rel_diff"],
                "회계_일치": res["epochs_match"] and res["steps_match"],
            },
            "checkpoints_after_crash": left, "crash_exit_code": rc,
        }
        print(json.dumps(rep[mode]["판정"], ensure_ascii=False, indent=2), flush=True)

    out = ROOT / "verify_resume.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}", flush=True)


if __name__ == "__main__":
    main()
