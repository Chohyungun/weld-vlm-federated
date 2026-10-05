"""통합형 재개 등가 시험(T-eq-uni) — 죽였다가 이어 간 학습이 중단 없는 학습과 같은 곳에 닿는가(12번 §4-2).

## 어떻게 재는가

같은 신원 · 같은 예산 · 같은 행으로 네 단계를 돈다. 단계마다 **자식 프로세스**다.

| 단계 | 무엇 |
|---|---|
| `baseline` | 중단 없이 끝낸다 |
| `baseline2` | 한 번 더 중단 없이 끝낸다 — 같은 장비의 비결정성 바닥을 잰다 |
| `crash` | epoch `die_after` 의 체크포인트가 **저장된 뒤** `os._exit(9)` 로 죽인다. 정리 루틴을 타지 않는다 |
| `resume` | `crash` 와 같은 재개 폴더로 다시 띄워 이어 간다 |

대조 대상은 12번 §4-2 의 다섯 — 최종 어댑터 배열, `optimizer_steps`, `supervised_tokens`, 학습률 궤적(epoch 끝의
학습률과 마지막 세 스텝), epoch 별 손실. 어댑터는 **비트 동일**이 기준이다. 서지 않으면 텐서마다의 최대 절대차를
그대로 적는다 — 허용오차를 먼저 정하지 않는다. `baseline` 과 `baseline2` 가 이미 다르면 그 차이를 바닥으로 함께 적는다.

## 무엇을 쓰나

기본은 파일럿 모델(`vlm.pilot_vlm.MODEL_ID`) · 파일럿 페어의 C1 학습 행 앞 `--rows` 장 · 3 epoch · epoch 1 뒤에 죽인다.
등가는 모델 크기와 무관하다고 보고 소형으로 돈다 — **결론을 본실험 모델로 일반화하지 않는다.** GPU 가 있어야 돈다.
12번 §4-2 의 1 은 "모델 판 · 페어 형식 · 창 경계 · 스케줄은 본실험과 같아야 한다" 와 "0.8B · 소표본으로 돌린다" 를 함께 적어
명세 안에서 갈린다 — 이 실행기는 뒤쪽을 따른다. 어느 쪽인지는 명세가 정한다. 기본 행 수는 누적 창 둘에 부분 창 하나가 남게 둔다.

자식은 **고의 중단 변수를 지운 환경**으로 띄운다(`vlm.fault.scrubbed_env`). `crash` 의 `os._exit(9)` 는 이 실행기의 단계이고
고의 중단 장치가 아니다 — 목적 `main` 은 모델 적재 이음새가 승인된 실제 구현만 받게 하려는 것이다.

    python -X utf8 scripts/verify_resume_uni.py --root outputs/probe_c/resume_verify_uni

CLI 에는 구현을 고르는 인자가 없다. 합성 시험은 단계 함수 `run_stage` 를 **파이썬 인자로** 대역 적재기와
죽음 대신 예외를 올리는 함수를 넘겨 프로세스 안에서 부른다 — 그 시험은 자식 프로세스의 사망 · 재기동을 재현하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

STAGES = ("baseline", "baseline2", "crash", "resume")
DEFAULT_ROOT = Path("outputs/probe_c/resume_verify_uni")
DEFAULT_ROWS = 72          # 누적 32 → epoch 당 3 스텝(마지막은 8 행의 부분 창)
DEFAULT_EPOCHS = 3
DEFAULT_DIE_AFTER = 1
BASE_SEED = 20260828


def _stage_dir(root: Path, stage: str) -> Path:
    return root / ("resume_run" if stage in ("crash", "resume") else stage)


def run_stage(stage: str, root: Path, *, rows: list[dict], epochs: int, die_after: int,
              model_loader: Callable | None = None, purpose: str = "main",
              die: Callable[[], None] | None = None, pairs_path: str | None = None) -> dict[str, Any]:
    """한 단계를 이 프로세스에서 돈다. `crash` 는 `die_after` 체크포인트가 **저장된 뒤** `die()` 를 부른다.

    `die` 의 기본은 `os._exit(9)` 다. 결과는 `<단계 폴더>/final.npz` · `epochs.jsonl`(epoch 행, 덧붙이기) ·
    `metrics.json` 이다. `crash` · `resume` 은 같은 폴더와 같은 재개 폴더를 쓴다.
    """
    from vlm.pilot_vlm import train_rounds

    if stage not in STAGES:
        raise ValueError(f"모르는 단계: {stage!r} — {list(STAGES)}")
    if not 0 <= die_after < epochs - 1:
        raise ValueError(f"die_after {die_after} 는 0 이상 {epochs - 1} 미만이어야 이어 갈 epoch 가 남는다")
    d = _stage_dir(root, stage)
    d.mkdir(parents=True, exist_ok=True)
    if stage in ("baseline", "baseline2", "crash") and any(d.iterdir()):
        raise FileExistsError(f"새로 도는 단계인데 폴더가 비어 있지 않다: {d} — 새 루트로 돈다")
    if stage == "resume" and not (d / "resume").is_dir():
        raise FileNotFoundError(f"이어 갈 재개 폴더가 없다: {d / 'resume'} — crash 를 먼저")
    die = die or (lambda: os._exit(9))

    def ledger_cb(v: dict[str, Any]) -> None:
        with (d / "epochs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**v, "stage": stage, "pid": os.getpid()}, ensure_ascii=False) + "\n")

    def after_ckpt(ep: int, ckpt: Any) -> None:
        # 저장이 성공했을 때만 불린다(`train_rounds`) — 체크포인트가 디스크에 있는 채로 죽는다.
        if stage == "crash" and ep == die_after:
            die()

    arrays, keys, m, _ = train_rounds(
        rows=rows, epochs=epochs, round_idx=0, client_idx=0, base_seed=BASE_SEED,
        resume_dir=str(d / "resume"), run_id="t-eq-uni", num_rounds=1, pairs_path=pairs_path,
        model_loader=model_loader, purpose=purpose, tag="t_eq_uni", ledger_cb=ledger_cb,
        after_ckpt_cb=after_ckpt,
    )
    if stage == "crash":
        raise RuntimeError(f"crash 단계가 epoch {die_after} 에서 죽지 않았다")
    np.savez(d / "final.npz", **dict(zip(keys, arrays)))
    keep = {k: m[k] for k in ("optimizer_steps", "supervised_tokens", "epochs_ran", "resumed_from_epoch",
                              "lr_trace_tail", "lr_trace_head", "resume_save_failures")}
    (d / "metrics.json").write_text(json.dumps(keep, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    return keep


def _epoch_rows(d: Path) -> dict[int, dict[str, Any]]:
    """epoch 별 마지막 기록 — 죽은 프로세스가 쓴 뒤 다시 돈 epoch 는 뒤의 것이 이긴다."""
    out: dict[int, dict[str, Any]] = {}
    for line in (d / "epochs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[int(r["epoch"])] = r
    return out


def compare(root: Path, a: str, b: str) -> dict[str, Any]:
    """두 단계의 최종 어댑터와 회계를 맞댄다. 어긋나면 크기를 그대로 적는다."""
    da, db = _stage_dir(root, a), _stage_dir(root, b)
    with np.load(da / "final.npz") as za, np.load(db / "final.npz") as zb:
        if list(za.files) != list(zb.files):
            return {"pair": [a, b], "keys_equal": False}
        diffs = {k: float(np.max(np.abs(za[k].astype(np.float64) - zb[k].astype(np.float64))))
                 if za[k].shape == zb[k].shape else float("inf") for k in za.files}
        bit = all(za[k].shape == zb[k].shape and za[k].dtype == zb[k].dtype and np.array_equal(za[k], zb[k])
                  for k in za.files)
    ma = json.loads((da / "metrics.json").read_text(encoding="utf-8"))
    mb = json.loads((db / "metrics.json").read_text(encoding="utf-8"))
    ea, eb = _epoch_rows(da), _epoch_rows(db)
    epochs = sorted(set(ea) | set(eb))
    per_epoch = {ep: {k: (ea.get(ep, {}).get(k), eb.get(ep, {}).get(k))
                      for k in ("mean_ce", "lr", "optimizer_steps", "supervised_tokens")} for ep in epochs}
    return {
        "pair": [a, b],
        "keys_equal": True,
        "adapter_bit_identical": bool(bit),
        "adapter_max_abs_diff": max(diffs.values()) if diffs else 0.0,
        "adapter_max_abs_diff_by_tensor": {k: v for k, v in sorted(diffs.items(), key=lambda kv: -kv[1])[:5]},
        "optimizer_steps": [ma["optimizer_steps"], mb["optimizer_steps"]],
        "supervised_tokens": [ma["supervised_tokens"], mb["supervised_tokens"]],
        "lr_trace_tail": [ma["lr_trace_tail"], mb["lr_trace_tail"]],
        "per_epoch": per_epoch,
        "per_epoch_equal": all(x == y for v in per_epoch.values() for x, y in v.values()),
    }


def judge(root: Path, *, epochs: int, die_after: int) -> dict[str, Any]:
    """판정 — 재개 쌍(`baseline`↔`resume`)이 비트 동일이고, 회계 · 학습률 궤적 · epoch 별 기록이 같고, 재개가 죽인 다음
    epoch 에서 시작했고, 두 단계 모두 예산의 epoch 를 다 돌았으면 통과. 바닥(`baseline`↔`baseline2`)을 함께 싣는다."""
    pair = compare(root, "baseline", "resume")
    floor = compare(root, "baseline", "baseline2")
    counts_equal = bool(pair.get("keys_equal") and pair["optimizer_steps"][0] == pair["optimizer_steps"][1]
                        and pair["supervised_tokens"][0] == pair["supervised_tokens"][1])
    # 학습률 궤적은 **프로세스마다** 쌓인다 — 이어 간 프로세스의 꼬리에는 재개 뒤의 스텝만 있다. 스텝 번호로 맞대고
    # (겹치는 스텝의 값이 같다) 마지막 스텝이 같아야 한다. 목록째 비교하면 스텝이 셋보다 적게 남은 재개가 늘 떨어진다.
    lr_equal = False
    if pair.get("keys_equal"):
        ta, tb = ({int(s): v for s, v in tail} for tail in pair["lr_trace_tail"])
        common = set(ta) & set(tb)
        lr_equal = bool(common) and max(ta) == max(tb) and all(ta[s] == tb[s] for s in common)
    resumed = json.loads((_stage_dir(root, "resume") / "metrics.json").read_text(encoding="utf-8"))
    base = json.loads((_stage_dir(root, "baseline") / "metrics.json").read_text(encoding="utf-8"))
    resumed_ok = resumed.get("resumed_from_epoch") == die_after + 1
    epochs_ok = base.get("epochs_ran") == epochs and resumed.get("epochs_ran") == epochs
    return {
        "resume_pair": pair,
        "nondeterminism_floor": floor,
        "resumed_from_epoch": resumed.get("resumed_from_epoch"),
        "checks": {"adapter_bit_identical": bool(pair.get("adapter_bit_identical")), "counts_equal": counts_equal,
                   "lr_trace_equal": lr_equal, "per_epoch_equal": bool(pair.get("per_epoch_equal")),
                   "resumed_from_next_epoch": resumed_ok, "epochs_ran_full": epochs_ok},
        "passed": bool(pair.get("adapter_bit_identical") and counts_equal and lr_equal and pair.get("per_epoch_equal")
                       and resumed_ok and epochs_ok),
        "note": "비트 동일이 서지 않으면 최대 절대차를 적고 허용오차를 그 실측으로 정한다(12번 §4-2 의 3). "
                "소형 모델 결과를 본실험 모델로 일반화하지 않는다.",
    }


def _rows_default(n: int) -> list[dict]:
    from vlm.pilot_vlm import PAIRS_PATH, load_pairs

    rows = load_pairs("train", "C1", pairs_path=PAIRS_PATH)
    if len(rows) < n:
        raise ValueError(f"행이 {len(rows)} 개뿐이다 — --rows {n}")
    return rows[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", nargs="?", choices=("all",) + STAGES, default="all")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--die-after", type=int, default=DEFAULT_DIE_AFTER)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if args.stage != "all":
        run_stage(args.stage, root, rows=_rows_default(args.rows), epochs=args.epochs, die_after=args.die_after)
        return 0
    if root.exists() and any(root.iterdir()):
        print(f"[t-eq-uni] 루트가 비어 있지 않다: {root} — 새 루트로 돈다", file=sys.stderr)
        return 2
    from vlm.fault import scrubbed_env

    py = [sys.executable, "-X", "utf8", str(Path(__file__).resolve())]
    common = ["--root", str(root), "--rows", str(args.rows), "--epochs", str(args.epochs),
              "--die-after", str(args.die_after)]
    env = scrubbed_env(os.environ)            # 고의 중단 변수를 물려주지 않는다
    for st in ("baseline", "baseline2"):
        subprocess.run(py + [st] + common, check=True, env=env)
    rc = subprocess.run(py + ["crash"] + common, env=env).returncode
    if rc != 9:
        print(f"[t-eq-uni] crash 단계의 종료 코드가 9 가 아니다: {rc}", file=sys.stderr)
        return 3
    left = sorted(p.name for p in (_stage_dir(root, "crash") / "resume").glob("resume_ep*.pt"))
    subprocess.run(py + ["resume"] + common, check=True, env=env)
    report = {**judge(root, epochs=args.epochs, die_after=args.die_after), "crash_exit_code": rc,
              "checkpoints_after_crash": left}
    (root / "t_eq_uni.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"],
                      "adapter_bit_identical": report["resume_pair"].get("adapter_bit_identical"),
                      "adapter_max_abs_diff": report["resume_pair"].get("adapter_max_abs_diff"),
                      "floor_max_abs_diff": report["nondeterminism_floor"].get("adapter_max_abs_diff")},
                     ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
