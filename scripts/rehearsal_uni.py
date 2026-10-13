"""통합형 배관 리허설의 오케스트레이터 — 리허설 2판 §1-2 의 단계 표(줄 A · 줄 B 의 로컬 칸) · §1-3 · §1-4 · §5-4 · §3 의 판정기.

    python -X utf8 scripts/rehearsal_uni.py run --plan configs/rehearsal_uni.yaml --snapshot <동결 스냅샷> --device <장비>
    python -X utf8 scripts/rehearsal_uni.py judge --run-root outputs/rehearsal_u/reh-YYYYMMDDTHHMMSS

| 하위 명령 | 무엇 |
|---|---|
| `run` | 루트 `outputs/rehearsal_u/reh-<시각>` 을 배타 생성하고 계획의 사본을 둔 뒤 단계 표를 하위 프로세스로 돈다 → `rehearsal_log.jsonl` |
| `lists` · `register` · `train` | 단계 0 · 0′ · 학습(1 · 2 · 2′ · 2″ · 줄 B) — `run` 이 자식으로 부른다 |
| `judge` | 조건 ① ② ④ 와 평가 쪽 산출물(③ ⑤)의 읽기 → `judge.json`. **지표 값을 싣지 않는다** |
| `fdiag prepare` | 착수 전 프레임 진단(2판 §3-1)의 흐름 1 — 진단 루트 `fdiag-<시각>` 을 배타 생성하고 목록 → 등록을 자식으로 돌린 뒤 **멈춘다** |
| `fdiag receipt` | 흐름 2 의 도구 — 본줄기에 커밋한 진단 등록을 루트의 준비 산출과 바이트로 맞대고 영수증(`kind="frame_diag"`)을 쓴다 |
| `fdiag run` | 흐름 3 — 영수증 커밋의 등록으로 루트의 목록 · 계획 · 등록을 맞대고 시도 수 · 첫 루트를 본 뒤 에코 생성 → 학습 전 에코 관문(평가 쪽 에코 전용 호출 — 시도로 세지 않는다) → 학습 → 생성 → 평가 쪽 진단 호출 → `frame_diag.json`. 관문이 통과가 아니면 학습을 시작하지 않는다 |

종료 코드 — 0 은 맞음, 2 는 거부(stderr 첫 줄이 `[사유 코드]`), 75 는 고의 중단(자식), 1 은 그 밖의 예외. `run` 은 단계가 기대와 어긋나면
그 자리에서 멈추고 1 을 낸다.

**이음새는 파이썬 인자로만**(2판 §1-4). 명령줄에는 구현을 고르는 인자가 없다 — `main(argv, measure_env=..., model_loader=..., ...)`.
기본값은 승인된 실제 구현이다. **모델 생성기의 실제 구현이 아직 없다** — 등록 단계(구현 식별자를 적는다)가 그 이음새 없이 시작하지 않는다.

**줄 B 의 연합 칸(`uni_fed`)** 은 본실험과 같은 `flwr run` 경로로 돈다(2판 §1-1 · §1-3 의 연합 행). 중단을 넣지 않는다. 학습 단계
(`train --tag uni_fed`)가 상주 SuperLink · 시뮬레이션 프로세스를 앞뒤로 내리고 남은 것이 없는지 본다. 출력은 `<루트>/train/uni_fed_s<n>/`
(2판 §13-4), 초기 어댑터는 등록 단계가 만든 `<루트>/init/initial.npz`, 학습 행은 로컬 칸의 목록이다.
계획의 `exclude.near_dup_edges` · `timeouts_s` 가 `null` 이면 시작하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXIT_OK, EXIT_HALT, EXIT_REFUSED = 0, 1, 2
NON_MAIN_PARENT_REL = Path("outputs/rehearsal_u")
SCORING_LEDGER_REL = Path("outputs/main_u/scoring_ledger.jsonl")
TIMEOUT_KEYS = ("lists", "register", "train", "export", "score")

__all__ = ["EXIT_HALT", "EXIT_OK", "EXIT_REFUSED", "TIMEOUT_KEYS", "Halt", "main", "run_rehearsal"]


class Halt(RuntimeError):
    """단계가 기대와 어긋났다 — 그 자리에서 멈춘다."""

    def __init__(self, code: str, msg: str, row: dict | None = None):
        self.code, self.row = code, row
        super().__init__(f"[{code}] {msg}")


def _refuse(code: str, msg: str) -> int:
    tag = f"[{code}] "
    while msg.startswith(tag):                  # 감싼 거부가 사유 코드를 겹쳐 싣는다 — 한 번만 쓴다
        msg = msg[len(tag):]
    print(f"{tag}{msg}", file=sys.stderr, flush=True)
    return EXIT_REFUSED


def _root_check(root: Path, plan_path: Path, parent: Path | None, purpose: str = "rehearsal") -> None:
    from vlm.rehearsal_run import StageRefused
    from vlm.run_root import run_root_problems

    bad = run_root_problems(purpose, root, [root, plan_path], parent=parent)
    if bad:
        raise StageRefused("run_root", " · ".join(bad))


# ---------------------------------------------------------------- 단계 표
def _single_receipt(root: Path) -> Path:
    found = sorted((root / "registration").glob("rehearsal-*.receipt.json"))
    if len(found) != 1:
        raise Halt("receipt_missing", f"등록 단계 뒤 영수증이 하나가 아니다: {[p.name for p in found]}")
    return found[0]


def fed_cleanup(*, procs: Callable[[], list[dict]] | None = None, stop: Callable[[list[dict]], None] | None = None) -> dict:
    """연합 단계 뒤 **살아 있는 부모 쪽** 정리 — 상주 SuperLink · 시뮬레이션 프로세스를 찾아 내리고 남은 것이 없는지 본다.
    신원이 불명확한 것은 내리지 않는다(`fl/flwr_procs.py`). 돌려주는 값은 실행 기록에 싣는 한 줄의 몸이다."""
    from fl.flwr_procs import flwr_processes, stop_processes

    procs = procs or flwr_processes
    stop = stop or stop_processes
    errors: list[str] = []                                      # 찾기 · 내리기의 예외 — 정리 실패로 적고 할 수 있는 데까지 간다
    try:
        found = procs()
    except Exception as exc:  # noqa: BLE001 - 정리 실패로 적는다
        errors.append(f"찾기: {type(exc).__name__}: {exc}"[:300])
        found = []
    unclear = [p for p in found if p.get("kind") != "target"]
    stopped: list[dict] = []
    if not unclear and found:
        try:
            stop(found)
            stopped = found
        except Exception as exc:  # noqa: BLE001 - 권한 거부 따위 — 정리 실패로 적고 잔류를 다시 본다
            errors.append(f"내리기: {type(exc).__name__}: {exc}"[:300])
    left: list[dict] | None
    try:
        left = procs()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"다시 찾기: {type(exc).__name__}: {exc}"[:300])
        left = None                                             # 잔류를 모른다 — 정리가 섰다고 하지 않는다
    return {"ok": not errors and not unclear and left == [],
            "errors": errors,
            "unclear": [{"pid": p.get("pid"), "name": p.get("name")} for p in unclear][:20],
            "stopped": [{"pid": p.get("pid"), "name": p.get("name")} for p in stopped][:20],
            "left": None if left is None else [{"pid": p.get("pid"), "name": p.get("name"), "kind": p.get("kind")}
                                               for p in left][:20]}


def with_fed_cleanup(step: Callable[[], Any], *, log: Callable[[dict], None], procs: Callable | None = None,
                     stop: Callable | None = None) -> Any:
    """연합 단계를 돌리고 **결과와 무관하게** 부모가 정리한다(외부 검토 §50 요청 1). 학습 자식 안의 정리(`finally`)는 시간 제한의
    트리 끊기(`_kill_tree`) · 비정상 사망 뒤에는 돌지 않는다 — 상주 SuperLink 는 그 트리 밖이다(2판 §1-3 의 연합 행).
    원래 단계의 실패와 정리의 실패를 나눠 적는다 — 단계가 실패했으면 그 실패를 그대로 올리고, 단계는 섰는데 정리가 실패하면 `fed_cleanup` 이다."""
    failure: BaseException | None = None
    out = None
    try:
        out = step()
    except BaseException as exc:  # noqa: BLE001 - 정리 뒤에 그대로 다시 올린다
        failure = exc
    try:
        row = fed_cleanup(procs=procs, stop=stop)
    except Exception as exc:  # noqa: BLE001 - 정리 자체가 무너져도 원래 실패를 덮지 않는다
        row = {"ok": False, "errors": [f"정리: {type(exc).__name__}: {exc}"[:300]], "unclear": [], "stopped": [], "left": None}
    try:
        log({"kind": "fed_cleanup", **row, "stage_failed": failure is not None,
             "stage_failure": str(failure)[:300] if failure is not None else None})
    except Exception:  # 기록이 실패해도 원래 실패를 먼저 올린다
        if failure is None:
            raise
    if failure is not None:
        raise failure
    if not row["ok"]:
        raise Halt("fed_cleanup", f"연합 단계 뒤 정리가 서지 않았다 — 오류 {row['errors']} · 불명 {row['unclear']} · 남음 {row['left']}")
    return out


def run_rehearsal(*, plan_path: Path, snapshot: Path, device: str, parent: Path | None = None,
                  checkout: Path | None = None, launcher: Callable[[str, Sequence[str]], list[str]] | None = None,
                  env: dict[str, str] | None = None, now: datetime | None = None,
                  exposure_ledger: Path | None = None, scoring_ledger: Path | None = None,
                  flwr_procs: Callable | None = None, flwr_stop: Callable | None = None) -> Path:
    """단계 표를 돈다. 돌려주는 값은 루트다. 어긋나면 `Halt`(실행 기록의 마지막 줄이 그 단계다)."""
    from evaluation.entry_gate import main_checkout
    from vlm.exposure import default_ledger
    from vlm.fault import FAULT_VARS
    from vlm.rehearsal_orch import (
        ANY_NONZERO,
        OrchestratorRefused,
        Stage,
        _append_log,
        default_launcher,
        export_stage_names,
        file_hashes,
        outputs_listing,
        preexisting_changes,
        run_stage,
        train_stage_names,
    )
    from vlm.rehearsal_plan import load_plan
    from vlm.rehearsal_run import LIST_FILES, StageRefused, synth_negatives

    env = dict(os.environ if env is None else env)
    hit = sorted({k for k in env if k.upper() in FAULT_VARS})
    if hit:
        raise OrchestratorRefused("fault_env", f"오케스트레이터의 환경에 고의 중단 변수가 있다: {hit} — 시작하지 않는다")
    plan = load_plan(plan_path, purpose="rehearsal")
    timeouts = plan.require("timeouts_s")
    plan.require("exclude.near_dup_edges")
    missing = [k for k in TIMEOUT_KEYS if not isinstance(timeouts, dict) or not isinstance(timeouts.get(k), (int, float))]
    if missing:
        raise StageRefused("timeouts", f"계획의 timeouts_s 에 단계 상한이 없다: {missing}")
    checkout = Path(checkout) if checkout is not None else main_checkout()
    parent = Path(parent) if parent is not None else checkout / NON_MAIN_PARENT_REL
    ledger = Path(exposure_ledger) if exposure_ledger is not None else default_ledger()
    sledger = Path(scoring_ledger) if scoring_ledger is not None else checkout / SCORING_LEDGER_REL
    now = now or datetime.now().astimezone()
    root = parent / f"reh-{now:%Y%m%dT%H%M%S}"
    parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError:
        raise OrchestratorRefused("root_exists", f"루트가 이미 있다: {root}") from None
    P = root / "plan.yaml"
    with open(P, "xb") as fh:
        fh.write(plan.raw)
    launcher = launcher or default_launcher
    outputs = checkout / "outputs"
    listing_before = outputs_listing(outputs, skip_dirs=[root], skip_files=[ledger, sledger])
    prefix = {name: (p.read_bytes() if p.exists() else b"") for name, p in (("exposure", ledger), ("scoring", sledger))}
    import hashlib

    _append_log(root, {"kind": "start", "plan_sha256": plan.sha256, "snapshot": str(snapshot), "device": device,
                       "outputs_listing_n": len(listing_before),
                       "ledger_prefix": {k: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                                         for k, v in prefix.items()}})
    seq = [0]

    def go(stage: Stage, extra: dict | None = None):
        seq[0] += 1
        res = run_stage(stage, root=root, seq=seq[0], launcher=launcher, env_base=env,
                        timeout_s=float(timeouts[stage.timeout_key]), extra=extra)
        if not res.ok:
            raise Halt("stage_mismatch", f"단계 {stage.no} {stage.name} 이 기대와 다르다 — 종료 {res.exit_code}", res.row)
        return res

    R, S = str(root), str(snapshot)
    idx = str(int(plan.get("seed_index")))
    lists = root / "lists"
    go(Stage("0", "lists", "lists", ("--run-root", R, "--plan", str(P), "--snapshot", S), 0, "lists"))
    go(Stage("0'", "register", "register", ("--run-root", R, "--plan", str(P)), 0, "register"))
    RC = str(_single_receipt(root))
    gen, echo = str(lists / LIST_FILES["gen"]), str(lists / LIST_FILES["echo"])
    exp = ("--purpose", "rehearsal", "--seed-index", idx, "--receipt", RC, "--snapshot", S, "--run-root", R,
           "--plan", str(P), "--folder", str(root / "export"), "--device", device)
    score = ("--registration", "probe", "--purpose", "rehearsal", "--receipt", RC, "--snapshot", S,
             "--bundles", str(root / "export"), "--generation-list", gen, "--scoring-list", gen, "--echo-list", echo,
             "--train-root", str(root / "train"), "--plan", str(P), "--device", device)
    go(Stage("E", "echo_export", "export", exp + ("--mode", "echo", "--tag", "uni_central", "--list", echo), 0,
             "export", tag="uni_central"))
    # E′ — 모델 묶음이 아직 없어 평가 쪽 상태는 fail(종료 2)이다. 학습 착수는 산출의 에코 카나리아로 가른다.
    go(Stage("E'", "echo_score", "score", score + ("--out", str(root / "echo_gate")), 2, "score"))
    gate = json.loads((root / "echo_gate" / "score" / "score_unified_rehearsal.json").read_text(encoding="utf-8"))
    from vlm.rehearsal_judge import echo_identity_ok

    entries = gate.get("canaries", {}).get("echo", [])
    states = [e.get("status") for e in entries]
    identity = [echo_identity_ok(e) for e in entries]                 # 기대 박스 수 > 0 과 등식들 — 불리언만 적는다
    _append_log(root, {"kind": "echo_gate", "statuses": states, "identity": identity})
    if not entries or not all(identity):
        raise Halt("echo_gate", f"에코 카나리아가 항등 검사를 지나지 않았다: {states} — 학습을 시작하지 않는다")

    faults = plan.get("faults") or {}
    tfaults, efaults = faults.get("train") or {}, faults.get("export") or {}
    train = ("--run-root", R, "--plan", str(P), "--snapshot", S)

    def train_cell(tag: str, no: str) -> None:
        f = tfaults.get(tag)
        args = train + ("--tag", tag)
        names = train_stage_names(tag, f is not None)
        if f is not None:
            go(Stage(no, names[0], "train", args, 0, "train", fault=f, tag=tag))
            if tag == "uni_central":
                src = sorted((root / "resume" / f"{tag}_s{idx}").glob("resume_ep*.pt"))
                if not src:
                    raise Halt("negative_source", "중단 뒤 재개 체크포인트가 없다 — 합성 구판을 만들 수 없다")
                made = synth_negatives(root, tag=tag, seed_index=int(idx), source=src[0])
                _append_log(root, {"kind": "negatives_made", "source": src[0].relative_to(root).as_posix(),
                                   "cases": {k: {"path": Path(v["path"]).relative_to(root).as_posix(),
                                                 "sha256": v["sha256"], "changed": v["changed"]}
                                             for k, v in made.items()}})
        go(Stage("2" if f is not None and tag == "uni_central" else no, names[-1], "train", args, 0, "train",
                 tag=tag))

    def export_cell(tag: str, first_no: int | None) -> list[str]:
        attempts = efaults.get(tag) or [{"to_end": True}]
        names = export_stage_names(tag, len(attempts))
        base = exp + ("--mode", "model", "--tag", tag, "--list", gen,
                      "--adapter-dir", str(root / "train" / f"{tag}_s{idx}"))
        last = base
        for k, att in enumerate(attempts):
            no = str(first_no + k) if first_no is not None else "B"
            if "fault" in att:
                go(Stage(no, names[k], "export", base, 0, "export", fault=att["fault"], tag=tag))
            elif "stop_after_lines" in att:
                last = base + ("--stop-after-lines", str(int(att["stop_after_lines"])))
                go(Stage(no, names[k], "export", last, 0, "export", tag=tag))
                last = base
            else:
                go(Stage(no, names[k], "export", base, 0, "export", tag=tag))
        return list(last)

    # 줄 A — 학습 시도 1 · 재개 · 구판 거부 · 덮어쓰기 거부
    train_cell("uni_central", "1")
    for case in plan.get("negatives") or []:
        go(Stage("2'", f"negative_{case}", "train", train + ("--tag", "uni_central", "--negative", case),
                 ANY_NONZERO, "train", tag="uni_central"))
    before = file_hashes(root)
    go(Stage("2''", "overwrite_same_stamp", "train", train + ("--tag", "uni_central"), 0, "train", tag="uni_central"))
    go(Stage("2''", "overwrite_other_stamp", "train",
             train + ("--tag", "uni_central", "--stamp", f"{root.name}_other"), ANY_NONZERO, "train", tag="uni_central"))
    _append_log(root, {"kind": "hashes", "around": "2''", "before": before, "after": file_hashes(root)})
    # 줄 A — 생성 시도 넷 · 재실행 거부
    last = export_cell("uni_central", 3)
    before = file_hashes(root)
    go(Stage("6'", "rerun_uni_central", "export", tuple(last), 2, "export", tag="uni_central"))
    _append_log(root, {"kind": "hashes", "around": "6'", "before": before, "after": file_hashes(root)})
    # 줄 B — 로컬 셋, 그리고 연합(flwr run 경로 — 중단 없음)
    for tag in [t for t in plan.get("cells") if t not in ("uni_central", "uni_fed")]:
        train_cell(tag, "B")
        export_cell(tag, None)
    if "uni_fed" in plan.get("cells"):
        # 부모가 정리한다 — 학습 자식이 시간 제한으로 끊겨도(그 안의 정리가 돌지 않아도) 상주 프로세스를 내린다(외부 검토 §50 요청 1)
        with_fed_cleanup(lambda: train_cell("uni_fed", "B"), log=lambda row: _append_log(root, row),
                         procs=flwr_procs, stop=flwr_stop)
        export_cell("uni_fed", None)
    go(Stage("7", "score", "score", score + ("--out", R), (0, 2), "score"))
    after = outputs_listing(outputs, skip_dirs=[root], skip_files=[ledger, sledger])
    # 루트 밖 — 실행 전에 있던 파일이 크기 · mtime_ns 그대로인가(2판 §3 ④). 새로 생긴 파일(노출 목록 사본 등)은 따로 적는다(외부 검토 2)
    changed_pre = preexisting_changes(listing_before, after)
    _append_log(root, {"kind": "end", "outputs_preexisting_preserved": not changed_pre,
                       "outputs_preexisting_changed": changed_pre[:50],
                       "outputs_new": sorted(set(after) - set(listing_before))[:50],
                       "outputs_new_n": len(set(after) - set(listing_before)),
                       "ledger_prefix_kept": {k: (p.read_bytes() if p.exists() else b"").startswith(prefix[k])
                                              for k, p in (("exposure", ledger), ("scoring", sledger))}})
    return root


# ---------------------------------------------------------------- 착수 전 프레임 진단(2판 §3-1)
FDIAG_FILE = "frame_diag.json"
D_DIAG_FILES = ("score/frame_diag_score.json", "score/frame_diag_not_judged.json")
#: 평가 쪽 진단 산출물에서 옮겨 싣는 칸 — 상태 · 후보 · 사유와 표본 수뿐이다. 기울기 · 비 · 표준오차의 값은 싣지 않는다(2판 §3-1 의 산출).
D_DIAG_KEEP = ("status", "candidate", "reason", "n_images", "n_pairs", "n_no_pred", "n_count_mismatch", "n_groups",
               "n_image_groups", "n_pair_groups", "n_limit_no_pair", "samples")
NOT_JUDGED = "판정하지 않음"
#: 장비 이름의 꼴 — `{이름}:{번호}`. 평가 쪽 묶음 검사가 시작 도장 · 시도 줄의 장비를 이 꼴로 본다(§28-3 나). 진단은 시도가 한 번이라
#: 꼴이 틀린 장비로 학습 · 생성까지 간 뒤 평가 쪽이 묶음을 거부하면 그 시도를 잃는다 — 시작 전에 본다.
DEVICE_FORM = re.compile(r"[^:]+(?::[^:]+)*:[0-9]+")


def _env_clean(env: dict[str, str]) -> None:
    from vlm.fault import FAULT_VARS
    from vlm.rehearsal_orch import OrchestratorRefused

    hit = sorted({k for k in env if k.upper() in FAULT_VARS})
    if hit:
        raise OrchestratorRefused("fault_env", f"오케스트레이터의 환경에 고의 중단 변수가 있다: {hit} — 시작하지 않는다")


def _diag_plan(plan_path: Path):
    from vlm.rehearsal_plan import load_plan
    from vlm.rehearsal_run import StageRefused

    plan = load_plan(plan_path, purpose="frame_diag")
    timeouts = plan.require("timeouts_s")
    plan.require("exclude.near_dup_edges")
    missing = [k for k in TIMEOUT_KEYS if not isinstance(timeouts, dict) or not isinstance(timeouts.get(k), (int, float))]
    if missing:
        raise StageRefused("timeouts", f"계획의 timeouts_s 에 단계 상한이 없다: {missing}")
    return plan, timeouts


def _stager(root: Path, timeouts: dict, launcher: Callable, env: dict[str, str]):
    """단계를 자식으로 돌리는 함수 — 같은 루트에서 다시 돌면 앞 실행의 로그 번호 뒤에서 잇는다(로그 파일은 배타 생성이다)."""
    from vlm.rehearsal_orch import run_stage

    logs = root / "logs"
    seq = [len(list(logs.glob("*.stdout"))) if logs.is_dir() else 0]

    def go(stage):
        seq[0] += 1
        res = run_stage(stage, root=root, seq=seq[0], launcher=launcher, env_base=env,
                        timeout_s=float(timeouts[stage.timeout_key]))
        if not res.ok:
            raise Halt("stage_mismatch", f"단계 {stage.no} {stage.name} 이 기대와 다르다 — 종료 {res.exit_code}", res.row)
        return res
    return go


def prepare_diag(*, plan_path: Path, snapshot: Path, rules_rel: str, parent: Path | None = None,
                 checkout: Path | None = None, launcher: Callable | None = None, env: dict[str, str] | None = None,
                 now: datetime | None = None) -> Path:
    """흐름 1 — 진단 루트를 배타 생성하고 계획 사본을 둔 뒤 목록(노출 원장의 `planned` 줄이 먼저) → 등록을 자식으로 돌리고 멈춘다.
    규칙 파일은 본줄기의 커밋에서 읽는다 — 루트를 만들기 전에 본다. 돌려주는 값은 루트다."""
    from evaluation.entry_gate import main_checkout
    from vlm.rehearsal_orch import OrchestratorRefused, Stage, _append_log, default_launcher
    from vlm.rehearsal_run import StageRefused, diag_rules_bytes

    env = dict(os.environ if env is None else env)
    _env_clean(env)
    plan, timeouts = _diag_plan(plan_path)
    from vlm.rehearsal_lists import ListsRefused, diag_hold

    try:
        diag_hold(plan)                                          # 진단은 보류 — 루트를 만들기 전에(결정 21 의 4)
    except ListsRefused as exc:
        raise StageRefused(exc.code, str(exc)) from None
    checkout = Path(checkout) if checkout is not None else main_checkout()
    rules_raw = diag_rules_bytes(checkout, rules_rel)
    parent = Path(parent) if parent is not None else checkout / NON_MAIN_PARENT_REL
    now = now or datetime.now().astimezone()
    root = parent / f"fdiag-{now:%Y%m%dT%H%M%S}"
    parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError:
        raise OrchestratorRefused("root_exists", f"루트가 이미 있다: {root}") from None
    P = root / "plan.yaml"
    with open(P, "xb") as fh:
        fh.write(plan.raw)
    import hashlib

    _append_log(root, {"kind": "fdiag_prepare", "plan_sha256": plan.sha256, "snapshot": str(snapshot),
                       "rules_path": rules_rel, "rules_sha256": hashlib.sha256(rules_raw).hexdigest()})
    go = _stager(root, timeouts, launcher or default_launcher, env)
    R = str(root)
    go(Stage("0", "lists", "lists", ("--run-root", R, "--plan", str(P), "--snapshot", str(snapshot)), 0, "lists"))
    go(Stage("0'", "register", "register", ("--run-root", R, "--plan", str(P), "--rules", rules_rel), 0, "register"))
    regs = sorted(p for p in (root / "registration").glob("frame_diag-*.json")
                  if not p.name.endswith((".train_config.json", ".receipt.json")))
    if len(regs) != 1:
        raise Halt("registration_missing", f"등록 단계 뒤 진단 등록이 하나가 아니다: {[p.name for p in regs]}")
    _append_log(root, {"kind": "fdiag_prepared", "registration": regs[0].name,
                       "registration_file_sha256": hashlib.sha256(regs[0].read_bytes()).hexdigest()})
    return root


def write_diag_receipt(*, root: Path, registration_rel: str, commit: str, checkout: Path | None = None,
                       parent: Path | None = None, now: datetime | None = None) -> Path:
    """흐름 2 의 도구 — 본줄기에 커밋한 진단 등록과 학습 설정 원문이 루트의 준비 산출과 **같은 바이트**일 때만 영수증을 쓴다.
    커밋은 본줄기(`refs/heads/main`)의 조상이어야 한다. 손으로 해시를 옮겨 적는 자리를 없앤다. 돌려주는 값은 영수증 경로다."""
    import hashlib
    import subprocess

    from evaluation.entry_gate import EntryRejected, blob_at, is_ancestor, main_checkout
    from evaluation.prereg_unified import (
        RegistrationFileError,
        RegistrationIncomplete,
        RegistrationInvalid,
        load_registration,
    )
    from vlm.export_preflight import TRAIN_CONFIG_SUFFIX, echo_train_config_rel
    from vlm.rehearsal_run import StageRefused, purpose_of_root

    root = Path(root)
    checkout = Path(checkout) if checkout is not None else main_checkout()
    if purpose_of_root(root) != "frame_diag":
        raise StageRefused("run_root", "영수증은 진단 루트(fdiag-*)에만 쓴다")
    _root_check(root, root / "plan.yaml", parent, "frame_diag")
    full = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--verify", f"{commit}^{{commit}}"],
                          capture_output=True, text=True, check=False)
    if full.returncode != 0 or not is_ancestor(checkout, full.stdout.strip()):
        raise StageRefused("receipt_not_ancestor", f"커밋 {commit} 이 본줄기의 조상이 아니다")
    sha = full.stdout.strip()
    try:
        raw = blob_at(checkout, sha, registration_rel)
        tc = blob_at(checkout, sha, echo_train_config_rel(registration_rel))
    except EntryRejected as exc:
        raise StageRefused("receipt_blob", str(exc)) from None
    stem = registration_rel.rsplit("/", 1)[-1]
    if not stem.startswith("frame_diag-") or not stem.endswith(".json"):
        raise StageRefused("receipt_blob", f"진단 등록 파일 이름이 frame_diag-*.json 이 아니다: {registration_rel}")
    stem = stem[: -len(".json")]
    d = root / "registration"
    local = d / f"{stem}.json"
    if not local.is_file() or local.read_bytes() != raw:
        raise StageRefused("registration_vs_root", "커밋한 등록이 루트의 준비 산출과 다르다")
    tc_local = d / f"{stem}{TRAIN_CONFIG_SUFFIX}"
    if not tc_local.is_file() or tc_local.read_bytes() != tc:
        raise StageRefused("train_config_vs_root", "커밋한 학습 설정 원문이 루트의 준비 산출과 다르다")
    try:
        reg = load_registration(raw)
        reg.require_complete(side="generation")
    except (RegistrationFileError, RegistrationIncomplete, RegistrationInvalid) as exc:
        raise StageRefused("registration_form", str(exc)) from None
    if reg.generation.purpose != "frame_diag":
        raise StageRefused("registration_form", "커밋한 등록의 목적이 frame_diag 가 아니다")
    now = now or datetime.now().astimezone()
    receipt = {"generation_sha256": reg.generation_sha256(), "scoring_sha256": reg.scoring_sha256(),
               "registered_at": now.isoformat(timespec="microseconds"), "main_commit": sha, "kind": "frame_diag",
               "registration_path": registration_rel, "registration_file_sha256": hashlib.sha256(raw).hexdigest()}
    rp = d / f"{stem}.receipt.json"
    with open(rp, "xb") as fh:
        fh.write((json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return rp


def _diag_attempts(ledger: Path, gen_sha: str) -> list[dict]:
    """채점 원장의 이 진단 등록의 호출(시도)과 그 판정 상태 — 모든 시도를 함께 싣는다(판정 07 의 19)."""
    from evaluation.scoring_ledger import read_ledger

    if not ledger.is_file():
        return []
    rows = read_ledger(ledger)
    verdicts = {r.get("call_sha256"): r.get("status") for _s, r in rows if r["kind"] == "frame_diag_verdict"}
    return [{"call_sha256": s, "verdict_status": verdicts.get(s)} for s, r in rows
            if r["kind"] == "call" and r.get("purpose") == "frame_diag" and r.get("generation_sha256") == gen_sha]


BINDINGS_DIR = "fdiag_bindings"
"""진단 등록과 그 등록을 처음 돌린 루트의 결속 — 비본실험 부모 아래, 루트 **밖**이다(2판 §3-1 의 시도의 뜻 — 루트 안에서 세면 새 루트마다 0)."""


ECHO_GATE_FILE = "score/echo_gate.json"
#: 묶음 생성 파일의 이름 — 평가 쪽이 묶음을 고르는 꼴(`scripts/probe/score_unified.py` 의 `_NAME`)과 같다. 쓰는 쪽도 이 꼴의 파일만 고른다 —
#: 꼴 밖의 파일(예 `backup.echo.generations.jsonl`)은 평가 쪽이 보지 않으므로 이 쪽도 읽지 않는다(외부 검토 회신 `echowire` 의 1). 시험이 두 꼴을 맞댄다.
GEN_NAME = re.compile(r"^(?P<tag>[A-Za-z0-9_]+)_s(?P<seed>\d+)(?P<echo>\.echo)?\.generations\.jsonl$")


def _bundle_files(root: Path, *, echo: bool | None = None) -> list[Path]:
    """`<루트>/export` 에서 평가 쪽과 같은 규칙으로 고른 생성 파일(평가 쪽 `_bundle_files` 와 같다). `echo` 가 참이면 에코만, 거짓이면 모델만."""
    d = Path(root) / "export"
    if not d.is_dir():
        return []
    out = []
    for q in sorted(d.iterdir()):
        m = GEN_NAME.match(q.name)
        if q.is_file() and m and (echo is None or bool(m["echo"]) == echo):
            out.append(q)
    return out


def _sha_under(q: Path, root: Path, *, code: str) -> str:
    """루트 아래 파일의 sha256 — `read_under` 와 같은 보호(최종 경로 · 보통 파일 · 이름 하나)로 읽는다."""
    import hashlib

    from vlm.rehearsal_run import read_under

    return hashlib.sha256(read_under(q, root, code=code)).hexdigest()


def _diag_out_rel(root: Path, checkout: Path) -> str:
    """평가 쪽 판정 산출의 자리(체크아웃 상대) — **파일 자체는 풀지 않는다**(푼 루트 + 이름). 그 파일이 밖으로 가는 링크여도 여기서 예외로 끝나지
    않고, 뒤의 보호된 읽기(`read_under`)가 사유 코드로 거부한다. 루트가 체크아웃 밖이면 `fdiag_output_path`(외부 검토 회신 `echorule` 의 4 · §59)."""
    from vlm.rehearsal_run import StageRefused

    try:
        return (Path(root).resolve() / D_DIAG_FILES[0]).relative_to(Path(checkout).resolve()).as_posix()
    except ValueError:
        raise StageRefused("fdiag_output_path", f"루트 {root} 가 체크아웃 {checkout} 아래가 아니다 — 평가 쪽 산출의 자리를 정할 수 없다") from None


def _score_dir_ok(root: Path) -> None:
    """`<루트>/score` 가 있으면 루트 아래의 보통 폴더여야 한다 — 밖으로 가는 정션 · 링크면 평가 쪽을 부르기 전에 사유 코드로 거부한다
    (외부 검토 회신 `echowire` 의 2 — 경로를 풀다가 예외로 끝나지 않게)."""
    from vlm.rehearsal_run import StageRefused

    s = Path(root) / "score"
    if not (s.exists() or s.is_symlink()):
        return
    if s.is_symlink() or not s.is_dir() or os.path.normcase(str(s.resolve())) != os.path.normcase(str(s)):
        raise StageRefused("fdiag_output_path", f"{s} 가 루트 아래의 보통 폴더가 아니다(정션 · 링크) — 평가 쪽 산출의 자리로 쓰지 않는다")


def d_echo_gate(*, root: Path, receipt: Path, snapshot: Path, device: str, go: Callable, gen_sha: str,
                registration_file_sha256: str) -> dict:
    """**승인된 실제 구현** — 학습 전 에코 관문. 평가 쪽의 **에코 전용 호출**(`--echo-only` — 채점 원장에 아무것도 쓰지 않아 시도로 세지 않는다)을
    자식 단계로 부르고 그 산출 `score/echo_gate.json` 을 읽는다. 인자 계약은 평가 쪽 13번 보고(지시 2026-10-05f 절)다 — 종료 0(통과) · 2(통과가 아님)는
    정상 종료로 받고, 3(진입 전 거부)은 단계 어긋남으로 멈춘다. 산출은 이 루트 아래의 보통 파일이고(`read_under`), 이 등록 · 이 루트의 에코 묶음 ·
    목록 · 영수증을 본 것일 때만 받는다."""

    from vlm.rehearsal_orch import Stage
    from vlm.rehearsal_run import LIST_FILES, StageRefused, read_under

    root = Path(root)
    lists = root / "lists"
    gen, echo = str(lists / LIST_FILES["gen"]), str(lists / LIST_FILES["echo"])
    go(Stage("E'", "fdiag_echo_gate", "score",
             ("--echo-only", "--registration", "probe", "--purpose", "frame_diag", "--receipt", str(receipt),
              "--snapshot", str(snapshot), "--bundles", str(root / "export"), "--generation-list", gen,
              "--scoring-list", gen, "--echo-list", echo, "--train-root", str(root / "train"),
              "--plan", str(root / "plan.yaml"), "--out", str(root), "--device", device), (0, 2), "score"))
    out = root / ECHO_GATE_FILE
    if not (out.exists() or out.is_symlink()):
        raise Halt("echo_gate_no_output", "평가 쪽 에코 전용 호출 뒤 산출이 없다 — 학습을 시작하지 않는다")
    body = json.loads(read_under(out, root, code="echo_gate_path").decode("utf-8"))

    def sha(q: Path) -> str:
        return _sha_under(q, root, code="echo_gate_path")

    fp = body.get("fingerprints") or {}
    inputs = body.get("inputs") or {}
    want = {q.name: sha(q) for q in _bundle_files(root, echo=True)}
    want |= {"generation_list": sha(lists / LIST_FILES["gen"]), "echo_list": sha(lists / LIST_FILES["echo"]),
             "receipt": sha(Path(receipt))}
    if body.get("kind") != "echo_gate" or body.get("purpose") != "frame_diag" \
            or fp.get("generation_sha256") != gen_sha or fp.get("registration_file_sha256") != registration_file_sha256 \
            or inputs != want:
        raise StageRefused("echo_gate_provenance", "에코 관문 산출이 이 등록 · 이 루트의 에코 묶음 · 목록 · 영수증을 본 것이 아니다 — 학습을 시작하지 않는다")
    return {"ok": body.get("status") == "pass", "detail": body.get("reasons")}


def _root_id(root: Path) -> str:
    """결속의 신원 — 정션 · 링크를 푼 **실제 루트의 경로**다. 끝 이름은 신원이 아니다(같은 이름의 별칭이 다른 준비 사본을 가리킬 수 있다)."""
    return Path(root).resolve().as_posix()


def _same_root(stored: str | None, root: Path) -> bool:
    return stored is not None and os.path.normcase(stored) == os.path.normcase(_root_id(root))


def _bindings_dir(parent: Path, *, create: bool) -> Path | None:
    """결속 폴더 — **해석된 비본실험 부모 바로 아래의 보통 폴더**여야 한다(외부 검토 §56). 부모를 푼 뒤(`outputs` 의 공유 정션은 그대로 받는다)
    그 아래 `fdiag_bindings` 가 정션 · 링크로 다른 자리를 가리키면 읽지도 만들지도 않고 거부한다. 없으면 `create` 일 때만 만든다(그 뒤에도 다시 본다)."""
    from vlm.rehearsal_run import StageRefused

    d = Path(parent).resolve() / BINDINGS_DIR
    if not (d.exists() or d.is_symlink()):
        if not create:
            return None
        try:
            d.mkdir()
        except FileExistsError:                                  # 끊긴 정션 · 링크 — 아래에서 거부한다
            pass
    if d.is_symlink() or not d.is_dir() or os.path.normcase(str(d.resolve())) != os.path.normcase(str(d)):
        raise StageRefused("fdiag_binding_path", f"결속 폴더 {d} 가 비본실험 부모 아래의 보통 폴더가 아니다(정션 · 링크) — 읽지도 쓰지도 않는다")
    return d


def _read_binding(d: Path, gen_sha: str) -> str | None:
    """결속 파일의 첫 루트 — 파일은 `read_under` 와 같은 보호(최종 경로가 결속 폴더 아래 · 보통 파일 · 이름 하나)로 읽는다. 없으면 `None`."""
    from vlm.rehearsal_run import read_under

    p = d / f"{gen_sha}.json"
    if not (p.exists() or p.is_symlink()):
        return None
    return json.loads(read_under(p, d, code="fdiag_binding_path").decode("utf-8")).get("run_root")


def _other_root(parent: Path, gen_sha: str, root: Path) -> None:
    """이 진단 등록을 **다른 루트**가 먼저 돌렸으면 거부한다 — 다른 검사보다 먼저 본다(재개만 — 2판 §3-1). 루트는 푼 경로로 견준다."""
    from vlm.rehearsal_run import StageRefused

    d = _bindings_dir(parent, create=False)
    first = _read_binding(d, gen_sha) if d is not None else None
    if first is not None and not _same_root(first, root):
        raise StageRefused("fdiag_other_root", f"이 진단 등록은 루트 {first} 에서 처음 돌았다 — 그 루트에서 재개만 한다(새 루트는 새 시도다)")


def _bind_root(parent: Path, gen_sha: str, root: Path, receipt_sha: str) -> bool:
    """진단 등록 → 처음 돌린 루트. 처음이면 배타 생성하고 참, 이미 이 루트면 거짓, **다른 루트면 거부**(재개만 — 2판 §3-1).
    결속 폴더 · 파일의 자리는 만들기 전과 뒤에 본다(§56)."""
    from vlm.rehearsal_run import StageRefused, read_under

    d = _bindings_dir(parent, create=True)
    p = d / f"{gen_sha}.json"
    body = {"generation_sha256": gen_sha, "run_root": _root_id(root), "receipt_sha256": receipt_sha,
            "at": datetime.now().astimezone().isoformat(timespec="microseconds")}
    try:
        with open(p, "xb") as fh:                                # 배타 생성 — 그 자리에 링크가 있으면(끊긴 것도) 만들지 않는다
            fh.write((json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        read_under(p, d, code="fdiag_binding_path")              # 만든 뒤에도 그 자리인지 본다
        return True
    except FileExistsError:
        first = _read_binding(d, gen_sha)
    if not _same_root(first, root):
        raise StageRefused("fdiag_other_root", f"이 진단 등록은 루트 {first} 에서 처음 돌았다 — 그 루트에서 재개만 한다(새 루트는 새 시도다)")
    return False


def _d_output(root: Path, d_out: Path, *, gen_sha: str, ledger: Path, checkout: Path) -> tuple[dict, bytes]:
    """평가 쪽 산출물이 **이 진단의 것인지** — 이름만 보고 받지 않는다(외부 검토 회신 `fdiag` 의 1).

    1. 최종 경로가 이 루트 아래인 보통 파일이다(`read_under`).
    2. 본문의 `generation_sha256` 이 이 등록이다.
    3. 판정 산출물이면 채점 원장에 그 판정 줄(`verdict_line_sha256`)이 있고, 본문이 **그 줄의 값 그대로**(종류 · 시각 · 앞 줄 sha 를 뺀 것)이며,
       줄의 `output` 이 이 파일 자리다. 판정하지 않은 산출물이면 종류가 `frame_diag_not_judged` 다.
    4. 그 산출물을 낸 `call` 줄이 원장에 있고(목적 `frame_diag` · 이 등록), 그 줄이 본 묶음의 생성 파일 해시가 이 루트의 생성 파일과 같다.
    """

    from evaluation.scoring_ledger import LedgerChainBroken, read_ledger
    from vlm.rehearsal_run import StageRefused, read_under

    def bad(msg: str):
        raise StageRefused("fdiag_output_provenance", f"{d_out.name}: {msg} — 이 진단의 결과로 받지 않는다")

    raw = read_under(d_out, root, code="fdiag_output_path")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        bad("JSON 이 아니다")
    if not isinstance(body, dict) or body.get("generation_sha256") != gen_sha:
        bad("다른 등록의 산출물이다")
    try:
        rows = dict(read_ledger(ledger)) if ledger.is_file() else {}
    except LedgerChainBroken as exc:
        bad(f"채점 원장을 읽지 못했다: {exc}")
    if d_out.name == "frame_diag_score.json":
        vsha = body.get("verdict_line_sha256")
        row = rows.get(vsha)
        if row is None or row.get("kind") != "frame_diag_verdict":
            bad("채점 원장에 그 판정 줄이 없다")
        want = {k: v for k, v in row.items() if k not in ("kind", "at", "prev_sha256")} | {"verdict_line_sha256": vsha}
        if body != want:
            bad("판정 줄의 값과 다르다")
        if row.get("output") != _diag_out_rel(root, checkout):          # d_out 은 위의 보호된 읽기를 지났다
            bad("판정 줄이 가리키는 산출물 자리가 이 파일이 아니다")
        call_sha = row.get("call_sha256")
    else:
        if body.get("kind") != "frame_diag_not_judged":
            bad("판정하지 않은 산출물의 종류가 아니다")
        call_sha = body.get("call_sha256")
    call = rows.get(call_sha)
    if call is None or call.get("kind") != "call" or call.get("purpose") != "frame_diag" \
            or call.get("generation_sha256") != gen_sha:
        bad("채점 원장에 그 호출이 없다")
    mine = sorted(_sha_under(q, root, code="fdiag_output_path") for q in _bundle_files(root))
    if sorted(str(b.get("generations_sha256")) for b in call.get("bundles") or []) != mine:
        bad("그 호출이 본 묶음이 이 루트의 생성 파일이 아니다")
    return body, raw


def _verdict_waiting(ledger: Path, *, gen_sha: str, out_rel: str) -> bool:
    """판정 줄은 원장에 있는데 산출물이 없는가 — 평가 쪽이 줄을 쓴 직후 죽은 꼴. 평가 쪽은 그 줄에서 산출물을 다시 만든다(새 시도 없이)."""
    from evaluation.scoring_ledger import read_ledger

    if not ledger.is_file():
        return False
    return any(r.get("kind") == "frame_diag_verdict" and r.get("generation_sha256") == gen_sha and r.get("output") == out_rel
               for _s, r in read_ledger(ledger))


def run_diag(*, root: Path, receipt: Path, snapshot: Path, device: str, parent: Path | None = None,
             checkout: Path | None = None, launcher: Callable | None = None, env: dict[str, str] | None = None,
             scoring_ledger: Path | None = None, echo_gate: Callable | None = None) -> dict:
    """흐름 3 — 영수증 커밋의 등록으로 루트를 맞대고 시도 수를 센 뒤 에코 생성 → **에코 관문** → 학습 → 생성 → 평가 쪽 진단 호출 → `frame_diag.json`.

    **모델을 올리기 전에** 거부한다 — 루트의 목록 · 계획 · 학습 행 · 등록이 등록과 다르다 · 이 등록을 **다른 루트**가 먼저 돌렸다 ·
    학습 전 에코 관문이 서지 않는다 · 루트에 남은 평가 쪽 산출물이 이 진단의 것이 아니다 · 평가 쪽 호출 수가 허용 시도 수에 닿았다.
    같은 루트에서 다시 돌면 닫힌 단계는 건너뛴다(재개만). 이 진단의 산출물이 남았으면 옮겨 싣기만 하고, 판정 줄만 남고 산출물이 없으면
    평가 쪽 진단 호출 하나만 다시 불러 그 줄에서 산출물을 다시 만든다(학습 · 생성 · 판정 계산을 다시 하지 않는다 — 평가 쪽은 새 시도로 세지 않는다).
    `echo_gate` 는 시험의 이음새다(파이썬 인자로만) — 커밋된 영수증과 함께 쓰면 거부한다.
    """
    import hashlib

    from evaluation.entry_gate import main_checkout
    from evaluation.eval_list import validate_eval_list
    from evaluation.scoring_ledger import count_calls
    from vlm.export_writer import stem_of
    from vlm.pilot_vlm import train_rows_digest
    from vlm.rehearsal_orch import OrchestratorRefused, Stage, _append_log, default_launcher
    from vlm.rehearsal_run import (
        CENTRAL_TAG,
        LIST_FILES,
        RECORD_FILE,
        TRAIN_LIST,
        StageRefused,
        diag_registration,
        purpose_of_root,
        read_under,
    )

    env = dict(os.environ if env is None else env)
    _env_clean(env)
    if not DEVICE_FORM.fullmatch(device):
        raise StageRefused("device_form", f"장비는 이름과 번호를 함께 적는다(예 cuda:0): {device!r}")
    root = Path(root).resolve()                                   # 별칭(정션 · 링크)을 푼 실제 루트 — 아래 검사 · 결속 · 단계가 모두 이 자리를 본다
    if purpose_of_root(root) != "frame_diag":
        raise StageRefused("run_root", "진단은 진단 루트(fdiag-*)에서만 돈다")
    P = root / "plan.yaml"
    _root_check(root, P, parent, "frame_diag")
    plan, timeouts = _diag_plan(P)
    if (root / FDIAG_FILE).exists():
        raise OrchestratorRefused("fdiag_done", f"이 루트의 진단은 끝났다: {root / FDIAG_FILE}")
    checkout = Path(checkout) if checkout is not None else main_checkout()
    reg, rc, committed = diag_registration(root, Path(receipt), checkout=checkout)
    g = reg.generation
    lists = root / "lists"
    for kind, want_file, want_set in (("gen", g.eval_list_file_sha256, g.eval_list_set_sha256),
                                      ("echo", g.echo_list_file_sha256, g.echo_list_set_sha256)):
        el = validate_eval_list(read_under(lists / LIST_FILES[kind], root, code="list_path"))
        if (el.file_sha256, el.set_sha256) != (want_file, want_set):
            raise StageRefused("lists_vs_registration", f"루트의 {kind} 목록이 등록의 해시와 다르다 — prepare 뒤에 바뀌었다")
    if plan.sha256 != g.plan_sha256:
        raise StageRefused("plan_vs_registration", "루트의 계획 파일이 등록의 plan_sha256 과 다르다")
    tname = TRAIN_LIST.format(tag=CENTRAL_TAG)
    traw = read_under(lists / tname, root, code="train_list_path")
    rec = json.loads(read_under(lists / RECORD_FILE, root, code="train_list_path").decode("utf-8"))
    if hashlib.sha256(traw).hexdigest() != (rec.get("files") or {}).get(tname) \
            or train_rows_digest([json.loads(x) for x in traw.decode("utf-8").splitlines() if x]) != g.train_rows_digest:
        raise StageRefused("train_rows_vs_registration", "루트의 학습 행 목록이 등록의 train_rows_digest 와 다르다")
    ledger = Path(scoring_ledger) if scoring_ledger is not None else checkout / SCORING_LEDGER_REL
    gen_sha = reg.generation_sha256()
    if echo_gate is not None and committed:
        raise StageRefused("seam_on_committed_receipt", "커밋된 진단 영수증으로는 에코 관문의 이음새를 쓸 수 없다")
    gate = echo_gate or d_echo_gate
    bind_parent = Path(parent) if parent is not None else checkout / NON_MAIN_PARENT_REL
    _other_root(bind_parent, gen_sha, root)
    used = count_calls(ledger, purpose="frame_diag", generation_sha256=gen_sha) if ledger.is_file() else 0
    allowed = int(plan.require("attempts_allowed"))
    d_out = next((root / q for q in D_DIAG_FILES if (root / q).exists() or (root / q).is_symlink()), None)
    _score_dir_ok(root)
    out_rel = _diag_out_rel(root, checkout)
    if d_out is not None:
        _d_output(root, d_out, gen_sha=gen_sha, ledger=ledger, checkout=checkout)    # 남은 산출물은 출처를 본 뒤에만 옮긴다
        mode = "resume_output"
    elif used >= allowed:
        if not _verdict_waiting(ledger, gen_sha=gen_sha, out_rel=out_rel):
            raise StageRefused("fdiag_attempts", f"이 등록의 진단 호출이 허용 시도 수에 닿았다({used}/{allowed}) — 새 등록이 필요하다")
        mode = "regenerate"
    else:
        mode = "run"
    receipt_sha = hashlib.sha256(Path(receipt).read_bytes()).hexdigest()
    first = _bind_root(bind_parent, gen_sha, root, receipt_sha)
    _append_log(root, {"kind": "fdiag_run", "receipt_sha256": receipt_sha, "generation_sha256": gen_sha,
                       "committed_receipt": committed, "attempts_used": used, "attempts_allowed": allowed, "mode": mode,
                       "first_run_of_registration": first})
    if mode != "resume_output":
        go = _stager(root, timeouts, launcher or default_launcher, env)
        R, S, RC = str(root), str(snapshot), str(receipt)
        idx = int(plan.get("seed_index"))
        gen, echo = str(lists / LIST_FILES["gen"]), str(lists / LIST_FILES["echo"])
        export = root / "export"
        if mode == "run":
            exp = ("--purpose", "frame_diag", "--seed-index", str(idx), "--receipt", RC, "--snapshot", S, "--run-root", R,
                   "--plan", str(P), "--folder", str(export), "--device", device)
            if not (export / f"{stem_of(CENTRAL_TAG, idx, 'echo')}.export_meta.json").is_file():
                go(Stage("E", "fdiag_echo_export", "export", exp + ("--mode", "echo", "--tag", CENTRAL_TAG, "--list", echo),
                         0, "export", tag=CENTRAL_TAG))
            # 학습 전 에코 관문 — 통과가 아니면 학습을 시작하지 않는다(2판 §3-1 · §1-2 의 E · E′)
            res = gate(root=root, receipt=Path(receipt), snapshot=Path(snapshot), device=device, go=go, gen_sha=gen_sha,
                       registration_file_sha256=rc.registration_file_sha256)
            _append_log(root, {"kind": "fdiag_echo_gate", "ok": bool(res.get("ok")), "detail": res.get("detail")})
            if not res.get("ok"):
                raise Halt("echo_gate", f"학습 전 에코 관문이 통과가 아니다: {res.get('detail')} — 학습을 시작하지 않는다")
            go(Stage("1", "fdiag_train", "train", ("--run-root", R, "--plan", str(P), "--snapshot", S, "--tag", CENTRAL_TAG,
                                                   "--receipt", RC), 0, "train", tag=CENTRAL_TAG))
            if not (export / f"{stem_of(CENTRAL_TAG, idx, 'model')}.export_meta.json").is_file():
                go(Stage("3", "fdiag_export", "export",
                         exp + ("--mode", "model", "--tag", CENTRAL_TAG, "--list", gen,
                                "--adapter-dir", str(root / "train" / f"{CENTRAL_TAG}_s{idx}")), 0, "export",
                         tag=CENTRAL_TAG))
        # 판정 · 판정 전 거부 모두 평가 쪽 호출 하나다. 다시 만들기(mode regenerate)는 같은 호출이 판정 줄에서 산출물만 되살린다
        go(Stage("D", "fdiag_score" if mode == "run" else "fdiag_score_regenerate", "score",
                 ("--registration", "probe", "--purpose", "frame_diag", "--receipt", RC, "--snapshot", S,
                  "--bundles", str(export), "--generation-list", gen, "--scoring-list", gen, "--echo-list", echo,
                  "--train-root", str(root / "train"), "--plan", str(P), "--out", R, "--device", device), (0, 2), "score"))
        d_out = next((root / q for q in D_DIAG_FILES if (root / q).exists() or (root / q).is_symlink()), None)
        if d_out is None:
            raise Halt("fdiag_no_output", "평가 쪽 진단 호출 뒤 산출물이 없다")
    body, raw = _d_output(root, d_out, gen_sha=gen_sha, ledger=ledger, checkout=checkout)
    judged = d_out.name == "frame_diag_score.json"
    out = {"kind": "frame_diag", "run_root": root.name, "generation_sha256": gen_sha,
           "d_output": {"path": d_out.relative_to(root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
                        "kind": "frame_diag_verdict" if judged else body.get("kind")},
           **({k: body.get(k) for k in D_DIAG_KEEP} if judged else {"status": NOT_JUDGED, "reason": body.get("reason")}),
           "committed_receipt": committed,
           "attempts": _diag_attempts(ledger, gen_sha), "attempts_allowed": allowed,
           "inputs": {"receipt_sha256": hashlib.sha256(Path(receipt).read_bytes()).hexdigest(),
                      "registration_file_sha256": rc.registration_file_sha256, "plan_sha256": plan.sha256,
                      "gen_list_file_sha256": g.eval_list_file_sha256, "echo_list_file_sha256": g.echo_list_file_sha256,
                      "train_rows_digest": g.train_rows_digest,
                      "generations": {q.name: _sha_under(q, root, code="fdiag_output_path") for q in _bundle_files(root)}}}
    with open(root / FDIAG_FILE, "xb") as fh:
        fh.write((json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    _append_log(root, {"kind": "fdiag_end", "status": out["status"],
                       "frame_diag_sha256": hashlib.sha256((root / FDIAG_FILE).read_bytes()).hexdigest()})
    return out


# ---------------------------------------------------------------- 명령줄
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True, type=Path)
    r.add_argument("--snapshot", required=True, type=Path)
    r.add_argument("--device", required=True)
    for name in ("lists", "register", "train", "judge"):
        s = sub.add_parser(name)
        s.add_argument("--run-root", required=True, type=Path)
        if name != "judge":
            s.add_argument("--plan", required=True, type=Path)
        if name in ("lists", "train"):
            s.add_argument("--snapshot", required=True, type=Path)
        if name == "register":
            s.add_argument("--rules", default=None, help="진단만 — 본줄기의 규칙 파일(저장소 상대 경로)")
        if name == "train":
            s.add_argument("--tag", required=True)
            s.add_argument("--stamp", default=None)
            s.add_argument("--negative", default=None)
            s.add_argument("--receipt", default=None, type=Path, help="진단만 — 진단 영수증")
    f = sub.add_parser("fdiag").add_subparsers(dest="fcmd", required=True)
    fp = f.add_parser("prepare")
    fp.add_argument("--plan", required=True, type=Path)
    fp.add_argument("--snapshot", required=True, type=Path)
    fp.add_argument("--rules", required=True)
    fr = f.add_parser("receipt")
    fr.add_argument("--run-root", required=True, type=Path)
    fr.add_argument("--registration-path", required=True)
    fr.add_argument("--commit", required=True)
    fx = f.add_parser("run")
    fx.add_argument("--run-root", required=True, type=Path)
    fx.add_argument("--receipt", required=True, type=Path)
    fx.add_argument("--snapshot", required=True, type=Path)
    fx.add_argument("--device", required=True)
    return ap


def main(argv: list[str] | None = None, *, measure_env: Callable | None = None, model_loader: Callable | None = None,
         generator_loader: Callable | None = None, checkout: Path | None = None, exposure_ledger: Path | None = None,
         scoring_ledger: Path | None = None, non_main_parent: Path | None = None, config_path: Path | None = None,
         launcher: Callable | None = None, env: dict[str, str] | None = None, now: datetime | None = None,
         standin_allowed: bool = False, fed_runner: Callable | None = None, flwr_procs: Callable | None = None,
         flwr_stop: Callable | None = None, token_counter: Callable | None = None,
         echo_gate: Callable | None = None) -> int:
    """명령줄 입구. 키워드 인자는 **시험의 이음새**다 — 명령줄로는 바꿀 수 없다.
    `fed_runner` · `flwr_procs` · `flwr_stop` 은 연합 칸의 flwr 실행 · 프로세스 찾기 · 내리기다(비면 실제 것).
    `token_counter` 는 목록 단계의 타깃 길이 셈이다(진단 계획 — 비면 프로세서의 토크나이저로 센다).
    `echo_gate` 는 진단의 학습 전 에코 관문이다(비면 실제 것 — 평가 쪽 에코 전용 호출)."""
    from evaluation.entry_gate import main_checkout
    from vlm.export_measure import measure_env as measure_real
    from vlm.rehearsal_orch import OrchestratorRefused
    from vlm.rehearsal_plan import PlanRejected, load_plan
    from vlm.rehearsal_run import (
        FED_TAG,
        StageRefused,
        purpose_of_root,
        stage_lists,
        stage_register,
        stage_train,
        stage_train_fed,
    )
    from vlm.uni_config import load_uni_config

    a = _parser().parse_args(argv)
    try:
        if a.cmd == "fdiag":
            if a.fcmd == "prepare":
                root = prepare_diag(plan_path=a.plan, snapshot=a.snapshot, rules_rel=a.rules, parent=non_main_parent,
                                    checkout=checkout, launcher=launcher, env=env, now=now)
                print(f"fdiag_prepared root={root}", flush=True)
            elif a.fcmd == "receipt":
                rp = write_diag_receipt(root=a.run_root, registration_rel=a.registration_path, commit=a.commit,
                                        checkout=checkout, parent=non_main_parent, now=now)
                print(f"fdiag_receipt receipt={rp}", flush=True)
            else:
                out = run_diag(root=a.run_root, receipt=a.receipt, snapshot=a.snapshot, device=a.device,
                               parent=non_main_parent, checkout=checkout, launcher=launcher, env=env,
                               scoring_ledger=scoring_ledger, echo_gate=echo_gate)
                print(f"fdiag status={out['status']}", flush=True)
            return EXIT_OK
        if a.cmd == "run":
            root = run_rehearsal(plan_path=a.plan, snapshot=a.snapshot, device=a.device, parent=non_main_parent,
                                 checkout=checkout, launcher=launcher, env=env, now=now,
                                 exposure_ledger=exposure_ledger, scoring_ledger=scoring_ledger,
                                 flwr_procs=flwr_procs, flwr_stop=flwr_stop)
            print(f"rehearsal_done root={root}", flush=True)
            return EXIT_OK
        if a.cmd == "judge":
            from vlm.rehearsal_judge import JudgeRefused, judge

            try:
                out = judge(a.run_root, checkout=checkout, exposure_ledger=exposure_ledger)
            except JudgeRefused as exc:
                return _refuse(exc.code, str(exc))
            print(f"judge status={out['status']}", flush=True)
            return EXIT_OK
        if a.cmd in ("lists", "register"):
            # 2판 §1-4 의 순서 2 — 인자 다음, 루트 검사보다 먼저(외부 검토 7). 학습은 칸의 가드가 같은 자리에서 본다
            from vlm.rehearsal_run import _fault_guard

            _fault_guard(a.run_root, a.cmd, env)
        purpose = purpose_of_root(a.run_root)
        _root_check(a.run_root, a.plan, non_main_parent, purpose)
        plan = load_plan(a.plan, purpose=purpose)
        cfg = load_uni_config(config_path)
        co = Path(checkout) if checkout is not None else main_checkout()
        if a.cmd == "lists":
            rec = stage_lists(root=a.run_root, plan=plan, config=cfg, snapshot=a.snapshot, checkout=co,
                              exposure_ledger=exposure_ledger, now=now, env=env, token_counter=token_counter)
            print(f"lists n_gen={rec['n_gen']} excluded={rec['excluded']}", flush=True)
        elif a.cmd == "register":
            from vlm.export_generator import load_generator as generator_real

            rp = stage_register(root=a.run_root, plan=plan, config=cfg, checkout=co,
                                measure_env=measure_env or measure_real, model_loader=model_loader,
                                generator_loader=generator_loader or generator_real, now=now,
                                standin_allowed=standin_allowed,
                                env=env, rules_rel=a.rules)
            print(f"registered {'registration' if plan.purpose == 'frame_diag' else 'receipt'}={rp.name}", flush=True)
        elif a.tag == FED_TAG:
            if plan.purpose != "rehearsal" or a.receipt is not None:
                return _refuse("fed_args", "연합 칸은 리허설만 돈다 — 진단 · 영수증 인자를 받지 않는다")
            if a.stamp is not None or a.negative is not None:
                return _refuse("fed_args", "연합 칸은 스탬프 · 구판 경우를 받지 않는다 — 중단 · 재개가 없다")
            res = stage_train_fed(root=a.run_root, plan=plan, plan_path=a.plan, config=cfg, snapshot=a.snapshot,
                                  runner=fed_runner, procs=flwr_procs, stop=flwr_stop, standin_allowed=standin_allowed,
                                  non_main_parent=non_main_parent, config_path=config_path)
            print(f"train {a.tag} status={res.status} step={res.adapter_step}", flush=True)
        else:
            res = stage_train(root=a.run_root, plan=plan, config=cfg, tag=a.tag, snapshot=a.snapshot, stamp=a.stamp,
                              negative=a.negative,
                              model_loader=model_loader, standin_allowed=standin_allowed,
                              non_main_parent=non_main_parent, config_path=config_path, receipt=a.receipt, checkout=co)
            print(f"train {a.tag} status={res.status} step={res.adapter_step}", flush=True)
        return EXIT_OK
    except (StageRefused, PlanRejected, OrchestratorRefused) as exc:
        return _refuse(exc.code, str(exc))
    except Halt as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return EXIT_HALT


if __name__ == "__main__":
    raise SystemExit(main())
