"""리허설 오케스트레이터의 단계 실행기 — 리허설 2판 §1-2 의 단계 표 · §1-3 의 고의 중단 장치 · §5-4 의 노출 기록.

자식은 **하위 프로세스**다. 단계마다 기대 종료 코드와 기대 표지 수(0 또는 1)를 실제와 맞대고, 종료 코드 · 시각 · stdout · stderr 파일의
sha256 을 `<루트>/rehearsal_log.jsonl` 에 남긴다. 기대와 다르면 그 자리에서 멈춘다.

| 장치 | 규칙 |
|---|---|
| 환경 지우기 | 오케스트레이터는 자기 환경에 중단 변수 둘 가운데 하나라도 있으면 시작하지 않는다(이름은 대문자로 맞춰 본다). 모든 자식의 환경은 두 변수를 지운 복사본이고, 계획된 호출에만 둘을 함께 넣는다(`vlm.fault.scrubbed_env`) |
| 표식 | 계획된 호출마다 16바이트 hex 표식을 만들고 **자식을 띄우기 전에** 계획 기록 `faults/planned/<표식>.json` 을 fsync 해 둔다 |
| 의도한 중단 | 종료 코드 75 · 새 활성화 기록 하나 · 새 발화 기록 하나 · 활성화 기록의 표식이 이번 표식 · 지점 문자열이 계획과 같음. 중단을 건 단계가 0 으로 끝나도 실패, 중단을 걸지 않은 단계에 새 기록이 생겨도 실패 |
| 시간 제한 | 단계마다 상한(계획의 `timeouts_s`)을 두고 넘으면 **프로세스 트리째** 끊는다(`taskkill /T /F`). 끊긴 단계는 실패다 |
| 자식 인코딩 | 모든 자식을 `-X utf8` 로 띄운다. stdout · stderr 는 바이트로 받아 `logs/` 에 파일로 남긴다 |

**이음새.** 자식 명령을 만드는 `launcher` 는 **파이썬 인자로만** 받는다 — 합성 시험이 대역을 끼운 자식(시험 전용 진입 파일)을 띄운다.
기본값은 실제 진입점이다(`scripts/rehearsal_uni.py` 의 하위 명령 · `scripts/export_uni.py` · 평가 쪽 `scripts.probe.score_unified`).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 상주 SuperLink · 시뮬레이션 프로세스의 찾기 · 내리기 — 정확한 진입점과 이 venv 의 소유로만 고른다(외부 검토 §50 요청 2).
# 명령줄의 부분 문자열로 고르던 종전 선택기는 걷었다. 학습 단계가 이 이름으로 부른다.
from fl.flwr_procs import flwr_processes, stop_processes  # noqa: F401

__all__ = [
    "LOG_FILE",
    "REQUIRED_TAGS",
    "OrchestratorRefused",
    "Stage",
    "StageResult",
    "default_launcher",
    "expected_stages",
    "export_stage_names",
    "file_hashes",
    "outputs_listing",
    "preexisting_changes",
    "run_stage",
    "train_stage_names",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = "rehearsal_log.jsonl"
ANY_NONZERO = "nonzero"
"""기대 종료 코드 — 0 이 아니고 75 도 아닌 값(거부)."""


class OrchestratorRefused(ValueError):
    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


@dataclass(frozen=True)
class Stage:
    no: str
    """단계 번호(2판 §1-2 의 표 — `0` · `0'` · `E` · `E'` · `1` · `2` · `2'` · `2''` · `3`~`6` · `6'` · `B` · `7`)."""
    name: str
    entry: str
    """자식의 진입 — `lists` · `register` · `train` · `export` · `score`."""
    argv: tuple[str, ...]
    expect: int | str | tuple[int, ...]
    """기대 종료 코드 — 정수 하나 · 정수 몇 · `ANY_NONZERO`."""
    timeout_key: str
    fault: str | None = None
    tag: str | None = None


@dataclass
class StageResult:
    stage: Stage
    exit_code: int | None
    ok: bool
    row: dict[str, Any] = field(default_factory=dict)
    stdout: bytes = b""
    stderr: bytes = b""


#: 리허설이 반드시 돌아야 하는 칸 — 다섯 모델(2판 §3 ② 의 다섯 칸 · ③ 의 다섯 모델과 에코). 계획의 `cells` 와 무관하다.
REQUIRED_TAGS = ("uni_central", "uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_fed")


def train_stage_names(tag: str, faulted: bool) -> list[str]:
    return [f"train_{tag}_1", f"train_{tag}_2"] if faulted else [f"train_{tag}"]


def export_stage_names(tag: str, n_attempts: int) -> list[str]:
    return [f"export_{tag}_{k}" for k in range(1, n_attempts + 1)]


def expected_stages(plan: Any) -> list[str]:
    """계획이 정하는 단계 표의 이름 전부 — 오케스트레이터가 이 이름으로 돌고, 판정기가 빠진 단계를 이것으로 찾는다(외부 검토 3)."""
    faults = plan.get("faults") or {}
    tf, ef = faults.get("train") or {}, faults.get("export") or {}
    out = ["lists", "register", "echo_export", "echo_score"]
    out += train_stage_names("uni_central", "uni_central" in tf)
    out += [f"negative_{c}" for c in plan.get("negatives") or []]
    out += ["overwrite_same_stamp", "overwrite_other_stamp"]
    out += export_stage_names("uni_central", len(ef.get("uni_central") or [None]))
    out += ["rerun_uni_central"]
    for tag in REQUIRED_TAGS[1:]:
        out += train_stage_names(tag, tag in tf)
        out += export_stage_names(tag, len(ef.get(tag) or [None]))
    return out + ["score"]


def default_launcher(entry: str, argv: Sequence[str]) -> list[str]:
    """실제 진입점의 명령. 자식은 모두 `-X utf8` 로 띄운다."""
    py = [sys.executable, "-X", "utf8"]
    if entry in ("lists", "register", "train"):
        return py + [str(REPO_ROOT / "scripts" / "rehearsal_uni.py"), entry, *argv]
    if entry == "export":
        return py + [str(REPO_ROOT / "scripts" / "export_uni.py"), *argv]
    if entry == "score":
        return py + ["-m", "scripts.probe.score_unified", *argv]
    raise OrchestratorRefused("entry_unknown", f"모르는 진입: {entry!r}")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fault_files(root: Path) -> dict[str, set[str]]:
    d = Path(root) / "faults"
    if not d.is_dir():
        return {"activated": set(), "fired": set()}
    return {"activated": {p.name for p in d.glob("*.activated.json")}, "fired": {p.name for p in d.glob("*.fired.json")}}


def _append_log(root: Path, row: Mapping[str, Any]) -> None:
    p = Path(root) / LOG_FILE
    line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if p.exists():
        raw = p.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise OrchestratorRefused("log_broken", f"{p.name} 의 끝이 LF 가 아니다 — 덧붙이지 않는다")
    with open(p, "ab") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, check=False)
    else:  # pragma: no cover - Windows 가 정본 환경이다
        import signal

        os.killpg(os.getpgid(pid), signal.SIGKILL)


def run_stage(stage: Stage, *, root: Path, seq: int, launcher: Callable[[str, Sequence[str]], list[str]],
              env_base: Mapping[str, str], timeout_s: float, extra: Mapping[str, Any] | None = None) -> StageResult:
    """단계 하나를 자식으로 돌리고 기대와 맞댄다. 로그 한 줄을 쓴다."""
    from vlm.fault import EXIT_CODE, scrubbed_env, write_planned

    root = Path(root)
    nonce = planned_sha = None
    if stage.fault is not None:
        nonce = secrets.token_hex(16)
        planned_sha = _sha(write_planned(root, nonce, stage.fault, stage_no=seq).read_bytes())
    env = scrubbed_env(env_base, planned=(stage.fault, nonce) if stage.fault is not None else None)
    before = _fault_files(root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out_p, err_p = logs / f"{seq:03d}_{stage.name}.stdout", logs / f"{seq:03d}_{stage.name}.stderr"
    cmd = launcher(stage.entry, stage.argv)
    started = datetime.now().astimezone()
    killed = False
    with open(out_p, "xb") as fo, open(err_p, "xb") as fe:
        kw: dict[str, Any] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else \
            {"start_new_session": True}
        proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, env=env, cwd=str(REPO_ROOT), **kw)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid)
            killed = True
            code = proc.wait()
    ended = datetime.now().astimezone()
    stdout, stderr = out_p.read_bytes(), err_p.read_bytes()
    after = _fault_files(root)
    new_act = sorted(after["activated"] - before["activated"])
    new_fired = sorted(after["fired"] - before["fired"])
    acts = []
    for name in new_act:
        raw = (root / "faults" / name).read_bytes()
        rec = json.loads(raw)
        acts.append({"name": name, "sha256": _sha(raw), "nonce": rec.get("nonce"), "fault": rec.get("fault")})
    fired = [{"name": n, "sha256": _sha((root / "faults" / n).read_bytes())} for n in new_fired]
    if stage.fault is not None:
        intended = (code == EXIT_CODE and len(acts) == 1 and len(fired) == 1 and acts[0]["nonce"] == nonce
                    and acts[0]["fault"] == stage.fault)
        exit_ok = intended
        markers_ok = intended
    else:
        intended = False
        markers_ok = not acts and not fired
        if stage.expect == ANY_NONZERO:
            exit_ok = code not in (0, EXIT_CODE)
        elif isinstance(stage.expect, tuple):
            exit_ok = code in stage.expect
        else:
            exit_ok = code == stage.expect
    ok = exit_ok and markers_ok and not killed
    row = {"seq": seq, "stage_no": stage.no, "name": stage.name, "entry": stage.entry, "tag": stage.tag,
           "argv": list(stage.argv), "exit_code": code,
           "expect": "fault_75" if stage.fault is not None else (list(stage.expect) if isinstance(stage.expect, tuple)
                                                                    else stage.expect), "ok": ok, "killed": killed,
           "timeout_s": timeout_s, "started_at": started.isoformat(timespec="microseconds"),
           "ended_at": ended.isoformat(timespec="microseconds"), "wall_s": round((ended - started).total_seconds(), 3),
           "stdout_path": out_p.relative_to(root).as_posix(), "stdout_sha256": _sha(stdout),
           "stderr_path": err_p.relative_to(root).as_posix(), "stderr_sha256": _sha(stderr),
           "fault": stage.fault, "nonce": nonce, "planned_sha256": planned_sha, "intended_fault": intended,
           "activated": acts, "fired": fired, **(dict(extra) if extra else {})}
    _append_log(root, row)
    return StageResult(stage, code, ok, row, stdout, stderr)


def file_hashes(root: Path, *, skip: Sequence[str] = (LOG_FILE, "logs", "judge.json")) -> dict[str, str]:
    """루트 안 파일의 상대 경로 → sha256(조건 ④ 의 전후 대조). 오케스트레이터 자신의 기록은 뺀다."""
    root = Path(root)
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if not p.is_file() or rel.split("/", 1)[0] in skip:
            continue
        out[rel] = _sha(p.read_bytes())
    return out


def preexisting_changes(before: dict[str, list], after: dict[str, list]) -> list[str]:
    """실행 전에 있던 파일 가운데 크기 · `mtime_ns` 가 바뀌었거나 사라진 것. 새로 생긴 파일은 보지 않는다(2판 §3 ④ — 보존)."""
    return sorted(k for k, v in before.items() if after.get(k) != v)


def outputs_listing(outputs: Path, *, skip_dirs: Sequence[Path] = (), skip_files: Sequence[Path] = ()) -> dict[str, list]:
    """`outputs/` 의 파일 → [크기, mtime_ns](조건 ④ 의 루트 밖 대조). 리허설 루트와 두 원장은 뺀다."""
    outputs = Path(outputs)
    sd = [Path(os.path.abspath(d)) for d in skip_dirs]
    sf = {Path(os.path.abspath(f)) for f in skip_files}
    out: dict[str, list] = {}
    if not outputs.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(outputs):
        here = Path(os.path.abspath(dirpath))
        dirnames[:] = [d for d in dirnames if not any((here / d) == s or (here / d).is_relative_to(s) for s in sd)]
        for f in filenames:
            p = here / f
            if p in sf or any(p.is_relative_to(s) for s in sd):
                continue
            st = p.stat()
            out[p.relative_to(Path(os.path.abspath(outputs))).as_posix()] = [st.st_size, st.st_mtime_ns]
    return dict(sorted(out.items()))
