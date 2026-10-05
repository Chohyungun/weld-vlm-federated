"""고의 중단 장치 — 리허설에서만, 계획된 호출에서만, 켜기 전에 기록을 남기고 죽는다(리허설 2판 §1-3).

## 켜는 법과 받는 목적

환경 변수 `WELD_REHEARSAL_FAULT`(값 `<단계>:<지점>:<태그>:<인자>`)와 1회용 표식 `WELD_REHEARSAL_NONCE`(16바이트 hex).
**`purpose="rehearsal"` 의 계획된 호출에서만 켜진다.** `main` · `frame_diag` 는 둘 가운데 하나라도 보이면 무엇보다 먼저
거부한다(§1-4 의 순서 2). 리허설은 둘이 **함께** 있고, 표식의 꼴이 맞고, `<루트>/faults/planned/<표식>.json` 이 있으며
그 기록의 지점이 변수 값과 같을 때만 켜진다. 하나라도 어긋나면 시작을 거부한다(종료 코드는 75 가 아니다).
환경 이름은 대문자로 맞춰 본다 — Windows 의 환경 이름은 대소문자를 가리지 않는다.

## 켜는 순서 — 지점에 닿으면

① **활성화 기록** `<루트>/faults/<순번>_<단계>-<지점>-<태그>-<인자>.activated.json` 을 `<표식>.activated.tmp` 에 쓰고
fsync 한 뒤 개명한다. 개명이 `FileExistsError` 면 켜지 않고 멈춘다(종료 75 아님). ② 효과(학습 지점은 없다 — 죽음이 효과다)
③ **발화 기록** `…fired.json` fsync ④ `os._exit(75)`. ① 이 실패하면 ②~④ 를 하지 않는다 — 출력과 학습 상태를 한 바이트도
바꾸지 않는다. 파일 이름에 `:` 를 쓰지 않는다(NTFS 의 대체 데이터 스트림 구분자). 지점 문자열 · 표식 · 시각은 파일 **안**에 싣는다.

## 한 번만

같은 (단계 · 지점 · 태그 · 인자)의 활성화 기록이 이미 있으면 다시 켜지지 않는다 — 가드(§1-4 의 순서 2~6) **뒤에** 선다.
계획된 호출이 이 규칙으로 켜지지 않으면 그 단계는 종료 0 으로 끝나 판정에서 실패로 드러난다.

## 훅의 가드

훅은 **목적과 정규화한 루트**를 호출한 쪽의 문맥 객체(`FaultContext`)에서 다시 받는다. 목적이 `rehearsal` 이 아니거나 루트가
켤 때의 루트와 다르면 켜지 않고 거부한다. 문맥 없이 불린 훅은 켜지지 않는다. 공유 코드의 훅이 다른 진입점에서 불려도 막힌다.

이 변수는 재개 신원에 넣지 않는다. 대신 활성화 · 발화 기록의 sha256 과 그 뒤의 복구 세그먼트를 `fault_events` 로 잇는다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = ["FAULT_VAR", "NONCE_VAR", "FAULT_VARS", "EXIT_CODE", "POINTS", "FaultRefused", "FaultSpec",
           "FaultContext", "ArmedFault", "parse_fault", "fault_env", "check_fault_env", "assert_no_fault_env",
           "scrubbed_env", "write_planned", "fault_events"]

FAULT_VAR = "WELD_REHEARSAL_FAULT"
NONCE_VAR = "WELD_REHEARSAL_NONCE"
FAULT_VARS = (FAULT_VAR, NONCE_VAR)
EXIT_CODE = 75
#: 단계 → 지점 → 인자의 이름(없으면 `-`). `export:after_stamp` 는 합성 시험에서만 쓴다.
POINTS: dict[str, dict[str, str | None]] = {
    "train": {"after_ckpt": "ep", "before_ckpt": "ep"},
    "export": {"after_lines": "n", "torn_line": "n", "after_all_lines": None, "after_stamp": None},
}
TAGS = ("uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_central", "uni_fed")
NONCE_RE = re.compile(r"[0-9a-f]{32}")          # `fullmatch` 로만 쓴다 — `$` 는 끝 개행을 받는다


class FaultRefused(ValueError):
    """고의 중단 설정 때문에 시작하지 않는다(또는 훅이 켜지지 않는다). 종료 코드는 75 가 아니다. `code` 가 사유다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


@dataclass(frozen=True)
class FaultSpec:
    raw: str
    stage: str
    point: str
    tag: str
    arg: str
    k: int | None

    @property
    def key(self) -> str:
        """파일 이름에 쓰는 열쇠 — `:` 대신 `-`, 인자의 `=` 도 `-`."""
        return f"{self.stage}-{self.point}-{self.tag}-{self.arg.replace('=', '-')}"


@dataclass(frozen=True)
class FaultContext:
    """훅이 다시 받는 호출한 쪽의 문맥 — 목적과 정규화한 루트."""

    purpose: str
    run_root: Path


def parse_fault(raw: str) -> FaultSpec:
    """`<단계>:<지점>:<태그>:<인자>`. 모르는 지점 · 태그 · 인자 꼴은 거부한다 — 오타가 조용히 무시되면 중단이 걸리지 않는다."""
    parts = str(raw).split(":")
    if len(parts) != 4:
        raise FaultRefused("fault_form", f"중단 값은 네 칸이어야 한다: {raw!r}")
    stage, point, tag, arg = parts
    if stage not in POINTS or point not in POINTS[stage]:
        raise FaultRefused("fault_point", f"모르는 단계 · 지점: {stage}:{point}")
    if tag not in TAGS:
        raise FaultRefused("fault_tag", f"모르는 태그: {tag!r}")
    name = POINTS[stage][point]
    if name is None:
        if arg != "-":
            raise FaultRefused("fault_arg", f"{stage}:{point} 는 인자가 `-` 다: {arg!r}")
        return FaultSpec(raw, stage, point, tag, arg, None)
    m = re.fullmatch(rf"{name}=([0-9]+)", arg)          # `\d` 는 유니코드 숫자도 받는다
    if not m:
        raise FaultRefused("fault_arg", f"{stage}:{point} 의 인자는 {name}=<정수> 다: {arg!r}")
    return FaultSpec(raw, stage, point, tag, arg, int(m.group(1)))


def fault_env(env: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    """두 변수의 값 — 이름을 대문자로 맞춰 찾는다."""
    src = os.environ if env is None else env
    up = {str(k).upper(): v for k, v in src.items()}
    return up.get(FAULT_VAR), up.get(NONCE_VAR)


def assert_no_fault_env(env: Mapping[str, str] | None = None, *, who: str = "") -> None:
    """두 변수 가운데 하나라도 보이면 거부한다 — 연합의 서버 · 클라이언트(목적과 무관하게) · 본실험 · 진단."""
    f, n = fault_env(env)
    if f is not None or n is not None:
        raise FaultRefused("fault_env_forbidden", f"{who} 고의 중단 변수를 받지 않는다: "
                           f"{[v for v, x in ((FAULT_VAR, f), (NONCE_VAR, n)) if x is not None]}")


def _fsync_write(path: Path, data: bytes, *, exclusive: bool = True) -> None:
    with open(path, "xb" if exclusive else "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def write_planned(run_root: str | Path, nonce: str, fault: str, *, stage_no: int) -> Path:
    """오케스트레이터 쪽 — 자식을 띄우기 **전에** 계획 기록을 fsync 해 둔다. 이미 있으면 쓰지 않는다(배타 생성)."""
    if not NONCE_RE.fullmatch(nonce):
        raise FaultRefused("nonce_form", f"표식은 16바이트 hex(32자)여야 한다: {nonce!r}")
    spec = parse_fault(fault)
    d = Path(run_root) / "faults" / "planned"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{nonce}.json"
    rec = {"nonce": nonce, "fault": spec.raw, "stage_no": int(stage_no), "stage": spec.stage, "point": spec.point,
           "tag": spec.tag, "arg": spec.arg, "time": time.time()}
    _fsync_write(p, (json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    return p


def scrubbed_env(env: Mapping[str, str], planned: tuple[str, str] | None = None) -> dict[str, str]:
    """자식 환경 — 두 변수를 **지운 복사본**에서 만들고, 계획된 리허설 호출에만 둘을 **함께** 넣는다.

    넘기지 않는 것과 지우는 것은 다르다 — 복사본에 새로 넣지 않아도 부모에 있던 값은 남는다. 이름은 대문자로 맞춰 지운다.
    """
    out = {k: v for k, v in env.items() if str(k).upper() not in FAULT_VARS}
    if planned is not None:
        fault, nonce = planned
        out[FAULT_VAR], out[NONCE_VAR] = str(fault), str(nonce)
    return out


@dataclass
class ArmedFault:
    """켤 준비가 된 중단 하나. 가드를 지난 뒤에만 만들어진다."""

    spec: FaultSpec
    nonce: str
    planned_sha256: str
    run_root: Path
    #: 죽는 함수. 없으면 부를 때 `os._exit` 를 찾는다 — 프로세스 안 합성 시험이 파이썬 인자로 바꿔 끼운다.
    exit_fn: Callable[[int], Any] | None = None

    @property
    def faults_dir(self) -> Path:
        return self.run_root / "faults"

    def already_activated(self) -> bool:
        """"한 번만" — 같은 열쇠의 활성화 기록이 이미 있다."""
        return any(self.faults_dir.glob(f"*_{self.spec.key}.activated.json"))

    def fire(self, ctx: FaultContext | None, *, point: str, tag: str, k: int | None, run_id: str = "",
             effect: Callable[[], None] | None = None) -> None:
        """지점에 닿았다. 이 중단의 지점 · 태그 · 인자와 같을 때만 켠다 — ① 활성화 기록 ② 효과 ③ 발화 기록 ④ 종료 75."""
        if ctx is None:
            raise FaultRefused("hook_no_context", "문맥 없이 불린 훅은 켜지지 않는다")
        if ctx.purpose != "rehearsal" or Path(ctx.run_root).resolve() != self.run_root.resolve():
            raise FaultRefused("hook_context", f"훅의 문맥이 켤 때와 다르다: 목적 {ctx.purpose!r} · 루트 {ctx.run_root}")
        s = self.spec
        if (point, tag, k) != (s.point, s.tag, s.k):
            return
        if self.already_activated():
            # "한 번만" 을 여기서도 지킨다 — 부르는 쪽이 앞 기록을 보지 않아도 다시 발화하지 않는다(검수 14번 M-11).
            # 이름의 순번 때문에 개명의 배타성만으로는 같은 열쇠의 두 번째 기록을 막지 못한다.
            raise FaultRefused("already_activated", f"같은 열쇠의 활성화 기록이 이미 있다: {s.key}")
        d = self.faults_dir
        d.mkdir(parents=True, exist_ok=True)
        seq = len(list(d.glob("*.activated.json"))) + 1
        name = f"{seq:03d}_{s.key}"
        act = {"run_id": run_id, "fault": s.raw, "stage": s.stage, "point": s.point, "tag": s.tag, "arg": s.arg,
               "nonce": self.nonce, "planned_sha256": self.planned_sha256, "pid": os.getpid(), "time": time.time()}
        tmp = d / f"{self.nonce}.activated.tmp"
        # ① — 실패하면 아무것도 바꾸지 않고 올린다(종료 75 가 아니다).
        _fsync_write(tmp, (json.dumps(act, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        final = d / f"{name}.activated.json"
        if final.exists():
            raise FaultRefused("activation_exists", f"활성화 기록이 이미 있다: {final.name}")
        os.rename(tmp, final)                      # 대상이 있으면 Windows 에서 FileExistsError — 켜지 않는다
        if effect is not None:
            effect()                               # ②
        fired = {"activated": final.name, "activated_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                 "nonce": self.nonce, "pid": os.getpid(), "time": time.time()}
        _fsync_write(d / f"{name}.fired.json",       # ③
                     (json.dumps(fired, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        (self.exit_fn or os._exit)(EXIT_CODE)      # ④


def check_fault_env(purpose: str, *, run_root: str | Path | None, stage: str, tags: Sequence[str],
                    bounds: Mapping[str, int] | None = None, env: Mapping[str, str] | None = None
                    ) -> ArmedFault | None:
    """진입점의 순서 2 — 두 변수를 목적 · 계획 기록과 맞댄다. 켤 것이 없으면 None, 어긋나면 `FaultRefused`.

    `tags` 는 이 호출이 도는 칸들이고 `bounds` 는 인자의 상한(`{"ep": epoch 수}` 따위, 인자 < 상한)이다.
    파일을 쓰지 않는다 — 계획 기록을 읽기만 한다.
    """
    fault, nonce = fault_env(env)
    if purpose != "rehearsal":
        assert_no_fault_env(env, who=f"purpose={purpose}")
        return None
    if fault is None and nonce is None:
        return None
    if fault is None or nonce is None:
        raise FaultRefused("fault_env_half", "중단 변수와 표식은 함께 있어야 한다")
    if not NONCE_RE.fullmatch(nonce):
        raise FaultRefused("nonce_form", "표식의 꼴이 틀리다(16바이트 hex)")
    spec = parse_fault(fault)
    if run_root is None:
        raise FaultRefused("fault_no_root", "루트 없이 중단을 켤 수 없다")
    planned = Path(run_root) / "faults" / "planned" / f"{nonce}.json"
    if not planned.exists():
        raise FaultRefused("planned_missing", f"표식의 계획 기록이 없다: {planned.name}")
    raw = planned.read_bytes()
    try:
        rec = json.loads(raw)
    except json.JSONDecodeError:
        raise FaultRefused("planned_broken", f"계획 기록을 읽을 수 없다: {planned.name}") from None
    if rec.get("fault") != fault or rec.get("nonce") != nonce:
        raise FaultRefused("planned_mismatch", "계획 기록의 지점 · 표식이 변수와 다르다")
    if spec.stage != stage or spec.tag not in tuple(tags):
        raise FaultRefused("fault_not_for_this_call", f"이 호출({stage} · {list(tags)})의 중단이 아니다: {fault}")
    name = POINTS[spec.stage][spec.point]
    if name is not None and bounds and name in bounds and not 0 <= int(spec.k) < int(bounds[name]):
        raise FaultRefused("fault_range", f"{spec.arg} 가 범위 0..{int(bounds[name]) - 1} 밖이다")
    return ArmedFault(spec=spec, nonce=nonce, planned_sha256=hashlib.sha256(raw).hexdigest(),
                      run_root=Path(run_root).resolve())


def fault_events(run_root: str | Path | None, *, stage: str, tags: Iterable[str],
                 segments: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """이 학습(단계 · 태그)에서 켜진 중단의 기록 — 활성화 · 발화의 sha256 과 그 뒤의 복구 세그먼트(`started_at`).

    복구 세그먼트는 활성화 시각 뒤에 처음 선 세그먼트다. 본실험은 루트가 없어 언제나 빈 목록이다.
    """
    if run_root is None:
        return []
    d = Path(run_root) / "faults"
    if not d.is_dir():
        return []
    want = set(tags)
    out: list[dict[str, Any]] = []
    for act in sorted(d.glob("*.activated.json")):
        rec = json.loads(act.read_text(encoding="utf-8"))
        if rec.get("stage") != stage or rec.get("tag") not in want:
            continue
        fired = act.with_name(act.name.replace(".activated.json", ".fired.json"))
        t = float(rec.get("time", 0.0))
        rec_seg = next((s for s in segments if float(s.get("started_at", 0.0)) > t), None)
        out.append({"activated": act.name, "activated_sha256": hashlib.sha256(act.read_bytes()).hexdigest(),
                    "fired": fired.name if fired.exists() else None,
                    "fired_sha256": hashlib.sha256(fired.read_bytes()).hexdigest() if fired.exists() else None,
                    "fault": rec.get("fault"), "nonce": rec.get("nonce"),
                    "recovered_process_started_unix": None if rec_seg is None else rec_seg["started_at"]})
    return out
