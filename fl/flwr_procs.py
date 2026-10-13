"""상주 flwr 프로세스의 신원 — **정확한 진입점과 이 실행의 소유(이 venv)** 로만 고른다(외부 검토 §50 요청 2).

명령줄 문자열의 부분 일치로 고르지 않는다 — `notepad.exe C:/notes/superlink.txt` 같은 무관한 프로세스가 걸린다.

| 판정 | 무엇 |
|---|---|
| 대상(`target`) | 프로세스 이미지가 **이 venv 의 `Scripts`**(이 인터프리터의 폴더 — `scripts/main_det.py::_flwr_run` 이 flwr 를 찾는 그 폴더) 아래의 flwr 진입점 실행 파일이거나, 파이썬 인터프리터가 그 진입점 파일을 **첫 인자(스크립트)** 로 돌린다. 진입점은 `flower-superlink` · `flwr-simulation` · `flower-supernode` · `flwr` 다 |
| 불명(`unclear`) | 진입점 이름은 맞는데 이 venv 의 것이 아니다 · 실행 파일 경로를 읽지 못했다 · `python -m flwr…` 꼴이다. **죽이지 않는다** — 부르는 쪽이 시작을 거부한다 |
| 무관 | 그 밖. 인자 · 문서 · 작업 폴더 · 상위 폴더 이름에 표지가 들어도 고르지 않는다 |

인터프리터의 경우 **파이썬 명령줄 옵션을 해석해** 실제로 도는 스크립트 · 모듈을 찾는다(`-u` · `-X utf8` · `-W ignore` 같은 옵션 뒤의 첫 인자).
일반 인자를 뒤져 표지를 찾지 않는다 — 옵션 해석이 정한 자리 하나만 본다(외부 검토 회신 `rehearsal_blockers` 의 2). 옵션을 해석하지 못한
명령줄(모르는 옵션)만은 인자에 진입점 이름이 있으면 불명으로 둔다 — 내리지 않고 시작을 거부할 뿐이다.
스크립트 자리가 상대 경로면 **대상 프로세스의 작업 디렉터리**로 해석한다 — 검사하는 쪽의 현재 디렉터리로 풀면 다른 자리에서 띄운
`python -u .venv/Scripts/flower-superlink-script.py` 를 이 venv 의 것으로 오인한다. 작업 디렉터리를 읽지 못했거나(권한)
드라이브 · 루트만 붙은 상대 경로(`C:x` · 역슬래시로 시작하는 경로)면 불명이다(외부 검토 회신 `rehearsal_blockers2` 의 2).

`stop_processes` 는 대상만 내린다 — 다른 판정이 섞이면 아무것도 내리지 않고 멈춘다. 내리다 권한 거부(`AccessDenied`) 따위로 실패한
프로세스가 있으면 나머지를 다 시도한 뒤 `StopFailed` 로 알린다.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

__all__ = ["ENTRIES", "StopFailed", "classify", "flwr_processes", "python_target", "stop_processes", "venv_scripts"]

ENTRIES = ("flower-superlink", "flwr-simulation", "flower-supernode", "flwr")
_PYTHONS = ("python", "python3", "pythonw")
_SUFFIXES = (".exe", "-script.py", ".py")


def venv_scripts() -> Path:
    """이 실행의 venv `Scripts` — 이 인터프리터의 폴더다(`_flwr_run` 이 flwr 실행 파일을 찾는 자리와 같다)."""
    return Path(sys.executable).resolve().parent


def _base(path: str) -> str:
    name = PureWindowsPath(path).name if "\\" in path else Path(path).name
    name = name.lower()
    for suf in _SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _entry(path: str) -> str | None:
    b = _base(path)
    return b if b in ENTRIES else None


#: CPython 의 명령줄 옵션 — 값을 받는 것과 받지 않는 것(`python --help`). 값은 붙여 쓰거나(`-Wignore`) 다음 인자로 쓴다.
_PY_ARG_OPTS = frozenset("WX")
_PY_FLAG_OPTS = frozenset("bBdEhiIOPqsSuvVx?")
_PY_LONG_ARG = ("--check-hash-based-pycs",)
_PY_LONG_FLAG = ("--help", "--version", "--help-env", "--help-xoptions", "--help-all")


def python_target(cmd: Sequence[str]) -> tuple[str, str] | None:
    """인터프리터 명령줄(`cmd[0]` 이 인터프리터) → `("script", 경로)` · `("module", 이름)` · `("code", "")` · 옵션만이면 `None`.
    옵션을 모르는 꼴이면 `("unknown", 그 인자)` — 부르는 쪽이 판정하지 않는다."""
    i = 1
    while i < len(cmd):
        a = cmd[i]
        if a == "--":
            return ("script", cmd[i + 1]) if i + 1 < len(cmd) else None
        if a in _PY_LONG_FLAG:
            i += 1
            continue
        if a in _PY_LONG_ARG:
            i += 2
            continue
        if a.startswith("--check-hash-based-pycs="):
            i += 1
            continue
        if a == "-":
            return ("code", "")
        if a.startswith("-") and len(a) > 1 and not a.startswith("--"):
            k = 1
            while k < len(a):
                ch = a[k]
                if ch == "c":
                    return ("code", "")
                if ch == "m":
                    rest = a[k + 1:]
                    if rest:
                        return ("module", rest)
                    return ("module", cmd[i + 1]) if i + 1 < len(cmd) else ("unknown", a)
                if ch in _PY_ARG_OPTS:
                    if k + 1 == len(a):              # 값이 다음 인자다
                        i += 1
                    break
                if ch in _PY_FLAG_OPTS:
                    k += 1
                    continue
                return ("unknown", a)
            i += 1
            continue
        if a.startswith("--"):
            return ("unknown", a)
        return ("script", a)
    return None


def _script_path(what: str, cwd: str | None) -> str | None:
    """스크립트 자리의 절대 경로 — 상대면 대상의 작업 디렉터리 `cwd` 로 잇는다. 정할 수 없으면 `None`."""
    p = Path(what)
    if p.is_absolute():
        return what
    if p.drive or p.root or not cwd or not Path(cwd).is_absolute():
        return None
    return str(Path(cwd) / p)


def _in_dir(path: str, d: Path) -> bool:
    try:
        return os.path.normcase(str(Path(path).resolve().parent)) == os.path.normcase(str(Path(d).resolve()))
    except OSError:
        return False


def classify(info: dict[str, Any], scripts_dir: Path) -> str | None:
    """프로세스 하나 — `target` · `unclear` · 무관이면 `None`. `info` 는 `{pid, name, exe, cmdline, cwd}`(`cwd` 는 대상의 작업 디렉터리,
    읽지 못했으면 없음)."""
    exe = str(info.get("exe") or "")
    name = str(info.get("name") or "")
    cmd = [str(x) for x in (info.get("cmdline") or [])]
    image = exe or name
    if _entry(image):                                          # 진입점 실행 파일(런처)
        return "target" if exe and _in_dir(exe, scripts_dir) else "unclear"
    if _base(image) in _PYTHONS and len(cmd) >= 2:            # 인터프리터 — 옵션을 해석해 실제 스크립트 · 모듈을 본다
        got = python_target(cmd)
        if got is None:
            return None
        kind, what = got
        if kind == "script" and _entry(what):
            full = _script_path(what, info.get("cwd"))
            return "target" if full is not None and _in_dir(full, scripts_dir) else "unclear"
        if kind == "module" and what.split(".")[0] == "flwr":
            return "unclear"                                   # 모듈로는 이 venv 의 것인지 가릴 수 없다
        if kind == "unknown" and any(_entry(x) for x in cmd[1:]):
            return "unclear"                                   # 옵션을 해석하지 못해 스크립트 자리를 모른다 — 내리지 않고 시작만 거부한다
    return None


def flwr_processes(*, scripts_dir: Path | None = None, iter_procs: Callable[[], Iterable[Any]] | None = None) -> list[dict]:
    """지금 떠 있는 flwr 프로세스 — `{pid, name, exe, cmdline, kind}` 목록(`kind` 는 `target` · `unclear`). 이 프로세스와 조상은 뺀다.
    `scripts_dir` · `iter_procs` 는 시험의 이음새다(파이썬 인자로만)."""
    import psutil

    scripts_dir = Path(scripts_dir) if scripts_dir is not None else venv_scripts()
    me = {os.getpid()}
    try:
        me |= {pp.pid for pp in psutil.Process().parents()}
    except psutil.Error:
        pass
    out = []
    for proc in (iter_procs or (lambda: psutil.process_iter(["pid", "name", "exe", "cmdline", "cwd"])))():
        info = proc.info if hasattr(proc, "info") else proc
        if info.get("pid") in me:
            continue
        kind = classify(info, scripts_dir)
        if kind is not None:
            out.append({"pid": info.get("pid"), "name": str(info.get("name") or ""), "exe": str(info.get("exe") or ""),
                        "cmdline": " ".join(str(x) for x in (info.get("cmdline") or [])), "kind": kind})
    return out


class StopFailed(RuntimeError):
    """대상 가운데 내리지 못한 것이 있다 — `failed` 에 `(pid, 사유)`."""

    def __init__(self, failed: list[tuple[Any, str]]):
        self.failed = failed
        super().__init__(f"내리지 못한 프로세스: {failed}")


def stop_processes(found: Sequence[dict]) -> None:
    """대상(`target`)만 트리째 내린다(자식 먼저). 다른 판정이 하나라도 있으면 **아무것도 내리지 않고** 멈춘다. 이미 끝난 것은 넘어간다.
    권한 거부 · 시간 초과로 내리지 못한 것이 있으면 나머지를 다 시도한 뒤 `StopFailed`."""
    import psutil

    bad = [p for p in found if p.get("kind") != "target"]
    if bad:
        raise ValueError(f"대상이 아닌 프로세스는 내리지 않는다: {[(p.get('pid'), p.get('name')) for p in bad]}")
    failed: list[tuple[Any, str]] = []
    for item in found:
        try:
            proc = psutil.Process(item["pid"])
            for child in proc.children(recursive=True):
                child.kill()
            proc.kill()
            proc.wait(timeout=10)
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, psutil.TimeoutExpired) as exc:
            failed.append((item.get("pid"), type(exc).__name__))
    if failed:
        raise StopFailed(failed)
