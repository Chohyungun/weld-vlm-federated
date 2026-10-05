"""자원 지킴과 쓰기 가드.

**자원 지킴**(3판 1-2) — 원천을 공유 드라이브에서 읽는 동안 구간마다 그 앞에서 본다.
  - C: 여유가 문턱 아래면 기다리고, 정한 시간 안에 돌아오지 않으면 멈춘다(드라이브 캐시가 C: 를 채운다),
  - 본체 게이트 락(`.git/CTO_GATE_OPEN`)이 있으면 풀릴 때까지 기다린다,
  - 여유 메모리가 문턱 아래면 기다린다.

**쓰기 가드**(VT-1b) — 빌더는 허락된 뿌리(스테이징 · 원천 사본 자리) 안에만 쓴다. 동결 명부의 경로와 RT 의
매니페스트 · 타일 · 라벨 · 원천 폴더에는 쓰지 않는다. 경로는 정션을 풀어 비교한다(`data/interim` 은 본체와 정션이다).
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

from data.frozen_guard import EXPECTED_SEALED, is_frozen

#: RT 쪽 자리 — VT 빌더가 쓰면 안 된다. 동결 명부(EXPECTED_SEALED)에 더해 이름으로 막는다.
RT_FORBIDDEN = (
    "data/raw/aihub71761",
    "data/interim/aihub_labels",
    "data/interim/manifest_v0",
    "data/interim/manifest_v1",
    "data/interim/manifest_v2_absorbed",
    "data/interim/manifest_v3_rawlabels",
    "data/interim/tiles_v1",
    "data/interim/tiles_v1_histmatch",
    "data/interim/tiles_v1_masked",
)


class GuardStop(RuntimeError):
    """지킴 조건이 정한 시간 안에 돌아오지 않았다 — 멈춘다."""


class WriteForbidden(RuntimeError):
    """허락되지 않은 자리에 쓰려고 했다."""


# ------------------------------------------------------------------------------------------
# 자원
# ------------------------------------------------------------------------------------------
def c_free_gb(drive: str = "C:/") -> float:
    return shutil.disk_usage(drive).free / 1e9


def mem_free_gb() -> float:
    if sys.platform == "win32":
        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPhys / 1e9
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    raise RuntimeError("여유 메모리를 잴 수 없다")


def gate_lock_path(repo_root: Path) -> Path:
    """본체 저장소의 `.git/CTO_GATE_OPEN`. 워크트리면 `.git` 파일의 gitdir 에서 공통 git 폴더를 찾는다."""
    dotgit = repo_root / ".git"
    if dotgit.is_dir():
        return dotgit / "CTO_GATE_OPEN"
    text = dotgit.read_text(encoding="utf-8").strip()
    if not text.startswith("gitdir:"):
        raise RuntimeError(f".git 파일을 읽을 수 없다: {dotgit}")
    gitdir = Path(text.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (repo_root / gitdir).resolve()
    # …/.git/worktrees/<이름> → …/.git
    if gitdir.parent.name != "worktrees":
        raise RuntimeError(f"워크트리 gitdir 꼴이 아니다: {gitdir}")
    return gitdir.parent.parent / "CTO_GATE_OPEN"


class Guard:
    """구간마다 그 앞에서 부른다. 조건이 서면 돌아오고, C: 가 시한 안에 돌아오지 않으면 `GuardStop`."""

    def __init__(
        self,
        *,
        c_free_min_gb: float,
        mem_free_min_gb: float,
        c_wait_max_s: float,
        wait_step_s: float,
        gate_path: Path | None,
        c_free_fn: Callable[[], float] = c_free_gb,
        mem_free_fn: Callable[[], float] = mem_free_gb,
        sleep_fn: Callable[[float], None] = time.sleep,
        log: Callable[[dict], None] | None = None,
        check_c: bool = True,
    ) -> None:
        self.c_min, self.mem_min = c_free_min_gb, mem_free_min_gb
        self.c_wait_max, self.step = c_wait_max_s, wait_step_s
        self.gate_path = gate_path
        self.c_free_fn, self.mem_free_fn, self.sleep_fn = c_free_fn, mem_free_fn, sleep_fn
        self.log = log or (lambda d: None)
        self.check_c = check_c          # 공유 드라이브를 읽지 않는 단계(E: 사본만)는 C: 여유를 보지 않는다
        self.waits = 0

    @classmethod
    def from_config(cls, cfg: dict, repo_root: Path, **kw) -> Guard:
        c = cfg["copy"]
        return cls(c_free_min_gb=float(c["c_free_min_gb"]), mem_free_min_gb=float(c["mem_free_min_gb"]),
                   c_wait_max_s=float(c["c_wait_max_s"]), wait_step_s=float(c["wait_step_s"]),
                   gate_path=gate_lock_path(repo_root), **kw)

    def gate_open(self) -> bool:
        return self.gate_path is not None and self.gate_path.exists()

    def wait(self) -> dict:
        """조건이 설 때까지 기다린다. 돌아올 때의 C: · 메모리 값을 준다."""
        c_waited = 0.0
        while True:
            cf, mf, gate = self.c_free_fn(), self.mem_free_fn(), self.gate_open()
            c_ok = cf >= self.c_min or not self.check_c
            if c_ok and mf >= self.mem_min and not gate:
                return {"c_free_gb": round(cf, 2), "mem_free_gb": round(mf, 2)}
            self.waits += 1
            self.log({"wait": True, "c_free_gb": round(cf, 2), "mem_free_gb": round(mf, 2), "gate": gate})
            if not c_ok and not gate:
                c_waited += self.step
                if c_waited > self.c_wait_max:
                    raise GuardStop(f"C: 여유가 {self.c_wait_max:.0f}초 안에 {self.c_min} GB 로 돌아오지 않았다")
            self.sleep_fn(self.step)


# ------------------------------------------------------------------------------------------
# 쓰기 가드 (VT-1b)
# ------------------------------------------------------------------------------------------
def _real(p: Path) -> str:
    return os.path.normcase(os.path.realpath(p))


def within(a: Path, b: Path) -> bool:
    """`a` 가 `b` 와 같거나 `b` 아래인가 — 정션을 풀고 대소문자를 맞춘 경로로 본다."""
    ra, rb = _real(a), _real(b)
    return ra == rb or ra.startswith(rb.rstrip(os.sep) + os.sep)


def assert_vt_writable(target: Path, *, allowed_roots: list[Path], repo_root: Path) -> None:
    """VT 빌더가 `target` 에 써도 되는가. 아니면 아무것도 쓰기 전에 `WriteForbidden`."""
    for rel in list(EXPECTED_SEALED) + list(RT_FORBIDDEN):
        bad = repo_root / rel
        if within(target, bad) or within(bad, target):
            raise WriteForbidden(f"동결 · RT 자리에 쓰려고 했다: {rel}")
    if not any(within(target, r) for r in allowed_roots):
        raise WriteForbidden(f"허락된 뿌리 밖이다: {target}")
    p = target
    while True:
        if p.exists() and p.is_dir() and is_frozen(p):
            raise WriteForbidden(f"동결 계약서가 있는 디렉터리다: {p}")
        if p.parent == p or not any(within(p, r) for r in allowed_roots):
            break
        p = p.parent
