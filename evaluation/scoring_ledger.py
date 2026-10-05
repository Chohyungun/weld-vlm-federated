"""통합형 **채점 원장** — 추가 전용 JSONL, 앞 행의 sha 로 잇는다. 07번 §12-5 · §31-4 · §32-1 · §32-8 · 진입점 미니스펙 3판 3-5 · 8-3.

| 종류 | 언제 | 무엇 |
|---|---|---|
| `call` | 계산 **전** | 채점 호출 하나. 실패한 호출도 남는다 — "돌았는데 산출물이 없다" 가 기록으로 드러난다 |
| `frame_diag_verdict` | 진단 판정을 계산한 뒤, **산출물보다 먼저** | 판정 한 번 규칙의 열쇠(내용 해시)와 산출물을 다시 만들 값 |
| `frame_diag_regenerated` | 판정 줄에서 산출물을 다시 만들 때 | 판정 줄의 sha · 산출물 이름 · 산출물 sha256. `purpose` 를 싣지 않는다 — 시도로 세지 않는다 |

**줄의 바이트.** 정규 JSON(키 정렬 · UTF-8 · 구분자 `,` `:`) 한 줄 + LF. 줄의 sha 는 LF 를 뺀 그 바이트의 sha256 이다.
`prev_sha256` 은 앞 줄의 sha(첫 줄은 0 64자리)다. 중간을 지우거나 고치면 사슬이 끊겨 드러난다 — **막지는 못한다**(3판 8-11).

**자리는 이 모듈이 정하지 않는다.** 정본 경로는 진입점이 본체 체크아웃 아래로 정한다(`entry_gate.canonical_ledger`) — 인자로 받지 않는다(§32-1).

**배타 잠금.** 원장 옆의 잠금 파일(`<원장>.lock`)에 운영체제 잠금을 건다(`ledger_lock`). 줄 추가는 늘 잠금 안에서 하고,
진입점은 **판정 한 번** 의 조회부터 판정 줄 추가까지를 한 잠금 안에 둔다 — 동시 호출 둘이 함께 조회를 지나 판정 줄을 둘 남기지 않게.
잠금은 프로세스가 죽으면 운영체제가 푼다. 같은 스레드 안에서는 다시 들어갈 수 있다. 기다림이 한도를 넘으면 `LedgerBusy`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LEDGER_KINDS = ("call", "frame_diag_verdict", "frame_diag_regenerated")
GENESIS = "0" * 64


LOCK_TIMEOUT_S = 600.0
"""잠금을 기다리는 한도(초). 진단 판정 하나의 계산이 이 안에 끝난다고 본다."""
WAIT_HOOK = None
"""잠금이 다른 쪽에 잡혀 기다리기 시작할 때 한 번 부르는 함수(잠금 파일 경로를 받는다). 시험이 본다 — 기본은 없음."""
_HELD = threading.local()


class LedgerChainBroken(RuntimeError):
    """원장의 사슬이 끊겼거나 줄의 꼴이 틀렸다. 몇째 줄인지 든다."""


class LedgerBusy(RuntimeError):
    """원장의 잠금을 한도 안에 얻지 못했다 — 다른 호출이 판정 · 기록 중이다."""


def _try_lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def ledger_lock(path: Path, *, timeout: float | None = None) -> Iterator[None]:
    """원장의 배타 잠금. 같은 스레드가 이미 쥐고 있으면 그대로 들어간다."""
    key = str(Path(path).resolve())
    held = getattr(_HELD, "keys", None)
    if held is None:
        held = _HELD.keys = set()
    if key in held:
        yield
        return
    lock = Path(path).with_name(Path(path).name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as fh:
        deadline = time.monotonic() + (LOCK_TIMEOUT_S if timeout is None else timeout)
        waited = False
        while not _try_lock(fh):
            if not waited:
                waited = True
                if WAIT_HOOK is not None:
                    WAIT_HOOK(lock)
            if time.monotonic() >= deadline:
                raise LedgerBusy(f"원장 잠금을 얻지 못했다({lock.name}) — 다른 호출이 판정 · 기록 중이다")
            time.sleep(0.05)
        held.add(key)
        try:
            yield
        finally:
            held.discard(key)
            _unlock(fh)


def _canon(obj: Mapping) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def line_sha(raw_line: bytes) -> str:
    return hashlib.sha256(raw_line).hexdigest()


def read_ledger(path: Path) -> list[tuple[str, dict]]:
    """원장 전체 → `(줄 sha, 줄)` 목록. 사슬과 꼴을 본다. 파일이 없으면 빈 목록."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw == b"":
        return []
    if not raw.endswith(b"\n") or b"\r" in raw:
        raise LedgerChainBroken("원장이 LF 로 끝나는 줄들이 아니다")
    out: list[tuple[str, dict]] = []
    prev = GENESIS
    for n, line in enumerate(raw[:-1].split(b"\n"), 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise LedgerChainBroken(f"{n}번째 줄이 JSON 이 아니다") from None
        if not isinstance(row, dict) or row.get("kind") not in LEDGER_KINDS:
            raise LedgerChainBroken(f"{n}번째 줄의 종류가 {list(LEDGER_KINDS)} 가 아니다")
        if _canon(row) != line:
            raise LedgerChainBroken(f"{n}번째 줄이 정규형이 아니다")
        if row.get("prev_sha256") != prev:
            raise LedgerChainBroken(f"{n}번째 줄의 prev_sha256 이 앞 줄과 맞지 않는다 — 사슬이 끊겼다")
        prev = line_sha(line)
        out.append((prev, row))
    return out


def append_line(path: Path, kind: str, payload: Mapping, *, at: datetime) -> str:
    """줄 하나를 덧붙이고 **되읽어** 확인한 뒤 그 줄의 sha 를 돌려준다. `at` 은 오프셋 있는 시각이어야 한다."""
    if kind not in LEDGER_KINDS:
        raise ValueError(f"원장 줄의 종류는 {list(LEDGER_KINDS)} 가운데 하나다")
    if at.tzinfo is None:
        raise ValueError("naive 시각은 받지 않는다")
    reserved = {"kind", "at", "prev_sha256"} & set(payload)
    if reserved:
        raise ValueError(f"줄의 예약 키를 payload 에 두지 않는다: {sorted(reserved)}")
    path = Path(path)
    with ledger_lock(path):
        rows = read_ledger(path)
        prev = rows[-1][0] if rows else GENESIS
        row = {**payload, "kind": kind, "at": at.isoformat(), "prev_sha256": prev}
        line = _canon(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as fh:
            fh.write(line + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        back = read_ledger(path)
    sha = line_sha(line)
    if not back or back[-1][0] != sha:
        raise LedgerChainBroken("쓴 줄을 되읽지 못했다 — 다른 쓰기가 끼었을 수 있다")
    return sha


def find_verdicts(path: Path, content_keys: Iterable[str]) -> list[tuple[str, dict]]:
    """판정 한 번 규칙 — 같은 **내용 해시**(`generations_content_sha256`)의 `frame_diag_verdict` 줄(07번 §32-2)."""
    keys = set(content_keys)
    out = []
    for sha, row in read_ledger(path):
        if row["kind"] != "frame_diag_verdict":
            continue
        judged = {b.get("generations_content_sha256") for b in row.get("bundles", [])}
        if judged & keys:
            out.append((sha, row))
    return out


def count_calls(path: Path, *, purpose: str, generation_sha256: str) -> int:
    """쓰는 쪽이 세는 시도 수의 정의 — 그 목적 · 그 생성 지문의 `call` 줄 수(07번 §31-4)."""
    return sum(1 for _sha, row in read_ledger(path)
               if row["kind"] == "call" and row.get("purpose") == purpose
               and row.get("generation_sha256") == generation_sha256)
