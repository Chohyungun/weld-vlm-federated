"""노출 원장 — 리허설 · 진단이 생성에 쓴 이미지 목록을 해시와 함께 남긴다(리허설 2판 §5-4).

`outputs/exposure/uni_exposure_ledger.jsonl` 는 **덧붙이기만** 하고 앞 바이트가 그대로 접두다. 목록 사본은
`outputs/exposure/lists/<실행 id>_<gen|echo>.txt` 에 배타 생성한다. 줄의 꼴:

    {event, run_id, purpose, tag, seed_index, mode, list_kind, list_path, list_file_sha256, list_set_sha256,
     n, snapshot_digest, commit, at}  (+ `n_written` — `generated` 줄만)

`event` 는 둘이다 — `planned`(목록 단계, **생성보다 먼저**) · `generated`(export 가 끝 줄과 곁 파일을 쓴 뒤).
`planned` 가 있으면 노출된 것으로 본다(보수적으로 읽는다). 본실험(`main`)은 이 원장에 쓰지 않는다.

덧붙이기 전에 파일의 끝 바이트가 LF 인지, 모든 줄이 JSON 객체인지 본다 — 아니면 덧붙이지 않고 멈춘다.
쓰는 동안 `.lock` 을 배타 생성으로 잡는다(여러 실행이 같은 원장에 쓸 때 한 줄이 섞이지 않게). 원장의 자리는
**본체 체크아웃** 아래다 — 정션으로 모든 작업 트리가 같은 곳을 본다. 시험은 경로를 파이썬 인자로 준다.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "EVENTS",
    "EXPOSURE_REL",
    "ROW_KEYS",
    "ExposureError",
    "append_row",
    "default_ledger",
    "find_planned",
    "read_rows",
    "write_list_copy",
]

EXPOSURE_REL = Path("outputs/exposure/uni_exposure_ledger.jsonl")
EVENTS = ("planned", "generated")
ROW_KEYS = ("event", "run_id", "purpose", "tag", "seed_index", "mode", "list_kind", "list_path",
            "list_file_sha256", "list_set_sha256", "n", "snapshot_digest", "commit", "at")
_LOCK_TRIES = 50


class ExposureError(ValueError):
    """노출 원장을 읽거나 쓰지 못한다. `code` 가 사유다 — 쓰지 않고 멈춘다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def default_ledger() -> Path:
    """정본 자리 — 본체 체크아웃의 `outputs/exposure/uni_exposure_ledger.jsonl`."""
    from evaluation.entry_gate import main_checkout

    return main_checkout() / EXPOSURE_REL


def read_rows(path: Path) -> list[dict]:
    """원장 전체를 읽는다. 바이트 규약(끝 LF · UTF-8 · 줄마다 JSON 객체)이 어긋나면 `ExposureError`. 없으면 빈 목록."""
    p = Path(path)
    if not p.exists():
        return []
    raw = p.read_bytes()
    if not raw:
        return []
    if not raw.endswith(b"\n") or b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
        raise ExposureError("ledger_broken", f"노출 원장의 바이트가 규약 밖이다(끝 LF · CR · BOM): {p}")
    rows = []
    for i, line in enumerate(raw.decode("utf-8")[:-1].split("\n"), 1):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            raise ExposureError("ledger_broken", f"노출 원장 {i} 번째 줄이 JSON 이 아니다") from None
        if not isinstance(row, dict) or row.get("event") not in EVENTS:
            raise ExposureError("ledger_broken", f"노출 원장 {i} 번째 줄의 꼴이 틀리다")
        rows.append(row)
    return rows


def _lock(path: Path):
    lock = Path(path).parent / ".lock"
    for _ in range(_LOCK_TRIES):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock
        except FileExistsError:
            time.sleep(0.1)
    raise ExposureError("ledger_locked", f"노출 원장의 잠금을 얻지 못했다: {lock}")


def append_row(path: Path, row: Mapping[str, Any]) -> bytes:
    """한 줄을 덧붙인다 — 잠금 → 앞 줄 검사 → `write` 한 번 · fsync → 잠금 해제. 쓴 줄의 바이트를 돌려준다."""
    missing = [k for k in ROW_KEYS if k not in row]
    if missing or row.get("event") not in EVENTS:
        raise ExposureError("row_form", f"노출 원장 줄의 꼴이 틀리다: 빠진 키 {missing}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock(p)
    try:
        read_rows(p)
        line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        with open(p, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        return line
    finally:
        lock.unlink(missing_ok=True)


def find_planned(path: Path, *, run_id: str, list_kind: str, list_file_sha256: str) -> dict | None:
    """같은 실행 id · 같은 목록 종류 · 같은 목록 파일 해시의 `planned` 줄. 없으면 None."""
    for row in read_rows(path):
        if (row["event"] == "planned" and row.get("run_id") == run_id and row.get("list_kind") == list_kind
                and row.get("list_file_sha256") == list_file_sha256):
            return row
    return None


def write_list_copy(ledger: Path, run_id: str, list_kind: str, raw: bytes, *, suffix: str = ".txt") -> Path:
    """목록 사본을 `lists/<실행 id>_<종류><suffix>` 에 배타 생성하고 fsync 한다. 이미 있으면 `ExposureError`.

    곁에 id 마다 지금 판의 `group_id` 를 싣는 파일(2판 §5-4 · X-13)은 `suffix=".groups.json"` 으로 같은 함수가 쓴다."""
    d = Path(ledger).parent / "lists"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}_{list_kind}{suffix}"
    try:
        with open(p, "xb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError:
        raise ExposureError("list_copy_exists", f"목록 사본이 이미 있다: {p.name}") from None
    return p
