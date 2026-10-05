"""스냅샷 밖 기록의 쓰기와 해시 확인 읽기 (2판 8절 m-11).

기록은 UTF-8 · LF 의 CSV 다. 쓰는 쪽은 바이트의 sha256 을 돌려주고, 읽는 쪽은 그 해시를 받아 맞댄 뒤에만 읽는다.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from collections.abc import Iterable
from pathlib import Path


class RecordHashMismatch(RuntimeError):
    """기록의 바이트가 적힌 해시와 다르다."""


def write_csv(path: Path, header: list[str], rows: Iterable[dict]) -> str:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n", extrasaction="raise")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    data = buf.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def read_csv_checked(path: Path, sha256: str) -> list[dict]:
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256.lower():
        raise RecordHashMismatch(f"{path.name} 의 해시가 다르다: {got[:16]}… ≠ {sha256[:16]}…")
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
