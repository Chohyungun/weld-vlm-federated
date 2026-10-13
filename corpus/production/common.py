"""Small, dependency-free file contracts shared by production stages."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def parse_json(raw: str | bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)


def read_json(path: str | Path) -> dict:
    value = parse_json(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")  # noqa: TRY004 -- invalid file, not caller type
    return value


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line: {path}:{line_number}")
            try:
                value = parse_json(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL object required: {path}:{line_number}")  # noqa: TRY004
            yield value


def write_json_new(path: str | Path, value: Any) -> None:
    """Publish a complete file, refusing any existing destination (including races).

    The temporary file and destination are siblings. A hard link is an atomic
    create-if-absent operation on the supported NTFS/Linux filesystems. A failure
    is reported instead of falling back to an overwriting rename.
    """
    path = Path(path)
    raw = canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def assert_gate_closed(repo_root: str | Path | None = None) -> None:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    git = root / ".git"
    if not git.is_dir():
        raise ValueError("prepare/write requires the primary checkout gate location")
    if any((git / name).exists() for name in ("CTO_GATE_OPEN", "MERGE_HEAD")):
        raise ValueError("repository gate/merge is open; stop repository writes")


def validate_release(path: str | Path) -> dict:
    """Use the independent release verifier; a boolean receipt alone is insufficient.

    No permissive fallback is provided while the release stage is being completed.
    """
    from .release import verify_release

    return verify_release(Path(path))
