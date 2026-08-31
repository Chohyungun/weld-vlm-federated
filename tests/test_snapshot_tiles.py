"""tiles.csv 선택적 스냅샷 멤버 — 함께 잠기고, 옛 스냅샷은 그대로 검증된다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.manifest_io import (
    REASON_TO_PROVENANCE,
    TILES_FILENAME,
    ManifestError,
    SnapshotVerificationError,
    load_snapshot,
    read_tiles,
    verify_snapshot,
    write_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK = REPO_ROOT / "data" / "mock" / "mock_aihub_v1"


def make_snapshot(tmp_path: Path, with_tiles: bool):
    snap = load_snapshot(MOCK)
    tiles = None
    if with_tiles:
        tiles = pd.DataFrame({
            "image_id": snap.manifest["image_id"],
            "provenance": "N-crop",
            "reason": "ok",
        }).astype("string")
    digest = write_snapshot(tmp_path, snap.manifest, snap.annotations,
                            snap.capabilities, tiles=tiles)
    return digest


def test_tiles_locked_together_and_loaded(tmp_path: Path) -> None:
    make_snapshot(tmp_path, with_tiles=True)
    assert TILES_FILENAME in (tmp_path / "SNAPSHOT.sha256").read_text(encoding="utf-8")
    snap = load_snapshot(tmp_path)
    assert snap.tiles is not None
    assert len(snap.tiles) == len(snap.manifest)
    assert set(snap.tiles["provenance"]) <= set(REASON_TO_PROVENANCE.values())


def test_snapshot_without_tiles_still_verifies(tmp_path: Path) -> None:
    """옛 규약 스냅샷(3파일)은 그대로 유효하다. tiles 는 None 으로 온다."""
    make_snapshot(tmp_path, with_tiles=False)
    snap = load_snapshot(tmp_path)
    assert snap.tiles is None


def test_tampered_tiles_fails_verification(tmp_path: Path) -> None:
    """tiles.csv 를 고치면 스냅샷 전체가 무효다 — 함께 잠겼다는 뜻이 이것이다."""
    make_snapshot(tmp_path, with_tiles=True)
    p = tmp_path / TILES_FILENAME
    p.write_text(p.read_text(encoding="utf-8").replace("N-crop", "N-tile", 1),
                 encoding="utf-8")
    with pytest.raises(SnapshotVerificationError):
        verify_snapshot(tmp_path)


def test_read_tiles_rejects_unknown_provenance(tmp_path: Path) -> None:
    p = tmp_path / TILES_FILENAME
    p.write_text("image_id,provenance,reason\na:1,N-magic,ok\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        read_tiles(p)
