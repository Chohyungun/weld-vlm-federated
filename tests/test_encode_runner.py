"""재인코딩 실행부 — 재개, 원천 조인, 좌표 안전장치."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image

from data.convert.encode_runner import build_v1, encode_all, load_progress, repo_rel
from data.convert.tiling import REASON_OK, REASON_TILED, TilePlan

SPEC = {
    "tile_w": 1280, "tile_h": 720, "mode": "L", "quality": 74,
    "progressive": False, "optimize": False,
}


@dataclass
class Row:
    image_id: str
    modality: str = "RT"
    material: str = "ST"


def make_zip(path: Path, members: dict[str, tuple[int, int]]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, (w, h) in members.items():
            buf = BytesIO()
            Image.new("RGB", (w, h), (120, 120, 120)).save(buf, format="JPEG", quality=95)
            z.writestr(name, buf.getvalue())


def make_args(tmp_path: Path, **kw) -> SimpleNamespace:
    return SimpleNamespace(
        out=tmp_path / "tiles", manifest_out=tmp_path / "v1",
        v0=tmp_path / "v0", **kw)


def test_repo_rel_is_posix_and_relative() -> None:
    from data.convert.encode_runner import REPO_ROOT

    assert repo_rel(REPO_ROOT / "data" / "interim" / "x.jpg") == "data/interim/x.jpg"


def test_load_progress_ignores_records_whose_file_vanished(tmp_path: Path) -> None:
    """기록만 있고 파일이 없으면 끝난 것으로 치지 않는다 — 쓰다 죽은 경우다."""
    p = tmp_path / "progress.jsonl"
    p.write_text(
        json.dumps({"image_id": "a", "rel_path": "data/does_not_exist.jpg"}) + "\n",
        encoding="utf-8")
    assert load_progress(p) == {}


def test_missing_source_stops_instead_of_skipping(tmp_path: Path) -> None:
    """원천을 못 찾으면 조용히 건너뛰지 않는다. 회계가 틀린 채 맞아 보이면 안 된다."""
    rows = [Row("aihub71761:1")]
    plans = {"aihub71761:1": TilePlan("aihub71761:1", REASON_OK, (0, 0, 1280, 720), 1)}
    rc = encode_all(rows, plans, {}, {"aihub71761:1": "RT_ST_00_9"}, SPEC,
                    make_args(tmp_path))
    assert rc == 4


def test_join_uses_file_name_not_id(tmp_path: Path) -> None:
    """id 와 파일명 꼬리가 다른 실데이터 10건이 이 경로로 살아난다."""
    zp = tmp_path / "src.zip"
    make_zip(zp, {"RT_ST_02_62757373.jpg": (1280, 720)})
    members = {"RT_ST_02_62757373": (zp, "RT_ST_02_62757373.jpg")}
    rows = [Row("aihub71761:13121")]                      # id 는 파일명과 무관하다
    plans = {"aihub71761:13121": TilePlan("aihub71761:13121", REASON_OK, (0, 0, 1280, 720), 1)}
    args = make_args(tmp_path)

    assert encode_all(rows, plans, members, {"aihub71761:13121": "RT_ST_02_62757373"},
                      SPEC, args) == 0
    recs = [json.loads(x) for x in
            (args.manifest_out / "encode_progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["image_id"] for r in recs] == ["aihub71761:13121"]
    # 출력 이름은 id 다. 원천 파일명은 클래스를 담고 있어 그대로 옮기면 지름길이 된다.
    assert recs[0]["rel_path"].endswith("/13121.jpg")


def test_encodes_to_locked_spec_and_resumes(tmp_path: Path) -> None:
    zp = tmp_path / "src.zip"
    make_zip(zp, {"RT_ST_00_1.jpg": (2560, 720)})
    members = {"RT_ST_00_1": (zp, "RT_ST_00_1.jpg")}
    rows = [Row("aihub71761:1")]
    plans = {"aihub71761:1": TilePlan("aihub71761:1", REASON_TILED, (640, 0, 1920, 720), 3)}
    args = make_args(tmp_path)
    names = {"aihub71761:1": "RT_ST_00_1"}

    assert encode_all(rows, plans, members, names, SPEC, args) == 0
    out = args.out / "RT" / "ST" / "1.jpg"
    with Image.open(out) as im:
        assert im.size == (1280, 720)
        assert im.mode == "L"                     # 그레이스케일 강제
    before = out.read_bytes()

    # 두 번째 호출은 재개다. 이미 끝난 장은 다시 쓰지 않는다.
    assert encode_all(rows, plans, members, names, SPEC, args) == 0
    assert out.read_bytes() == before
    lines = (args.manifest_out / "encode_progress.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_build_v1_stops_when_defect_image_was_cropped(tmp_path: Path) -> None:
    """결함 이미지를 잘랐다면 bbox 를 옮겨야 하는데 그 경로는 규격에 없다. 멈춘다."""
    v0 = tmp_path / "v0"
    v0.mkdir()
    # 계약 모양 그대로 쓴다. 첫 행의 image_id 만 갈아 끼운다.
    mock = Path(__file__).resolve().parents[1] / "data" / "mock" / "mock_aihub_v1"
    for name in ("manifest.csv", "annotations.csv"):
        df = pd.read_csv(mock / name, dtype=str, keep_default_na=False).head(1)
        df["image_id"] = "aihub71761:1"
        with (v0 / name).open("w", encoding="utf-8", newline="\n") as fh:
            df.to_csv(fh, index=False, na_rep="", lineterminator="\n")
    args = make_args(tmp_path)
    args.manifest_out.mkdir(parents=True)
    tile = tmp_path / "t.jpg"
    Image.new("L", (1280, 720)).save(tile)
    (args.manifest_out / "encode_progress.jsonl").write_text(
        json.dumps({"image_id": "aihub71761:1", "rel_path": repo_rel(tile)}) + "\n",
        encoding="utf-8")
    plans = {"aihub71761:1": TilePlan("aihub71761:1", REASON_TILED, (640, 0, 1920, 720), 3)}
    assert load_progress(args.manifest_out / "encode_progress.jsonl")
    assert build_v1(args, SPEC, plans, {"cells": {}}) == 6
