"""검산 스크립트 공용 — 동결본 읽기·지문 검증·산출 경로 제한.

세 스크립트가 같은 규약을 쓰도록 한 곳에 모은다. 규약이 스크립트마다 다르면 같은 자료에서
다른 수가 나온다.

  · 동결본은 **읽기 전에 지문을 검증한다.** 손으로 한 번 확인한 것은 다음 사람에게 남지 않는다.
  · 결측 행은 **버리지 않고 센다.** 조용히 사라지면 개수 차이의 사유를 산출만 보고 알 수 없다.
  · 산출은 **허용한 곳에만 쓴다.** 동결 디렉터리와 외부 드라이브에 쓰지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT      # 정션으로 본체 자산이 풀린다. 로컬 절대경로를 코드에 두지 않는다
MANI = MAIN / "data/interim/manifest_v1"
ZIPS = MAIN / "data/raw/aihub71761/_zips"
csv.field_size_limit(10_000_000)

DIGITS = re.compile(r"(\d+)")

#: 상자 자르기 상한. **`W`·`H` 다. `W-1` 이 아니다.**
#: 동결본에 폭 1280 영상의 `bbox_x2_px = 1280` 인 행이 실제로 있다(188행). 실물로 확인한 값이다.
CLAMP_NOTE = "x 는 [0, W], y 는 [0, H] 로 자른다. 상한은 W·H 이고 W-1·H-1 이 아니다"


class SnapshotChanged(RuntimeError):
    """동결본의 지문이 기록과 다르다. 검산을 진행하지 않는다."""


def verify_snapshot(mani: Path = MANI) -> dict:
    """`SNAPSHOT.sha256` 의 항목을 **코드 안에서** 검증한다.

    손으로 한 번 맞춰 본 것은 다음 사람이 돌릴 때 남지 않는다. 어긋나면 값을 내지 않고 멈춘다.
    """
    lines = (mani / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines()
    checked, digest = {}, None
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            if "snapshot_digest" in line:
                digest = line.split()[-1]
            continue
        if not line:
            continue
        want, name = line.split(None, 1)
        got = hashlib.sha256((mani / name.strip()).read_bytes()).hexdigest()
        if got != want:
            raise SnapshotChanged(f"{name.strip()}: 기록 {want[:12]}… 실제 {got[:12]}…")
        checked[name.strip()] = want
    return {"검증한_항목": sorted(checked), "snapshot_digest": digest}


def safe_out(path: Path) -> Path:
    """산출 경로를 제한한다. 동결 디렉터리·원본·외부 드라이브에 쓰지 않는다."""
    p = Path(path).resolve()
    for bad in (MANI.resolve(), (MAIN / "data/raw").resolve()):
        if p == bad or bad in p.parents:
            raise ValueError(f"이 경로에는 쓰지 않는다: {p}")
    if p.drive.upper() not in {ROOT.drive.upper(), MAIN.drive.upper()}:
        raise ValueError(f"저장소 밖 드라이브에 쓰지 않는다: {p}")
    return p


def num_key(s: str) -> str:
    """식별자의 숫자부. 매니페스트는 `출처:숫자` 꼴이다."""
    m = DIGITS.findall(s or "")
    return m[-1] if m else ""


def id_tail(s: str) -> str:
    """jsonl `image_id`(`RT_AL_01_14487428`)의 **맨 뒤 마디**.

    `orig_info_id` 와 **다른 결합 키다.** 둘이 갈리는 장이 있어 같은 키로 취급하면 안 된다.
    """
    return (s or "").rsplit("_", 1)[-1]


def load_manifest(mani: Path = MANI) -> dict:
    """동결 매니페스트. 두 번 읽지 않도록 필요한 열을 한 번에 담는다."""
    out = {}
    with (mani / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["image_id"]] = {
                "num": num_key(r["image_id"]),
                "stem": Path(r["rel_path"]).stem,
                "modality": r["modality"], "material": r["material"],
                "client": r["client"], "split": r["split"],
                "has_defect": r["has_defect"], "n_defects": r["n_defects"],
                "wh": (r["width_px"], r["height_px"]),
            }
    return out


def load_boxes(want: set, mani: Path = MANI) -> tuple[dict, Counter]:
    """결함 상자. **결측 행을 버리지 않고 센다.**

    `annotations.csv` 에는 좌표가 빈 행이 있다(`geom_valid=False`). 조용히 버리면 장마다 상자 수가
    모자라 보이고, 그 차이의 사유를 산출만 보고는 알 수 없다. 그래서 수와 표시를 함께 돌려준다.
    """
    boxes, acct = {}, Counter()
    with (mani / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["image_id"] not in want:
                continue
            acct["겹친_장의_주석행"] += 1
            try:
                boxes.setdefault(r["image_id"], []).append(
                    (int(float(r["bbox_x1_px"])), int(float(r["bbox_y1_px"])),
                     int(float(r["bbox_x2_px"])), int(float(r["bbox_y2_px"]))))
            except (KeyError, TypeError, ValueError):
                boxes.setdefault(r["image_id"], [])
                acct["상자가_빈_행"] += 1
                acct[f"빈_행의_표시:{(r.get('geom_flags') or '').strip() or '(없음)'}"] += 1
    return boxes, acct


def clamp(box: tuple, w: int, h: int) -> tuple:
    """상자를 프레임 안으로 자른다. 상한은 **W·H** 다(`CLAMP_NOTE`)."""
    x1, y1, x2, y2 = box
    return (min(max(x1, 0), w), min(max(y1, 0), h),
            min(max(x2, 0), w), min(max(y2, 0), h))
