"""동결 스냅샷 → 평가셋 정답. **다섯 칸이 같은 정답으로 채점된다는 보증의 한 지점.**

채점기가 하나여도 정답을 칸마다 따로 만들면 공정성은 그 지점에서 무너진다. 그래서
매니페스트·어노테이션 읽기를 여기 한 곳에 둔다 — 65번(검출 3칸)과 66번(통합형 2칸)이
같은 함수를 호출한다.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

Box = tuple[float, float, float, float]


def read_manifest(snapshot: str | Path) -> list[dict[str, str]]:
    with (Path(snapshot) / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_gold(
    snapshot: str | Path, eval_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, list[tuple[str, Box]]]]:
    """평가셋 GT — 이미지 수준 클래스 집합과 bbox 목록.

    bbox 는 원본 픽셀 그대로다. 정규화·클리핑을 하지 않는다.
    """
    codes: dict[str, set[str]] = defaultdict(set)
    boxes: dict[str, list[tuple[str, Box]]] = defaultdict(list)
    with (Path(snapshot) / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            iid = r["image_id"]
            if iid not in eval_ids:
                continue
            codes[iid].add(r["iso_code"])
            if r.get("bbox_x1_px"):
                boxes[iid].append((
                    r["iso_code"],
                    (float(r["bbox_x1_px"]), float(r["bbox_y1_px"]),
                     float(r["bbox_x2_px"]), float(r["bbox_y2_px"])),
                ))
    return codes, boxes


def eval_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["split"] == "eval"]


def image_sizes(rows: list[dict[str, str]]) -> dict[str, tuple[int, int]]:
    """image_id → (W, H). 경계 이탈 집계용이며 좌표를 고치는 데 쓰지 않는다."""
    return {r["image_id"]: (int(r["width_px"]), int(r["height_px"])) for r in rows}
