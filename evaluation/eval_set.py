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

ISO_SEP = ";"
"""매니페스트 `iso_codes` 열의 구분자. 정본은 `data/ingest/base.py`(트랙 A)의
`";".join(codes)` 이고 `scripts/build_manifest_v0.py` 도 같다.

**`|` 는 `strata_key` 의 구분자다**(예: `AL|__normal__`). 두 열이 같은 파일에 있어서
헷갈리기 쉽고, 잘못 쪼개면 다중 라벨 이미지가 `"2011;301"` 이라는 없는 코드 하나로
집계돼 클래스 수가 조용히 틀어진다 — 유병률에만 의존하는 자명하한이 그대로 어긋난다.
"""


def parse_iso_codes(value: str | None) -> tuple[str, ...]:
    """매니페스트 `iso_codes` 셀 → 코드 튜플. **구분자를 코드에 하드코딩하지 않는다.**"""
    return tuple(c for c in (value or "").split(ISO_SEP) if c)


_DIGESTS: dict[str, str] = {}
VERIFIED: set[str] = set()
"""이번 프로세스에서 이미 검증한 스냅샷 경로. 같은 스냅샷을 여러 번 열어도 해시는 한 번만
다시 센다 — 검증을 건너뛰기 위한 장치가 아니라 **매번 부르되 비싸지 않게** 하는 장치다."""


def ensure_verified(snapshot: str | Path, *, force: bool = False) -> str:
    """스냅샷 해시를 대조한다. **채점 리더는 이 함수를 지나서만 파일을 연다.**

    이전 판은 `manifest.csv` 를 직접 열었고, 저장소 전체에서 `verify_snapshot` 호출처가
    `manifest_io` 내부와 시험뿐이었다. 변조 사본으로 실증됐다 — 승인 로더는 거부하는데
    채점 리더는 eval 653장·gold 388장을 경고 없이 반환했다(80번 D1).

    "잠금은 OS 읽기 전용이 아니라 이 검증이다"(Q7 확정)라고 계약이 못 박아 두고, 정작
    채점 경로가 그 검증을 지나지 않았다. 여기가 그 배선이다.
    """
    from data.manifest_io import verify_snapshot

    key = str(Path(snapshot).resolve())
    cached = _DIGESTS.get(key)
    if cached is not None and not force:
        return cached
    digest = verify_snapshot(snapshot)
    VERIFIED.add(key)
    _DIGESTS[key] = digest
    return digest


def read_manifest(snapshot: str | Path) -> list[dict[str, str]]:
    ensure_verified(snapshot)
    with (Path(snapshot) / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_gold(
    snapshot: str | Path, eval_ids: set[str]
) -> tuple[dict[str, set[str]], dict[str, list[tuple[str, Box]]]]:
    """평가셋 GT — 이미지 수준 클래스 집합과 bbox 목록.

    bbox 는 원본 픽셀 그대로다. 정규화·클리핑을 하지 않는다.
    """
    ensure_verified(snapshot)
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
