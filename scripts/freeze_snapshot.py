"""분할 동결 — tiles.csv 승격 + SNAPSHOT 잠금 + 동결 구성 기록.

R3 잔존 수용 판정(57번 문서)에 따른 동결이다. **분할·층화·묶음을 다시 계산하지 않는다.**
현행 매니페스트(정합본)를 그대로 잠근다.

    uv run python scripts/freeze_snapshot.py

하는 일 세 가지.
1. `encode_progress.jsonl` 의 이미지별 출처를 `tiles.csv` 로 승격 (1:1·누락 0 확인)
2. `write_snapshot` 으로 4파일을 하나의 SNAPSHOT.sha256 에 잠그고 읽기 전용 속성 부여
3. 동결 시점 구성(분할·클라이언트·출처·클래스)을 표로 출력

잠근 뒤 재생성 금지. 해시는 논문에 싣는다.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.manifest_io import (
    ANNOTATIONS_FILENAME,
    CAPABILITIES_FILENAME,
    MANIFEST_FILENAME,
    REASON_TO_PROVENANCE,
    SNAPSHOT_FILENAME,
    TILES_FILENAME,
    file_sha256,
    read_annotations,
    read_manifest,
    write_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"
SNAPSHOT_ID = "aihub71761_rt_v1"


def build_tiles(manifest: pd.DataFrame) -> pd.DataFrame:
    """진행 로그의 출처를 동결 산출물로 승격한다. 매니페스트와 1:1 이어야 한다."""
    reasons: dict[str, str] = {}
    for line in (V1 / "encode_progress.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            reasons[rec["image_id"]] = rec["reason"]

    ids = list(manifest["image_id"])
    missing = [i for i in ids if i not in reasons]
    if missing:
        raise SystemExit(f"출처가 없는 이미지 {len(missing)}장: {missing[:5]}")

    tiles = pd.DataFrame({
        "image_id": ids,
        "provenance": [REASON_TO_PROVENANCE[reasons[i]] for i in ids],
        "reason": [reasons[i] for i in ids],
    }).astype("string")

    extra = len(set(reasons) - set(ids))
    print(f"tiles.csv: {len(tiles):,}행 · 매니페스트와 1:1 · 누락 0건 "
          f"(로그에만 있는 폐기분 {extra:,}건은 제외)")
    assert len(tiles) == len(manifest) and tiles["image_id"].is_unique
    return tiles


def main() -> int:
    m = read_manifest(V1 / MANIFEST_FILENAME)
    a = read_annotations(V1 / ANNOTATIONS_FILENAME)
    print(f"매니페스트 {len(m):,}행 · 결함 인스턴스 {len(a):,}행")

    tiles = build_tiles(m)
    split_meta = json.loads((V1 / "split_meta_e3.json").read_text(encoding="utf-8"))


    capabilities = {
        "generated_at": "2026-08-31T00:00:00+09:00",
        "snapshot_id": SNAPSHOT_ID,
        "source": "aihub71761",
        "is_mock": False,
        "counts": {
            "images_total": len(m),
            "with_thickness": int((m["thickness_source"] != "none").sum()),
            "with_pixel_scale": int((m["scale_source"] != "none").sum()),
            "with_quality_level": int((m["quality_level"] != "").sum()),
        },
        "capabilities": {
            "localization": True,
            "thickness_mm": False,
            "pixel_scale": False,
            "size_mm": False,
            "verdict_mode": "clause_only",
        },
        "assumptions": {"thickness_mm": None, "px_per_mm": None,
                        "quality_level": None, "rationale": None},
        "split_meta": {
            "seed": split_meta["seed"],
            "eval_folds": 5, "val_folds": 10,
            "dirichlet": split_meta["dirichlet"],
            "groups": split_meta["groups"],
        },
        "preprocessing": {
            "lineage": ["tiling_v1", "border_mask_8pct_fill114", "histmatch_trainpool_mean"],
            "tile_rule_version": "v1",
            "encode": {"mode": "L", "quality": 74},
            "border_mask": {"frac": 0.08, "fill": 114},
            "histmatch_reference": {"p1": 6, "p50": 114, "p99": 226, "mean": 113.97},
            "images_dir": "data/interim/tiles_v1_histmatch",
        },
    }

    digest = write_snapshot(V1, m, a, capabilities, tiles=tiles)
    print(f"\nSNAPSHOT 잠금 완료 · snapshot_digest {digest}")
    locked = [MANIFEST_FILENAME, ANNOTATIONS_FILENAME, TILES_FILENAME,
              CAPABILITIES_FILENAME, SNAPSHOT_FILENAME]
    for name in locked:
        print(f"  {file_sha256(V1 / name)}  {name}")

    # 읽기 전용 속성 (불변조건 2). 실질 잠금은 verify_snapshot 이고 속성은 이중 안전장치다.
    for name in locked:
        (V1 / name).chmod(stat.S_IREAD)
    print("읽기 전용 속성 부여 완료")

    # ---- 동결 구성 기록 ----
    df = m.copy()
    df["defect"] = df["has_defect"].astype(bool)
    df["prov"] = df["image_id"].map(dict(zip(tiles["image_id"], tiles["provenance"],
                                             strict=True)))
    print("\n[split × 결함/정상]")
    print(df.groupby(["split", "defect"]).size().unstack(fill_value=0).to_string())
    print("\n[client × 재질]")
    print(df[df["client"].notna()].groupby(["client", "material"]).size()
          .unstack(fill_value=0).to_string())
    print("\n[출처 분포 (전체 / split 별)]")
    print(df["prov"].value_counts().to_string())
    print(df.groupby(["split", "prov"]).size().unstack(fill_value=0).to_string())
    print("\n[클래스 × client] (결함 이미지, defect_types 포함 기준)")
    for cls in ("crack", "lack_of_fusion", "porosity", "slag_inclusion"):
        has = df["defect_types"].fillna("").str.contains(cls)
        row = df[has].groupby("client", dropna=False).size()
        counts = {str(k): int(v) for k, v in row.items()}
        print(f"  {cls:16s} {counts}")
    print("\n[split × 클래스]")
    for cls in ("crack", "lack_of_fusion", "porosity", "slag_inclusion"):
        has = df["defect_types"].fillna("").str.contains(cls)
        row = df[has].groupby("split").size()
        print(f"  {cls:16s} {({str(k): int(v) for k, v in row.items()})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
