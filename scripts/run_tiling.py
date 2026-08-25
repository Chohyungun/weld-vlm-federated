"""타일링 + RT 전량 재인코딩 → 매니페스트 v1 (실행 순서 ④⑤).

    uv run python scripts/run_tiling.py --zips data/raw/aihub71761/_zips \
        --out data/interim/tiles_v1 --manifest-out data/interim/manifest_v1

규격은 `configs/base.yaml` 의 `preprocess.tile` 에서 읽는다. 이 스크립트에 기본값을 두지
않는다. 8/25 게이트에서 잠긴 값이라 코드가 임의로 바꿀 수 없어야 한다.

하는 일

1. v0 매니페스트를 읽어 이미지별 처리 계획을 세운다 (라벨만으로 결정. 픽셀 미접근)
2. 원천 zip 에서 이미지를 읽어 계획대로 잘라 **새 경로**에 다시 인코딩한다
3. 폐기분을 빼고 `rel_path`·`sha256`·`width_px`·`height_px` 를 갱신해 v1 을 쓴다
4. 폐기 회계를 셀별로 남기고 정상 셀 5% 상한과 대조한다

**원본을 고치지 않는다.** zip 은 읽기만 하고 결과는 `--out` 아래에만 쓴다.
**잠그지 않는다.** SNAPSHOT 잠금은 프로브 판정 뒤 총괄이 연다.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.convert.encode_runner import build_v1, encode_all
from data.convert.tiling import (
    REASON_CROPPED_BAND,
    REASON_OK,
    REASON_TILED,
    plan_tile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_spec() -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    t = cfg["preprocess"]["tile"]
    if t["k"] != 1 or t["padding"] != "forbidden":
        raise ValueError("확정 규격과 다르다. configs/base.yaml 을 확인하라")
    if t["tau_mode"] != "band_containment":
        raise ValueError(f"tau_mode 가 확정값과 다르다: {t['tau_mode']}")
    return {
        "tile_w": int(t["tile_size"][0]), "tile_h": int(t["tile_size"][1]),
        "stride_x": int(t["stride_x"]), "align": 8 if t["align8"] else 1,
        "seed": int(t["selection_seed"]),
        "mode": t["encode"]["mode"],
        "quality": t["encode"]["quality"],
        "progressive": bool(t["encode"]["progressive"]),
        "optimize": bool(t["encode"]["optimize"]),
        "discard_limits": cfg["preprocess"]["discard_limits"],
    }


def index_zip_members(zip_dir: Path) -> dict[str, tuple[Path, str]]:
    """파일명(확장자 제외) → (zip 경로, 멤버 이름).

    **id 로 색인하지 않는다.** 라벨의 `info.id` 는 10건에서 파일명 꼬리와 다르다
    (예: id 13121 의 파일은 `RT_ST_02_62757373.jpg`). id 로 원천을 찾으면 그 10장을
    조용히 잃는다. 라벨이 들고 있는 `image_data.file_name` 만이 원천과 이어지는 키다.
    """
    idx: dict[str, tuple[Path, str]] = {}
    for zp in sorted(zip_dir.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx[Path(name).stem] = (zp, name)
    return idx


def band_of(annotations_normal: dict[str, tuple[int, int, int, int]], image_id: str):
    return annotations_normal.get(image_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0", type=Path, default=REPO_ROOT / "data/interim/manifest_v0")
    ap.add_argument("--zips", type=Path, default=REPO_ROOT / "data/raw/aihub71761/_zips")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/interim/aihub_labels")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/interim/tiles_v1")
    ap.add_argument("--manifest-out", type=Path, default=REPO_ROOT / "data/interim/manifest_v1")
    ap.add_argument("--limit", type=int, default=0, help="시험 실행용. 0 이면 전량")
    ap.add_argument("--encode", action="store_true",
                    help="본 실행. 붙이지 않으면 계획만 낸다")
    args = ap.parse_args()

    spec = load_spec()
    # 품질이 비어 있어도 계획 단계는 돈다. 계획은 라벨만 쓰고 픽셀을 읽지 않는다.
    # 재인코딩에만 품질이 필요하므로 그 직전에 막는다.

    m = pd.read_csv(args.v0 / "manifest.csv", dtype=str, keep_default_na=False)
    print(f"v0 매니페스트 {len(m):,}행")

    # 정상 이미지의 밴드 폴리곤 (라벨에서 직접 읽는다. annotations.csv 에는 없다)
    from scripts.measure_tiling_geometry import read_labels

    bands: dict[str, tuple[int, int, int, int]] = {}
    file_names: dict[str, str] = {}
    for r in read_labels(args.labels):
        if r.modality != "RT":
            continue
        image_id = f"aihub71761:{r.image_id}"
        file_names[image_id] = r.file_name
        if not r.is_normal or not r.polys:
            continue
        xs, ys = max(r.polys, key=lambda p: max(p[1]) - min(p[1]))
        bands[image_id] = (min(xs), min(ys), max(xs), max(ys))
    print(f"정상 밴드 폴리곤 {len(bands):,}개 · 원천 파일명 {len(file_names):,}개")

    print("멤버 색인 작성 중 (zip 읽기)")
    members = index_zip_members(args.zips)
    print(f"  원천 이미지 {len(members):,}개")

    rows = list(m.itertuples(index=False))
    if args.limit:
        rows = rows[: args.limit]

    plans = []
    for row in rows:
        is_normal = row.has_defect != "True"
        plans.append(plan_tile(
            image_id=row.image_id, width=int(row.width_px), height=int(row.height_px),
            is_normal=is_normal, band=bands.get(row.image_id),
            tile_w=spec["tile_w"], tile_h=spec["tile_h"], stride_x=spec["stride_x"],
            align=spec["align"], seed=spec["seed"],
        ))

    acct: dict[tuple[str, str, str], int] = defaultdict(int)
    for row, plan in zip(rows, plans, strict=True):
        cell = ("normal" if row.has_defect != "True" else "defect", row.material, plan.reason)
        acct[cell] += 1

    print("\n처리 계획 (픽셀 미접근):")
    for (kind, mat, reason), n in sorted(acct.items()):
        print(f"  {kind:7s} {mat:3s} {reason:24s} {n:,}")

    keep = sum(1 for p in plans if p.keep)
    print(f"\n유지 {keep:,} / 폐기 {len(plans)-keep:,}")

    # 폐기 회계와 상한 대조
    report = accounting_report(rows, plans, spec["discard_limits"])
    for line in report["lines"]:
        print("  " + line)

    args.manifest_out.mkdir(parents=True, exist_ok=True)
    with (args.manifest_out / "tile_accounting.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n회계 기록: {args.manifest_out / 'tile_accounting.json'}")
    if spec["quality"] is None:
        print("\n재인코딩은 아직 못 돈다: configs/base.yaml 의 encode.quality 가 비어 있다.")
        return 3
    if not args.encode:
        print("\n계획만 냈다. 본 실행은 --encode 를 붙인다.")
        return 0

    plan_by_id = {p.image_id: p for p in plans}
    rc = encode_all(rows, plan_by_id, members, file_names, spec, args)
    if rc:
        return rc
    return build_v1(args, spec, plan_by_id, report)


def accounting_report(rows, plans, limits: dict) -> dict:
    per: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row, plan in zip(rows, plans, strict=True):
        kind = "normal" if row.has_defect != "True" else "defect"
        per[f"{kind}|{row.material}"]["total"] += 1
        per[f"{kind}|{row.material}"][plan.reason] += 1
        if not plan.keep:
            per[f"{kind}|{row.material}"]["discarded"] += 1

    lines, cells = [], {}
    normal_rates = {}
    for cell, c in sorted(per.items()):
        rate = c["discarded"] / c["total"] * 100 if c["total"] else 0.0
        cells[cell] = {
            "total": c["total"], "discarded": c["discarded"],
            "discard_rate_pct": round(rate, 3),
            "oversized_band_cropped": c.get(REASON_CROPPED_BAND, 0),
            "tiled": c.get(REASON_TILED, 0),
            "passthrough": c.get(REASON_OK, 0),
        }
        if cell.startswith("normal|"):
            normal_rates[cell.split("|")[1]] = rate
            limit = limits["normal_per_material"] * 100
            flag = "초과" if rate > limit else "이내"
            lines.append(f"{cell:12s} 폐기 {rate:6.3f}% (상한 {limit}%) {flag} · "
                         f"밴드중심크롭 {c.get(REASON_CROPPED_BAND, 0):,}")
    if len(normal_rates) == 2:
        gap = abs(normal_rates.get("AL", 0) - normal_rates.get("ST", 0))
        limit = limits["material_gap_pp"] * 100
        lines.append(f"재질 간 정상 폐기율 차 {gap:.3f}%p (상한 {limit}%p) "
                     f"{'초과' if gap > limit else '이내'}")
    return {"cells": cells, "lines": lines, "limits": limits}


if __name__ == "__main__":
    raise SystemExit(main())
