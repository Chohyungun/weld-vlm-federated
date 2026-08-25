"""매니페스트 v0 + 예비 분할 + 타일 규칙 캘리브레이션 (실행 순서 ①②③).

40번 4-1 의 순서를 그대로 따른다. **②를 ③보다 먼저** 하는 이유는 전처리 상수가 평가셋
픽셀 통계에서 유도되면 평가 격리가 실질에서 깨지기 때문이다. 캘리브레이션은 예비 분할의
**학습 풀에서만** 돈다.

    uv run python scripts/build_manifest_v0.py

산출물 (`data/interim/manifest_v0/`)

| 파일 | 내용 |
|---|---|
| `manifest.csv` · `annotations.csv` | 계약 #2 스키마. **잠그지 않는다** |
| `calibration.json` | 학습 풀에서만 유도한 타일 규칙 값과 폐기 회계 |

**v0 는 스냅샷이 아니다.** 원천 이미지가 로컬에 없어 `sha256` 을 채울 수 없고, 규격은 라벨
JSON 의 width/height 를 읽는다. 타일링 후 실파일에서 `sha256` 과 규격을 다시 읽어 v1 을
만들고 그때 잠근다. 혼동을 막으려고 스냅샷 디렉터리에 쓰지 않고 `data/interim/` 에 둔다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.convert.geometry import polygon_metrics
from data.label_map import load_label_map
from data.manifest_io import (
    ANNOTATION_COLUMNS,
    FLOAT_FORMAT,
    MANIFEST_COLUMNS,
)
from data.split.pipeline import assign_splits
from scripts.measure_tiling_geometry import (
    TILE_H,
    TILE_W,
    ImageRec,
    read_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "aihub71761"
INGEST_VERSION = "aihub-v0-labels"
#: 라벨의 `case` 값 → 계약 #1 L2 키. 사상표가 원본 한글 라벨을 받으므로 여기서 한 번 잇는다.
CASE_TO_RAW = {
    "crack": "균열",
    "porosity": "기공",
    "lackoffusion": "융합불량",
    "slaginclusion": "슬래그혼입",
}
#: 계약 #1 의 `sources.aihub71761` 에도, `eval_spaces.main_rt` 에도 없는 클래스.
#: RT 에 실제로 2장 있다(전수 집계). 계약이 동결 상태라 임의로 넓히지 않고 폐기하며
#: 건수를 회계에 남긴다.
OUT_OF_LABEL_SPACE = {"incompletepenetration"}


def _norm_case(case: str) -> str:
    return case.replace("_", "").replace(" ", "").lower()


def _raw_label(case: str) -> str | None:
    """라벨 case → 사상표 입력(한글 원본 라벨). 결함 종류가 아니면 None.

    `case` 가 빈 폴리곤이 RT 전체에 1건 있다(id 14520899). 이미지 전체를 덮는 1298×723
    폴리곤이라 결함 표시가 아니라 이상 어노테이션이다. 그 폴리곤만 건너뛰고 같은 이미지의
    실제 결함(기공)은 살린다. 결함 이미지가 밴드 폴리곤을 갖는 사례는 아니다.
    """
    key = _norm_case(case)
    if not key or key in OUT_OF_LABEL_SPACE:
        return None
    if key not in CASE_TO_RAW:
        raise KeyError(f"라벨 case {case!r} 를 사상표 입력으로 바꿀 수 없다")
    return CASE_TO_RAW[key]


def group_key_of(rec: ImageRec, runs: dict[int, str]) -> str:
    return runs[rec.image_id]


def build_runs(recs: list[ImageRec]) -> dict[int, str]:
    """연속 id 묶음. 같은 (모달, 재질) 안에서 id 가 1씩 이어지면 한 촬영 묶음으로 본다.

    원천 이미지가 없어 pHash 를 못 돌리므로 v0 의 묶음은 이 축 하나로 만든다.
    타일링 후 v1 에서 pHash(E3)를 더한다. 묶음은 합쳐질 뿐 쪼개지지 않으므로,
    v0 묶음을 기준으로 한 분할은 v1 에서 더 보수적으로만 바뀐다.
    """
    runs: dict[int, str] = {}
    by_axis: dict[tuple[str, str], list[ImageRec]] = defaultdict(list)
    for r in recs:
        by_axis[(r.modality, r.material)].append(r)
    for (mod, mat), group in by_axis.items():
        group.sort(key=lambda r: r.image_id)
        start = group[0].image_id
        prev = start
        for r in group:
            if r.image_id - prev > 1:
                start = r.image_id
            runs[r.image_id] = f"run_{mod}_{mat}_{start:d}"
            prev = r.image_id
    return runs


def to_frames(
    recs: list[ImageRec], lm, runs: dict[int, str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    m_rows, a_rows = [], []
    dropped_out_of_space: list[str] = []
    skipped_polys: list[str] = []
    for rec in recs:
        image_id = f"{SOURCE}:{rec.image_id}"
        defects = []
        polys = [] if rec.is_normal else list(zip(rec.polys, rec.poly_cases, strict=True))
        # 라벨 공간 밖 클래스가 한 장이라도 있으면 그 이미지를 통째로 뺀다.
        # 폴리곤만 빼면 "결함이 있는데 라벨에 없는" 이미지가 정상처럼 남는다.
        if any(_norm_case(c) in OUT_OF_LABEL_SPACE for _, c in polys):
            dropped_out_of_space.append(image_id)
            continue
        seq = 0
        for (xs, ys), case in polys:
            raw = _raw_label(case)
            if raw is None:
                skipped_polys.append(image_id)
                continue
            l2 = lm.to_defect_type(SOURCE, raw)
            if l2 is None:
                continue
            g = polygon_metrics(list(zip(xs, ys, strict=True)), rec.width, rec.height)
            bbox = g.bbox_px or (None, None, None, None)
            defects.append({
                "ann_id": f"{image_id}#{seq}",
                "image_id": image_id,
                "src_label_raw": raw,
                "defect_type": l2,
                "iso_code": lm.iso_code(l2),
                "polygon_json": json.dumps(
                    [[int(x), int(y)] for x, y in zip(xs, ys, strict=True)], separators=(",", ":")
                ),
                "bbox_x1_px": bbox[0], "bbox_y1_px": bbox[1],
                "bbox_x2_px": bbox[2], "bbox_y2_px": bbox[3],
                "area_px": round(g.area_px, 2) if g.area_px else pd.NA,
                "major_axis_px": round(g.major_axis_px, 2) if g.major_axis_px else pd.NA,
                "minor_axis_px": round(g.minor_axis_px, 2) if g.minor_axis_px else pd.NA,
                "equiv_diameter_px": round(g.equiv_diameter_px, 2) if g.equiv_diameter_px else pd.NA,
                "major_axis_mm": pd.NA, "equiv_diameter_mm": pd.NA,
                "geom_valid": g.valid, "geom_flags": g.flags_str,
            })
            seq += 1
        a_rows.extend(defects)
        types = sorted({d["defect_type"] for d in defects})
        m_rows.append({
            "image_id": image_id,
            "source": SOURCE,
            "rel_path": f"data/raw/{SOURCE}/{rec.modality}/{rec.material}/{rec.image_id}.jpg",
            "sha256": "",                       # v0 에는 없다. 타일링 후 실파일에서 읽는다
            "width_px": rec.width,
            "height_px": rec.height,
            "modality": rec.modality,
            "material": rec.material,
            "has_defect": bool(defects),
            "n_defects": len(defects),
            "defect_types": ";".join(types),
            "iso_codes": ";".join(sorted({d["iso_code"] for d in defects})),
            "src_labels_raw": ";".join(sorted({d["src_label_raw"] for d in defects})),
            "label_type": "polygon",
            "has_localization": True,
            "phash_hex": pd.NA,                 # 원천 이미지가 없다. v1 에서 채운다
            "group_id": runs[rec.image_id],
            "group_size": pd.NA,
            "strata_key": pd.NA,
            "split": pd.NA, "client": pd.NA, "eval_subset": pd.NA,
            "thickness_mm": pd.NA, "thickness_source": "none",
            "px_per_mm": pd.NA, "scale_source": "none",
            "quality_level": pd.NA,
            "ingest_version": INGEST_VERSION,
            "label_map_version": lm.version,
            "notes": "",
        })
    m = pd.DataFrame(m_rows, columns=list(MANIFEST_COLUMNS))
    a = pd.DataFrame(a_rows, columns=list(ANNOTATION_COLUMNS))
    m["group_size"] = m.groupby("group_id")["image_id"].transform("size").astype(int)
    return m, a, dropped_out_of_space, skipped_polys


def calibrate(manifest: pd.DataFrame, recs_by_id: dict[str, ImageRec]) -> dict:
    """③ 학습 풀에서만 타일 규칙을 캘리브레이션한다. 평가셋 행은 읽지 않는다."""
    pool = manifest.loc[manifest["split"] != "eval"]
    normals = pool.loc[~pool["has_defect"].astype(bool)]

    heights, contained, oversized = [], 0, 0
    per_material_over: dict[str, int] = defaultdict(int)
    per_material_normal: dict[str, int] = defaultdict(int)
    for row in normals.itertuples():
        rec = recs_by_id[row.image_id]
        # 타일을 실제로 뜨는 대상만 센다. 이미 1280×720 이거나 타일보다 작은 이미지는
        # 밴드 포함 판정 자체가 일어나지 않는다. 그것까지 세면 이미지 경계를 넘는 이상
        # 폴리곤이 '밴드 초과'로 잡혀 수치가 부풀려진다 (실측: 1,124 대 94).
        if (rec.width, rec.height) == (TILE_W, TILE_H):
            continue
        if rec.width < TILE_W or rec.height < TILE_H:
            continue
        per_material_normal[rec.material] += 1
        hs = [max(ys) - min(ys) for _, ys in rec.polys if max(ys) > min(ys)]
        if not hs:
            continue
        h = max(hs)                              # 이미지 안에서 가장 높은 밴드가 기준이다
        heights.append(h)
        if h <= TILE_H:
            contained += 1
        else:
            oversized += 1
            per_material_over[rec.material] += 1

    n = len(heights)
    return {
        "derived_from": "preliminary_split_train_pool_only",
        "pool_images": len(pool),
        "pool_normal_images": len(normals),
        "tile_size": [TILE_W, TILE_H],
        "tau_mode": "band_containment",
        "band_height": {
            "n": n,
            "median": statistics.median(heights) if heights else None,
            "q75": sorted(heights)[int(0.75 * (n - 1))] if n else None,
            "q95": sorted(heights)[int(0.95 * (n - 1))] if n else None,
            "max": max(heights) if heights else None,
        },
        "containment": {
            "contained": contained,
            "oversized": oversized,
            "contained_pct": round(contained / n * 100, 3) if n else None,
            "oversized_pct": round(oversized / n * 100, 3) if n else None,
        },
        "out_of_label_space": {
            "decision": "discard",
            "classes": sorted(OUT_OF_LABEL_SPACE),
            "reason": (
                "계약 #1 sources.aihub71761 과 eval_spaces.main_rt 어디에도 없는 클래스다. "
                "계약이 동결 상태라 임의로 넓히지 않는다. 폴리곤만 빼면 결함 이미지가 "
                "정상처럼 남으므로 이미지 단위로 뺀다."
            ),
            "accounting_cell": "out_of_label_space",
        },
        "oversized_band_policy": {
            "decision": "crop_within_band",
            "reason": (
                "밴드가 타일보다 높으면 포함 관계가 뒤집혀 tile ⊆ band 가 된다. 이때 타일의 "
                "모든 화소가 검사된 용접부 안에 있으므로 정상 승계가 오히려 더 안전하다. "
                "미검사 영역 문제는 반대 방향(tile ⊋ band)에서만 생긴다. 따라서 밴드 중심 "
                "크롭을 허용하고 폐기하지 않는다. "
                "폐기를 택했다면 AL 정상 폐기율이 6.415% 로 4-4 의 5% 상한을 넘어 진행이 "
                "막혔을 것이다 — 상한을 피하려고 고른 규칙이 아니라, 포함 방향이 뒤집힌다는 "
                "사실이 먼저이고 상한 충돌은 그 결과다."
            ),
            "accounting_cell": "oversized_band_cropped",
            "discarded": 0,
            "would_have_discarded_if_policy_were_discard": oversized,
            "cropped_by_material": dict(per_material_over),
            "normal_by_material": dict(per_material_normal),
            "crop_rate_by_material": {
                mat: round(per_material_over.get(mat, 0) / cnt * 100, 3)
                for mat, cnt in per_material_normal.items() if cnt
            },
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/interim/aihub_labels")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/interim/manifest_v0")
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    lm = load_label_map()
    print("① 매니페스트 v0")
    recs = [r for r in read_labels(args.labels) if r.modality == "RT"]
    runs = build_runs(recs)
    manifest, annotations, dropped, skipped = to_frames(recs, lm, runs)
    if dropped:
        print(f"   라벨 공간 밖 클래스로 제외 {len(dropped)}장 (incomplete penetration)")
    if skipped:
        print(f"   결함 종류가 없는 폴리곤 {len(skipped)}개 건너뜀")
    print(f"   이미지 {len(manifest):,} / 결함 인스턴스 {len(annotations):,} / 묶음 {manifest['group_id'].nunique():,}")

    print("② 예비 분할 (잠그지 않는다)")
    iso = {k: v.iso_code for k, v in lm.defect_types.items()}
    manifest, meta = assign_splits(manifest, iso, args.seed)
    print(f"   split {manifest['split'].value_counts().to_dict()}")
    print(f"   client {manifest['client'].value_counts(dropna=True).to_dict()}")
    if meta.dirichlet:
        print(f"   dirichlet {meta.dirichlet}")
    for line in meta.band_report:
        print(f"   [밴드 보고] {line}")

    print("③ 캘리브레이션 (학습 풀에서만)")
    cal = calibrate(manifest, {f"{SOURCE}:{r.image_id}": r for r in recs})
    c = cal["containment"]
    print(f"   밴드 포함 {c['contained']:,} ({c['contained_pct']}%) / "
          f"초과 {c['oversized']:,} ({c['oversized_pct']}%)")
    op = cal["oversized_band_policy"]
    print(f"   밴드 초과 {op['would_have_discarded_if_policy_were_discard']:,}장은 "
          f"폐기하지 않고 밴드 중심 크롭 (재질별 {op['crop_rate_by_material']})")

    args.out.mkdir(parents=True, exist_ok=True)
    for df, name, sort_col in ((manifest, "manifest.csv", "image_id"),
                               (annotations, "annotations.csv", "ann_id")):
        out = df.sort_values(sort_col, kind="stable").reset_index(drop=True)
        for col in out.columns:
            if str(out[col].dtype) == "boolean":
                out[col] = out[col].map({True: "True", False: "False"}).astype("string")
        with (args.out / name).open("w", encoding="utf-8", newline="\n") as fh:
            out.to_csv(fh, index=False, na_rep="", float_format=FLOAT_FORMAT, lineterminator="\n")
    cal["split_meta"] = {
        "seed": meta.seed, "eval_folds": meta.eval_folds, "val_folds": meta.val_folds,
        "dirichlet": meta.dirichlet, "band_report": list(meta.band_report),
    }
    with (args.out / "calibration.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(cal, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"\n기록: {args.out}  (스냅샷 아님. 잠그지 않았다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
