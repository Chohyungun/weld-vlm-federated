"""히스토그램 정합이 결함 대비를 얼마나 깎는가 — 검출 성능 영향의 대리 지표.

결함 화소와 **인접 배경** 화소의 밝기 차를 정합 전후로 비교한다. 정합은 이미지별 단조
변환이라 화소 순서는 보존되지만, 영역 평균의 차이는 압축될 수 있다. 대비가 줄면 검출이
어려워진다.

**순서 역전도 함께 본다.** 단조 사상이므로 화소 수준 역전은 원리적으로 불가능하지만,
영역 평균은 분포가 겹칠 때 뒤집힐 수 있다. 그것이 실제로 일어나는지 센다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"

#: 인접 배경 고리의 두께(화소). 결함 주변을 두르되 다른 결함까지 삼키지 않을 폭.
RING_PX = 12


def ring_mask(poly: np.ndarray) -> np.ndarray:
    """폴리곤 바깥을 두르는 고리. 결함 자신은 뺀다."""
    kernel = np.ones((RING_PX * 2 + 1, RING_PX * 2 + 1), np.uint8)
    dilated = cv2.dilate(poly.astype(np.uint8), kernel, iterations=1).astype(bool)
    return dilated & ~poly


def run_contrast(args) -> int:
    from data.frozen_guard import legacy_path
    from data.manifest_io import read_manifest
    from scripts.measure_clamp_feasibility import poly_mask
    from scripts.measure_tiling_geometry import read_labels

    # 정합 이전 판은 attic/ 으로 격리됐다 (80번 G11-1). 경로를 직접 쓰지 않는다.
    before = read_manifest(legacy_path("manifest_pre_histmatch.csv")).set_index("image_id")
    after = read_manifest(V1 / "manifest.csv").set_index("image_id")

    prog = {}
    for line in (V1 / "encode_progress.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            prog[r["image_id"]] = r

    labels = {}
    for r in read_labels(args.labels):
        if r.modality == "RT":
            labels[f"aihub71761:{r.image_id}"] = r

    pool = before[(before["split"] != "eval") & before["has_defect"].astype(bool)]
    pool = pool.sort_index()
    if len(pool) > args.per_pop:
        pool = pool.iloc[:: len(pool) // args.per_pop].head(args.per_pop)
    print(f"결함 이미지 {len(pool):,}장에서 결함 대비를 잰다 (학습 풀)")

    rows: list[dict] = []
    for image_id, r in pool.iterrows():
        lab = labels.get(image_id)
        if lab is None or not lab.polys:
            continue
        arr_b = np.asarray(Image.open(REPO_ROOT / r.rel_path).convert("L")).astype(np.float64)
        arr_a = np.asarray(
            Image.open(REPO_ROOT / after.loc[image_id, "rel_path"]).convert("L")
        ).astype(np.float64)
        box = prog[image_id]["box"]
        h, w = arr_b.shape
        for (xs, ys), case in zip(lab.polys, lab.poly_cases, strict=True):
            if not case:
                continue
            poly = poly_mask([(xs, ys)], box, (w, h))
            if poly.sum() < 20:
                continue
            ring = ring_mask(poly)
            if ring.sum() < 20:
                continue
            rows.append({
                "case": case,
                "before": float(arr_b[ring].mean() - arr_b[poly].mean()),
                "after": float(arr_a[ring].mean() - arr_a[poly].mean()),
            })

    if not rows:
        print("!! 잰 폴리곤이 없다.")
        return 3

    b = np.array([x["before"] for x in rows])
    a = np.array([x["after"] for x in rows])
    keep = np.abs(b) > 1e-9
    ratio = a[keep] / b[keep]
    flips = int((np.sign(a[keep]) != np.sign(b[keep])).sum())

    print(f"\n폴리곤 {len(rows):,}개 · 인접 고리 {RING_PX}px")
    print("대비 = 인접 배경 평균 밝기 − 결함 평균 밝기 (양수면 결함이 더 어둡다)")
    print(f"  정합 전  중앙 {np.median(b):7.2f} · 평균 {b.mean():7.2f} · "
          f"결함이 더 어두운 비율 {(b>0).mean()*100:.1f}%")
    print(f"  정합 후  중앙 {np.median(a):7.2f} · 평균 {a.mean():7.2f} · "
          f"결함이 더 어두운 비율 {(a>0).mean()*100:.1f}%")
    print(f"  보존 비율(후/전) 중앙 {np.median(ratio):.3f} · "
          f"p10 {np.percentile(ratio,10):.3f} · p90 {np.percentile(ratio,90):.3f}")
    print(f"  **부호 역전(대비가 뒤집힘) {flips}건 ({flips/len(ratio)*100:.2f}%)**")

    per_case = {}
    print("\n  결함 종류별 대비 중앙값")
    for case in sorted({x["case"] for x in rows}):
        bb = np.array([x["before"] for x in rows if x["case"] == case])
        aa = np.array([x["after"] for x in rows if x["case"] == case])
        per_case[case] = {"n": len(bb), "before_median": round(float(np.median(bb)), 2),
                          "after_median": round(float(np.median(aa)), 2)}
        print(f"    {case:14s} n={len(bb):5,}  {np.median(bb):7.2f} → {np.median(aa):7.2f}")

    out = V1 / "histmatch_contrast.json"
    out.write_text(json.dumps({
        "ring_px": RING_PX, "n_polygons": len(rows),
        "before_median": round(float(np.median(b)), 3),
        "after_median": round(float(np.median(a)), 3),
        "retention_median": round(float(np.median(ratio)), 4),
        "retention_p10": round(float(np.percentile(ratio, 10)), 4),
        "retention_p90": round(float(np.percentile(ratio, 90)), 4),
        "sign_flips": flips, "sign_flip_pct": round(flips / len(ratio) * 100, 3),
        "per_case": per_case,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n기록: {out}")
    return 0


if __name__ == "__main__":
    print("이 모듈은 run_hist_match.py --stage contrast 로 호출한다.")
    sys.exit(1)
