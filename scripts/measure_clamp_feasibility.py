"""R3 판단 재료 — 어두운 쪽 클램프가 결함 신호를 건드리는가.

방사선 사진에서 결함은 **어두운 특징**이다. 어두운 화소를 상수로 눌러 출처 채널을 닫는
변환은 그래서 위험하다. 쓸 수 있는지는 한 가지로 갈린다.

    결함 폴리곤 내부 화소의 분포와, 타일의 어두운 배경(밴드 폴리곤 밖) 화소의 분포가
    **겹치는가.** 겹치지 않으면 배경만 눌러 채널을 닫을 수 있고, 겹치면 그 방법은 죽는다.

측정은 **학습 풀에서만** 한다. 평가셋 화소에서 전처리 상수를 유도하면 불변조건 1-4 가
실질에서 깨진다.

    uv run python scripts/measure_clamp_feasibility.py

**사다리를 추월하지 않는다.** 이 스크립트는 재료만 만든다. 적용 여부는 총괄이 정한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"

#: 후보 클램프 임계. 지시된 3개에 위아래를 붙여 곡선이 보이게 한다.
CANDIDATES = (50, 70, 90, 110, 130)


def pct_from_hist(hist: np.ndarray, qs) -> dict[str, float]:
    """256빈 히스토그램에서 백분위. 전수라 표본 오차가 없다."""
    total = hist.sum()
    if total == 0:
        return {}
    cum = np.cumsum(hist)
    return {f"p{q}": float(np.searchsorted(cum, q / 100 * total)) for q in qs}


def poly_mask(polys, box, size: tuple[int, int]) -> np.ndarray:
    """폴리곤들을 타일 좌표로 옮겨 래스터화한다.

    폴리곤은 **원본 좌표**로 저장돼 있다. 타일은 원본에서 상자만큼 잘라낸 것이라
    상자 원점을 빼야 맞는다. 결함 통과분은 상자가 (0,0,...) 이라 이동량이 0 이다.
    """
    w, h = size
    img = Image.new("L", (w, h), 0)
    drw = ImageDraw.Draw(img)
    ox, oy = box[0], box[1]
    for xs, ys in polys:
        pts = [(float(x) - ox, float(y) - oy) for x, y in zip(xs, ys, strict=True)]
        if len(pts) >= 3:
            drw.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/interim/aihub_labels")
    ap.add_argument("--per-pop", type=int, default=1500, help="모집단별 표본 상한")
    ap.add_argument("-o", "--out", type=Path, default=V1 / "clamp_feasibility.json")
    args = ap.parse_args()

    from data.frozen_guard import legacy_path
    from data.manifest_io import read_manifest
    from scripts.measure_tiling_geometry import read_labels

    # 마스킹 전 경로를 쓴다. 마스킹본은 테두리가 상수 114 라 배경 분포가 오염된다.
    # 그 판은 attic/ 으로 격리됐다 (80번 G11-1). 경로를 직접 쓰지 않는다.
    m = read_manifest(legacy_path("manifest_pre_mask.csv"))
    m = m[m["split"] != "eval"]
    print(f"학습 풀 {len(m):,}장 (평가셋 제외)")

    prog = {}
    for line in (V1 / "encode_progress.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            prog[r["image_id"]] = r

    labels = {}
    for r in read_labels(args.labels):
        if r.modality == "RT":
            labels[f"aihub71761:{r.image_id}"] = r
    print(f"라벨 {len(labels):,}건")

    defect_rows = m[m["has_defect"].astype(bool)].sort_values("image_id")
    normal = m[~m["has_defect"].astype(bool)].copy()
    normal["reason"] = normal["image_id"].map(lambda i: prog[i]["reason"])
    tiled = normal[normal["reason"] == "tiled"].sort_values("image_id")
    cropped = normal[normal["reason"] == "ok"].sort_values("image_id")

    def take(frame):
        if len(frame) <= args.per_pop:
            return frame
        step = len(frame) // args.per_pop
        return frame.iloc[::step].head(args.per_pop)      # 결정론적 균등 추출

    hist_defect = np.zeros(256, dtype=np.int64)
    hist_defect_by_type: dict[str, np.ndarray] = {}
    hist_bg_tile = np.zeros(256, dtype=np.int64)
    hist_bg_crop = np.zeros(256, dtype=np.int64)
    hist_band_tile = np.zeros(256, dtype=np.int64)

    sel = take(defect_rows)
    print(f"\n결함 폴리곤 내부: 이미지 {len(sel):,}장")
    n_poly = 0
    for r in sel.itertuples():
        lab = labels.get(r.image_id)
        if lab is None or not lab.polys:
            continue
        arr = np.asarray(Image.open(REPO_ROOT / r.rel_path).convert("L"))
        h, w = arr.shape
        box = prog[r.image_id]["box"]
        for (xs, ys), case in zip(lab.polys, lab.poly_cases, strict=True):
            if not case:
                continue
            mask = poly_mask([(xs, ys)], box, (w, h))
            if not mask.any():
                continue
            vals = arr[mask]
            hist_defect += np.bincount(vals, minlength=256)
            hist_defect_by_type.setdefault(case, np.zeros(256, dtype=np.int64))
            hist_defect_by_type[case] += np.bincount(vals, minlength=256)
            n_poly += 1
    print(f"  폴리곤 {n_poly:,}개 · 화소 {hist_defect.sum():,}")

    for frame, hist_bg, hist_band, label in (
        (tiled, hist_bg_tile, hist_band_tile, "정상 타일"),
        (cropped, hist_bg_crop, None, "정상 크롭"),
    ):
        sel = take(frame)
        print(f"{label} 배경(밴드 폴리곤 밖): 이미지 {len(sel):,}장")
        for r in sel.itertuples():
            lab = labels.get(r.image_id)
            if lab is None or not lab.polys:
                continue
            arr = np.asarray(Image.open(REPO_ROOT / r.rel_path).convert("L"))
            h, w = arr.shape
            box = prog[r.image_id]["box"]
            band = poly_mask(lab.polys, box, (w, h))
            outside = arr[~band]
            if outside.size:
                hist_bg += np.bincount(outside, minlength=256)
            if hist_band is not None and band.any():
                hist_band += np.bincount(arr[band], minlength=256)
        print(f"  배경 화소 {hist_bg.sum():,}")

    qs = (1, 5, 25, 50, 75, 95, 99)
    report: dict[str, object] = {
        "note": "학습 풀 전용. 마스킹 전 tiles_v1 화소.",
        "defect_polygon": pct_from_hist(hist_defect, qs),
        "background_tile": pct_from_hist(hist_bg_tile, qs),
        "background_crop": pct_from_hist(hist_bg_crop, qs),
        "band_interior_tile": pct_from_hist(hist_band_tile, qs),
        "defect_by_type": {k: pct_from_hist(v, qs) for k, v in sorted(hist_defect_by_type.items())},
        "pixels": {
            "defect": int(hist_defect.sum()),
            "background_tile": int(hist_bg_tile.sum()),
            "background_crop": int(hist_bg_crop.sum()),
        },
    }

    print("\n=== 분포 (화소 단위 전수) ===")
    for key in ("defect_polygon", "background_tile", "background_crop", "band_interior_tile"):
        d = report[key]
        if d:
            print(f"  {key:20s} " + " ".join(f"{k}={v:.0f}" for k, v in d.items()))

    print("\n=== 겹침: 후보 임계 미만 화소 비율 ===")
    print("   T   결함폴리곤 내부   정상타일 배경   정상크롭 배경")
    rows = []
    cum_d = np.cumsum(hist_defect) / max(hist_defect.sum(), 1)
    cum_bt = np.cumsum(hist_bg_tile) / max(hist_bg_tile.sum(), 1)
    cum_bc = np.cumsum(hist_bg_crop) / max(hist_bg_crop.sum(), 1)
    for t in CANDIDATES:
        row = {"t": t, "defect_below_pct": round(float(cum_d[t - 1]) * 100, 3),
               "bg_tile_below_pct": round(float(cum_bt[t - 1]) * 100, 3),
               "bg_crop_below_pct": round(float(cum_bc[t - 1]) * 100, 3)}
        rows.append(row)
        print(f"  {t:3d}   {row['defect_below_pct']:12.3f}%   "
              f"{row['bg_tile_below_pct']:11.3f}%   {row['bg_crop_below_pct']:11.3f}%")
    report["overlap_table"] = rows

    report["clamp_effect"] = clamp_effect(m, prog, take)

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\n기록: {args.out}")
    return 0


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """순위 기반 AUC. 두 모집단을 가르는 힘이고, 0.5 면 못 가른다."""
    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # 동점 보정. 클램프 후에는 같은 값이 무더기로 생기므로 반드시 필요하다.
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    return max(auc, 1 - auc)


def clamp_effect(m, prog, take) -> list[dict]:
    """후보 임계별로 두 모집단의 1백분위가 어떻게 움직이는지.

    **현재 상태(마스킹본) 위에서 잰다.** R3 는 R2-a 위에 쌓이므로 그 순서로 봐야 한다.
    """
    from data.manifest_io import read_manifest

    cur = read_manifest(V1 / "manifest.csv")
    cur = cur[cur["split"] != "eval"].copy()
    normal = cur[~cur["has_defect"].astype(bool)].copy()
    normal["reason"] = normal["image_id"].map(lambda i: prog[i]["reason"])
    pops = {
        "N-crop": take(normal[normal["reason"] == "ok"].sort_values("image_id")),
        "N-tile": take(normal[normal["reason"] == "tiled"].sort_values("image_id")),
    }
    hists: dict[str, np.ndarray] = {}
    for name, frame in pops.items():
        acc = np.zeros((len(frame), 256), dtype=np.int32)
        for k, r in enumerate(frame.itertuples()):
            arr = np.asarray(Image.open(REPO_ROOT / r.rel_path).convert("L"))
            acc[k] = np.bincount(arr.ravel(), minlength=256)
        hists[name] = acc
        print(f"\n{name} {len(frame):,}장 히스토그램 수집")

    def stat(acc: np.ndarray, t: int, q: int) -> np.ndarray:
        """임계 t 로 아래를 눌렀을 때의 q백분위. t=0 이면 원본."""
        h = acc.copy()
        if t > 0:
            h[:, t] += h[:, :t].sum(axis=1)
            h[:, :t] = 0
        cum = np.cumsum(h, axis=1)
        target = q / 100 * cum[:, -1]
        return np.argmax(cum >= target[:, None], axis=1).astype(np.float64)

    rows = []
    print("\n=== 클램프 후 1백분위 (학습 풀, 마스킹본 위) ===")
    print("   T   N-crop p1   N-tile p1     차이   p1 단독 AUC   p25 단독 AUC")
    for t in (0, *CANDIDATES):
        pc, pt = stat(hists["N-crop"], t, 1), stat(hists["N-tile"], t, 1)
        qc, qt = stat(hists["N-crop"], t, 25), stat(hists["N-tile"], t, 25)
        row = {
            "t": t, "n_crop_p1": round(float(pc.mean()), 1),
            "n_tile_p1": round(float(pt.mean()), 1),
            "gap": round(float(pc.mean() - pt.mean()), 1),
            "auc_p1": round(_auc(pc, pt), 4), "auc_p25": round(_auc(qc, qt), 4),
        }
        rows.append(row)
        tag = "원본" if t == 0 else f"{t:3d}"
        print(f"  {tag:>4}   {row['n_crop_p1']:9.1f}   {row['n_tile_p1']:9.1f}   "
              f"{row['gap']:6.1f}   {row['auc_p1']:11.4f}   {row['auc_p25']:12.4f}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
