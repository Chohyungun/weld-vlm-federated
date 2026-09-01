"""content-free 기준선 재산출 — 동결본 정본값. 68번 §5-(가).

**모든 값을 `evaluation/score.py` 의 단일 채점기로 낸다.** 해석식은 검산용으로만 쓰고,
표에 박히는 값은 채점기 출력이다. 두 갈래가 어긋나면 그 자체를 보고한다.

content-free 예측기란 **이미지 화소를 보지 않고 메타데이터(출처·재질)만으로** 예측하는
규칙이다. 이 계열의 최댓값이 게이트 선이 된다 — 자명하한 하나로는 지름길 구간이 걸러지지
않는다는 것이 66번의 소득이었다.

    uv run python scripts/recompute_baselines.py

**평가셋만 채점 모집단으로 쓴다.** 다만 상수 박스는 train+val 에서만 유도한다 —
평가셋 통계에서 예측 상수를 뽑으면 불변조건 1-4 가 실질에서 깨진다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.manifest_io import load_snapshot
from evaluation.schema import Defect, PredictionRecord
from evaluation.score import score_records

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "data/interim/manifest_v1"


def build_gold(snap, image_ids: Sequence[str]):
    """평가 모집단의 정답. 정상 이미지는 빈 집합으로 들어간다(오탐 원천)."""
    keep = set(image_ids)
    ann = snap.annotations[snap.annotations["image_id"].isin(keep)]
    codes: dict[str, list[str]] = {i: [] for i in image_ids}
    boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {
        i: [] for i in image_ids
    }
    n_nobox = 0
    for r in ann.itertuples():
        codes[r.image_id].append(str(r.iso_code))
        vals = (r.bbox_x1_px, r.bbox_y1_px, r.bbox_x2_px, r.bbox_y2_px)
        if any(pd.isna(v) for v in vals):
            n_nobox += 1        # N1(위치 없음). 코드는 살리고 박스 축에서만 빠진다
            continue
        boxes[r.image_id].append((str(r.iso_code), tuple(float(v) for v in vals)))
    if n_nobox:
        print(f"  bbox 결측 어노테이션 {n_nobox:,}건 — 코드 축에는 남기고 위치 축에서만 제외")
    return {k: sorted(set(v)) for k, v in codes.items()}, boxes


def records_from_rule(
    image_ids: Sequence[str], cell_of: Mapping[str, str], subsets: Mapping[str, tuple[str, ...]],
    box: tuple[float, float, float, float] | None = None,
) -> list[PredictionRecord]:
    """셀 → 코드 집합 규칙을 계약 #4 레코드로 만든다. 화소를 보지 않는다."""
    out = []
    for img in image_ids:
        defects = [
            Defect(iso_code=c, bbox_px=list(box) if box else None,
                   score=1.0 if box else None)
            for c in subsets.get(cell_of[img], ())
        ]
        out.append(PredictionRecord(
            schema_version="1.3", image_id=img, cell="sep_central", seed=20260825,
            defects=defects, verdict="판정불가", cited_clauses=[], parse_ok=True,
        ))
    return out


def analytic_macro_f1(
    cells: Sequence[str], classes: Sequence[str],
    n_by_cell: Mapping[str, int], pos_by_cell: Mapping[tuple[str, str], int],
    chosen: Mapping[str, tuple[str, ...]],
) -> float:
    """셀별 코드 집합이 주어졌을 때의 Macro-F1 해석값 (채점기 검산용).

    클래스 c: TP = Σ_{셀에 c 포함} n_{셀,c}, 예측양성 = Σ_{셀에 c 포함} N_셀,
    실양성 = Σ_셀 n_{셀,c}. F1 = 2TP / (예측양성 + 실양성).
    """
    per = []
    for c in classes:
        actual = sum(pos_by_cell.get((g, c), 0) for g in cells)
        if actual == 0:
            continue                       # GT 0 인 클래스는 macro 에서 빠진다(채점기 계약)
        tp = sum(pos_by_cell.get((g, c), 0) for g in cells if c in chosen.get(g, ()))
        pp = sum(n_by_cell[g] for g in cells if c in chosen.get(g, ()))
        per.append(0.0 if tp == 0 else 2 * tp / (pp + actual))
    return float(np.mean(per)) if per else 0.0


def best_content_free(
    cells: Sequence[str], classes: Sequence[str],
    n_by_cell: Mapping[str, int], pos_by_cell: Mapping[tuple[str, str], int],
) -> tuple[dict[str, tuple[str, ...]], float]:
    """셀 축만 쓰는 예측기의 **정확한** 최적해.

    Macro-F1 은 클래스별 F1 의 평균이고 클래스 c 의 F1 은 "어느 셀에서 c 를 주장하는가"
    에만 의존한다. 따라서 **클래스끼리 독립**이고, 클래스마다 셀 부분집합 2^|cells| 를
    전수 탐색하면 전역 최적이다. 셀 조합 16^|cells| 를 뒤질 필요가 없다.
    """
    include: dict[str, set[str]] = {g: set() for g in cells}
    for c in classes:
        actual = sum(pos_by_cell.get((g, c), 0) for g in cells)
        if actual == 0:
            continue
        best_f1, best_sel = 0.0, ()
        for k in range(len(cells) + 1):
            for sel in itertools.combinations(cells, k):
                tp = sum(pos_by_cell.get((g, c), 0) for g in sel)
                pp = sum(n_by_cell[g] for g in sel)
                f1 = 0.0 if tp == 0 else 2 * tp / (pp + actual)
                if f1 > best_f1:
                    best_f1, best_sel = f1, sel
        for g in best_sel:
            include[g].add(c)
    chosen = {g: tuple(sorted(include[g])) for g in cells}
    return chosen, analytic_macro_f1(cells, classes, n_by_cell, pos_by_cell, chosen)


def cell_stats(rows: pd.DataFrame, gold: Mapping[str, Sequence[str]], cell_col: str,
               classes: Sequence[str]):
    n_by_cell: dict[str, int] = {}
    pos_by_cell: dict[tuple[str, str], int] = {}
    for cell, part in rows.groupby(cell_col, observed=True):
        n_by_cell[str(cell)] = len(part)
        for c in classes:
            pos_by_cell[(str(cell), c)] = sum(
                1 for i in part["image_id"] if c in gold[i])
    return n_by_cell, pos_by_cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("-o", "--out", type=Path,
                    default=REPO_ROOT / "data/interim/manifest_v1/baselines_recomputed.json")
    args = ap.parse_args()

    snap = load_snapshot(args.snapshot)
    m, t = snap.manifest, snap.tiles
    prov = dict(zip(t["image_id"], t["provenance"], strict=True))
    m = m.assign(prov=m["image_id"].map(prov))
    ev = m[m["split"] == "eval"].reset_index(drop=True)
    print(f"동결 스냅샷 {snap.snapshot_id} · 평가셋 {len(ev):,}장")

    gold_codes, gold_boxes = build_gold(snap, list(ev["image_id"]))
    classes = sorted({c for v in gold_codes.values() for c in v})
    print(f"평가셋 GT 코드 {classes}")

    report: dict[str, object] = {"snapshot_id": snap.snapshot_id, "classes": classes,
                                 "populations": {}, "constants": {}}

    populations = {
        "eval12461": ev,
        "eval_ncrop7740": ev[ev["prov"] == "N-crop"].reset_index(drop=True),
    }

    for pop_name, rows in populations.items():
        ids = list(rows["image_id"])
        g_codes = {i: gold_codes[i] for i in ids}
        g_boxes = {i: gold_boxes[i] for i in ids}
        pop_classes = sorted({c for v in g_codes.values() for c in v})
        print(f"\n=== {pop_name} · {len(ids):,}장 · 클래스 {pop_classes} ===")
        report["populations"][pop_name] = {"n": len(ids), "classes": pop_classes}

        def run(name: str, cell_col: str, subsets: Mapping[str, tuple[str, ...]],
                rows=rows, ids=ids, g_codes=g_codes, g_boxes=g_boxes,
                pop_classes=pop_classes, pop_name=pop_name) -> float:
            cell_of = dict(zip(rows["image_id"], rows[cell_col].astype(str), strict=True))
            recs = records_from_rule(ids, cell_of, subsets)
            res = score_records(recs, g_codes, g_boxes, pop_classes)
            n_by, pos_by = cell_stats(rows, g_codes, cell_col, pop_classes)
            ana = analytic_macro_f1(sorted(n_by), pop_classes, n_by, pos_by, subsets)
            mf1 = float(res["macro_f1"])
            flag = "" if abs(mf1 - ana) < 1e-9 else f"  !! 해석식 {ana:.6f} 과 불일치"
            print(f"  {name:34s} macro_f1 {mf1:.4f}  miss {res['miss_rate']:.4f}{flag}")
            report["constants"][f"{name}__{pop_name}"] = {
                "macro_f1": round(mf1, 6), "analytic": round(ana, 6),
                "miss_rate": round(float(res["miss_rate"]), 6),
                "defect_recall": round(float(res["defect_recall"]), 6),
                "class_jaccard": round(float(res["class_jaccard"]), 6),
                "rule": {k: list(v) for k, v in subsets.items()},
            }
            return mf1

        allc = tuple(pop_classes)
        m_all = rows.assign(_all="all")
        run("trivial_all_positive", "_all", {"all": allc}, rows=m_all)
        run("constant_porosity", "_all", {"all": ("2011",)}, rows=m_all)
        run("source_conditional_porosity", "prov",
            {"N-crop": ("2011",), "N-tile": (), "N-band": ()})

        for axis, col in (("material_only", "material"), ("source_only", "prov"),
                          ("source_x_material", "cell_sm")):
            work = rows.assign(cell_sm=rows["prov"].astype(str) + "|" + rows["material"].astype(str))
            n_by, pos_by = cell_stats(work, g_codes, col, pop_classes)
            chosen, _ = best_content_free(sorted(n_by), pop_classes, n_by, pos_by)
            run(f"{axis}_best", col, chosen, rows=work)

    # ---- 위치 축: 상수 박스. train+val 에서만 유도한다 ----
    print("\n=== 위치 축 기준선 (상수 박스) ===")
    train_ids = set(m.loc[m["split"] != "eval", "image_id"])
    tr_ann = snap.annotations[snap.annotations["image_id"].isin(train_ids)]
    const_box = (
        float(tr_ann["bbox_x1_px"].median()), float(tr_ann["bbox_y1_px"].median()),
        float(tr_ann["bbox_x2_px"].median()), float(tr_ann["bbox_y2_px"].median()),
    )
    print(f"  상수 박스 = train+val GT 중앙값 {const_box} (평가셋 미사용, n={len(tr_ann):,})")
    ids = list(ev["image_id"])
    cell_of = dict.fromkeys(ids, "all")
    recs = records_from_rule(ids, cell_of, {"all": tuple(classes)}, box=const_box)
    res = score_records(recs, gold_codes, gold_boxes, classes)
    per_image_max = []
    for img in ids:
        g = gold_boxes[img]
        if not g:
            continue
        per_image_max.append(max(_iou(const_box, b) for _, b in g))
    ge50 = int(res["n_matched_ge_50"]) / max(int(res["n_gold"]), 1)
    print(f"  GT 박스 {int(res['n_gold']):,}개 중 IoU>=0.5 {int(res['n_matched_ge_50']):,} "
          f"({ge50*100:.2f}%)")
    print(f"  bbox_iou(미매칭 0 포함) {res['bbox_iou']:.4f} · "
          f"이미지당 최대 IoU 평균 {np.mean(per_image_max):.4f}")
    report["constants"]["constant_box__eval12461"] = {
        "box_xyxy": list(const_box), "derived_from": "train+val GT 중앙값 (eval 미사용)",
        "n_gold": int(res["n_gold"]),
        "n_ge_50": int(res["n_matched_ge_50"]),
        "frac_ge_50": round(ge50, 6),
        "bbox_iou_mean_all": round(float(res["bbox_iou"]), 6),
        "per_image_max_iou_mean": round(float(np.mean(per_image_max)), 6),
        "map_50": round(float(res.get("map_50", 0.0)), 6),
    }

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\n기록: {args.out}")
    return 0


def _iou(a, b) -> float:
    from evaluation.metrics.localization import iou
    return iou(tuple(a), tuple(b))


if __name__ == "__main__":
    raise SystemExit(main())
