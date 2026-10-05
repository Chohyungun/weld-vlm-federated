"""1단계 빌드가 멈춘 자리를 센다 — v1 상자와 원좌표 상자의 차이가 `clip_kind` 규칙으로 설명되지 않는 주석.

    uv run python -m scripts.crosscheck.diag_rawmeta_unexplained OUT_DIR

v1 만 읽는다. `annotations.csv` 의 `polygon_json` 은 원본 좌표를 `int()` 로만 거친 것이라 원좌표 상자를 거기서 낸다.
**평가 행은 수만 센다** — 평가 행의 식별자 · 값 · 플래그 분포는 내지 않는다. 학습 · 검증 행만 자세히 적는다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from data import absorb, rawmeta
from data.convert.geometry import polygon_metrics
from data.manifest_io import load_snapshot

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "data/interim/manifest_v1"


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = load_snapshot(V1, verify=True)
    man, ann = snap.manifest, snap.annotations
    split = dict(zip(man["image_id"], man["split"].fillna("")))
    size = {i: (int(w), int(h)) for i, w, h in zip(man["image_id"], man["width_px"], man["height_px"])}
    n_eval = 0
    rows = []
    flags_tv = Counter()
    shape_tv = Counter()
    for r in ann.itertuples(index=False):
        iid = str(r.image_id)
        pts = json.loads(r.polygon_json)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        raw = rawmeta.raw_box(xs, ys)
        v1 = absorb._box(r)
        w, h = size[iid]
        try:
            rawmeta.classify_clip(v1, raw, str(r.geom_flags or ""), w, h)
            continue
        except rawmeta.RawMetaError:
            pass
        if split[iid] == "eval":
            n_eval += 1
            continue
        flags = str(r.geom_flags or "")
        flags_tv[flags] += 1
        inside = v1[0] >= raw[0] and v1[1] >= raw[1] and v1[2] <= raw[2] and v1[3] <= raw[3]
        g = polygon_metrics(list(zip(xs, ys)), w, h)
        same_again = (tuple(g.bbox_px) if g.bbox_px else None) == v1
        shape_tv[("v1_상자가_원좌표_안" if inside else "v1_상자가_원좌표_밖으로"),
                 ("다시_내면_같다" if same_again else "다시_내면_다르다")] += 1
        rows.append({"ann_id": str(r.ann_id), "split": split[iid], "geom_flags": flags,
                     "v1_box": list(v1), "raw_box": list(raw), "n_points": len(pts)})
    summary = {
        "unexplained_rows_total": len(rows) + n_eval,
        "unexplained_rows_eval": n_eval,
        "unexplained_rows_trainval": len(rows),
        "trainval_geom_flags": dict(flags_tv.most_common()),
        "trainval_shape": {" · ".join(k): v for k, v in shape_tv.most_common()},
        "평가_행": "수만 센다. 식별자 · 값 · 플래그 분포를 내지 않는다",
    }
    (out_dir / "unexplained_trainval.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8", newline="\n")
    (out_dir / "unexplained_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
