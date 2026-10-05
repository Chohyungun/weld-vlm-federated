"""원본 라벨 판(1단계)과 정리된 파일을 겹치는 장에서 맞댄다 — 미니스펙 2판 5-2 · 5-3.

    ABSORB_JSONL=<정리된 파일> uv run python -m scripts.crosscheck.xc_rawmeta_team <스냅샷> <출력 폴더>

**평가 행은 식별자 넷 밖의 통로를 열지 않는다.** 평가 장의 맞대기는 프로세스 안에서만 돌리고
"전부 같다 / 같지 않다" 한 비트와 같지 않은 장 수만 낸다. 차이표 · 사유별 수는 학습 · 검증 장만 담는다.

여기서 낸 수는 **단일 구현의 값**이다. 원저자 아닌 독립 구현이 같은 값을 낼 때까지 인용하지 않는다.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data import absorb, rawmeta

#: 판정 03 추기 · 08 §5-3 의 101 건. 겹치는 결함 장, 행 그레인과 장 그레인.
EXPECTED_101 = {"rows": {"empty_bbox": 54, "boundary": 43, "polygon_repair": 9, "both": 0},
                "images": {"empty_bbox": 53, "boundary": 39, "polygon_repair": 9}}


def _ints(v) -> list[int]:
    if isinstance(v, str):
        v = json.loads(v)
    return [int(x) for x in (v or [])]


def compare(root: Path, jsonl: Path) -> tuple[dict, list[dict]]:
    snap = absorb.load_absorbed(root)
    man = snap.manifest
    split = dict(zip(man["image_id"], man["split"].fillna("")))
    has_defect = dict(zip(man["image_id"], man["has_defect"].astype(bool)))
    by_num = {i.split(":")[1]: i for i in man["image_id"]}
    img = snap.absorb_image.set_index("image_id")
    iso = dict(zip(snap.annotations["ann_id"], snap.annotations["iso_code"].astype(str)))
    ours_def: dict[str, list] = defaultdict(list)
    for r in snap.absorb_defect.itertuples(index=False):
        ours_def[str(r.image_id)].append(r)
    for v in ours_def.values():
        v.sort(key=lambda r: int(str(r.ann_id).rsplit("#", 1)[-1]))
    ours_reg: dict[str, list] = defaultdict(list)
    for r in snap.absorb_region.itertuples(index=False):
        ours_reg[str(r.image_id)].append(tuple(map(tuple, json.loads(r.polygon_json))))

    tv = Counter()
    ev_images_differ = 0
    ev_images = 0
    diffs: list[dict] = []
    overlap_defect_images: set[str] = set()
    for d in absorb.stream_jsonl(jsonl):
        iid = by_num.get(str(d.get("orig_info_id", "")).strip())
        if iid is None:
            continue
        is_eval = split[iid] == "eval"
        anns = d.get("annotations") or []
        team_def = sorted((a for a in anns if a.get("annotation_role") == "defect"), key=lambda a: a.get("ann_idx", 0))
        team_reg = [a for a in anns if a.get("annotation_role") == "normal_region"]
        reasons = []
        # 원본 크기 — 둘 다 라벨이 선언한 크기다
        if (int(d.get("width_px")), int(d.get("height_px"))) != (int(img.loc[iid, "orig_width_px"]),
                                                                  int(img.loc[iid, "orig_height_px"])):
            reasons.append("orig_size")
        # 결함 — 순서 키와 내용 키로 따로 짝짓는다
        mine = ours_def.get(iid, [])
        if has_defect[iid]:
            overlap_defect_images.add(iid)
        if len(mine) != len(team_def):
            reasons.append("defect_count")
        else:
            ob = [((int(r.raw_bbox_x1_px), int(r.raw_bbox_y1_px), int(r.raw_bbox_x2_px), int(r.raw_bbox_y2_px)),
                   iso[str(r.ann_id)]) for r in mine]
            tb = [((int(a["bbox_xmin"]), int(a["bbox_ymin"]), int(a["bbox_xmax"]), int(a["bbox_ymax"])),
                   str(a.get("class_std_iso6520", ""))) for a in team_def]
            order_same = ob == tb
            content_same = Counter(ob) == Counter(tb)
            if not content_same:
                if any(o[0] != t[0] for o, t in zip(ob, tb)):
                    reasons.append("defect_raw_box")
                if any(o[1] != t[1] for o, t in zip(ob, tb)):
                    reasons.append("defect_iso")
            elif not order_same:
                reasons.append("defect_order_only")
        # 정상영역 — 좌표 키로 짝짓는다. region_id 로 잇지 않는다
        tp = Counter(tuple(zip(_ints(a.get("polygon_x")), _ints(a.get("polygon_y")))) for a in team_reg)
        op = Counter(ours_reg.get(iid, []))
        if tp != op:
            reasons.append("region_polygons")
        if is_eval:
            ev_images += 1
            ev_images_differ += bool(reasons)
            continue
        tv["images"] += 1
        tv["defect_rows"] += len(mine)
        tv["region_rows_ours"] += sum(op.values())
        tv["region_rows_team"] += sum(tp.values())
        for r in reasons:
            tv[f"differ_{r}"] += 1
        if reasons:
            tv["images_differ"] += 1
            diffs.append({"image_id": iid, "split": split[iid], "reasons": reasons})

    # 101 건 — 겹치는 결함 장 전부(평가 포함)를 프로세스 안에서 맞대고, 학습 · 검증 분해만 낸다
    rows_all, rows_tv = Counter(), Counter()
    kinds_all: dict[str, list] = {}
    kinds_tv: dict[str, list] = {}
    for iid in overlap_defect_images:
        ks = [str(r.clip_kind) for r in ours_def.get(iid, [])]
        for k in ks:
            if k != "none":
                rows_all[k] += 1
                if split[iid] != "eval":
                    rows_tv[k] += 1
        kinds_all[iid] = ks
        if split[iid] != "eval":
            kinds_tv[iid] = ks
    img_all = rawmeta.fold_images(kinds_all)
    rows_all_full = {k: int(rows_all.get(k, 0)) for k in EXPECTED_101["rows"]}
    img_all_full = {k: int(img_all.get(k, 0)) for k in EXPECTED_101["images"]}
    summary = {
        "단일_구현": "독립 구현이 같은 값을 낼 때까지 인용하지 않는다",
        "trainval": dict(sorted(tv.items())),
        "eval": {"all_equal": ev_images_differ == 0, "images_differ": ev_images_differ,
                 "비고": "평가 장은 한 비트와 같지 않은 장 수만 낸다"},
        "expected_101": {"rows_equal": rows_all_full == EXPECTED_101["rows"],
                         "images_equal": img_all_full == EXPECTED_101["images"],
                         "trainval_rows": {k: int(rows_tv.get(k, 0)) for k in EXPECTED_101["rows"]},
                         "trainval_images": {k: int(v) for k, v in sorted(rawmeta.fold_images(kinds_tv).items())},
                         "비고": "겹치는 결함 장 전부(평가 포함)로 프로세스 안에서 맞대고, 같은지 한 비트와 학습 · 검증 분해만 낸다"},
        "overlap_images_total": ev_images + tv["images"],
    }
    return summary, diffs


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    jsonl = Path(os.environ.get("ABSORB_JSONL", "").strip() or "-")
    if not jsonl.exists():
        raise SystemExit("환경 변수 ABSORB_JSONL 이 정리된 파일을 가리키지 않는다")
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    summary, diffs = compare(root, jsonl)
    summary["team_file_sha256"] = absorb.file_digest(jsonl)
    out.mkdir(parents=True, exist_ok=True)
    (out / "xc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
    (out / "xc_diffs_trainval.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in diffs),
                                                encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
