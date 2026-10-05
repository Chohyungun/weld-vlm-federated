"""VT 라벨의 집계를 따로 다시 센다 — VT 스냅샷 미니스펙의 실측 표를 원저자 아닌 쪽이 검산한다.

라벨 JSON 만 읽고 원천 이미지는 열지 않는다. 장마다의 값은 내지 않고 집계만 낸다. 독립성의 범위는 검산 보고에 적는다.

    python -X utf8 -B -m scripts.crosscheck.vt_census_independent \
        --vt-labels <VT 라벨 zip 이 든 폴더들> --rt-labels <RT 라벨 zip 폴더> --out <_workspace 아래의 없는 새 .json>

**쓰기.** 출력은 `--out` 파일 하나이고, 저장소 `_workspace/` 아래의 **없는 새 경로**만 받는다(실제 경로로 본다 —
`scripts.crosscheck.xc_out_guard`). 입력 폴더와 겹치거나 동결 디렉터리 아래면 아무것도 읽기 전에 멈춘다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FOLDER_RE = re.compile(r"(결함_\d\. [^.]+|정상)\.zip$")
CODE_RE = re.compile(r"_(\d{2})_\d+$")


def folder_of(zip_name: str) -> str:
    m = FOLDER_RE.search(zip_name)
    if not m:
        raise SystemExit(f"폴더 이름을 읽지 못했다: {zip_name}")
    return m.group(1).split(". ")[-1] if "결함" in m.group(1) else "정상"


def read(zips: list[Path]):
    for zp in sorted(zips, key=lambda p: p.name):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.lower().endswith(".json"):
                    yield zp.name, json.loads(z.read(name).decode("utf-8-sig"))


def gaps(ids: list[int], cap: int = 20) -> dict:
    ids = sorted(ids)
    diffs = [b - a for a, b in zip(ids, ids[1:])]
    g = [d - 1 for d in diffs if d > 1]
    small = [x for x in g if x <= cap]
    runs, cur, lens = 0, 1, []
    for d in diffs:
        if d == 1:
            cur += 1
        else:
            lens.append(cur)
            cur = 1
    lens.append(cur)
    missing = sum(small)
    p = missing / (len(ids) + missing) if ids else 0.0
    c = Counter(small)
    n = len(small)
    return {
        "n": len(ids), "id_min": ids[0], "id_max": ids[-1], "dup_ids": len(ids) - len(set(ids)),
        "gaps": len(g), f"gaps_over_{cap}": len(g) - n, "p_from_small_gaps": round(p, 4),
        "mean_gap_small": round(missing / n, 4) if n else None,
        "geom_mean_expected": round(1 / (1 - p), 4) if p < 1 else None,
        "ratio_len1_2_3": [round(c[k] / n, 4) for k in (1, 2, 3)] if n else None,
        "geom_expected_1_2_3": [round((1 - p) * p ** (k - 1), 4) for k in (1, 2, 3)],
        "runs": len(lens), "runs_expected_n_times_p": round(len(ids) * p, 1),
        "run_mean": round(sum(lens) / len(lens), 2), "run_top5": sorted(lens, reverse=True)[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vt-labels", type=Path, nargs="+", required=True)
    ap.add_argument("--rt-labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="_workspace 아래의 없는 새 .json")
    a = ap.parse_args()

    from scripts.crosscheck.xc_out_guard import guard_new_output, recheck_inside

    guard_new_output(a.out, repo=Path(__file__).resolve().parents[2], inputs=(*a.vt_labels, a.rt_labels))
    vt_zips = [p for d in a.vt_labels for p in d.glob("*VTST*.zip")]
    out: dict = {"vt_zips": sorted(p.name for p in vt_zips)}
    count = Counter()
    split_count = Counter()
    sizes = defaultdict(Counter)
    vertical = Counter()
    mismatch = Counter()
    oos_imgs, oos_inst = 0, Counter()
    multi = Counter()
    oob_inst = oob_img = 0
    few_inst = few_img = 0
    nonint = 0
    inst_cls = Counter()
    vt_ids, id_folder, id_1280, id_vert = [], {}, {}, {}
    for zname, o in read(vt_zips):
        folder = folder_of(zname)
        info, img = o["info"], o["image_data"]
        iid = int(info["id"])
        w, h = int(img["width"]), int(img["height"])
        count[folder] += 1
        split_count[(folder, zname[:2])] += 1
        sizes[folder][f"{w}x{h}"] += 1
        vertical[folder] += int(h > w)
        code = CODE_RE.search(str(img["file_name"]))
        info_ko = str(img.get("information", ""))
        if info_ko != folder:
            mismatch["information_vs_folder"] += 1
        if str(info["type"]) != "VT" or str(info["material"]) != "ST":
            mismatch["type_or_material"] += 1
        vt_ids.append(iid)
        id_folder[iid] = folder
        id_1280[iid] = (w, h) == (1280, 720)
        anns = o.get("annotations") or []
        cases = {str(x.get("case", "")) for x in anns if x.get("class") == "defect" and x.get("case")}
        if "crack" in cases:
            oos_imgs += 1
            oos_inst["crack"] += sum(1 for x in anns if x.get("case") == "crack")
        if len(cases) > 1:
            multi[" + ".join(sorted(cases))] += 1
        any_oob = any_few = False
        for x in anns:
            inst_cls[str(x.get("class"))] += 1
            c = x.get("coordinate") or {}
            xs, ys = c.get("x") or [], c.get("y") or []
            if len(xs) < 3:
                few_inst += 1
                any_few = True
            if any(isinstance(v, float) and not float(v).is_integer() for v in (*xs, *ys)):
                nonint += 1
            if any(v < 0 or v > w for v in xs) or any(v < 0 or v > h for v in ys):
                oob_inst += 1
                any_oob = True
        oob_img += any_oob
        few_img += any_few
        if code is None:
            mismatch["filename_code_missing"] += 1
        else:
            out.setdefault("_codes", defaultdict(set))[folder].add(code.group(1))

    out["counts"] = dict(count)
    out["counts_by_split"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(split_count.items())}
    out["total"] = sum(count.values())
    out["codes_per_folder"] = {k: sorted(v) for k, v in out.pop("_codes").items()}
    out["sizes"] = {k: dict(v.most_common()) for k, v in sizes.items()}
    out["vertical"] = dict(vertical)
    n1280_def = sum(sizes[f]["1280x720"] for f in sizes if f != "정상")
    n_def = sum(count[f] for f in count if f != "정상")
    n1280_norm = sizes["정상"]["1280x720"]
    out["size_rule"] = {"defect_1280": n1280_def, "defect_total": n_def, "normal_1280": n1280_norm,
                        "normal_total": count["정상"],
                        "accuracy": round((n1280_def + count["정상"] - n1280_norm) / out["total"], 4)}
    out["mismatch"] = dict(mismatch)
    out["out_of_space"] = {"images": oos_imgs, "instances": dict(oos_inst)}
    out["multi_type"] = dict(multi)
    out["oob"] = {"instances": oob_inst, "images": oob_img}
    out["few_vertices"] = {"instances": few_inst, "images": few_img}
    out["noninteger_instances"] = nonint
    out["instance_class"] = dict(inst_cls)
    out["vt_gaps"] = gaps(vt_ids)

    # id 십분위 — 폴더 구성 · 1280×720 비율
    ids_sorted = sorted(vt_ids)
    dec = []
    for q in range(10):
        lo, hi = q * len(ids_sorted) // 10, (q + 1) * len(ids_sorted) // 10
        part = ids_sorted[lo:hi]
        fc = Counter(id_folder[i] for i in part)
        dec.append({"q": q, "n": len(part), "normal": round(fc["정상"] / len(part), 3),
                    "r1280": round(sum(id_1280[i] for i in part) / len(part), 3), "folders": dict(fc)})
    out["vt_id_deciles"] = dec

    # RT
    rt = defaultdict(list)
    for zname, o in read(list(a.rt_labels.glob("*.zip"))):
        info = o["info"]
        if str(info["type"]) == "RT":
            rt[str(info["material"])].append(int(info["id"]))
    rt_all = sorted(i for v in rt.values() for i in v)
    out["rt_gaps"] = {m: gaps(v) for m, v in rt.items()}
    vt_set = set(vt_ids)
    out["rt_vt"] = {
        "rt_n": len(rt_all), "rt_min": rt_all[0], "rt_max": rt_all[-1],
        "overlap": len(vt_set & set(rt_all)),
        "rt_below_vt_min_nearest_gap": min(ids_sorted[0] - i for i in rt_all if i < ids_sorted[0]) - 1,
        "rt_above_vt_max_count": sum(1 for i in rt_all if i > ids_sorted[-1]),
        "rt_inside_vt_span": sum(1 for i in rt_all if ids_sorted[0] < i < ids_sorted[-1]),
    }
    merged = sorted([(i, "VT") for i in vt_ids] + [(i, "RT") for i in rt_all])
    out["rt_vt"]["modality_switches"] = sum(1 for x, y in zip(merged, merged[1:]) if x[1] != y[1])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    recheck_inside(a.out.parent, repo=Path(__file__).resolve().parents[2])
    with a.out.open("x", encoding="utf-8", newline="\n") as fh:   # 배타 생성 — 그사이 생겼으면 덮지 않고 멈춘다
        fh.write(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("total", "counts", "size_rule", "out_of_space", "oob", "few_vertices",
                                          "vt_gaps", "rt_vt")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
