"""전수 메타데이터 1단계의 독립 구현 — 원본 라벨에서 곁파일 셋을 다시 내고, 후보 판 · 팀원 파일과 맞댄다.

근거는 전수 메타데이터화 미니스펙(`docs/dev_log/2026-10-01-전수메타데이터/01_미니스펙_전수메타데이터화_A.md`)의
2-2 · 2-3 · 4 · 5-2 · 5-3 · 6절과 총괄 결정 04 의 1(원좌표 상자는 최솟값 내림 · 최댓값 올림)이다. 독립성의
범위(무엇을 읽지 않고 짰는지)는 이 스크립트의 보고에 적는다.

    python -X utf8 -B -m scripts.crosscheck.rawmeta_independent \
        --labels <aihub_labels> --v1 <manifest_v1> --label-map configs/label_map.yaml \
        --jsonl <images.jsonl> --candidate <후보 판 폴더> --out <_workspace 아래의 없는 새 폴더>

**쓰기.** 출력은 `--out` 하나이고, 저장소 `_workspace/` 아래의 **없는 새 폴더**만 받는다(실제 경로로 본다 —
`scripts.crosscheck.xc_out_guard`). 입력 · 후보 판 · 동결 디렉터리와 겹치면 아무것도 읽거나 만들기 전에 멈춘다.
원본 라벨 zip · 동결 스냅샷 · 팀원 파일 · 후보 판은 읽기만 한다.

**평가 행 — 전수 메타데이터 검산의 예외.** 이 스크립트는 일반 판독기(`data.manifest_view` — 평가 행은 식별자
넷만)를 쓰지 않고 매니페스트 · 주석 · 타일 CSV 전체와 원본 라벨을 직접 읽는다. 곁파일 셋이 평가 행까지 전 장을
담는 정본이라, 그 바이트를 다시 내려면 평가 행의 재질 · 좌표 · 결함 종류가 필요하다. 미니스펙 5-3 이 정한 범위
안에서만 쓴다.
- 평가 행의 값은 프로세스 안에서만 대조한다. 요약(`summary_G.json`)에는 평가 행의 식별자 · 값 · 열별 불일치를
  싣지 않고, "전부 같다" 한 비트와 같지 않은 장(행) 수만 싣는다. 차이표(`*_trainval*.jsonl`)는 학습 · 검증 행만 담는다.
- 전 행 곁파일(평가 행 포함)은 `_workspace/` 의 새 폴더에만 쓴다 — 추적되지 않는다.
- 학습 경로로 새지 않는다: 이 모듈은 독립 검산 진입점이고 학습 · 목록 · 등록 코드가 import 하지 않으며, 그 쪽은
  일반 판독기의 학습 · 검증 뷰와 페어 파일만 읽는다. 이 모듈의 산출 경로(`_workspace/`)는 학습 설정 어디에도 없다.

가져다 쓰는 것은 명세가 "가져다 쓰고 사본을 만들지 않는다" 고 정한 사상 딕셔너리 둘뿐이다
(`scripts.build_manifest_v0` 의 `CASE_TO_RAW` · `OUT_OF_LABEL_SPACE`). 정규화 · 거르기 ·
사상표 해석 · 상자 · 직렬화 · 분류는 여기서 따로 짠다. 사상표는 YAML 을 직접 읽는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[2]
SOURCE = "aihub71761"
TILE = (1280, 720)

IMAGE_COLS = ("image_id", "covered", "source_zip", "material_source",
              "orig_width_px", "orig_height_px", "frame_match", "absorb_note")
DEFECT_COLS = ("ann_id", "image_id", "raw_bbox_x1_px", "raw_bbox_y1_px", "raw_bbox_x2_px",
               "raw_bbox_y2_px", "clip_applied", "clip_kind", "raw_source")
REGION_COLS = ("region_id", "image_id", "polygon_json", "n_vertices", "frame", "usable_in_tile_frame")


class Stop(RuntimeError):
    """명세가 멈추라고 한 자리."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_case(case: str) -> str:
    """규칙 4 — 밑줄 · 공백을 지우고 소문자."""
    return case.replace("_", "").replace(" ", "").lower()


def poly_json(xs, ys) -> str:
    """2-2 — 원본 좌표를 `int()` 로 거쳐 `separators=(",",":")`."""
    return json.dumps([[int(x), int(y)] for x, y in zip(xs, ys)], separators=(",", ":"))


def _has_float(xs, ys) -> bool:
    return any(isinstance(v, float) for v in (*xs, *ys))


def _has_nonint(xs, ys) -> bool:
    return any(isinstance(v, float) and not v.is_integer() for v in (*xs, *ys))


def raw_box(xs, ys) -> tuple[int, int, int, int]:
    """원좌표 상자 — 최솟값 내림 · 최댓값 올림(총괄 결정 04 의 1). 자르지 않는다."""
    return (math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys)))


# ------------------------------------------------------------------------------------------
# 원본 라벨 읽기 — 규칙 0 · 1 · 3 · 4
# ------------------------------------------------------------------------------------------

def read_raw(labels: Path) -> tuple[dict[int, dict], dict]:
    recs: dict[int, dict] = {}
    acct = Counter()
    zips = []
    for zp in sorted(labels.glob("*.zip"), key=lambda p: p.name):
        n_json = 0
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if not name.lower().endswith(".json"):
                    continue
                n_json += 1
                try:
                    o = json.loads(z.read(name).decode("utf-8-sig"))
                    info, img = o["info"], o["image_data"]
                    anns = o.get("annotations") or []
                    iid = int(info["id"])
                    typ = str(info["type"])
                    mat = str(info["material"])
                    w, h = int(img["width"]), int(img["height"])
                    is_normal = (not anns) or all(a.get("class") != "defect" for a in anns)
                    polys = []
                    for a in anns:
                        c = a.get("coordinate") or {}
                        xs, ys = c.get("x") or [], c.get("y") or []
                        if len(xs) >= 3 and len(xs) == len(ys):
                            polys.append((xs, ys, str(a.get("case", ""))))
                except (KeyError, ValueError, TypeError, UnicodeDecodeError):
                    acct["parse_failed_images"] += 1
                    continue
                if typ != "RT":
                    acct["non_rt_images"] += 1
                    continue
                if iid in recs:
                    raise Stop(f"규칙 8 — info.id 가 겹친다 (zip {zp.name})")
                recs[iid] = {"zip": zp.name, "material": mat, "w": w, "h": h,
                             "is_normal": is_normal, "polys": polys}
        zips.append({"name": zp.name, "sha256": sha256_file(zp), "json_entries": n_json})
    acct["rt_images"] = len(recs)
    return recs, {"zips": zips, **acct}


# ------------------------------------------------------------------------------------------
# 결함 다각형의 사상 — 규칙 2 · 5 · 6 · 7
# ------------------------------------------------------------------------------------------

def load_iso(label_map: Path) -> tuple[dict[str, str], set[str], dict[str, str]]:
    lm = yaml.safe_load(label_map.read_text(encoding="utf-8"))
    src = lm["sources"][SOURCE]
    iso = {k: str(v["iso_code"]) for k, v in lm["defect_types"].items()}
    return dict(src["mapping"]), set(src.get("normal_labels") or []), iso


def defects_of(rec: dict, case_to_raw: dict, oos: set, mapping: dict, normals: set, iso: dict):
    """결함 장의 주석 목록. 장째 빠지면 None."""
    if rec["is_normal"]:
        return []
    if any(norm_case(c) in oos for _, _, c in rec["polys"]):
        return None
    out = []
    for pidx, (xs, ys, case) in enumerate(rec["polys"]):
        key = norm_case(case)
        if not key:
            continue                                   # 규칙 5 — 순번은 늘지 않는다
        if key not in case_to_raw:
            raise Stop(f"규칙 5 — 사상할 수 없는 case {case!r}")
        raw = case_to_raw[key]
        if raw in normals:
            continue                                   # 규칙 7 — to_defect_type 이 None
        if raw not in mapping:
            raise Stop(f"계약 #1 — 사상표에 없는 라벨 {raw!r}")
        l2 = mapping[raw]
        out.append({"_pidx": pidx, "src_label_raw": raw, "defect_type": l2, "iso_code": iso[l2],
                    "polygon_json": poly_json(xs, ys), "raw": raw_box(xs, ys)})
    return out


# ------------------------------------------------------------------------------------------
# v1 읽기
# ------------------------------------------------------------------------------------------

def read_csv_rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def int_or_none(s: str):
    return None if s in ("", None) else int(float(s))


def clip_kind(v1box, raw, w: int, h: int, flags: str) -> str:
    """2-3 의 분류. 멈춤은 Stop."""
    if any(v is None for v in v1box):
        return "empty_bbox"
    if tuple(v1box) == tuple(raw):
        return "none"
    clamped = (min(max(raw[0], 0), w), min(max(raw[1], 0), h),
               min(max(raw[2], 0), w), min(max(raw[3], 0), h))
    boundary = tuple(v1box) == clamped
    poly = "multipart_largest_kept" in (flags or "")
    if boundary and poly:
        return "both"
    if boundary:
        return "boundary"
    if poly:
        return "polygon_repair"
    raise Stop("2-3 — 상자가 다른데 어느 갈래도 아니다")


# ------------------------------------------------------------------------------------------
# 정본 CSV — `_canonical_csv` 와 같은 꼴(LF · QUOTE_MINIMAL · 정렬 키 · 불리언 True/False)
# ------------------------------------------------------------------------------------------

def cell(v) -> str:
    if v is True:
        return "True"
    if v is False:
        return "False"
    if v is None:
        return ""
    return str(v)


def write_canonical(rows: list[dict], cols: tuple[str, ...], key: str, path: Path) -> str:
    rows = sorted(rows, key=lambda r: r[key])
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    wr.writerow(cols)
    for r in rows:
        wr.writerow([cell(r[c]) for c in cols])
    data = buf.getvalue().encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------------
# 본체
# ------------------------------------------------------------------------------------------

def build(args) -> dict:
    from scripts.build_manifest_v0 import CASE_TO_RAW, OUT_OF_LABEL_SPACE  # 명세가 정한 공유

    zips_before = {p.name: sha256_file(p) for p in sorted(args.labels.glob("*.zip"))}
    recs, raw_acct = read_raw(args.labels)
    mapping, normals, iso = load_iso(args.label_map)

    man = read_csv_rows(args.v1 / "manifest.csv")
    ann = read_csv_rows(args.v1 / "annotations.csv")
    tiles = {r["image_id"]: r["reason"] for r in read_csv_rows(args.v1 / "tiles.csv")}
    man_by = {r["image_id"]: r for r in man}
    ann_by_img: dict[str, list[dict]] = defaultdict(list)
    for r in ann:
        ann_by_img[r["image_id"]].append(r)
    for v in ann_by_img.values():
        v.sort(key=lambda r: int(r["ann_id"].rsplit("#", 1)[1]))

    img_rows, def_rows, reg_rows = [], [], []
    acct = Counter()
    v1_mismatch_tv = Counter()                       # 열별 집계는 학습 · 검증 행만(명세 5-3)
    label_defect_v1_normal_tv = Counter()
    ev_v1_bad: set[str] = set()                      # 평가 행은 어긋난 장의 집합만 — 내보내는 것은 한 비트와 장 수
    for iid_s, m in man_by.items():
        iid = int(iid_s.rsplit(":", 1)[1])
        is_ev = m["split"] == "eval"

        def miss(col: str, _ev: bool = is_ev, _iid: str = iid_s) -> None:
            if _ev:
                ev_v1_bad.add(_iid)
            else:
                v1_mismatch_tv[col] += 1
        rec = recs.get(iid)
        if rec is None:
            raise Stop("v1 의 장이 원본에 없다")
        if rec["material"] != m["material"]:
            miss("material")
        w, h = rec["w"], rec["h"]
        fm = "same" if (w, h) == TILE else "tile_of_orig"
        if (fm == "same") != (tiles[iid_s] == "ok"):
            miss("frame_match_vs_reason")
        img_rows.append({"image_id": iid_s, "covered": True, "source_zip": rec["zip"],
                         "material_source": "label_json", "orig_width_px": w, "orig_height_px": h,
                         "frame_match": fm, "absorb_note": ""})

        ds = defects_of(rec, CASE_TO_RAW, OUT_OF_LABEL_SPACE, mapping, normals, iso)
        if ds is None:
            raise Stop("v1 의 장이 규칙 6 으로 빠진다")
        v1a = ann_by_img.get(iid_s, [])
        if len(ds) != len(v1a):
            miss("defect_count")
            continue
        if not rec["is_normal"] and not ds and not is_ev:
            label_defect_v1_normal_tv[m["split"]] += 1
        W, H = int(m["width_px"]), int(m["height_px"])
        for seq, (d, a) in enumerate(zip(ds, v1a)):
            ann_id = f"{iid_s}#{seq}"
            if a["ann_id"] != ann_id:
                miss("ann_id")
            for k in ("polygon_json", "src_label_raw", "defect_type", "iso_code"):
                if d[k] != a[k]:
                    miss(k)
            v1box = tuple(int_or_none(a[c]) for c in ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"))
            kind = clip_kind(v1box, d["raw"], W, H, a["geom_flags"])
            xs_, ys_ = rec["polys"][d["_pidx"]][:2]
            def_rows.append({"_float": _has_float(xs_, ys_), "_nonint": _has_nonint(xs_, ys_),
                             "ann_id": ann_id, "image_id": iid_s,
                             "raw_bbox_x1_px": d["raw"][0], "raw_bbox_y1_px": d["raw"][1],
                             "raw_bbox_x2_px": d["raw"][2], "raw_bbox_y2_px": d["raw"][3],
                             "clip_applied": kind != "none", "clip_kind": kind, "raw_source": "raw_label",
                             "_split": m["split"], "_client": m["client"], "_material": m["material"],
                             "_defect_type": d["defect_type"], "_iso": d["iso_code"]})
        if rec["is_normal"]:
            for k, (xs, ys, _c) in enumerate(rec["polys"]):
                pj = poly_json(xs, ys)
                reg_rows.append({"region_id": f"{iid_s}#r{k}", "image_id": iid_s, "polygon_json": pj,
                                 "n_vertices": len(xs), "frame": "orig", "usable_in_tile_frame": False,
                                 "_split": m["split"], "_float": _has_float(xs, ys), "_nonint": _has_nonint(xs, ys)})
            if m["split"] != "eval":
                acct["trainval_normal_images_with_zero_regions"] += int(not rec["polys"])
    if v1_mismatch_tv or ev_v1_bad:
        print("!! v1 일치 검사 어긋남 — 학습 · 검증:", dict(v1_mismatch_tv), "· 평가 장 수:", len(ev_v1_bad))

    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {
        "absorb_image.csv": write_canonical(img_rows, IMAGE_COLS, "image_id", args.out / "absorb_image.csv"),
        "absorb_defect.csv": write_canonical(def_rows, DEFECT_COLS, "ann_id", args.out / "absorb_defect.csv"),
        "absorb_region.csv": write_canonical(reg_rows, REGION_COLS, "region_id", args.out / "absorb_region.csv"),
    }
    zips_after = {p.name: sha256_file(p) for p in sorted(args.labels.glob("*.zip"))}
    # 회계 — 원본에는 있고 v1 에 없는 장, 매니페스트의 iso_codes 짝 (전 장 집계, 행 값 없음)
    v1_ids = {int(k.rsplit(":", 1)[1]) for k in man_by}
    not_in = [i for i in recs if i not in v1_ids]
    oos = sum(1 for i in not_in if not recs[i]["is_normal"]
              and any(norm_case(c) in OUT_OF_LABEL_SPACE for _, _, c in recs[i]["polys"]))
    iso_fixed = 0
    for m in man:
        dts = [t for t in m["defect_types"].split(";") if t]
        rebuilt = ";".join(iso[t] for t in dts)
        iso_fixed += rebuilt != m["iso_codes"]
    return {
        "labels_not_in_v1": {"total": len(not_in), "out_of_label_space": oos, "other": len(not_in) - oos},
        "iso_pair_fixed": iso_fixed,
        "img_rows": img_rows, "def_rows": def_rows, "reg_rows": reg_rows, "hashes": hashes,
        "raw_acct": raw_acct, "acct": dict(acct),
        "v1_mismatch_trainval": dict(v1_mismatch_tv),
        "v1_check_eval": {"all_equal": not ev_v1_bad, "unequal_images": len(ev_v1_bad)},
        "label_defect_v1_normal_trainval": dict(label_defect_v1_normal_tv),
        "zips_unchanged": zips_before == zips_after, "man_by": man_by, "recs": recs,
    }


# ------------------------------------------------------------------------------------------
# 후보 판과 맞대기 — 바이트, 다르면 칸 단위 값
# ------------------------------------------------------------------------------------------

def compare_candidate(res: dict, cand: Path, out: Path) -> dict:
    split_of = {k: v["split"] for k, v in res["man_by"].items()}
    report = {"candidate_snapshot_sha256": sha256_file(cand / "SNAPSHOT.sha256")}
    lock = (cand / "SNAPSHOT.sha256").read_text(encoding="utf-8")
    members = [ln for ln in lock.splitlines() if ln.strip() and not ln.startswith("#")]
    report["candidate_digest_recomputed"] = hashlib.sha256(("\n".join(members) + "\n").encode("utf-8")).hexdigest()
    report["candidate_member_lines"] = members
    diffs_tv = []
    for name, cols, key in (("absorb_image.csv", IMAGE_COLS, "image_id"),
                            ("absorb_defect.csv", DEFECT_COLS, "ann_id"),
                            ("absorb_region.csv", REGION_COLS, "region_id")):
        theirs_hash = sha256_file(cand / name)
        mine_hash = res["hashes"][name]
        ent = {"mine": mine_hash, "theirs": theirs_hash, "byte_equal": mine_hash == theirs_hash}
        if not ent["byte_equal"]:
            mine = {r[key]: r for r in read_csv_rows(out / name)}
            theirs = {r[key]: r for r in read_csv_rows(cand / name)}
            ent["header_equal"] = list(read_csv_rows(out / name)[0].keys()) == list(read_csv_rows(cand / name)[0].keys())
            only_m, only_t = set(mine) - set(theirs), set(theirs) - set(mine)
            col_diff_tv = Counter()                       # 열별 집계는 학습 · 검증 행만(명세 5-3)
            bad_rows = {"train_val": 0, "eval": 0}
            for k in set(mine) & set(theirs):
                cs = [c for c in cols if mine[k].get(c) != theirs[k].get(c)]
                if cs:
                    sp = "eval" if split_of.get(mine[k]["image_id"]) == "eval" else "train_val"
                    bad_rows[sp] += 1
                    if sp == "train_val":
                        for c in cs:
                            col_diff_tv[c] += 1
                        diffs_tv.append({"file": name, "key": k, "cols": cs,
                                         "mine": {c: mine[k][c] for c in cs},
                                         "theirs": {c: theirs[k][c] for c in cs}})
            ev_keys = lambda d: sorted(k for k, r in d.items() if split_of.get(r["image_id"]) == "eval")
            ev_mine = [(k, [mine[k].get(c) for c in cols]) for k in ev_keys(mine)]
            ev_theirs = [(k, [theirs[k].get(c) for c in cols]) for k in ev_keys(theirs)]
            ent.update({"only_mine": len(only_m), "only_theirs": len(only_t),
                        "rows_differing": bad_rows, "cols_differing_trainval": dict(col_diff_tv),
                        "eval_block_equal": ev_mine == ev_theirs})
        report[name] = ent
    with (out / "cand_diffs_trainval.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for d in diffs_tv:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    # 회계 — 값 같음(zip 이름 · sha256 · JSON 엔트리 수)
    caps = yaml.safe_load((cand / "data_capabilities.yaml").read_text(encoding="utf-8"))
    absn = caps.get("absorption") or {}
    report["capabilities_absorption_keys"] = sorted(absn)
    report["capabilities_zips_value_equal"] = _zips_equal(absn, res["raw_acct"]["zips"])
    return report


def _zips_equal(absn: dict, mine: list[dict]):
    """후보 회계의 zip 목록을 찾아 이름 · sha256 · 엔트리 수를 맞댄다. 꼴을 모르면 None."""
    for k, v in absn.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "sha256" in v[0]:
            theirs = {(d.get("name") or d.get("zip")): (d.get("sha256"), d.get("json_entries") or d.get("entries"))
                      for d in v}
            m = {d["name"]: (d["sha256"], d["json_entries"]) for d in mine}
            return {"key": k, "equal": theirs == m, "n_theirs": len(theirs), "n_mine": len(m)}
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()) and \
                any("sha256" in x for x in v.values()):
            theirs = {n: (d.get("sha256"), d.get("json_entries") or d.get("entries")) for n, d in v.items()}
            m = {d["name"]: (d["sha256"], d["json_entries"]) for d in mine}
            return {"key": k, "equal": theirs == m, "n_theirs": len(theirs), "n_mine": len(m)}
    return None


# ------------------------------------------------------------------------------------------
# 팀원 파일과의 교차 검산 — 겹치는 장 (명세 5-2 · 5-3)
# ------------------------------------------------------------------------------------------

def crosscheck_teammate(res: dict, jsonl: Path, out: Path) -> dict:
    man_by, recs = res["man_by"], res["recs"]
    split_of = {k: v["split"] for k, v in man_by.items()}
    ours_def = defaultdict(list)
    for r in res["def_rows"]:
        ours_def[r["image_id"]].append(r)
    for v in ours_def.values():
        v.sort(key=lambda r: int(r["ann_id"].rsplit("#", 1)[1]))
    ours_reg = defaultdict(list)
    for r in res["reg_rows"]:
        ours_reg[r["image_id"]].append(r["polygon_json"])

    res_tv = Counter()
    res_ev = Counter()
    ev_bad_images = set()
    rows_tv = []
    overlap = 0
    clip_rows = {"tv": Counter(), "all": Counter()}
    clip_imgs = {"tv": Counter(), "all": Counter()}
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            o = json.loads(line)
            iid_s = f"{SOURCE}:{int(o['orig_info_id'])}"
            if iid_s not in man_by:
                continue
            overlap += 1
            ev = split_of[iid_s] == "eval"
            tally = res_ev if ev else res_tv
            tally["images"] += 1
            rec = recs[int(o["orig_info_id"])]
            img_bad = []
            # 원본 크기
            if (int(o["width_px"]), int(o["height_px"])) != (rec["w"], rec["h"]):
                tally["size_mismatch_images"] += 1
                img_bad.append("size")
            anns = o.get("annotations") or []
            td = sorted([a for a in anns if a.get("annotation_role") == "defect"], key=lambda a: int(a["ann_idx"]))
            tr = [a for a in anns if a.get("annotation_role") == "normal_region"]
            tally["other_role_annotations"] += sum(1 for a in anns if a.get("annotation_role") not in ("defect", "normal_region"))
            od = ours_def.get(iid_s, [])
            if td or od:
                tally["defect_images"] += 1
                if len(td) != len(od):
                    tally["defect_count_mismatch_images"] += 1
                    img_bad.append("defect_count")
                else:
                    order_ok = 0
                    iso_bad = False
                    for a, r in zip(td, od):
                        tb = (int(a["bbox_xmin"]), int(a["bbox_ymin"]), int(a["bbox_xmax"]), int(a["bbox_ymax"]))
                        ob = (r["raw_bbox_x1_px"], r["raw_bbox_y1_px"], r["raw_bbox_x2_px"], r["raw_bbox_y2_px"])
                        tally["defect_pairs"] += 1
                        if tb == ob:
                            tally["pair_box_equal"] += 1
                            order_ok += 1
                        else:
                            tally["pair_box_diff"] += 1
                        if str(a.get("class_std_iso6520", "")) == r["_iso"]:
                            tally["pair_iso_equal"] += 1
                        else:
                            tally["pair_iso_diff"] += 1
                            iso_bad = True
                    if iso_bad:                               # ISO 가 다른 짝이 있으면 그 장은 같지 않다(명세 5-2)
                        tally["iso_mismatch_images"] += 1
                        img_bad.append("iso")
                    # 내용 키 — 원좌표 상자의 다중집합
                    tset = Counter((int(a["bbox_xmin"]), int(a["bbox_ymin"]), int(a["bbox_xmax"]), int(a["bbox_ymax"])) for a in td)
                    oset = Counter((r["raw_bbox_x1_px"], r["raw_bbox_y1_px"], r["raw_bbox_x2_px"], r["raw_bbox_y2_px"]) for r in od)
                    if tset == oset:
                        tally["content_key_equal_images"] += 1
                        if order_ok == len(od):
                            tally["two_keys_agree_images"] += 1
                        else:                                 # 두 키가 다른 짝을 낸다 — 같지 않다(명세 5-2)
                            tally["content_equal_order_differs_images"] += 1
                            img_bad.append("defect_order")
                    else:
                        tally["content_key_differs_images"] += 1
                        img_bad.append("defect_box")
                # 101 건의 꼴 — 겹치는 결함 장의 clip_kind (평가 행 포함 전수는 내부에서만 센다)
                kinds = Counter(r["clip_kind"] for r in od)
                for grp in (("all",) if ev else ("all", "tv")):
                    for k2, n in kinds.items():
                        clip_rows[grp][k2] += n
                    if kinds.get("empty_bbox"):
                        clip_imgs[grp]["empty_bbox"] += 1
                    elif kinds.get("boundary") or kinds.get("both"):
                        clip_imgs[grp]["boundary"] += 1
                    elif kinds.get("polygon_repair"):
                        clip_imgs[grp]["polygon_repair"] += 1
            if rec["is_normal"]:
                tally["normal_images"] += 1
                tp = Counter()
                for a in tr:
                    px = a.get("polygon_x")
                    py = a.get("polygon_y")
                    px = json.loads(px) if isinstance(px, str) else px
                    py = json.loads(py) if isinstance(py, str) else py
                    tp[json.dumps([[int(x), int(y)] for x, y in zip(px or [], py or [])], separators=(",", ":"))] += 1
                op = Counter(ours_reg.get(iid_s, []))
                tally["region_rows_ours"] += sum(op.values())
                tally["region_rows_team"] += sum(tp.values())
                if tp == op:
                    tally["region_multiset_equal_images"] += 1
                else:
                    tally["region_multiset_differs_images"] += 1
                    img_bad.append("region")
            if img_bad:
                if ev:
                    ev_bad_images.add(iid_s)
                else:
                    rows_tv.append({"image_id": iid_s, "what": img_bad})
    with (out / "xc_diffs_trainval_G.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows_tv:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "teammate_sha256": sha256_file(jsonl),
        "overlap_images": overlap,
        "train_val": dict(res_tv),
        "eval": {"images": res_ev["images"], "all_equal": not ev_bad_images,
                 "unequal_images": len(ev_bad_images)},
        "clip_kind_overlap_rows_trainval": dict(clip_rows["tv"]),
        "clip_kind_overlap_images_trainval": dict(clip_imgs["tv"]),
        # 전 겹침(평가 포함)은 값을 내지 않고 판정 03 · 08 의 공개 기대값과 같은지 한 비트만
        "clip_kind_overlap_all_equals_expected": (
            {k: v for k, v in clip_rows["all"].items() if k != "none"} == {"empty_bbox": 54, "boundary": 43, "polygon_repair": 9}
            and clip_rows["all"].get("both", 0) == 0
            and dict(clip_imgs["all"]) == {"empty_bbox": 53, "boundary": 39, "polygon_repair": 9}),
    }


# ------------------------------------------------------------------------------------------
# 결정 규칙 표 — 명세 6절
# ------------------------------------------------------------------------------------------

KINDS = ("none", "boundary", "polygon_repair", "both", "empty_bbox")


def chi2_stat(tab: np.ndarray) -> tuple[float, np.ndarray]:
    tab = tab[:, tab.sum(axis=0) > 0]
    r, c = tab.sum(axis=1, keepdims=True), tab.sum(axis=0, keepdims=True)
    exp = r @ c / tab.sum()
    return float(((tab - exp) ** 2 / exp).sum()), exp


def decision_table(res: dict) -> dict:
    from scipy.stats import chi2 as chi2_dist

    tv = [r for r in res["def_rows"] if r["_split"] != "eval"]
    clients = sorted({r["_client"] for r in tv})
    tab = np.array([[sum(1 for r in tv if r["_client"] == cl and r["clip_kind"] == k) for k in KINDS]
                    for cl in clients], dtype=float)
    stat, exp = chi2_stat(tab)
    keep = tab.sum(axis=0) > 0
    dof = (tab.shape[0] - 1) * (int(keep.sum()) - 1)
    p_asym = float(chi2_dist.sf(stat, dof))
    out = {"rows": len(tv), "clients": clients, "kinds": list(KINDS),
           "table": tab.astype(int).tolist(), "chi2": stat, "dof": dof, "p_asymptotic": p_asym,
           "min_expected": float(exp.min())}
    if exp.min() < 5:
        # 정확 검정 대용 — 행 라벨을 섞는 순열 검정(시드 고정)
        rng = np.random.default_rng(20261001)
        cl_idx = np.array([clients.index(r["_client"]) for r in tv])
        k_idx = np.array([KINDS.index(r["clip_kind"]) for r in tv])
        n_perm, ge = 20000, 0
        for _ in range(n_perm):
            perm = rng.permutation(k_idx)
            t = np.bincount(cl_idx * len(KINDS) + perm, minlength=tab.shape[0] * len(KINDS)).reshape(tab.shape).astype(float)
            s, _e = chi2_stat(t)
            ge += s >= stat - 1e-9
        out["p_permutation"] = (ge + 1) / (n_perm + 1)
        out["n_permutation"] = n_perm
    p = out.get("p_permutation", p_asym)
    out["decision"] = "진단 전용으로 내린다" if p < 0.01 else "허용 목록에 둔다"
    # (나) 결함 종류별 · (다) ST 안 id 구간별 — 해석용
    dts = sorted({r["_defect_type"] for r in tv})
    out["by_defect_type"] = {d: dict(Counter(r["clip_kind"] for r in tv if r["_defect_type"] == d)) for d in dts}
    ids_tv = np.array([int(k.rsplit(":", 1)[1]) for k, v in res["man_by"].items() if v["split"] != "eval"], dtype=float)
    for K in (4, 8, 16):
        qs = np.unique(np.quantile(ids_tv, np.linspace(0, 1, K + 1)[1:-1]))
        if qs.size < K - 1:
            qs = np.concatenate([qs, np.full(K - 1 - qs.size, np.inf)])
        st = defaultdict(Counter)
        for r in tv:
            if r["_material"] != "ST":
                continue
            b = int(np.searchsorted(qs, float(r["image_id"].rsplit(":", 1)[1]), side="right"))
            st[b][r["clip_kind"]] += 1
        out[f"st_by_idq{K}"] = {str(b): dict(st[b]) for b in sorted(st)}
    return out


def extra_acct(res: dict) -> dict:
    """학습 · 검증 행의 추가 회계 — `frame_match` · 결함 · 영역 행 수와 정수 아닌 좌표. 평가 행은 넣지 않는다."""
    split_of = {k: v["split"] for k, v in res["man_by"].items()}
    tv_def = [r for r in res["def_rows"] if r["_split"] != "eval"]
    tv_reg = [r for r in res["reg_rows"] if r["_split"] != "eval"]
    fm = Counter(r["frame_match"] for r in res["img_rows"] if split_of[r["image_id"]] != "eval")
    per_img = defaultdict(Counter)
    for r in tv_def:
        per_img[r["image_id"]][r["clip_kind"]] += 1
    imgs = Counter()
    for c in per_img.values():
        if c.get("empty_bbox"):
            imgs["empty_bbox"] += 1
        elif c.get("boundary") or c.get("both"):
            imgs["boundary"] += 1
        elif c.get("polygon_repair"):
            imgs["polygon_repair"] += 1
    rpi = Counter(Counter(r["image_id"] for r in tv_reg).values())
    return {
        "trainval_images": sum(fm.values()),
        "trainval.frame_match": dict(fm),
        "trainval.defect_rows": len(tv_def),
        "trainval.clip_kind_images": dict(imgs),
        "trainval.region_rows": len(tv_reg),
        "trainval.regions_per_image": {str(k): v for k, v in sorted(rpi.items())},
        "trainval.defect_rows_float_type": sum(r["_float"] for r in tv_def),
        "trainval.defect_rows_noninteger_value": sum(r["_nonint"] for r in tv_def),
        "trainval.region_rows_float_type": sum(r["_float"] for r in tv_reg),
        "trainval.region_rows_noninteger_value": sum(r["_nonint"] for r in tv_reg),
        "trainval.images_with_noninteger_value": len({r["image_id"] for r in tv_def + tv_reg if r["_nonint"]}),
        "labels_not_in_v1": res["labels_not_in_v1"],
        "iso_pair_fixed": res["iso_pair_fixed"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--label-map", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="_workspace 아래의 없는 새 폴더")
    args = ap.parse_args()

    from scripts.crosscheck.xc_out_guard import guard_new_output, recheck_inside

    # 입력을 읽거나 폴더를 만들기 전에 — 새 폴더만, _workspace 아래만, 입력 · 동결 디렉터리와 겹치지 않게
    guard_new_output(args.out, repo=REPO,
                     inputs=(args.labels, args.v1, args.label_map, args.jsonl, args.candidate))
    args.out.mkdir(parents=True)
    recheck_inside(args.out, repo=REPO)
    res = build(args)
    summary = {
        "sidecar_sha256": res["hashes"], "raw_acct": res["raw_acct"], "acct": res["acct"],
        "v1_mismatch_trainval": res["v1_mismatch_trainval"], "v1_check_eval": res["v1_check_eval"],
        "label_defect_v1_normal_trainval": res["label_defect_v1_normal_trainval"],
        "zips_unchanged": res["zips_unchanged"],
        "rows": {"absorb_image": len(res["img_rows"]), "absorb_defect": len(res["def_rows"]),
                 "absorb_region": len(res["reg_rows"])},
        "clip_kind_rows_trainval": dict(Counter(r["clip_kind"] for r in res["def_rows"] if r["_split"] != "eval")),
    }
    (args.out / "summary_G.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("곁파일 셋:", json.dumps(res["hashes"], indent=1))
    summary["candidate"] = compare_candidate(res, args.candidate, args.out)
    summary["teammate"] = crosscheck_teammate(res, args.jsonl, args.out)
    summary["decision"] = decision_table(res)
    summary["acct_trainval_extra"] = extra_acct(res)
    (args.out / "summary_G.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("candidate", "zips_unchanged", "v1_mismatch_trainval", "v1_check_eval")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Stop as exc:
        print(f"!! 멈춤: {exc}")
        sys.exit(3)
