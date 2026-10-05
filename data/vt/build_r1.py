"""VT 구현 회차 1 의 실행 — 파싱 · 화소 꼴 계획 · 재인코딩 품질(R-9) · 다시 인코딩 (4판 5절).

    uv run python -X utf8 -m data.vt.build_r1 plan      # 라벨 → 계획 · 자른 상자 기록 · 제외 장부 (픽셀을 읽지 않는다)
    uv run python -X utf8 -m data.vt.build_r1 quality   # R-9 — 헤더(DQT)만 읽는다
    uv run python -X utf8 -m data.vt.build_r1 encode    # 계획대로 다시 인코딩 (병렬도 2 이하, 이어 하기)

**평가 분할을 만들지 않는다.** 산출은 스테이징(`staging_root`)에만 쓰고 분할 칸을 두지 않는다(VT-1d).
장 단위 값(라벨 · 상자 · 영상)은 스테이징에만 있고 추적하지 않는다. 요약에는 집계와 해시만 싣는다.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data.vt.config import REPO_ROOT, load_config, repo_path
from data.vt.copy import load_jsonl_tolerant, repair_jsonl_tail
from data.vt.guard import Guard, assert_vt_writable
from data.vt.labels import (
    Ann,
    VtLabel,
    author_split_of,
    parse_label_zips,
    zip_member_index,
    zip_name_of,
)
from data.vt.pixel import band_inside, encode, plan
from data.vt.records import read_csv_checked, write_csv

CROP_HEADER = ["image_id", "rule_version", "src_w", "src_h", "rot90_k", "x0", "y0", "x1", "y1", "provenance", "reason"]
EXCL_HEADER = ["image_id", "folder", "src_w", "src_h", "reason"]
# 계약 #2 의 `tiles.csv`(출처 칸)와 이름이 겹치지 않게 한다 — 같은 이름 다른 뜻을 만들지 않는다
ENCODED_FILE = "encoded_tiles.csv"
ENCODED_HEADER = ["image_id", "path", "sha256", "bytes"]


def label_to_json(lab: VtLabel, excluded: str | None, cfg: dict) -> dict:
    return {"vt_id": lab.vt_id, "image_id": lab.image_id, "folder": lab.folder,
            "src_zip": zip_name_of(lab.author_split, lab.folder, "image", cfg), "file_stem": lab.file_stem,
            "w": lab.width, "h": lab.height, "is_normal": lab.is_normal, "label_excluded": excluded,
            "anns": [{"cls": a.cls, "case": a.case, "xs": list(a.xs), "ys": list(a.ys)} for a in lab.anns]}


def label_from_json(d: dict, cfg: dict) -> VtLabel:
    split = author_split_of(d["src_zip"], cfg)
    return VtLabel(vt_id=d["vt_id"], image_id=d["image_id"], folder=d["folder"], author_split=split,
                   file_stem=d["file_stem"], width=d["w"], height=d["h"], is_normal=d["is_normal"],
                   anns=tuple(Ann(a["cls"], a["case"], tuple(a["xs"]), tuple(a["ys"])) for a in d["anns"]))


def load_labels(staging: Path) -> list[dict]:
    """파싱 산출을 읽는다. 계획 요약에 해시가 있으면 맞댄 뒤에만 읽는다(06b I-7 ②)."""
    p = staging / "parse" / "labels.jsonl"
    if not p.exists():
        raise SystemExit(f"파싱 산출이 없다 — plan 단계를 먼저 돈다: {p}")
    raw = p.read_bytes()
    sp = staging / "records" / "plan_summary.json"
    if sp.exists():
        want = json.loads(sp.read_text(encoding="utf-8")).get("records", {}).get("labels.jsonl")
        if want is None:
            raise SystemExit("계획 요약에 labels.jsonl 의 해시가 없다 — plan 단계를 다시 돈다(같은 바이트를 낸다)")
        if hashlib.sha256(raw).hexdigest() != want:
            raise SystemExit("labels.jsonl 의 바이트가 계획 요약의 해시와 다르다")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def require_copy_verified(staging: Path) -> None:
    """원천 E: 사본을 읽는 단계의 선행 조건 — 복사 대조가 지났다(4판 4절 · 5-2)."""
    v = staging / "copy" / "copy_verify.json"
    d = json.loads(v.read_text(encoding="utf-8")) if v.exists() else {}
    if not d.get("all_ok"):
        raise SystemExit("원천 복사 대조가 지나지 않았다 — data.vt.copy 를 먼저 돈다")
    if not d.get("image_content_evidence_all"):
        # 원천 이미지 zip 은 서버 md5 가 유일한 내용 증거다 — 없으면 크기만 맞은 사본을 읽게 된다(06b I-6)
        raise SystemExit("원천 이미지 zip 가운데 내용 증거(서버 md5)가 없는 것이 있다 — 대조를 다시 돈다")


def _paths(cfg: dict) -> dict:
    staging = repo_path(cfg["staging_root"])
    return {"staging": staging, "raw": repo_path(cfg["copy"]["raw_zip_dir"]),
            "labels": repo_path(cfg["copy"]["label_zip_dir"]), "records": staging / "records",
            "tiles": staging / "tiles" / "VT" / "ST"}


def _writable(cfg: dict, *targets: Path) -> None:
    roots = [repo_path(cfg["staging_root"])]
    for t in targets:
        assert_vt_writable(t, allowed_roots=roots, repo_root=REPO_ROOT)


def _wait(cfg: dict) -> None:
    """단계의 시작에서 본체 게이트 락 · 여유 메모리를 본다(06b m-5)."""
    Guard.from_config(cfg, REPO_ROOT, log=lambda d: print(json.dumps(d, ensure_ascii=False)), check_c=False).wait()


def _discard_limits() -> dict:
    import yaml
    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    return base["preprocess"]["discard_limits"]


# ------------------------------------------------------------------------------------------
# plan
# ------------------------------------------------------------------------------------------
def cmd_plan(cfg: dict) -> int:
    P = _paths(cfg)
    _writable(cfg, P["staging"] / "parse", P["records"])
    require_copy_verified(P["staging"])                 # 라벨 zip 도 복사 대조를 지난 사본만 읽는다(06b m-11)
    _wait(cfg)
    res = parse_label_zips(P["labels"], cfg)
    labels_out = [(lab, None) for lab in res.labels]
    excl_rows, no_label = [], []
    for e in res.excluded:
        if "label" in e:
            labels_out.append((e["label"], e["reason"]))
            lab = e["label"]
            excl_rows.append({"image_id": lab.image_id, "folder": lab.folder, "src_w": lab.width,
                              "src_h": lab.height, "reason": e["reason"]})
        else:
            no_label.append(e)
            excl_rows.append({"image_id": f"{cfg['source_key']}:{e['vt_id']}", "folder": e["folder"],
                              "src_w": "", "src_h": "", "reason": e["reason"]})
    labels_out.sort(key=lambda x: x[0].vt_id)
    (P["staging"] / "parse").mkdir(parents=True, exist_ok=True)
    lab_bytes = "".join(json.dumps(label_to_json(lab, ex, cfg), ensure_ascii=False) + "\n"
                        for lab, ex in labels_out).encode("utf-8")
    lab_tmp = P["staging"] / "parse" / "labels.jsonl.tmp"          # 임시 이름에 쓰고 바꾼다(06b I-7 ②)
    lab_tmp.write_bytes(lab_bytes)
    os.replace(lab_tmp, P["staging"] / "parse" / "labels.jsonl")
    h_labels = hashlib.sha256(lab_bytes).hexdigest()

    crop_rows = []
    acct: dict[str, Counter] = defaultdict(Counter)
    band_out = Counter()
    for lab, ex in labels_out:
        a = acct[lab.folder]
        a["total"] += 1
        a["ann_missing_case"] += sum(1 for x in lab.anns
                                     if x.cls == cfg["labels"]["defect_class"] and x.case == cfg["labels"]["missing_case"])
        if ex:
            a[f"excluded:{ex}"] += 1
            continue
        p = plan(lab, cfg)
        if not p.keep:
            a[f"excluded:{p.reason}"] += 1
            excl_rows.append({"image_id": p.image_id, "folder": lab.folder, "src_w": p.src_w, "src_h": p.src_h,
                              "reason": p.reason})
            continue
        a["kept"] += 1
        a[f"prov:{p.provenance}"] += 1
        a[f"reason:{p.reason}"] += 1
        if p.rot90_k:
            a["rotated"] += 1
        bi = band_inside(lab, p, cfg)
        if bi is not None:
            band_out["checked"] += 1
            band_out["band_not_inside"] += (not bi)
        crop_rows.append({"image_id": p.image_id, "rule_version": cfg["tile"]["rule_version"], "src_w": p.src_w,
                          "src_h": p.src_h, "rot90_k": p.rot90_k, "x0": p.box[0], "y0": p.box[1], "x1": p.box[2],
                          "y1": p.box[3], "provenance": p.provenance, "reason": p.reason})
    excl_rows.sort(key=lambda r: int(r["image_id"].split(":")[1]))
    h_crop = write_csv(P["records"] / "crop_boxes.csv", CROP_HEADER, crop_rows)
    h_excl = write_csv(P["records"] / "exclusions.csv", EXCL_HEADER, excl_rows)

    lim = _discard_limits()
    cells = {}
    for f, a in sorted(acct.items()):
        disc = a["total"] - a["kept"]
        rate = disc / a["total"]
        is_normal = f == cfg["labels"]["normal_folder"]
        cap = lim["normal_per_material"] if is_normal else max(lim["defect_per_cell_abs"] / a["total"],
                                                               lim["defect_per_cell_ratio"])
        cells[f] = {**dict(a), "discarded": disc, "discard_rate_pct": round(rate * 100, 3),
                    "cap_pct": round(cap * 100, 3), "over_cap": rate > cap}
    n_lab = len(labels_out) + len(no_label)
    summary = {"labels_parsed": n_lab, "kept": len(crop_rows), "excluded": len(excl_rows),
               "accounting_ok": n_lab == len(crop_rows) + len(excl_rows), "cells": cells,
               "band_containment": dict(band_out),
               "records": {"crop_boxes.csv": h_crop, "exclusions.csv": h_excl, "labels.jsonl": h_labels}}
    known = set(cfg["tile"].get("known_over_cap", []))
    summary["over_cap_unregistered"] = sorted(f for f, c in cells.items() if c["over_cap"] and f not in known)
    (P["records"] / "plan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                                                    encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, ensure_ascii=False))
    for f, c in cells.items():
        print(f, json.dumps(c, ensure_ascii=False))
    if summary["over_cap_unregistered"]:
        print(f"등록된 예외 밖의 셀이 폐기 상한을 넘었다: {summary['over_cap_unregistered']}")     # 06b m-2
        return 8
    return 0 if summary["accounting_ok"] else 5


# ------------------------------------------------------------------------------------------
# quality — R-9
# ------------------------------------------------------------------------------------------
def cmd_quality(cfg: dict) -> int:
    from scripts.measure_jpeg_fingerprint import estimate_quality, parse_jpeg_header

    P = _paths(cfg)
    _writable(cfg, P["records"])
    require_copy_verified(P["staging"])
    _wait(cfg)
    labs = load_labels(P["staging"])
    by_zip: dict[str, list[dict]] = defaultdict(list)
    for d in labs:
        by_zip[d["src_zip"]].append(d)
    pops: dict[str, list[float]] = defaultdict(list)
    dqt: dict[str, Counter] = defaultdict(Counter)
    fails = 0
    for zname, rows in sorted(by_zip.items()):
        with zipfile.ZipFile(P["raw"] / zname) as zf:
            idx = zip_member_index(zf)
            for d in rows:
                with zf.open(idx[d["file_stem"]]) as fh:
                    head = fh.read(1 << 16)
                info = parse_jpeg_header(head)
                if not info or not info["dqt"]:
                    fails += 1
                    continue
                lum = next((t for tq, t in info["dqt"] if tq == 0), info["dqt"][0][1])
                pop = "normal" if d["is_normal"] else "defect"
                pops[pop].append(estimate_quality(lum))
                dqt[pop][hashlib.sha256(bytes(lum)).hexdigest()[:12]] += 1
        print(zname, len(rows), flush=True)

    def q(s, f):
        return s[min(len(s) - 1, int(f * len(s)))]
    out = {"rule": "두 모집단 최저치 가운데 작은 값의 내림 (R-9)", "header_fail": fails, "populations": {}}
    for pop, qs in sorted(pops.items()):
        s = sorted(qs)
        out["populations"][pop] = {"n": len(s), "min": s[0], "p01": q(s, 0.01), "p05": q(s, 0.05),
                                   "median": statistics.median(s), "max": s[-1], "distinct_dqt": len(dqt[pop])}
    value = int(min(v["min"] for v in out["populations"].values()))
    out["quality"] = value
    out["stop"] = value < int(cfg["encode"]["quality_floor_stop"]) or fails > 0
    out["n_total"] = sum(v["n"] for v in out["populations"].values())
    (P["records"] / "r9_quality.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps(out, ensure_ascii=False))
    return 6 if out["stop"] else 0


# ------------------------------------------------------------------------------------------
# encode
# ------------------------------------------------------------------------------------------
def _encode_batch(args: tuple) -> list[dict]:
    zpath, items, cfg, tiles_dir = args
    from data.vt.pixel import VtPlan
    out = []
    with zipfile.ZipFile(zpath) as zf:
        idx = zip_member_index(zf)
        for stem, p_dict in items:
            p = VtPlan(**{**p_dict, "box": tuple(p_dict["box"])})
            data, rec = encode(zf.read(idx[stem]), p, cfg)
            dst = Path(tiles_dir) / f"{p.vt_id}.jpg"
            tmp = dst.with_name(dst.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            out.append(rec)
    return out


def check_r9(cfg: dict, r9: dict) -> None:
    """설정의 품질이 R-9 산출과 같고 멈춤이 아니다 — 사람이 옮긴 값을 기계가 맞댄다."""
    if r9.get("stop"):
        raise SystemExit("R-9 가 멈춤을 냈다 — 4판 9절로 올린다")
    if cfg["encode"]["quality"] != r9["quality"]:
        raise SystemExit(f"설정의 품질 {cfg['encode']['quality']} 이 R-9 산출 {r9['quality']} 과 다르다")


def encode_header(cfg: dict, crop_sha: str, r9_sha: str) -> dict:
    """이어 하기의 열쇠(핵심) — 이것이 달라진 진행 기록은 이어 받지 않는다."""
    return {"crop_boxes_sha256": crop_sha, "r9_quality_sha256": r9_sha, "rule_version": cfg["tile"]["rule_version"],
            "encode": {k: cfg["encode"][k] for k in ("mode", "quality", "progressive", "optimize", "strip_exif")}}


def encode_provenance(staging: Path) -> dict:
    """진행 기록 머리의 출처 — 파싱 산출 · 복사 기록 · 화소 꼴 코드의 바이트 · Pillow 판(06b I-7 ③)."""
    import PIL

    from data.vt import pixel

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return {"labels_jsonl_sha256": sha(staging / "parse" / "labels.jsonl"),
            "copy_record_sha256": sha(staging / "copy" / "copy_record.jsonl"),
            "pixel_module_sha256": sha(Path(pixel.__file__)), "pillow": PIL.__version__}


def read_progress(path: Path, header: dict, provenance: dict | None = None) -> set[str]:
    """진행 기록의 첫 줄은 머리(`header`)다. 머리가 다르면 멈춘다. 마지막 줄만 잘렸으면 그 줄을 버린다.

    출처(`provenance`)를 주면 머리의 출처와도 맞댄다. 출처가 없는 옛 머리는 핵심만 맞댄다(10-04 의 실물 — 그 실행은
    한 번에 끝나 섞이지 않았다, 06b I-7 ③).
    """
    rows = load_jsonl_tolerant(path)
    if not rows:
        return set()
    stored = dict(rows[0].get("header") or {})
    got_prov = stored.pop("provenance", None)
    if provenance is not None and got_prov is not None and got_prov != provenance:
        raise SystemExit(f"진행 기록의 출처가 지금과 다르다 — 옆 이름으로 비켜 두고 다시 돈다: {path}")
    if stored != header:
        raise SystemExit(f"진행 기록의 머리가 지금 설정 · 계획과 다르다 — 진행 기록과 영상 폴더를 옆 이름으로 비켜 두고(지우지 않는다) 다시 돈다: {path}")
    return {r["image_id"] for r in rows[1:]}


def cmd_encode(cfg: dict, workers: int, batch: int) -> int:
    if workers > 2:
        raise SystemExit("병렬도는 2 이하다")
    P = _paths(cfg)
    _writable(cfg, P["tiles"], P["records"])
    require_copy_verified(P["staging"])
    if cfg["encode"]["quality"] is None:
        raise SystemExit("encode.quality 가 비어 있다 — quality 단계(R-9)를 먼저 돌리고 설정에 적는다")
    r9_path = P["records"] / "r9_quality.json"
    check_r9(cfg, json.loads(r9_path.read_text(encoding="utf-8")))
    summ = json.loads((P["records"] / "plan_summary.json").read_text(encoding="utf-8"))
    crops = read_csv_checked(P["records"] / "crop_boxes.csv", summ["records"]["crop_boxes.csv"])
    labs = {d["image_id"]: d for d in load_labels(P["staging"])}
    P["tiles"].mkdir(parents=True, exist_ok=True)
    prog_path = P["records"] / "encode_progress.jsonl"
    header = encode_header(cfg, summ["records"]["crop_boxes.csv"], hashlib.sha256(r9_path.read_bytes()).hexdigest())
    prov = encode_provenance(P["staging"])
    done = read_progress(prog_path, header, prov)
    repair_jsonl_tail(prog_path)                        # 잘린 꼬리 뒤에 붙이지 않는다(06b I-7 ①)
    if not prog_path.exists() or not prog_path.stat().st_size:
        with prog_path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"header": {**header, "provenance": prov}}, ensure_ascii=False) + "\n")
    by_zip: dict[str, list] = defaultdict(list)
    for r in crops:
        if r["image_id"] in done:
            continue
        d = labs[r["image_id"]]
        p = {"image_id": r["image_id"], "vt_id": d["vt_id"], "keep": True, "reason": r["reason"],
             "provenance": r["provenance"], "rot90_k": int(r["rot90_k"]),
             "box": [int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])],
             "src_w": int(r["src_w"]), "src_h": int(r["src_h"])}
        by_zip[d["src_zip"]].append((d["file_stem"], p))
    tasks = [(str(P["raw"] / z), items[i:i + batch], cfg, str(P["tiles"]))
             for z, items in sorted(by_zip.items()) for i in range(0, len(items), batch)]
    logf = (P["records"] / "encode_log.jsonl").open("a", encoding="utf-8", newline="\n")

    def log(d):
        d = {"t": time.strftime("%H:%M:%S"), **d}
        logf.write(json.dumps(d, ensure_ascii=False) + "\n")
        logf.flush()
        print(json.dumps(d, ensure_ascii=False), flush=True)

    guard = Guard.from_config(cfg, REPO_ROOT, log=log, check_c=False)      # E: 사본만 읽는다
    log({"start": True, "todo_batches": len(tasks), "done_before": len(done)})
    t0 = time.monotonic()
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex, prog_path.open("a", encoding="utf-8", newline="\n") as pf:
        pending = []
        for t in tasks:
            guard.wait()
            pending.append(ex.submit(_encode_batch, t))
            while len(pending) >= workers:
                for rec in pending.pop(0).result():
                    pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                pf.flush()
                log({"encoded": n, "seconds": round(time.monotonic() - t0, 1)})
        for f in pending:
            for rec in f.result():
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        log({"encoded": n, "seconds": round(time.monotonic() - t0, 1), "end": True})
    return finalize_tiles(cfg)


def finalize_tiles(cfg: dict) -> int:
    """진행 기록에서 영상 목록을 내고 계획 · **파일**과 맞댄다.

    - 영상 파일 전부를 다시 읽어 sha256 · 크기를 진행 기록과 맞댄다.
    - 원천 zip 의 영상 멤버 집합이 라벨 전수와 같은지 본다(회계와 따로 선 대조).
    """
    from PIL import Image

    P = _paths(cfg)
    _writable(cfg, P["records"])
    _wait(cfg)
    summ = json.loads((P["records"] / "plan_summary.json").read_text(encoding="utf-8"))
    crops = read_csv_checked(P["records"] / "crop_boxes.csv", summ["records"]["crop_boxes.csv"])
    r9_path = P["records"] / "r9_quality.json"
    header = encode_header(cfg, summ["records"]["crop_boxes.csv"], hashlib.sha256(r9_path.read_bytes()).hexdigest())
    rows_all = load_jsonl_tolerant(P["records"] / "encode_progress.jsonl")
    stored = dict((rows_all[0].get("header") or {}) if rows_all else {})
    stored_prov = stored.pop("provenance", None)
    if not rows_all or stored != header:
        raise SystemExit("진행 기록의 머리가 지금 설정 · 계획과 다르다")
    recs = {r["image_id"]: r for r in rows_all[1:]}
    want = [r["image_id"] for r in crops]
    missing = [i for i in want if i not in recs]
    rows, sizes, bad = [], Counter(), []
    for i in want:
        if i not in recs:
            continue
        r = recs[i]
        rel = f"VT/ST/{i.split(':')[1]}.jpg"
        f = P["staging"] / "tiles" / rel
        data = f.read_bytes() if f.exists() else b""
        if hashlib.sha256(data).hexdigest() != r["sha256"]:
            bad.append(i)
            continue
        with Image.open(io.BytesIO(data)) as im:
            sizes[im.size] += 1
        rows.append({"image_id": i, "path": rel, "sha256": r["sha256"], "bytes": r["bytes"]})
    h_tiles = write_csv(P["records"] / ENCODED_FILE, ENCODED_HEADER, rows)
    labs = load_labels(P["staging"])
    stems: dict[str, set] = defaultdict(set)
    for d in labs:
        stems[d["src_zip"]].add(d["file_stem"])
    members_ok = True
    for z, want_stems in stems.items():
        with zipfile.ZipFile(P["raw"] / z) as zf:
            have = set(zip_member_index(zf))
        members_ok &= have == want_stems
    # 폴더의 파일 집합 = 계획 — 남은 임시 파일이나 옛 영상이 봉인 때 딸려 가지 않게(06b m-3)
    planned_files = {f"{i.split(':')[1]}.jpg" for i in want}
    on_disk = {p.name for p in (P["staging"] / "tiles" / "VT" / "ST").iterdir()}
    extra = sorted(on_disk - planned_files)
    out = {"planned": len(want), "encoded": len(rows), "missing": len(missing), "file_hash_mismatch": len(bad),
           "extra_files_in_tile_dir": len(extra), "provenance": stored_prov or "옛 머리 — 출처 없음(10-04 실행)",
           "sizes": {f"{w}x{h}": c for (w, h), c in sizes.items()},
           "total_bytes": sum(int(r["bytes"]) for r in rows),
           "image_members_equal_labels": members_ok, "header": header,
           "records": {ENCODED_FILE: h_tiles}}
    (P["records"] / "encode_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                      encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in out.items() if k != "header"}, ensure_ascii=False))
    ok = not missing and not bad and not extra and list(sizes) == [(1280, 720)] and members_ok
    return 0 if ok else 7


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "quality", "encode", "finalize"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=250)
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.stage == "plan":
        return cmd_plan(cfg)
    if args.stage == "quality":
        return cmd_quality(cfg)
    if args.stage == "encode":
        return cmd_encode(cfg, args.workers, args.batch)
    return finalize_tiles(cfg)


if __name__ == "__main__":
    sys.exit(main())
