"""원본 라벨에서 전 장의 곁파일 셋을 낸다 — 전수 메타데이터화 1단계.

절 번호는 전수 메타데이터화 미니스펙 2판의 것이다.

  §2    곁파일 셋(`absorb_image` · `absorb_defect` · `absorb_region` 의 원본 프레임 행)을 원본 라벨에서 낸다.
        새 경로 `data/interim/manifest_v3_rawlabels`. 동결본은 읽기만 한다
  §2-3  `clip_kind` 의 분류 규칙 — 흡수 판과 같은 규칙에 `empty_bbox` 하나를 더했다
  §4    입력 규약. 수집 단계(v1 을 만든 코드)의 읽기 · 거르기 · 순번을 그대로 재현하고,
        **v1 과 주석마다 바이트로 맞대어** 어긋나면 쓰기 전에 멈춘다
  §5-3  평가 행의 값은 회계에 싣지 않는다. 평가 행은 장 수만 센다
  §7    V-8 원본 zip 은 읽기만 한다 — 시작과 끝의 sha256 이 같아야 한다.
        V-12 잠긴 판이 있는 경로에는 쓰지 않는다

**원본 라벨 zip 은 압축을 풀지 않고 읽기만 한다**(불변조건 1-1). 화소는 읽지 않는다 — 1단계의 범위다.
사상은 수집 스크립트의 것을 **가져다 쓴다.** 사본을 만들면 두 사상이 갈릴 수 있다.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data import absorb
from data.convert.geometry import polygon_metrics
from data.frozen_guard import assert_writable
from data.label_map import load_label_map
from data.manifest_io import (
    ABSORB_DEFECT_COLUMNS,
    ABSORB_IMAGE_COLUMNS,
    ABSORB_REGION_COLUMNS,
    SNAPSHOT_FILENAME,
    load_snapshot,
    verify_snapshot,
    write_snapshot,
)
from scripts.build_manifest_v0 import CASE_TO_RAW, OUT_OF_LABEL_SPACE, _norm_case, _raw_label

SOURCE = "aihub71761"
SNAPSHOT_ID = "aihub71761_rt_v3_rawlabels"
TILE_W, TILE_H = 1280, 720
#: §7 V-12. 이 이름의 경로에는 쓰지 않는다 — 동결본과 잠정판이다.
PROTECTED_NAMES = frozenset({"manifest_v1", "manifest_v2_absorbed"})
#: §11. 두께 · 화소 크기 키를 찾는 사전. **수를 보기 전에 적었다.** 키 이름(소문자)에 이 조각이 들면 후보다.
SCALE_KEY_PARTS: tuple[str, ...] = (
    "thick", "mm", "pixel", "px", "scale", "resolution", "dpi", "spacing", "iqi",
    "두께", "픽셀", "해상도", "배율")
#: 문자열 값에 숫자 + mm 꼴이 들면 후보다.
SCALE_VALUE = re.compile(r"\d+(?:\.\d+)?\s*mm\b", re.IGNORECASE)


class RawMetaError(RuntimeError):
    """원본 라벨에서 곁파일을 낼 수 없다. 값을 내지 않고 멈춘다."""


@dataclass
class LabelRec:
    """라벨 JSON 한 장. 수집 단계의 `ImageRec` 과 같은 값을 갖고, 어느 zip 에서 왔는지를 더한다."""

    info_id: int
    modality: str
    material: str
    width: int
    height: int
    is_normal: bool
    file_name: str
    zip_name: str
    polys: list[tuple[list, list]] = field(default_factory=list)
    poly_cases: list[str] = field(default_factory=list)

    @property
    def image_id(self) -> str:
        return f"{SOURCE}:{self.info_id}"


def _walk_keys(o: Any, prefix: str, paths: Counter, hits: Counter) -> None:
    """키 경로를 **전 원소**에 대해 센다. 두께 · 화소 크기 후보 키와 값을 센다(§11)."""
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            paths[p] += 1
            lk = str(k).lower()
            if any(part in lk for part in SCALE_KEY_PARTS):
                hits[f"key:{p}"] += 1
            _walk_keys(v, p, paths, hits)
    elif isinstance(o, list):
        for v in o:
            _walk_keys(v, prefix + "[]", paths, hits)
    elif isinstance(o, str) and SCALE_VALUE.search(o):
        hits[f"value:{prefix}"] += 1


def read_label_zips(label_root: Path) -> dict[str, Any]:
    """§4 규칙 0 ~ 3. 원본 라벨 zip 을 전수 읽는다. 압축을 풀지 않는다.

    수집 단계의 `read_labels` 와 같은 순서 · 같은 예외 처리다 — 파싱이 실패한 장은 통째로 빠지고 센다.
    `info.type` 이 RT 가 아닌 장은 뺀다. 반환은 기록 목록과 회계다.
    """
    zips = sorted(Path(label_root).glob("*.zip"))
    if not zips:
        raise RawMetaError(f"라벨 zip 을 하나도 찾지 못했다: {label_root}")
    recs: list[LabelRec] = []
    failures: Counter = Counter()
    non_rt = 0
    entries: dict[str, int] = {}
    paths: Counter = Counter()
    hits: Counter = Counter()
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".json")]
            entries[zp.name] = len(names)
            for name in names:
                try:
                    o = json.loads(z.read(name).decode("utf-8-sig"))
                    info, img = o["info"], o["image_data"]
                    anns = o.get("annotations") or []
                    is_normal = all(a.get("class") != "defect" for a in anns) or not anns
                    rec = LabelRec(info_id=int(info["id"]), modality=str(info["type"]),
                                   material=str(info["material"]), width=int(img["width"]),
                                   height=int(img["height"]), is_normal=is_normal,
                                   file_name=str(img.get("file_name") or ""), zip_name=zp.name)
                    for a in anns:
                        c = a.get("coordinate") or {}
                        xs, ys = c.get("x") or [], c.get("y") or []
                        if len(xs) >= 3 and len(xs) == len(ys):
                            rec.polys.append((xs, ys))
                            rec.poly_cases.append(str(a.get("case", "")))
                except (KeyError, ValueError, TypeError, UnicodeDecodeError):
                    failures[zp.name] += 1
                    continue
                _walk_keys(o, "", paths, hits)
                if rec.modality != "RT":
                    non_rt += 1
                    continue
                recs.append(rec)
    return {"records": recs, "parse_failures": dict(failures), "non_rt": non_rt,
            "json_entries": entries, "key_paths": dict(paths), "scale_key_hits": dict(hits)}


def zip_digests(label_root: Path) -> dict[str, str]:
    out = {}
    for zp in sorted(Path(label_root).glob("*.zip")):
        h = hashlib.sha256()
        with zp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[zp.name] = h.hexdigest()
    return out


def polygon_json(xs, ys) -> str:
    """수집 단계와 같은 직렬화 — `int()` · 공백 없는 구분자."""
    return json.dumps([[int(x), int(y)] for x, y in zip(xs, ys, strict=True)], separators=(",", ":"))


def raw_box(xs, ys) -> tuple[int, int, int, int]:
    """원본 좌표의 최솟값 · 최댓값. 자르지 않는다.

    **정수가 아닌 좌표는 바깥으로 정수화한다** — 최솟값은 내림, 최댓값은 올림. v1 상자 계산
    (`polygon_metrics` 의 floor · ceil)과 같은 방향이다. `int()` 로 깎으면 실수 좌표가 있는 장에서
    원좌표 상자가 v1 상자보다 작아져 자르기로 설명되지 않는 차이가 생긴다(1단계 실물에서 4 장).
    정수 좌표에서는 `int()` 와 같다. (명세 2판의 개정 1)
    """
    return (math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys)))


def has_noninteger(xs, ys) -> bool:
    return any(isinstance(v, float) and not v.is_integer() for v in (*xs, *ys))


def classify_clip(v1_box, raw, geom_flags: str, w: int, h: int) -> str:
    """§2-3. v1 상자와 원좌표 상자로 `clip_kind` 를 낸다. 설명되지 않는 차이는 멈춘다."""
    if v1_box is None:
        return absorb.CLIP_EMPTY
    if v1_box == raw:
        return absorb.CLIP_NONE
    boundary = v1_box == absorb._clamp(raw, w, h)
    poly = "multipart_largest_kept" in (geom_flags or "")
    if boundary and poly:
        return absorb.CLIP_BOTH
    if boundary:
        return absorb.CLIP_BOUNDARY
    if poly:
        return absorb.CLIP_POLY
    raise RawMetaError("v1 상자와 원좌표 상자의 차이를 문서화된 정책으로 설명할 수 없다")


def fold_images(kinds_by_image: dict[str, list[str]]) -> Counter:
    """§2-3 장으로 접는 순서 — `empty_bbox` > `boundary` · `both` > `polygon_repair`."""
    out: Counter = Counter()
    for kinds in kinds_by_image.values():
        ks = set(kinds)
        if absorb.CLIP_EMPTY in ks:
            out[absorb.CLIP_EMPTY] += 1
        elif ks & {absorb.CLIP_BOUNDARY, absorb.CLIP_BOTH}:
            out[absorb.CLIP_BOUNDARY] += 1
        elif absorb.CLIP_POLY in ks:
            out[absorb.CLIP_POLY] += 1
    return out


def _v1_box(r) -> tuple[int, int, int, int] | None:
    return absorb._box(r)


def _guard_out(out_root: Path, protected: list[Path]) -> None:
    """§7 V-12. 아무것도 쓰기 전에 부른다."""
    out_root = Path(out_root)
    if out_root.name in PROTECTED_NAMES:
        raise RawMetaError(f"{out_root.name} 에는 쓰지 않는다 — 동결본 · 잠정판의 이름이다")
    norm = os.path.normcase(os.path.abspath(out_root))
    for p in protected:
        if p is not None and os.path.normcase(os.path.abspath(p)) == norm:
            raise RawMetaError("출력 경로가 입력 스냅샷과 같다")
    if (out_root / SNAPSHOT_FILENAME).exists():
        raise RawMetaError(f"{out_root} 에 잠긴 판이 있다 — 잠긴 판이 있는 경로에는 쓰지 않는다. 새 경로를 준다")
    assert_writable(out_root, what="출력 경로")


def build(v1_root: Path, label_root: Path, out_root: Path, label_map_path: Path,
          v2_root: Path | None = None) -> dict:
    """v1 을 옮기고 `iso_codes` 를 다시 만들고 원본 라벨에서 곁파일 셋을 내어 새 판을 쓴다.

    반환값은 회계다. **v1 · v2 · 원본 zip 은 한 바이트도 건드리지 않는다** — 끝에 다시 재어 확인한다.
    """
    v1_root, label_root, out_root = Path(v1_root), Path(label_root), Path(out_root)
    _guard_out(out_root, [v1_root, v2_root])
    v1_digest = verify_snapshot(v1_root)
    v2_digest = verify_snapshot(v2_root) if v2_root is not None and (Path(v2_root) / SNAPSHOT_FILENAME).exists() else None
    zips_before = zip_digests(label_root)

    snap = load_snapshot(v1_root, verify=True)
    lm = load_label_map(label_map_path)
    lm_raw = yaml.safe_load(Path(label_map_path).read_text(encoding="utf-8"))
    man = snap.manifest.copy()
    ann = snap.annotations.copy()
    tiles = snap.tiles
    if tiles is None:
        raise RawMetaError("v1 에 tiles.csv 가 없다 — 프레임 관계의 v1 일치 검사를 할 수 없다")
    prov = dict(zip(tiles["image_id"], tiles["provenance"]))

    read = read_label_zips(label_root)
    recs: list[LabelRec] = read["records"]
    # §4 규칙 8. info.id 가 겹치면 멈춘다.
    dup = [k for k, n in Counter(r.info_id for r in recs).items() if n > 1]
    if dup:
        raise RawMetaError(f"info.id 가 겹치는 장이 {len(dup)}개 있다 — 어느 것을 쓸지 정하지 않는다")
    by_id = {r.image_id: r for r in recs}

    split = dict(zip(man["image_id"], man["split"].fillna("")))
    eval_ids = {i for i, s in split.items() if s == "eval"}
    missing = [i for i in man["image_id"] if i not in by_id]
    if missing:
        raise RawMetaError(f"v1 의 {len(missing)}장이 원본 라벨에 없다 — 원천이 전 장을 덮지 않는다")

    ours: dict[str, list] = {}
    for r in ann.itertuples(index=False):
        ours.setdefault(str(r.image_id), []).append(r)
    for v in ours.values():
        v.sort(key=lambda r: int(str(r.ann_id).rsplit("#", 1)[-1]))

    img_rows, def_rows, reg_rows = [], [], []
    acct = Counter()
    clip_rows = Counter()
    kinds_by_image: dict[str, list[str]] = {}
    label_defect_v1_normal: Counter = Counter()
    regions_per_image: Counter = Counter()

    for row in man.itertuples(index=False):
        iid = str(row.image_id)
        rec = by_id[iid]
        is_eval = iid in eval_ids
        # V-9
        if rec.material != str(row.material) or rec.modality != "RT":
            raise RawMetaError(f"{iid}: 라벨의 재질 · 종류가 매니페스트와 다르다")
        same = (rec.width, rec.height) == (TILE_W, TILE_H)
        fm = absorb.MATCH_SAME if same else absorb.MATCH_TILE_OF_ORIG
        # V-10 — v1 일치 검사. 같은 라벨 크기에서 나온 값이라 독립 대조가 아니다.
        if (prov.get(iid) == "N-crop") != same:
            raise RawMetaError(f"{iid}: 라벨 크기의 프레임 관계가 v1 의 출처와 다르다")
        img_rows.append({"image_id": iid, "covered": True, "source_zip": rec.zip_name,
                         "material_source": absorb.MAT_LABEL_JSON,
                         "orig_width_px": rec.width, "orig_height_px": rec.height,
                         "frame_match": fm, "absorb_note": ""})

        # --- 결함 주석 — §4 규칙 1 · 2 · 4 · 5 · 6 · 7 ----------------------------------
        mine = ours.get(iid, [])
        produced = []
        if not rec.is_normal:
            if any(_norm_case(c) in OUT_OF_LABEL_SPACE for c in rec.poly_cases):
                raise RawMetaError(f"{iid}: 라벨 공간 밖 클래스가 있는데 v1 에 있다")
            seq = 0
            for (xs, ys), case in zip(rec.polys, rec.poly_cases, strict=True):
                try:
                    raw_label = _raw_label(case)      # 규칙 5 — 사상표 입력이 없으면 멈춘다. 건너뛰지 않는다
                except KeyError as exc:
                    raise RawMetaError(f"{iid}: 라벨 case 를 사상표 입력으로 바꿀 수 없다 — {exc}") from exc
                if raw_label is None:
                    continue
                l2 = lm.to_defect_type(SOURCE, raw_label)
                if l2 is None:
                    continue
                produced.append((f"{iid}#{seq}", xs, ys, raw_label, l2, lm.iso_code(l2)))
                seq += 1
        elif bool(row.has_defect):
            raise RawMetaError(f"{iid}: 라벨은 정상인데 v1 에 결함이 있다")
        if not rec.is_normal and not produced and not bool(row.has_defect):
            # 검토 m-2 (나) — 라벨은 결함인데 거른 뒤 v1 정상이 된 장. 분할별로 수만 센다.
            label_defect_v1_normal["eval" if is_eval else split[iid] or "?"] += 1

        # --- v1 일치 검사 — 순번과 내용이라는 서로 다른 두 키로 맞댄다 -----------------
        if len(produced) != len(mine):
            raise RawMetaError(f"{iid}: 결함 주석 수가 v1 과 다르다")
        if produced and fm != absorb.MATCH_SAME:
            raise RawMetaError(f"{iid}: 결함 주석이 원본 프레임 장에 있다")          # V-6
        kinds = []
        for (aid, xs, ys, raw_label, l2, iso), r in zip(produced, mine):
            if aid != str(r.ann_id):
                raise RawMetaError(f"{iid}: 주석 순번이 v1 과 다르다")
            if polygon_json(xs, ys) != str(r.polygon_json):
                raise RawMetaError(f"{iid}: 원본 다각형의 직렬화가 v1 polygon_json 과 바이트로 다르다")
            if (raw_label, l2, iso) != (str(r.src_label_raw), str(r.defect_type), str(r.iso_code)):
                raise RawMetaError(f"{iid}: 주석의 원본 라벨 · 종류 · ISO 가 v1 과 다르다")   # V-7
            v1_box = _v1_box(r)
            g = polygon_metrics(list(zip(xs, ys, strict=True)), rec.width, rec.height)
            if (tuple(g.bbox_px) if g.bbox_px else None) != v1_box:
                raise RawMetaError(f"{iid}: 원본 다각형으로 다시 낸 상자가 v1 상자와 다르다")
            raw = raw_box(xs, ys)
            if has_noninteger(xs, ys):
                acct["noninteger_rows_eval" if is_eval else "noninteger_rows_trainval"] += 1
            kind = classify_clip(v1_box, raw, str(r.geom_flags or ""), int(row.width_px), int(row.height_px))
            kinds.append(kind)
            def_rows.append({"ann_id": aid, "image_id": iid,
                             "raw_bbox_x1_px": raw[0], "raw_bbox_y1_px": raw[1],
                             "raw_bbox_x2_px": raw[2], "raw_bbox_y2_px": raw[3],
                             "clip_applied": kind != absorb.CLIP_NONE, "clip_kind": kind,
                             "raw_source": absorb.RAW_LABEL})
            if not is_eval:
                clip_rows[kind] += 1
        if kinds and not is_eval:
            kinds_by_image[iid] = kinds

        # --- 정상영역 — 원본 프레임 행만(§2-2 · §4 규칙 9) ------------------------------
        if rec.is_normal:
            for k, (xs, ys) in enumerate(rec.polys):
                pj = polygon_json(xs, ys)
                reg_rows.append({"region_id": f"{iid}#r{k}", "image_id": iid, "polygon_json": pj,
                                 "n_vertices": len(xs), "frame": absorb.FRAME_ORIG,
                                 "usable_in_tile_frame": False})
            if not is_eval:
                regions_per_image[len(rec.polys)] += 1
        acct["eval_images" if is_eval else "trainval_images"] += 1

    if len(def_rows) != len(ann):
        raise RawMetaError(f"결함 곁파일 {len(def_rows)}행 ≠ 주석 {len(ann)}행")

    # 원본에 있으나 v1 에 없는 장 — 사유별로 수만 센다
    v1_ids = set(man["image_id"])
    not_in_v1 = Counter()
    for r in recs:
        if r.image_id in v1_ids:
            continue
        oos = (not r.is_normal) and any(_norm_case(c) in OUT_OF_LABEL_SPACE for c in r.poly_cases)
        not_in_v1["out_of_label_space" if oos else "other_ingest_or_tiling_discard"] += 1

    # §8-1 과 같다 — iso_codes 를 계약 사상표로 다시 만든다
    new_iso = [absorb._iso_of(t, lm_raw) for t in man["defect_types"].fillna("")]
    changed = int(sum(1 for a, b in zip(man["iso_codes"].fillna(""), new_iso) if a != b))
    man["iso_codes"] = pd.Series(new_iso, index=man.index, dtype="string")
    changed_cols = [c for c in man.columns if not man[c].equals(snap.manifest[c])]

    covered = len(img_rows)
    absorption: dict[str, Any] = {
        "source": absorb.SOURCE_RAW_LABELS,
        "status": "complete" if covered == len(man) else "provisional",
        "join_key": "info.id",
        "covered_images": covered,
        "total_images": len(man),
        "label_archives": [{"name": n, "sha256": zips_before[n], "json_entries": read["json_entries"][n]}
                           for n in sorted(zips_before)],
        "parse_failures": read["parse_failures"],
        "non_rt_records": int(read["non_rt"]),
        "labels_not_in_v1": dict(sorted(not_in_v1.items())),
        "평가_행": "§5-3 — 값의 분포는 학습 · 검증 행에서만 센다. 평가 행은 장 수만 적는다",
        "eval_images": int(acct["eval_images"]),
        "trainval_images": int(acct["trainval_images"]),
        "trainval": {
            "frame_match": dict(sorted(Counter(r["frame_match"] for r in img_rows
                                               if r["image_id"] not in eval_ids).items())),
            "defect_rows": int(sum(clip_rows.values())),
            "clip_kind_rows": dict(sorted(clip_rows.items())),
            "clip_kind_images": dict(sorted(fold_images(kinds_by_image).items())),
            "region_rows": int(sum(1 for r in reg_rows if r["image_id"] not in eval_ids)),
            "regions_per_image": {int(k): int(v) for k, v in sorted(regions_per_image.items())},
            "label_defect_v1_normal": int(sum(v for k, v in label_defect_v1_normal.items() if k != "eval")),
            "noninteger_coordinate_rows": int(acct["noninteger_rows_trainval"]),
        },
        "원좌표_정수화": "정수가 아닌 좌표는 바깥으로(최솟값 내림 · 최댓값 올림) — v1 상자 계산과 같은 방향. "
                     "polygon_json 은 v1 과 같이 int() 다(그 장들의 부분 화소는 v1 에서 이미 사라졌다)",
        "label_defect_v1_normal_eval_images": int(label_defect_v1_normal.get("eval", 0)),
        "v1_consistency": "주석마다 순번 · polygon_json 바이트 · 원본 라벨 · 종류 · ISO · 다시 낸 상자가 v1 과 같다 — 전 행",
        "iso_pair_fixed": changed,
        "changed_columns_vs_v1": changed_cols,
        "key_paths": dict(sorted(read["key_paths"].items())),
        "scale_key_hits": dict(sorted(read["scale_key_hits"].items())),
        "scale_key_dictionary": {"key_parts": list(SCALE_KEY_PARTS), "value_pattern": SCALE_VALUE.pattern},
        "case_to_raw": dict(CASE_TO_RAW),
        "out_of_label_space": sorted(OUT_OF_LABEL_SPACE),
        "digest_history": [],
    }

    # V-8 — 읽는 동안 원본 zip 이 바뀌지 않았다. 바뀌었으면 아무것도 쓰지 않는다.
    if zip_digests(label_root) != zips_before:
        raise RawMetaError("원본 라벨 zip 의 sha256 이 빌드 중에 바뀌었다 — 쓰지 않았다")

    caps = dict(snap.capabilities)
    caps["snapshot_id"] = SNAPSHOT_ID
    caps["absorption"] = absorption
    digest = write_snapshot(
        out_root, man, ann, caps, tiles=tiles,
        absorb_image=pd.DataFrame(img_rows, columns=list(ABSORB_IMAGE_COLUMNS)),
        absorb_defect=pd.DataFrame(def_rows, columns=list(ABSORB_DEFECT_COLUMNS)),
        absorb_region=pd.DataFrame(reg_rows, columns=list(ABSORB_REGION_COLUMNS)),
    )

    # --- 끝 검사 — V-12 · 읽는 쪽 검사 ------------------------------------------------
    if verify_snapshot(v1_root) != v1_digest:
        raise RawMetaError("v1 의 지문이 빌드 중에 바뀌었다")
    if v2_digest is not None and verify_snapshot(v2_root) != v2_digest:
        raise RawMetaError("v2 의 지문이 빌드 중에 바뀌었다")
    absorb.load_absorbed(out_root)
    absorption["snapshot_digest"] = digest
    absorption["v1_digest"] = v1_digest
    absorption["v2_digest"] = v2_digest
    return absorption
