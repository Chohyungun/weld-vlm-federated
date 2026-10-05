"""정리된 이미지 메타데이터를 새 스냅샷에 얹는다 — 흡수 미니스펙 구현.

절 번호는 흡수 미니스펙의 것이다.

  §1-3  두 핵심 파일의 컬럼을 늘리지 않고 **곁파일 셋**으로 얹는다
  §2    새 경로 `data/interim/manifest_v2_absorbed`. 동결본은 읽기만 한다
  §3    곁파일의 컬럼·자료형·결측 표시. 결함 곁파일은 **모든 주석에 한 행**이고 안 덮은 주석은 값이 빈다
  §3-4  🔴 정상영역 폴리곤의 **좌표 프레임**을 반드시 적는다. 원본 프레임 행도 버리지 않는다.
        결함 곁파일은 전량 타일 프레임이다 — 관측이 아니라 **불변식**으로 막는다
  §4    조인 키는 `orig_info_id` ↔ `image_id` 의 숫자부. **파일 이름으로 잇지 않는다**
  §5    덮임 회계를 `data_capabilities.yaml` 의 `absorption` 블록에 넣는다
  §6-2  학습 재료로 나가는 경로를 **하나만** 두고, 그 안에서 스냅샷을 검증하고 평가셋을 뗀다
  §7-2  학습 산출은 **허용 목록의 칸만** 낸다. 원본 크기와 그 요약 비트, 출처·덮임 칸이 나가지 않는다
  §8-1  `iso_codes` 를 v1 에서 옮기지 않고 계약 사상표로 **다시 만든다**
  §8-2  빈 상자 85행은 표시 그대로 두고 원좌표는 곁파일로 **복구 가능하게** 담는다
  §9-1  `status` 는 사람이 적지 않고 회계에서 유도한다
  §9-2  같은 경로에 다시 낼 때 앞 판의 지문을 `digest_history` 에 남긴다.
        잠정판을 읽으면 로그에 남긴다

**읽기만 하는 것** — 동결본, 정리된 파일(외부 드라이브), 원본 아카이브 이름. 어느 것도 고치지 않는다.
정리된 파일은 한 줄씩 흘려 읽는다. 병렬을 쓰지 않는다.

**곁파일을 읽는 자리는 이 모듈 하나다.** `load_absorbed` 가 잠금 소속과 값 공간을 검사하고 잠정판을
경고한다. 다른 모듈이 `Snapshot.absorb_*` 를 직접 읽지 않는다는 것은 시험이 고정한다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data.manifest_io import (ABSORB_DEFECT_COLUMNS, ABSORB_DEFECT_FILENAME,
                              ABSORB_IMAGE_COLUMNS, ABSORB_IMAGE_FILENAME,
                              ABSORB_REGION_COLUMNS, ABSORB_REGION_FILENAME,
                              CAPABILITIES_FILENAME, SNAPSHOT_FILENAME, Snapshot,
                              load_snapshot, read_capabilities, verify_snapshot,
                              write_snapshot)

_log = logging.getLogger(__name__)

DIGITS = re.compile(r"(\d+)")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
#: §3-4 4번. 크기가 같으면 타일 프레임, 다르면 원본 프레임이다.
FRAME_TILE, FRAME_ORIG = "tile", "orig"
FRAME_VALUES = frozenset({FRAME_TILE, FRAME_ORIG})
#: §3-2 5번. 값 공간을 폐집합으로 둔다.
CLIP_NONE, CLIP_BOUNDARY, CLIP_POLY, CLIP_BOTH = "none", "boundary", "polygon_repair", "both"
#: 원본 라벨 원천에서만 쓰는 값 — 우리 상자가 빈 행. `none` 에 두 뜻을 싣지 않는다.
CLIP_EMPTY = "empty_bbox"
CLIP_VALUES = frozenset({CLIP_NONE, CLIP_BOUNDARY, CLIP_POLY, CLIP_BOTH, CLIP_EMPTY})
#: §3-1 7번 · §3-2 6번.
MATCH_SAME, MATCH_TILE_OF_ORIG = "same", "tile_of_orig"
FRAME_MATCH_VALUES = frozenset({MATCH_SAME, MATCH_TILE_OF_ORIG, ""})
RAW_ABSORBED = "absorbed"
RAW_LABEL = "raw_label"
RAW_SOURCE_VALUES = frozenset({RAW_ABSORBED, RAW_LABEL, ""})
MAT_FOLDER_NOMINAL, MAT_LABEL_JSON = "folder_nominal", "label_json"
#: 곁파일의 원천. `absorption.source` 가 없으면 정리된 파일을 흡수한 판이다.
#: 같은 칸 이름이 원천마다 다른 값 공간을 쓴다 — **값으로 원천이 갈린다.** 한 판에 두 원천의 값이 섞이면 멈춘다.
SOURCE_TEAM_FILE, SOURCE_RAW_LABELS = "team_file", "raw_labels"
SOURCE_VALUES = frozenset({SOURCE_TEAM_FILE, SOURCE_RAW_LABELS})
#: 원천마다 쓰는 값. (원좌표 출처, 재질 근거의 허용 값, `clip_kind` 의 허용 값)
_SOURCE_RULES = {
    SOURCE_TEAM_FILE: (RAW_ABSORBED, frozenset({MAT_FOLDER_NOMINAL, ""}),
                       frozenset({CLIP_NONE, CLIP_BOUNDARY, CLIP_POLY, CLIP_BOTH})),
    SOURCE_RAW_LABELS: (RAW_LABEL, frozenset({MAT_LABEL_JSON}), CLIP_VALUES),
}
#: 곁파일 셋 — (`Snapshot` 의 속성 이름, 파일 이름).
SIDECARS: tuple[tuple[str, str], ...] = (
    ("absorb_image", ABSORB_IMAGE_FILENAME),
    ("absorb_defect", ABSORB_DEFECT_FILENAME),
    ("absorb_region", ABSORB_REGION_FILENAME),
)
_BOX = ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")
_RAW = ("raw_bbox_x1_px", "raw_bbox_y1_px", "raw_bbox_x2_px", "raw_bbox_y2_px")
_HISTORY_KEYS = ("status", "digest", "at", "archives_present")


class AbsorptionError(RuntimeError):
    """흡수를 진행할 수 없다. 값을 내지 않고 멈춘다."""


def num_key(image_id: str) -> str:
    """우리 `image_id` 의 숫자부. §4-1 — 파일 이름을 파싱해 잇지 않는다."""
    m = DIGITS.findall(image_id or "")
    return m[-1] if m else ""


def _clamp(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    """프레임 안으로 자른다. **상한은 W·H 다. W-1·H-1 이 아니다.**

    동결본에 폭 1280 영상의 `bbox_x2_px = 1280` 인 행이 있다. 실물로 확인한 값이다.
    """
    x1, y1, x2, y2 = box
    return (min(max(x1, 0), w), min(max(y1, 0), h), min(max(x2, 0), w), min(max(y2, 0), h))


def _box(r) -> tuple[int, int, int, int] | None:
    """우리 주석 행의 상자. 네 칸이 비어 있으면 None 이다(§8-2 의 빈 상자)."""
    try:
        return (int(r.bbox_x1_px), int(r.bbox_y1_px), int(r.bbox_x2_px), int(r.bbox_y2_px))
    except (TypeError, ValueError):
        return None


def _iso_of(defect_types: str, label_map: dict) -> str:
    """§8-1. `defect_types` 의 **순서대로** 계약 사상표에서 코드를 찾아 다시 만든다.

    v1 의 `iso_codes` 를 옮기지 않는다 — 그 열은 두 컬럼이 따로 정렬돼 짝이 풀려 있다.
    파생값을 옮기지 않고 다시 만드는 것이 원칙에 맞다.
    """
    types = [t for t in (defect_types or "").split(";") if t]
    codes = []
    for t in types:
        spec = label_map["defect_types"].get(t)
        if spec is None:
            raise AbsorptionError(f"계약 사상표에 없는 결함 종류: {t!r}")
        codes.append(str(spec["iso_code"]))
    return ";".join(codes)


def stream_jsonl(path: Path):
    """한 줄씩 흘려 읽는다. 파일 전체를 메모리에 올리지 않는다."""
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise AbsorptionError(f"{n}번째 줄을 JSON 으로 읽지 못했다") from exc


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_archive_names(dataset_root: Path) -> list[str]:
    """원본 폴더의 라벨 아카이브 **이름만** 센다. 열지도 풀지도 않는다.

    0건이면 멈춘다 — 빈 목록이 "빠진 것 없음" 으로 읽히는 것을 막는다.
    """
    names = sorted(p.name for p in dataset_root.rglob("*.zip") if "02.라벨링데이터" in str(p))
    if not names:
        raise AbsorptionError(f"라벨 아카이브를 하나도 찾지 못했다: {dataset_root}")
    return names


def _locked_members(root: Path) -> set[str]:
    """`SNAPSHOT.sha256` 에 잠긴 파일 이름. 형식은 `verify_snapshot` 이 읽는 것과 같다."""
    names = set()
    for line in (root / SNAPSHOT_FILENAME).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        names.add(line.split(maxsplit=1)[1].strip())
    return names


def _history_entry_ok(e: dict) -> bool:
    return (isinstance(e, dict) and all(k in e for k in _HISTORY_KEYS)
            and isinstance(e["digest"], str) and bool(HEX64.match(e["digest"])))


def _carry_history(out_root: Path, seed: list[dict] | None) -> list[dict]:
    """§9-2. 같은 경로에 이미 있는 판의 지문을 **덮기 전에** 기록으로 옮긴다.

    항목은 `{status, digest, at, archives_present}` 다. 잠정판 결과를 인용한 문서가 어느 지문을
    봤는지 나중에 가릴 수 있어야 한다.

    - 이미 있는 판은 **검증한 뒤에만** 기록한다. 검증되지 않는 판을 덮으면 무엇을 덮었는지 모르게 된다.
    - `seed` 는 이 장치가 생기기 전에 사라져 디스크에서 찾을 수 없는 판이다. 부르는 쪽이 출처와 함께
      준다. 디스크에 있는 어떤 판보다 앞선 것이므로 앞에 둔다.
    """
    hist: list[dict] = []
    prev: dict | None = None
    lock = out_root / SNAPSHOT_FILENAME
    if lock.exists():
        digest = verify_snapshot(out_root)
        ab = (read_capabilities(out_root / CAPABILITIES_FILENAME) or {}).get("absorption") or {}
        hist = [dict(e) for e in ab.get("digest_history") or []]
        at = datetime.fromtimestamp(lock.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        prev = {"status": ab.get("status"), "digest": digest, "at": at,
                "archives_present": (ab.get("archives") or {}).get("present")}
    known = {e.get("digest") for e in hist} | ({prev["digest"]} if prev else set())
    lead = [dict(e) for e in (seed or []) if e.get("digest") not in known]
    hist = lead + hist
    if prev is not None and prev["digest"] not in {e.get("digest") for e in hist}:
        hist.append(prev)
    bad = [e for e in hist if not _history_entry_ok(e)]
    if bad:
        raise AbsorptionError(f"digest_history 항목의 형식이 다르다 — 필요한 칸 {_HISTORY_KEYS}: {bad[:2]}")
    return hist


def build(v1_root: Path, jsonl: Path, out_root: Path, label_map_path: Path,
          dataset_root: Path | None = None, check=None,
          seed_history: list[dict] | None = None) -> dict:
    """v1 을 옮기고 파생값 하나를 고치고 곁파일을 더해 v2 를 낸다 (§8-3).

    반환값은 회계다. **v1 은 한 바이트도 건드리지 않는다.**

    `check` 는 회계를 받아 어긋난 자리 목록을 내는 함수다. 하나라도 있으면
    **쓰기 전에** 멈춘다 — 낸 뒤에 걸러도 이미 인용할 수 있는 파일이 남는다.
    기대값은 이 모듈이 갖지 않는다. 부르는 쪽이 준다.

    `seed_history` 는 `_carry_history` 의 `seed` 다.
    """
    out_root = Path(out_root)
    snap = load_snapshot(v1_root, verify=True)          # 지문 검증을 건너뛰지 않는다
    label_map = yaml.safe_load(Path(label_map_path).read_text(encoding="utf-8"))
    man = snap.manifest.copy()
    ann = snap.annotations.copy()

    by_num: dict[str, str] = {}
    dup_num = 0
    for iid in man["image_id"]:
        k = num_key(iid)
        if k in by_num:
            dup_num += 1
        by_num.setdefault(k, iid)

    meta = {iid: (str(r.width_px), str(r.height_px), bool(r.has_defect))
            for iid, r in zip(man["image_id"], man.itertuples(index=False))}

    # §5. 결측 참여자를 `eval` 칸에 모으는 것은 **이름 붙이기가 아니라 확인**이어야 한다.
    # 참여자가 빈 장과 평가셋이 같은 집합일 때만 그 이름이 맞다.
    no_client = set(man.loc[man["client"].fillna("") == "", "image_id"])
    eval_split = set(man.loc[man["split"] == "eval", "image_id"])
    if no_client != eval_split:
        raise AbsorptionError(
            f"참여자가 빈 장 {len(no_client)} 과 평가셋 {len(eval_split)} 이 같은 집합이 아니다 "
            f"(차 {len(no_client ^ eval_split)}). 결측 칸에 `eval` 이라는 이름을 붙일 수 없다")

    # --- 정리된 파일을 흘려 읽으며 겹치는 장만 담는다 --------------------------
    cov: dict[str, dict] = {}
    only_j = Counter()
    lines = 0
    idx_gap = 0
    zips_src: set[str] = set()                 # 정리된 파일 **전체 줄**에 나타난 아카이브
    zips_src_mod: dict[str, set[str]] = defaultdict(set)
    for d in stream_jsonl(jsonl):
        lines += 1
        z0 = (d.get("source_zip") or "").strip()
        if z0:
            zips_src.add(z0)
            zips_src_mod[str(d.get("modality", ""))].add(z0)
        oid = str(d.get("orig_info_id", "")).strip()
        iid = by_num.get(oid)
        if iid is None:
            only_j[d.get("modality", "")] += 1
            continue
        jw, jh = str(d.get("width_px", "")), str(d.get("height_px", ""))
        anns = d.get("annotations") or []
        defects = sorted((a for a in anns if a.get("annotation_role") == "defect"),
                         key=lambda a: a.get("ann_idx", 0))
        if defects and [a.get("ann_idx") for a in defects] != list(range(len(defects))):
            idx_gap += 1
        regions = [a for a in anns if a.get("annotation_role") == "normal_region"]
        cov[iid] = {"zip": d.get("source_zip", "") or "", "mat_src": d.get("material_source", "") or "",
                    "jw": jw, "jh": jh, "defects": defects, "regions": regions}

    # --- absorb_image (§3-1) — 62,308행. 덮지 않은 장도 행은 있고 값이 빈다 -----
    img_rows = []
    frame_match = Counter()
    for iid in man["image_id"]:
        c = cov.get(iid)
        if c is None:
            img_rows.append({"image_id": iid, "covered": False, "source_zip": "",
                             "material_source": "", "orig_width_px": pd.NA,
                             "orig_height_px": pd.NA, "frame_match": "", "absorb_note": ""})
            continue
        same = (c["jw"], c["jh"]) == meta[iid][:2]
        fm = MATCH_SAME if same else MATCH_TILE_OF_ORIG
        frame_match[fm] += 1
        img_rows.append({"image_id": iid, "covered": True, "source_zip": c["zip"],
                         "material_source": c["mat_src"],
                         "orig_width_px": int(c["jw"]), "orig_height_px": int(c["jh"]),
                         "frame_match": fm, "absorb_note": ""})

    # --- absorb_defect (§3-2) — **모든 주석에 한 행.** 안 덮은 주석은 값이 빈다 --------
    # 짝짓기 규칙: 우리 `ann_id` 꼬리 순서 ↔ 정리된 파일 `ann_idx` 순서. 명세 §4 는 장 단위 결합만
    # 정했으므로 주석 단위 짝은 여기서 정한다. **개수가 같아야 하고, 짝마다 ISO 코드가 같아야 한다.**
    # 위치로만 짝지으면 우리 상자가 빈 행은 비교 없이 짝지어진다 — 코드 대조가 그 행까지 본다.
    ours: dict[str, list] = defaultdict(list)
    for r in ann.itertuples(index=False):
        ours[str(r.image_id)].append(r)
    for v in ours.values():
        v.sort(key=lambda r: int(str(r.ann_id).rsplit("#", 1)[-1]))

    def_rows = []
    clip_kinds = Counter()
    #: 장 단위 회계. 앞선 두 검산의 53·39·9 는 **장** 을 센 수이고 곁파일은 **행** 을 센다.
    #: 두 그레인을 함께 적지 않으면 받는 쪽이 106 과 101 을 같은 것으로 견준다.
    per_img_clip: dict[str, Counter] = defaultdict(Counter)
    recovered = 0
    missing_bbox_rows = 0
    pairs_checked = 0
    absorbed_rows = 0

    def _empty_row(r, iid):
        return {"ann_id": str(r.ann_id), "image_id": iid,
                "raw_bbox_x1_px": pd.NA, "raw_bbox_y1_px": pd.NA,
                "raw_bbox_x2_px": pd.NA, "raw_bbox_y2_px": pd.NA,
                "clip_applied": False, "clip_kind": CLIP_NONE, "raw_source": ""}

    for iid in man["image_id"]:
        rows = ours.get(iid, [])
        missing_bbox_rows += sum(1 for r in rows if _box(r) is None)
        c = cov.get(iid)
        if c is None:
            def_rows.extend(_empty_row(r, iid) for r in rows)
            continue
        ds = c["defects"]
        if len(rows) != len(ds):
            raise AbsorptionError(
                f"{iid}: 결함 주석 수가 다르다 — 우리 {len(rows)} · 정리된 파일 {len(ds)}. 짝지을 수 없다")
        if not rows:
            continue
        # §3-4 · T-4 ③. 결함 주석은 **크기가 같은 장**(타일 프레임)에만 있어야 한다. 지금 덮인 자료에서
        # 참인 것은 관측이고, 빠진 결함 아카이브가 채워지면 처음 시험받는다. 관측이 아니라 단언으로 둔다.
        if (c["jw"], c["jh"]) != meta[iid][:2]:
            raise AbsorptionError(
                f"{iid}: 결함 주석이 원본 프레임 장에 있다 — 정리된 파일 {c['jw']}×{c['jh']} · "
                f"우리 {meta[iid][0]}×{meta[iid][1]}. 원좌표를 타일 좌표로 옮길 수 없다")
        w, h = int(meta[iid][0]), int(meta[iid][1])
        for r, a in zip(rows, ds):
            if str(r.iso_code) != str(a.get("class_std_iso6520", "")):
                raise AbsorptionError(
                    f"{iid}: 짝지은 주석의 ISO 코드가 다르다 — 우리 {r.iso_code!r} · "
                    f"정리된 파일 {a.get('class_std_iso6520')!r}. 위치 순서 짝짓기가 틀렸다")
            pairs_checked += 1
            absorbed_rows += 1
            raw = (int(a["bbox_xmin"]), int(a["bbox_ymin"]), int(a["bbox_xmax"]), int(a["bbox_ymax"]))
            mine = _box(r)
            if mine is None:
                # §8-2. 우리 상자가 빈 행이다. 행을 지우지 않고 원좌표만 복구 가능하게 담는다.
                # 값 공간(§3-2 5번)에 이 경우의 값이 없어 `none` 으로 둔다. `none` 의 뜻은 둘이다 —
                # "우리 상자와 원좌표가 같다" 또는 "우리 상자가 비어 `geom_valid=False` 다".
                # 그 밖의 `none` 은 `load_absorbed` 가 막는다.
                recovered += 1
                per_img_clip[iid]["empty"] += 1
                def_rows.append({"ann_id": str(r.ann_id), "image_id": iid,
                                 "raw_bbox_x1_px": raw[0], "raw_bbox_y1_px": raw[1],
                                 "raw_bbox_x2_px": raw[2], "raw_bbox_y2_px": raw[3],
                                 "clip_applied": False, "clip_kind": CLIP_NONE,
                                 "raw_source": RAW_ABSORBED})
                continue
            boundary = mine == _clamp(raw, w, h) and mine != raw
            poly = "multipart_largest_kept" in str(r.geom_flags or "") and mine != raw
            kind = (CLIP_BOTH if boundary and poly else CLIP_BOUNDARY if boundary
                    else CLIP_POLY if poly else CLIP_NONE)
            if mine != raw and kind == CLIP_NONE:
                # 값 공간이 폐집합이므로 설명되지 않는 차이를 `none` 으로 적으면 **사라진다.**
                # 명세는 상자 차이 전건이 설명된다고 단언한다. 단언이 깨지면 값을 내지 않고 멈춘다.
                raise AbsorptionError(
                    f"{iid} 의 상자 차이를 문서화된 정책으로 설명할 수 없다 (ann_id 는 산출에 있다)")
            clip_kinds[kind] += 1
            if kind != CLIP_NONE:
                per_img_clip[iid][kind] += 1
            def_rows.append({"ann_id": str(r.ann_id), "image_id": iid,
                             "raw_bbox_x1_px": raw[0], "raw_bbox_y1_px": raw[1],
                             "raw_bbox_x2_px": raw[2], "raw_bbox_y2_px": raw[3],
                             "clip_applied": mine != raw, "clip_kind": kind,
                             "raw_source": RAW_ABSORBED})
    if len(def_rows) != len(ann):
        raise AbsorptionError(f"결함 곁파일 {len(def_rows)}행 ≠ 주석 {len(ann)}행 — 모든 주석에 한 행이어야 한다")

    # --- absorb_region (§3-3·§3-4) — 정상 이미지만. 프레임을 반드시 적는다 -------
    reg_rows = []
    frames = Counter()
    frame_imgs = Counter()         # 같은 그레인 문제. 명세 §3-4 표의 5,885·1,379 는 **장수** 다
    n_regions_per_img = Counter()
    regions_on_defect = 0          # 결함 장의 정상영역은 담지 않는다(§3-3 2번). 버린 수를 센다
    for iid, c in cov.items():
        if meta[iid][2]:           # has_defect 면 정상영역을 담지 않는다 (§1-3)
            regions_on_defect += len(c["regions"])
            continue
        same = (c["jw"], c["jh"]) == meta[iid][:2]
        frame = FRAME_TILE if same else FRAME_ORIG
        frame_imgs[frame] += 1
        n_regions_per_img[len(c["regions"])] += 1
        for n, a in enumerate(c["regions"]):
            try:
                px = json.loads(a["polygon_x"]) if isinstance(a.get("polygon_x"), str) else a.get("polygon_x")
                py = json.loads(a["polygon_y"]) if isinstance(a.get("polygon_y"), str) else a.get("polygon_y")
            except json.JSONDecodeError as exc:
                raise AbsorptionError(f"{iid} 의 정상영역 폴리곤을 읽지 못했다") from exc
            poly = [[int(x), int(y)] for x, y in zip(px or [], py or [])]
            frames[frame] += 1
            reg_rows.append({"region_id": f"{iid}#r{n}", "image_id": iid,
                             "polygon_json": json.dumps(poly, separators=(",", ":")),
                             "n_vertices": len(poly), "frame": frame,
                             "usable_in_tile_frame": frame == FRAME_TILE})

    # --- §8-1. iso_codes 를 다시 만든다. v2 매니페스트가 v1 과 다른 **유일한** 자리 ---
    new_iso = [_iso_of(t, label_map) for t in man["defect_types"].fillna("")]
    changed = int(sum(1 for a, b in zip(man["iso_codes"].fillna(""), new_iso) if a != b))
    man["iso_codes"] = pd.Series(new_iso, index=man.index, dtype="string")
    # 명세 10-2 R-5. v1 과 다른 열을 **세어** 적는다. 쓰기 전 검사가 이 값을 본다.
    changed_cols = [c for c in man.columns if not man[c].equals(snap.manifest[c])]

    # --- §5. 회계 --------------------------------------------------------------
    per_zip = Counter(c["zip"] for c in cov.values())
    seen_zip = {z for z in per_zip if z}
    archives = None
    if dataset_root is not None:
        names = label_archive_names(Path(dataset_root))
        # **모수가 둘이다.** 명세 §5 의 `present: 21` 은 정리된 파일 전체 줄에 나타난 수이고,
        # 덮인 장에만 나타난 수는 14 다(VT 는 우리 동결본에 한 장도 없다). 이름 하나에
        # 둘을 담으면 받는 쪽이 21 과 14 중 무엇을 본 것인지 알 수 없다.
        miss_src = [n for n in names if n not in zips_src]
        miss_cov = [n for n in names if n not in seen_zip]
        archives = {
            "total_label_archives": len(names),
            "present": len(zips_src), "missing_rt": sum(1 for n in miss_src if "RT" in n),
            "missing_vt": sum(1 for n in miss_src if "VT" in n), "missing": miss_src,
            "present_모수": "정리된 파일 전체 줄",
            "present_in_source_by_modality": {k: len(v) for k, v in sorted(zips_src_mod.items())},
            "present_in_covered": len(seen_zip),
            "missing_from_covered": {"rt": sum(1 for n in miss_cov if "RT" in n),
                                     "vt": sum(1 for n in miss_cov if "VT" in n)},
            "per_archive": {k: int(v) for k, v in sorted(per_zip.items()) if k},
        }

    def by(field, na_key=None):
        """층별 포함률. 결측을 어느 칸에 모을지 **부르는 쪽이 정한다.**

        `string` dtype 의 결측은 `pd.NA` 이고 `str(pd.NA)` 는 `"<NA>"` 로 참이다.
        `or` 로 걸러지지 않으므로 결측을 명시적으로 본다. `na_key` 를 주지 않은 축에
        결측이 있으면 그 자리를 지어 넣지 않고 멈춘다.
        """
        tot, hit = Counter(), Counter()
        for iid, r in zip(man["image_id"], man.itertuples(index=False)):
            v = getattr(r, field)
            if pd.isna(v) or str(v) == "":
                if na_key is None:
                    raise AbsorptionError(f"{field} 열에 결측이 있으나 모을 칸이 정해지지 않았다")
                k = na_key
            else:
                k = str(v)
            tot[k] += 1
            if iid in cov:
                hit[k] += 1
        return {k: {"covered": int(hit[k]), "total": int(tot[k])} for k in sorted(tot)}

    # 장 단위로 굴려 올린다. 분기 순서는 앞선 두 검산의 탐침과 같다 — 개수 → 경계 → 폴리곤.
    clip_by_image: dict[str, int] = Counter()
    for c in per_img_clip.values():
        if c["empty"]:
            clip_by_image["count_differs"] += 1
        elif c[CLIP_BOUNDARY] or c[CLIP_BOTH]:
            clip_by_image[CLIP_BOUNDARY] += 1
        elif c[CLIP_POLY]:
            clip_by_image[CLIP_POLY] += 1
    clip_by_image = {k: int(v) for k, v in sorted(clip_by_image.items())}

    absorption: dict[str, Any] = {
        "status": "provisional",
        "source_digest": file_digest(Path(jsonl)),
        "source_lines": lines,
        "join_key": "orig_info_id",
        "covered_images": len(cov),
        "total_images": int(len(man)),
        "not_inserted": {"vt": int(only_j.get("VT", 0)), "rt_missing_from_v1": int(only_j.get("RT", 0))},
        "coverage_by_client": by("client", na_key="eval"),
        "coverage_by_client_eval_칸": "참여자가 빈 장과 평가셋이 같은 집합임을 확인한 뒤 붙인 이름",
        "coverage_by_material": by("material"),
        "region_frames": {FRAME_TILE: int(frames[FRAME_TILE]), FRAME_ORIG: int(frames[FRAME_ORIG])},
        "region_frames_그레인": "정상영역 1개 = 1건",
        "region_images_by_frame": {FRAME_TILE: int(frame_imgs[FRAME_TILE]),
                                   FRAME_ORIG: int(frame_imgs[FRAME_ORIG])},
        "regions_per_image": {int(k): int(v) for k, v in sorted(n_regions_per_img.items())},
        "regions_not_stored_on_defect_images": int(regions_on_defect),
        "frame_match": {k: int(v) for k, v in sorted(frame_match.items())},
        "defect_rows": {"total": int(len(def_rows)), "absorbed": int(absorbed_rows),
                        "not_covered": int(len(def_rows) - absorbed_rows)},
        "defect_pairs_iso_checked": int(pairs_checked),
        "clip_kinds": {k: int(v) for k, v in sorted(clip_kinds.items()) if k != CLIP_NONE},
        "clip_kinds_그레인": "결함 주석 1건 = 1건",
        "clip_by_image": clip_by_image,
        "clip_by_image_그레인": "이미지 1장 = 1건. 명세 §3-2 의 53·39·9 가 이 그레인이다",
        "iso_pair_fixed": changed,
        "changed_columns_vs_v1": changed_cols,
        # §8-2 5번의 두 이름. 85 는 동결본 전체의 빈 상자 행, 54 는 그중 곁파일로 원좌표를 되찾은 행이다.
        "missing_bbox_rows": int(missing_bbox_rows),
        "recovered_from_absorb": int(recovered),
        "missing_bbox_images": int(sum(1 for c in per_img_clip.values() if c["empty"])),
        "missing_bbox_images_그레인": "빈 상자 행이 있는 덮인 장의 수. 명세 §5 예시 블록의 53 이 이 값이다",
        "duplicate_numeric_keys": dup_num,
        "ann_idx_not_contiguous": idx_gap,
    }
    if archives is not None:
        absorption["archives"] = archives
        # §9-1. 사람이 적지 않고 회계에서 유도한다.
        absorption["status"] = "complete" if archives["missing_rt"] == 0 else "provisional"

    # §9-2. 앞 판을 덮기 전에 그 지문을 옮긴다. 쓰기 전 검사보다 먼저 한다 — 검증되지 않는 앞 판이면
    # 여기서 멈추고 아무것도 쓰지 않는다.
    absorption["digest_history"] = _carry_history(out_root, seed_history)

    if check is not None:
        bad = check(absorption)
        if bad:
            raise AbsorptionError("회계가 기대와 다르다. 쓰지 않았다:\n  - " + "\n  - ".join(bad))

    caps = dict(snap.capabilities)
    caps["snapshot_id"] = "aihub71761_rt_v2_absorbed"
    caps["absorption"] = absorption

    digest = write_snapshot(
        out_root, man, ann, caps, tiles=snap.tiles,
        absorb_image=pd.DataFrame(img_rows, columns=list(ABSORB_IMAGE_COLUMNS)),
        absorb_defect=pd.DataFrame(def_rows, columns=list(ABSORB_DEFECT_COLUMNS)),
        absorb_region=pd.DataFrame(reg_rows, columns=list(ABSORB_REGION_COLUMNS)),
    )
    absorption["snapshot_digest"] = digest
    return absorption


# --------------------------------------------------------------------------------------
# 곁파일을 읽는 **유일한** 자리
# --------------------------------------------------------------------------------------

def _fail(name: str, what: str, n: int) -> None:
    if n:
        raise AbsorptionError(f"{name}: {what} — {n}행")


def _check_sidecars(snap: Snapshot) -> None:
    """곁파일의 값 공간과 칸 사이 정합을 읽는 쪽에서 막는다.

    지금 값이 맞는 것이 `build` 가 그렇게 만들었기 때문이면, 다른 손이 쓴 파일은 아무것도 막지 않는다.
    """
    man, ann = snap.manifest, snap.annotations
    img, dfc, reg = snap.absorb_image, snap.absorb_defect, snap.absorb_region
    source = (snap.capabilities.get("absorption") or {}).get("source", SOURCE_TEAM_FILE)
    _fail("absorption", "`source` 가 값 공간 밖이다", int(source not in SOURCE_VALUES))
    raw_src = source == SOURCE_RAW_LABELS
    own_raw, own_mat, own_clip = _SOURCE_RULES[source]

    # absorb_image — 모든 장에 한 행. 안 덮은 행은 값이 빈다(§3-1).
    _fail("absorb_image", "매니페스트와 장 집합이 다르다",
          len(set(img["image_id"]) ^ set(man["image_id"])))
    _fail("absorb_image", "`frame_match` 가 값 공간 밖이다", int((~img["frame_match"].isin(FRAME_MATCH_VALUES)).sum()))
    cov = img["covered"].astype(bool)
    _fail("absorb_image", "안 덮은 행에 값이 있다", int((~cov & (
        (img["source_zip"] != "") | (img["material_source"] != "") | (img["frame_match"] != "")
        | img["orig_width_px"].notna() | img["orig_height_px"].notna())).sum()))
    _fail("absorb_image", "덮인 행의 `frame_match` 가 비어 있다", int((cov & (img["frame_match"] == "")).sum()))
    # 원천을 값이 가른다 — 다른 원천의 재질 근거 값이 섞이면 멈춘다.
    _fail("absorb_image", "`material_source` 가 이 원천의 값이 아니다",
          int((cov & ~img["material_source"].isin(own_mat)).sum()))
    # 원본 라벨은 전 장을 덮는다 — 안 덮인 장이 있으면 원천이 다른 판이다.
    _fail("absorb_image", "원본 라벨 원천인데 안 덮인 장이 있다", int((~cov).sum()) if raw_src else 0)
    fm = dict(zip(img["image_id"], img["frame_match"]))

    # absorb_defect — 모든 주석에 한 행(§3-2). 값 공간과 칸 사이 정합.
    _fail("absorb_defect", "주석과 행 집합이 다르다", len(set(dfc["ann_id"]) ^ set(ann["ann_id"])))
    _fail("absorb_defect", "`clip_kind` 가 값 공간 밖이다", int((~dfc["clip_kind"].isin(own_clip)).sum()))
    _fail("absorb_defect", "`raw_source` 가 값 공간 밖이다",
          int((~dfc["raw_source"].isin({own_raw} if raw_src else {own_raw, ""})).sum()))
    raw_na = dfc[list(_RAW)].isna()
    absorbed = dfc["raw_source"] == own_raw
    _fail("absorb_defect", "`raw_source` 와 원좌표의 유무가 어긋난다",
          int((absorbed & raw_na.any(axis=1)).sum() + (~absorbed & ~raw_na.all(axis=1)).sum()))
    applied = dfc["clip_applied"].astype(bool)
    _fail("absorb_defect", "`clip_applied` 와 `clip_kind` 가 어긋난다",
          int((applied != (dfc["clip_kind"] != CLIP_NONE)).sum()))
    # T-4 ③. 원좌표가 있는 결함 행은 크기가 같은 장에만 있다.
    _fail("absorb_defect", "원좌표가 원본 프레임 장에 있다",
          int((absorbed & (dfc["image_id"].map(fm) != MATCH_SAME)).sum()))
    # `none` 의 뜻은 둘뿐이다 — 우리 상자와 원좌표가 같거나, 우리 상자가 비어 `geom_valid=False` 다.
    j = dfc.merge(ann[["ann_id", "image_id", *_BOX, "geom_valid"]], on="ann_id", how="left",
                  suffixes=("", "_ann"), validate="one_to_one")
    _fail("absorb_defect", "`image_id` 가 주석의 것과 다르다", int((j["image_id"] != j["image_id_ann"]).sum()))
    none_abs = (j["raw_source"] == own_raw) & (j["clip_kind"] == CLIP_NONE)
    same = pd.Series(True, index=j.index)
    for a, b in zip(_BOX, _RAW):
        same &= (j[a] == j[b]).fillna(False).astype(bool)
    empty = j[list(_BOX)].isna().all(axis=1) & ~j["geom_valid"].fillna(True).astype(bool)
    # 흡수 판은 빈 상자 행을 `none` 으로 적었다(두 뜻). 원본 라벨 판은 `empty_bbox` 로 가른다 — `none` 은 같음 하나다.
    none_ok = same if raw_src else (same | empty)
    _fail("absorb_defect", "`none` 인데 우리 상자와 원좌표가 다르다", int((none_abs & ~none_ok).sum()))
    _fail("absorb_defect", "`empty_bbox` 가 우리 상자가 빈 행과 어긋난다",
          int(((j["clip_kind"] == CLIP_EMPTY) != empty).sum()) if raw_src else 0)

    # absorb_region — 정상 장만, 프레임 값 공간, 칸 사이 정합(§3-3·§3-4).
    _fail("absorb_region", "`frame` 이 비었거나 값 공간 밖이다", int((~reg["frame"].isin(FRAME_VALUES)).sum()))
    _fail("absorb_region", "`usable_in_tile_frame` 이 `frame` 과 어긋난다",
          int((reg["usable_in_tile_frame"].astype(bool) != (reg["frame"] == FRAME_TILE)).sum()))
    n_poly = reg["polygon_json"].map(lambda s: len(json.loads(s)))
    _fail("absorb_region", "`n_vertices` 가 좌표 수와 다르다", int((n_poly != reg["n_vertices"]).sum()))
    defect_imgs = set(man.loc[man["has_defect"].astype(bool), "image_id"])
    _fail("absorb_region", "결함 장에 정상영역이 있다", int(reg["image_id"].isin(defect_imgs).sum()))
    # 흡수 판은 장의 크기 대조로 프레임을 정했다. 원본 라벨 판(1단계)은 원본 프레임 행만 둔다 —
    # 같은 `region_id` 가 두 판에서 다른 영역을 가리킬 수 있으므로 두 판을 `region_id` 로 잇지 않는다.
    if raw_src:
        want_frame = pd.Series(FRAME_ORIG, index=reg.index)
    else:
        want_frame = reg["image_id"].map(fm).map({MATCH_SAME: FRAME_TILE, MATCH_TILE_OF_ORIG: FRAME_ORIG})
    _fail("absorb_region", "`frame` 이 원천의 프레임 규칙과 어긋난다", int((want_frame != reg["frame"]).sum()))


def load_absorbed(root: Path | str) -> Snapshot:
    """흡수 스냅샷을 읽는 **유일한** 자리다.

    1. 경로를 받아 스스로 `load_snapshot(verify=True)` 로 읽는다. 이미 읽은 `Snapshot` 은 받지 않는다 —
       `verify=False` 로 읽은 객체가 그대로 통과한다.
    2. 곁파일은 **잠금 목록에 있을 때만** 받는다. `load_snapshot` 은 곁파일을 디스크에 있는가로 읽고
       `verify_snapshot` 은 잠금 목록에 있는가로 검사해서, 잠금 밖 곁파일은 검증 없이 읽힌다.
    3. 값 공간과 칸 사이 정합을 검사한다(`_check_sidecars`).
    4. §5-2·§9-2. `complete` 가 아니면 로그에 남긴다. **잠정판으로 낸 수치를 논문에 싣지 않는다.**
    """
    if isinstance(root, Snapshot):
        raise TypeError("이미 읽은 Snapshot 은 받지 않는다 — 검증 없이 읽힌 객체가 통과한다. 경로를 준다")
    root = Path(root)
    snap = load_snapshot(root, verify=True)
    locked = _locked_members(root)
    for attr, name in SIDECARS:
        if (root / name).exists() and name not in locked:
            raise AbsorptionError(f"{name} 이 폴더에 있으나 잠금 목록에 없다 — 검증 없이 읽힌다")
        if getattr(snap, attr) is None:
            raise AbsorptionError(f"흡수 스냅샷이 아니다 — {name} 이 없다")
    _check_sidecars(snap)
    status = ((snap.capabilities.get("absorption") or {}).get("status"))
    if status != "complete":
        _log.warning("흡수 스냅샷이 %s 이다(%s). 누락 아카이브가 남아 있어 참여자별 덮임이 고르지 않다. "
                     "이 판으로 낸 수치를 논문에 싣지 않는다", status, root)
    return snap


# --------------------------------------------------------------------------------------
# §6-2. 학습 재료로 나가는 **유일한** 경로
# --------------------------------------------------------------------------------------

#: 학습 산출로 나가는 칸. **허용 목록이다** — 여기 없는 칸은 이름을 몰라도 빠진다.
#: 규칙: 결합 키와 타일 프레임의 기하만 나간다. 원본 크기와 그 요약 비트(`frame_match`), 출처
#: (`source_zip` 은 아카이브 이름에 결함 여부와 종류가 박혀 있다), 덮임을 적은 칸(`covered`·`raw_source`),
#: 재질 근거, 자유 서술의 **칸**은 나가지 않는다. 금지 목록은 칸이 늘 때마다 샌다.
#: **칸을 막아도 덮임은 막히지 않는다.** `absorb_defect` 의 원좌표가 비어 있는가가 덮임 비트를 그대로 싣고,
#: 덮임은 참여자·재질·결함 종류와 얽혀 있다. 이 목록으로는 닫히지 않는다 — 원좌표를 학습에 쓸지는 명세가 정한다.
#: **정상영역은 학습 산출에 나가지 않는다.** 정상 장에만 있어 행이 있다는 것이 곧 정상이라는 라벨이다 —
#: 입력 · 마스크 · 표집 어느 꼴로도 쓰면 라벨이 샌다. 진단은 `load_absorbed` 로 읽는다.
TRAINING_COLUMNS: dict[str, tuple[str, ...]] = {
    "absorb_image": ("image_id",),
    "absorb_defect": ("ann_id", "image_id", *_RAW, "clip_applied", "clip_kind"),
}
#: 허용 목록이 스스로 지키는 규칙. 이름에 이 조각이 든 칸은 허용 목록에 들 수 없다.
FORBIDDEN_NAME_PARTS: tuple[str, ...] = (
    "orig_", "width", "height", "frame_match", "zip", "source", "covered", "note")
_SIDECAR_SCHEMA = {"absorb_image": ABSORB_IMAGE_COLUMNS, "absorb_defect": ABSORB_DEFECT_COLUMNS,
                   "absorb_region": ABSORB_REGION_COLUMNS}
#: §7-2 가 막으라고 한 두 이름을 포함해, 허용 목록 밖의 곁파일 칸 전부.
BLOCKED_COLUMNS: frozenset[str] = frozenset(
    c for name, cols in _SIDECAR_SCHEMA.items() for c in cols if c not in TRAINING_COLUMNS.get(name, ()))


def check_training_allowlist() -> None:
    """허용 목록이 규칙을 지키는지 본다. 칸이 곁파일에 없거나 금지 조각을 담으면 멈춘다."""
    for name, cols in TRAINING_COLUMNS.items():
        unknown = [c for c in cols if c not in _SIDECAR_SCHEMA[name]]
        banned = [c for c in cols if any(p in c for p in FORBIDDEN_NAME_PARTS)]
        if unknown or banned or "image_id" not in cols:
            raise AbsorptionError(f"{name} 허용 목록이 규칙과 다르다 — 모르는 칸 {unknown} · 금지 {banned}")


def export_for_training(root: Path | str) -> dict[str, pd.DataFrame]:
    """곁파일에서 학습 재료를 뽑는다. **스냅샷을 검증하고, 평가셋을 떼고, 허용 목록의 칸만 낸다.**

    경로를 받는다(§6-2 1단계). 다른 경로를 만들지 않는다. 거른 뒤 **한 번 더 세는 것**이 요점이다 —
    거르는 코드가 틀려도 이 단언이 잡는다. 경고로 넘어가지 않고 예외로 멈춘다.
    """
    snap = load_absorbed(root)
    check_training_allowlist()
    man = snap.manifest
    allowed = set(man.loc[man["split"] != "eval", "image_id"])
    eval_ids = set(man.loc[man["split"] == "eval", "image_id"])

    out: dict[str, pd.DataFrame] = {}
    for name, _file in SIDECARS:
        if name not in TRAINING_COLUMNS:          # 허용 목록에 없는 곁파일은 통째로 나가지 않는다
            continue
        frame = getattr(snap, name)
        kept = frame[frame["image_id"].isin(allowed)]
        kept = kept[list(TRAINING_COLUMNS[name])].copy()
        # 산출 **직전에** 다시 센다. 거르는 코드와 따로 선 검사다.
        leaked = set(kept["image_id"]) & eval_ids
        if leaked:
            raise AbsorptionError(f"{name}: 평가셋 {len(leaked)}장이 학습 산출에 남았다")
        if tuple(kept.columns) != TRAINING_COLUMNS[name]:
            raise AbsorptionError(f"{name}: 산출 칸이 허용 목록과 다르다 — {list(kept.columns)}")
        banned = [c for c in kept.columns if any(p in c for p in FORBIDDEN_NAME_PARTS)]
        if banned:
            raise AbsorptionError(f"{name}: 금지된 칸이 산출에 남았다 — {banned}")
        out[name] = kept.reset_index(drop=True)
    if "absorb_region" in out:
        raise AbsorptionError("absorb_region: 정상영역이 학습 산출에 남았다 — 있음이 곧 정상이라는 라벨이다")
    return out
