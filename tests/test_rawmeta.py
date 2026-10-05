"""원본 라벨에서 곁파일 셋을 내는 빌더를 고정한다 — 전수 메타데이터화 미니스펙 2판 12절.

**합성 자료만 쓴다. 동결 자산 · 원본 zip · 외부 드라이브를 읽지 않는다.**
합성 v1 은 손으로 꾸미지 않는다 — v1 을 실제로 만든 수집 코드(`read_labels` · `to_frames`)로 합성 라벨 zip 에서 만든다.
그래야 빌더의 "v1 과 주석마다 바이트로 같다" 가 수집 단계와 맞대는 시험이 된다.
"""
from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from data import absorb, rawmeta
from data.label_map import load_label_map
from data.manifest_io import load_snapshot, verify_snapshot, write_snapshot
from scripts.build_manifest_v0 import build_runs, to_frames
from scripts.measure_tiling_geometry import read_labels

REPO = Path(__file__).resolve().parents[1]
LM = REPO / "configs" / "label_map.yaml"
SQ = [(100, 100), (120, 100), (120, 120), (100, 120)]


def _poly(pts):
    return [p[0] for p in pts], [p[1] for p in pts]


def _ann(cls, case, pts):
    xs, ys = _poly(pts)
    return {"class": cls, "case": case, "tool": "polygon", "coordinate": {"x": xs, "y": ys}}


def _label(iid, anns=(), *, w=1280, h=720, mat="ST", typ="RT", extra=None):
    o = {"info": {"id": iid, "type": typ, "material": mat},
         "image_data": {"width": w, "height": h, "file_name": f"RT_{mat}_01_{iid}.jpg",
                        "format": "jpg", "information": "x"},
         "annotations": list(anns), "meta": {"is_crowd": 0}}
    if extra:
        o.update(extra)
    return o


#: 기본 합성 원본 — 각 장이 겨누는 것을 이름에 적었다.
BASE = {
    "TL_RTST_결함_2. 기공.zip": [
        # 101: 안쪽 기공(none) + 경계 밖으로 나간 균열(boundary)
        _label(101, [_ann("defect", "porosity", SQ),
                     _ann("defect", "crack", [(-5, 10), (30, 10), (30, 40), (-5, 40)])]),
        _label(102, [_ann("defect", "porosity", SQ)]),                    # 평가셋 결함
        # 105: case 가 빈 다각형은 건너뛰고 순번이 늘지 않는다(규칙 5)
        _label(105, [_ann("defect", "", [(0, 0), (1280, 0), (1280, 720), (0, 720)]),
                     _ann("defect", "porosity", SQ)]),
        # 106: 면적 0 인 다각형 — v1 상자가 빈다(empty_bbox). 같은 장의 정상 기공은 살린다
        _label(106, [_ann("defect", "porosity", [(10, 10), (20, 20), (30, 30)]),
                     _ann("defect", "porosity", SQ)]),
        # 109: 자기교차(나비꼴) — 가장 큰 조각만 남아 상자가 줄어든다(polygon_repair)
        _label(109, [_ann("defect", "slag_inclusion", [(100, 100), (140, 140), (140, 100), (100, 140)])]),
        # 112: 결함 장의 class=normal · case 빈 다각형은 건너뛴다(규칙 2 · 5)
        _label(112, [_ann("normal", "", SQ), _ann("defect", "porosity", SQ)]),
        # 113: 정수가 아닌 좌표 — v1 상자는 바깥으로(내림 · 올림), polygon_json 은 int() 다
        _label(113, [_ann("defect", "porosity", [(100.4, 100.2), (120.6, 100.2), (120.6, 120.7), (100.4, 120.7)])]),
    ],
    "TL_RTST_정상.zip": [
        _label(103, [_ann("normal", "", [(0, 300), (4000, 300), (4000, 900), (0, 900)]),
                     _ann("normal", "", [(10, 10), (50, 10), (50, 50)])], w=4000, h=1272),
        _label(104, [_ann("normal", "", [(0, 100), (1280, 100), (1280, 600), (0, 600)])]),
        _label(110, [_ann("normal", "", [(0, 100), (1280, 100), (1280, 600), (0, 600)])]),  # 평가셋 정상
    ],
    "VL_RTST_결함_1. 균열.zip": [
        _label(107, [_ann("defect", "incomplete_penetration", SQ)]),   # 라벨 공간 밖 — v1 에 없다
        b"{not json",                                                   # 파싱 실패 — 장째 빠진다
        _label(111, [_ann("defect", "porosity", SQ)], typ="VT"),        # RT 가 아니다
    ],
}
EVAL = {"aihub71761:102", "aihub71761:110"}


def _write_zips(root: Path, groups: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, items in groups.items():
        with zipfile.ZipFile(root / name, "w") as z:
            for n, item in enumerate(items):
                data = item if isinstance(item, bytes) else (
                    "﻿" + json.dumps(item, ensure_ascii=False)).encode("utf-8")   # BOM — utf-8-sig
                z.writestr(f"label/{n:03d}.json", data)
    return root


def _make_v1(label_root: Path, out: Path) -> Path:
    """수집 코드로 v1 꼴의 스냅샷을 낸다. 타일 뒤의 크기 · 분할만 채운다."""
    lm = load_label_map(LM)
    recs = [r for r in read_labels(label_root) if r.modality == "RT"]
    m, a, _dropped, _skipped = to_frames(recs, lm, build_runs(recs))
    size = {f"aihub71761:{r.image_id}": (r.width, r.height) for r in recs}
    m = m.copy()
    m["sha256"] = [i.split(":")[1].zfill(64) for i in m["image_id"]]
    m["width_px"], m["height_px"] = 1280, 720
    m["phash_hex"], m["strata_key"], m["eval_subset"] = "0" * 16, "ST|porosity", ""
    m["split"] = ["eval" if i in EVAL else "train" for i in m["image_id"]]
    m["client"] = ["" if i in EVAL else "C1" for i in m["image_id"]]
    tiles = pd.DataFrame({
        "image_id": m["image_id"],
        "provenance": ["N-crop" if size[i] == (1280, 720) else "N-tile" for i in m["image_id"]],
        "reason": ["ok" if size[i] == (1280, 720) else "tiled" for i in m["image_id"]]})
    write_snapshot(out, m, a, {"snapshot_id": "synthetic_v1",
                               "capabilities": {"verdict_mode": "clause_only"}}, tiles=tiles)
    return out


@pytest.fixture
def env(tmp_path):
    labels = _write_zips(tmp_path / "labels", BASE)
    v1 = _make_v1(labels, tmp_path / "v1")
    return {"labels": labels, "v1": v1, "tmp": tmp_path}


def _build(env, out="v3", labels=None, **kw):
    return rawmeta.build(env["v1"], labels or env["labels"], env["tmp"] / out, LM, **kw)


def _kinds(root: Path) -> dict[str, str]:
    snap = absorb.load_absorbed(root)
    return dict(zip(snap.absorb_defect["ann_id"], snap.absorb_defect["clip_kind"]))


# --- 정상 경로 -------------------------------------------------------------------------

def test_전_장의_곁파일을_내고_읽는_쪽_검사를_지난다(env):
    acct = _build(env)
    snap = absorb.load_absorbed(env["tmp"] / "v3")
    v1 = load_snapshot(env["v1"])
    assert set(snap.absorb_image["image_id"]) == set(v1.manifest["image_id"])
    assert snap.absorb_image["covered"].all() and set(snap.absorb_image["material_source"]) == {"label_json"}
    assert set(snap.absorb_defect["ann_id"]) == set(v1.annotations["ann_id"])
    assert set(snap.absorb_defect["raw_source"]) == {"raw_label"}
    assert set(snap.absorb_region["frame"]) == {"orig"} and not snap.absorb_region["usable_in_tile_frame"].any()
    assert acct["source"] == "raw_labels" and acct["status"] == "complete"
    assert snap.capabilities["snapshot_id"] == rawmeta.SNAPSHOT_ID


def test_clip_kind_가_다섯_갈래로_나온다(env):
    _build(env)
    k = _kinds(env["tmp"] / "v3")
    assert k["aihub71761:101#0"] == "none"
    assert k["aihub71761:101#1"] == "boundary"
    assert k["aihub71761:106#0"] == "empty_bbox"
    assert k["aihub71761:109#0"] == "polygon_repair"
    assert k["aihub71761:105#0"] == "none"          # 빈 case 를 건너뛰어 순번이 0 이다


def test_원좌표_상자는_자르지_않은_값이다(env):
    _build(env)
    d = absorb.load_absorbed(env["tmp"] / "v3").absorb_defect.set_index("ann_id")
    assert tuple(d.loc["aihub71761:101#1", ["raw_bbox_x1_px", "raw_bbox_y1_px", "raw_bbox_x2_px",
                                            "raw_bbox_y2_px"]]) == (-5, 10, 30, 40)


def test_영역은_원본_프레임_행만이고_순번은_배열_순서다(env):
    _build(env)
    reg = absorb.load_absorbed(env["tmp"] / "v3").absorb_region.set_index("region_id")
    assert {"aihub71761:103#r0", "aihub71761:103#r1", "aihub71761:104#r0", "aihub71761:110#r0"} == set(reg.index)
    assert reg.loc["aihub71761:103#r1", "polygon_json"] == "[[10,10],[50,10],[50,50]]"


def test_회계는_원본에_있으나_v1_에_없는_장과_파싱_실패를_센다(env):
    acct = _build(env)
    assert acct["labels_not_in_v1"] == {"out_of_label_space": 1}
    assert acct["parse_failures"] == {"VL_RTST_결함_1. 균열.zip": 1}
    assert acct["non_rt_records"] == 1
    assert {a["name"] for a in acct["label_archives"]} == set(BASE)


def test_T18_회계에_평가_장의_값이_없다(env):
    """평가 행은 장 수만 센다. 평가 장의 식별자 · 값이 회계 어디에도 나오지 않는다."""
    acct = _build(env)
    text = json.dumps(acct, ensure_ascii=False)
    for iid in EVAL:
        assert iid not in text
    assert acct["eval_images"] == 2
    tv = acct["trainval"]
    # 110 의 영역 · 102 의 결함은 빠졌다. 평가 행이 섞이면 none 이 6, 결함 행이 9, 영역 행이 4 가 된다
    assert tv["region_rows"] == 3 and tv["defect_rows"] == 8
    assert tv["clip_kind_rows"] == {"boundary": 1, "empty_bbox": 1, "none": 5, "polygon_repair": 1}


def test_T7_같은_입력은_같은_지문을_낸다(env):
    a = _build(env, out="a")
    b = _build(env, out="b")
    assert a["snapshot_digest"] == b["snapshot_digest"]


def test_T9_분할과_묶음은_v1과_전_행_같고_다른_열은_iso_codes_뿐이다(env):
    acct = _build(env)
    v1, v3 = load_snapshot(env["v1"]), load_snapshot(env["tmp"] / "v3")
    for c in ("split", "client", "eval_subset", "group_id", "group_size", "strata_key"):
        assert v3.manifest[c].equals(v1.manifest[c]), c
    assert set(acct["changed_columns_vs_v1"]) <= {"iso_codes"}
    assert v3.annotations.equals(v1.annotations)


# --- 입력 규약(T-10) -----------------------------------------------------------------------

def _variant(env, name, mutate) -> Path:
    groups = {k: list(v) for k, v in BASE.items()}
    mutate(groups)
    return _write_zips(env["tmp"] / name, groups)


def _replace(groups, zip_name, idx, item):
    groups[zip_name][idx] = item


@pytest.mark.parametrize("case", ["weird", "normal"])
def test_T10_사상표에_없는_case_는_건너뛰지_않고_멈춘다(env, case):
    labels = _variant(env, f"bad_{case}", lambda g: _replace(
        g, "TL_RTST_결함_2. 기공.zip", 0,
        _label(101, [_ann("defect", case, SQ), _ann("defect", "crack", [(-5, 10), (30, 10), (30, 40), (-5, 40)])])))
    with pytest.raises(rawmeta.RawMetaError, match="사상표 입력"):
        _build(env, out=f"o_{case}", labels=labels)
    assert not (env["tmp"] / f"o_{case}").exists()


def test_T10_info_id_가_겹치면_멈춘다(env):
    labels = _variant(env, "dup", lambda g: g["TL_RTST_정상.zip"].append(_label(101, [_ann("normal", "", SQ)])))
    with pytest.raises(rawmeta.RawMetaError, match="info.id"):
        _build(env, out="o_dup", labels=labels)


def test_T10_v1_에_있는_장이_원본에_없으면_멈춘다(env):
    labels = _variant(env, "gone", lambda g: g["TL_RTST_정상.zip"].pop(1))
    with pytest.raises(rawmeta.RawMetaError, match="원본 라벨에 없다"):
        _build(env, out="o_gone", labels=labels)


def test_T10b_원본_읽기가_수집_단계의_읽기와_같다(env):
    ours = rawmeta.read_label_zips(env["labels"])
    theirs = read_labels(env["labels"])
    a = {r.info_id: (r.modality, r.material, r.width, r.height, r.is_normal, r.file_name, r.polys, r.poly_cases)
         for r in ours["records"]}
    b = {r.image_id: (r.modality, r.material, r.width, r.height, r.is_normal, r.file_name, r.polys, r.poly_cases)
         for r in theirs if r.modality == "RT"}
    assert a == b and len(a) == 11
    assert ours["non_rt"] == 1 and ours["parse_failures"] == {"VL_RTST_결함_1. 균열.zip": 1}


# --- v1 일치 검사(T-11) ----------------------------------------------------------------------

def _tamper_v1(env, name, **cols):
    s = load_snapshot(env["v1"])
    ann = s.annotations.copy()
    i = ann.index[ann["ann_id"] == "aihub71761:101#0"][0]
    for c, v in cols.items():
        ann.loc[i, c] = v
    out = env["tmp"] / name
    write_snapshot(out, s.manifest, ann, s.capabilities, tiles=s.tiles)
    return out


@pytest.mark.parametrize("cols, msg", [
    ({"polygon_json": "[[100,100],[120,100],[120,120],[100,121]]"}, "바이트로 다르다"),
    ({"iso_code": "100"}, "ISO"),
    ({"bbox_x1_px": 99}, "다시 낸 상자"),
], ids=["polygon_json", "iso", "상자"])
def test_T11_v1_과_주석이_다르면_쓰기_전에_멈춘다(env, cols, msg):
    bad_v1 = _tamper_v1(env, "v1bad", **cols)
    with pytest.raises(rawmeta.RawMetaError, match=msg):
        rawmeta.build(bad_v1, env["labels"], env["tmp"] / "o", LM)
    assert not (env["tmp"] / "o").exists()


# --- clip_kind 규칙(T-12) ------------------------------------------------------------------

@pytest.mark.parametrize("v1_box, raw, flags, want", [
    (None, (1, 1, 5, 5), "zero_area", "empty_bbox"),
    ((1, 1, 5, 5), (1, 1, 5, 5), "", "none"),
    ((0, 1, 5, 5), (-3, 1, 5, 5), "", "boundary"),
    ((1, 1, 3, 5), (1, 1, 5, 5), "self_intersect;multipart_largest_kept", "polygon_repair"),
    ((0, 1, 5, 5), (-3, 1, 5, 5), "multipart_largest_kept", "both"),
])
def test_T12_clip_kind_의_다섯_값(v1_box, raw, flags, want):
    assert rawmeta.classify_clip(v1_box, raw, flags, 1280, 720) == want


def test_정수가_아닌_좌표는_바깥으로_정수화해_v1_상자와_같다(env):
    """`int()` 로 깎으면 원좌표 상자가 v1 상자보다 작아 자르기로 설명되지 않는다 — 실물 1단계가 멈춘 자리다."""
    acct = _build(env)
    d = absorb.load_absorbed(env["tmp"] / "v3").absorb_defect.set_index("ann_id")
    row = d.loc["aihub71761:113#0"]
    assert tuple(row[["raw_bbox_x1_px", "raw_bbox_y1_px", "raw_bbox_x2_px", "raw_bbox_y2_px"]]) == (100, 100, 121, 121)
    assert row["clip_kind"] == "none"
    assert acct["trainval"]["noninteger_coordinate_rows"] == 1
    assert rawmeta.raw_box([1.5, 3], [2, 4.2]) == (1, 2, 3, 5)
    assert rawmeta.raw_box([1, 3], [2, 4]) == (1, 2, 3, 4)


def test_T12_설명되지_않는_차이는_멈춘다():
    with pytest.raises(rawmeta.RawMetaError, match="설명할 수 없다"):
        rawmeta.classify_clip((2, 2, 5, 5), (1, 1, 5, 5), "", 1280, 720)


def test_T12_장으로_접는_순서():
    got = rawmeta.fold_images({"a": ["empty_bbox", "boundary"], "b": ["both", "polygon_repair"],
                               "c": ["polygon_repair", "none"], "d": ["none"]})
    assert dict(got) == {"empty_bbox": 1, "boundary": 1, "polygon_repair": 1}


# --- 읽는 쪽 검사 — 원본 라벨 판(T-14) -------------------------------------------------------

def _rewrite(src: Path, dst: Path, caps=None, **frames) -> Path:
    s = load_snapshot(src)
    parts = {"absorb_image": s.absorb_image, "absorb_defect": s.absorb_defect,
             "absorb_region": s.absorb_region} | frames
    write_snapshot(dst, s.manifest, s.annotations, caps or s.capabilities, tiles=s.tiles, **parts)
    return dst


def _m(name, fn):
    def go(s):
        return {name: fn(getattr(s, name).copy())}
    return go


def _set(df, key, keyval, **vals):
    i = df.index[df[key] == keyval][0]
    for k, v in vals.items():
        df.loc[i, k] = v
    return df


def _uncover(d):
    i = d.index[d["image_id"] == "aihub71761:104"][0]
    d.loc[i, ["covered", "source_zip", "material_source", "frame_match"]] = [False, "", "", ""]
    d.loc[i, ["orig_width_px", "orig_height_px"]] = [pd.NA, pd.NA]
    return d


#: 원본 라벨 판에서만 서는 읽는 쪽 검사. (이름, 바꾸는 법, 문구)
RAW_READ_CHECKS = [
    ("raw_안덮인장", _m("absorb_image", _uncover), "absorb_image: 원본 라벨 원천인데 안 덮인 장이 있다"),
    ("raw_empty_bbox", _m("absorb_defect", lambda d: _set(d, "ann_id", "aihub71761:101#0",
                                                           clip_kind="empty_bbox", clip_applied=True)),
     "absorb_defect: `empty_bbox` 가 우리 상자가 빈 행과 어긋난다"),
]
#: 두 원천의 값이 섞이면 값 공간 검사가 잡는다(T-14).
MIXED = [
    ("raw_판에_absorbed", _m("absorb_defect", lambda d: _set(d, "ann_id", "aihub71761:101#0", raw_source="absorbed")),
     "absorb_defect: `raw_source` 가 값 공간 밖이다"),
    ("raw_판에_folder_nominal", _m("absorb_image", lambda d: _set(d, "image_id", "aihub71761:104",
                                                                  material_source="folder_nominal")),
     "absorb_image: `material_source` 가 이 원천의 값이 아니다"),
    ("raw_판에_tile_영역", _m("absorb_region", lambda d: _set(d, "region_id", "aihub71761:104#r0", frame="tile",
                                                            usable_in_tile_frame=True)),
     "absorb_region: `frame` 이 원천의 프레임 규칙과 어긋난다"),
]


@pytest.mark.parametrize("mutate, message", [(m, msg) for _n, m, msg in RAW_READ_CHECKS + MIXED],
                         ids=[n for n, _m_, _msg in RAW_READ_CHECKS + MIXED])
def test_T14_원본_라벨_판의_읽는_쪽_검사(env, mutate, message):
    _build(env)
    bad = _rewrite(env["tmp"] / "v3", env["tmp"] / "bad", **mutate(load_snapshot(env["tmp"] / "v3")))
    with pytest.raises(absorb.AbsorptionError, match=re.escape(message)):
        absorb.load_absorbed(bad)


def test_T14_원천_값이_값_공간_밖이면_멈춘다(env):
    _build(env)
    s = load_snapshot(env["tmp"] / "v3")
    caps = dict(s.capabilities)
    caps["absorption"] = dict(caps["absorption"], source="somewhere")
    bad = _rewrite(env["tmp"] / "v3", env["tmp"] / "bad", caps=caps)
    with pytest.raises(absorb.AbsorptionError, match=re.escape("absorption: `source` 가 값 공간 밖이다")):
        absorb.load_absorbed(bad)


# --- 학습 산출(T-15) -----------------------------------------------------------------------

def test_T15_정상영역은_학습_산출에_없고_지름길_칸도_없다(env):
    _build(env)
    out = absorb.export_for_training(env["tmp"] / "v3")
    assert set(out) == {"absorb_image", "absorb_defect"}
    cols = set().union(*(set(f.columns) for f in out.values()))
    assert not cols & {"orig_width_px", "orig_height_px", "frame_match", "source_zip", "material_source",
                       "covered", "absorb_note", "raw_source"}
    assert not any(set(f["image_id"]) & EVAL for f in out.values())


# --- 부르는 자리(T-16) ----------------------------------------------------------------------

#: `export_for_training` · `load_absorbed` 를 부를 수 있는 파일. 여기 밖에 생기면 시험이 떨어진다.
CALLERS_ALLOWED = {"data/absorb.py", "data/rawmeta.py", "tests/test_absorb.py", "tests/test_rawmeta.py",
                   "scripts/crosscheck/run_rawmeta_stage1.py", "scripts/crosscheck/xc_rawmeta_team.py",
                   "scripts/crosscheck/rawmeta_clipkind_table.py",
                   "tests/test_score_unified.py"}     # 마지막은 채점 진입점이 부르지 못하게 이름을 금지 목록에 적는 시험이다


def test_T16_곁파일을_올리는_함수를_부르는_자리가_정해져_있다():
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=False)
    if r.returncode != 0:
        pytest.skip("git 으로 파일 목록을 낼 수 없다")
    files = [f for f in r.stdout.splitlines() if f]
    pat = re.compile(r"\b(export_for_training|load_absorbed)\b")
    hits = {f for f in files if pat.search((REPO / f).read_text(encoding="utf-8", errors="replace"))}
    assert "data/absorb.py" in hits and len(files) > 50, "검사 범위가 비었다"
    assert not (hits - CALLERS_ALLOWED), f"허용 목록 밖에서 부른다: {sorted(hits - CALLERS_ALLOWED)}"


# --- 출력 경로 가드(T-17) --------------------------------------------------------------------

@pytest.mark.parametrize("name", ["manifest_v1", "manifest_v2_absorbed"])
def test_T17_동결본_잠정판_이름의_경로에는_쓰지_않는다(env, name):
    with pytest.raises(rawmeta.RawMetaError, match="쓰지 않는다"):
        _build(env, out=name)
    assert not (env["tmp"] / name).exists()


def test_T17_잠긴_판이_있는_경로에는_쓰지_않는다(env):
    _build(env, out="done")
    before = verify_snapshot(env["tmp"] / "done")
    with pytest.raises(rawmeta.RawMetaError, match="잠긴 판"):
        _build(env, out="done")
    assert verify_snapshot(env["tmp"] / "done") == before


def test_T17_입력_스냅샷과_같은_경로에는_쓰지_않는다(env):
    with pytest.raises(rawmeta.RawMetaError, match="입력 스냅샷"):
        rawmeta.build(env["v1"], env["labels"], env["v1"], LM)


def test_V8_원본_zip_이_빌드_중에_바뀌면_멈춘다(env, monkeypatch):
    real = rawmeta.zip_digests
    calls = []

    def fake(root):
        calls.append(1)
        d = real(root)
        return d if len(calls) == 1 else {k: "0" * 64 for k in d}
    monkeypatch.setattr(rawmeta, "zip_digests", fake)
    with pytest.raises(rawmeta.RawMetaError, match="zip"):
        _build(env)


# --- 두께 탐색기의 양성 대조(T-19) ------------------------------------------------------------

def test_T19_두께_탐색기는_심은_키와_값을_잡는다(env):
    labels = _variant(env, "scale", lambda g: _replace(
        g, "TL_RTST_정상.zip", 1,
        _label(104, [_ann("normal", "", [(0, 100), (1280, 100), (1280, 600), (0, 600)])],
               extra={"meta": {"is_crowd": 0, "thickness_mm": 12, "note": "IQI 3.2mm"}})))
    hits = rawmeta.read_label_zips(labels)["scale_key_hits"]
    assert hits.get("key:meta.thickness_mm") == 1
    assert hits.get("value:meta.note") == 1
    assert not rawmeta.read_label_zips(env["labels"])["scale_key_hits"], "기본 합성 원본에는 없어야 한다"


# --- 교차 검산의 평가 행(T-18) ----------------------------------------------------------------

def _team_records(root: Path, tweak: dict[str, int]) -> list[dict]:
    """빌드된 판의 값으로 정리된 파일 꼴의 줄을 만든다. `tweak` 의 장은 첫 결함 상자를 그만큼 민다."""
    snap = absorb.load_absorbed(root)
    iso = dict(zip(snap.annotations["ann_id"], snap.annotations["iso_code"]))
    img = snap.absorb_image.set_index("image_id")
    recs = []
    for iid in snap.manifest["image_id"]:
        d = snap.absorb_defect[snap.absorb_defect["image_id"] == iid].sort_values("ann_id")
        anns = []
        for n, r in enumerate(d.itertuples(index=False)):
            dx = tweak.get(iid, 0) if n == 0 else 0
            anns.append({"annotation_role": "defect", "ann_idx": n, "class_std_iso6520": str(iso[r.ann_id]),
                         "bbox_xmin": int(r.raw_bbox_x1_px) + dx, "bbox_ymin": int(r.raw_bbox_y1_px),
                         "bbox_xmax": int(r.raw_bbox_x2_px), "bbox_ymax": int(r.raw_bbox_y2_px)})
        for r in snap.absorb_region[snap.absorb_region["image_id"] == iid].itertuples(index=False):
            pts = json.loads(r.polygon_json)
            anns.append({"annotation_role": "normal_region", "polygon_x": json.dumps([p[0] for p in pts]),
                         "polygon_y": json.dumps([p[1] for p in pts])})
        recs.append({"orig_info_id": iid.split(":")[1], "modality": "RT",
                     "width_px": int(img.loc[iid, "orig_width_px"]), "height_px": int(img.loc[iid, "orig_height_px"]),
                     "annotations": anns})
    return recs


def test_T18_교차_검산은_평가_장을_한_비트와_장_수로만_낸다(env):
    from scripts.crosscheck.xc_rawmeta_team import compare
    _build(env)
    root = env["tmp"] / "v3"
    jl = env["tmp"] / "team.jsonl"
    jl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                          for r in _team_records(root, {"aihub71761:102": 3, "aihub71761:101": 2})),
                  encoding="utf-8", newline="\n")
    summary, diffs = compare(root, jl)
    assert summary["eval"] == {"all_equal": False, "images_differ": 1,
                               "비고": "평가 장은 한 비트와 같지 않은 장 수만 낸다"}
    assert [d["image_id"] for d in diffs] == ["aihub71761:101"]
    assert diffs[0]["reasons"] == ["defect_raw_box"]
    text = json.dumps(summary, ensure_ascii=False) + json.dumps(diffs, ensure_ascii=False)
    assert not any(i in text for i in EVAL), "평가 장의 식별자가 산출에 나왔다"


def test_T18_같은_값이면_차이가_없다(env):
    from scripts.crosscheck.xc_rawmeta_team import compare
    _build(env)
    root = env["tmp"] / "v3"
    jl = env["tmp"] / "team.jsonl"
    jl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _team_records(root, {})),
                  encoding="utf-8", newline="\n")
    summary, diffs = compare(root, jl)
    assert diffs == [] and summary["eval"]["all_equal"] and "images_differ" not in summary["trainval"]


# --- 정본 승격 — 지문이 같을 때만 옮기고 잠근다 -------------------------------------------------

def test_정본_승격은_지문이_다르면_정본_경로를_만들지_않는다(env):
    import os
    import stat

    from scripts.crosscheck.run_rawmeta_stage1 import promote
    acct = _build(env, out="staging")
    out = env["tmp"] / "canonical"
    with pytest.raises(rawmeta.RawMetaError, match="정본 경로에 쓰지 않았다"):
        promote(env["tmp"] / "staging", out, "0" * 64)
    assert not out.exists() and (env["tmp"] / "staging").exists()
    assert promote(env["tmp"] / "staging", out, acct["snapshot_digest"]) == acct["snapshot_digest"]
    assert not (env["tmp"] / "staging").exists() and verify_snapshot(out) == acct["snapshot_digest"]
    for p in out.iterdir():
        assert not (os.stat(p).st_mode & stat.S_IWRITE), f"{p.name} 이 읽기 전용이 아니다"


# --- 정본 쓰기 가드 — 실행기 수준 ---------------------------------------------------------------
#
# 정본 뿌리를 임시 폴더로 바꿔 실행기 `main` 을 그대로 부른다. 빌더는 진짜를 부르고 부른 목적지만 적는다.

@pytest.fixture
def runner(env, monkeypatch):
    import scripts.crosscheck.run_rawmeta_stage1 as R
    canon = env["tmp"] / "interim"
    canon.mkdir()
    for name, value in {"V1": env["v1"], "LABELS": env["labels"], "LABEL_MAP": LM, "V2": None,
                        "CANONICAL_ROOT": canon, "OUT": canon / "manifest_v3_rawlabels"}.items():
        monkeypatch.setattr(R, name, value)
    calls: list[Path] = []
    real = rawmeta.build
    monkeypatch.setattr(rawmeta, "build", lambda *a, **k: (calls.append(Path(a[2])), real(*a, **k))[1])
    return {"main": R.main, "out": canon / "manifest_v3_rawlabels", "canon": canon, "calls": calls, "tmp": env["tmp"]}


@pytest.mark.parametrize("argv", [[], ["--out", "{canon}/other"]], ids=["기본_정본", "정본_뿌리_아래"])
def test_정본_목적지에_지문_대조가_빠지면_빌더를_부르기_전에_거부한다(runner, capsys, argv):
    argv = [a.format(canon=runner["canon"]) for a in argv]
    assert runner["main"](argv) == 2
    assert "필수다" in capsys.readouterr().out
    assert runner["calls"] == [] and list(runner["canon"].iterdir()) == []


@pytest.mark.parametrize("staging", ["{out}", "{out}/stage", "{tmp}", "{canon}/stage"],
                         ids=["같다", "목적지_아래", "목적지를_품는다", "정본_뿌리_안"])
def test_임시_경로가_목적지와_겹치면_빌더를_부르기_전에_거부한다(runner, capsys, staging):
    st = staging.format(out=runner["out"], tmp=runner["tmp"], canon=runner["canon"])
    assert runner["main"](["--expect-digest", "0" * 64, "--staging", st]) == 2
    assert "임시 경로" in capsys.readouterr().out
    assert runner["calls"] == [] and list(runner["canon"].iterdir()) == []


def test_지문이_다르면_실행기가_정본을_만들지_않는다(runner):
    stage = runner["tmp"] / "stage"
    assert runner["main"](["--expect-digest", "0" * 64, "--staging", str(stage)]) == 2
    assert runner["calls"] == [stage]
    assert not runner["out"].exists() and list(runner["canon"].iterdir()) == []
    assert verify_snapshot(stage)              # 대조에 진 판은 임시 경로에 남는다


def test_지문이_같으면_실행기가_정본에_옮기고_잠근다(runner):
    import os
    import stat
    cand = runner["tmp"] / "cand"
    assert runner["main"](["--out", str(cand)]) == 0      # 정본 뿌리 밖의 후보는 대조 없이 낸다
    digest = verify_snapshot(cand)
    stage = runner["tmp"] / "stage"
    assert runner["main"](["--expect-digest", digest, "--staging", str(stage)]) == 0
    assert runner["calls"] == [cand, stage]
    assert verify_snapshot(runner["out"]) == digest and not stage.exists()
    assert all(not (os.stat(p).st_mode & stat.S_IWRITE) for p in runner["out"].iterdir())


@pytest.mark.parametrize("rel", ["cand", "cand/stage", "."], ids=["같다", "목적지_아래", "목적지를_품는다"])
def test_후보_목적지에서도_임시_경로가_겹치면_거부한다(runner, capsys, rel):
    """정본 뿌리 밖에서도 겹침 규칙만으로 거부한다 — 정본 뿌리 규칙에 기대지 않는 경우."""
    out = runner["tmp"] / "cand"
    st = (runner["tmp"] / rel) if rel != "." else runner["tmp"]
    assert runner["main"](["--out", str(out), "--expect-digest", "0" * 64, "--staging", str(st)]) == 2
    assert "같거나 한쪽이 다른 쪽 아래다" in capsys.readouterr().out
    assert runner["calls"] == [] and not out.exists()
