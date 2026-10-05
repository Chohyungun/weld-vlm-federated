"""흡수 곁파일의 불변식을 고정한다 — 미니스펙 10-1 의 T-1 ~ T-9 와 검수가 짚은 자리.

**합성 자료만 쓴다. 동결 자산과 외부 드라이브를 읽지 않는다.**
무거운 것을 만들지 않는다.

각 시험이 무엇을 막는지는 시험 이름과 본문 주석에 적었다. 규약이 박는 쪽에만 있고 읽기를
강제하는 자리가 없으면 규약이 아니다 — 그래서 산출 스키마와 예외까지 시험이 본다.
위반을 막는 장치는 **위반 입력을 심어서** 시험한다. 지금 자료에서 참인 관측만 확인하면
장치가 없어도 통과한다.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from data import absorb
from data.manifest_io import (ABSORB_DEFECT_COLUMNS, ABSORB_IMAGE_COLUMNS,
                              ABSORB_REGION_COLUMNS, ANNOTATION_COLUMNS, MANIFEST_COLUMNS,
                              SnapshotVerificationError, load_snapshot, verify_snapshot,
                              write_snapshot)


def _man_row(image_id, split="train", client="C1", has_defect=True,
             defect_types="porosity", iso_codes="2011", w=1280, h=720, stem=None):
    tail = image_id.split(":")[-1]
    return {c: "" for c in MANIFEST_COLUMNS} | {
        "image_id": image_id, "source": "aihub71761", "rel_path": f"x/{stem or tail}.jpg",
        "sha256": tail.zfill(8), "width_px": w, "height_px": h,
        "modality": "RT", "material": "ST", "has_defect": has_defect,
        "n_defects": 1 if has_defect else 0, "defect_types": defect_types,
        "iso_codes": iso_codes, "src_labels_raw": "", "label_type": "polygon",
        "has_localization": True, "phash_hex": "0" * 16, "group_id": f"g{image_id[-1]}",
        "group_size": 1, "strata_key": "ST|porosity", "split": split, "client": client,
        "eval_subset": "", "thickness_mm": None, "thickness_source": "unavailable",
        "px_per_mm": None, "scale_source": "unavailable", "quality_level": "",
        "ingest_version": "t", "label_map_version": 1, "notes": "",
    }


def _ann_row(image_id, idx=0, box=(10, 10, 20, 20), flags="", iso="2011", valid=True):
    return {c: "" for c in ANNOTATION_COLUMNS} | {
        "ann_id": f"{image_id}#{idx}", "image_id": image_id, "src_label_raw": "기공",
        "defect_type": "porosity", "iso_code": iso, "polygon_json": "[]",
        "bbox_x1_px": box[0], "bbox_y1_px": box[1], "bbox_x2_px": box[2], "bbox_y2_px": box[3],
        "area_px": 1.0, "major_axis_px": 1.0, "minor_axis_px": 1.0, "equiv_diameter_px": 1.0,
        "major_axis_mm": None, "equiv_diameter_mm": None, "geom_valid": valid, "geom_flags": flags,
    }


def _caps():
    return {"snapshot_id": "synthetic_v1",
            "capabilities": {"verdict_mode": "clause_only", "localization": True}}


def _label_map(tmp_path):
    p = tmp_path / "label_map.yaml"
    p.write_text("defect_types:\n  porosity:\n    iso_code: '2011'\n"
                 "  crack:\n    iso_code: '100'\n", encoding="utf-8")
    return p


def _jsonl(tmp_path, records, name="images.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                 encoding="utf-8", newline="\n")
    return p


def _rec(oid, image_id_str, *, w=1280, h=720, defects=(), regions=(), zip_name="TL_RTST_정상.zip",
         isos=None):
    anns = []
    isos = list(isos) if isos is not None else ["2011"] * len(defects)
    for i, b in enumerate(defects):
        anns.append({"ann_idx": i, "annotation_role": "defect", "class_std_iso6520": isos[i],
                     "bbox_xmin": b[0], "bbox_ymin": b[1], "bbox_xmax": b[2], "bbox_ymax": b[3]})
    for i, poly in enumerate(regions):
        anns.append({"ann_idx": len(anns), "annotation_role": "normal_region",
                     "polygon_x": json.dumps([p[0] for p in poly]),
                     "polygon_y": json.dumps([p[1] for p in poly])})
    return {"orig_info_id": oid, "image_id": image_id_str, "modality": "RT",
            "material_nominal": "ST", "material_source": "folder_nominal",
            "width_px": w, "height_px": h, "source_zip": zip_name, "annotations": anns}


@pytest.fixture
def v1(tmp_path):
    """합성 v1 스냅샷. 동결 자산을 읽지 않는다."""
    root = tmp_path / "v1"
    man = pd.DataFrame([
        _man_row("aihub71761:11", split="train"),
        _man_row("aihub71761:12", split="eval", client=""),
        _man_row("aihub71761:13", split="train", has_defect=False, defect_types="", iso_codes=""),
        _man_row("aihub71761:14", split="train", has_defect=False, defect_types="", iso_codes=""),
        # §8-1 을 겨눈 행 — 두 컬럼의 짝이 일부러 뒤집혀 있다.
        _man_row("aihub71761:15", split="train", defect_types="crack;porosity",
                 iso_codes="2011;100"),
        # 평가셋 **정상** 장. 영역 곁파일의 평가셋 격리를 시험하려면 이 장의 영역이 있어야 한다.
        _man_row("aihub71761:16", split="eval", client="", has_defect=False, defect_types="",
                 iso_codes=""),
    ])
    ann = pd.DataFrame([_ann_row("aihub71761:11"), _ann_row("aihub71761:12"),
                        _ann_row("aihub71761:15", 0), _ann_row("aihub71761:15", 1)])
    write_snapshot(root, man, ann, _caps())
    return root


def _build(v1, tmp_path, records, out="v2", **kw):
    return absorb.build(v1, _jsonl(tmp_path, records), tmp_path / out, _label_map(tmp_path), **kw)


#: 여러 시험이 쓰는 입력 — 평가셋 결함 장(12), 타일 정상(13), 원본 프레임 정상(14), 평가셋 정상(16) 을 덮는다.
FULL = [
    ("11", dict(defects=[(10, 10, 20, 20)])),
    ("12", dict(defects=[(10, 10, 20, 20)])),
    ("13", dict(regions=[[(1, 1), (2, 2), (3, 3)]])),
    ("14", dict(w=4000, h=1272, regions=[[(5, 5), (6, 6), (7, 7)]])),
    ("16", dict(regions=[[(4, 4), (5, 5), (6, 6)]])),
]


def _full(v1, tmp_path, out="v2", **kw):
    return _build(v1, tmp_path, [_rec(o, f"RT_ST_01_{o}", **a) for o, a in FULL], out=out, **kw)


def _rewrite(src: Path, dst: Path, **frames) -> Path:
    """검증된 스냅샷을 읽어 곁파일 하나를 바꿔 **새로 잠근** 사본을 낸다. 다른 손이 쓴 파일을 흉내 낸다."""
    s = load_snapshot(src)
    parts = {"absorb_image": s.absorb_image, "absorb_defect": s.absorb_defect,
             "absorb_region": s.absorb_region} | frames
    write_snapshot(dst, s.manifest, s.annotations, s.capabilities, tiles=s.tiles, **parts)
    return dst


# --- T-1 평가셋 격리 -------------------------------------------------------------------

def test_T1_평가셋이_학습_산출에_한_줄도_남지_않는다(v1, tmp_path):
    _full(v1, tmp_path)
    snap = load_snapshot(tmp_path / "v2")
    ev = set(snap.manifest.loc[snap.manifest["split"] == "eval", "image_id"])
    # 전제를 먼저 단언한다 — 입력에 평가셋 행이 없으면 이 시험은 빈 채로 통과한다.
    for name in ("absorb_image", "absorb_defect", "absorb_region"):
        assert set(getattr(snap, name)["image_id"]) & ev, f"{name} 입력에 평가셋 행이 없다"
    out = absorb.export_for_training(tmp_path / "v2")
    for name, frame in out.items():
        assert not (set(frame["image_id"]) & ev), f"{name} 에 평가셋이 남았다"


def test_T1_거르기가_망가지면_단언이_예외를_던진다(v1, tmp_path, monkeypatch):
    # 거르는 코드를 일부러 무력화한다. **경고가 아니라 예외**여야 한다.
    _full(v1, tmp_path)
    snap = absorb.load_absorbed(tmp_path / "v2")   # 읽기 쪽 검사는 망가뜨리기 전에 통과시킨다
    monkeypatch.setattr(absorb, "load_absorbed", lambda root: snap)
    monkeypatch.setattr(pd.Series, "isin", lambda self, other: pd.Series(True, index=self.index))
    with pytest.raises(absorb.AbsorptionError, match="평가셋"):
        absorb.export_for_training(tmp_path / "v2")


# --- T-2 조인 키 둘이 **서로 다른 시험**이어야 한다 -------------------------------------

def test_T2_원본식별자와_파일명꼬리가_다른_결과를_낸다(v1, tmp_path):
    """정리된 파일 쪽. 꼬리와 `orig_info_id` 가 다른 줄을 심는다. 두 결합이 **같은 결과면 실패**다."""
    recs = [_rec("11", "RT_ST_01_99999", defects=[(10, 10, 20, 20)])]   # 꼬리 99999 ≠ oid 11
    _build(v1, tmp_path, recs)
    snap = load_snapshot(tmp_path / "v2")
    ai = snap.absorb_image.set_index("image_id")

    by_oid = {absorb.num_key(i) for i in ai.index[ai["covered"].fillna(False)]}
    by_tail = {r["image_id"].rsplit("_", 1)[-1] for r in recs}
    assert by_oid == {"11"}, "채택 키는 orig_info_id 다"
    assert by_oid != by_tail, "두 결합이 같은 결과를 냈다 — 시험이 아무것도 가르지 못한다"


def test_T2_우리_쪽_파일명_어간으로_잇지_않는다(tmp_path):
    """우리 쪽. `rel_path` 어간과 `image_id` 숫자부가 **엇갈린** 두 장을 심는다(§4-1).

    고정물의 어간이 늘 숫자부와 같으면 "숫자부로 잇기" 와 "어간으로 잇기" 가 가려지지 않는다.
    """
    root = tmp_path / "v1s"
    man = pd.DataFrame([_man_row("aihub71761:61", stem="62"),
                        _man_row("aihub71761:62", stem="61", has_defect=False,
                                 defect_types="", iso_codes="")])
    write_snapshot(root, man, pd.DataFrame([_ann_row("aihub71761:61")]), _caps())
    absorb.build(root, _jsonl(tmp_path, [_rec("61", "RT_ST_01_61", defects=[(10, 10, 20, 20)])]),
                 tmp_path / "v2s", _label_map(tmp_path))
    ai = load_snapshot(tmp_path / "v2s").absorb_image.set_index("image_id")
    covered = set(ai.index[ai["covered"].fillna(False)])
    by_stem = {iid for iid, rp in zip(man["image_id"], man["rel_path"]) if Path(rp).stem == "61"}
    assert covered == {"aihub71761:61"}, "숫자부로 잇지 않았다"
    assert covered != by_stem, "어간 결합과 같은 결과다 — 시험이 가르지 못한다"


# --- T-3 덮임 회계 ---------------------------------------------------------------------

def test_T3_회계가_손으로_센_값과_같고_안_덮은_행은_비어_있다(v1, tmp_path):
    acct = _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)]),
                                 _rec("13", "RT_ST_01_13", regions=[[(1, 1), (2, 2), (3, 3)]])])
    assert acct["covered_images"] == 2 and acct["total_images"] == 6
    ai = load_snapshot(tmp_path / "v2").absorb_image.set_index("image_id")
    un = ai[~ai["covered"].fillna(False)]
    assert (un["source_zip"] == "").all() and (un["frame_match"] == "").all()
    assert un["orig_width_px"].isna().all(), "안 덮은 행에 값을 지어 넣었다"


def test_결측_참여자_칸은_평가셋과_같은_집합일_때만_eval_이다(tmp_path):
    """이름 붙이기가 아니라 확인이다. 평가셋이 아닌 장의 참여자가 비면 멈춘다."""
    root = tmp_path / "v1c"
    man = pd.DataFrame([_man_row("aihub71761:71", split="eval", client=""),
                        _man_row("aihub71761:72", split="train", client="")])   # 어긋난 장
    write_snapshot(root, man, pd.DataFrame([_ann_row("aihub71761:71"), _ann_row("aihub71761:72")]),
                   _caps())
    with pytest.raises(absorb.AbsorptionError, match="같은 집합이 아니다"):
        absorb.build(root, _jsonl(tmp_path, []), tmp_path / "v2c", _label_map(tmp_path))


# --- T-4 좌표 프레임 -------------------------------------------------------------------

def test_T4_프레임이_비면_안_되고_원본프레임은_학습산출에_없다(v1, tmp_path):
    _full(v1, tmp_path)
    snap = load_snapshot(tmp_path / "v2")
    reg = snap.absorb_region
    assert (reg["frame"] != "").all(), "프레임이 빈 행이 있다"
    assert set(reg["frame"]) == {"tile", "orig"}
    orig_ids = set(reg.loc[reg["frame"] == "orig", "region_id"])
    assert orig_ids
    # 정상영역은 프레임과 무관하게 학습 산출에 나가지 않는다 — 있음이 곧 정상이라는 라벨이다.
    assert "absorb_region" not in absorb.export_for_training(tmp_path / "v2")


def test_T4_결함_곁파일은_전량_타일_프레임이다(v1, tmp_path):
    """지금 자료의 관측. 아래 위반 입력 시험이 이것을 **불변식**으로 만든다."""
    _full(v1, tmp_path)
    snap = load_snapshot(tmp_path / "v2")
    ai = snap.absorb_image.set_index("image_id")
    for iid in snap.absorb_defect.loc[snap.absorb_defect["raw_source"] == "absorbed", "image_id"]:
        assert ai.loc[iid, "frame_match"] == "same", "결함 주석이 원본 프레임 장에 있다"


@pytest.mark.parametrize("empty_box", [False, True], ids=["상자있음", "우리상자빔"])
def test_T4_결함이_원본_프레임_장에_있으면_멈춘다(tmp_path, empty_box):
    """크기가 다른 장에 결함을 심는다. **우리 상자가 빈 행**은 상자 비교가 없어 따로 심는다.

    빠진 라벨 아카이브 여섯 중 다섯이 결함 아카이브라, 완성판에서 처음 이 입력을 만날 수 있다.
    """
    root = tmp_path / "v1f"
    man = pd.DataFrame([_man_row("aihub71761:81")])
    box = (None, None, None, None) if empty_box else (10, 10, 20, 20)
    write_snapshot(root, man, pd.DataFrame([_ann_row("aihub71761:81", box=box, valid=not empty_box)]),
                   _caps())
    with pytest.raises(absorb.AbsorptionError, match="원본 프레임"):
        absorb.build(root, _jsonl(tmp_path, [_rec("81", "RT_ST_01_81", w=4000, h=1272,
                                                  defects=[(10, 10, 20, 20)])]),
                     tmp_path / "v2f", _label_map(tmp_path))
    assert not (tmp_path / "v2f").exists()


# --- T-5 학습 산출은 허용 목록의 칸만 --------------------------------------------------

def test_T5_학습_산출_칸이_허용_목록과_같다(v1, tmp_path):
    """두 이름이 **없음**만 보면 같은 정보가 다른 칸으로 나가도 통과한다. 칸 집합이 **같음**을 본다."""
    _full(v1, tmp_path)
    out = absorb.export_for_training(tmp_path / "v2")
    for name, frame in out.items():
        assert tuple(frame.columns) == absorb.TRAINING_COLUMNS[name], name
    leaked = {"orig_width_px", "orig_height_px", "frame_match", "source_zip", "covered",
              "material_source", "raw_source", "absorb_note"}
    for name, frame in out.items():
        assert not (leaked & set(frame.columns)), f"{name}: 지름길 칸이 나갔다"


def test_T5_곁파일에_새_칸을_심어도_학습_산출에_나가지_않는다(v1, tmp_path, monkeypatch):
    """허용 목록이라 이름을 몰라도 빠진다. 금지 목록이었다면 새 칸이 그대로 샌다."""
    _full(v1, tmp_path)
    snap = absorb.load_absorbed(tmp_path / "v2")
    planted = snap.absorb_image.assign(orig_aspect_ratio=1.0, defect_hint="결함")
    monkeypatch.setattr(absorb, "load_absorbed",
                        lambda root: dataclasses.replace(snap, absorb_image=planted))
    out = absorb.export_for_training(tmp_path / "v2")
    assert not ({"orig_aspect_ratio", "defect_hint"} & set(out["absorb_image"].columns))


def test_T5_허용_목록이_금지_조각을_담으면_멈춘다(v1, tmp_path, monkeypatch):
    _full(v1, tmp_path)
    monkeypatch.setitem(absorb.TRAINING_COLUMNS, "absorb_image", ("image_id", "source_zip"))
    with pytest.raises(absorb.AbsorptionError, match="허용 목록"):
        absorb.export_for_training(tmp_path / "v2")


def test_T5_허용_목록은_곁파일_스키마_안에_있고_규칙을_지킨다():
    absorb.check_training_allowlist()
    schema = {"absorb_image": ABSORB_IMAGE_COLUMNS, "absorb_defect": ABSORB_DEFECT_COLUMNS,
              "absorb_region": ABSORB_REGION_COLUMNS}
    assert set(absorb.TRAINING_COLUMNS) == set(schema) - {"absorb_region"}, "정상영역은 학습 산출에 없다"
    assert {"orig_width_px", "orig_height_px"} <= absorb.BLOCKED_COLUMNS, "명세 §7-2 의 두 이름"
    assert set(ABSORB_REGION_COLUMNS) - {"image_id"} <= absorb.BLOCKED_COLUMNS


# --- T-6 ISO 짝 ------------------------------------------------------------------------

def test_T6_iso_코드가_결함종류_순서와_짝이_맞는다(v1, tmp_path):
    acct = _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])])
    man = load_snapshot(tmp_path / "v2").manifest
    lm = {"porosity": "2011", "crack": "100"}
    for t, c in zip(man["defect_types"].fillna(""), man["iso_codes"].fillna("")):
        ts = [x for x in t.split(";") if x]
        cs = [x for x in c.split(";") if x]
        assert [lm[x] for x in ts] == cs, f"짝이 어긋났다: {t} ↔ {c}"
    assert acct["iso_pair_fixed"] >= 1, "일부러 심은 어긋난 행을 고치지 않았다"


# --- T-7 지문 재현 ---------------------------------------------------------------------

def test_T7_같은_입력은_같은_지문을_낸다(v1, tmp_path):
    a = _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])], out="v2a")
    b = _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])], out="v2b")
    assert a["snapshot_digest"] == b["snapshot_digest"]


def test_T7_멤버_순서를_바꾸면_지문이_달라진다(v1, tmp_path):
    _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])])
    snap = tmp_path / "v2" / "SNAPSHOT.sha256"
    lines = snap.read_text(encoding="utf-8").splitlines()
    body = [x for x in lines if not x.startswith("#")]
    import hashlib
    same = hashlib.sha256(("\n".join(body) + "\n").encode()).hexdigest()
    swapped = hashlib.sha256(("\n".join([body[-1], *body[:-1]]) + "\n").encode()).hexdigest()
    assert same != swapped, "순서가 지문에 영향을 주지 않는다"


# --- T-8 옛 스냅샷 하위호환 -------------------------------------------------------------

def test_T8_곁파일이_없는_스냅샷이_그대로_통과한다(v1):
    verify_snapshot(v1)
    snap = load_snapshot(v1)
    assert snap.absorb_image is None and snap.absorb_defect is None and snap.absorb_region is None


# --- T-9 분할 불변 ---------------------------------------------------------------------

def test_T9_분할과_묶음_컬럼이_v1과_전_행_동일하다(v1, tmp_path):
    _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])])
    a = load_snapshot(v1).manifest.set_index("image_id")
    b = load_snapshot(tmp_path / "v2").manifest.set_index("image_id")
    for col in ("split", "client", "eval_subset", "group_id", "group_size", "strata_key",
                "phash_hex", "sha256"):
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)


def test_v2가_v1과_다른_열은_iso_codes_하나뿐이고_회계도_그렇게_센다(v1, tmp_path):
    acct = _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])])
    a = load_snapshot(v1).manifest.set_index("image_id")
    b = load_snapshot(tmp_path / "v2").manifest.set_index("image_id")
    diff = [c for c in a.columns if not a[c].equals(b[c])]
    assert diff == ["iso_codes"], f"달라진 열: {diff}"
    assert acct["changed_columns_vs_v1"] == ["iso_codes"], "쓰기 전 검사가 볼 값이 파일과 다르다"


# --- 결함 곁파일의 행 범위와 빈 상자 (§3-2 · §8-2) -----------------------------------------

def test_결함_곁파일은_모든_주석에_한_행이고_안_덮은_주석은_비어_있다(v1, tmp_path):
    acct = _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])])
    snap = load_snapshot(tmp_path / "v2")
    d = snap.absorb_defect.set_index("ann_id")
    assert set(d.index) == set(snap.annotations["ann_id"]), "주석과 행 집합이 다르다"
    un = d[d["image_id"] != "aihub71761:11"]
    assert len(un) == 3 and (un["raw_source"] == "").all() and un["raw_bbox_x1_px"].isna().all()
    assert acct["defect_rows"] == {"total": 4, "absorbed": 1, "not_covered": 3}


def test_빈_상자_행을_지우지_않고_원좌표를_복구_가능하게_담는다(tmp_path):
    """85 는 동결본 전체의 빈 상자 행이고 54 는 그중 되찾은 행이다. 분모가 회계에 있어야 한다."""
    root = tmp_path / "v1b"
    man = pd.DataFrame([_man_row("aihub71761:21"), _man_row("aihub71761:22")])
    empty = (None, None, None, None)
    ann = pd.DataFrame([_ann_row("aihub71761:21", 0, box=empty, flags="self_intersect;zero_area", valid=False),
                        _ann_row("aihub71761:22", 0, box=empty, flags="self_intersect;zero_area", valid=False)])
    write_snapshot(root, man, ann, _caps())
    acct = absorb.build(root, _jsonl(tmp_path, [_rec("21", "RT_ST_01_21", defects=[(-1, 0, 30, 30)])]),
                        tmp_path / "v2c", _label_map(tmp_path))
    snap = absorb.load_absorbed(tmp_path / "v2c")
    assert len(snap.annotations) == 2, "빈 상자 행을 지웠다"
    d = snap.absorb_defect.set_index("ann_id").loc["aihub71761:21#0"]
    assert d["raw_source"] == "absorbed" and int(d["raw_bbox_x1_px"]) == -1
    assert acct["missing_bbox_rows"] == 2, "안 덮은 장의 빈 상자가 분모에서 빠졌다"
    assert acct["recovered_from_absorb"] == 1


# --- 주석 짝짓기 (명세 §4 는 장 단위만 정했다) -------------------------------------------

def test_짝지을_주석_수가_다르면_멈춘다(v1, tmp_path):
    with pytest.raises(absorb.AbsorptionError, match="주석 수가 다르다"):
        _build(v1, tmp_path, [_rec("15", "RT_ST_01_15", defects=[(10, 10, 20, 20)])])


def test_짝지은_주석의_ISO_코드가_다르면_멈춘다(v1, tmp_path):
    """위치로만 짝지으면 우리 상자가 빈 행은 비교 없이 짝지어진다. 코드 대조가 그 행까지 본다."""
    with pytest.raises(absorb.AbsorptionError, match="ISO 코드가 다르다"):
        _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)], isos=["100"])])


# --- 잠정판의 재발행 이력과 읽기 경고 (§9-2) ---------------------------------------------

def test_같은_경로에_다시_내면_앞_판의_지문이_이력에_남는다(v1, tmp_path):
    first = _full(v1, tmp_path, out="v2h")
    second = _full(v1, tmp_path, out="v2h")
    hist = second["digest_history"]
    assert [e["digest"] for e in hist] == [first["snapshot_digest"]]
    assert set(hist[0]) >= {"status", "digest", "at", "archives_present"}
    caps = load_snapshot(tmp_path / "v2h").capabilities
    assert caps["absorption"]["digest_history"] == hist, "이력이 산출 파일에 실리지 않았다"


def test_장치_전에_사라진_판은_출처와_함께_맨_앞에_둔다(v1, tmp_path):
    seed = [{"status": "provisional", "digest": "a" * 64, "at": None, "archives_present": 1,
             "note": "시험용"}]
    first = _full(v1, tmp_path, out="v2s", seed_history=seed)
    assert [e["digest"] for e in first["digest_history"]] == ["a" * 64]
    second = _full(v1, tmp_path, out="v2s", seed_history=seed)
    assert [e["digest"] for e in second["digest_history"]] == ["a" * 64, first["snapshot_digest"]]


def test_검증되지_않는_앞_판은_덮지_않는다(v1, tmp_path):
    _full(v1, tmp_path, out="v2t")
    target = tmp_path / "v2t" / "absorb_region.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    before = target.read_bytes()
    with pytest.raises(SnapshotVerificationError):
        _full(v1, tmp_path, out="v2t")
    assert target.read_bytes() == before, "무엇을 덮는지 모르는 채로 덮었다"


def test_이력_항목의_형식이_다르면_멈춘다(v1, tmp_path):
    with pytest.raises(absorb.AbsorptionError, match="digest_history"):
        _full(v1, tmp_path, out="v2x", seed_history=[{"digest": "짧다"}])


def test_잠정판을_읽으면_로그에_남긴다(v1, tmp_path, caplog):
    _full(v1, tmp_path)
    with caplog.at_level(logging.WARNING, logger="data.absorb"):
        absorb.load_absorbed(tmp_path / "v2")
    assert any("논문에 싣지 않는다" in r.getMessage() for r in caplog.records)


def test_완성판은_경고하지_않는다(v1, tmp_path, caplog):
    _full(v1, tmp_path, dataset_root=_dataset(tmp_path, ["TL_RTST_정상.zip"]))
    with caplog.at_level(logging.WARNING, logger="data.absorb"):
        snap = absorb.load_absorbed(tmp_path / "v2")
    assert snap.capabilities["absorption"]["status"] == "complete"
    assert not [r for r in caplog.records if r.name == "data.absorb"]


# --- 읽기 쪽이 막는 것 (§6-2 1단계 · 값 공간) ---------------------------------------------

def test_이미_읽은_Snapshot_은_받지_않는다(v1, tmp_path):
    """`verify=False` 로 읽은 객체가 그대로 통과하면 함수 안의 검증이 뜻이 없다."""
    _full(v1, tmp_path)
    unverified = load_snapshot(tmp_path / "v2", verify=False)
    # 경로 API 는 `Path(Snapshot)` 에서도 TypeError 를 낸다. 그 우연에 기대지 않게 문구까지 본다.
    with pytest.raises(TypeError, match="검증 없이"):
        absorb.export_for_training(unverified)


def test_잠금_밖_곁파일은_받지_않는다(v1, tmp_path):
    """`load_snapshot` 은 곁파일을 디스크 기준으로 읽는다. 잠금에 없는 파일이 검증 없이 들어온다."""
    _full(v1, tmp_path)
    s = load_snapshot(tmp_path / "v2")
    bare = tmp_path / "bare"
    write_snapshot(bare, s.manifest, s.annotations, s.capabilities)      # 곁파일 없이 잠근다
    (bare / "absorb_image.csv").write_bytes((tmp_path / "v2" / "absorb_image.csv").read_bytes())
    load_snapshot(bare)                                                  # 기존 경로는 통과한다
    with pytest.raises(absorb.AbsorptionError, match="잠금 목록에 없다"):
        absorb.load_absorbed(bare)


def test_바뀐_곁파일은_검증에서_걸린다(v1, tmp_path):
    _full(v1, tmp_path)
    p = tmp_path / "v2" / "absorb_defect.csv"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(SnapshotVerificationError):
        absorb.export_for_training(tmp_path / "v2")


def _at(df, **eq):
    """조건에 맞는 첫 행의 색인."""
    m = pd.Series(True, index=df.index)
    for k, v in eq.items():
        m &= (df[k] == v).fillna(False).astype(bool)
    return df.index[m][0]


def _m_image(fn):
    def go(s):
        img = s.absorb_image.copy()
        return {"absorb_image": fn(img)}
    return go


def _m_defect(fn):
    def go(s):
        d = s.absorb_defect.copy()
        return {"absorb_defect": fn(d)}
    return go


def _m_region(fn):
    def go(s):
        r = s.absorb_region.copy()
        return {"absorb_region": fn(r)}
    return go


def _set(df, idx, **vals):
    for k, v in vals.items():
        df.loc[idx, k] = v
    return df


_A11 = "aihub71761:11#0"          # 덮인 결함 행 — 우리 상자와 원좌표가 같아 `none`
_R13 = "aihub71761:13#r0"         # 타일 프레임 정상영역

#: `_check_sidecars` 의 검사 열일곱. (이름, 바꾸는 법, 그 검사의 문구). 앞 검사를 건드리지 않는 입력만 심는다.
READ_CHECKS = [
    ("image_장집합", _m_image(lambda d: d.drop(index=_at(d, covered=False))),
     "absorb_image: 매니페스트와 장 집합이 다르다"),
    ("image_frame_match_값공간", _m_image(lambda d: _set(d, _at(d, covered=True), frame_match="panorama")),
     "absorb_image: `frame_match` 가 값 공간 밖이다"),
    ("image_안덮은행에_값", _m_image(lambda d: _set(d, _at(d, covered=False), source_zip="X.zip")),
     "absorb_image: 안 덮은 행에 값이 있다"),
    ("image_덮인행_빈_frame_match", _m_image(lambda d: _set(d, _at(d, image_id="aihub71761:11"), frame_match="")),
     "absorb_image: 덮인 행의 `frame_match` 가 비어 있다"),
    ("defect_행집합", _m_defect(lambda d: d.drop(index=_at(d, raw_source=""))),
     "absorb_defect: 주석과 행 집합이 다르다"),
    ("defect_clip_kind_값공간", _m_defect(lambda d: _set(d, _at(d, ann_id=_A11), clip_kind="trimmed")),
     "absorb_defect: `clip_kind` 가 값 공간 밖이다"),
    ("defect_raw_source_값공간", _m_defect(lambda d: _set(d, _at(d, ann_id=_A11), raw_source="copied")),
     "absorb_defect: `raw_source` 가 값 공간 밖이다"),
    ("defect_원좌표_유무", _m_defect(lambda d: _set(d, _at(d, ann_id=_A11), raw_source="")),
     "absorb_defect: `raw_source` 와 원좌표의 유무가 어긋난다"),
    ("defect_applied_kind", _m_defect(lambda d: _set(d, _at(d, ann_id=_A11), clip_applied=True)),
     "absorb_defect: `clip_applied` 와 `clip_kind` 가 어긋난다"),
    # I-4 의 읽는 쪽 절반. 결함이 붙은 장을 원본 프레임 장으로 바꿔 놓는다.
    ("defect_원본프레임_장", _m_image(lambda d: _set(d, _at(d, image_id="aihub71761:11"), frame_match="tile_of_orig")),
     "absorb_defect: 원좌표가 원본 프레임 장에 있다"),
    ("defect_image_id_불일치", _m_defect(lambda d: _set(d, _at(d, ann_id=_A11), image_id="aihub71761:13")),
     "absorb_defect: `image_id` 가 주석의 것과 다르다"),
    ("defect_none인데_다름", _m_defect(lambda d: _set(d, _at(d, ann_id=_A11), raw_bbox_x1_px=500)),
     "absorb_defect: `none` 인데 우리 상자와 원좌표가 다르다"),
    ("region_frame_값공간", _m_region(lambda d: _set(d, _at(d, region_id=_R13), frame="panorama")),
     "absorb_region: `frame` 이 비었거나 값 공간 밖이다"),
    ("region_usable", _m_region(lambda d: _set(d, _at(d, region_id=_R13), usable_in_tile_frame=False)),
     "absorb_region: `usable_in_tile_frame` 이 `frame` 과 어긋난다"),
    ("region_꼭짓점수", _m_region(lambda d: _set(d, _at(d, region_id=_R13), n_vertices=99)),
     "absorb_region: `n_vertices` 가 좌표 수와 다르다"),
    ("region_결함장", _m_region(lambda d: _set(d, _at(d, region_id=_R13), image_id="aihub71761:11")),
     "absorb_region: 결함 장에 정상영역이 있다"),
    ("region_크기대조", _m_region(lambda d: _set(d, _at(d, region_id=_R13), frame="orig", usable_in_tile_frame=False)),
     "absorb_region: `frame` 이 원천의 프레임 규칙과 어긋난다"),
    # 원천을 값이 가른다 — 흡수 판에 원본 라벨 원천의 재질 근거 값이 섞이면 멈춘다.
    ("image_재질근거_원천", _m_image(lambda d: _set(d, _at(d, image_id="aihub71761:11"), material_source="label_json")),
     "absorb_image: `material_source` 가 이 원천의 값이 아니다"),
]


@pytest.mark.parametrize("mutate, message", [(m, msg) for _n, m, msg in READ_CHECKS],
                         ids=[n for n, _m, _msg in READ_CHECKS])
def test_읽는_쪽_검사를_하나씩_심는다(v1, tmp_path, mutate, message):
    """다른 손이 쓴 곁파일을 흉내 낸다. 각 입력이 **그 검사의 문구로** 걸려야 한다 — 앞선 다른 검사가
    대신 잡으면 그 검사는 시험되지 않은 채 남는다.
    """
    _full(v1, tmp_path)
    bad = _rewrite(tmp_path / "v2", tmp_path / "bad", **mutate(load_snapshot(tmp_path / "v2")))
    with pytest.raises(absorb.AbsorptionError, match=re.escape(message)):
        absorb.load_absorbed(bad)


def test_읽는_쪽_검사의_목록이_코드와_같다():
    """시험 목록과 코드의 검사가 어긋나면 새 검사가 시험 없이 들어온다.

    흡수 판에서 서는 검사는 이 파일이, 원본 라벨 판에서만 서는 검사와 원천 값 검사는 `tests/test_rawmeta.py` 가 심는다.
    """
    from tests.test_rawmeta import RAW_READ_CHECKS
    src = Path(absorb.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _check_sidecars"):src.index("def load_absorbed")]
    calls = re.findall(r'_fail\(\s*"(absorb_\w+|absorption)",\s*"([^"]+)"', body)
    in_code = {f"{a}: {m}" for a, m in calls}
    planted = {msg for _n, _m, msg in READ_CHECKS} | {msg for _n, _m, msg in RAW_READ_CHECKS} \
        | {"absorption: `source` 가 값 공간 밖이다"}           # test_rawmeta 의 원천 값 시험이 심는다
    assert len(calls) == len(in_code) == len(READ_CHECKS) + len(RAW_READ_CHECKS) + 1 == 21
    assert in_code == planted


# --- 곁파일을 읽는 자리는 하나다 -------------------------------------------------------

def test_곁파일을_읽는_자리는_흡수_모듈_하나다():
    """경로가 하나라는 규약과 잠정판 경고는 **다른 모듈이 곁파일을 직접 읽지 않을 때만** 선다."""
    root = Path(absorb.__file__).resolve().parents[1]
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
                       cwd=root, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        pytest.skip("git 으로 파일 목록을 낼 수 없다")
    files = [f for f in r.stdout.splitlines() if f]
    pat = re.compile(r"\.absorb_(image|defect|region)\b|\bread_absorb\s*\(")
    allow = {"data/absorb.py", "data/manifest_io.py", "tests/test_absorb.py", "tests/test_rawmeta.py",
             "scripts/crosscheck/diag_absorb_gap.py",      # 진단은 첫 판을 보려고 쓴 일회성이다
             "scripts/crosscheck/xc_rawmeta_team.py",      # 원본 라벨 판과 정리된 파일의 교차 검산 — `load_absorbed` 로 연다
             "scripts/crosscheck/rawmeta_clipkind_table.py"}   # 6절 결정 규칙의 표 — 평가 행을 넣지 않는다
    hits = {f for f in files if pat.search((root / f).read_text(encoding="utf-8", errors="replace"))}
    # 검사 범위가 비어 0 이 나오는 것을 막는다 — 흡수 모듈 자신은 반드시 걸려야 한다.
    assert "data/absorb.py" in hits and len(files) > 50
    assert not (hits - allow), f"곁파일을 직접 읽는 모듈: {sorted(hits - allow)}"


# --- 회계의 모수와 그레인 (첫 실물 실행이 드러낸 자리) ---------------------------------
#
# 실물 회계와 두 검산의 수가 다섯 자리에서 갈렸고 셋이 값이 아니라 **세는 단위**였다.
# 아래는 그 셋이 다시 뭉개지지 않게 고정한다. 값이 맞아도 단위가 다르면 인용이 틀어진다.

def test_결측_참여자를_eval_칸으로_모으고_NA_표기를_키로_쓰지_않는다(v1, tmp_path):
    """`str(pd.NA)` 는 `"<NA>"` 로 참이라 `or` 로 걸러지지 않는다.

    값(4,105/12,461)이 맞아도 키가 `<NA>` 면 받는 쪽이 명세의 `eval` 칸을 못 찾는다.
    """
    acct = _build(v1, tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)])])
    by_client = acct["coverage_by_client"]
    assert "eval" in by_client, f"결측을 모을 칸이 없다: {sorted(by_client)}"
    assert "<NA>" not in by_client, "pandas 결측 표기가 회계의 키로 새어 나왔다"
    assert by_client["eval"]["total"] == 2


def test_결측을_모을_칸이_없는_축은_지어_넣지_않고_멈춘다(tmp_path):
    root = tmp_path / "v1na"
    man = pd.DataFrame([_man_row("aihub71761:31")])
    man.loc[:, "material"] = pd.NA
    write_snapshot(root, man, pd.DataFrame([_ann_row("aihub71761:31")]), _caps())
    with pytest.raises(absorb.AbsorptionError, match="결측"):
        absorb.build(root, _jsonl(tmp_path, []), tmp_path / "v2na", _label_map(tmp_path))


def _dataset(tmp_path, names):
    d = tmp_path / "ds" / "02.라벨링데이터"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_bytes(b"")          # 이름만 센다. 열지도 풀지도 않는다
    return tmp_path / "ds"


def test_아카이브_모수_둘을_한_이름에_담지_않는다(tmp_path, v1):
    """`present` 는 정리된 파일 전체 줄이고, 덮인 장만 보면 더 작다.

    실물에서 21 과 14 였다. 한 이름에 담으면 받는 쪽이 어느 모수를 본 것인지 알 수 없다.
    """
    vt = _rec("99", "VT_ST_01_99", zip_name="TL_VTST_정상.zip") | {"modality": "VT"}
    acct = absorb.build(
        v1, _jsonl(tmp_path, [_rec("11", "RT_ST_01_11", defects=[(10, 10, 20, 20)]), vt]),
        tmp_path / "v2zip", _label_map(tmp_path),
        dataset_root=_dataset(tmp_path, ["TL_RTST_정상.zip", "TL_VTST_정상.zip",
                                         "TL_RTAL_정상.zip"]))
    a = acct["archives"]
    assert a["present"] == 2, "덮이지 않은 줄의 아카이브가 모수에서 빠졌다"
    assert a["present_in_covered"] == 1, "덮인 장의 모수가 전체 줄과 섞였다"
    assert a["present"] != a["present_in_covered"], "두 모수가 구별되지 않는다"
    assert a["missing_vt"] == 0 and a["missing_from_covered"]["vt"] == 1


def test_자르기_회계를_행과_장_두_그레인으로_함께_적는다(tmp_path):
    """한 장에 경계 자르기 행이 둘이면 행 2 · 장 1 이다.

    실물에서 행 43 · 장 39 였고 앞선 검산의 39 는 장이었다. 한 그레인만 적으면
    106 과 101 을 같은 것으로 견주게 된다.
    """
    root = tmp_path / "v1g"
    man = pd.DataFrame([_man_row("aihub71761:41", defect_types="porosity;porosity",
                                 iso_codes="2011;2011")])
    ann = pd.DataFrame([_ann_row("aihub71761:41", 0, box=(0, 0, 30, 30)),
                        _ann_row("aihub71761:41", 1, box=(1200, 100, 1280, 200))])
    write_snapshot(root, man, ann, _caps())
    acct = absorb.build(root, _jsonl(tmp_path, [
        _rec("41", "RT_ST_01_41", defects=[(-1, 0, 30, 30), (1200, 100, 1300, 200)])]),
        tmp_path / "v2g", _label_map(tmp_path))
    assert acct["clip_kinds"]["boundary"] == 2, "행 그레인이 장 수로 뭉개졌다"
    assert acct["clip_by_image"]["boundary"] == 1, "장 그레인이 없다"


def test_정상영역_회계도_영역과_장을_함께_적는다(v1, tmp_path):
    """명세 §3-4 표의 5,885·1,379 는 장수이고 곁파일 행수는 5,925·1,379 였다."""
    acct = _build(v1, tmp_path, [
        _rec("13", "RT_ST_01_13", w=4000, h=1272,
             regions=[[(0, 0), (9, 0), (9, 9)], [(20, 20), (29, 20), (29, 29)]])])
    assert acct["region_frames"]["orig"] == 2, "영역 그레인이 장 수로 뭉개졌다"
    assert acct["region_images_by_frame"]["orig"] == 1, "장 그레인이 없다"
    assert acct["regions_per_image"][2] == 1


def test_설명되지_않는_상자_차이를_none_으로_적지_않고_멈춘다(tmp_path):
    """`clip_kind` 값 공간이 폐집합이라 설명 못 한 차이를 `none` 에 담으면 사라진다.

    명세는 상자 차이 전건이 문서화된 정책으로 설명된다고 단언한다. 그 단언이 깨지는
    입력에서 조용히 통과하면 단언을 읽는 자리가 없어진다.
    """
    root = tmp_path / "v1u"
    man = pd.DataFrame([_man_row("aihub71761:51")])
    ann = pd.DataFrame([_ann_row("aihub71761:51", 0, box=(10, 10, 20, 20))])
    write_snapshot(root, man, ann, _caps())
    with pytest.raises(absorb.AbsorptionError, match="설명할 수 없다"):
        absorb.build(root, _jsonl(tmp_path, [
            _rec("51", "RT_ST_01_51", defects=[(100, 100, 200, 200)])]),
            tmp_path / "v2u", _label_map(tmp_path))


def test_회계가_기대와_다르면_쓰기_전에_멈춘다(v1, tmp_path):
    """낸 뒤에 걸러도 이미 인용할 수 있는 파일이 남는다. 첫 실행이 그랬다."""
    out = tmp_path / "v2chk"
    with pytest.raises(absorb.AbsorptionError, match="쓰지 않았다"):
        absorb.build(v1, _jsonl(tmp_path, []), out, _label_map(tmp_path),
                     check=lambda a: ["일부러 어긋낸다"])
    assert not out.exists(), "멈췄다고 적었는데 스냅샷이 남았다"
