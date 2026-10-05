"""전수 메타데이터 독립 검산(`scripts.crosscheck.rawmeta_independent`)의 평가 행 판정 — 합성 입력.

- 평가 장의 ISO 불일치 · 순서 키만 다른 장은 "같지 않음" 이다(명세 5-2). 평가 장은 한 비트와 장 수만 낸다.
- 후보 판과의 열별 집계에는 학습 · 검증 행만 든다(명세 5-3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crosscheck import rawmeta_independent as R


def _res(iso_ours: str):
    """평가 장 하나(결함 둘) · 학습 장 하나(결함 하나)의 합성 산출."""
    man_by = {"aihub71761:1": {"split": "eval"}, "aihub71761:2": {"split": "train"}}
    recs = {1: {"w": 1280, "h": 720, "is_normal": False}, 2: {"w": 1280, "h": 720, "is_normal": False}}

    def d(iid, seq, box, iso):
        return {"ann_id": f"aihub71761:{iid}#{seq}", "image_id": f"aihub71761:{iid}", "raw_bbox_x1_px": box[0],
                "raw_bbox_y1_px": box[1], "raw_bbox_x2_px": box[2], "raw_bbox_y2_px": box[3], "clip_kind": "none",
                "_iso": iso}
    def_rows = [d(1, 0, (1, 1, 5, 5), iso_ours), d(1, 1, (10, 10, 20, 20), "2011"), d(2, 0, (3, 3, 9, 9), "2011")]
    return {"man_by": man_by, "recs": recs, "def_rows": def_rows, "reg_rows": []}


def _team(tmp: Path, order=(0, 1), iso0="2011") -> Path:
    boxes = [(1, 1, 5, 5, iso0), (10, 10, 20, 20, "2011")]
    anns = [{"annotation_role": "defect", "ann_idx": k, "bbox_xmin": boxes[i][0], "bbox_ymin": boxes[i][1],
             "bbox_xmax": boxes[i][2], "bbox_ymax": boxes[i][3], "class_std_iso6520": boxes[i][4]}
            for k, i in enumerate(order)]
    rows = [{"orig_info_id": 1, "width_px": 1280, "height_px": 720, "annotations": anns},
            {"orig_info_id": 2, "width_px": 1280, "height_px": 720,
             "annotations": [{"annotation_role": "defect", "ann_idx": 0, "bbox_xmin": 3, "bbox_ymin": 3,
                              "bbox_xmax": 9, "bbox_ymax": 9, "class_std_iso6520": "2011"}]}]
    p = tmp / "team.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_같은_평가_장은_같다(tmp_path):
    out = R.crosscheck_teammate(_res("2011"), _team(tmp_path), tmp_path)
    assert out["eval"] == {"images": 1, "all_equal": True, "unequal_images": 0}


def test_평가_장의_ISO_불일치는_같지_않음이다(tmp_path):
    out = R.crosscheck_teammate(_res("4011"), _team(tmp_path), tmp_path)
    assert out["eval"] == {"images": 1, "all_equal": False, "unequal_images": 1}


def test_평가_장의_순서_키만_다른_장도_같지_않음이다(tmp_path):
    out = R.crosscheck_teammate(_res("2011"), _team(tmp_path, order=(1, 0)), tmp_path)
    assert out["eval"]["all_equal"] is False and out["eval"]["unequal_images"] == 1


def test_평가_결과에는_한_비트와_장_수만_있다(tmp_path):
    out = R.crosscheck_teammate(_res("4011"), _team(tmp_path), tmp_path)
    assert set(out["eval"]) == {"images", "all_equal", "unequal_images"}
    assert (tmp_path / "xc_diffs_trainval_G.jsonl").read_text(encoding="utf-8") == ""   # 평가 장은 차이표에 없다


def _cand(tmp: Path, their_rows) -> Path:
    cand = tmp / "cand"
    cand.mkdir()
    (cand / "SNAPSHOT.sha256").write_text("aa  absorb_image.csv\n", encoding="utf-8")
    (cand / "data_capabilities.yaml").write_text("absorption: {}\n", encoding="utf-8")
    R.write_canonical(their_rows, R.IMAGE_COLS, "image_id", cand / "absorb_image.csv")
    for name, cols, key in (("absorb_defect.csv", R.DEFECT_COLS, "ann_id"),
                            ("absorb_region.csv", R.REGION_COLS, "region_id")):
        R.write_canonical([], cols, key, cand / name)
    return cand


def test_후보_판의_열별_집계에는_학습_검증_행만_든다(tmp_path):
    def img(iid, fm):
        return {"image_id": iid, "covered": True, "source_zip": "z", "material_source": "label_json",
                "orig_width_px": 1, "orig_height_px": 1, "frame_match": fm, "absorb_note": ""}
    ours = [img("aihub71761:1", "same"), img("aihub71761:2", "same")]
    theirs = [img("aihub71761:1", "tile_of_orig"), img("aihub71761:2", "same")]   # 평가 행만 다르다
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    hashes = {"absorb_image.csv": R.write_canonical(ours, R.IMAGE_COLS, "image_id", out_dir / "absorb_image.csv")}
    for name, cols, key in (("absorb_defect.csv", R.DEFECT_COLS, "ann_id"),
                            ("absorb_region.csv", R.REGION_COLS, "region_id")):
        hashes[name] = R.write_canonical([], cols, key, out_dir / name)
    res = {"man_by": {"aihub71761:1": {"split": "eval"}, "aihub71761:2": {"split": "train"}}, "hashes": hashes,
           "raw_acct": {"zips": []}}
    rep = R.compare_candidate(res, _cand(tmp_path, theirs), out_dir)
    ent = rep["absorb_image.csv"]
    assert ent["byte_equal"] is False and ent["rows_differing"] == {"train_val": 0, "eval": 1}
    assert ent["cols_differing_trainval"] == {} and ent["eval_block_equal"] is False
    assert "cols_differing" not in ent


# ---------------------------------------------------------------------------------------------
# build() — v1 일치 검사의 열별 집계도 학습 · 검증 행만(평가 행은 한 비트와 장 수)
# ---------------------------------------------------------------------------------------------

def _snapshot(tmp: Path, iso_eval: str, iso_train: str):
    """평가 장 1 · 학습 장 2 의 합성 원본 라벨 zip 과 v1 스냅샷."""
    import csv
    import io
    import zipfile
    from types import SimpleNamespace

    labels = tmp / "labels"
    labels.mkdir()
    poly = {"x": [10, 50, 50, 10], "y": [10, 10, 40, 40]}
    with zipfile.ZipFile(labels / "TL_rt.zip", "w") as z:
        for iid in (1, 2):
            z.writestr(f"{iid}.json", json.dumps({
                "info": {"id": iid, "type": "RT", "material": "ST"},
                "image_data": {"width": 1280, "height": 720},
                "annotations": [{"class": "defect", "case": "porosity", "coordinate": poly}]}))
    v1 = tmp / "v1"
    v1.mkdir()

    def write(name, cols, rows):
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        (v1 / name).write_text(buf.getvalue(), encoding="utf-8")
    write("manifest.csv", ["image_id", "split", "material", "width_px", "height_px", "client", "defect_types",
                           "iso_codes"],
          [{"image_id": f"aihub71761:{i}", "split": s, "material": "ST", "width_px": 1280, "height_px": 720,
            "client": c, "defect_types": "porosity", "iso_codes": "2011"}
           for i, s, c in ((1, "eval", ""), (2, "train", "C1"))])
    write("annotations.csv", ["ann_id", "image_id", "polygon_json", "src_label_raw", "defect_type", "iso_code",
                              "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "geom_flags"],
          [{"ann_id": f"aihub71761:{i}#0", "image_id": f"aihub71761:{i}", "polygon_json": R.poly_json(poly["x"], poly["y"]),
            "src_label_raw": "기공", "defect_type": "porosity", "iso_code": iso, "bbox_x1_px": 10, "bbox_y1_px": 10,
            "bbox_x2_px": 50, "bbox_y2_px": 40, "geom_flags": ""} for i, iso in ((1, iso_eval), (2, iso_train))])
    write("tiles.csv", ["image_id", "reason"], [{"image_id": f"aihub71761:{i}", "reason": "ok"} for i in (1, 2)])
    repo = Path(__file__).resolve().parents[1]
    return SimpleNamespace(labels=labels, v1=v1, label_map=repo / "configs" / "label_map.yaml", out=tmp / "out")


def test_build_은_평가_행의_v1_어긋남을_열별로_싣지_않는다(tmp_path):
    res = R.build(_snapshot(tmp_path, iso_eval="9999", iso_train="2011"))
    assert res["v1_mismatch_trainval"] == {}
    assert res["v1_check_eval"] == {"all_equal": False, "unequal_images": 1}
    assert "v1_mismatch" not in res and "label_defect_v1_normal_by_split" not in res


def test_build_은_학습_행의_v1_어긋남을_열별로_싣는다(tmp_path):
    res = R.build(_snapshot(tmp_path, iso_eval="2011", iso_train="9999"))
    assert res["v1_mismatch_trainval"] == {"iso_code": 1}
    assert res["v1_check_eval"] == {"all_equal": True, "unequal_images": 0}
