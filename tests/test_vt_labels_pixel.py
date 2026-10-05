"""VT-2 · VT-3 · VT-3b · VT-3c · VT-4 — 라벨 파싱 · 화소 꼴 · 다시 인코딩. 실물을 읽지 않는다."""
from __future__ import annotations

import copy
import io
import json
import re
import zipfile

import numpy as np
import pytest
import yaml
from PIL import Image

from data.convert.tiling import candidate_origins, select_index
from data.vt.config import REPO_ROOT, load_config
from data.vt.labels import (
    REASON_OUT_OF_SPACE,
    REASON_SCHEMA,
    LabelInconsistency,
    parse_label_zips,
    parse_one,
)
from data.vt.pixel import (
    PROV_BAND,
    PROV_CROP,
    PROV_TILE,
    REASON_NO_WINDOW,
    PixelFormError,
    VtPlan,
    band_inside,
    encode,
    plan,
    rotate_ccw_points,
)

CFG = load_config()
NAME_RE = re.compile(CFG["labels"]["name_pattern"])
CODES = {f: v["code"] for f, v in CFG["labels"]["folders"].items()}
INFO = {f: v["information"] for f, v in CFG["labels"]["folders"].items()}
POR, LOF, UND, NOR = "결함_1. 기공", "결함_3. 융합불량", "결함_4. 언더컷", "정상"


def _label(folder, vid, w=1280, h=720, anns=None, **over):
    stem = f"VT_ST_{CODES[folder]}_{vid}"
    d = {"info": {"id": vid, "type": "VT", "material": "ST"},
         "image_data": {"file_name": stem, "format": "jpg", "information": INFO[folder], "width": w, "height": h},
         "meta": {"is_crowd": 0},
         "annotations": anns if anns is not None else [
             {"tool": "polygon", "coordinate": {"x": [10, 20, 20], "y": [10, 10, 20]},
              "class": "normal" if folder == NOR else "defect", "case": "" if folder == NOR else "porosity"}]}
    for k, v in over.items():
        sec, key = k.split("__")
        d[sec][key] = v
    return f"/{stem}.json", json.dumps(d, ensure_ascii=False).encode("utf-8")


def _parse(folder, vid, **kw):
    member, raw = _label(folder, vid, **kw)
    return parse_one(raw, member=member, folder=folder, author_split="Training", cfg=CFG, name_re=NAME_RE)


def _poly(x0, y0, x1, y1, cls="defect", case="lack of fusion"):
    return {"tool": "polygon", "coordinate": {"x": [x0, x1, x1, x0], "y": [y0, y0, y1, y1]}, "class": cls, "case": case}


# ------------------------------------------------------------------------------------------
# 라벨 (VT-2)
# ------------------------------------------------------------------------------------------
def test_일관된_라벨은_VT_원천_키로_읽힌다():
    r = _parse(POR, 14599003)
    assert r.image_id == "aihub71761_vt:14599003" and r.folder == POR and not r.is_normal
    assert r.cases("defect") == ["porosity"]


@pytest.mark.parametrize("over", [
    {"image_data__information": "융합불량"}, {"info__type": "RT"}, {"info__material": "AL"},
    {"info__id": 1}, {"image_data__file_name": "VT_ST_02_9"},
])
def test_전_장이_같아야_하는_값이_어긋나면_멈춘다(over):
    with pytest.raises(LabelInconsistency):
        _parse(POR, 14599003, **over)


def test_파일명_코드가_폴더와_다르면_멈춘다():
    """파일명 코드만 다르고 file_name 은 파일명과 같다 — 코드 검사 하나만 걸리게 둔다(06b I-1)."""
    member, raw = _label(POR, 5)
    d = json.loads(raw)
    d["image_data"]["file_name"] = "VT_ST_04_5"
    with pytest.raises(LabelInconsistency, match="파일명 코드"):
        parse_one(json.dumps(d, ensure_ascii=False).encode(), member=member.replace("_02_", "_04_"), folder=POR,
                  author_split="Training", cfg=CFG, name_re=NAME_RE)


def test_정상_폴더의_결함_주석_결함_폴더의_결함_없음_모르는_case_와_class_는_멈춘다():
    with pytest.raises(LabelInconsistency):
        _parse(NOR, 1, anns=[_poly(0, 0, 9, 9)])
    with pytest.raises(LabelInconsistency):
        _parse(POR, 2, anns=[_poly(0, 0, 9, 9, cls="normal", case="")])
    with pytest.raises(LabelInconsistency):
        _parse(POR, 3, anns=[_poly(0, 0, 9, 9, case="slag")])
    with pytest.raises(LabelInconsistency):
        _parse(POR, 4, anns=[_poly(0, 0, 9, 9, cls="other")])


def test_case_가_빈_결함_주석은_이름_있는_결함이_함께_있으면_남고_홀로면_멈춘다():
    r = _parse(POR, 21, anns=[_poly(0, 0, 9, 9, case="porosity"), _poly(20, 0, 29, 9, case="")])
    assert r.cases("defect") == ["porosity", ""]
    with pytest.raises(LabelInconsistency):
        _parse(POR, 22, anns=[_poly(0, 0, 9, 9, case="")])


def test_스키마가_어긋난_장과_라벨_공간_밖_장은_제외_장부로_간다():
    member, raw = _label(POR, 7)
    d = json.loads(raw)
    del d["image_data"]["width"]
    r = parse_one(json.dumps(d).encode(), member=member, folder=POR, author_split="Training", cfg=CFG,
                  name_re=NAME_RE)
    assert r["reason"] == REASON_SCHEMA and r["vt_id"] == 7
    r = _parse(POR, 8, anns=[_poly(0, 0, 9, 9, case="porosity"), _poly(20, 0, 29, 9, case="crack")])
    assert r["reason"] == REASON_OUT_OF_SPACE and r["detail"] == "crack" and r["label"].vt_id == 8


def _write_zips(tmp_path, extra=None):
    d = tmp_path / "labels"
    d.mkdir(parents=True)
    vid = 100
    for pre in ("TL", "VL"):
        for folder in CFG["labels"]["folders"]:
            with zipfile.ZipFile(d / f"{pre}_VTST_{folder}.zip", "w") as zf:
                vid += 1
                zf.writestr(*_label(folder, vid))
                for f, v in (extra or {}).get((pre, folder), []):
                    zf.writestr(*_label(f, v))
    return d


def test_라벨_zip_열_개를_읽고_같은_id_가_두_번이면_멈춘다(tmp_path):
    res = parse_label_zips(_write_zips(tmp_path), CFG)
    assert len(res.labels) == 10 and res.excluded == []
    assert {r.author_split for r in res.labels} == {"Training", "Validation"}
    with pytest.raises(LabelInconsistency):
        parse_label_zips(_write_zips(tmp_path / "b", {("VL", NOR): [(NOR, 101)]}), CFG)


# ------------------------------------------------------------------------------------------
# 화소 꼴 계획 (VT-3 · VT-4)
# ------------------------------------------------------------------------------------------
def test_결함_1280x720_은_그대로_N_crop():
    p = plan(_parse(POR, 1), CFG)
    assert p.keep and p.provenance == PROV_CROP and p.box == (0, 0, 1280, 720) and p.rot90_k == 0


@pytest.mark.parametrize("x0,x1,want", [(100, 600, [0]), (1300, 1700, [520]), (600, 1200, [0, 520]),
                                        (100, 1700, []), (519, 1281, [])])
def test_1800x720_융합불량은_모든_결함이_드는_창으로_자르고_없으면_뺀다(x0, x1, want):
    assert candidate_origins(1800, 1280, 640, 8) == [0, 520]
    lab = _parse(LOF, 9, w=1800, anns=[_poly(x0, 100, x1, 300)])
    p = plan(lab, CFG)
    if not want:
        assert not p.keep and p.reason == REASON_NO_WINDOW
        return
    pick = want[select_index(lab.image_id, CFG["tile"]["selection_seed"], len(want))]
    assert p.keep and p.provenance == PROV_TILE and p.box == (pick, 0, pick + 1280, 720)


def test_창_판정은_바깥_정수화_상자로_한다():
    # 1280.4 는 올림해 1281 이라 창 [0, 1280] 에 들지 않는다 — 창 [520, 1800] 하나만 남는다
    lab = _parse(LOF, 10, w=1800, anns=[{"tool": "polygon", "coordinate": {"x": [600, 1280.4, 700], "y": [1, 2, 3]},
                                         "class": "defect", "case": "lack of fusion"}])
    p = plan(lab, CFG)
    assert p.box == (520, 0, 1800, 720)


def test_정한_결함_크기는_빼고_그_밖의_결함_크기는_멈춘다():
    assert plan(_parse(UND, 11, w=720, h=1280), CFG).reason == "portrait_undercut"
    assert plan(_parse(POR, 12, w=2603, h=1464), CFG).reason == "odd_size_defect"
    with pytest.raises(PixelFormError):
        plan(_parse(UND, 13, w=1800, h=720), CFG)        # 융합불량이 아닌 1800×720


def test_정상_정사각은_빼고_1280x720_은_그대로():
    nb = [_poly(0, 300, 1280, 700, cls="normal", case="")]
    assert plan(_parse(NOR, 14, w=1280, h=1280, anns=nb), CFG).reason == "square_normal"
    p = plan(_parse(NOR, 15, anns=[_poly(0, 300, 1280, 600, cls="normal", case="")]), CFG)
    assert p.keep and p.provenance == PROV_CROP


def test_정상_가로_사진은_P1_비드_세로_중심과_해시_가로_원점():
    lab = _parse(NOR, 16, w=3840, h=2160, anns=[_poly(0, 1000, 3840, 1400, cls="normal", case="")])
    p = plan(lab, CFG)
    xs = candidate_origins(3840, 1280, 640, 8)
    x0 = xs[select_index(lab.image_id, CFG["tile"]["selection_seed"], len(xs))]
    assert p.keep and p.provenance == PROV_TILE and p.box == (x0, 840, x0 + 1280, 1560)
    assert band_inside(lab, p, CFG) is True
    big = plan(_parse(NOR, 17, w=3840, h=2160, anns=[_poly(0, 500, 3840, 1500, cls="normal", case="")]), CFG)
    assert big.provenance == PROV_BAND


def test_세로_정상은_반시계_90도로_돌린_뒤_타일이고_원점은_돌린_뒤_프레임():
    # 폭 2160 의 원천에서 x 1000 ~ 1445 의 세로 비드 → 돌린 뒤 y 715 ~ 1160
    lab = _parse(NOR, 18, w=2160, h=3840, anns=[_poly(1000, 0, 1445, 3840, cls="normal", case="")])
    p = plan(lab, CFG)
    assert p.rot90_k == 1 and p.provenance == PROV_TILE
    cy = (715 + 1160) // 2
    assert p.box[1] == ((cy - 360) // 8) * 8 and p.box[3] - p.box[1] == 720
    assert p.box[2] <= 3840


def test_돌림_좌표는_np_rot90_과_PIL_ROTATE_90_과_같다():
    W, H = 7, 5
    a = np.zeros((H, W), dtype=np.uint8)
    for (x, y) in [(0, 0), (6, 1), (3, 4)]:
        a[:] = 0
        a[y, x] = 255
        rx, ry = rotate_ccw_points([x + 0.5], [y + 0.5], W)
        col, row = int(rx[0]), int(ry[0])
        assert np.rot90(a, k=1)[row, col] == 255
        assert np.asarray(Image.fromarray(a).transpose(Image.Transpose.ROTATE_90))[row, col] == 255


# ------------------------------------------------------------------------------------------
# 다시 인코딩 (VT-3b)
# ------------------------------------------------------------------------------------------
def _cfg(q=90):
    c = copy.deepcopy(CFG)
    c["encode"]["quality"] = q
    return c


def _jpeg(arr, mode="RGB", exif_orientation=None):
    im = Image.fromarray(arr) if mode == "RGB" else Image.fromarray(arr).convert(mode)
    buf = io.BytesIO()
    kw = {}
    if exif_orientation:
        ex = Image.Exif()
        ex[0x0112] = exif_orientation
        kw["exif"] = ex.tobytes()
    im.save(buf, format="JPEG", quality=95, **kw)
    return buf.getvalue()


def _quad(h, w):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[: h // 2, : w // 2] = (250, 10, 10)
    a[: h // 2, w // 2:] = (10, 250, 10)
    a[h // 2:, : w // 2] = (10, 10, 250)
    a[h // 2:, w // 2:] = (250, 250, 10)
    return a


def test_다시_인코딩은_1280x720_RGB_EXIF_없이_계획의_상자를_뜬다():
    src = _quad(720, 1800)
    p = VtPlan("aihub71761_vt:1", 1, True, "defect_window", PROV_TILE, 0, (520, 0, 1800, 720), 1800, 720)
    data, rec = encode(_jpeg(src), p, _cfg())
    im = Image.open(io.BytesIO(data))
    assert im.size == (1280, 720) and im.mode == "RGB" and rec["w"] == 1280
    assert not im.getexif() and "icc_profile" not in im.info and not im.info.get("progressive")
    diff = np.abs(np.asarray(im, dtype=int) - src[:, 520:1800].astype(int)).mean()
    assert diff < 3


def test_돌린_장의_다시_인코딩은_np_rot90_의_상자와_같다():
    src = _quad(1920, 960)                                # 세로 960×1920
    p = VtPlan("aihub71761_vt:2", 2, True, "tiled", PROV_TILE, 1, (320, 96, 1600, 816), 960, 1920)
    data, _ = encode(_jpeg(src), p, _cfg())
    got = np.asarray(Image.open(io.BytesIO(data)), dtype=int)
    want = np.rot90(src, k=1)[96:816, 320:1600].astype(int)
    assert np.abs(got - want).mean() < 3


@pytest.mark.parametrize("bad", ["size", "exif", "mode"])
def test_선언_크기_EXIF_방향_모드가_어긋나면_멈춘다(bad):
    src = _quad(720, 1280)
    raw = {"size": _jpeg(src[:, :1200]), "exif": _jpeg(src, exif_orientation=6), "mode": _jpeg(src, mode="L")}[bad]
    p = VtPlan("aihub71761_vt:3", 3, True, "ok", PROV_CROP, 0, (0, 0, 1280, 720), 1280, 720)
    with pytest.raises(PixelFormError):
        encode(raw, p, _cfg())


def test_품질이_비었거나_멈춤_문턱_아래면_멈춘다():
    raw = _jpeg(_quad(720, 1280))
    p = VtPlan("aihub71761_vt:4", 4, True, "ok", PROV_CROP, 0, (0, 0, 1280, 720), 1280, 720)
    for q in (None, 49):
        with pytest.raises(PixelFormError):
            encode(raw, p, _cfg(q))


def test_VT_3c_타일_규칙의_값이_RT_base_yaml_과_같다():
    base = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))["preprocess"]["tile"]
    vt = CFG["tile"]
    for k in ("tile_size", "stride_x", "vertical_anchor", "tau_mode", "align8", "padding", "selection"):
        assert vt[k] == base[k], k
    assert CFG["encode"]["progressive"] == base["encode"]["progressive"]
    assert CFG["encode"]["optimize"] == base["encode"]["optimize"]
    assert CFG["encode"]["strip_exif"] == base["encode"]["strip_exif"]


def test_돌린_정상의_상자는_정수이고_CSV_왕복_뒤에도_읽힌다(tmp_path):
    from data.vt.build_r1 import CROP_HEADER
    from data.vt.records import read_csv_checked, write_csv
    lab = _parse(NOR, 31, w=2160, h=3840, anns=[_poly(1000, 0, 1445, 3840, cls="normal", case="")])
    p = plan(lab, CFG)
    assert all(type(v) is int for v in p.box)
    row = {"image_id": p.image_id, "rule_version": "vt1", "src_w": p.src_w, "src_h": p.src_h, "rot90_k": p.rot90_k,
           "x0": p.box[0], "y0": p.box[1], "x1": p.box[2], "y1": p.box[3], "provenance": p.provenance,
           "reason": p.reason}
    h = write_csv(tmp_path / "c.csv", CROP_HEADER, [row])
    back = read_csv_checked(tmp_path / "c.csv", h)[0]
    assert [int(back[k]) for k in ("x0", "y0", "x1", "y1")] == list(p.box)


def _segments(data: bytes) -> list[int]:
    """SOS 앞까지의 JPEG 표지 목록."""
    i, out = 2, []
    while i < len(data) - 1:
        assert data[i] == 0xFF
        m = data[i + 1]
        out.append(m)
        if m == 0xDA:
            break
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return out


def test_다시_인코딩은_원천의_주석_EXIF_ICC_를_싣지_않는다():
    im = Image.fromarray(_quad(720, 1280))
    ex = Image.Exif()
    ex[0x0131] = "camera-software"
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=95, exif=ex.tobytes(), comment=b"made by defect cropper",
            icc_profile=b"\0" * 128)
    src = buf.getvalue()
    assert 0xFE in _segments(src) and 0xE1 in _segments(src) and 0xE2 in _segments(src)
    p = VtPlan("aihub71761_vt:5", 5, True, "ok", PROV_CROP, 0, (0, 0, 1280, 720), 1280, 720)
    data, _ = encode(src, p, _cfg())
    seg = _segments(data)
    assert 0xFE not in seg and not ({0xE1, 0xE2, 0xED, 0xEE} & set(seg))
    assert not [m for m in seg if 0xE0 < m <= 0xEF]


def test_창으로_자를_결함에_좌표가_없으면_정해진_멈춤이다():
    lab = _parse(LOF, 32, w=1800, anns=[_poly(100, 100, 300, 300)])
    from dataclasses import replace

    from data.vt.labels import Ann
    lab = replace(lab, anns=(Ann("defect", "lack of fusion", (), ()), Ann("normal", "", (1.0,), (1.0,))))
    with pytest.raises(PixelFormError):
        plan(lab, CFG)


def test_메타데이터를_남기는_설정은_멈춘다():
    c = _cfg()
    c["encode"]["strip_exif"] = False
    p = VtPlan("aihub71761_vt:6", 6, True, "ok", PROV_CROP, 0, (0, 0, 1280, 720), 1280, 720)
    with pytest.raises(PixelFormError):
        encode(_jpeg(_quad(720, 1280)), p, c)


def test_R_9_산출과_설정의_품질이_다르거나_멈춤이면_인코딩을_시작하지_않는다():
    from data.vt.build_r1 import check_r9
    check_r9(_cfg(74), {"quality": 74, "stop": False})
    with pytest.raises(SystemExit):
        check_r9(_cfg(74), {"quality": 71, "stop": False})
    with pytest.raises(SystemExit):
        check_r9(_cfg(74), {"quality": 74, "stop": True})


def test_진행_기록의_머리가_다르면_이어_받지_않는다(tmp_path):
    from data.vt.build_r1 import encode_header, read_progress
    h = encode_header(_cfg(74), "a" * 64, "b" * 64)
    p = tmp_path / "prog.jsonl"
    p.write_text(json.dumps({"header": h}) + "\n" + json.dumps({"image_id": "x"}) + "\n{\"image_id\": ",
                 encoding="utf-8")
    assert read_progress(p, h) == {"x"}
    with pytest.raises(SystemExit):
        read_progress(p, encode_header(_cfg(75), "a" * 64, "b" * 64))
    with pytest.raises(SystemExit):
        read_progress(p, encode_header(_cfg(74), "c" * 64, "b" * 64))


def test_라벨_zip_이_열_개가_아니면_멈춘다(tmp_path):
    d = _write_zips(tmp_path)
    next(d.glob("VL_*.zip")).unlink()
    with pytest.raises(LabelInconsistency):
        parse_label_zips(d, CFG)


def test_zip_안의_같은_줄기_멤버는_멈춘다(tmp_path):
    from data.vt.labels import zip_member_index
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("/A/VT_ST_00_1.jpg", b"x")
        zf.writestr("/B/VT_ST_00_1.jpg", b"y")
    with zipfile.ZipFile(p) as zf, pytest.raises(LabelInconsistency):
        zip_member_index(zf)


def test_인코딩은_프레임_밖_상자에서_멈춘다():
    """패딩 금지를 인코딩 단계에서도 — 프레임 밖은 crop 이 검은색으로 채운다(06b I-2)."""
    raw = _jpeg(_quad(720, 1280))
    for box in [(80, 0, 1360, 720), (-8, 0, 1272, 720), (0, 0, 1272, 720), (0, 8, 1280, 728), (0, -8, 1280, 712)]:
        p = VtPlan("aihub71761_vt:7", 7, True, "ok", PROV_CROP, 0, box, 1280, 720)
        with pytest.raises(PixelFormError):
            encode(raw, p, _cfg())


def test_돌린_세로_정상의_비드가_아래쪽이어도_상자는_돌린_프레임_안이다():
    lab = _parse(NOR, 33, w=2160, h=3840, anns=[_poly(0, 0, 400, 3840, cls="normal", case="")])
    p = plan(lab, CFG)
    assert p.rot90_k == 1 and 0 <= p.box[1] and p.box[3] <= 2160 and p.box[2] <= 3840
    assert p.box[3] == 2160 and p.n_candidates == len(candidate_origins(3840, 1280, 640, 8))


def test_설정의_규칙_문자열이_이_코드의_동작과_다르면_멈춘다():
    for k, v in [("vertical_anchor", "center"), ("tau_mode", "area_ratio"), ("padding", "allowed"),
                 ("selection", "center")]:
        c = copy.deepcopy(CFG)
        c["tile"][k] = v
        with pytest.raises(PixelFormError):
            plan(_parse(POR, 34), c)


def test_돌릴_세로_사진의_폭이_8_의_배수가_아니면_멈춘다():
    lab = _parse(NOR, 35, w=2164, h=3840, anns=[_poly(1000, 0, 1445, 3840, cls="normal", case="")])
    with pytest.raises(PixelFormError):
        plan(lab, CFG)


def test_창의_오른쪽_경계와_품질_50_은_든다():
    lab = _parse(LOF, 36, w=1800, anns=[_poly(600, 100, 1280, 300)])
    assert plan(lab, CFG).n_candidates == 2                 # x1 = 1280 은 창 [0, 1280] 에 든다
    p = VtPlan("aihub71761_vt:8", 8, True, "ok", PROV_CROP, 0, (0, 0, 1280, 720), 1280, 720)
    data, _ = encode(_jpeg(_quad(720, 1280)), p, _cfg(50))
    assert Image.open(io.BytesIO(data)).size == (1280, 720)
