"""VT 회차 1 — 실행 단계를 도는 합성 시험(06b I-1). 실물을 읽지 않는다.

합성 저장소(라벨 zip 열 · 이미지 zip 열 · 복사 대조 기록)에서 `cmd_plan` → `cmd_quality` → `cmd_encode` → `finalize_tiles`
를 돌린다. 산출 파일의 칸 · 키가 **허용 목록과 같은지** 단언한다 — 기본 실행의 VT-1d(평가 행 · 분할 칸이 없다)다.
복사(`copy.main`)의 세 길(새로 받기 · 기록에 있어 건너뛰기 · 기록 없이 남은 최종 사본)도 돈다.
"""
from __future__ import annotations

import copy as _copy
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import data.vt.build_r1 as B
import data.vt.copy as CP
from data.vt.config import load_config

CFG = load_config()
FOLDERS = list(CFG["labels"]["folders"])


def _cfg() -> dict:
    """합성 자료는 장 수가 작아 한 장만 빼도 셀이 폐기 상한을 넘는다 — 상한 판독 시험 밖에서는 모든 셀을 등록 예외로 둔다."""
    c = _copy.deepcopy(CFG)
    c["tile"]["known_over_cap"] = list(FOLDERS)
    return c
NOR = CFG["labels"]["normal_folder"]


class _NoWait:
    def __init__(self, *a, **k):
        self.c_free_fn = lambda: 99.0

    @classmethod
    def from_config(cls, *a, **k):
        return cls()

    def wait(self):
        return {"c_free_gb": 99.0, "mem_free_gb": 99.0}


def _jpeg(w, h, q=92, seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.random((h // 16 + 1, w // 16 + 1, 3)) * 255).astype(np.uint8)
    im = Image.fromarray(a).resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def _label(folder, vid, w, h, anns):
    stem = f"VT_ST_{CFG['labels']['folders'][folder]['code']}_{vid}"
    d = {"info": {"id": vid, "type": "VT", "material": "ST"},
         "image_data": {"file_name": stem, "format": "jpg",
                        "information": CFG["labels"]["folders"][folder]["information"], "width": w, "height": h},
         "meta": {}, "annotations": anns}
    return stem, json.dumps(d, ensure_ascii=False).encode("utf-8")


def _poly(x0, y0, x1, y1, cls, case):
    return {"coordinate": {"x": [x0, x1, x1, x0], "y": [y0, y0, y1, y1]}, "class": cls, "case": case}


def _repo(tmp: Path, q: int = 92, q_normal: int | None = None) -> Path:
    """합성 저장소 — 폴더마다 Training 두 장 · Validation 한 장. 정상에는 가로 · 세로 · 정사각, 융합불량에 1800×720."""
    root = tmp / "repo"
    lab = root / CFG["copy"]["label_zip_dir"]
    raw = root / CFG["copy"]["raw_zip_dir"]
    lab.mkdir(parents=True)
    raw.mkdir(parents=True)
    vid = 1000
    case = {"결함_1. 기공": "porosity", "결함_2. 용입부족": "incomplete penetration",
            "결함_3. 융합불량": "lack of fusion", "결함_4. 언더컷": "undercut"}
    for split in ("Training", "Validation"):
        for folder in FOLDERS:
            sizes = [(1280, 720)] if split == "Validation" else [(1280, 720), (1280, 720)]
            if folder == NOR and split == "Training":
                sizes = [(2560, 1440), (1440, 2560), (1280, 1280)]
            if folder == "결함_3. 융합불량" and split == "Training":
                sizes = [(1280, 720), (1800, 720)]
            lz = zipfile.ZipFile(lab / B.zip_name_of(split, folder, "label", CFG), "w")
            iz = zipfile.ZipFile(raw / B.zip_name_of(split, folder, "image", CFG), "w")
            for (w, h) in sizes:
                vid += 1
                if folder == NOR:
                    if h > w:
                        anns = [_poly(w // 2 - 200, 0, w // 2 + 200, h, "normal", "")]
                    else:
                        anns = [_poly(0, h // 2 - 200, w, h // 2 + 200, "normal", "")]
                else:
                    anns = [_poly(100, 100, 400, 300, "defect", case[folder])]
                stem, data = _label(folder, vid, w, h, anns)
                lz.writestr(f"/{stem}.json", data)
                iz.writestr(f"/{stem}.jpg", _jpeg(w, h, q=q_normal if (q_normal and folder == NOR) else q, seed=vid))
            lz.close()
            iz.close()
    (root / "configs").mkdir()
    (root / "configs" / "base.yaml").write_bytes((B.REPO_ROOT / "configs" / "base.yaml").read_bytes())
    st = root / CFG["staging_root"] / "copy"
    st.mkdir(parents=True)
    (st / "copy_verify.json").write_text(json.dumps({"all_ok": True, "image_content_evidence_all": True}),
                                         encoding="utf-8")
    (st / "copy_record.jsonl").write_text('{"name": "x"}\n', encoding="utf-8")
    return root


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    monkeypatch.setattr(B, "repo_path", lambda rel, r=None: root / rel)
    monkeypatch.setattr(B, "REPO_ROOT", root)
    monkeypatch.setattr(B, "Guard", _NoWait)
    return root


def _header(p: Path) -> list[str]:
    with p.open(encoding="utf-8") as fh:
        return next(csv.reader(fh))


#: 산출의 칸 — 허용 목록. 분할 · 평가 · 묶음 · 참여자 칸이 생기면 여기서 떨어진다(VT-1d)
ALLOWED = {
    "crop_boxes.csv": ["image_id", "rule_version", "src_w", "src_h", "rot90_k", "x0", "y0", "x1", "y1", "provenance",
                       "reason"],
    "exclusions.csv": ["image_id", "folder", "src_w", "src_h", "reason"],
    "encoded_tiles.csv": ["image_id", "path", "sha256", "bytes"],
}
LABEL_KEYS = {"vt_id", "image_id", "folder", "src_zip", "file_stem", "w", "h", "is_normal", "label_excluded", "anns"}


def test_실행_단계를_끝까지_돌고_산출의_칸은_허용_목록과_같다(repo):
    cfg = _cfg()
    assert B.cmd_plan(cfg) == 0
    st = repo / CFG["staging_root"]
    s = json.loads((st / "records" / "plan_summary.json").read_text(encoding="utf-8"))
    assert s["accounting_ok"] and s["labels_parsed"] == s["kept"] + s["excluded"] == 16
    assert {r["reason"] for r in csv.DictReader((st / "records" / "exclusions.csv").open(encoding="utf-8"))} == {
        "square_normal"}
    assert B.cmd_quality(cfg) == 0
    r9 = json.loads((st / "records" / "r9_quality.json").read_text(encoding="utf-8"))
    cfg["encode"]["quality"] = r9["quality"]
    assert B.cmd_encode(cfg, workers=1, batch=4) == 0
    for name, cols in ALLOWED.items():
        assert _header(st / "records" / name) == cols, name
    with (st / "parse" / "labels.jsonl").open(encoding="utf-8") as fh:
        assert {k for line in fh for k in json.loads(line)} == LABEL_KEYS
    e = json.loads((st / "records" / "encode_summary.json").read_text(encoding="utf-8"))
    assert e["encoded"] == s["kept"] and e["sizes"] == {"1280x720": s["kept"]} and e["extra_files_in_tile_dir"] == 0
    assert e["image_members_equal_labels"] and e["provenance"]["pillow"]
    # 이어 하기 — 다시 돌리면 할 것이 없고 같은 판정이다. 진행 기록의 꼬리가 잘려 있어도 고친 뒤 잇는다
    prog = st / "records" / "encode_progress.jsonl"
    prog.write_bytes(prog.read_bytes() + b'{"image_id": "aihub71761_vt:10')
    assert B.cmd_encode(cfg, workers=1, batch=4) == 0
    log = [json.loads(x) for x in (st / "records" / "encode_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d for d in log if d.get("start")][-1]["todo_batches"] == 0
    assert all(json.loads(x) for x in prog.read_text(encoding="utf-8").splitlines())
    # 폴더에 계획 밖 파일이 있으면 마무리가 떨어진다(06b m-3)
    (st / "tiles" / "VT" / "ST" / "999.jpg.tmp").write_bytes(b"x")
    assert B.finalize_tiles(cfg) == 7


def test_파싱_산출은_계획_요약의_해시와_맞을_때만_읽힌다(repo):
    cfg = _cfg()
    assert B.cmd_plan(cfg) == 0
    st = repo / CFG["staging_root"]
    assert len(B.load_labels(st)) == 16
    p = st / "parse" / "labels.jsonl"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(SystemExit):
        B.load_labels(st)


def test_계획은_복사_대조와_내용_증거가_없으면_시작하지_않는다(repo):
    st = repo / CFG["staging_root"] / "copy" / "copy_verify.json"
    st.write_text(json.dumps({"all_ok": True, "image_content_evidence_all": False}), encoding="utf-8")
    with pytest.raises(SystemExit):
        B.cmd_plan(_copy.deepcopy(CFG))
    st.write_text(json.dumps({"all_ok": False, "image_content_evidence_all": True}), encoding="utf-8")
    with pytest.raises(SystemExit):
        B.cmd_plan(_copy.deepcopy(CFG))


def test_등록되지_않은_셀이_폐기_상한을_넘으면_계획이_멈춘다(repo):
    cfg = _cfg()
    assert B.cmd_plan(cfg) == 0
    cfg["tile"]["known_over_cap"] = [f for f in FOLDERS if f != NOR]
    assert B.cmd_plan(cfg) == 8                      # 정상 4 장 가운데 1 장(정사각)을 빼면 상한 5 % 를 넘는다


def test_R_9_는_헤더만_읽고_품질이_50_아래면_멈춘다(tmp_path, monkeypatch):
    root = _repo(tmp_path, q=40)
    monkeypatch.setattr(B, "repo_path", lambda rel, r=None: root / rel)
    monkeypatch.setattr(B, "REPO_ROOT", root)
    monkeypatch.setattr(B, "Guard", _NoWait)
    cfg = _cfg()
    assert B.cmd_plan(cfg) == 0
    # 머리 64 KB 뒤를 망가뜨려도 R-9 는 같은 값을 낸다 — 화소를 해독하지 않는다
    raw = root / CFG["copy"]["raw_zip_dir"]
    for z in raw.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            items = [(n, zf.read(n)) for n in zf.namelist()]
        with zipfile.ZipFile(z, "w") as zf:
            for n, d in items:
                zf.writestr(n, d[:1 << 16] + b"\x00" * 1000)
    assert B.cmd_quality(cfg) == 6
    r9 = json.loads((root / CFG["staging_root"] / "records" / "r9_quality.json").read_text(encoding="utf-8"))
    assert r9["stop"] and r9["quality"] < 50 and r9["header_fail"] == 0


# ------------------------------------------------------------------------------------------
# 복사의 세 길
# ------------------------------------------------------------------------------------------
def _drive(tmp: Path):
    drive = tmp / "drive"
    names = ["Training/02/TL_VTST_정상.zip", "Training/01/TS_VTST_정상.zip"]
    data = {}
    for rel in names:
        p = drive / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        b = (rel * 4000).encode("utf-8")
        p.write_bytes(b)
        data[rel] = b
    listing = tmp / "listing.tsv"
    listing.write_text("".join(f"{len(b)}\t{rel}\n" for rel, b in data.items()), encoding="utf-8")
    census = tmp / "census.json"
    census.write_text(json.dumps({"inputs": {"vt_label_zips": [
        {"name": "TL_VTST_정상.zip", "sha256": hashlib.sha256(data[names[0]]).hexdigest()}]}}), encoding="utf-8")
    return drive, listing, census, data


def _run_copy(tmp, monkeypatch, root, drive, listing, census):
    monkeypatch.setattr(CP, "repo_path", lambda rel, r=None: root / rel)
    monkeypatch.setattr(CP, "REPO_ROOT", root)
    monkeypatch.setattr(CP, "Guard", _NoWait)
    monkeypatch.setattr(CP, "default_drivefs_db", lambda: None)
    args = ["--drive-root", str(drive), "--listing", str(listing), "--listing-sha256", CP.sha256_of(listing)[:16],
            "--label-census", str(census), "--label-census-sha256", CP.sha256_of(census)[:16]]
    return CP.main(args)


def test_복사_main_은_새로_받고_기록에_있으면_건너뛰고_기록_없는_최종_사본은_원천과_맞댄다(tmp_path, monkeypatch):
    drive, listing, census, _ = _drive(tmp_path)
    root = tmp_path / "repo"
    # 목록이 스무 개 · 라벨 해시가 열 개가 아니면 main 이 멈추므로 그 수만 이 시험의 합성에 맞춘다
    monkeypatch.setattr(CP, "LISTING_N", (2, 1))
    rc = _run_copy(tmp_path, monkeypatch, root, drive, listing, census)
    assert rc == 0                                    # 대조가 지났다(캐시가 없어 서버 md5 는 세지 않는다)
    rec = CP._load_records(root / CFG["staging_root"] / "copy" / "copy_record.jsonl")
    assert set(rec) == {"TL_VTST_정상.zip", "TS_VTST_정상.zip"}
    n_lines = len((root / CFG["staging_root"] / "copy" / "copy_record.jsonl").read_text(encoding="utf-8").splitlines())
    # 둘째 실행 — 기록에 있으니 다시 받지 않는다
    _run_copy(tmp_path, monkeypatch, root, drive, listing, census)
    assert len((root / CFG["staging_root"] / "copy" / "copy_record.jsonl").read_text(
        encoding="utf-8").splitlines()) == n_lines
    # 기록을 지우고 최종 사본을 원천과 다르게 바꾸면 기록하지 않고 멈춘다(06b I-6)
    (root / CFG["staging_root"] / "copy" / "copy_record.jsonl").write_text("", encoding="utf-8")
    raw = root / CFG["copy"]["raw_zip_dir"] / "TS_VTST_정상.zip"
    raw.write_bytes(b"different" + raw.read_bytes()[9:])
    with pytest.raises(CP.CopyMismatch):
        _run_copy(tmp_path, monkeypatch, root, drive, listing, census)
    # 라벨 zip 은 원천과 같아 기록됐고, 다른 원천 zip 은 기록되지 않았다
    rec = CP._load_records(root / CFG["staging_root"] / "copy" / "copy_record.jsonl")
    assert set(rec) == {"TL_VTST_정상.zip"} and rec["TL_VTST_정상.zip"]["recovered_matched_source"]


def test_R_9_는_두_모집단_최저치_가운데_작은_값이다(tmp_path, monkeypatch):
    """결함 92 · 정상 60 으로 저장한 원천 — 품질은 작은 쪽(정상)의 내림이다(06c Q04)."""
    from scripts.measure_jpeg_fingerprint import estimate_quality, parse_jpeg_header
    root = _repo(tmp_path, q=92, q_normal=60)
    monkeypatch.setattr(B, "repo_path", lambda rel, r=None: root / rel)
    monkeypatch.setattr(B, "REPO_ROOT", root)
    monkeypatch.setattr(B, "Guard", _NoWait)
    cfg = _cfg()
    assert B.cmd_plan(cfg) == 0 and B.cmd_quality(cfg) == 0
    r9 = json.loads((root / CFG["staging_root"] / "records" / "r9_quality.json").read_text(encoding="utf-8"))
    info = parse_jpeg_header(_jpeg(64, 64, q=60))
    want = int(estimate_quality(next(t for tq, t in info["dqt"] if tq == 0)))
    assert r9["quality"] == want and r9["populations"]["defect"]["min"] > r9["populations"]["normal"]["min"]


def test_단계는_스테이징이_동결_자리면_아무것도_쓰기_전에_멈춘다(repo):
    """쓰기 가드가 단계에서 실제로 막는다 — 스테이징을 동결 명부의 자리로 돌리면 계획이 쓰지 않고 멈춘다(06c I-1 ④)."""
    from data.vt.guard import WriteForbidden
    cfg = _cfg()
    cfg["staging_root"] = "data/interim/manifest_v1/vt_build_r1"
    with pytest.raises(WriteForbidden):
        B.cmd_plan(cfg)
    assert not (repo / "data/interim/manifest_v1").exists()
