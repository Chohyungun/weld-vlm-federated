"""VT 구현 회차 1 — 실물 전제 시험 (4판 5-5).

**`VT_REAL=1` 로 켤 때만 돈다.** 켜지 않으면 그 사유로 건너뛴다. **켰는데 스테이징 산출이 없으면 실패한다** —
자산 부재를 환경 잡음으로 넘기지 않는다. 게이트의 전량 시험이 6.6 만 장을 읽지 않게 기본은 끈다.

    VT_REAL=1 uv run python -X utf8 -m pytest -q tests/test_vt_real.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from data.vt.config import load_config, repo_path
from data.vt.records import read_csv_checked

pytestmark = pytest.mark.skipif(os.environ.get("VT_REAL") != "1", reason="실물 전제 — VT_REAL=1 로 따로 돈다")

CFG = load_config()
ST = repo_path(CFG["staging_root"])
REC = ST / "records"


def _json(p: Path) -> dict:
    if not p.exists():
        pytest.fail(f"스테이징 산출이 없다 — 자산 부재: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def test_원천_스무_개가_세_참조와_맞고_다시_해시하면_기록과_같다():
    v = _json(ST / "copy" / "copy_verify.json")
    assert v["all_ok"] and v["n"] == 20 and v["label_sha256_checked"] == 10
    assert v["server_md5_control"].startswith("라벨 zip") and v["server_md5_checked"] == 20
    assert all(z.get("server_md5_ok") and z["bytes_ok"] and z.get("on_disk_ok") for z in v["zips"])
    recs = {}
    for line in (ST / "copy" / "copy_record.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[r["name"]] = r
    for d, pre in ((repo_path(CFG["copy"]["raw_zip_dir"]), "image_prefixes"),
                   (repo_path(CFG["copy"]["label_zip_dir"]), "label_prefixes")):
        files = sorted(p for p in d.glob("*.zip") if p.name.startswith(tuple(CFG["copy"][pre])))
        assert len(files) == 10 and not list(d.glob("*.part"))
        for p in files:
            assert p.stat().st_size == recs[p.name]["bytes"]
    # 다시 해시 — 라벨 zip 열 개 전부와 원천 zip 가운데 작은 둘(전수 재해시는 회차 보고의 별도 점검)
    from data.vt.copy import hash_file
    lab = sorted(repo_path(CFG["copy"]["label_zip_dir"]).glob("*.zip"))
    raw = sorted(repo_path(CFG["copy"]["raw_zip_dir"]).glob("*.zip"), key=lambda q: q.stat().st_size)[:2]
    for p in lab + raw:
        sha, md5, n = hash_file(p, 1 << 22)
        assert (sha, md5, n) == (recs[p.name]["sha256"], recs[p.name]["md5"], recs[p.name]["bytes"]), p.name


def test_계획의_회계와_기록의_해시():
    s = _json(REC / "plan_summary.json")
    assert s["accounting_ok"] and s["labels_parsed"] == s["kept"] + s["excluded"]
    crops = read_csv_checked(REC / "crop_boxes.csv", s["records"]["crop_boxes.csv"])
    excl = read_csv_checked(REC / "exclusions.csv", s["records"]["exclusions.csv"])
    assert len(crops) == s["kept"] and len(excl) == s["excluded"]
    assert not {r["image_id"] for r in crops} & {r["image_id"] for r in excl}
    for r in crops:
        w, h = int(r["x1"]) - int(r["x0"]), int(r["y1"]) - int(r["y0"])
        fw, fh = (int(r["src_h"]), int(r["src_w"])) if r["rot90_k"] == "1" else (int(r["src_w"]), int(r["src_h"]))
        assert (w, h) == (1280, 720) and 0 <= int(r["x0"]) and int(r["x1"]) <= fw and int(r["y1"]) <= fh
        assert r["provenance"] in ("N-crop", "N-tile", "N-band")
        assert int(r["x0"]) % 8 == 0 and int(r["y0"]) % 8 == 0 or r["provenance"] == "N-crop"


def test_VT_1d_스테이징_기록에_분할_칸이_없다():
    for p in [REC / "crop_boxes.csv", REC / "exclusions.csv", REC / "encoded_tiles.csv"]:
        with p.open(encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        assert not {"split", "eval_subset", "client", "group_id"} & set(header), p.name
    with (ST / "parse" / "labels.jsonl").open(encoding="utf-8") as fh:
        keys = set(json.loads(fh.readline()))
    assert not {"split", "eval_subset", "client", "group_id"} & keys


def test_영상은_전부_1280x720_이고_계획과_장_수가_같고_해시가_맞는다():
    e = _json(REC / "encode_summary.json")
    s = _json(REC / "plan_summary.json")
    assert e["missing"] == 0 and e["file_hash_mismatch"] == 0 and e["image_members_equal_labels"]
    assert e["encoded"] == s["kept"] and e["sizes"] == {"1280x720": s["kept"]}
    tiles = read_csv_checked(REC / "encoded_tiles.csv", e["records"]["encoded_tiles.csv"])
    root = ST / "tiles"
    # 해시는 sha256 순서 앞 2,000 장을 다시 읽어 맞댄다(전수는 회차 보고의 별도 점검)
    sample = sorted(tiles, key=lambda r: hashlib.sha256(r["image_id"].encode()).hexdigest())[:2000]
    for r in sample:
        data = (root / r["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r["sha256"]
        im = Image.open(io.BytesIO(data))
        assert im.size == (1280, 720) and im.mode == CFG["encode"]["mode"] and not im.getexif()
        assert "comment" not in im.info and "icc_profile" not in im.info


def test_재인코딩_품질은_설정값이고_R_9_기록과_같다():
    from scripts.measure_jpeg_fingerprint import estimate_quality, parse_jpeg_header
    r9 = _json(REC / "r9_quality.json")
    assert r9["quality"] == CFG["encode"]["quality"] and not r9["stop"]
    e = _json(REC / "encode_summary.json")
    tiles = read_csv_checked(REC / "encoded_tiles.csv", e["records"]["encoded_tiles.csv"])
    # 기록은 id 순이라 앞 몇백 장은 한 클래스다 — 해시 순서로 뽑아 클래스 · 출처 · 돌림을 섞는다(06b m-4)
    sample = sorted(tiles, key=lambda r: hashlib.sha256(r["image_id"].encode()).hexdigest())[:400]
    heads = set()
    for r in sample:
        data = (ST / "tiles" / r["path"]).read_bytes()
        info = parse_jpeg_header(data[:1 << 16])
        lum = next(t for tq, t in info["dqt"] if tq == 0)
        assert abs(estimate_quality(lum) - CFG["encode"]["quality"]) <= 1
        heads.add(data[:data.index(b"\xff\xda")])          # SOS 앞의 머리 — 양자화 · 허프만 · 표본화가 한 가지다
    assert len(heads) == 1


def test_근사_중복_후보는_전_장을_덮고_점수는_후보마다_하나():
    meta = _json(ST / "neardup" / "features_meta.json")
    s = _json(REC / "plan_summary.json")
    assert meta["n"] == s["labels_parsed"]
    cm = _json(ST / "neardup" / "candidates_meta.json")
    assert cm["excluded_images_in_some_pair"] > 0
    with (ST / "neardup" / "candidates.csv").open(encoding="utf-8") as fh:
        n_cand = sum(1 for _ in fh) - 1
    with (ST / "neardup" / "scores.csv").open(encoding="utf-8") as fh:
        n_score = sum(1 for _ in fh) - 1
    assert n_cand == cm["pairs"] == n_score


def test_원천_zip_의_멤버가_라벨_전수와_같다():
    with (ST / "parse" / "labels.jsonl").open(encoding="utf-8") as fh:
        labs = [json.loads(x) for x in fh if x.strip()]
    by_zip: dict[str, set[str]] = {}
    for d in labs:
        by_zip.setdefault(d["src_zip"], set()).add(d["file_stem"])
    raw = repo_path(CFG["copy"]["raw_zip_dir"])
    for z, stems in by_zip.items():
        with zipfile.ZipFile(raw / z) as zf:
            have = {Path(n).stem for n in zf.namelist() if n.lower().endswith(".jpg")}
        assert stems == have, z


def test_R_8_산출은_3판_측정을_다시_내고_귀결은_등록된_문장이다():
    r8 = _json(ST / "r8" / "r8.json")
    assert r8["reproduces_3rd_edition"] and all(r8["reproduces_3rd_edition"].values())
    assert set(r8["folders"]) == set(CFG["labels"]["folders"])
    assert r8["consequence"].startswith(("양성 대조가 선다", "결함 폴더의 '닮지 않았다' 를 근거로 쓰지 않고 4판 9절 1 로 올린다"))

