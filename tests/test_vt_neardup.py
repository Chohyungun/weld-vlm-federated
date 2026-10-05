"""VT-5i — 근사 중복 기술자 · 후보 생성기 · 국소 특징 검증. 실물을 읽지 않는다(합성 영상)."""
from __future__ import annotations

import copy
import csv
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image

import data.vt.neardup as nd
from data.vt.config import load_config

CFG = load_config()


def _texture(h, w, seed):
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 3)
    a = (a - a.min()) / (a.max() - a.min()) * 200 + 20
    for _ in range(40):
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        cv2.rectangle(a, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))),
                      float(rng.integers(0, 255)), -1)
    return np.clip(a, 0, 255).astype(np.uint8)


def _jpeg(gray, q=92, gain=1.0):
    rgb = np.clip(np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) * gain, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def _cos(a, b):
    return float(np.dot(a, b))


def test_기술자는_밝기_이동과_배율에_흔들리지_않고_다른_장면은_멀다():
    t = _texture(360, 640, 1)
    d = nd.coarse_desc(t, (64, 36), (1.0, 4.0))
    assert abs(np.linalg.norm(d) - 1) < 1e-5
    assert _cos(d, nd.coarse_desc(t.astype(np.float32) * 1.1 + 15, (64, 36), (1.0, 4.0))) > 0.99
    assert _cos(d, nd.coarse_desc(_texture(360, 640, 2), (64, 36), (1.0, 4.0))) < 0.5


def test_기술자는_행마다_열마다의_밝기_띠를_지운다():
    t = _texture(360, 640, 11).astype(np.float32)
    rng = np.random.default_rng(12)
    banded = t + rng.normal(0, 40, size=(360, 1)) + rng.normal(0, 40, size=(1, 640))
    d0 = nd.coarse_desc(t, (64, 36), (1.0, 4.0))
    assert _cos(d0, nd.coarse_desc(banded, (64, 36), (1.0, 4.0))) > 0.98


def test_창은_배율마다_간격_절반으로_깔리고_경계에_붙는다():
    boxes = nd.window_boxes(3840, 2160, [1.0, 1.5, 2.0], (1280, 720))
    sizes = {(b[2] - b[0], b[3] - b[1]) for b in boxes}
    assert sizes == {(1280, 720), (853, 480), (640, 360)}
    assert all(0 <= b[0] and b[2] <= 3840 and 0 <= b[1] and b[3] <= 2160 for b in boxes)
    assert sum(1 for b in boxes if b[2] - b[0] == 1280) == 25
    assert sum(1 for b in boxes if b[2] - b[0] == 640) == 121
    assert (3840 - 853, 2160 - 480, 3840, 2160) in boxes          # 마지막 창은 경계에 붙는다
    assert nd.window_boxes(1000, 600, [1.0], (1280, 720)) == []


def test_G_id_는_같은_폴더_안_id_차_span_이내만():
    folder_of = {1: "A", 2: "A", 3: "A", 10: "A", 4: "B", 5: "B"}
    assert nd.id_pairs(folder_of, 2) == {(1, 2), (1, 3), (2, 3), (4, 5)}


def test_G_nn_은_심은_근사_중복_짝을_낸다():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 32)).astype(np.float32)
    X[7] = X[3] + 0.01 * rng.normal(size=32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = np.arange(100, 150)
    got = nd.nn_pairs(ids, X, 1, block=8)
    assert (103, 107) in got and all(a < b for a, b in got)


def test_G_x_는_사진마다_한_창으로_세고_사진_안의_창_번호를_든다():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(4, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    xt = nd.CrossTopK(q, 2)
    w = rng.normal(size=(9, 16)).astype(np.float32) * 0.01
    w[5] = q[0]                       # 사진 77 의 셋째 창
    w[6] = q[0] * 0.99                # 같은 사진의 넷째 창 — 사진은 하나로 센다
    w[1] = q[0] * 0.5 + q[1] * 0.5    # 사진 76 의 둘째 창
    owner = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2])
    xt.update(np.array([76, 77, 78]), w, owner)
    assert xt.photo[0].tolist() == [77, 76] and xt.win[0].tolist() == [2, 1]
    xt.update(np.array([90]), q[[0]] * 1.0001, np.array([0]))
    assert xt.photo[0, 0] == 90 and xt.photo[0, 1] == 77


def _feat(raw, normal=False):
    return nd.image_features(raw, is_normal=normal, rot90_k=0, cfg=CFG)


def _v(fa, fb):
    v = CFG["neardup"]["verify"]
    return nd.verify_pair(fa["orb_xy"], fa["orb_desc"], fa["orb_f"], fb["orb_xy"], fb["orb_desc"], fb["orb_f"],
                          ratio=float(v["ratio"]), reproj=float(v["ransac_reproj_px"]))


def test_겹친_창_다시_인코딩_밝기는_정합점이_많고_다른_장면은_적다():
    T = _texture(1000, 1700, 3)
    a = _feat(_jpeg(T[0:720, 0:1280]))
    b = _feat(_jpeg(T[60:780, 100:1380], q=70, gain=1.1))
    c = _feat(_jpeg(_texture(720, 1280, 4)))
    pos, neg = _v(a, b), _v(a, c)
    assert pos["inliers"] >= 40 and abs(pos["scale"] - 1) < 0.05
    assert neg["inliers"] < 12 and pos["inliers"] > 4 * max(1, neg["inliers"])


def test_결함_크롭과_그를_담은_정상_사진은_축소_해독이_달라도_배율_1_로_맞는다():
    T = _texture(1440, 2560, 5)
    crop = _feat(_jpeg(T[400:1120, 900:2180]))
    photo = _feat(_jpeg(T), normal=True)
    assert photo["orb_f"] == pytest.approx(4.0) and crop["orb_f"] == pytest.approx(2.0)
    r = _v(crop, photo)
    assert r["inliers"] >= 20 and abs(r["scale"] - 1) < 0.1
    assert "win_desc" in photo and photo["win_desc"].shape[1] == 32 * 18 and "desc32" in crop


# ------------------------------------------------------------------------------------------
# 종단간 — features → merge → candidates → verify (작은 합성 스테이징)
# ------------------------------------------------------------------------------------------
class _NoWait:
    @classmethod
    def from_config(cls, *a, **k):
        return cls()

    def wait(self):
        return {"c_free_gb": 99.0, "mem_free_gb": 99.0}


def test_종단간_제외할_장도_후보에_들고_겹친_결함_짝이_검증된다(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    staging = root / "data/interim/vt_build_r1"
    raw = root / "data/raw/aihub71761_vt/_zips"
    (staging / "parse").mkdir(parents=True)
    (staging / "records").mkdir(parents=True)
    raw.mkdir(parents=True)
    T = _texture(1000, 1700, 7)
    imgs = {
        11: ("결함_1. 기공", T[0:720, 0:1280], False),
        12: ("결함_1. 기공", T[50:770, 80:1360], False),        # 11 과 겹친다 — id 차 1
        13: ("결함_1. 기공", _texture(720, 1280, 8), False),     # 라벨 공간 밖으로 뺀 장
        20: ("정상", _texture(1440, 2560, 9), True),
        21: ("정상", _texture(1440, 2560, 10), True),
    }
    labels = []
    for vid, (folder, g, normal) in imgs.items():
        zname = f"TS_VTST_{folder}.zip"
        with zipfile.ZipFile(raw / zname, "a") as zf:
            zf.writestr(f"/VT_ST_00_{vid}.jpg", _jpeg(g))
        labels.append({"vt_id": vid, "image_id": f"aihub71761_vt:{vid}", "folder": folder, "src_zip": zname,
                       "file_stem": f"VT_ST_00_{vid}", "w": g.shape[1], "h": g.shape[0], "is_normal": normal,
                       "label_excluded": "out_of_label_space" if vid == 13 else None, "anns": []})
    (staging / "parse" / "labels.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in labels),
                                                    encoding="utf-8")
    cfg = copy.deepcopy(CFG)
    monkeypatch.setattr(nd, "repo_path", lambda rel, root_=None: root / rel)
    monkeypatch.setattr(nd, "REPO_ROOT", root)
    monkeypatch.setattr(nd, "Guard", _NoWait)
    # 원천 복사 대조가 지나지 않았으면 원천을 읽지 않는다
    with pytest.raises(SystemExit):
        nd.cmd_features(cfg, workers=1, chunk=2)
    (staging / "copy").mkdir()
    (staging / "copy" / "copy_verify.json").write_text('{"all_ok": true, "image_content_evidence_all": true}',
                                                        encoding="utf-8")
    (staging / "copy" / "copy_record.jsonl").write_text('{"name": "x"}\n', encoding="utf-8")
    assert nd.cmd_features(cfg, workers=1, chunk=2) == 0
    meta = json.loads((staging / "neardup" / "features_meta.json").read_text(encoding="utf-8"))
    assert meta["n"] == 5 and meta["n_defect"] == 3 and meta["windows_total"] > 0
    assert nd.cmd_candidates(cfg, threads=1) == 0
    cm = json.loads((staging / "neardup" / "candidates_meta.json").read_text(encoding="utf-8"))
    with (staging / "neardup" / "candidates.csv").open(encoding="utf-8") as fh:
        cand = {(int(r["a_id"]), int(r["b_id"])): r["gens"] for r in csv.DictReader(fh)}
    assert "id" in cand[(11, 12)] and "id" in cand[(12, 13)]        # 뺀 장도 후보에 든다
    assert cm["excluded_images_in_some_pair"] == 1
    assert any("x" in g for g in cand.values())
    # 새 배선 — 무작위 대조군이 후보에 들고(06c N-1 · N-9) 생성기별 수가 메타와 같다
    assert any("rand" in g for g in cand.values())
    assert cm["pairs"] == len(cand) and cm["by_generator"]["rand"] == sum("rand" in g for g in cand.values())
    assert nd.cmd_verify(cfg, workers=1, chunk=3) == 0
    with (staging / "neardup" / "scores.csv").open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["a_id", "b_id", "gens", "matches", "inliers", "scale", "cover_a", "cover_b", "tx", "ty"]
    assert all(len(r) == 10 for r in rows)
    sc = {(int(r[0]), int(r[1])): r for r in rows[1:]}
    assert set(sc) == set(cand)
    assert int(sc[(11, 12)][4]) >= 40 and int(sc[(12, 13)][4]) < 12
    # 겹친 짝(B = A − (80, 50))의 이동량과 덮는 범위가 실린다
    assert abs(float(sc[(11, 12)][8]) + 80) < 4 and abs(float(sc[(11, 12)][9]) + 50) < 4
    assert float(sc[(11, 12)][6]) > 0.4
    sm = json.loads((staging / "neardup" / "scores_meta.json").read_text(encoding="utf-8"))
    assert sm["rows"] == len(cand) == cm["pairs"]
    # 후보 · 검증의 결속 — 입력 해시 · 모듈 해시 · OpenCV 판(06c N-4)
    assert set(cm["inputs"]) == {"chunks_meta_sha256", "labels_jsonl_sha256", "module_sha256", "opencv"}
    vm = json.loads((staging / "neardup" / "verify_chunks" / "_meta.json").read_text(encoding="utf-8"))
    assert vm["module_sha256"] == cm["inputs"]["module_sha256"] and vm["opencv"]
    # 대조 단계(06c N-2 · N-3) — 양성 대조는 폴더마다 R-8b 가 고른 장, 결함–정상 무작위 짝은 결함 폴더마다 정한 수
    c = copy.deepcopy(cfg)
    c["neardup_control"]["cross_random"]["per_defect_folder"] = 7
    assert nd.cmd_control(c) == 0
    ctl = staging / "neardup" / "control"
    with (ctl / "positive.csv").open(encoding="utf-8") as fh:
        pos = list(csv.DictReader(fh))
    with (ctl / "cross_random.csv").open(encoding="utf-8") as fh:
        crs = list(csv.DictReader(fh))
    assert len(pos) == 5 and len(crs) == 7
    assert all(int(r["inliers"]) >= 40 for r in pos)            # 옮긴 겹침은 ORB 가 잡는다(합성 무늬)
    meta = json.loads((ctl / "control_meta.json").read_text(encoding="utf-8"))
    assert set(meta["judgement"]) == {"결함_1. 기공", "정상"} and meta["scores_sha256"]
    with pytest.raises(SystemExit):                              # 대조 산출이 있으면 다시 쓰지 않는다
        nd.cmd_control(c)


def test_실행은_설정의_top_k_와_비_문턱을_쓴다(tmp_path, monkeypatch):
    """설정의 값이 실행에 닿는다 — G-x 의 top_k · G-nn 의 nn_k · 검증의 비 문턱을 바꾸면 산출이 바뀐다(06c I-3 · N-9)."""
    root = tmp_path / "repo"
    staging = root / "data/interim/vt_build_r1"
    raw = root / "data/raw/aihub71761_vt/_zips"
    (staging / "parse").mkdir(parents=True)
    (staging / "copy").mkdir(parents=True)
    raw.mkdir(parents=True)
    labels = []
    for vid, folder, normal, seed in [(1, "결함_1. 기공", False, 51), (2, "결함_1. 기공", False, 52),
                                      (3, "결함_1. 기공", False, 53), (4, "결함_1. 기공", False, 54),
                                      (5, "정상", True, 55), (6, "정상", True, 56), (7, "정상", True, 57)]:
        g = _texture(1440, 2560, seed) if normal else _texture(720, 1280, seed)
        zname = f"TS_VTST_{folder}.zip"
        with zipfile.ZipFile(raw / zname, "a") as zf:
            zf.writestr(f"/VT_ST_00_{vid}.jpg", _jpeg(g))
        labels.append({"vt_id": vid, "image_id": f"aihub71761_vt:{vid}", "folder": folder, "src_zip": zname,
                       "file_stem": f"VT_ST_00_{vid}", "w": g.shape[1], "h": g.shape[0], "is_normal": normal,
                       "label_excluded": None, "anns": []})
    (staging / "parse" / "labels.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in labels),
                                                    encoding="utf-8")
    (staging / "copy" / "copy_verify.json").write_text('{"all_ok": true, "image_content_evidence_all": true}',
                                                        encoding="utf-8")
    (staging / "copy" / "copy_record.jsonl").write_text('{"name": "x"}\n', encoding="utf-8")
    monkeypatch.setattr(nd, "repo_path", lambda rel, root_=None: root / rel)
    monkeypatch.setattr(nd, "REPO_ROOT", root)
    monkeypatch.setattr(nd, "Guard", _NoWait)
    cfg = copy.deepcopy(CFG)
    assert nd.cmd_features(cfg, workers=1, chunk=4) == 0

    def run(c):
        assert nd.cmd_candidates(c, threads=1) == 0
        return json.loads((staging / "neardup" / "candidates_meta.json").read_text(encoding="utf-8"))
    base = run(cfg)
    assert base["cross_rows"] == 4 * 3                     # 결함 넷 × 정상 사진 셋(top_k 3)
    c1 = copy.deepcopy(cfg)
    c1["neardup"]["cross"]["top_k"] = 1
    c1["neardup"]["nn_k"] = 1
    one = run(c1)
    assert one["cross_rows"] == 4 and one["by_generator"]["nn"] < base["by_generator"]["nn"]
    # 비 문턱 0 이면 어떤 정합도 남지 않는다 — 검증이 설정의 비를 쓴다
    run(cfg)
    c2 = copy.deepcopy(cfg)
    c2["neardup"]["verify"]["ratio"] = 0.0
    assert nd.cmd_verify(c2, workers=1, chunk=50) == 0
    with (staging / "neardup" / "scores.csv").open(encoding="utf-8") as fh:
        assert all(r["matches"] == "0" for r in csv.DictReader(fh))


def test_돌림은_세로_정상에만이다():
    assert nd._rot({"is_normal": True, "w": 2160, "h": 3840}, CFG) == 1
    assert nd._rot({"is_normal": True, "w": 3840, "h": 2160}, CFG) == 0
    assert nd._rot({"is_normal": False, "w": 720, "h": 1280}, CFG) == 0


def test_기술자는_행마다의_밝기_띠만으로도_흔들리지_않는다():
    """행 평균 제거를 빼면 행마다의 큰 밝기 띠가 기술자를 끌고 간다(06c N05)."""
    t = _texture(360, 640, 61).astype(np.float32)
    rng = np.random.default_rng(62)
    rows = t + rng.normal(0, 120, size=(360, 1))
    assert _cos(nd.coarse_desc(t, (64, 36), (1.0, 4.0)), nd.coarse_desc(rows, (64, 36), (1.0, 4.0))) > 0.98


def test_재투영_문턱은_짝의_순서에_달리지_않는다(monkeypatch):
    """`cv2` 에 넘어가는 문턱을 직접 잡는다 — 두 순서에서 같고 거친 쪽 축소 배율의 4 화소다(06b I-1 · I-3)."""
    seen = []
    real = nd.cv2.estimateAffinePartial2D

    def spy(a, b, **kw):
        seen.append(kw["ransacReprojThreshold"])
        return real(a, b, **kw)
    monkeypatch.setattr(nd.cv2, "estimateAffinePartial2D", spy)
    T = _texture(1440, 2560, 21)
    crop = _feat(_jpeg(T[400:1120, 900:2180]))
    photo = _feat(_jpeg(T), normal=True)
    ab, ba = _v(crop, photo), _v(photo, crop)
    assert seen == [16.0, 16.0]
    assert ab["inliers"] >= 20 and ba["inliers"] >= 20
    assert abs(ab["scale"] - 1) < 0.1 and abs(ba["scale"] - 1) < 0.1


def test_근사_중복_설정은_등록값과_같다():
    """4판 5-4 와 5판 1절의 등록값. 설정을 바꾸면 이 시험과 명세를 함께 고친다(06b I-3 · 06c N-8)."""
    assert CFG["neardup"] == {
        "id_span": 5, "nn_k": 5, "descriptor_size": [64, 36], "dog_sigmas": [1.0, 4.0],
        "cross": {"descriptor_size": [32, 18], "window_scales": [1.0, 1.5, 2.0], "top_k": 3},
        "random_control": {"seed": 20261005, "per_folder": 2000},
        "verify": {"orb_features": 1000, "defect_reduce": 2, "normal_reduce": 4, "ratio": 0.8,
                   "ransac_reproj_px": 4.0}}


def test_G_nn_의_동률은_자리_순서로_정해진다():
    X = np.zeros((6, 4), np.float32)
    X[:, 0] = 1.0                                       # 전부 같은 기술자 — 모두 동률
    got = nd.nn_pairs(np.arange(10, 16), X, 2, block=4)
    assert got == nd.nn_pairs(np.arange(10, 16), X, 2, block=3)
    # 행마다 자기를 뺀 가장 작은 자리 둘이 이웃이다
    assert got == {(10, 11), (10, 12), (11, 12), (10, 13), (11, 13), (10, 14), (11, 14), (10, 15), (11, 15)}
    # 등록값 k = 5 · 큰 행에서도 — 정렬되지 않은 부분 선택은 여기서 다른 답을 낸다(06c X13)
    Y = np.zeros((400, 4), np.float32)
    Y[:, 0] = 1.0
    big = nd.nn_pairs(np.arange(400), Y, 5, block=64)
    want = {(min(i, j), max(i, j)) for i in range(400) for j in [x for x in range(7) if x != i][:5]}
    assert big == want


def test_돌린_정상의_축소_해독은_np_rot90_과_같고_크기가_바뀐다():
    raw = _jpeg(_texture(1920, 960, 31))
    g0, w0, h0 = nd._gray_draft(raw, 8, 0)
    g1, w1, h1 = nd._gray_draft(raw, 8, 1)
    assert (w1, h1) == (h0, w0) and np.array_equal(g1, np.rot90(g0, k=1))
    f = nd.image_features(raw, is_normal=True, rot90_k=1, cfg=CFG)
    assert f["frame_wh"] == (1920, 960) and int(f["win_box"][:, 2].max()) == 1920


def test_G_rand_는_같은_폴더의_무작위_짝이고_결정적이다():
    folder_of = {i: ("A" if i < 50 else "B") for i in range(80)}
    a = nd.random_pairs(folder_of, ["A", "B"], 7, 30)
    assert a == nd.random_pairs(folder_of, ["A", "B"], 7, 30) and len(a) == 60
    assert all(x < y and folder_of[x] == folder_of[y] for x, y in a)
    assert a != nd.random_pairs(folder_of, ["A", "B"], 8, 30)


def test_겹친_창의_검증은_덮는_범위와_이동량을_낸다():
    T = _texture(1000, 1700, 41)
    a = _feat(_jpeg(T[0:720, 0:1280]))
    b = _feat(_jpeg(T[60:780, 100:1380]))
    v = CFG["neardup"]["verify"]
    r = nd.verify_pair(a["orb_xy"], a["orb_desc"], a["orb_f"], b["orb_xy"], b["orb_desc"], b["orb_f"],
                       ratio=float(v["ratio"]), reproj=float(v["ransac_reproj_px"]),
                       wh_a=a["frame_wh"], wh_b=b["frame_wh"])
    assert abs(r["tx"] + 100) < 4 and abs(r["ty"] + 60) < 4          # B 의 좌표 = A 의 좌표 − (100, 60)
    assert r["cover_a"] > 0.4 and r["cover_b"] > 0.4
    # 겹친 범위(1180 × 660 / 1280 × 720 ≈ 0.845)를 넘지 못하고, 두 장에서 비슷하다
    assert r["cover_a"] < 0.86 and r["cover_b"] < 0.86 and abs(r["cover_a"] - r["cover_b"]) < 0.1
    assert nd.verify_pair(a["orb_xy"], a["orb_desc"], a["orb_f"], b["orb_xy"], b["orb_desc"], b["orb_f"],
                          ratio=0.8, reproj=4.0)["cover_a"] == ""       # 프레임 크기를 주지 않으면 비운다


def test_이어_하던_폴더가_다른_입력에_묶였으면_멈춘다(tmp_path):
    p = tmp_path / "_meta.json"
    nd.bind_meta(p, {"a": 1})
    nd.bind_meta(p, {"a": 1})
    with pytest.raises(SystemExit):
        nd.bind_meta(p, {"a": 2})


def test_양성_대조의_판정은_무작위_짝의_바닥을_넘는_몫이다():
    rule = {"rand_percentile": 0.99, "min_share": 0.9}
    rand = list(range(100))                                       # 99 백분위 = 99
    ok = nd.control_judgement([150] * 95 + [10] * 5, rand, rule)
    assert ok["stands"] and ok["rand_floor"] == 99 and ok["share_above"] == 0.95
    no = nd.control_judgement([150] * 85 + [10] * 15, rand, rule)
    assert no["stands"] is False
    assert nd.control_judgement([], rand, rule)["stands"] is None


def test_대조_설정은_등록값과_같다():
    """6판 1절의 등록값. 바꾸면 이 시험과 명세를 함께 고친다."""
    assert CFG["neardup_control"] == {"positive_rule": {"rand_percentile": 0.99, "min_share": 0.9},
                                      "cross_random": {"seed": 20261006, "per_defect_folder": 2000}}
