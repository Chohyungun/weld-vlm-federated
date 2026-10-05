"""R-8 — 양성 대조 측정의 성질. 실물을 읽지 않는다."""
from __future__ import annotations

import io
import random

import cv2
import numpy as np
from PIL import Image

from data.vt.r8 import feat, neighbor_and_random, pair_stats, similar, synth_pair


def _jpeg(seed, h=720, w=1280):
    """비드 띠 하나와 저주파 무늬 — 구조가 창의 이동(5 %)보다 크다. 무늬가 이동보다 잘면 겹친 창도
    상관이 0 에 가깝다(측정의 성질 — R-8b 가 실물에서 이것을 드러낸다)."""
    rng = np.random.default_rng(seed)
    a = cv2.GaussianBlur(rng.random((h // 8, w // 8)).astype(np.float32), (0, 0), 10)
    a = (a - a.min()) / (a.max() - a.min())
    yy = np.arange(h // 8)[:, None]
    c, half = rng.integers(20, h // 8 - 20), rng.integers(5, 15)
    a = a + 1.5 * (np.abs(yy - c) < half)
    a = cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    a = ((a - a.min()) / (a.max() - a.min()) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.repeat(a[:, :, None], 3, axis=2)).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_겹친_창의_합성_짝은_닮았고_무작위_짝은_닮지_않았다():
    F, sp = {}, []
    for k in range(12):
        a, b = synth_pair(_jpeg(k), 0.9, 0.05, 90)
        F[2 * k], F[2 * k + 1] = feat(a), feat(b)
        sp.append((2 * k, 2 * k + 1))
    R = {100 + k: feat(_jpeg(1000 + k)) for k in range(12)}
    rp = [(100 + k, 100 + (k + 1) % 12) for k in range(12)]
    s_sy, s_rd = pair_stats(sp, F), pair_stats(rp, R)
    # 겹친 창은 무작위보다 가깝고 상관이 높다. R-5 의 "절반 이하" 를 지나는지는 무늬의 크기에 달린다 —
    # 이 합성에서 16.5 대 32 로 경계에 있다. 그 판정은 실물(R-8b)이 낸다. 여기서는 성질만 고정한다.
    assert s_sy["dhash_median"] < 0.75 * s_rd["dhash_median"]
    assert s_sy["corr_median"] > 0.5 and abs(s_rd["corr_median"]) < 0.3
    assert not similar(s_rd, s_rd)


def test_옮기지_않은_다시_인코딩_짝은_규칙상_닮았다():
    F, sp = {}, []
    for k in range(12):
        a, b = synth_pair(_jpeg(k), 0.9, 0.0, 90)
        F[2 * k], F[2 * k + 1] = feat(a), feat(b)
        sp.append((2 * k, 2 * k + 1))
    R = {100 + k: feat(_jpeg(1000 + k)) for k in range(12)}
    rp = [(100 + k, 100 + (k + 1) % 12) for k in range(12)]
    assert similar(pair_stats(sp, F), pair_stats(rp, R))


def test_합성_창은_원본의_90_퍼센트이고_5_퍼센트_옮긴다():
    a, b = synth_pair(_jpeg(1, 720, 1280), 0.9, 0.05, 90)
    assert Image.open(io.BytesIO(a)).size == (1152, 648) == Image.open(io.BytesIO(b)).size


def test_이웃과_무작위_짝은_같은_시드에서_같고_같은_크기에서만_뽑는다():
    F = {i: {"size": (160, 90) if i % 3 else (240, 135)} for i in [1, 2, 3, 4, 5, 7, 8, 10]}
    nb1, r1 = neighbor_and_random(F, random.Random(20261004))
    nb2, r2 = neighbor_and_random(F, random.Random(20261004))
    assert nb1 == [(1, 2), (2, 3), (3, 4), (4, 5), (7, 8)] and (r1, nb1) == (r2, nb2)
    assert all(F[a]["size"] == F[b]["size"] and a != b for a, b in r1)


def test_판정은_무작위_중앙값의_절반_이하():
    assert similar({"dhash_median": 16.0}, {"dhash_median": 32.0})
    assert not similar({"dhash_median": 16.5}, {"dhash_median": 32.0})
    assert not similar({"dhash_median": None}, {"dhash_median": 32.0})


def test_결함_폴더_순서는_설정에서_나오고_3판의_순서와_같다():
    from data.vt.config import load_config
    from data.vt.r8 import defect_order
    assert defect_order(load_config()) == ["결함_1. 기공", "결함_2. 용입부족", "결함_3. 융합불량", "결함_4. 언더컷"]


def test_합성_짝은_같은_바이트가_아니고_옮긴_자리에서_맞대면_더_가깝다():
    """창을 옮기지 않으면 같은 바이트 짝이 되어 양성 대조가 동어반복이 된다 — 그것을 막는다(06b I-5)."""
    from data.vt.config import load_config
    r8 = load_config()["r8"]
    raw = _jpeg(5)
    a, b = synth_pair(raw, float(r8["synth_window_frac"]), float(r8["synth_shift_frac"]), int(r8["synth_quality"]))
    assert a != b
    A = np.asarray(Image.open(io.BytesIO(a)).convert("L"), dtype=float)
    Bm = np.asarray(Image.open(io.BytesIO(b)).convert("L"), dtype=float)
    W, H = Image.open(io.BytesIO(raw)).size
    dx, dy = int(W * float(r8["synth_shift_frac"])), int(H * float(r8["synth_shift_frac"]))
    shifted = np.abs(A[dy:, dx:] - Bm[:A.shape[0] - dy, :A.shape[1] - dx]).mean()
    as_is = np.abs(A - Bm).mean()
    assert shifted < as_is / 3


def test_R_8_설정은_등록값과_같다():
    from data.vt.config import load_config
    assert load_config()["r8"] == {"random_seed": 20261004, "synth_per_folder": 200, "synth_rank_salt": "20261004|r8b|",
                                   "synth_window_frac": 0.9, "synth_shift_frac": 0.05, "synth_quality": 90}


def _f(nb, sy):
    return {"neighbors_similar": nb, "synthetic_similar": sy}


def test_귀결은_R_8b_가_떨어진_결함_폴더를_근거로_쓰지_않는다():
    """R-8a 가 서도 합성 겹침을 잡지 못한 결함 폴더의 "닮지 않았다" 는 근거가 아니다(5판 2절 · 06c I-5)."""
    from data.vt.r8 import consequence
    d = ["A", "B"]
    assert "근거로 쓴다" in consequence({"N": _f(True, True), "A": _f(False, True), "B": _f(False, True)}, "N", d)
    c = consequence({"N": _f(True, True), "A": _f(False, True), "B": _f(False, False)}, "N", d)
    assert "근거로 쓰지 않는다" in c and "B" in c and "A" not in c.split("(")[1]
    assert "거의 같은 구도가 아니다" in consequence({"N": _f(False, True), "A": _f(False, False),
                                                  "B": _f(False, False)}, "N", d)
    assert "잡지 못한다" in consequence({"N": _f(False, False), "A": _f(False, False), "B": _f(False, False)}, "N", d)


def test_r8_main_은_3판_재현이_어긋나면_3_으로_끝난다(tmp_path, monkeypatch):
    """`main` 을 합성 스테이징에서 돈다 — 산출의 꼴과 `--prior` 의 종료 코드(06c I-5 · X43)."""
    import json
    import zipfile

    import data.vt.guard as G
    import data.vt.r8 as R
    from data.vt.config import load_config
    cfg = load_config()
    root = tmp_path / "repo"
    st = root / cfg["staging_root"]
    raw = root / cfg["copy"]["raw_zip_dir"]
    (st / "parse").mkdir(parents=True)
    (st / "copy").mkdir(parents=True)
    raw.mkdir(parents=True)
    labels, vid = [], 100
    for folder in cfg["labels"]["folders"]:
        zname = f"TS_VTST_{folder}.zip"
        with zipfile.ZipFile(raw / zname, "w") as zf:
            for k in range(6):
                vid += 1
                zf.writestr(f"/VT_ST_00_{vid}.jpg", _jpeg(vid, 360, 640))
                labels.append({"vt_id": vid, "image_id": f"aihub71761_vt:{vid}", "folder": folder, "src_zip": zname,
                               "file_stem": f"VT_ST_00_{vid}", "w": 640, "h": 360,
                               "is_normal": folder == cfg["labels"]["normal_folder"], "label_excluded": None,
                               "anns": []})
    (st / "parse" / "labels.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in labels),
                                              encoding="utf-8")
    (st / "copy" / "copy_verify.json").write_text('{"all_ok": true, "image_content_evidence_all": true}',
                                                 encoding="utf-8")

    class _NoWait:
        @classmethod
        def from_config(cls, *a, **k):
            return cls()

        def wait(self):
            return {}
    monkeypatch.setattr(R, "repo_path", lambda rel, r=None: root / rel)
    monkeypatch.setattr(R, "REPO_ROOT", root)
    monkeypatch.setattr(G, "Guard", _NoWait)
    assert R.main([]) == 0
    out = json.loads((st / "r8" / "r8.json").read_text(encoding="utf-8"))
    assert set(out["folders"]) == set(cfg["labels"]["folders"]) and "consequence" in out
    assert set(out["defect_not_similar_usable"]) == set(R.defect_order(cfg))
    prior = {"folders": {f: {k: out["folders"][f][k] for k in ("neighbors", "random_same_size")}
                         for f in R.defect_order(cfg)}}
    pp = tmp_path / "prior.json"
    pp.write_text(json.dumps(prior, ensure_ascii=False), encoding="utf-8")
    assert R.main(["--prior", str(pp)]) == 0
    prior["folders"][R.defect_order(cfg)[0]]["neighbors"]["dhash_median"] = -1
    pp.write_text(json.dumps(prior, ensure_ascii=False), encoding="utf-8")
    assert R.main(["--prior", str(pp)]) == 3

