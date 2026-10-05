"""간선 v2 — 덩어리 스캔의 이어 하기 · 지킴 · 결정성(합성 기술자)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from data.nd2 import scan as SC


def _set(n=12, seed=0):
    rng = np.random.default_rng(seed)
    D = np.stack([cv2.GaussianBlur(rng.normal(size=(36, 64)).astype(np.float32), (0, 0), 1.2) for _ in range(n)])
    D[5] = D[2]                                     # 같은 사진
    D[7][:, :40] = D[3][:, 24:]                     # 부분 중첩 — 7[y, x] = 3[y, x + 24] (앞 40 칸)
    ids = [f"x:{i:03d}" for i in range(n)]
    return ids, D


def _go(out, ids, D, **kw):
    return SC.run_pairs("t", ids, ids, D, D, within=True, out_dir=out, floors={"g3": 0.5, "g4": 0.45, "p": 0.8},
                        wait=lambda: {"mem_free_gb": 9.0}, mem_now=lambda: 9.0, start_min_gb=8.0, block=5, workers=1,
                        log=lambda s: None, **kw)


def test_같은_사진과_부분_중첩이_남는다(tmp_path):
    ids, D = _set()
    s = _go(tmp_path, ids, D)
    assert s["n_pairs"] == 12 * 11 // 2
    got = {(r["a_id"], r["b_id"]): r for r in SC.read_rows(tmp_path / "t")}
    assert float(got[("x:002", "x:005")]["g3"]) > 0.999
    r = got[("x:003", "x:007")]
    assert float(r["p"]) > 0.95
    assert (int(r["p_dx"]), int(r["p_dy"])) == (24, 0)           # 3[y, x] = 7[y, x - 24] — a 기준 b 의 위치


def test_뒤집힌_짝은_이동의_부호를_바꾼다(tmp_path):
    ids, D = _set()
    ids = ids[::-1]
    D = D[::-1].copy()
    _go(tmp_path, ids, D)
    r = {(r["a_id"], r["b_id"]): r for r in SC.read_rows(tmp_path / "t")}[("x:003", "x:007")]
    assert (int(r["p_dx"]), int(r["p_dy"])) == (24, 0)


def test_이어_하기는_마친_덩어리를_건너뛰고_같은_바이트(tmp_path):
    ids, D = _set()
    _go(tmp_path / "1", ids, D)
    full = sorted((tmp_path / "1" / "t").glob("blk_*.csv"))
    _go(tmp_path / "2", ids, D)
    (tmp_path / "2" / "t" / "blk_00001.json").unlink()            # 둘째 덩어리 중간에 죽은 것처럼
    (tmp_path / "2" / "t" / "blk_00001.csv").unlink()
    _go(tmp_path / "2", ids, D)
    assert len(full) == 3
    for p in full:
        assert p.read_bytes() == (tmp_path / "2" / "t" / p.name).read_bytes()


def test_시작_조건_아래면_시작하지_않는다(tmp_path):
    ids, D = _set()
    with pytest.raises(SC.StartRefused):
        SC.run_pairs("t", ids, ids, D, D, within=True, out_dir=tmp_path, floors={"g3": 0.5, "g4": 0.45, "p": 0.8},
                     wait=dict, mem_now=lambda: 6.0, start_min_gb=8.0, workers=1, log=lambda s: None)
    assert not list((tmp_path / "t").glob("blk_*"))


def test_덩어리마다_지킴을_부른다(tmp_path):
    ids, D = _set()
    calls = []
    SC.run_pairs("t", ids, ids, D, D, within=True, out_dir=tmp_path, floors={"g3": 0.5, "g4": 0.45, "p": 0.8},
                 wait=lambda: calls.append(1) or {"mem_free_gb": 7.0}, mem_now=lambda: 9.0, start_min_gb=8.0, block=5,
                 workers=1, log=lambda s: None)
    assert len(calls) == 3


def test_교차는_모든_열을_본다(tmp_path):
    ids, D = _set()
    s = SC.run_pairs("c", ids[:3], ids[3:], D[:3], D[3:], within=False, out_dir=tmp_path,
                     floors={"g3": 0.5, "g4": 0.45, "p": 0.8}, wait=dict, mem_now=lambda: 9.0, start_min_gb=8.0,
                     workers=1, log=lambda s: None)
    assert s["n_pairs"] == 3 * 9
    assert ("x:002", "x:005") in {(r["a_id"], r["b_id"]) for r in SC.read_rows(tmp_path / "c")}


def test_분포의_비율():
    h = [0] * 200
    h[150] = 3                                                     # 0.50 칸
    h[100] = 1                                                     # 0.00 칸
    assert SC.rate_at(h, 0.45) == pytest.approx(0.75)
    assert SC.rate_at(h, 0.51) == 0.0
