"""간선 v2 파일럿 — 그림을 읽지 않는 부분(표본의 결정성 · 부분 집합 순서 · 시작 조건)."""
from __future__ import annotations

import numpy as np
import pytest

from data.nd2 import pilot as P
from data.nd2 import scan as SC


def test_부분_집합은_요청한_순서를_지킨다():
    D = np.arange(5 * 36 * 64, dtype=np.float32).reshape(5, 36, 64)
    pos = {f"i{k}": k for k in range(5)}
    out = P._subset(D, pos, ["i3", "i0", "i4"])
    assert np.array_equal(out, D[[3, 0, 4]])


def test_순위_열쇠는_시드와_id_로만_정해진다():
    assert P.rank_key(1, "a") == P.rank_key(1, "a")
    assert P.rank_key(1, "a") != P.rank_key(2, "a")


def test_표본은_두_번_뽑아도_같은_바이트(tmp_path, monkeypatch):
    man = {f"aihub71761:{k}": {"split": ("eval" if k % 5 == 0 else "train")} for k in range(200)}
    monkeypatch.setattr(P, "read_manifest", lambda cfg: man)
    monkeypatch.setattr(P, "desc_index", lambda cfg: (sorted(man), None, None))
    cfg = {"pilot": {"seed": 7, "n_eval": 10, "n_trainval": 30}}
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    P.step_sample(cfg, tmp_path / "1")
    P.step_sample(cfg, tmp_path / "2")
    a = (tmp_path / "1" / "sample.csv").read_bytes()
    assert a == (tmp_path / "2" / "sample.csv").read_bytes()
    E, T = P.read_sample(tmp_path / "1")
    assert len(E) == 10 and len(T) == 30
    assert all(man[i]["split"] == "eval" for i in E) and all(man[i]["split"] != "eval" for i in T)


def test_실물_단계는_메모리가_모자라면_시작하지_않는다(monkeypatch):
    monkeypatch.setattr(P, "mem_free_gb", lambda: 5.9)
    with pytest.raises(SC.StartRefused):
        P.require_start({"resources": {"start_min_gb": 8.0}}, "scan")


def test_홉의_G3_문턱은_G4_문턱_이상이면_따로_세지_않는다(tmp_path, monkeypatch):
    import json
    runs = {"g3": "a" * 64, "overlap": "b" * 64}
    monkeypatch.setattr(P, "require_calib_current", lambda cfg, out: runs)    # 결속은 test_nd2_consume_binding 이 본다
    monkeypatch.setattr(P, "calib_sources", lambda out: {"calib_g3.csv": "c", "calib_overlap.csv": "d"})
    cfg = {"pilot": {"hop_g4": 0.45}}
    (tmp_path / "calib_summary.json").write_text(json.dumps({"g3_synthetic_positive": {
        "N-crop": {"p5_small": 0.6}, "N-tile": {"p5_small": 0.5}}, "runs": runs,
        "sources": {"calib_g3.csv": "c", "calib_overlap.csv": "d"}}), encoding="utf-8")
    assert P.hop_g3_taus(cfg, tmp_path) == ([], [0.5])
    (tmp_path / "calib_summary.json").write_text(json.dumps({"g3_synthetic_positive": {
        "N-crop": {"p5_small": 0.6}, "N-tile": {"p5_small": 0.3179}}, "runs": runs,
        "sources": {"calib_g3.csv": "c", "calib_overlap.csv": "d"}}), encoding="utf-8")
    assert P.hop_g3_taus(cfg, tmp_path) == ([0.31], [])
