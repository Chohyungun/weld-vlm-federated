"""간선 v2 — 실행 결속 · 메타 대조 관문 · 홉 정의(공유 회신 §63-1 ~ 3). 합성 입력, 실데이터를 읽지 않는다."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from data.nd2 import pilot as P
from data.nd2 import scan as SC


def _set(n=12, seed=0):
    rng = np.random.default_rng(seed)
    D = np.stack([cv2.GaussianBlur(rng.normal(size=(36, 64)).astype(np.float32), (0, 0), 1.2) for _ in range(n)])
    D[5] = D[2]
    return [f"x:{i:03d}" for i in range(n)], D


def _go(out, ids, D, floors=None, block=5):
    return SC.run_pairs("t", ids, ids, D, D, within=True, out_dir=out, floors=floors or {"g3": 0.5, "g4": 0.45, "p": 0.8},
                        wait=dict, mem_now=lambda: 9.0, start_min_gb=8.0, block=block, workers=1, log=lambda s: None)


def _snapshot(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


# --- §63-1 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("change", ["표본_축소", "바닥", "덩어리_크기", "기술자"])
def test_다른_실행으로_같은_자리를_이으면_보존하고_멈춘다(tmp_path, change):
    ids, D = _set()
    _go(tmp_path, ids, D)
    before = _snapshot(tmp_path / "t")
    with pytest.raises(SC.RunMismatch):
        if change == "표본_축소":
            _go(tmp_path, ids[:6], D[:6])
        elif change == "바닥":
            _go(tmp_path, ids, D, floors={"g3": 0.4, "g4": 0.45, "p": 0.8})
        elif change == "덩어리_크기":
            _go(tmp_path, ids, D, block=4)
        else:
            D2 = D.copy()
            D2[0] += 1e-3
            _go(tmp_path, ids, D2)
    assert _snapshot(tmp_path / "t") == before                      # 지우거나 덮지 않았다


def test_실행_식별_없는_덩어리가_있으면_멈춘다(tmp_path):
    ids, D = _set()
    d = tmp_path / "t"
    d.mkdir()
    (d / "blk_00000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SC.RunMismatch):
        _go(tmp_path, ids, D)
    assert not (d / "RUN.json").exists()


def test_목록_밖_덩어리가_있으면_멈춘다(tmp_path):
    ids, D = _set()
    _go(tmp_path, ids, D)
    (tmp_path / "t" / "blk_00009.csv").write_text("a_id,b_id\n", encoding="utf-8")
    with pytest.raises(SC.RunMismatch):
        _go(tmp_path, ids, D)


def test_요약은_그_실행의_덩어리만_해시를_확인하며_읽는다(tmp_path):
    ids, D = _set()
    s = _go(tmp_path, ids, D)
    assert s["blocks"] == 3 and s["n_pairs"] == 66
    p = tmp_path / "t" / "blk_00001.csv"
    p.write_text(p.read_text(encoding="utf-8") + "x:000,x:999,0,0,0,0,,0,0\n", encoding="utf-8")
    with pytest.raises(SC.RunMismatch):
        SC.summarize(tmp_path / "t")
    with pytest.raises(SC.RunMismatch):
        SC.read_rows(tmp_path / "t")


def test_끝나지_않은_실행은_요약하지_않는다(tmp_path):
    ids, D = _set()
    _go(tmp_path, ids, D)
    (tmp_path / "t" / "blk_00002.json").unlink()
    with pytest.raises(SC.RunMismatch):
        SC.summarize(tmp_path / "t")


def test_같은_실행은_이어_한다(tmp_path):
    ids, D = _set()
    _go(tmp_path, ids, D)
    full = _snapshot(tmp_path / "t")
    (tmp_path / "t" / "blk_00001.json").unlink()
    _go(tmp_path, ids, D)
    after = _snapshot(tmp_path / "t")
    for name in ("blk_00000.csv", "blk_00001.csv", "blk_00002.csv", "RUN.json"):
        assert after[name] == full[name]


def test_파일럿_산출_결속(tmp_path):
    out = tmp_path / "o.csv"
    P.bind_outputs(tmp_path / "x_RUN.json", {"a": 1}, [out])
    P.bind_outputs(tmp_path / "x_RUN.json", {"a": 1}, [out])            # 같은 실행은 통과
    with pytest.raises(SC.RunMismatch):
        P.bind_outputs(tmp_path / "x_RUN.json", {"a": 2}, [out])
    out2 = tmp_path / "o2.csv"
    out2.write_text("x", encoding="utf-8")
    with pytest.raises(SC.RunMismatch):
        P.bind_outputs(tmp_path / "y_RUN.json", {"a": 1}, [out2])      # 기록 없이 산출만 있다
    assert out2.read_text(encoding="utf-8") == "x"


def test_표본은_다른_설정으로_덮어쓰지_않는다(tmp_path, monkeypatch):
    man = {f"aihub71761:{k}": {"split": ("eval" if k % 5 == 0 else "train")} for k in range(200)}
    monkeypatch.setattr(P, "read_manifest", lambda cfg: man)
    monkeypatch.setattr(P, "desc_index", lambda cfg: (sorted(man), None, None))
    P.step_sample({"pilot": {"seed": 7, "n_eval": 10, "n_trainval": 30}}, tmp_path)
    before = (tmp_path / "sample.csv").read_bytes()
    P.step_sample({"pilot": {"seed": 7, "n_eval": 10, "n_trainval": 30}}, tmp_path)   # 같으면 통과
    with pytest.raises(SC.RunMismatch):
        P.step_sample({"pilot": {"seed": 7, "n_eval": 5, "n_trainval": 30}}, tmp_path)
    assert (tmp_path / "sample.csv").read_bytes() == before


# --- §63-2 ---------------------------------------------------------------------------------
def _cfg():
    return {"inputs": {"manifest": "m.csv"}, "chain": {"quality": 74}}


def test_메타_대조_없이는_실물_단계가_시작하지_않는다(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "input_hashes", lambda cfg: {"manifest": "aa"})
    monkeypatch.setattr(P, "mem_free_gb", lambda: 99.0)
    with pytest.raises(P.MetaCheckFailed):
        P.require_start(_cfg(), "scan", tmp_path)                     # 기록이 없다
    rec = {"ok": False, "failures": ["원점"], "inputs_cfg": _cfg()["inputs"], "chain_cfg": _cfg()["chain"],
           "input_sha256": {"manifest": "aa"}}
    (tmp_path / "metacheck.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(P.MetaCheckFailed):
        P.require_start(_cfg(), "scan", tmp_path)                     # 실패한 대조
    rec["ok"] = True
    rec["input_sha256"] = {"manifest": "bb"}
    (tmp_path / "metacheck.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(P.MetaCheckFailed):
        P.require_start(_cfg(), "scan", tmp_path)                     # 대조 뒤 입력이 바뀌었다
    rec["input_sha256"] = {"manifest": "aa"}
    rec["chain_cfg"] = {"quality": 75}
    (tmp_path / "metacheck.json").write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(P.MetaCheckFailed):
        P.require_start(_cfg(), "scan", tmp_path)                     # 사슬 설정이 다르다
    rec["chain_cfg"] = _cfg()["chain"]
    (tmp_path / "metacheck.json").write_text(json.dumps(rec), encoding="utf-8")
    assert P.require_start({**_cfg(), "resources": {"start_min_gb": 8.0}}, "scan", tmp_path) == 99.0


def test_메타_대조_실패는_비정상_종료(tmp_path, monkeypatch):
    def boom(cfg, out):
        raise P.MetaCheckFailed("원점 집합이 실제 상자를 품지 못함 1")
    called = []
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "load_cfg", lambda: {"pilot": {"out": "_workspace/p"}})   # 실제 검출은 test_nd2_consume_binding
    monkeypatch.setitem(P.STEPS, "metacheck", boom)
    monkeypatch.setitem(P.STEPS, "scan", lambda cfg, out: called.append(1))
    monkeypatch.setitem(P.STEPS, "sample", lambda cfg, out: None)
    assert P.main(["all"]) == 3
    assert called == []                                                # 다음 실물 단계로 가지 않았다


# --- §63-3 ---------------------------------------------------------------------------------
def test_묶음_제외_2홉은_h1_에서만_넓힌다():
    groups = {"t1": {"t1", "t2"}, "t2": {"t1", "t2"}, "t3": {"t3"}}
    edges = {("t2", "t3")}

    def expand(s):
        return {b if a == t else a for t in s for (a, b) in edges if t in (a, b)}

    def with_groups(s):
        return set().union(*(groups[t] for t in s)) if s else set()

    hs = P.hop_sets({"t1"}, expand, with_groups)
    assert hs["h2"] == {"t1"}                                          # 회신의 반례 — 묶음 제외
    assert hs["h2_with_groups"] == {"t1", "t2", "t3"}                  # 묶음 포함
    assert hs["h1_with_groups"] == {"t1", "t2"}


def test_메타의_덩어리_번호가_파일_번호와_다르면_멈춘다(tmp_path):
    ids, D = _set()
    _go(tmp_path, ids, D)
    d = tmp_path / "t"
    m = json.loads((d / "blk_00000.json").read_text(encoding="utf-8"))
    m["block"] = 2                                                     # CSV 해시는 그대로 — 번호만 다른 덩어리를 가리킨다
    (d / "blk_00000.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SC.RunMismatch, match="block"):
        SC.read_rows(d)
    with pytest.raises(SC.RunMismatch, match="block"):
        SC.summarize(d)


def test_코드_지문은_들여올_때의_내용으로_잰다(monkeypatch):
    before = SC.code_version(("scan.py", "ncc.py"))
    from pathlib import Path as _P

    def boom(self):
        raise AssertionError("실행 중에 디스크의 코드를 다시 읽었다")
    monkeypatch.setattr(_P, "read_bytes", boom)
    assert SC.code_version(("scan.py", "ncc.py")) == before
