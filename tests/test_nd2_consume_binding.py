"""간선 v2 파일럿 — 소비 쪽 결속 · 메타 대조의 실제 실패 · base.yaml · 출력 경로(공유 회신 §63 재검토의 반례). 합성 고정물만 쓴다."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from data.nd2 import pilot as P
from data.nd2 import scan as SC

N_EVAL, N_TV = 4, 10


def _fixture(tmp: Path, *, bad_box: bool = False) -> dict:
    """작은 저장소 꼴 — 매니페스트 · 출처 · 자르기 기록 · 원래 폭 · 기준 파일 · 기술자 · base.yaml."""
    (tmp / ".git").mkdir()
    d = tmp / "in"
    d.mkdir()
    ids = [f"aihub71761:{1000 + k}" for k in range(N_EVAL + N_TV)]
    man = ["image_id,split,group_id,rel_path,sha256"]
    tiles, prog, v0 = ["image_id,provenance,reason"], [], ["image_id,width_px,height_px"]
    for k, i in enumerate(ids):
        split = "eval" if k < N_EVAL else "train"
        man.append(f"{i},{split},g{k // 2},data/interim/tiles_v1_histmatch/RT/ST/{1000 + k}.jpg,{k:064x}")
        pano = k % 3 == 0
        tiles.append(f"{i},{'N-tile' if pano else 'N-crop'},{'tiled' if pano else 'ok'}")
        x0 = 100 if (bad_box and k == 0) else (640 if pano else 0)
        prog.append(json.dumps({"image_id": i, "box": [x0, 40, x0 + 1280, 760], "reason": "tiled" if pano else "ok"}))
        v0.append(f"{i},{2000 if pano else 1280},{800 if pano else 720}")
    (d / "manifest.csv").write_text("\n".join(man) + "\n", encoding="utf-8")
    (d / "tiles.csv").write_text("\n".join(tiles) + "\n", encoding="utf-8")
    (d / "progress.jsonl").write_text("\n".join(prog) + "\n", encoding="utf-8")
    (d / "v0.csv").write_text("\n".join(v0) + "\n", encoding="utf-8")
    (d / "href.json").write_text(json.dumps({"histogram": [1] * 256}), encoding="utf-8")
    (d / "bm.json").write_text(json.dumps({"fill_value": 114, "border_frac": 0.08}), encoding="utf-8")
    rng = np.random.default_rng(0)
    np.save(d / "desc.npy", rng.normal(size=(len(ids), 36, 64)).astype(np.float32))
    (d / "ids.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    (tmp / "configs").mkdir()
    (tmp / "configs" / "base.yaml").write_text(yaml.safe_dump({"preprocess": {"tile": {
        "tile_size": [1280, 720], "stride_x": 640, "align8": True,
        "encode": {"mode": "L", "quality": 74, "progressive": False, "optimize": False}}}}), encoding="utf-8")
    return {
        "inputs": {"manifest": "in/manifest.csv", "tiles": "in/tiles.csv", "encode_progress": "in/progress.jsonl",
                   "manifest_v0": "in/v0.csv", "histmatch_reference": "in/href.json", "border_mask_calibration": "in/bm.json",
                   "desc": "in/desc.npy", "desc_ids": "in/ids.txt"},
        "chain": {"quality": 74, "tile_w": 1280, "tile_h": 720, "stride_x": 640, "align": 8},
        "resources": {"start_min_gb": 8.0, "chunk_min_gb": 0.0, "wait_step_s": 0.01, "fft_workers": 1, "block_rows": 3},
        "pilot": {"out": "_workspace/p", "seed": 7, "n_eval": 3, "n_trainval": 6,
                  "record_floors": {"g3": -1.0, "g4": -1.0, "p": -1.0}, "p_min_overlap": 0.2, "hop_g4": 0.45,
                  "hop_seed_cap": 5000, "hop_seed_steps": [1, 3], "attach_max_pairs": 4, "recut_check_n": 2},
        "features": {"window_cells": 1, "compute_min_overlap": 0.05},
        "calib": {"g3_per_provenance": 2, "overlap_panoramas": 2, "overlaps": [0.9, 0.5]},
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "mem_free_gb", lambda: 99.0)
    cfg = _fixture(tmp_path)
    out = tmp_path / "_workspace" / "p"
    out.mkdir(parents=True)
    return tmp_path, cfg, out


def _snap(d: Path) -> dict:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


# --- §63-2: 실제 메타 대조가 실패를 잡고, 실패면 다음 실물 단계로 가지 않는다 ------------------------------------
def test_실제_메타_대조가_원점_밖_상자를_잡는다(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ROOT", tmp_path)
    cfg = _fixture(tmp_path, bad_box=True)
    out = tmp_path / "_workspace" / "p"
    out.mkdir(parents=True)
    with pytest.raises(P.MetaCheckFailed):
        P.step_metacheck(cfg, out)                                    # 대역이 아니라 실제 함수
    rec = json.loads((out / "metacheck.json").read_text(encoding="utf-8"))
    assert rec["ok"] is False and rec["n_origin_not_contained"] == 1


def test_같은_고정물에서_상자가_맞으면_통과한다(env):
    _, cfg, out = env
    assert P.step_metacheck(cfg, out)["ok"] is True                   # 위 실패가 상자 때문임을 보이는 대조


def test_메타_대조_실패면_all_이_스캔으로_가지_않는다(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(P, "mem_free_gb", lambda: 99.0)
    cfg = _fixture(tmp_path, bad_box=True)
    monkeypatch.setattr(P, "load_cfg", lambda: cfg)
    assert P.main(["all"]) == 3
    assert not (tmp_path / "_workspace" / "p" / "scan").exists()


def test_base_yaml_이_바뀌면_지난_대조를_다시_쓰지_않는다(env):
    root, cfg, out = env
    P.step_metacheck(cfg, out)
    P.require_metacheck(cfg, out, "scan")                              # 바뀌기 전에는 통과
    b = root / "configs" / "base.yaml"
    d = yaml.safe_load(b.read_text(encoding="utf-8"))
    d["preprocess"]["tile"]["encode"]["quality"] = 75
    b.write_text(yaml.safe_dump(d), encoding="utf-8")
    with pytest.raises(P.MetaCheckFailed, match="base_yaml"):
        P.require_metacheck(cfg, out, "scan")


# --- §63-1: 소비 쪽 결속 ------------------------------------------------------------------------------------
def test_기술자가_바뀐_뒤의_규칙_입력은_옛_스캔을_소비하지_않는다(env):
    root, cfg, out = env
    P.step_sample(cfg, out)
    P.step_metacheck(cfg, out)
    P.step_scan(cfg, out)
    D = np.load(root / "in" / "desc.npy")
    D[0] += 0.5
    np.save(root / "in" / "desc.npy", D)
    P.step_metacheck(cfg, out)                                        # 새 입력으로 대조는 통과한다
    before = _snap(out / "scan")
    with pytest.raises(SC.RunMismatch, match="스캔"):
        P.step_attach(cfg, out)
    assert _snap(out / "scan") == before
    assert not (out / "attach.csv").exists() and not (out / "attach_RUN.json").exists()


def test_기록_바닥이_바뀐_뒤의_규칙_입력도_멈춘다(env):
    _, cfg, out = env
    P.step_sample(cfg, out)
    P.step_metacheck(cfg, out)
    P.step_scan(cfg, out)
    cfg["pilot"]["record_floors"] = {"g3": 0.0, "g4": 0.0, "p": 0.0}
    with pytest.raises(SC.RunMismatch):
        P.step_attach(cfg, out)


def _fake_calib(cfg, out):
    """보정 산출이 있는 것처럼 — 생산 단계와 같은 식별 · 요약의 runs."""
    plan = P.calib_plan(cfg, out)
    P.bind_outputs(out / "calib_g3_RUN.json", plan["g3_ident"], [out / "calib_g3.csv"])
    P.bind_outputs(out / "calib_overlap_RUN.json", plan["ov_ident"], [out / "calib_overlap.csv"])
    (out / "calib_g3.csv").write_text("provenance,image_id\n", encoding="utf-8")
    (out / "calib_overlap.csv").write_text("image_id,width\n", encoding="utf-8")
    runs = P.require_calib_current(cfg, out)
    P.write_json(out / "calib_summary.json", {"g3_synthetic_positive": {"N-crop": {"p5_small": 0.3}, "N-tile": {"p5_small": 0.4}},
                                              "runs": runs, "sources": P.calib_sources(out)})


def test_보정_설정이_바뀌면_홉_규칙_입력이_옛_문턱을_소비하지_않는다(env):
    _, cfg, out = env
    P.step_metacheck(cfg, out)
    _fake_calib(cfg, out)
    assert P.hop_g3_taus(cfg, out) == ([0.3], [])                      # 같은 설정이면 읽는다
    cfg["calib"]["g3_per_provenance"] = 3
    with pytest.raises(SC.RunMismatch, match="보정"):
        P.hop_g3_taus(cfg, out)


def test_보정_요약이_그_실행의_것이_아니면_멈춘다(env):
    _, cfg, out = env
    P.step_metacheck(cfg, out)
    _fake_calib(cfg, out)
    s = json.loads((out / "calib_summary.json").read_text(encoding="utf-8"))
    s["runs"]["g3"] = "0" * 64
    P.write_json(out / "calib_summary.json", s)
    with pytest.raises(SC.RunMismatch, match="calib_summary"):
        P.hop_g3_taus(cfg, out)


# --- 출력 경로 ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("where", ["data/interim/x", "../elsewhere", "_workspace/frozen/inner"])
def test_출력_경로가_허락된_뿌리_밖이거나_동결이면_만들지_않는다(tmp_path, monkeypatch, where):
    monkeypatch.setattr(P, "ROOT", tmp_path)
    cfg = _fixture(tmp_path)
    (tmp_path / "_workspace" / "frozen").mkdir(parents=True)
    (tmp_path / "_workspace" / "frozen" / "SNAPSHOT.sha256").write_text("x", encoding="utf-8")
    cfg["pilot"]["out"] = where
    monkeypatch.setattr(P, "load_cfg", lambda: cfg)
    assert P.main(["metacheck"]) == 5
    assert not (tmp_path / where).exists()


@pytest.mark.parametrize("cap", [0, 5000])
def test_홉은_2_홉_시작_영상이_없어도_멈추지_않고_있으면_잰다(env, cap):
    root, cfg, out = env
    D = np.load(root / "in" / "desc.npy")
    D[N_EVAL + 1] = D[0]                                               # 평가 0 과 학습검증 하나가 같은 그림
    np.save(root / "in" / "desc.npy", D)
    cfg["pilot"]["hop_seed_cap"] = cap
    cfg["pilot"]["n_eval"] = N_EVAL
    P.step_sample(cfg, out)
    P.step_metacheck(cfg, out)
    _fake_calib(cfg, out)
    res = P.step_hops(cfg, out)
    row = res["seed_steps"]["3"]["g4"]
    if cap == 0:
        assert res["h2_seeds"] == 0 and row["h2"] is None
    else:
        assert res["h2_seeds"] > 0 and row["h2"] is not None and row["h2"] >= row["h1"]


# --- 공유 회신 §64 의 두 반례 — 개별 진입점 그대로 ---------------------------------------------------------------
def test_반례1_옛_입력의_온전한_스캔은_새_대조_아래서_규칙_입력이_받지_않는다(env):
    """이전 입력으로 스캔 완료 → 기술자 변경 → 메타 대조 갱신 → 스캔을 건너뛰고 attach 를 처음 부른다."""
    root, cfg, out = env
    P.step_sample(cfg, out)
    P.step_metacheck(cfg, out)
    P.step_scan(cfg, out)
    _fake_calib(cfg, out)
    D = np.load(root / "in" / "desc.npy")
    D[3] *= 1.5
    np.save(root / "in" / "desc.npy", D)
    assert P.step_metacheck(cfg, out)["ok"] is True                    # 새 대조는 통과
    before = _snap(out)
    with pytest.raises(SC.RunMismatch, match="스캔"):
        P.step_attach(cfg, out)
    assert _snap(out) == before                                       # 옛 스캔 · 보정을 지우지도 덮지도 않았다


def test_스캔_읽기는_기대_실행_해시가_다르면_읽지_않는다(env):
    _, cfg, out = env
    P.step_sample(cfg, out)
    P.step_metacheck(cfg, out)
    P.step_scan(cfg, out)
    d = out / "scan" / "TT"
    good = json.loads((d / "RUN.json").read_text(encoding="utf-8"))["run"]
    assert SC.read_rows(d, expect_run=good) == SC.read_rows(d)
    with pytest.raises(SC.RunMismatch, match="기대"):
        SC.read_rows(d, expect_run="0" * 64)
    with pytest.raises(SC.RunMismatch, match="기대"):
        SC.summarize(d, expect_run="0" * 64)


@pytest.mark.parametrize("entry", ["hops", "attach"])
def test_반례2_보정_설정을_바꾸고_개별_단계로_들어가면_옛_문턱을_쓰지_않는다(env, entry):
    _, cfg, out = env
    P.step_sample(cfg, out)
    P.step_metacheck(cfg, out)
    P.step_scan(cfg, out)
    _fake_calib(cfg, out)
    cfg["calib"]["overlaps"] = [0.9, 0.5, 0.3]                       # 보정 설정이 바뀌었다
    before = _snap(out)
    with pytest.raises(SC.RunMismatch, match="보정"):
        (P.step_hops if entry == "hops" else P.step_attach)(cfg, out)
    assert _snap(out) == before
    assert not (out / "hops").exists() and not (out / "attach.csv").exists()


def test_보정_요약을_낸_원천_CSV_가_바뀌면_소비하지_않는다(env):
    _, cfg, out = env
    P.step_metacheck(cfg, out)
    _fake_calib(cfg, out)
    assert P.hop_g3_taus(cfg, out) == ([0.3], [])
    (out / "calib_g3.csv").write_text("provenance,image_id\nN-crop,x\n", encoding="utf-8")
    with pytest.raises(SC.RunMismatch, match="원천"):
        P.hop_g3_taus(cfg, out)
