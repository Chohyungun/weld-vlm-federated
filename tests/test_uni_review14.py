"""학습 쪽 통합형 구현의 원저자 아닌 읽기(검수 14번)가 짚은 자리 — Important 다섯과 고친 Minor.

연합 칸이 로컬 · 중앙보다 약하게 지켜지던 다섯(I-1 ~ I-5)이 중심이다. 모델은 올리지 않는다 — 합성 적재기(리허설 목적)와
합성 산출물로 돈다. 본실험 목적의 연합 서버는 모델을 올리지 않고 가드와 신원 계산까지만 부른다.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tests.test_uni_fed_cell as FC
from fl.atomic_log import END_METRIC, AtomicLog, last_data_row, new_run_id
from fl.uni_fed import UniFedRejected, UniFedRun, check_fed_ledger
from scripts import main_uni as M
from tests.test_main_uni import _cfg_yaml
from tests.test_uni_fed_cell import TAGS, E, R, _run_fed, _server_get
from tests.test_uni_train_cell import _init, _rows, _spec, reh
from tests.uni_fakes import CountingLoader
from vlm import fault as F
from vlm.train_cell import (
    LOCAL_TAGS,
    CellRejected,
    cell_dir,
    close_ledger,
)

NONCE = "0123456789abcdef0123456789abcdef"
fed = FC.fed          # 연합 칸 시험의 합성 세계(리허설 루트 · 페어 · 공통 초기 어댑터)를 그대로 쓴다


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """GPU 를 쓰지 않고, 고의 중단 변수를 물려받지 않는다."""
    import torch

    was = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for k in list(os.environ):
        if k.upper() in F.FAULT_VARS:
            monkeypatch.delenv(k)
    yield
    assert was or not torch.cuda.is_initialized(), "이 시험이 CUDA 를 초기화했다"


# ================================================================ I-1 실행기의 연합 건너뜀
def _main_cache(path: Path, arrays, keys, *, seed: int, revision: str) -> None:
    """본실험 proof 가 붙은 공통 초기 어댑터 캐시(승인 구현 식별자 · 세 증빙)."""
    from vlm.init_adapter import adapter_proof, init_adapter_digest

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **dict(zip(keys, arrays)))
    impl = [{"seam": "model_loader", "module": "vlm.pilot_vlm", "qualname": "_load_model",
             "source_path": "vlm/pilot_vlm.py", "blob_sha1": "b" * 40, "approved": True}]
    proof = {"model_id": "Qwen/Qwen3.5-4B", "revision": revision, "seed": seed, "purpose": "main",
             "impl_ids": impl, "init_adapter_digest": init_adapter_digest(arrays), **adapter_proof(arrays, keys)}
    path.with_suffix(".proof.json").write_text(json.dumps(proof), encoding="utf-8")


def _fed_ledger(path: Path, run_id: str, *, rounds=range(2), close=True) -> None:
    log = AtomicLog(path, run_id=run_id, seed=11, cell="uni_fed", split_hash="s" * 64)
    for rd in rounds:
        for i in range(3):
            log.log_round(round_idx=rd, client_id=i, n_train_samples=1, metrics={"optimizer_steps": 1.0})
        log.log_round(round_idx=rd, client_id="server", n_train_samples=0, metrics={"global_l2": 1.0})
    if close:
        log.close(client_id="server", step=2)


def _server_from_run_config(cfg, p: Path, out: Path, fed_dir: Path) -> UniFedRun:
    """flwr 서버가 받는 것과 같은 실행 설정(선언 기본값 + 실행기의 run-config)으로 연합 서버의 앞을 만든다."""
    from flwr.common.config import parse_config_args

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    run_cfg = {**declared, **parse_config_args([M.fed_run_config(cfg, 1, out=out)])}
    return UniFedRun(get=lambda k, d: run_cfg.get(k, d), out_dir=fed_dir, run_id=new_run_id("uni_fed", 11, "main_s1"),
                     base_seed=int(run_cfg["base-seed"]), split_hash=str(run_cfg["split-hash"]),
                     num_rounds=int(run_cfg["num-server-rounds"]), local_epochs=int(run_cfg["local-epochs"]),
                     total_epochs=int(run_cfg["total-epochs"]), client_tags=TAGS, config_path=p)


def _main_fed(tmp: Path) -> SimpleNamespace:
    """본실험 설정 · 공통 초기 어댑터 · **서버의 신원 계산으로 만든** 닫힌 연합 산출물."""
    from vlm.init_adapter import init_adapter_digest
    from vlm.train_cell import _sha, save_adapter_cell

    p = _cfg_yaml(tmp)
    cfg = M.load_main_config(p)
    out = tmp / "out"
    fed_dir = M.fed_dir(out, 1)                            # 연합 산출 — 학습 루트 아래 uni_fed_s1(2판 §13-4)
    keys = ["m.lora_A.weight", "m.lora_B.weight"]
    arrays = [np.arange(6, dtype=np.float32).reshape(2, 3), np.zeros((3, 2), dtype=np.float32)]
    _main_cache(M.seed_dir(out, 1) / "fl" / "uni_fed" / "initial.npz", arrays, keys, seed=11, revision="a" * 40)
    server = _server_from_run_config(cfg, p, out, fed_dir)
    ident = server.expected_identity(init_adapter_digest(arrays))
    run_id = new_run_id("uni_fed", 11, "main_s1")
    save_adapter_cell(fed_dir, [a + 1 for a in arrays], keys,
                      {"adapter_step": 2, "adapter_step_unit": "fed_rounds", "identity": ident,
                       "identity_sha256": _sha(ident), "train_run_id": run_id,
                       "init_adapter_digest": init_adapter_digest(arrays)})
    _fed_ledger(fed_dir / "atomic_log.csv", run_id)
    return SimpleNamespace(p=p, cfg=cfg, out=out, fed=fed_dir, arrays=arrays, keys=keys, ident=ident, run_id=run_id)


def test_I1_연합_신원은_서버의_실행_설정과_실행기의_설정에서_같게_난다(tmp_path):
    from vlm.init_adapter import init_adapter_digest

    w = _main_fed(tmp_path)
    spec = M.spec_for(w.cfg, 1, out=w.out, config_path=w.p)
    assert M.fed_expected_identity(spec, init_adapter_digest(w.arrays)) == w.ident


def test_I1_같은_설정의_닫힌_연합은_되짚고_건너뛴다(tmp_path):
    w = _main_fed(tmp_path)
    calls: list = []
    assert M.stage_fed(w.cfg, 1, out=w.out, config_path=w.p, runner=lambda *a: calls.append(a)) == "skipped"
    assert calls == []


def _tamper_meta(w, fn, *, rehash=False):
    from vlm.train_cell import _sha

    mp = w.fed / "adapter_last.meta.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    fn(m)
    if rehash:
        m["identity_sha256"] = _sha(m["identity"])
    mp.write_text(json.dumps(m), encoding="utf-8")


@pytest.mark.parametrize("how, code", [
    ("config", "identity_mismatch"),
    ("init", "identity_mismatch"),
    ("identity_edit", "identity_sha_broken"),
    ("identity_rehash", "identity_mismatch"),
    ("run_id", "identity_mismatch"),
    ("adapter", "adapter_sha_mismatch"),
    ("meta", "meta_missing"),
    ("ledger", "fed_server_rounds"),
])
def test_I1_다른_실행의_연합_산출물은_건너뛰지_않는다(tmp_path, how, code, capsys):
    w = _main_fed(tmp_path)
    p = w.p
    if how == "config":
        p = _cfg_yaml(tmp_path, uni_processor_kwargs={"max_pixels": 1})      # 설정이 바뀌었다
    elif how == "init":
        other = [a + 5 for a in w.arrays]                                     # 공통 초기 어댑터를 다시 만들었다
        _main_cache(M.seed_dir(w.out, 1) / "fl" / "uni_fed" / "initial.npz", other, w.keys, seed=11, revision="a" * 40)
    elif how == "identity_edit":
        _tamper_meta(w, lambda m: m["identity"].update(processor_kwargs={"max_pixels": 1}))
    elif how == "identity_rehash":
        _tamper_meta(w, lambda m: m["identity"].update(processor_kwargs={"max_pixels": 1}), rehash=True)
    elif how == "run_id":
        _tamper_meta(w, lambda m: m.update(train_run_id="uni_fed_s11_other"))
    elif how == "adapter":
        with np.load(w.fed / "adapter_last.npz") as z:
            arrs = {k: z[k] + 1 for k in z.files}
        np.savez(w.fed / "adapter_last.npz", **arrs)
    elif how == "meta":
        (w.fed / "adapter_last.meta.json").unlink()
    elif how == "ledger":
        (w.fed / "atomic_log.csv").unlink()
        _fed_ledger(w.fed / "atomic_log.csv", w.run_id, rounds=[0])      # 닫혔지만 라운드 1 의 server 행이 없다
    cfg = M.load_main_config(p)
    with pytest.raises(SystemExit) as exc:
        M.stage_fed(cfg, 1, out=w.out, config_path=p, runner=lambda *a: pytest.fail("학습을 돌렸다"))
    assert exc.value.reason == code, capsys.readouterr().err


def test_M14_연합_잔해의_거부는_실제로_할_수_있는_복구를_말한다(tmp_path, capsys):
    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    fed_dir = M.fed_dir(tmp_path / "out", 1)
    fed_dir.mkdir(parents=True)
    _fed_ledger(fed_dir / "atomic_log.csv", "uni_fed_s11_main_s1", rounds=[0], close=False)
    with pytest.raises(SystemExit) as exc:
        M.stage_fed(cfg, 1, out=tmp_path / "out", config_path=p, runner=lambda *a: None)
    assert exc.value.reason == "fed_ledger_not_fresh"
    err = capsys.readouterr().err
    assert "옆으로 옮긴" in err and "새 스탬프" not in err


# ================================================================ I-2 서버가 등록값을 정본으로 둔다
@pytest.mark.parametrize("key", ["uni-model-id", "uni-model-revision", "uni-prompt-sha256"])
def test_I2_세_참여자가_같은_값으로_등록값과_다르면_닫지_않는다(fed, key):
    def tamper(sr, i, strings):
        strings[key] = "0" * 64                         # 셋이 **같은** 값 — 참여자 사이의 대조로는 걸리지 않는다

    with pytest.raises(UniFedRejected) as exc:
        _run_fed(fed, tamper=tamper)
    assert exc.value.code == "client_registration_mismatch" and key in str(exc.value)
    assert not (fed.out / "adapter_last.npz").exists()
    assert last_data_row(fed.out / "atomic_log.csv")["metric_name"] != END_METRIC


def test_I2_클라이언트는_실제로_올린_모델_판을_보고한다(fed):
    seen: list[str] = []
    _run_fed(fed, tamper=lambda sr, i, s: seen.append(s["uni-model-revision"]))
    assert seen == ["r1"] * (R * len(TAGS))


# ================================================================ I-3 연합의 루트 규칙
@pytest.mark.parametrize("over", [{"resume-root": "OUTSIDE/resume"}, {"plan": "OUTSIDE/plan.yaml"}])
def test_I3_재개_폴더나_계획_파일이_루트_밖이면_읽기_전에_거부한다(fed, over):
    over = {k: str(fed.tmp / v) for k, v in over.items()}     # 계획 파일은 없다 — 읽으려 들면 다른 예외가 난다
    get = _server_get(fed.tmp, **over)
    with pytest.raises(UniFedRejected) as exc:
        UniFedRun(get=get, out_dir=fed.out, run_id="x", base_seed=7, split_hash="s" * 64, num_rounds=R,
                  local_epochs=E, total_epochs=R * E, client_tags=TAGS, non_main_parent=fed.tmp)
    assert exc.value.code == "run_root"


def test_I3_루트_아래의_재개_폴더는_받는다(fed):
    run = UniFedRun(get=_server_get(fed.tmp, **{"resume-root": str(reh(fed.tmp) / "fl" / "resume")}),
                    out_dir=fed.out, run_id="x", base_seed=7, split_hash="s" * 64, num_rounds=R, local_epochs=E,
                    total_epochs=R * E, client_tags=TAGS, non_main_parent=fed.tmp)
    assert run.resume_root == reh(fed.tmp) / "fl" / "resume"


# ================================================================ I-4 두 서버 진입점을 끝까지 돈다
class _FakeStrategy:
    """`WeldFedAvg` 자리 — 라운드마다 세 참여자를 클라이언트 앱의 통합형 함수로 돌리고 라운드 끝 콜백을 부른다.

    Flower 전송과 집계 산술만 대신한다(산술 평균). 서버 진입점의 앞뒤(가드 · 원장 · 초기 어댑터 · 관찰 · 감사 · 저장 · 끝 행)는
    진짜다. `die_after_rounds` 면 라운드를 다 돈 뒤 죽는다.
    """

    die_after_rounds = False

    def __init__(self, *, expected_nodes, canonical_keys, reference_state_dict, on_round_end):
        self.keys, self.cb = list(canonical_keys), on_round_end

    def start(self, *, grid, initial_arrays, num_rounds, train_config):
        from fl.client_app import uni_client_round

        glob = initial_arrays.to_numpy_ndarrays()
        cfg = dict(train_config)
        for sr in range(1, num_rounds + 1):
            cells, outs = [], []
            for i in range(len(TAGS)):
                arr, metrics, strings = uni_client_round(cfg, arrays_in=glob, canonical_keys=self.keys,
                                                         round_idx=sr - 1, client_idx=i)
                cells.append({"node_id": i, **metrics, **strings})
                outs.append(arr)
            glob = [np.mean([o[j] for o in outs], axis=0).astype(outs[0][j].dtype) for j in range(len(self.keys))]
            self.cb(sr, cells, SimpleNamespace(ndarrays=glob, total_examples=6, global_norm=0.0,
                                               bn_buffer_divergence=0.0, missing_variance_ratio=0.0))
        if self.die_after_rounds:
            raise RuntimeError("라운드를 다 돈 뒤 전략이 죽었다")


def _uni_vals(tmp: Path) -> dict[str, str]:
    return {"uni-model": "tiny", "uni-model-revision": "r1", "uni-pairs": str(tmp / "pairs.jsonl"),
            "uni-pairs-digest": "p" * 64, "uni-chat-template-kwargs": '{"enable_thinking": false}',
            "uni-coord-space": "ABS_ORIG", "purpose": "rehearsal", "plan": str(reh(tmp) / "plan.yaml"),
            "uni-seed-index": "1", "rehearsal-root": str(reh(tmp))}


def _serve(fed, monkeypatch, entry: str, *, die=False) -> None:
    """두 서버 진입점 가운데 하나를 부른다. 리허설 루트의 부모를 임시 폴더로 바꿔 끼운다(파이썬 속성, 시험만)."""
    import torch

    from fl import pilot_sim, server_app
    from vlm import run_root

    monkeypatch.setattr(run_root, "NON_MAIN_PARENT", fed.tmp)
    strat = type("S", (_FakeStrategy,), {"die_after_rounds": die})
    if entry == "flwr":
        monkeypatch.setattr(server_app, "WeldFedAvg", strat)
        cfg = {"cell": "uni_fed", "num-server-rounds": R, "local-epochs": E, "total-epochs": R * E,
               "project": str(reh(fed.tmp)), "base-seed": 7, "run-stamp": "t1", "split-hash": "s" * 64,
               "client-tags": ",".join(TAGS), "resume-root": "", **_uni_vals(fed.tmp)}
        server_app.main(None, SimpleNamespace(run_config=cfg))
    else:
        monkeypatch.setattr(pilot_sim, "WeldFedAvg", strat)
        cfg = {"cell": "uni_fed", "out_dir": str(fed.out), "num_rounds": R, "num_clients": len(TAGS),
               "canonical_keys": fed.keys, "local_epochs": E, "total_epochs": R * E, "run_stamp": "t1",
               "base_seed": 7, "split_hash": "s" * 64, "client_tags": TAGS, "initial_arrays": fed.arrays,
               "reference_sd": {k: torch.as_tensor(a) for k, a in zip(fed.keys, fed.arrays)}, "resume_root": "",
               **{k.replace("-", "_"): v for k, v in _uni_vals(fed.tmp).items()}}
        monkeypatch.setattr(pilot_sim, "PILOT_CFG", cfg)
        pilot_sim._server_main(None, None)


@pytest.mark.parametrize("entry", ["flwr", "sim"])
def test_I4_두_서버_진입점이_통합형_연합을_끝까지_돈다(fed, monkeypatch, entry):
    _serve(fed, monkeypatch, entry)
    meta = json.loads((fed.out / "adapter_last.meta.json").read_text(encoding="utf-8"))
    assert meta["adapter_step"] == R and meta["identity"]["model_revision"] == "r1"
    check_fed_ledger(fed.out / "atomic_log.csv", num_rounds=R, num_clients=len(TAGS), closed=True)


@pytest.mark.parametrize("entry", ["flwr", "sim"])
def test_I4_라운드_뒤에_전략이_죽으면_어댑터도_끝_행도_없다(fed, monkeypatch, entry):
    """저장과 끝 행은 `finally` 밖이다 — 전략이 죽은 실행은 감사가 지나도 닫지 않는다."""
    with pytest.raises(RuntimeError, match="전략이 죽었다"):
        _serve(fed, monkeypatch, entry, die=True)
    assert not (fed.out / "adapter_last.npz").exists()
    assert last_data_row(fed.out / "atomic_log.csv")["metric_name"] != END_METRIC


def test_I4_연합_저장이_실패하면_끝_행을_쓰지_않는다(fed, monkeypatch):
    import vlm.train_cell as tc

    def boom(*a, **k):
        raise OSError("저장 실패")

    monkeypatch.setattr(tc, "save_adapter_cell", boom)
    with pytest.raises(OSError):
        _run_fed(fed)
    assert last_data_row(fed.out / "atomic_log.csv")["metric_name"] != END_METRIC


def test_M7_서버는_원장보다_중단_변수를_먼저_본다(fed, monkeypatch):
    """데이터 행이 있는 원장 + 중단 변수 — 사유는 중단 변수다(2판 §1-4 의 순서 2 가 멱등 · 원장보다 앞)."""
    from fl import server_app

    AtomicLog(fed.out / "atomic_log.csv", run_id=new_run_id("uni_fed", 7, "t1"), seed=7, cell="uni_fed",
              split_hash="s" * 64).log_round(round_idx=0, client_id="server", n_train_samples=0,
                                             metrics={"global_l2": 1.0})
    monkeypatch.setenv(F.NONCE_VAR, NONCE)
    cfg = {"cell": "uni_fed", "num-server-rounds": R, "local-epochs": E, "total-epochs": R * E,
           "project": str(reh(fed.tmp)), "base-seed": 7, "run-stamp": "t1", "split-hash": "s" * 64,
           **_uni_vals(fed.tmp)}
    with pytest.raises(UniFedRejected) as exc:
        server_app.main(None, SimpleNamespace(run_config=cfg))
    assert exc.value.code == "fault_env_forbidden"


def test_M7_클라이언트는_받은_설정을_읽기_전에_중단_변수를_본다(monkeypatch):
    from fl.client_app import uni_client_round

    monkeypatch.setenv(F.FAULT_VAR, "train:after_ckpt:uni_local_C1:ep=0")
    with pytest.raises(F.FaultRefused):
        uni_client_round({"client-tag-0": "C1", "local-epochs": 1, "num-rounds": 1, "base-seed": 1},
                         arrays_in=[], canonical_keys=["k"], round_idx=0, client_idx=0)   # 통합형 키가 없다


def test_M9_인프로세스_연합은_본실험을_받지_않는다(monkeypatch, tmp_path):
    from fl import pilot_sim

    monkeypatch.setattr(pilot_sim, "PILOT_CFG", {
        "cell": "uni_fed", "out_dir": str(tmp_path), "num_rounds": 1, "num_clients": 3, "canonical_keys": ["k"],
        "local_epochs": 1, "total_epochs": 1, "run_stamp": "t", "base_seed": 1, "purpose": "main"})
    with pytest.raises(UniFedRejected) as exc:
        pilot_sim._server_main(None, None)
    assert exc.value.code == "pilot_sim_main"


# ================================================================ M-3 (f) · M-13 연합 서버의 거부 사유
def _cell(run: UniFedRun, idx: int, **over) -> dict:
    """참여자 하나가 한 라운드에 보내는 값 — 서버의 등록값과 맞는 합성 값."""
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import TARGET_CONTRACT_SHA256, template_mode_of

    c = {"client-idx": float(idx), "supervised-tokens": 5.0, "num-examples": 1.0,
         "uni-client-tag": TAGS[idx], "uni-model-id": run.model_id, "uni-model-revision": run.model_revision or "",
         "uni-train-rows-digest": f"d{idx}", "uni-grid-observed": '{"[1, 2, 4]": 1}',
         "uni-processor-config-sha256": "c" * 64,
         "uni-config-blocks": json.dumps({k: {} for k in ("lora", "quant", "optimizer", "loss", "lr_schedule")}),
         "uni-impl-ids": "[]", "uni-prompt-sha256": run.prompt_sha256,
         "uni-template-mode": template_mode_of({"enable_thinking": False}),
         "uni-coord-cfg-hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")),
         "uni-target-contract-sha256": TARGET_CONTRACT_SHA256}
    c.update(over)
    return {k: v for k, v in c.items() if v is not None}


def _run(fed, **get_over) -> UniFedRun:
    return UniFedRun(get=_server_get(fed.tmp, **get_over), out_dir=fed.out, run_id="x", base_seed=7,
                     split_hash="s" * 64, num_rounds=R, local_epochs=E, total_epochs=R * E, client_tags=TAGS,
                     non_main_parent=fed.tmp)


_AGG = SimpleNamespace(ndarrays=[np.zeros(1)])


@pytest.mark.parametrize("over, code", [
    ({"uni-model-id": None}, "client_strings_missing"),
    ({"uni-model-id": ""}, "client_strings_empty"),
    ({"uni-train-rows-digest": ""}, "client_strings_empty"),
    ({"supervised-tokens": None}, "client_tokens_missing"),
    ({"num-examples": 2.0}, "grid_rows_mismatch"),
])
def test_M13_라운드의_참여자_값이_모자라면_그_라운드에서_거부한다(fed, over, code):
    run = _run(fed)
    with pytest.raises(UniFedRejected) as exc:
        run.observe(1, [_cell(run, 0, **over)], _AGG)
    assert exc.value.code == code


def _finish(run: UniFedRun, fed, rounds: dict, *, final=True) -> None:
    run.rounds = rounds
    run.final = (R, [np.zeros(1)]) if final else None
    _fed_ledger(fed.out / "atomic_log.csv", run.run_id, rounds=range(R), close=False)
    run.finish(atomic=None, report=SimpleNamespace(ok=True))


def _observed(run: UniFedRun, cells_of) -> dict:
    for sr in range(1, R + 1):
        run.observe(sr, [cells_of(sr, i) for i in range(len(TAGS))], _AGG)
    return run.rounds


@pytest.mark.parametrize("case, code", [
    ("initial", "initial_missing"),
    ("rounds", "rounds_observed"),
    ("clients", "clients_observed"),
    ("final", "final_missing"),
    ("tags", "client_tags"),
    ("drift", "client_value_drift"),
    ("grid", "grid_missing"),
    ("impl", "client_value_split"),
])
def test_M3_연합_서버가_닫기_전에_거부하는_사유(fed, case, code):
    run = _run(fed)
    if case != "initial":
        run.set_initial(fed.arrays, fed.keys)
    over = {
        "tags": lambda sr, i: {"uni-client-tag": TAGS[::-1][i]},
        "drift": lambda sr, i: {"uni-processor-config-sha256": ("c" if sr == 1 else "e") * 64},
        "grid": lambda sr, i: {"uni-grid-observed": "{}"},
        "impl": lambda sr, i: {"uni-impl-ids": json.dumps([{"i": i}])},
    }.get(case, lambda sr, i: {})
    rounds = _observed(run, lambda sr, i: _cell(run, i, **over(sr, i)))
    if case == "rounds":
        rounds = {1: rounds[1]}
    elif case == "clients":
        rounds = {sr: {k: v for k, v in per.items() if k != 2} for sr, per in rounds.items()}
    with pytest.raises(UniFedRejected) as exc:
        _finish(run, fed, rounds, final=case != "final")
    assert exc.value.code == code


@pytest.mark.parametrize("over, code", [
    ({"uni-expected-supervised-tokens": '{"C1": 1, "C2": 1}'}, "tokens_expected_keys"),
])
def test_M3_연합_서버가_시작하지_않는_사유(fed, over, code):
    with pytest.raises(UniFedRejected) as exc:
        _run(fed, **over)
    assert exc.value.code == code
    with pytest.raises(UniFedRejected) as exc:
        UniFedRun(get=_server_get(fed.tmp), out_dir=fed.out, run_id="x", base_seed=7, split_hash="s",
                  num_rounds=2, local_epochs=1, total_epochs=3, client_tags=TAGS, non_main_parent=fed.tmp)
    assert exc.value.code == "budget"


# ================================================================ I-5 토큰 기대값의 참여자 셋
@pytest.mark.parametrize("exp", [{"C1": 1, "C2": 1}, {}])
def test_I5_참여자가_빠진_기대값은_학습_전에_거부한다(tmp_path, exp):
    loader = CountingLoader()
    spec = _spec(tmp_path, expected_supervised_tokens=exp)
    _init(spec, loader)
    n = loader.n
    from vlm.train_cell import run_uni_central_cell

    with pytest.raises(CellRejected) as exc:
        run_uni_central_cell(spec=spec, model_loader=loader, rows=_rows(tmp_path)["C1"])
    assert exc.value.code == "tokens_expected_keys"
    assert loader.n == n                                   # 모델을 올리지 않았다


@pytest.mark.parametrize("exp", [{"C1": 1, "C2": 1}, {}, {"C1": 1, "C2": 1, "C3": 1, "central": 3}])
def test_I5_설정의_기대값도_참여자_셋이_정확해야_한다(exp):
    from tests.test_uni_path_passing import _doc
    from vlm.uni_config import UniConfigIncomplete, uni_config_from

    with pytest.raises(UniConfigIncomplete) as exc:
        uni_config_from(_doc(fixed={"uni_expected_supervised_tokens": exp}))
    assert any("참여자" in p for p in exc.value.problems)


# ================================================================ M-1 설정 밖에서 오는 값
def test_M1_프롬프트와_좌표_규약은_설정에서_온다(tmp_path):
    p = _cfg_yaml(tmp_path, uni_prompt_path="vlm/prompts/unified_pilot_v1.txt", uni_coord_space="NORM_1000")
    cfg = M.load_main_config(p)
    spec = M.spec_for(cfg, 1, out=tmp_path / "out", config_path=p)
    assert (spec.prompt_path, spec.coord_space) == ("vlm/prompts/unified_pilot_v1.txt", "NORM_1000")
    s = M.fed_run_config(cfg, 1, out=tmp_path / "out")
    assert 'uni-prompt="vlm/prompts/unified_pilot_v1.txt"' in s and 'uni-coord-space="NORM_1000"' in s


@pytest.mark.parametrize("drop", ["uni_prompt_path", "uni_coord_space"])
def test_M1_프롬프트나_좌표_규약이_없으면_학습하지_않는다(tmp_path, drop):
    p = _cfg_yaml(tmp_path, **{drop: None})
    with pytest.raises(SystemExit) as exc:
        M.load_main_config(p)
    assert exc.value.reason == "config_incomplete"


def test_M1_출력_루트는_저장소_루트_기준이고_다른_폴더에서는_시작하지_않는다(tmp_path, monkeypatch, capsys):
    assert M.OUT.is_absolute() and M.OUT == M.REPO_ROOT / "outputs" / "main_u"
    p = _cfg_yaml(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert M.main(["local", "--seed", "1"], out=tmp_path / "out", config_path=p) == 2
    assert "[cwd]" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fixed", [{"uni_prompt_sha256": "a" * 64}, {"uni_coord_space": "abs_orig"},
                                   {"uni_processor_kwargs": [1]}, {"uni_expected_supervised_tokens": {"C1": -1, "C2": 1, "C3": 1}},
                                   {"uni_expected_supervised_tokens": {"C1": True, "C2": 1, "C3": 1}},
                                   {"uni_expected_supervised_tokens": [1, 2, 3]}])
def test_M6_M1_설정_키의_꼴이_틀리면_읽을_때_멈춘다(fixed):
    from tests.test_uni_path_passing import _doc
    from vlm.uni_config import UniConfigIncomplete, uni_config_from

    with pytest.raises(UniConfigIncomplete):
        uni_config_from(_doc(fixed=fixed))


# ================================================================ M-2 초기 어댑터 캐시의 내구 저장
def test_M2_초기_어댑터_캐시는_찢긴_채로_남지_않는다(tmp_path, monkeypatch):
    import vlm.init_adapter as ia

    real = np.savez

    def torn(target, **kw):
        data = b"PK\x03\x04 torn"
        if hasattr(target, "write"):
            target.write(data)
        else:
            Path(target).write_bytes(data)
        raise RuntimeError("저장 도중에 죽었다")

    cache = reh(tmp_path) / "init" / "initial.npz"
    monkeypatch.setattr(np, "savez", torn)
    with pytest.raises(RuntimeError):
        ia.build_initial_adapter(model_id="tiny", seed=7, cache_path=cache, revision="r1", purpose="rehearsal",
                                 model_loader=CountingLoader())
    assert not cache.exists()                              # 찢긴 배열이 캐시로 읽히지 않는다
    monkeypatch.setattr(np, "savez", real)
    arrays, _keys, _ = ia.build_initial_adapter(model_id="tiny", seed=7, cache_path=cache, revision="r1",
                                               purpose="rehearsal", model_loader=CountingLoader())
    again, _, _ = ia.build_initial_adapter(model_id="tiny", seed=7, cache_path=cache, revision="r1",
                                           purpose="rehearsal", model_loader=CountingLoader())
    assert all(np.array_equal(a, b) for a, b in zip(arrays, again))


# ================================================================ M-3 (b)(c)(d)(f) · M-10 · M-11 · M-12 · M-16
def test_M3_실행기는_설정을_읽기_전에_중단_변수를_본다(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(F.FAULT_VAR, "train:after_ckpt:uni_central:ep=0")
    assert M.main(["local", "--seed", "1"], out=tmp_path / "out", config_path=tmp_path / "없는_설정.yaml") == 2
    assert "[fault_env]" in capsys.readouterr().err


def test_M3_이름의_대소문자는_사전으로_시험한다():
    """Windows 의 환경은 이름을 대문자로 저장한다 — 소문자 이름은 사전을 넘겨야 시험이 된다."""
    with pytest.raises(SystemExit) as exc:
        M.guard_env({"weld_rehearsal_fault": "x"})
    assert exc.value.reason == "fault_env"
    M.guard_env({"PATH": "x"})
    with pytest.raises(F.FaultRefused):
        F.assert_no_fault_env({"Weld_Rehearsal_Nonce": NONCE})


def test_M3_범위의_경계(tmp_path):
    root = reh(tmp_path)
    for ep, ok in ((1, True), (2, False)):
        fault, nonce = f"train:after_ckpt:uni_local_C1:ep={ep}", f"{ep:032x}"
        F.write_planned(root, nonce, fault, stage_no=3)
        env = {F.FAULT_VAR: fault, F.NONCE_VAR: nonce}
        if ok:
            assert F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_local_C1"],
                                     bounds={"ep": 2}, env=env) is not None
        else:
            with pytest.raises(F.FaultRefused) as exc:
                F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_local_C1"],
                                  bounds={"ep": 2}, env=env)
            assert exc.value.code == "fault_range"


class _Exit(RuntimeError):
    pass


def _armed(tmp_path) -> F.ArmedFault:
    root = reh(tmp_path)
    fault = "train:after_ckpt:uni_local_C1:ep=0"
    F.write_planned(root, NONCE, fault, stage_no=3)
    armed = F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_local_C1"], bounds={"ep": 2},
                              env={F.FAULT_VAR: fault, F.NONCE_VAR: NONCE})

    def exit_fn(code):
        raise _Exit(code)

    return dataclasses.replace(armed, exit_fn=exit_fn)


def test_M11_같은_열쇠의_활성화_기록이_있으면_fire_가_스스로_거부한다(tmp_path):
    armed = _armed(tmp_path)
    ctx = F.FaultContext(purpose="rehearsal", run_root=reh(tmp_path))
    with pytest.raises(_Exit):
        armed.fire(ctx, point="after_ckpt", tag="uni_local_C1", k=0)
    effects: list = []
    with pytest.raises(F.FaultRefused) as exc:
        armed.fire(ctx, point="after_ckpt", tag="uni_local_C1", k=0, effect=lambda: effects.append(1))
    assert exc.value.code == "already_activated" and effects == []
    assert len(list((reh(tmp_path) / "faults").glob("*.activated.json"))) == 1


def test_M3_활성화_기록의_이름이_겹치면_켜지지_않는다(tmp_path, monkeypatch):
    armed = _armed(tmp_path)
    d = armed.faults_dir
    # 기록이 하나 있으니 다음 순번은 2 다 — 그 이름의 기록이 이미 있다(앞 검사를 비켜 간 경합을 흉내 낸다).
    (d / f"002_{armed.spec.key}.activated.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(F.ArmedFault, "already_activated", lambda self: False)
    effects: list = []
    with pytest.raises(F.FaultRefused) as exc:
        armed.fire(F.FaultContext(purpose="rehearsal", run_root=reh(tmp_path)), point="after_ckpt",
                   tag="uni_local_C1", k=0, effect=lambda: effects.append(1))
    assert exc.value.code == "activation_exists" and effects == []


def test_M10_켜진_중단에_재개_폴더가_없으면_적재_전에_거부한다(tmp_path, monkeypatch):
    from vlm.train_cell import run_uni_local_cell

    spec = _spec(tmp_path, resume_root=None)
    loader = CountingLoader()
    _init(spec, loader)
    n = loader.n
    F.write_planned(reh(tmp_path), NONCE, "train:after_ckpt:uni_local_C1:ep=0", stage_no=3)
    monkeypatch.setenv(F.FAULT_VAR, "train:after_ckpt:uni_local_C1:ep=0")
    monkeypatch.setenv(F.NONCE_VAR, NONCE)
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=spec, model_loader=loader, rows_by_client=_rows(tmp_path))
    assert exc.value.code == "fault_needs_resume_root" and loader.n == n


def test_M7_원장_닫기도_가드가_먼저다(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    F.write_planned(reh(tmp_path), NONCE, "train:after_ckpt:uni_local_C1:ep=0", stage_no=3)
    monkeypatch.setenv(F.FAULT_VAR, "train:after_ckpt:uni_local_C1:ep=0")
    monkeypatch.setenv(F.NONCE_VAR, NONCE)
    with pytest.raises(CellRejected) as exc:
        close_ledger(spec, LOCAL_TAGS["C1"], rows=[])       # 산출물이 없다 — 그래도 사유는 가드다
    assert exc.value.code == "fault_not_for_this_call"


def test_M12_실행기는_중단_변수의_이름을_장치에서_가져온다():
    assert M.FAULT_VAR is F.FAULT_VAR and M.FAULT_VARS == (F.FAULT_VAR, F.NONCE_VAR)


@pytest.mark.parametrize("name", ["reh-20261001T000000\n", "reh-２０２６１００１T000000"])
def test_M16_루트_이름은_끝_개행과_유니코드_숫자를_받지_않는다(tmp_path, name):
    from vlm.run_root import run_root_problems

    root = tmp_path / name
    assert run_root_problems("rehearsal", root, [root / "x"], parent=tmp_path)


def test_M16_표식과_인자는_끝_개행과_유니코드_숫자를_받지_않는다(tmp_path):
    with pytest.raises(F.FaultRefused) as exc:
        F.check_fault_env("rehearsal", run_root=reh(tmp_path), stage="train", tags=["uni_local_C1"],
                          env={F.FAULT_VAR: "train:after_ckpt:uni_local_C1:ep=0", F.NONCE_VAR: NONCE + "\n"})
    assert exc.value.code == "nonce_form"
    with pytest.raises(F.FaultRefused) as exc:
        F.parse_fault("train:after_ckpt:uni_local_C1:ep=٣")
    assert exc.value.code == "fault_arg"


# ================================================================ 로컬 칸의 건너뜀도 어댑터 파일을 본다
def test_건너뛰기_전에_어댑터_파일의_sha_를_meta_와_맞댄다(tmp_path):
    from vlm.train_cell import run_uni_local_cell

    loader = CountingLoader()
    spec = _spec(tmp_path)
    _init(spec, loader)
    rows = _rows(tmp_path)
    run_uni_local_cell(["C1"], spec=spec, model_loader=loader, rows_by_client=rows)
    npz = cell_dir(spec, LOCAL_TAGS["C1"]) / "adapter_last.npz"
    with np.load(npz) as z:
        arrs = {k: z[k] + 1 for k in z.files}
    np.savez(npz, **arrs)
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=spec, model_loader=loader, rows_by_client=rows)
    assert exc.value.code == "adapter_sha_mismatch"


# ================================================================ M-4 등가 시험 실행기
def _stages(root: Path, tmp: Path):
    from scripts import verify_resume_uni as V
    from tests.uni_fakes import make_rows

    class _Died(RuntimeError):
        pass

    def die():
        raise _Died()

    kw = {"rows": make_rows(tmp / "img", "C1", 40), "epochs": 3, "die_after": 1, "model_loader": CountingLoader(),
          "purpose": "rehearsal", "die": die}
    V.run_stage("baseline", root, **kw)
    V.run_stage("baseline2", root, **kw)
    with pytest.raises(_Died):
        V.run_stage("crash", root, **kw)
    V.run_stage("resume", root, **kw)
    return V


def _edit_json(p: Path, fn) -> None:
    m = json.loads(p.read_text(encoding="utf-8"))
    fn(m)
    p.write_text(json.dumps(m), encoding="utf-8")


@pytest.mark.parametrize("how, check", [
    ("lr", "lr_trace_equal"), ("resumed", "resumed_from_next_epoch"), ("epochs", "epochs_ran_full"),
    ("counts", "counts_equal"), ("per_epoch", "per_epoch_equal"),
])
def test_M4_등가_판정은_회계_궤적_재개_지점을_모두_본다(tmp_path, how, check):
    root = tmp_path / "root"
    V = _stages(root, tmp_path)
    assert V.judge(root, epochs=3, die_after=1)["passed"]
    rm = root / "resume_run" / "metrics.json"
    if how == "lr":
        _edit_json(rm, lambda m: m["lr_trace_tail"][-1].__setitem__(1, 1.0))
    elif how == "resumed":
        _edit_json(rm, lambda m: m.update(resumed_from_epoch=1))
    elif how == "epochs":
        _edit_json(rm, lambda m: m.update(epochs_ran=2))
    elif how == "counts":
        _edit_json(rm, lambda m: m.update(optimizer_steps=m["optimizer_steps"] + 1))
    else:
        ep = root / "resume_run" / "epochs.jsonl"
        lines = ep.read_text(encoding="utf-8").splitlines()
        last = json.loads(lines[-1])
        last["mean_ce"] += 1.0
        ep.write_text("\n".join(lines[:-1] + [json.dumps(last)]) + "\n", encoding="utf-8")
    rep = V.judge(root, epochs=3, die_after=1)
    assert not rep["passed"] and rep["checks"][check] is False


def test_M4_기본_행_수는_부분_창을_탄다_그리고_자식은_중단_변수를_받지_않는다(tmp_path, monkeypatch):
    from scripts import verify_resume_uni as V
    from vlm.pilot_vlm import GRAD_ACCUM

    assert V.DEFAULT_ROWS % GRAD_ACCUM != 0
    monkeypatch.setenv(F.FAULT_VAR, "train:after_ckpt:uni_local_C1:ep=0")
    seen: list = []

    class _Stop(RuntimeError):
        pass

    def fake_run(cmd, **kw):
        seen.append(kw.get("env"))
        raise _Stop()

    monkeypatch.setattr(V.subprocess, "run", fake_run)
    with pytest.raises(_Stop):
        V.main(["--root", str(tmp_path / "r")])
    env, = seen
    assert env is not None and F.fault_env(env) == (None, None)
