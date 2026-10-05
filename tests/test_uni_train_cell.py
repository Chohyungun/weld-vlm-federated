"""통합형 로컬·중앙 칸 실행 모듈(미니스펙 12번 C1) — 리허설 2판 §2-6 · §2-7 · §1-4 · §6-1 의 10 · 11 · 12 · 26 · 41.

모델 없이 돈다. `train_rounds` 를 **실제로** 부르되 모델 적재기만 합성 대역(`tests/uni_fakes.py`)으로 바꾼다 —
대역은 CPU 에 있어 CUDA 를 건드리지 않는다. 대역을 받는 목적은 `rehearsal` 뿐이므로 학습이 도는 시험은 리허설
목적으로 돌고, 본실험 목적은 **대역을 부르기 전에 거부되는 것**을 본다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl.atomic_log import END_METRIC, AtomicLog, LedgerClosed  # noqa: E402
from tests.uni_fakes import CountingLoader, make_rows, write_pairs  # noqa: E402
from vlm import pilot_vlm  # noqa: E402
from vlm.init_adapter import InitCacheRejected, adapter_proof, build_initial_adapter  # noqa: E402
from vlm.seams import SeamRejected  # noqa: E402
from vlm.train_cell import (CENTRAL_TAG, LOCAL_TAGS, CellRejected, LedgerOpen, UniRunSpec,  # noqa: E402
                            build_train_config, cell_dir, check_local_end_row, close_ledger,
                            ledger_file_sha256, ledger_view, read_ledger_rows, run_uni_central_cell,
                            run_uni_local_cell, _effective_epoch_rows)

PLAN = "a" * 64
REH = "reh-20261001T000000"


def reh(tmp: Path) -> Path:
    """임시 폴더를 비본실험 부모로 삼은 리허설 루트. 리허설 칸의 모든 경로가 이 아래다(2판 §1-4 의 순서 4)."""
    return tmp / REH


def _spec(tmp: Path, **over) -> UniRunSpec:
    r = reh(tmp)
    kw = dict(purpose="rehearsal", run_stamp="t1", seed_index=1, seed_value=7, snapshot_digest="s" * 64,
              model_id="tiny", model_revision="r1", pairs_path=str(tmp / "pairs.jsonl"), pairs_digest="p" * 64,
              chat_template_kwargs={"enable_thinking": False}, num_rounds=2, local_epochs=1, total_epochs=2,
              train_root=r / "train", init_adapter_path=r / "init" / "initial.npz",
              resume_root=r / "resume", plan_sha256=PLAN, run_root=r, non_main_parent=tmp)
    kw.update(over)
    return UniRunSpec(**kw)


def _init(spec: UniRunSpec, loader) -> None:
    build_initial_adapter(model_id=spec.model_id, seed=spec.seed_value, cache_path=spec.init_adapter_path,
                          revision=spec.model_revision, purpose="rehearsal", model_loader=loader)


def _rows(tmp: Path, sizes=(("C1", 3), ("C2", 2), ("C3", 1))) -> dict[str, list[dict]]:
    return {c: make_rows(tmp / "img", c, n) for c, n in sizes}


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """GPU 를 쓰지 않는다. 재개 파일 저장은 CUDA 가 **보이면** 그 난수 상태를 읽어 CUDA 를 초기화한다
    (`detection/resume._rng_snapshot`) — 보이지 않게 막는다. 이 시험이 CUDA 를 처음 올렸다면 떨어진다."""
    import torch

    was = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    assert was or not torch.cuda.is_initialized(), "이 시험이 CUDA 를 초기화했다"


@pytest.fixture
def world(tmp_path):
    loader = CountingLoader()
    spec = _spec(tmp_path)
    _init(spec, loader)
    return SimpleNamespace(tmp=tmp_path, spec=spec, loader=loader, rows=_rows(tmp_path))


def _meta(d: Path) -> dict:
    return json.loads((d / "adapter_last.meta.json").read_text(encoding="utf-8"))


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree_bytes(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): _sha_file(p) for p in sorted(d.rglob("*")) if p.is_file()}


# ================================================================ 이음새(§1-4 · §6-1 의 26)
def test_본실험은_인자로_온_대역을_부르기_전에_거부한다(tmp_path):
    loader = CountingLoader()
    with pytest.raises(SeamRejected):
        pilot_vlm.train_rounds(rows=[], epochs=1, round_idx=0, client_idx=0, base_seed=1,
                               model_loader=loader, purpose="main")
    assert loader.n == 0


def test_본실험은_모듈_속성으로_바꿔_끼운_대역도_거부한다(monkeypatch):
    loader = CountingLoader()
    monkeypatch.setattr(pilot_vlm, "_load_model", loader)
    with pytest.raises(SeamRejected):
        pilot_vlm.resolve_model_loader(None, purpose="main")
    with pytest.raises(SeamRejected):
        pilot_vlm.train_rounds(rows=[], epochs=1, round_idx=0, client_idx=0, base_seed=1)
    assert loader.n == 0


def test_실제_적재기는_승인되고_대역은_리허설에서만_받는다():
    fn, ids = pilot_vlm.resolve_model_loader(None, purpose="main")
    assert fn is pilot_vlm._APPROVED_MODEL_LOADERS[0] and ids[0]["approved"] is True
    assert ids[0]["source_path"] == "vlm/pilot_vlm.py" and ids[0]["qualname"] == "_load_model"
    loader = CountingLoader()
    _, ids = pilot_vlm.resolve_model_loader(loader, purpose="rehearsal")
    assert ids[0]["approved"] is False and ids[0]["qualname"] == "CountingLoader"
    with pytest.raises(SeamRejected):
        pilot_vlm.resolve_model_loader(loader, purpose="frame_diag")
    pilot_vlm.resolve_model_loader(loader, purpose="frame_diag", standin_allowed=True)
    with pytest.raises(ValueError):
        pilot_vlm.resolve_model_loader(loader, purpose="mian")
    assert loader.n == 0


# ================================================================ 초기 어댑터 proof(§1-4 · X-16)
def test_초기_어댑터_proof_에_목적과_구현이_실린다(world):
    proof = json.loads(world.spec.init_adapter_path.with_suffix(".proof.json").read_text(encoding="utf-8"))
    assert proof["purpose"] == "rehearsal" and proof["impl_ids"][0]["approved"] is False
    assert proof["revision"] == "r1" and len(proof["init_adapter_digest"]) == 64


def test_본실험은_proof_없는_캐시_목적이_다른_캐시_대역이_만든_캐시를_받지_않는다(tmp_path):
    cache = tmp_path / "initial.npz"
    np.savez(cache, k=np.zeros(2, np.float32))
    proof_p = cache.with_suffix(".proof.json")
    for proof in (None,
                  {"model_id": "M", "seed": 1, "revision": None, "purpose": "rehearsal",
                   "impl_ids": [{"approved": True}]},
                  {"model_id": "M", "seed": 1, "revision": None, "purpose": "main",
                   "impl_ids": [{"approved": False}]},
                  {"model_id": "M", "seed": 1, "revision": None}):
        if proof is None:
            proof_p.unlink(missing_ok=True)
        else:
            proof_p.write_text(json.dumps(proof), encoding="utf-8")
        before = _sha_file(cache)
        with pytest.raises(InitCacheRejected):
            build_initial_adapter(model_id="M", seed=1, cache_path=cache, purpose="main")
        assert _sha_file(cache) == before            # 캐시는 읽기만 한다
    from vlm.init_adapter import init_adapter_digest

    arr = [np.zeros(2, np.float32)]
    full_impl = {"seam": "model_loader", "approved": True, "module": "vlm.pilot_vlm", "qualname": "_load_model",
                 "source_path": "vlm/pilot_vlm.py", "blob_sha1": "b" * 40}
    proof_p.write_text(json.dumps({"model_id": "M", "seed": 1, "revision": None, "purpose": "main",
                                   "impl_ids": [full_impl], "init_adapter_digest": init_adapter_digest(arr),
                                   **adapter_proof(arr, ["k"])}), encoding="utf-8")
    _, keys, _ = build_initial_adapter(model_id="M", seed=1, cache_path=cache, purpose="main")
    assert keys == ["k"]
    loader = CountingLoader()
    with pytest.raises(SeamRejected):                 # 대역은 캐시를 보기 전에 걸린다
        build_initial_adapter(model_id="M", seed=1, cache_path=cache, purpose="main", model_loader=loader)
    assert loader.n == 0


# ================================================================ 목적의 입력(§1-4 의 순서 5)
@pytest.mark.parametrize("case, code", [("plan", "main_with_plan"), ("rows", "main_with_train_list"),
                                        ("hooks", "main_with_hooks")])
def test_본실험은_계획_학습목록_훅을_받지_않는다(tmp_path, case, code):
    spec = _spec(tmp_path, purpose="main", plan_sha256=PLAN if case == "plan" else None)
    kw = {}
    if case == "rows":
        kw["rows_by_client"] = _rows(tmp_path)
    if case == "hooks":
        kw["hooks"] = {"after_ckpt": lambda ep, ck: None}
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=spec, **kw)
    assert exc.value.code == code
    assert not (tmp_path / "train").exists()       # 파일을 하나도 쓰지 않았다


def test_리허설은_계획_해시가_있어야_한다(tmp_path):
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_spec(tmp_path, plan_sha256=None), model_loader=CountingLoader(),
                           rows_by_client=_rows(tmp_path))
    assert exc.value.code == "plan_missing"


def test_초기_어댑터가_없으면_멈춘다(tmp_path):
    loader = CountingLoader()
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_spec(tmp_path), model_loader=loader, rows_by_client=_rows(tmp_path))
    assert exc.value.code == "init_adapter_missing" and loader.n == 0


# ================================================================ 한 번 끝까지(§2-6 · §2-7 · 6-1 의 10 · 11)
def test_로컬_세_칸이_각자_공통_초기_어댑터에서_출발해_닫힌_원장과_meta_를_낸다(world):
    res = run_uni_local_cell(spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    with np.load(world.spec.init_adapter_path) as z:
        init_keys = list(z.files)
        init_arrays = [z[k] for k in init_keys]
    want = adapter_proof(init_arrays, init_keys)
    shas = set()
    for c, r in res.items():
        assert r.status == "trained"
        d = r.cell_dir
        assert d == cell_dir(world.spec, LOCAL_TAGS[c])
        m = _meta(d)
        # 주입이 공통 초기 어댑터와 같다 — 앞 칸의 어댑터를 이어 쓰지 않았다
        assert m["metrics"]["injected_proof"]["tensor_digest"] == want["tensor_digest"]
        # 파일 · 스텝 · 원장
        assert m["adapter_sha256"] == _sha_file(d / "adapter_last.npz") == r.adapter_sha256
        assert m["adapter_step_unit"] == "optimizer_steps"
        led = d / "train_ledger.csv"
        view = ledger_view(led)
        assert view == {"run_id": m["train_run_id"], "file_sha256": ledger_file_sha256(led),
                        "final_step": m["adapter_step"], "cell": "uni_local", "client": c}
        assert check_local_end_row(led) == m["adapter_step"] == 2      # ceil(n/32) × 2 epoch
        rows = read_ledger_rows(led)
        assert [r_["metric_name"] for r_ in rows[:2]] == ["process_start", "process_started_unix"]
        assert rows[-1]["metric_name"] == END_METRIC and rows[-1]["client_id"] == c
        assert {r_["client_id"] for r_ in rows} == {c}                  # 칸 태그가 아니라 참여자 이름
        # 신원 · 세그먼트 · 목적 칸
        assert m["identity"]["tag"] == LOCAL_TAGS[c] and m["identity"]["purpose"] == "rehearsal"
        assert m["identity_sha256"] == hashlib.sha256(
            pilot_vlm.canonical_json(m["identity"]).encode("utf-8")).hexdigest()
        seg, = m["resume_segments"]
        assert (seg["start_epoch"], seg["end_epoch"], seg["identity_sha256"]) == (0, 1, m["identity_sha256"])
        assert seg["ended_at"] is not None and seg["ended_at"] >= seg["started_at"]
        assert m["plan_sha256"] == PLAN and m["fault_events"] == []
        assert m["impl_ids"][0]["approved"] is False and m["impl_ids"][0]["qualname"] == "CountingLoader"
        assert m["train_rows_digest"] == pilot_vlm.train_rows_digest(world.rows[c])
        assert m["train_input_grid_observed"] == {"[1, 2, 4]": len(world.rows[c])}
        assert m["train_ledger_path"].endswith(f"train/{LOCAL_TAGS[c]}_s1/train_ledger.csv")
        shas.add(m["adapter_sha256"])
    assert len(shas) == 3
    assert world.loader.n == 1 + 3                  # 초기 어댑터 한 번 + 칸마다 한 번


def test_원장_해시는_원시_바이트이고_CRLF_를_고치지_않는다(world):
    res = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    led = res["C1"].cell_dir / "train_ledger.csv"
    raw = led.read_bytes()
    assert b"\r\n" in raw
    assert ledger_file_sha256(led) == hashlib.sha256(raw).hexdigest()
    assert ledger_file_sha256(led) != hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def test_평가_쪽_검증기가_학습_출처를_받는다(world):
    from evaluation.bundle_v14 import MODE_MODEL, LedgerView, TrainingArtifact, _check_training
    from evaluation.schema_v14 import tag_parts

    res = run_uni_local_cell(["C2"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    cen = run_uni_central_cell(spec=world.spec, model_loader=world.loader,
                               rows=world.rows["C1"] + world.rows["C2"])
    for tag, r in ((LOCAL_TAGS["C2"], res["C2"]), (CENTRAL_TAG, cen)):
        m = _meta(r.cell_dir)
        view = ledger_view(r.cell_dir / "train_ledger.csv")
        cell, client = tag_parts(tag)
        assert (view["cell"], view["client"]) == (cell, client)
        side = {"scored_adapter_sha256": m["adapter_sha256"], "scored_adapter_step": m["adapter_step"],
                "train_run_id": m["train_run_id"], "train_ledger_sha256": view["file_sha256"],
                "cell": cell, "client": client}
        # 칸이 학습한 행의 digest 는 어댑터 meta 의 자리에서 읽는다(평가 쪽 `TrainingArtifact.train_rows_digest`)
        art = TrainingArtifact(adapter_sha256=m["adapter_sha256"], step=m["adapter_step"],
                               train_rows_digest=m["train_rows_digest"])
        rej: list = []
        _check_training(side, MODE_MODEL, art, LedgerView(**view), None, SimpleNamespace(stem=f"{tag}_s1"), rej,
                        SimpleNamespace(train_rows_digest=m["train_rows_digest"]))
        assert rej == [], rej
        rej = []                                            # 등록의 학습 행 digest 가 다르면 거부된다
        _check_training(side, MODE_MODEL, art, LedgerView(**view), None, SimpleNamespace(stem=f"{tag}_s1"), rej,
                        SimpleNamespace(train_rows_digest="0" * 64))
        assert [x.detail.get("item") for x in rej] == ["train_rows_digest"]
        # 중간 스텝을 채점하려 하면 거부된다
        rej = []
        _check_training({**side, "scored_adapter_step": 1}, MODE_MODEL,
                        TrainingArtifact(adapter_sha256=m["adapter_sha256"], step=1), LedgerView(**view),
                        None, SimpleNamespace(stem=f"{tag}_s1"), rej)
        assert rej


# ================================================================ 학습 설정 원문(§2-5 · 6-1 의 41)
def test_학습_설정_원문은_칸과_무관하고_한_칸을_바꾸면_해시가_바뀐다(world):
    res = run_uni_local_cell(["C1", "C3"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    cen = run_uni_central_cell(spec=world.spec, model_loader=world.loader, rows=world.rows["C1"])
    metas = [_meta(r.cell_dir) for r in (res["C1"], res["C3"], cen)]
    assert len({m["train_config"] for m in metas}) == 1
    m = metas[0]
    assert hashlib.sha256(m["train_config"].encode("utf-8")).hexdigest() == m["train_config_sha256"]
    tc = json.loads(m["train_config"])
    assert m["train_config"] == pilot_vlm.canonical_json(tc)
    assert tc["template_mode"] == ",".join(f"{k}={v!r}" for k, v in sorted({"enable_thinking": False}.items()))
    assert (tc["num_rounds"], tc["local_epochs"], tc["total_epochs"]) == (2, 1, 2)
    assert tc["lora"] == {"r": 2, "alpha": 4, "dropout": 0.0, "bias": "none", "target_modules": ["proj"]}
    assert tc["quant"] is None and tc["optimizer"]["name"] == "AdamW"
    assert tc["loss"] == {"normalization": "supervised_token_sum", "supervised_logits_only": True}
    assert tc["lr_schedule"] == {"kind": "cosine_global_offset", "warmup_steps": 0}
    base = build_train_config(world.spec, prompt_sha256=tc["prompt_sha256"],
                              init_digest=tc["init_adapter_digest"], metrics=m["metrics"])
    assert pilot_vlm.canonical_json(base) == m["train_config"]
    for block, key, val in (("lora", "r", 3), ("quant", None, {"load_in_4bit": True}),
                            ("optimizer", "eps", 1.0), ("loss", "supervised_logits_only", False),
                            ("lr_schedule", "warmup_steps", 5)):
        mm = json.loads(json.dumps(m["metrics"]))
        if key is None:
            mm["config_blocks"][block] = val
        else:
            mm["config_blocks"][block][key] = val
        alt = build_train_config(world.spec, prompt_sha256=tc["prompt_sha256"],
                                 init_digest=tc["init_adapter_digest"], metrics=mm)
        assert pilot_vlm.canonical_json(alt) != m["train_config"], block


def test_타깃_계약이_build_target_의_출력과_맞는다():
    from vlm.coords import ImageGeom

    row = {"skeleton": {"defects": [{"type": "2011", "bbox_px": [1, 2, 3, 4]}], "verdict": None}}
    out = json.loads(pilot_vlm.build_target(row, ImageGeom(orig_w=64, orig_h=32)))
    assert list(out) == pilot_vlm.TARGET_CONTRACT["keys"]
    assert list(out["defects"][0]) == pilot_vlm.TARGET_CONTRACT["defect_keys"]
    assert out["verdict"] == pilot_vlm.TARGET_CONTRACT["verdict_missing"]
    assert out["cited_clauses"] == pilot_vlm.TARGET_CONTRACT["clauses_missing"]
    assert pilot_vlm.TARGET_CONTRACT_SHA256 == pilot_vlm.sha256_text(
        pilot_vlm.canonical_json(pilot_vlm.TARGET_CONTRACT))


def test_관측_격자는_행마다_한_번_센다(world):
    rows = make_rows(world.tmp / "img", "C1", 2) + make_rows(world.tmp / "img_w", "C1", 1, w=96, h=48, start=5)
    res = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client={"C1": rows})
    assert _meta(res["C1"].cell_dir)["train_input_grid_observed"] == {"[1, 2, 4]": 2, "[1, 3, 6]": 1}


# ================================================================ 멱등(§2-6 · 6-1 의 12)
def test_같은_신원은_모델_적재_없이_건너뛰고_다른_스탬프는_적재_전에_멈춘다(world):
    run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    n = world.loader.n
    again = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    assert again["C1"].status == "skipped" and world.loader.n == n
    other = _spec(world.tmp, run_stamp="t2")
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=other, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "identity_mismatch" and world.loader.n == n
    # 다른 학습 목록(부분 목록이 바뀜)도 다른 신원이다
    with pytest.raises(CellRejected):
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader,
                           rows_by_client={"C1": world.rows["C1"][:2]})
    assert world.loader.n == n


def test_닫힌_원장에는_덧붙이지_않는다(world):
    res = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    led = res["C1"].cell_dir / "train_ledger.csv"
    before = led.read_bytes()
    log = AtomicLog(led, run_id=_meta(res["C1"].cell_dir)["train_run_id"], seed=7, cell="uni_local",
                    split_hash="s" * 64)
    assert log.closed
    with pytest.raises(LedgerClosed):
        log.log_round(round_idx=9, client_id="C1", n_train_samples=1, metrics={"x": 1.0})
    assert led.read_bytes() == before


# ================================================================ 멈춤과 재개
# 아래의 "죽으면" 은 **같은 시험 프로세스 안에서 훅이 예외를 올려 학습을 멈춘 뒤 다시 부르는 것**이다.
# 콜백 순서 · 재개 분기 · 원장 · 세그먼트를 본다. 2판 §6-1 의 하위 프로세스 사망 · 재기동(`os._exit`) 시험이
# 아니다 — 그 시험은 고의 중단 장치와 함께 07 의 순서 2 에서 짠다.
class _Die(RuntimeError):
    pass


def _die_at(ep_die: int):
    def hook(ep, ckpt):
        if ep == ep_die:
            raise _Die(f"ep {ep}")
    return hook


def test_체크포인트_뒤에_죽으면_이어서_끝내고_세그먼트가_원장에서_나온다(world, capsys):
    spec = world.spec
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows,
                           hooks={"after_ckpt": _die_at(0)})
    d = cell_dir(spec, LOCAL_TAGS["C1"])
    assert not (d / "adapter_last.npz").exists()
    capsys.readouterr()
    res = run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows)
    err = capsys.readouterr().err
    assert "[resume]" in err and "epoch 1 부터" in err
    m = _meta(res["C1"].cell_dir)
    s1, s2 = m["resume_segments"]
    assert (s1["start_epoch"], s1["end_epoch"], s1["ended_at"]) == (0, 0, None)
    assert (s2["start_epoch"], s2["end_epoch"]) == (1, 1) and s2["ended_at"] is not None
    assert s1["identity_sha256"] == s2["identity_sha256"] == m["identity_sha256"]   # 재개 사슬로 이어졌다
    assert m["metrics"]["resumed_from_epoch"] == 1
    assert check_local_end_row(d / "train_ledger.csv") == m["adapter_step"] == 2


def test_체크포인트_전에_죽으면_그_epoch_를_다시_돌고_겹친_행은_뒤의_것이_이긴다(world):
    spec = world.spec
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows,
                           hooks={"before_ckpt": _die_at(1)})
    d = cell_dir(spec, LOCAL_TAGS["C1"])
    rows_dead = read_ledger_rows(d / "train_ledger.csv")
    assert sum(r["metric_name"] == "epochs_ran" and r["round"] == "1" for r in rows_dead) == 1
    run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows)
    rows = read_ledger_rows(d / "train_ledger.csv")
    assert sum(r["metric_name"] == "epochs_ran" and r["round"] == "1" for r in rows) == 2   # 겹친 행
    assert check_local_end_row(d / "train_ledger.csv") == 2
    segs = _meta(d)["resume_segments"]
    assert [(s["start_epoch"], s["end_epoch"]) for s in segs] == [(0, 1), (1, 1)]


def test_이음매_없이_같은_행이_두_번이면_원장이_깨진_것이다():
    rows = [{"metric_name": "process_start", "round": "0", "client_id": "C1", "metric_value": "0"},
            {"metric_name": "process_started_unix", "round": "0", "client_id": "C1", "metric_value": "1.5"},
            {"metric_name": "optimizer_steps", "round": "0", "client_id": "C1", "metric_value": "1"},
            {"metric_name": "optimizer_steps", "round": "0", "client_id": "C1", "metric_value": "1"}]
    with pytest.raises(ValueError, match="두 번"):
        _effective_epoch_rows(rows)


def test_재개_파일의_신원_해시가_다르면_적용하지_않고_파일을_지우지_않는다(world):
    """재개 신원(`UniResumeIdentity`)에 없는 차이 — 같은 수의 **다른** 학습 행 — 는 재개 파일에 실린
    신원 해시가 잡는다. 적용 전에 멈추고 재개 파일 · 원장을 건드리지 않는다."""
    spec = world.spec
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows,
                           hooks={"after_ckpt": _die_at(0)})
    resume = Path(spec.resume_root)
    before = _tree_bytes(resume)
    led = cell_dir(spec, LOCAL_TAGS["C1"]) / "train_ledger.csv"
    n_rows = len(read_ledger_rows(led))
    other = make_rows(world.tmp / "img_o", "C1", len(world.rows["C1"]), start=40)
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client={"C1": other})
    assert exc.value.code == "resume_identity_mismatch"
    assert _tree_bytes(resume) == before
    assert len(read_ledger_rows(led)) == n_rows          # 거부된 실행은 이음매 행을 쓰지 않았다


def test_프롬프트가_바뀌면_재개_신원이_적용_전에_거부한다(world):
    spec = world.spec
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows,
                           hooks={"after_ckpt": _die_at(0)})
    other_prompt = world.tmp / "prompt_b.txt"
    other_prompt.write_text("다른 프롬프트", encoding="utf-8")
    resume = Path(spec.resume_root)
    before = _tree_bytes(resume)
    with pytest.raises(ValueError, match="prompt_sha256"):
        run_uni_local_cell(["C1"], spec=_spec(world.tmp, prompt_path=str(other_prompt)),
                           model_loader=world.loader, rows_by_client=world.rows)
    assert _tree_bytes(resume) == before


def test_재개_신원이_어긋나면_원장을_더럽히지_않는다(world):
    spec = world.spec
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows,
                           hooks={"after_ckpt": _die_at(0)})
    led = cell_dir(spec, LOCAL_TAGS["C1"]) / "train_ledger.csv"
    before = led.read_bytes()
    # 페어 경로 문자열만 다르다 — 원장·멱등 신원은 같고 재개 신원(`data`)만 어긋난다
    with pytest.raises(ValueError):
        run_uni_local_cell(["C1"], spec=_spec(world.tmp, pairs_path=str(world.tmp / "other.jsonl")),
                           model_loader=world.loader, rows_by_client=world.rows)
    assert led.read_bytes() == before


def test_끝_행_앞에서_죽으면_숨기지_않고_close_ledger_가_세_대조_뒤에만_닫는다(world, monkeypatch):
    spec = world.spec
    real_close = AtomicLog.close

    def boom(self, **kw):
        raise _Die("끝 행 직전")

    monkeypatch.setattr(AtomicLog, "close", boom)
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows)
    monkeypatch.setattr(AtomicLog, "close", real_close)
    d = cell_dir(spec, LOCAL_TAGS["C1"])
    assert (d / "adapter_last.npz").exists()
    n = world.loader.n
    with pytest.raises(LedgerOpen) as exc:
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "ledger_open" and "close-ledger" in str(exc.value) and world.loader.n == n
    with pytest.raises(ValueError):
        ledger_view(d / "train_ledger.csv")              # 닫히지 않은 원장은 채점 쪽이 받지 않는다

    # 대조가 어긋나면 쓰지 않는다
    meta_p = d / "adapter_last.meta.json"
    good_meta = meta_p.read_bytes()
    bad = json.loads(good_meta)
    bad["adapter_step"] = 99
    meta_p.write_text(json.dumps(bad), encoding="utf-8")
    rows1 = world.rows["C1"]
    with pytest.raises(CellRejected) as exc:
        close_ledger(spec, LOCAL_TAGS["C1"], rows=rows1)
    assert exc.value.code == "close_step"
    meta_p.write_bytes(good_meta)
    npz = d / "adapter_last.npz"
    good_npz = npz.read_bytes()
    npz.write_bytes(good_npz + b"\0")
    with pytest.raises(CellRejected) as exc:
        close_ledger(spec, LOCAL_TAGS["C1"], rows=rows1)
    assert exc.value.code == "close_adapter_sha"
    npz.write_bytes(good_npz)
    led_before = (d / "train_ledger.csv").read_bytes()
    with pytest.raises(CellRejected) as exc:
        close_ledger(_spec(world.tmp, run_stamp="t9"), LOCAL_TAGS["C1"], rows=rows1)
    assert exc.value.code == "identity_mismatch" and (d / "train_ledger.csv").read_bytes() == led_before

    assert close_ledger(spec, LOCAL_TAGS["C1"], rows=rows1) == 2
    assert check_local_end_row(d / "train_ledger.csv") == 2
    with pytest.raises(CellRejected) as exc:
        close_ledger(spec, LOCAL_TAGS["C1"], rows=rows1)
    assert exc.value.code == "already_closed"
    again = run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows)
    assert again["C1"].status == "skipped" and world.loader.n == n


def test_중앙은_페어의_학습_분할_전체를_읽는다(world):
    rows = world.rows["C1"] + world.rows["C2"] + world.rows["C3"]
    val = make_rows(world.tmp / "img_v", "C1", 1, start=50)
    for r in val:
        r["split"] = "val"
    write_pairs(Path(world.spec.pairs_path), rows + val)
    cen = run_uni_central_cell(spec=world.spec, model_loader=world.loader)
    m = _meta(cen.cell_dir)
    assert m["identity"]["n_train_rows"] == len(rows)
    assert m["train_rows_digest"] == pilot_vlm.train_rows_digest(rows)
    assert ledger_view(cen.cell_dir / "train_ledger.csv")["client"] is None
    assert read_ledger_rows(cen.cell_dir / "train_ledger.csv")[-1]["client_id"] == "central"


# ================================================================ train_rounds 의 자리(소스 순서)
def test_이음매_훅과_stderr_줄이_상태_적용_앞에_있다():
    src = (ROOT / "vlm/pilot_vlm.py").read_text(encoding="utf-8")
    body = src[src.index("state = latest_resume("):src.index("start_ep = state.next_epoch")]
    i_hook = body.index("on_identity_ok(")
    i_err = body.index("file=sys.stderr")
    i_apply = body.index("apply_resume(")
    assert i_hook < i_err < i_apply


def test_원장_콜백_값과_체크포인트_extra_합치기():
    import ast

    src = (ROOT / "vlm/pilot_vlm.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "train_rounds")
    body = ast.unparse(fn)
    for k in ("epochs_ran", "optimizer_steps", "supervised_tokens", "mean_ce", "lr", "param_l2", "peak_vram_gb"):
        assert f"'{k}'" in body, k
    assert "**dict(ckpt.extra or {})" in body and "**dict(resume_extra or {})" in body
    i_led = body.index("ledger_cb({")
    i_ck = body.index("ckpt(_AdapterTrainerView(model, opt, epoch=ep))")
    assert i_led < i_ck                                   # 원장 행은 체크포인트 전에 쓴다
