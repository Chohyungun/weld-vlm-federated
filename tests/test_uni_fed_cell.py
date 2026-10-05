"""통합형 연합 칸의 어댑터 저장과 끝 행(리허설 2판 §0-1 의 ① 연합 행 · §2-6 의 연합 규칙 · §6-1 의 43).

Flower 전송 없이 서버 루프의 두 끝을 잇는다 — 클라이언트 라운드(`fl.client_vlm.run_client_round`)는 **실제
`train_rounds`** 로 돌고(모델 적재기만 합성 대역, CPU), 서버 쪽은 두 서버 경로가 함께 쓰는 라운드 기록 ·
회계 감사 · `fl.uni_fed.UniFedRun` 을 그대로 부른다. 집계는 산술 평균으로 대신한다(집계 산술은 이 시험의 대상이 아니다).

대역은 `rehearsal` 목적에서만 받으므로 리허설 목적으로 돈다. 연합 경로에는 적재기를 고르는 인자가 없어서
모듈 속성을 바꿔 끼운다 — 본실험 목적이면 그 대역이 부르기 전에 거부된다(`tests/test_uni_train_cell.py`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl.atomic_log import AtomicLog, RoundTimer, new_run_id  # noqa: E402
from fl.uni_fed import UniFedRejected, UniFedRun, assert_fresh_ledger, check_fed_ledger  # noqa: E402
from fl.uni_run_config import client_run_cfg, down_config  # noqa: E402
from tests.test_uni_train_cell import reh  # noqa: E402
from tests.uni_fakes import CountingLoader, fake_loader, make_rows, write_pairs  # noqa: E402

R, E = 2, 1
TAGS = ["C1", "C2", "C3"]


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    import torch

    was = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    assert was or not torch.cuda.is_initialized(), "이 시험이 CUDA 를 초기화했다"


def _server_get(tmp: Path, **over):
    vals = {"uni-model": "tiny", "uni-model-revision": "r1", "uni-pairs": str(tmp / "pairs.jsonl"),
            "uni-pairs-digest": "p" * 64, "uni-chat-template-kwargs": '{"enable_thinking": false}',
            "uni-coord-space": "ABS_ORIG", "purpose": "rehearsal", "plan": str(reh(tmp) / "plan.yaml"),
            "uni-seed-index": "1", "rehearsal-root": str(reh(tmp))}
    vals.update(over)
    return lambda k, d: vals.get(k, d)


@pytest.fixture
def fed(tmp_path, monkeypatch):
    from vlm import pilot_vlm
    from vlm.init_adapter import build_initial_adapter

    loader = CountingLoader(fake_loader)
    monkeypatch.setattr(pilot_vlm, "_load_model", loader)       # 연합 경로의 적재기 — 리허설만 받는다
    reh(tmp_path).mkdir(parents=True, exist_ok=True)
    (reh(tmp_path) / "plan.yaml").write_text("kind: rehearsal\n", encoding="utf-8")   # 계획 파일도 루트 아래다(2판 §1-4)
    rows = {c: make_rows(tmp_path / "img", c, n) for c, n in zip(TAGS, (3, 2, 1))}
    write_pairs(tmp_path / "pairs.jsonl", [r for c in TAGS for r in rows[c]])
    out = reh(tmp_path) / "fl" / "uni_fed"
    arrays, keys, _ = build_initial_adapter(model_id="tiny", seed=7, cache_path=out / "initial.npz",
                                            revision="r1", purpose="rehearsal", model_loader=loader)
    return SimpleNamespace(tmp=tmp_path, out=out, arrays=arrays, keys=keys, rows=rows, loader=loader)


def _run_fed(f, *, get=None, stamp="t1", tamper=None):
    """R 라운드 × 세 참여자. 두 서버 경로와 같은 부품으로 기록 · 감사 · 저장 · 끝 행까지 간다."""
    from fl.client_vlm import run_client_round
    from fl.round_wiring import finalize_accounting, make_round_recorder
    from fl.server_app import _cell_from_metrics, _with_uni_observer, build_accounting

    get = get or _server_get(f.tmp)
    run_id = new_run_id("uni_fed", 7, stamp)
    ledger = f.out / "atomic_log.csv"
    assert_fresh_ledger(ledger)
    uni_run = UniFedRun(get=get, out_dir=f.out, run_id=run_id, base_seed=7, split_hash="s" * 64,
                        num_rounds=R, local_epochs=E, total_epochs=R * E, client_tags=TAGS,
                        non_main_parent=f.tmp)
    uni_run.set_initial(f.arrays, f.keys)
    atomic = AtomicLog(ledger, run_id=run_id, seed=7, cell="uni_fed", split_hash="s" * 64)
    accounting = build_accounting(num_rounds=R, client_ids=[0, 1, 2], local_epochs=E, total_epochs=R * E)
    on_round_end = _with_uni_observer(make_round_recorder(
        accounting=accounting, atomic=atomic, timer=RoundTimer(), cell_from_metrics=_cell_from_metrics),
        uni_run)
    uni = client_run_cfg(down_config(get))
    glob = [a.copy() for a in f.arrays]
    for sr in range(1, R + 1):
        cells, outs = [], []
        for i, tag in enumerate(TAGS):
            arr, metrics, strings = run_client_round(
                adapter_in=glob, canonical_keys=f.keys, round_idx=sr - 1, client_idx=i,
                cfg={"client_tag": tag, "local_epochs": E, "num_rounds": R, "base_seed": 7,
                     "resume_root": None, "run_id": stamp, "uni": uni})
            if tamper is not None:
                tamper(sr, i, strings)
            cells.append({"node_id": i, **metrics, **strings})
            outs.append(arr)
        glob = [np.mean([o[j] for o in outs], axis=0).astype(outs[0][j].dtype) for j in range(len(f.keys))]
        on_round_end(sr, cells, SimpleNamespace(ndarrays=glob, total_examples=6, global_norm=0.0,
                                                bn_buffer_divergence=0.0, missing_variance_ratio=0.0))
    report = finalize_accounting(accounting=accounting, atomic=atomic, out_dir=f.out, num_rounds=R,
                                 client_ids=[0, 1, 2], raise_on_failure=False)
    assert report.ok, report.failures
    files = uni_run.finish(atomic=atomic, report=report)
    return SimpleNamespace(files=files, glob=glob, ledger=ledger, run_id=run_id,
                           meta=json.loads((f.out / "adapter_last.meta.json").read_text(encoding="utf-8")))


# ================================================================ 끝까지
def test_연합_칸이_키_있는_어댑터와_meta_를_내고_server_끝_행으로_닫힌다(fed):
    from vlm.train_cell import ledger_file_sha256, ledger_view

    r = _run_fed(fed)
    with np.load(fed.out / "adapter_last.npz") as z:
        assert list(z.files) == fed.keys
        for k, a in zip(fed.keys, r.glob):
            assert np.array_equal(z[k], a)                         # 마지막 라운드의 글로벌 어댑터
    m = r.meta
    assert (m["adapter_step"], m["adapter_step_unit"]) == (R, "fed_rounds")
    assert m["adapter_sha256"] == r.files["adapter_sha256"]
    view = ledger_view(r.ledger)
    assert view == {"run_id": r.run_id, "file_sha256": ledger_file_sha256(r.ledger), "final_step": R,
                    "cell": "uni_fed", "client": None}
    check_fed_ledger(r.ledger, num_rounds=R, num_clients=3, closed=True)
    ident = m["identity"]
    assert (ident["tag"], ident["cell"], ident["client"], ident["seed_index"], ident["purpose"]) == \
        ("uni_fed", "uni_fed", None, 1, "rehearsal")
    assert ident["model_id"] == "tiny"                             # 클라이언트가 실제로 학습한 모델
    assert list(m["train_rows_digest"]) == TAGS
    assert m["train_input_grid_observed"] == {"[1, 2, 4]": 6}      # 세 참여자의 행 합
    assert m["plan_sha256"] and m["fault_events"] == []
    assert all(i["approved"] is False for i in m["impl_ids"])
    seg, = m["resume_segments"]
    assert (seg["start_epoch"], seg["end_epoch"], seg["unit"]) == (0, R - 1, "fed_rounds")


def test_연합의_학습_설정_원문은_로컬_칸과_같다(fed):
    """칸과 무관한 등록값이다(리허설 2판 §2-5) — 같은 시드의 로컬 칸과 한 글자도 다르지 않아야 한다."""
    from vlm.train_cell import UniRunSpec, run_uni_local_cell

    r = _run_fed(fed)
    spec = UniRunSpec(purpose="rehearsal", run_stamp="t1", seed_index=1, seed_value=7, snapshot_digest="s" * 64,
                      model_id="tiny", model_revision="r1", pairs_path=str(fed.tmp / "pairs.jsonl"),
                      pairs_digest="p" * 64, chat_template_kwargs={"enable_thinking": False},
                      num_rounds=R, local_epochs=E, total_epochs=R * E, train_root=reh(fed.tmp) / "train",
                      init_adapter_path=fed.out / "initial.npz", plan_sha256="a" * 64, run_root=reh(fed.tmp),
                      non_main_parent=fed.tmp)
    loc = run_uni_local_cell(["C2"], spec=spec, rows_by_client={"C2": fed.rows["C2"]})
    lm = json.loads((loc["C2"].cell_dir / "adapter_last.meta.json").read_text(encoding="utf-8"))
    assert r.meta["train_config"] == lm["train_config"]
    assert r.meta["init_adapter_digest"] == lm["init_adapter_digest"]


def test_평가_쪽_검증기가_연합_출처를_받는다(fed):
    from evaluation.bundle_v14 import MODE_MODEL, LedgerView, TrainingArtifact, _check_training
    from evaluation.schema_v14 import tag_parts
    from vlm.train_cell import ledger_view

    r = _run_fed(fed)
    view = ledger_view(r.ledger)
    cell, client = tag_parts("uni_fed")
    side = {"scored_adapter_sha256": r.meta["adapter_sha256"], "scored_adapter_step": R,
            "train_run_id": r.meta["train_run_id"], "train_ledger_sha256": view["file_sha256"],
            "cell": cell, "client": client}
    rows = r.meta["train_rows_digest"]                     # 연합은 참여자별 사전이다 — 평가 쪽이 같은 꼴로 받는다
    assert isinstance(rows, dict) and list(rows) == TAGS
    art = TrainingArtifact(adapter_sha256=r.meta["adapter_sha256"], step=R, train_rows_digest=rows)
    rej: list = []
    _check_training(side, MODE_MODEL, art, LedgerView(**view), None, SimpleNamespace(stem="uni_fed_s1"), rej,
                    SimpleNamespace(train_rows_digest=dict(rows)))
    assert rej == [], rej


# ================================================================ 시작 전
def test_데이터_행이_있는_원장에서는_시작하지_않는다(fed):
    _run_fed(fed)
    with pytest.raises(UniFedRejected) as exc:
        assert_fresh_ledger(fed.out / "atomic_log.csv")
    assert exc.value.code == "fed_ledger_not_fresh"


def test_서버는_원장_검사를_초기_어댑터_적재_전에_한다(fed, monkeypatch):
    """원장은 **서버가 보는 폴더**(`project/fl/uni_fed`)에 쓰고 사유를 맞댄다 — 다른 폴더에 쓰면 신선도 검사가 파일을
    보지 못하고 지나가, 다른 사유로 떨어져도 이 시험이 통과한다(검수 14번 I-4 가)."""
    from fl import server_app
    from vlm import run_root

    monkeypatch.setattr(run_root, "NON_MAIN_PARENT", fed.tmp)      # 리허설 루트의 부모 — 파이썬 속성, 시험만
    run_id = new_run_id("uni_fed", 7, "t1")
    AtomicLog(fed.out / "atomic_log.csv", run_id=run_id, seed=7, cell="uni_fed", split_hash="s" * 64).log_round(
        round_idx=0, client_id="server", n_train_samples=0, metrics={"global_l2": 1.0})
    calls = []
    monkeypatch.setattr(server_app, "_load_initial", lambda *a, **k: calls.append(1))
    vals = {k: v for k, v in (("uni-model", "tiny"), ("uni-model-revision", "r1"),
                              ("uni-pairs", str(fed.tmp / "pairs.jsonl")), ("uni-pairs-digest", "p" * 64),
                              ("uni-chat-template-kwargs", '{"enable_thinking": false}'),
                              ("uni-coord-space", "ABS_ORIG"), ("uni-seed-index", "1"))}
    cfg = {"cell": "uni_fed", "num-server-rounds": R, "local-epochs": E, "total-epochs": R * E,
           "project": str(reh(fed.tmp)), "base-seed": 7, "run-stamp": "t1", "split-hash": "s" * 64,
           "purpose": "rehearsal", "plan": str(reh(fed.tmp) / "plan.yaml"), "rehearsal-root": str(reh(fed.tmp)),
           **vals}
    assert Path(cfg["project"]) / "fl" / "uni_fed" == fed.out                   # 서버가 보는 폴더에 썼다
    with pytest.raises(UniFedRejected) as exc:
        server_app.main(None, SimpleNamespace(run_config=cfg))
    assert exc.value.code == "fed_ledger_not_fresh"
    assert calls == []


_MAIN_FULL = {"purpose": "main", "uni-prompt": "vlm/prompts/unified_v2_absorig.txt",
              "uni-processor-kwargs": "{}", "uni-expected-supervised-tokens": '{"C1": 1, "C2": 1, "C3": 1}'}


@pytest.mark.parametrize("over, code", [(_MAIN_FULL, "main_with_plan"),
                                        ({"plan": ""}, "plan_missing")])
def test_목적과_계획이_어긋나면_시작하지_않는다(fed, over, code):
    with pytest.raises(UniFedRejected) as exc:
        UniFedRun(get=_server_get(fed.tmp, **over), out_dir=fed.out, run_id="x", base_seed=7,
                  split_hash="s", num_rounds=R, local_epochs=E, total_epochs=R * E, client_tags=TAGS,
                  non_main_parent=fed.tmp)
    assert exc.value.code == code


# ================================================================ 닫기 전의 대조
def test_참여자마다_학습_조건이_갈리면_닫지_않는다(fed):
    def tamper(sr, i, strings):
        if i == 2:
            strings["uni-prompt-sha256"] = "0" * 64

    with pytest.raises(UniFedRejected) as exc:
        _run_fed(fed, tamper=tamper)
    assert exc.value.code == "client_value_split"
    assert not (fed.out / "adapter_last.npz").exists()
    from fl.atomic_log import END_METRIC, last_data_row

    assert last_data_row(fed.out / "atomic_log.csv")["metric_name"] != END_METRIC


def test_라운드당_감독_토큰이_기대값과_다르면_닫지_않는다(fed):
    get = _server_get(fed.tmp, **{"uni-expected-supervised-tokens": '{"C1": 1, "C2": 1, "C3": 1}'})
    with pytest.raises(UniFedRejected) as exc:
        _run_fed(fed, get=get)
    assert exc.value.code == "supervised_tokens"
    assert not (fed.out / "adapter_last.npz").exists()


def test_프로세서_설정이_연합_학습과_meta_에_닿는다(fed):
    r = _run_fed(fed, get=_server_get(fed.tmp, **{"uni-processor-kwargs": '{"max_pixels": 4096}'}))
    assert r.meta["identity"]["processor_kwargs"] == {"max_pixels": 4096}
    from tests.uni_fakes import FakeProcessor
    from vlm.pilot_vlm import processor_config_sha256

    assert r.meta["processor_config_sha256"] == processor_config_sha256(FakeProcessor(max_pixels=4096))
    assert r.meta["processor_config_sha256"] != processor_config_sha256(FakeProcessor())


def test_감사를_통과하지_않은_실행은_닫지_않는다(fed):
    run = UniFedRun(get=_server_get(fed.tmp), out_dir=fed.out, run_id="x", base_seed=7, split_hash="s",
                    num_rounds=R, local_epochs=E, total_epochs=R * E, client_tags=TAGS, non_main_parent=fed.tmp)
    run.set_initial(fed.arrays, fed.keys)
    with pytest.raises(UniFedRejected) as exc:
        run.finish(atomic=None, report=SimpleNamespace(ok=False))
    assert exc.value.code == "audit_not_ok"


# ================================================================ 원장 규칙 ①~④(6-1 의 43)
def _ledger(tmp: Path, rows: list[tuple]) -> Path:
    p = tmp / "led.csv"
    log = AtomicLog(p, run_id="r", seed=1, cell="uni_fed", split_hash="s")
    for rd, cid, name, val in rows:
        log.log_round(round_idx=rd, client_id=cid, n_train_samples=0, metrics={name: val})
    return p


def _good(n_rounds=2, k=3):
    rows = []
    for rd in range(n_rounds):
        rows += [(rd, i, "optimizer_steps", 1.0) for i in range(k)]
        rows.append((rd, "server", "global_l2", 1.0))
    return rows


def test_옳은_연합_원장은_받는다(tmp_path):
    p = _ledger(tmp_path, _good() + [(2, "server", "final_adapter_step", 2.0)])
    check_fed_ledger(p, num_rounds=2, num_clients=3, closed=True)


@pytest.mark.parametrize("rows, code", [
    (_good()[:-1] + [(2, "server", "final_adapter_step", 2.0)], "fed_server_rounds"),
    (_good() + [(1, 0, "optimizer_steps", 1.0), (2, "server", "final_adapter_step", 2.0)], "fed_duplicate_row"),
    ([r if r[1] != 2 else (r[0], -1, r[2], r[3]) for r in _good()] + [(2, "server", "final_adapter_step", 2.0)],
     "fed_client_ids"),
    (_good() + [(2, 0, "final_adapter_step", 2.0)], "fed_end_value"),
    (_good() + [(2, "server", "final_adapter_step", 3.0)], "fed_end_value"),
    (_good(), "fed_end_missing"),
])
def test_연합_원장의_라운드_규칙이_어긋나면_거부한다(tmp_path, rows, code):
    p = _ledger(tmp_path, rows)
    with pytest.raises(UniFedRejected) as exc:
        check_fed_ledger(p, num_rounds=2, num_clients=3, closed=True)
    assert exc.value.code == code
