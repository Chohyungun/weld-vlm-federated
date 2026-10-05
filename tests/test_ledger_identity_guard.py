"""34번 §3-1 8-4 — 새 정책의 기존 출력 충돌 거부.

`loader-reseed-per-epoch=True` 같은 **정책만 다른** 실행이 기존 `outputs/main_c/seedN` 위에
돌면 원자 로그가 append 돼 두 실행이 한 원장에 섞인다(26번 §2-1·§7). 경로 접두사 규칙은
검토에서 기각됐으므로 **기존 원장(첫 데이터 행)·기존 meta 와의 신원 대조**로 거부한다.
정책은 원장 열에 없으므로 run_id 의 stamp 에 접미(`_rs1`)로 새긴다.

뒤호환(총괄 게이트 조건): 정책이 꺼진 실행의 run_id 는 한 글자도 바뀌지 않고, 접미 없는
옛 원장·옛 meta 는 고치지 않고 그대로 읽힌다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fl.server_app

from fl.atomic_log import (FIELDS, IDENTITY_FIELDS, POLICY_SUFFIX_RESEED, AtomicLog,
                           LedgerIdentityMismatch, assert_ledger_compatible, new_run_id,
                           policy_stamp, read_ledger_identity)

SEED1_RUN_ID = "sep_fed_s20260828_main_s1"      # 시드 1 ④ 원장의 실제 run_id 문자열


# ==========================================================================
# 정책 접미 규약 — 뒤호환
# ==========================================================================

def test_정책이_꺼진_stamp_는_한_글자도_바뀌지_않는다():
    assert policy_stamp("main_s1", False) == "main_s1"
    assert new_run_id("sep_fed", 20260828, policy_stamp("main_s1", False)) == SEED1_RUN_ID


def test_정책이_켜진_stamp_는_접미가_붙고_run_id_가_갈린다():
    assert POLICY_SUFFIX_RESEED == "_rs1"
    assert policy_stamp("main_s1", True) == "main_s1_rs1"
    assert new_run_id("sep_fed", 1, policy_stamp("x", True)) != new_run_id("sep_fed", 1, policy_stamp("x", False))


@pytest.mark.parametrize("bad", ["true", 1, None, 0.0])
def test_정책은_진짜_bool_만_받는다(bad):
    """문자열 'true' 가 조용히 참이 되면 run_id 에 정책이 잘못 새겨진다."""
    with pytest.raises(ValueError):
        policy_stamp("x", bad)


# ==========================================================================
# 원장 신원 읽기·대조 — 접미 없는 옛 원장이 그대로 읽힌다
# ==========================================================================

def _write_ledger(path: Path, *, run_id: str, seed: int, cell: str, split_hash: str, rows: int = 1) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i in range(rows):
            w.writerow({"run_id": run_id, "seed": seed, "cell": cell, "split_hash": split_hash,
                        "client_id": 0, "round": i, "n_train_samples": 1, "metric_name": "m",
                        "metric_value": 1.0, "bytes_up": 0, "bytes_down": 0, "wall_time": 0.0})


def test_옛_원장의_첫_행이_신원이다(tmp_path):
    p = tmp_path / "atomic_log.csv"
    _write_ledger(p, run_id=SEED1_RUN_ID, seed=20260828, cell="sep_fed", split_hash="1f80e98b", rows=3)
    assert read_ledger_identity(p) == {"run_id": SEED1_RUN_ID, "seed": "20260828",
                                       "cell": "sep_fed", "split_hash": "1f80e98b"}


def test_신원은_첫_데이터_행에서_읽는다__뒤_행이_섞여도(tmp_path):
    """원장은 append 전용이라 **첫 행이 소유자**다. 마지막 행을 읽으면 혼입된 원장에서
    나중 실행이 소유자로 보여 검사가 뒤집힌다."""
    p = tmp_path / "atomic_log.csv"
    _write_ledger(p, run_id="owner", seed=7, cell="sep_fed", split_hash="h", rows=2)
    with p.open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=FIELDS).writerow(
            {"run_id": "intruder", "seed": 8, "cell": "uni_fed", "split_hash": "other",
             "client_id": 0, "round": 9, "n_train_samples": 1, "metric_name": "m",
             "metric_value": 1.0, "bytes_up": 0, "bytes_down": 0, "wall_time": 0.0})
    assert read_ledger_identity(p)["run_id"] == "owner"
    with pytest.raises(LedgerIdentityMismatch, match="run_id"):
        AtomicLog(p, run_id="intruder", seed=8, cell="uni_fed", split_hash="other")
    AtomicLog(p, run_id="owner", seed=7, cell="sep_fed", split_hash="h")     # 소유자는 통과


def test_잘린_첫_데이터_행은_KeyError_대신_사유와_함께_죽는다(tmp_path):
    """첫 `write` 도중 사망하면 첫 행이 잘려 있을 수 있다. 소유자를 모르면 통과시키지 않는다."""
    p = tmp_path / "atomic_log.csv"
    p.write_text(",".join(FIELDS) + "\nsep_fed_s7_x,7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="손상돼 신원을 읽을 수 없다"):
        read_ledger_identity(p)
    with pytest.raises(ValueError, match="손상돼 신원을 읽을 수 없다"):
        AtomicLog(p, run_id="sep_fed_s7_x", seed=7, cell="sep_fed", split_hash="h")


def test_신원_4열만_온전하면_뒤가_잘려도_읽는다(tmp_path):
    p = tmp_path / "atomic_log.csv"
    p.write_text(",".join(FIELDS) + "\nrid,7,sep_fed,h,0\n", encoding="utf-8")
    assert read_ledger_identity(p) == {"run_id": "rid", "seed": "7", "cell": "sep_fed",
                                       "split_hash": "h"}


def test_신원_필드_상수는_넷이다():
    assert IDENTITY_FIELDS == ("run_id", "seed", "cell", "split_hash")


def test_부재_빈파일_헤더만은_신원이_없다(tmp_path):
    p = tmp_path / "atomic_log.csv"
    assert read_ledger_identity(p) is None
    p.write_text("", encoding="utf-8")
    assert read_ledger_identity(p) is None
    p.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
    assert read_ledger_identity(p) is None
    assert assert_ledger_compatible(p, run_id="x", seed=0, cell="sep_fed", split_hash="h") is None


def test_같은_run_id_재기동은_옛_원장_위에_이어_쓴다(tmp_path):
    """(b) 같은 실행의 재기동 — 원장은 그 실행의 것이다. 접미 없는 옛 값과 그대로 맞아야 한다."""
    p = tmp_path / "atomic_log.csv"
    _write_ledger(p, run_id=SEED1_RUN_ID, seed=20260828, cell="sep_fed", split_hash="1f80e98b")
    log = AtomicLog(p, run_id=new_run_id("sep_fed", 20260828, policy_stamp("main_s1", False)),
                    seed=20260828, cell="sep_fed", split_hash="1f80e98b")
    log.log_round(round_idx=1, client_id=0, n_train_samples=1, metrics={"m": 2.0})
    assert [r["run_id"] for r in log.read_rows()] == [SEED1_RUN_ID, SEED1_RUN_ID]


def test_같은_경로_다른_run_id_는_거부하고_원장을_건드리지_않는다(tmp_path):
    """(a)"""
    p = tmp_path / "atomic_log.csv"
    _write_ledger(p, run_id=new_run_id("sep_fed", 7, "main_s1"), seed=7, cell="sep_fed", split_hash="h")
    before = p.read_bytes()
    with pytest.raises(LedgerIdentityMismatch, match="다른 실행의 원장 위에 쓰지 않는다"):
        AtomicLog(p, run_id=new_run_id("sep_fed", 7, "main_s9"), seed=7, cell="sep_fed", split_hash="h")
    assert p.read_bytes() == before


def test_정책만_다른_실행은_run_id_접미로_거부된다(tmp_path):
    """(c) 원장 열에 정책 필드가 없어도 run_id 가 갈려 잡힌다."""
    p = tmp_path / "atomic_log.csv"
    _write_ledger(p, run_id=new_run_id("sep_fed", 7, policy_stamp("main_s1", False)),
                  seed=7, cell="sep_fed", split_hash="h")
    with pytest.raises(LedgerIdentityMismatch, match="run_id"):
        AtomicLog(p, run_id=new_run_id("sep_fed", 7, policy_stamp("main_s1", True)),
                  seed=7, cell="sep_fed", split_hash="h")


@pytest.mark.parametrize("field,value", [("seed", 8), ("split_hash", "other"), ("cell", "uni_fed")])
def test_seed_split_cell_이_다르면_거부한다(tmp_path, field, value):
    p = tmp_path / "atomic_log.csv"
    _write_ledger(p, run_id="rid", seed=7, cell="sep_fed", split_hash="h")
    kw = dict(run_id="rid", seed=7, cell="sep_fed", split_hash="h")
    kw[field] = value
    with pytest.raises(LedgerIdentityMismatch, match=field):
        AtomicLog(p, **kw)


def test_빈_out_dir_는_통과하고_헤더를_쓴다(tmp_path):
    """(d)"""
    p = tmp_path / "new" / "atomic_log.csv"
    AtomicLog(p, run_id="rid", seed=7, cell="sep_fed", split_hash="h")
    assert p.read_text(encoding="utf-8").splitlines() == [",".join(FIELDS)]


# ==========================================================================
# ④ fl/server_app.main — 초기 가중치를 읽기 전에 거부(LedgerIdentityMismatch 전파)
# ==========================================================================

def _server_cfg(project: Path, *, stamp="main_s1", policy=False, seed=7, split="h"):
    return {
        "cell": "sep_fed", "num-server-rounds": 2, "local-epochs": 1, "total-epochs": 2,
        "num-clients": 3, "num-classes": 4, "base-seed": seed, "run-stamp": stamp,
        "split-hash": split, "model": "yolo11s.pt", "project": str(project),
        "views-root": str(project / "views"), "profile": "main", "num-examples": "1,1,1",
        "loader-reseed-per-epoch": policy, "resume-root": "",
    }


def _ledger_run_id(atomic) -> str:
    """서버가 연 원장에 한 줄 써서 **파일에 남는** run_id 를 읽는다."""
    atomic.log_round(round_idx=0, client_id=0, n_train_samples=1, metrics={"m": 1.0})
    return atomic.read_rows()[0]["run_id"]


def _drive_server(monkeypatch):
    """server_app.main 을 학습 없이 끝까지 통과시키는 가짜들. 무엇이 불렸는지 기록한다."""
    from fl import server_app

    calls: dict[str, object] = {"load_initial": 0, "start": None, "finalize": 0, "atomic": None}

    def fake_load_initial(cell, cfg):
        calls["load_initial"] += 1
        return [np.zeros(1, dtype=np.float32)], ["w"], {"w": None}

    class FakeStrategy:
        def __init__(self, **kw):
            pass

        def start(self, *, grid, initial_arrays, num_rounds, train_config):
            calls["start"] = dict(train_config)

    def fake_finalize(**kw):
        calls["finalize"] += 1
        # 서버가 **실제로 연 원장 객체**를 붙잡는다. train_config 만 보면 AtomicLog 에
        # 다른 run_id 를 넘기는 결함(변이 srvledger)이 잡히지 않는다.
        calls["atomic"] = kw["atomic"]

    monkeypatch.setattr(server_app, "_load_initial", fake_load_initial)
    monkeypatch.setattr(server_app, "WeldFedAvg", FakeStrategy)
    monkeypatch.setattr(server_app, "finalize_accounting", fake_finalize)
    return server_app, calls


def test_서버는_빈_out_dir_에서_통과하고_run_stamp_가_그대로다(monkeypatch, tmp_path):
    """(d) + 뒤호환: 정책 off 면 run_id·run-stamp 에 접미가 없다."""
    server_app, calls = _drive_server(monkeypatch)
    server_app.main(object(), SimpleNamespace(run_config=_server_cfg(tmp_path)))
    assert calls["load_initial"] == 1 and calls["finalize"] == 1
    assert calls["start"]["run-stamp"] == "main_s1"
    ledger = tmp_path / "fl" / "sep_fed" / "atomic_log.csv"
    assert ledger.read_text(encoding="utf-8").splitlines() == [",".join(FIELDS)]
    assert _ledger_run_id(calls["atomic"]) == SEED1_RUN_ID.replace("s20260828", "s7")


def test_서버는_다른_run_id_의_원장_위에서_죽고_가중치를_읽지_않는다(monkeypatch, tmp_path):
    """(a) 거부는 `_load_initial` 앞이다 — 모델을 올린 뒤 죽으면 GPU·시간을 버린다."""
    server_app, calls = _drive_server(monkeypatch)
    ledger = tmp_path / "fl" / "sep_fed" / "atomic_log.csv"
    ledger.parent.mkdir(parents=True)
    _write_ledger(ledger, run_id=new_run_id("sep_fed", 7, "main_s1"), seed=7, cell="sep_fed", split_hash="h")
    before = ledger.read_bytes()
    with pytest.raises(LedgerIdentityMismatch, match="다른 실행의 원장 위에 쓰지 않는다"):
        server_app.main(object(), SimpleNamespace(run_config=_server_cfg(tmp_path, stamp="main_s9")))
    assert calls["load_initial"] == 0 and calls["start"] is None and calls["finalize"] == 0
    assert ledger.read_bytes() == before


def test_거부_예외는_flwr_의_except_Exception_이_받는_종류다():
    """`SystemExit`(BaseException)로 던지면 SuperLink 는 run 을 FAILED 로 못 적고,
    인프로세스 시뮬레이션은 서버 **스레드**의 SystemExit 를 무시해 성공으로 끝난다(실측).
    그래서 이 거부는 `Exception` 계열이어야 하고, 서버는 감싸지 않는다."""
    import ast

    assert issubclass(LedgerIdentityMismatch, ValueError)
    assert not issubclass(LedgerIdentityMismatch, SystemExit)
    tree = ast.parse(Path(fl.server_app.__file__).read_text(encoding="utf-8"))
    raised = [
        getattr(getattr(node.exc, "func", node.exc), "id", "")
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef) and fn.name == "main"
        for node in ast.walk(fn)
        if isinstance(node, ast.Raise) and node.exc is not None
    ]
    assert "SystemExit" not in raised, raised


def test_서버는_같은_run_id_재기동을_통과시킨다(monkeypatch, tmp_path):
    """(b)"""
    server_app, calls = _drive_server(monkeypatch)
    ledger = tmp_path / "fl" / "sep_fed" / "atomic_log.csv"
    ledger.parent.mkdir(parents=True)
    _write_ledger(ledger, run_id=new_run_id("sep_fed", 7, "main_s1"), seed=7, cell="sep_fed", split_hash="h")
    server_app.main(object(), SimpleNamespace(run_config=_server_cfg(tmp_path)))
    assert calls["start"]["run-stamp"] == "main_s1"


def test_서버는_정책만_다른_실행을_거부한다(monkeypatch, tmp_path):
    """(c) 기존 원장은 정책 off(접미 없음). `loader-reseed-per-epoch=true` 로 같은 경로에 오면 거부."""
    server_app, calls = _drive_server(monkeypatch)
    ledger = tmp_path / "fl" / "sep_fed" / "atomic_log.csv"
    ledger.parent.mkdir(parents=True)
    _write_ledger(ledger, run_id=new_run_id("sep_fed", 7, "main_s1"), seed=7, cell="sep_fed", split_hash="h")
    with pytest.raises(LedgerIdentityMismatch, match="run_id"):
        server_app.main(object(), SimpleNamespace(run_config=_server_cfg(tmp_path, policy=True)))
    assert calls["load_initial"] == 0


def test_서버는_정책_on_이면_run_id_와_run_stamp_에_접미를_새긴다(monkeypatch, tmp_path):
    """정책 실행은 새 경로에서 통과하고, 원장 run_id 와 클라이언트 run-stamp 가 같은 접미를 받는다."""
    server_app, calls = _drive_server(monkeypatch)
    server_app.main(object(), SimpleNamespace(run_config=_server_cfg(tmp_path, policy=True)))
    assert calls["start"]["run-stamp"] == "main_s1_rs1"
    assert calls["start"]["loader-reseed-per-epoch"] is True
    # 원장에 남는 run_id 도 같은 접미를 받아야 한다 — 아니면 정책 on 원장 위에서
    # 정책 off 실행이 통과해 (c) 보장이 깨진다.
    assert _ledger_run_id(calls["atomic"]) == f"sep_fed_s7_main_s1{POLICY_SUFFIX_RESEED}"


def test_서버_원장_검사는_초기_가중치_적재보다_앞에_있다():
    """소스 순서 고정 — 검사가 `_load_initial` 뒤로 밀리면 거부가 늦어진다."""
    src = Path(fl.server_app.__file__).read_text(encoding="utf-8")
    body = src[src.index("def main(grid"):]
    assert body.index("assert_ledger_compatible(") < body.index("_load_initial(cell, cfg)")


# ==========================================================================
# ⑦ fl/pilot_sim — in-process 파일럿도 같은 규약(F01 이 정책을 연결한 경로)
# ==========================================================================

def _pilot_cfg(out_dir: Path, *, stamp="pilot_s1", policy=False, seed=7, split="h"):
    return {
        "cell": "sep_fed", "out_dir": str(out_dir), "num_rounds": 2, "num_clients": 3,
        "local_epochs": 1, "total_epochs": 2, "base_seed": seed, "run_stamp": stamp,
        "split_hash": split, "canonical_keys": ["w"], "reference_sd": {},
        "initial_arrays": [np.zeros(1, dtype=np.float32)], "resume_root": "",
        "views_root": str(out_dir / "views"), "model": "yolo11s.pt", "profile": "main",
        "num_examples": [1, 1, 1], "loader_reseed_per_epoch": policy,
    }


def _drive_pilot(monkeypatch, cfg):
    from fl import pilot_sim

    calls: dict[str, object] = {"start": None, "atomic": None}

    class FakeStrategy:
        def __init__(self, **kw):
            pass

        def start(self, *, grid, initial_arrays, num_rounds, train_config):
            calls["start"] = dict(train_config)

    monkeypatch.setattr(pilot_sim, "WeldFedAvg", FakeStrategy)
    monkeypatch.setattr(pilot_sim, "finalize_accounting",
                        lambda **kw: calls.__setitem__("atomic", kw["atomic"]))
    monkeypatch.setattr(pilot_sim, "PILOT_CFG", cfg)
    return pilot_sim, calls


@pytest.mark.parametrize("policy,suffix", [(False, ""), (True, POLICY_SUFFIX_RESEED)])
def test_파일럿도_정책을_stamp_에_새기고_원장에_같은_run_id_를_쓴다(monkeypatch, tmp_path, policy, suffix):
    cfg = _pilot_cfg(tmp_path, policy=policy)
    pilot_sim, calls = _drive_pilot(monkeypatch, cfg)
    pilot_sim._server_main(object(), SimpleNamespace())
    assert calls["start"]["run-stamp"] == f"pilot_s1{suffix}"
    assert _ledger_run_id(calls["atomic"]) == f"sep_fed_s7_pilot_s1{suffix}"


def test_파일럿은_다른_신원의_원장_위에서_거부한다(monkeypatch, tmp_path):
    ledger = tmp_path / "atomic_log.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    _write_ledger(ledger, run_id="sep_fed_s7_pilot_s1", seed=7, cell="sep_fed", split_hash="h")
    pilot_sim, _ = _drive_pilot(monkeypatch, _pilot_cfg(tmp_path, policy=True))
    with pytest.raises(LedgerIdentityMismatch, match="run_id"):
        pilot_sim._server_main(object(), SimpleNamespace())


def test_서버와_파일럿이_같은_설정에서_같은_run_id_를_낸다(monkeypatch, tmp_path):
    """두 진입점이 갈리면 한쪽만 보호된다(85번 ⑤ 와 같은 부류)."""
    from fl.atomic_log import new_run_id as nrid

    pilot_sim, pilot_calls = _drive_pilot(monkeypatch, _pilot_cfg(tmp_path / "p", stamp="x", policy=True))
    pilot_sim._server_main(object(), SimpleNamespace())
    server_app, server_calls = _drive_server(monkeypatch)
    server_app.main(object(), SimpleNamespace(run_config=_server_cfg(tmp_path / "s", stamp="x", policy=True)))
    assert _ledger_run_id(pilot_calls["atomic"]) == _ledger_run_id(server_calls["atomic"]) \
        == nrid("sep_fed", 7, "x_rs1")
    assert pilot_calls["start"]["run-stamp"] == server_calls["start"]["run-stamp"] == "x_rs1"


# ==========================================================================
# ②③ detection/train_cell — 기존 원장 + 기존 meta 와 대조
# ==========================================================================

def _drive_cells(monkeypatch):
    from detection import train_cell
    from detection.round_runner import RoundResult, derive_seed

    calls: list[dict] = []

    def fake_train(**kw):
        calls.append(kw)
        return RoundResult(
            ndarrays=[np.zeros(2, dtype=np.float32)], num_examples=1, round_idx=0,
            client_idx=kw["client_idx"], epochs_ran=2,
            seed=derive_seed(kw["base_seed"], 0, kw["client_idx"]),
            optimizer_steps=1, param_l2_norm=0.0, payload_bytes=8,
            loader_reseed_per_epoch=kw["loader_reseed_per_epoch"],
        )

    monkeypatch.setattr(train_cell, "train_round", fake_train)
    return train_cell, calls


def _run(train_cell, kind: str, out: Path, *, policy=False, split="h", seed=7, stamp="main_s1"):
    common = dict(model="m", total_epochs=2, base_seed=seed, out_dir=out, split_hash=split,
                  run_stamp=stamp, loader_reseed_per_epoch=policy)
    if kind == "local":
        return train_cell.run_local_cell(client_data_yamls={0: "a.yaml", 1: "b.yaml"},
                                         client_num_examples={0: 1, 1: 1}, **common)
    return train_cell.run_central_cell(data_yaml="a.yaml", num_examples=1, **common)


def _tags(kind: str) -> list[str]:
    return ["sep_local_c0", "sep_local_c1"] if kind == "local" else ["sep_central"]


@pytest.mark.parametrize("kind", ["local", "central"])
def test_칸은_빈_out_dir_에서_돌고_meta_에_identity_를_남긴다(monkeypatch, tmp_path, kind):
    """(d)(e) 정책 off 면 run_id·재개 run_id 에 접미가 없다(뒤호환)."""
    train_cell, calls = _drive_cells(monkeypatch)
    _run(train_cell, kind, tmp_path)
    cell = "sep_local" if kind == "local" else "sep_central"
    for tag in _tags(kind):
        meta = json.loads((tmp_path / f"{tag}.meta.json").read_text(encoding="utf-8"))
        assert meta["identity"] == {"run_id": f"{cell}_s7_main_s1", "base_seed": 7,
                                    "split_hash": "h", "loader_reseed_per_epoch": False}
    assert all(kw["run_id"] == "main_s1" for kw in calls)
    rows = list(csv.DictReader((tmp_path / "atomic_log.csv").open(encoding="utf-8")))
    assert {r["run_id"] for r in rows} == {f"{cell}_s7_main_s1"}


@pytest.mark.parametrize("kind", ["local", "central"])
def test_칸은_같은_신원의_재실행을_통과시킨다(monkeypatch, tmp_path, kind):
    """(b)(e)"""
    train_cell, calls = _drive_cells(monkeypatch)
    _run(train_cell, kind, tmp_path)
    n = len(calls)
    _run(train_cell, kind, tmp_path)
    assert len(calls) == 2 * n


@pytest.mark.parametrize("kind", ["local", "central"])
@pytest.mark.parametrize("change", [dict(split="other"), dict(seed=8), dict(stamp="main_s9"), dict(policy=True)])
def test_칸은_신원이_다른_실행을_학습_전에_거부한다(monkeypatch, tmp_path, kind, change):
    """(a)(c)(e) — 원장이 있는 경우. 학습(train_round)은 한 번도 불리지 않는다."""
    train_cell, calls = _drive_cells(monkeypatch)
    _run(train_cell, kind, tmp_path)
    n = len(calls)
    snapshot = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(LedgerIdentityMismatch):
        _run(train_cell, kind, tmp_path, **change)
    assert len(calls) == n
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == snapshot


@pytest.mark.parametrize("kind", ["local", "central"])
@pytest.mark.parametrize("change", [dict(split="other"), dict(seed=8), dict(stamp="main_s9"), dict(policy=True)])
def test_칸은_원장이_없어도_meta_로_거부한다(monkeypatch, tmp_path, kind, change):
    """(e) 두 번째 방어선 — 원장이 지워진 뒤 가중치·meta 만 남은 경우."""
    train_cell, calls = _drive_cells(monkeypatch)
    _run(train_cell, kind, tmp_path)
    (tmp_path / "atomic_log.csv").unlink()
    n = len(calls)
    with pytest.raises(LedgerIdentityMismatch, match="다른 실행의 산출물 위에"):
        _run(train_cell, kind, tmp_path, **change)
    assert len(calls) == n


@pytest.mark.parametrize("kind", ["local", "central"])
def test_구판_meta_는_고치지_않고_읽으며_정책_변경만_잡는다(monkeypatch, tmp_path, kind):
    """뒤호환: 세 시드 본실험 meta(`identity`·`loader_reseed_per_epoch` 없음). 같은 정책(off)은
    통과, 정책 on 은 거부, 파생 시드가 다르면 거부."""
    train_cell, calls = _drive_cells(monkeypatch)
    _run(train_cell, kind, tmp_path)
    (tmp_path / "atomic_log.csv").unlink()
    for tag in _tags(kind):
        p = tmp_path / f"{tag}.meta.json"
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta.pop("identity")
        meta.pop("loader_reseed_per_epoch")
        p.write_text(json.dumps(meta), encoding="utf-8")
    legacy = {tag: (tmp_path / f"{tag}.meta.json").read_bytes() for tag in _tags(kind)}

    with pytest.raises(LedgerIdentityMismatch, match="loader_reseed_per_epoch"):
        _run(train_cell, kind, tmp_path, policy=True)
    assert {tag: (tmp_path / f"{tag}.meta.json").read_bytes() for tag in _tags(kind)} == legacy
    (tmp_path / "atomic_log.csv").unlink(missing_ok=True)
    with pytest.raises(LedgerIdentityMismatch, match="seed"):
        _run(train_cell, kind, tmp_path, seed=8)
    (tmp_path / "atomic_log.csv").unlink(missing_ok=True)
    n = len(calls)
    _run(train_cell, kind, tmp_path)          # 같은 정책·같은 시드 → 통과(재학습)
    assert len(calls) > n


@pytest.mark.parametrize("kind", ["local", "central"])
def test_칸은_정책_on_이면_run_id_와_재개_run_id_에_접미를_새긴다(monkeypatch, tmp_path, kind):
    train_cell, calls = _drive_cells(monkeypatch)
    _run(train_cell, kind, tmp_path, policy=True)
    cell = "sep_local" if kind == "local" else "sep_central"
    assert all(kw["run_id"] == "main_s1_rs1" and kw["loader_reseed_per_epoch"] is True for kw in calls)
    rows = list(csv.DictReader((tmp_path / "atomic_log.csv").open(encoding="utf-8")))
    assert {r["run_id"] for r in rows} == {f"{cell}_s7_main_s1_rs1"}
