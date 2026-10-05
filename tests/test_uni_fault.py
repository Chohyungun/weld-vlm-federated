"""고의 중단 장치와 가드의 순서(리허설 2판 §1-3 · §1-4 · §6-1 의 13 · 27~33 · 48) — 학습 지점과 연합의 거부.

export 지점과 오케스트레이터(계획 기록을 쓰고 자식 환경을 지우는 쪽)는 이 구현의 범위 밖이다 — 오케스트레이터 쪽 도우미
(`write_planned` · `scrubbed_env`)만 여기서 본다. **죽는 시험은 자식 프로세스로 돈다** — `os._exit` 는 시험 프로세스를 함께 죽인다.
자식은 CPU 만 보게(`CUDA_VISIBLE_DEVICES=""`) 띄운다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_uni_train_cell import PLAN, _init, _meta, _rows, _spec, reh  # noqa: E402
from tests.uni_fakes import CountingLoader  # noqa: E402
from vlm import fault as F  # noqa: E402
from vlm.train_cell import LOCAL_TAGS, CellRejected, cell_dir, close_ledger, run_uni_local_cell  # noqa: E402

NONCE = "0123456789abcdef0123456789abcdef"
FAULT = "train:after_ckpt:uni_local_C1:ep=0"


@pytest.fixture(autouse=True)
def _no_cuda_no_env(monkeypatch):
    import torch

    was = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for k in list(os.environ):
        if k.upper() in F.FAULT_VARS:
            monkeypatch.delenv(k)
    yield
    assert was or not torch.cuda.is_initialized(), "이 시험이 CUDA 를 초기화했다"


# ================================================================ 값의 꼴
@pytest.mark.parametrize("raw, code", [
    ("train:after_ckpt:uni_local_C1", "fault_form"),
    ("train:after_save:uni_local_C1:ep=0", "fault_point"),
    ("train:after_ckpt:uni_local_c1:ep=0", "fault_tag"),
    ("train:after_ckpt:uni_local_C1:n=0", "fault_arg"),
    ("export:after_all_lines:uni_central:n=1", "fault_arg"),
])
def test_모르는_지점_태그_인자는_거부한다(raw, code):
    with pytest.raises(F.FaultRefused) as exc:
        F.parse_fault(raw)
    assert exc.value.code == code


def test_파일_이름의_열쇠에_콜론이_없다():
    s = F.parse_fault("export:torn_line:uni_central:n=21")
    assert s.key == "export-torn_line-uni_central-n-21" and ":" not in s.key
    assert F.parse_fault("export:after_all_lines:uni_local_C3:-").k is None


# ================================================================ 순서 2 — 변수 · 표식 · 계획 기록
def _planned(root: Path, fault: str = FAULT, nonce: str = NONCE) -> Path:
    return F.write_planned(root, nonce, fault, stage_no=3)


@pytest.mark.parametrize("purpose", ["main", "frame_diag"])
def test_본실험과_진단은_변수가_하나라도_보이면_거부한다(purpose, tmp_path):
    for env in ({"weld_rehearsal_fault": FAULT}, {"WELD_REHEARSAL_NONCE": NONCE}):     # 소문자 이름으로 심어도
        with pytest.raises(F.FaultRefused) as exc:
            F.check_fault_env(purpose, run_root=tmp_path, stage="train", tags=["uni_local_C1"], env=env)
        assert exc.value.code == "fault_env_forbidden"
    assert F.check_fault_env(purpose, run_root=None, stage="train", tags=[], env={}) is None


@pytest.mark.parametrize("env_of, code", [
    (lambda: {F.FAULT_VAR: FAULT}, "fault_env_half"),
    (lambda: {F.NONCE_VAR: NONCE}, "fault_env_half"),
    (lambda: {F.FAULT_VAR: FAULT, F.NONCE_VAR: "xyz"}, "nonce_form"),
    (lambda: {F.FAULT_VAR: FAULT, F.NONCE_VAR: "f" * 32}, "planned_missing"),
    (lambda: {F.FAULT_VAR: "train:before_ckpt:uni_local_C1:ep=0", F.NONCE_VAR: NONCE}, "planned_mismatch"),
])
def test_리허설은_둘이_함께_계획_기록과_같을_때만_켜진다(tmp_path, env_of, code):
    root = reh(tmp_path)
    _planned(root)
    with pytest.raises(F.FaultRefused) as exc:
        F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_local_C1"], env=env_of())
    assert exc.value.code == code


def test_이_호출의_중단이_아니거나_범위_밖이면_거부한다(tmp_path):
    root = reh(tmp_path)
    _planned(root)
    env = {F.FAULT_VAR: FAULT, F.NONCE_VAR: NONCE}
    with pytest.raises(F.FaultRefused) as exc:
        F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_central"], env=env)
    assert exc.value.code == "fault_not_for_this_call"
    with pytest.raises(F.FaultRefused) as exc:
        F.check_fault_env("rehearsal", run_root=root, stage="export", tags=["uni_local_C1"], env=env)
    assert exc.value.code == "fault_not_for_this_call"
    far = "train:after_ckpt:uni_local_C1:ep=5"
    F.write_planned(root, "1" * 32, far, stage_no=3)
    with pytest.raises(F.FaultRefused) as exc:
        F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_local_C1"], bounds={"ep": 2},
                          env={F.FAULT_VAR: far, F.NONCE_VAR: "1" * 32})
    assert exc.value.code == "fault_range"
    armed = F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_local_C1"], bounds={"ep": 2},
                              env=env)
    assert armed is not None and armed.spec.raw == FAULT


def test_계획_기록은_배타_생성이다(tmp_path):
    _planned(reh(tmp_path))
    with pytest.raises(FileExistsError):
        _planned(reh(tmp_path))


def test_자식_환경은_지운_복사본에서_만든다():
    parent = {"PATH": "x", "weld_rehearsal_fault": "old", "Weld_Rehearsal_Nonce": "old"}
    kid = F.scrubbed_env(parent)
    assert F.fault_env(kid) == (None, None) and kid["PATH"] == "x"
    planned = F.scrubbed_env(parent, (FAULT, NONCE))
    assert F.fault_env(planned) == (FAULT, NONCE)
    assert sum(k.upper() == F.FAULT_VAR for k in planned) == 1   # 옛 소문자 값은 남지 않는다


# ================================================================ 켜는 순서
class _Exit(RuntimeError):
    pass


def _armed(tmp_path) -> F.ArmedFault:
    root = reh(tmp_path)
    _planned(root)
    a = F.check_fault_env("rehearsal", run_root=root, stage="train", tags=["uni_local_C1"],
                          env={F.FAULT_VAR: FAULT, F.NONCE_VAR: NONCE})

    def _exit(code):
        raise _Exit(code)

    a.exit_fn = _exit
    return a


def _ctx(tmp_path, purpose="rehearsal"):
    return F.FaultContext(purpose=purpose, run_root=reh(tmp_path))


def test_지점에_닿으면_활성화_발화_기록을_남기고_75_로_끝난다(tmp_path):
    a = _armed(tmp_path)
    a.fire(_ctx(tmp_path), point="after_ckpt", tag="uni_local_C1", k=1)       # 다른 인자 — 켜지지 않는다
    assert not list((reh(tmp_path) / "faults").glob("*.activated.json"))
    with pytest.raises(_Exit) as exc:
        a.fire(_ctx(tmp_path), point="after_ckpt", tag="uni_local_C1", k=0, run_id="r")
    assert exc.value.args == (75,)
    d = reh(tmp_path) / "faults"
    act, = d.glob("*.activated.json")
    fired, = d.glob("*.fired.json")
    assert act.name == "001_train-after_ckpt-uni_local_C1-ep-0.activated.json" and ":" not in act.name
    rec = json.loads(act.read_text(encoding="utf-8"))
    assert rec["fault"] == FAULT and rec["nonce"] == NONCE and len(rec["planned_sha256"]) == 64
    assert json.loads(fired.read_text(encoding="utf-8"))["activated"] == act.name
    assert a.already_activated()


@pytest.mark.parametrize("ctx_of, code", [(lambda t: None, "hook_no_context"),
                                          (lambda t: _ctx(t, "main"), "hook_context"),
                                          (lambda t: F.FaultContext("rehearsal", t / "reh-20261001T999999"),
                                           "hook_context")])
def test_훅은_문맥의_목적과_루트를_다시_보고_어긋나면_켜지지_않는다(tmp_path, ctx_of, code):
    a = _armed(tmp_path)
    with pytest.raises(F.FaultRefused) as exc:
        a.fire(ctx_of(tmp_path), point="after_ckpt", tag="uni_local_C1", k=0)
    assert exc.value.code == code
    assert not list((reh(tmp_path) / "faults").glob("*.activated.json"))


def test_활성화_기록을_쓰지_못하면_아무것도_바꾸지_않고_75_가_아니다(tmp_path, monkeypatch):
    a = _armed(tmp_path)
    real = F._fsync_write

    def fail(path, data, *, exclusive=True):
        if str(path).endswith(".activated.tmp"):
            raise OSError("디스크 가득")
        return real(path, data, exclusive=exclusive)

    monkeypatch.setattr(F, "_fsync_write", fail)
    effects: list[int] = []
    with pytest.raises(OSError):
        a.fire(_ctx(tmp_path), point="after_ckpt", tag="uni_local_C1", k=0, effect=lambda: effects.append(1))
    assert effects == [] and not list((reh(tmp_path) / "faults").glob("*.json"))


def test_효과_직후에_죽으면_활성화_기록만_남는다(tmp_path):
    a = _armed(tmp_path)

    def die():
        raise _Exit("효과 직후")

    with pytest.raises(_Exit):
        a.fire(_ctx(tmp_path), point="after_ckpt", tag="uni_local_C1", k=0, effect=die)
    d = reh(tmp_path) / "faults"
    assert len(list(d.glob("*.activated.json"))) == 1 and not list(d.glob("*.fired.json"))
    ev, = F.fault_events(reh(tmp_path), stage="train", tags=["uni_local_C1"])
    assert ev["fired"] is None and ev["activated_sha256"]


# ================================================================ 칸과 가드의 순서
def test_중단_변수는_다른_어떤_검사보다_먼저_본다(tmp_path, monkeypatch):
    """본실험 설정이 온통 틀려도 사유는 중단 변수다 — 가드의 순서 2 가 1 다음이다(§1-4)."""
    monkeypatch.setenv(F.FAULT_VAR, FAULT)      # 이름의 대소문자는 사전을 넘기는 시험이 본다(Windows 는 대문자로 저장한다)
    spec = _spec(tmp_path, purpose="main", plan_sha256=PLAN, run_root=None)
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=spec, model_loader=CountingLoader())
    assert exc.value.code == "fault_env_forbidden"


def test_리허설_루트_밖에서는_시작하지_않는다(tmp_path):
    loader = CountingLoader()
    for bad in (dict(run_root=tmp_path / "reh-x"), dict(run_root=tmp_path / "reh-20261001T000000" / "sub"),
                dict(train_root=tmp_path / "elsewhere"), dict(run_root=tmp_path / "fdiag-20261001T000000")):
        with pytest.raises(CellRejected) as exc:
            run_uni_local_cell(["C1"], spec=_spec(tmp_path, **bad), model_loader=loader,
                               rows_by_client=_rows(tmp_path))
        assert exc.value.code == "run_root", bad
    assert loader.n == 0


def test_원장_닫기는_중단을_받지_않는다(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    _planned(reh(tmp_path))
    monkeypatch.setenv(F.FAULT_VAR, FAULT)
    monkeypatch.setenv(F.NONCE_VAR, NONCE)
    with pytest.raises(CellRejected) as exc:
        close_ledger(spec, LOCAL_TAGS["C1"], rows=[])
    assert exc.value.code == "fault_not_for_this_call"        # 산출물이 없어도 가드가 먼저다(검수 14번 M-7 나)
    d = cell_dir(spec, LOCAL_TAGS["C1"])
    d.mkdir(parents=True)
    for n in ("adapter_last.npz", "adapter_last.meta.json", "train_ledger.csv"):
        (d / n).write_bytes(b"x")
    with pytest.raises(CellRejected) as exc:
        close_ledger(spec, LOCAL_TAGS["C1"], rows=[])
    assert exc.value.code == "fault_not_for_this_call"


def test_앞_호출의_활성화_기록이_있으면_다시_켜지지_않고_끝까지_돈다(tmp_path, monkeypatch):
    """"한 번만" 은 활성화 기록으로 가른다 — 표식이 다른 앞 호출의 기록이어도 같은 열쇠면 켜지지 않는다.

    `os._exit` 를 예외로 바꿔 끼운다 — "한 번만" 검사가 사라지면 이 시험의 pytest 프로세스가 죽는 대신 이 시험이 떨어진다."""
    class _Exited(BaseException):
        pass

    def no_exit(code):
        raise _Exited(code)

    monkeypatch.setattr(F.os, "_exit", no_exit)
    spec = _spec(tmp_path)
    loader = CountingLoader()
    _init(spec, loader)
    d = reh(tmp_path) / "faults"
    d.mkdir(parents=True)
    (d / "001_train-after_ckpt-uni_local_C1-ep-0.activated.json").write_text(
        json.dumps({"stage": "train", "tag": "uni_local_C1", "fault": FAULT, "nonce": "e" * 32, "time": 1.0}),
        encoding="utf-8")
    _planned(reh(tmp_path))
    monkeypatch.setenv(F.FAULT_VAR, FAULT)
    monkeypatch.setenv(F.NONCE_VAR, NONCE)
    res = run_uni_local_cell(["C1"], spec=spec, model_loader=loader, rows_by_client=_rows(tmp_path))
    assert res["C1"].status == "trained"
    assert len(list(d.glob("*.activated.json"))) == 1                          # 새 활성화 기록이 없다


def test_연합의_서버와_클라이언트는_목적과_무관하게_변수를_거부한다(tmp_path, monkeypatch):
    import vlm.pilot_vlm as pv
    from fl import client_vlm
    from fl.uni_fed import UniFedRejected, UniFedRun

    monkeypatch.setenv(F.NONCE_VAR, NONCE)
    get = {"purpose": "rehearsal", "plan": str(tmp_path / "p"), "rehearsal-root": str(reh(tmp_path))}.get
    with pytest.raises(UniFedRejected) as exc:
        UniFedRun(get=lambda k, d: get(k, d), out_dir=reh(tmp_path) / "fl" / "uni_fed", run_id="x", base_seed=7,
                  split_hash="s", num_rounds=1, local_epochs=1, total_epochs=1, client_tags=["C1", "C2", "C3"],
                  non_main_parent=tmp_path)
    assert exc.value.code == "fault_env_forbidden"
    calls: list[int] = []
    monkeypatch.setattr(pv, "load_pairs", lambda *a, **k: calls.append(1))
    with pytest.raises(F.FaultRefused):
        client_vlm.run_client_round(adapter_in=None, canonical_keys=["k"], round_idx=0, client_idx=0,
                                    cfg={"client_tag": "C1", "local_epochs": 1, "num_rounds": 1, "base_seed": 1})
    assert calls == []


# ================================================================ 자식 프로세스 — 실제로 죽고 이어 간다
CHILD = r'''
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from tests.test_uni_train_cell import _spec
from tests.uni_fakes import CountingLoader, make_rows
from vlm.train_cell import run_uni_local_cell
tmp = Path(sys.argv[2])
rows = make_rows(tmp / "img", "C1", 3)
run_uni_local_cell(["C1"], spec=_spec(tmp), model_loader=CountingLoader(), rows_by_client={"C1": rows})
print("DONE")
'''


def test_자식이_after_ckpt_에서_75_로_죽고_다음_프로세스가_이어_간다(tmp_path):
    spec = _spec(tmp_path)
    loader = CountingLoader()
    _init(spec, loader)
    _planned(reh(tmp_path))
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    env = F.scrubbed_env(dict(os.environ), (FAULT, NONCE))
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-X", "utf8", str(script), str(ROOT), str(tmp_path)], env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == F.EXIT_CODE, r.stderr[-2000:]
    faults = reh(tmp_path) / "faults"
    assert len(list(faults.glob("*.activated.json"))) == 1 and len(list(faults.glob("*.fired.json"))) == 1
    ck = list((reh(tmp_path) / "resume").rglob("resume_ep*.pt"))
    assert ck, "체크포인트가 디스크에 있는 채로 죽어야 한다"
    # 다음 프로세스 — 변수 없이 같은 명령. 재개해 끝내고 meta 의 fault_events 가 복구 세그먼트에 잇는다.
    r2 = subprocess.run([sys.executable, "-X", "utf8", str(script), str(ROOT), str(tmp_path)],
                        env=F.scrubbed_env({**os.environ, "CUDA_VISIBLE_DEVICES": ""}),
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r2.returncode == 0 and "DONE" in r2.stdout, r2.stderr[-2000:]
    m = _meta(cell_dir(spec, LOCAL_TAGS["C1"]))
    ev, = m["fault_events"]
    s1, s2 = m["resume_segments"]
    assert ev["fired"] and ev["recovered_process_started_unix"] == s2["started_at"]
    assert (s1["end_epoch"], s2["start_epoch"]) == (0, 1)
