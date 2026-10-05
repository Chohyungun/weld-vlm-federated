"""연합 단계 뒤 살아 있는 부모 쪽 정리(`scripts/rehearsal_uni.py::with_fed_cleanup`, 외부 검토 §50 요청 1).

학습 자식 안의 정리(`finally`)는 시간 제한의 트리 끊기 뒤에는 돌지 않는다. 그 경우와 **별도로 떠 있는 상주 프로세스**를 실제 프로세스로 나눠 본다 —
대상은 이 시험이 띄운 합성 프로세스뿐이다(진입점 이름의 스크립트를 임시 `Scripts` 폴더에 두고 그 폴더를 이 실행의 venv 로 준다).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl.flwr_procs import flwr_processes
from scripts.rehearsal_uni import Halt, with_fed_cleanup
from vlm.rehearsal_orch import Stage, run_stage

SLEEPER = "import time\ntime.sleep(120)\n"


@pytest.fixture
def resident(tmp_path):
    """상주 SuperLink 를 흉내 낸 합성 프로세스 — 연합 단계(학습 자식)의 프로세스 트리 **밖**에서 따로 띄운다."""
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    entry = scripts / "flower-superlink"
    entry.write_text(SLEEPER, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(entry)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    import psutil

    mine = {proc.pid, *(c.pid for c in psutil.Process(proc.pid).children(recursive=True))}

    def procs():
        return [p for p in flwr_processes(scripts_dir=scripts) if p["pid"] in mine]

    yield proc, procs
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=15)


def _stage_killed_by_timeout(tmp_path):
    """학습 자식을 시간 제한으로 끊는다 — 자식의 `finally` 는 돌지 않는다(표지 파일이 남지 않는다)."""
    child = tmp_path / "child.py"
    marks = tmp_path / "marks"
    marks.mkdir()
    child.write_text(f"""import time
from pathlib import Path
Path(r"{marks / 'started'}").write_text("1")
try:
    time.sleep(120)
finally:
    Path(r"{marks / 'finally_ran'}").write_text("1")
""", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()

    def step():
        res = run_stage(Stage("B", "train_uni_fed", "train", (), 0, "train", tag="uni_fed"), root=root, seq=1,
                        launcher=lambda entry, argv: [sys.executable, str(child)], env_base={}, timeout_s=4)
        if not res.ok:
            raise Halt("stage_mismatch", f"단계 B train_uni_fed 이 기대와 다르다 — 종료 {res.exit_code}", res.row)
        return res
    return step, marks


def test_시간_제한으로_끊긴_연합_단계_뒤에도_부모가_상주_프로세스를_내린다(tmp_path, resident):
    proc, procs = resident
    step, marks = _stage_killed_by_timeout(tmp_path)
    rows = []
    with pytest.raises(Halt) as exc:
        with_fed_cleanup(step, log=rows.append, procs=procs)
    assert exc.value.code == "stage_mismatch"                                  # 원래 단계의 실패가 그대로 올라온다
    assert (marks / "started").exists() and not (marks / "finally_ran").exists()   # 자식 안의 정리는 돌지 않았다
    proc.wait(timeout=15)
    assert proc.poll() is not None                                             # 부모가 내렸다
    (row,) = rows
    assert row["kind"] == "fed_cleanup" and row["ok"] is True and row["stage_failed"] is True
    assert proc.pid in {p["pid"] for p in row["stopped"]} and row["left"] == []


def test_단계가_섰어도_떠_있는_상주_프로세스를_부모가_내린다(resident):
    proc, procs = resident
    rows = []
    assert with_fed_cleanup(lambda: "trained", log=rows.append, procs=procs) == "trained"
    proc.wait(timeout=15)
    assert proc.poll() is not None and rows[0]["ok"] is True and rows[0]["stage_failed"] is False


def test_정리가_서지_않으면_단계의_실패와_나눠_적는다():
    unclear = [{"pid": 7, "name": "flower-superlink.exe", "kind": "unclear"}]
    stopped = []
    rows = []
    with pytest.raises(Halt) as exc:                                           # 단계는 섰는데 정리가 실패 — fed_cleanup
        with_fed_cleanup(lambda: "trained", log=rows.append, procs=lambda: unclear, stop=stopped.append)
    assert exc.value.code == "fed_cleanup" and stopped == []                   # 불명은 내리지 않는다
    assert rows[-1]["ok"] is False and rows[-1]["stage_failed"] is False and rows[-1]["unclear"]

    def boom():
        raise Halt("stage_mismatch", "끊겼다")
    with pytest.raises(Halt) as exc:                                           # 단계도 실패 — 원래 실패를 올리고 정리 실패는 줄에 따로
        with_fed_cleanup(boom, log=rows.append, procs=lambda: unclear, stop=stopped.append)
    assert exc.value.code == "stage_mismatch"
    assert rows[-1]["ok"] is False and rows[-1]["stage_failed"] is True and "끊겼다" in rows[-1]["stage_failure"]


def _denied(found):
    import psutil

    raise psutil.AccessDenied(found[0]["pid"])


def test_내리다_권한이_거부돼도_원래_단계의_실패를_덮지_않고_정리_실패로_적는다():
    """`stop()` 이 `AccessDenied` 를 내도 잔류를 다시 보고, 원래 단계의 실패를 그대로 올린다(외부 검토 회신 `rehearsal_blockers` 의 1)."""
    left = [{"pid": 7, "name": "flower-superlink.exe", "kind": "target"}]
    seen = {"n": 0}

    def procs():
        seen["n"] += 1
        return left

    def boom():
        raise Halt("stage_mismatch", "끊겼다")

    rows = []
    with pytest.raises(Halt) as exc:
        with_fed_cleanup(boom, log=rows.append, procs=procs, stop=_denied)
    assert exc.value.code == "stage_mismatch"                                  # 정리 예외가 원래 실패를 갈아치우지 않는다
    (row,) = rows
    assert row["ok"] is False and row["stage_failed"] is True and "끊겼다" in row["stage_failure"]
    assert any("AccessDenied" in e for e in row["errors"]) and row["stopped"] == []
    assert seen["n"] == 2 and row["left"] == [{"pid": 7, "name": "flower-superlink.exe", "kind": "target"}]   # 잔류를 다시 봤다

    rows.clear()
    with pytest.raises(Halt) as exc:                                           # 단계는 섰다 — 정리 실패로 멈춘다
        with_fed_cleanup(lambda: "trained", log=rows.append, procs=procs, stop=_denied)
    assert exc.value.code == "fed_cleanup" and rows[0]["ok"] is False and rows[0]["stage_failed"] is False


def test_찾기가_예외를_내도_할_수_있는_만큼_보고_원래_실패를_올린다():
    calls = {"n": 0}

    def procs():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("process_iter 실패")
        return []

    def boom():
        raise Halt("stage_mismatch", "끊겼다")

    rows = []
    with pytest.raises(Halt) as exc:
        with_fed_cleanup(boom, log=rows.append, procs=procs, stop=_denied)
    assert exc.value.code == "stage_mismatch" and calls["n"] == 2              # 첫 찾기가 실패해도 다시 찾는다
    assert rows[0]["ok"] is False and rows[0]["left"] == [] and any("process_iter" in e for e in rows[0]["errors"])

    def always():
        raise OSError("못 읽는다")

    rows.clear()
    with pytest.raises(Halt) as exc:                                           # 잔류를 끝내 모르면 정리가 섰다고 하지 않는다
        with_fed_cleanup(lambda: "trained", log=rows.append, procs=always, stop=_denied)
    assert exc.value.code == "fed_cleanup" and rows[0]["left"] is None and len(rows[0]["errors"]) == 2
