"""상주 flwr 프로세스의 신원(`fl/flwr_procs.py`, 외부 검토 §50 요청 2) — 정확한 진입점 · 이 venv 의 소유로만 고르고, 불명확하면 죽이지 않는다.

실제 프로세스를 띄우는 시험은 **이 시험이 띄운 합성 프로세스만** 본다 — 판정은 그 pid 로 거르고, 내리는 것도 그 pid 뿐이다.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl.flwr_procs import classify, flwr_processes, stop_processes

SCRIPTS = Path("C:/venv/Scripts")


@pytest.mark.parametrize(("info", "want"), [
    ({"name": "flower-superlink.exe", "exe": "C:/venv/Scripts/flower-superlink.exe", "cmdline": ["flower-superlink"]}, "target"),
    ({"name": "python.exe", "exe": "C:/py/python.exe", "cmdline": ["python.exe", "C:/venv/Scripts/flwr-simulation.exe", "--app"]}, "target"),
    ({"name": "flwr.exe", "exe": "C:/venv/Scripts/flwr.exe", "cmdline": ["flwr", "run", "."]}, "target"),
    ({"name": "flower-superlink.exe", "exe": "C:/other/Scripts/flower-superlink.exe", "cmdline": []}, "unclear"),   # 다른 venv
    ({"name": "flower-superlink.exe", "exe": None, "cmdline": None}, "unclear"),                                      # 경로를 읽지 못했다
    ({"name": "python.exe", "exe": "C:/py/python.exe", "cmdline": ["python", "-m", "flwr.simulation"]}, "unclear"),
    ({"name": "python.exe", "exe": "C:/py/python.exe", "cmdline": ["python", "D:/x/flwr-simulation.py"]}, "unclear"),
    ({"name": "notepad.exe", "exe": "C:/Windows/notepad.exe", "cmdline": ["notepad.exe", "C:/notes/superlink.txt"]}, None),
    ({"name": "python.exe", "exe": "C:/py/python.exe", "cmdline": ["python", "C:/superlink/run.py"]}, None),       # 상위 폴더 이름
    ({"name": "python.exe", "exe": "C:/py/python.exe", "cmdline": ["python", "x.py", "C:/venv/Scripts/flwr-simulation.exe"]}, None),
    ({"name": "code.exe", "exe": "C:/apps/code.exe", "cmdline": ["code", "docs/flower-superlink.md"]}, None),      # 문서
])
def test_진입점과_소유로만_고른다(info, want):
    assert classify(info, SCRIPTS) == want


def _py(*args):
    return {"name": "python.exe", "exe": "C:/py/python.exe", "cmdline": ["C:/py/python.exe", *args]}


@pytest.mark.parametrize(("info", "want"), [
    (_py("-u", "C:/other/Scripts/flower-superlink-script.py"), "unclear"),          # 회신의 반례 — 옵션 뒤의 다른 자리
    (_py("-u", "C:/venv/Scripts/flower-superlink-script.py"), "target"),
    (_py("-X", "utf8", "-B", "C:/venv/Scripts/flwr-simulation.exe", "--app", "."), "target"),   # 값을 받는 옵션 · 붙여 쓴 플래그
    (_py("-Xutf8", "-W", "ignore", "C:/other/flwr.py"), "unclear"),
    (_py("-uB", "D:/x/flower-supernode.py"), "unclear"),                              # 묶은 플래그
    (_py("-I", "-m", "flwr.simulation"), "unclear"),
    (_py("-mflwr", "run"), "unclear"),                                                # 붙여 쓴 모듈
    (_py("--check-hash-based-pycs", "never", "C:/other/flwr.py"), "unclear"),
    (_py("--", "C:/other/flwr.py"), "unclear"),
    (_py("--unknown", "C:/other/flwr.py"), "unclear"),                                # 해석 못 한 꼴 — 내리지 않고 시작만 거부
    (_py("-u", "x.py", "C:/venv/Scripts/flwr-simulation.exe"), None),                 # 스크립트 뒤의 인자는 보지 않는다
    (_py("-W", "C:/venv/Scripts/flwr.exe", "x.py"), None),                            # 옵션 값은 스크립트가 아니다
    (_py("-c", "import time", "C:/venv/Scripts/flwr.exe"), None),                     # 코드 문자열 뒤는 인자다
    (_py("-u"), None),
    (_py("-m", "pytest", "C:/venv/Scripts/flwr.exe"), None),
])
def test_파이썬_옵션을_해석해_실제_스크립트를_본다(info, want):
    """`cmd[1]` 만 보면 `python -u <script>` 를 놓친다 — 옵션을 해석한 자리 하나만 본다(외부 검토 회신 `rehearsal_blockers` 의 2)."""
    assert classify(info, SCRIPTS) == want


@pytest.mark.parametrize(("rel", "cwd", "want"), [
    (".venv/Scripts/flower-superlink-script.py", "C:/other", "unclear"),   # 회신의 반례 — 다른 작업 디렉터리의 같은 상대 경로
    ("venv/Scripts/flower-superlink-script.py", "C:/", "target"),          # 대상의 작업 디렉터리로 풀면 이 venv 다
    ("venv/Scripts/flower-superlink-script.py", None, "unclear"),          # 작업 디렉터리를 읽지 못했다
    ("C:venv/Scripts/flwr.py", "C:/", "unclear"),                          # 드라이브만 붙은 상대 경로
    ("\\venv\\Scripts\\flwr.py", "C:/", "unclear"),                   # 루트만 붙은 상대 경로
])
def test_상대_스크립트_경로는_대상의_작업_디렉터리로_푼다(rel, cwd, want):
    """검사하는 쪽의 현재 디렉터리로 풀지 않는다(외부 검토 회신 `rehearsal_blockers2` 의 2)."""
    assert classify({**_py("-u", rel), "cwd": cwd}, SCRIPTS) == want


def test_실제_프로세스의_상대_경로는_검사하는_쪽의_현재_디렉터리로_풀지_않는다(tmp_path, monkeypatch):
    """검사하는 쪽은 X 에 서 있고 X 의 venv 가 이 실행의 것이다. Y 에서 같은 상대 경로로 띄운 것은 불명이고, X 에서 띄운 것만 대상이다."""
    import psutil

    rel = Path(".venv") / "Scripts" / "flower-superlink-script.py"
    roots = {}
    for k in ("X", "Y"):
        (tmp_path / k / rel).parent.mkdir(parents=True)
        (tmp_path / k / rel).write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
        roots[k] = tmp_path / k
    procs = {k: subprocess.Popen([sys.executable, "-u", str(rel)], cwd=str(roots[k]),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for k in roots}
    try:
        time.sleep(1.0)
        monkeypatch.chdir(roots["X"])
        pids = {k: {p.pid, *(c.pid for c in psutil.Process(p.pid).children(recursive=True))} for k, p in procs.items()}
        kinds = {p["pid"]: p["kind"] for p in flwr_processes(scripts_dir=roots["X"] / rel.parent)
                 if p["pid"] in pids["X"] | pids["Y"]}
        assert {kinds[q] for q in pids["Y"] if q in kinds} == {"unclear"}
        assert {kinds[q] for q in pids["X"] if q in kinds} == {"target"}
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
                p.wait(timeout=15)


def test_내리다_권한이_거부된_것은_나머지를_다_시도한_뒤_알린다(monkeypatch):
    import psutil

    from fl.flwr_procs import StopFailed

    killed = []

    class Fake:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            return []

        def kill(self):
            if self.pid == 1:
                raise psutil.AccessDenied(self.pid)
            killed.append(self.pid)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(psutil, "Process", Fake)
    found = [{"pid": 1, "name": "flower-superlink.exe", "kind": "target"},
             {"pid": 2, "name": "flwr-simulation.exe", "kind": "target"}]
    with pytest.raises(StopFailed) as exc:
        stop_processes(found)
    assert killed == [2] and exc.value.failed == [(1, "AccessDenied")]


def _spawn(argv):
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_실제_프로세스에서_대상만_내리고_불명확하면_아무것도_내리지_않는다(tmp_path):
    sleeper = "import time\ntime.sleep(120)\n"
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    target_script = scripts / "flower-superlink"                     # 이 실행의 venv 를 흉내 낸 폴더의 진입점
    target_script.write_text(sleeper, encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "flower-superlink").write_text(sleeper, encoding="utf-8")      # 이름은 같고 다른 폴더 — 불명
    folder = tmp_path / "superlink"
    folder.mkdir()
    (folder / "run.py").write_text(sleeper, encoding="utf-8")               # 상위 폴더 이름에 표지 — 무관
    procs = {
        "target": _spawn([sys.executable, str(target_script)]),
        "unclear": _spawn([sys.executable, str(other / "flower-superlink")]),
        "folder": _spawn([sys.executable, str(folder / "run.py")]),
        "arg": _spawn([sys.executable, "-c", "import time; time.sleep(120)", "C:/notes/superlink.txt"]),
    }
    try:
        time.sleep(1.0)
        import psutil

        mine = {p.pid for p in procs.values()}
        mine |= {c.pid for p in procs.values() for c in psutil.Process(p.pid).children(recursive=True)}

        def ours():
            return [x for x in flwr_processes(scripts_dir=scripts) if x["pid"] in mine]

        found = ours()
        kinds = {p["pid"]: p["kind"] for p in found}
        tpids = {procs["target"].pid, *(c.pid for c in psutil.Process(procs["target"].pid).children(recursive=True))}
        upids = {procs["unclear"].pid, *(c.pid for c in psutil.Process(procs["unclear"].pid).children(recursive=True))}
        assert {k for k, v in kinds.items() if v == "target"} <= tpids and tpids & set(kinds)
        assert {k for k, v in kinds.items() if v == "unclear"} <= upids and upids & set(kinds)
        assert not ({procs["folder"].pid, procs["arg"].pid} & set(kinds))         # 무관한 둘은 고르지 않는다
        with pytest.raises(ValueError):                                           # 불명이 섞이면 아무것도 내리지 않는다
            stop_processes(found)
        assert all(p.poll() is None for p in procs.values())
        stop_processes([p for p in found if p["kind"] == "target"])
        procs["target"].wait(timeout=15)
        assert procs["target"].poll() is not None
        assert all(procs[k].poll() is None for k in ("unclear", "folder", "arg"))
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
                p.wait(timeout=15)


# ================================================================ 검출 실행의 cp949 재시도 — 우회 경로가 남지 않는다
def _flwr_fake(calls, *, first_rc=1):
    def run(cmd, **kw):
        calls.append(cmd)
        if kw.get("stdout") is not None and len(calls) == 1:
            kw["stdout"].write("UnicodeDecodeError: 'cp949' codec can't decode ... install_from_fab\n")
        return subprocess.CompletedProcess(cmd, first_rc if len(calls) == 1 else 0)
    return run


def test_검출의_cp949_재시도는_정확한_신원으로만_내린다(tmp_path, monkeypatch):
    import fl.flwr_procs as FP
    import scripts.main_det as MD

    calls, stopped = [], []
    monkeypatch.setattr(MD.subprocess, "run", _flwr_fake(calls))
    monkeypatch.setattr(MD.time, "sleep", lambda s: None)
    monkeypatch.setattr(FP, "flwr_processes", lambda **k: [{"pid": 1, "name": "flower-superlink.exe", "kind": "target"}])
    monkeypatch.setattr(FP, "stop_processes", lambda found: stopped.append(found))
    MD._flwr_run("cell=\"x\"", tmp_path / "flwr.log")
    assert len(calls) == 2 and stopped == [[{"pid": 1, "name": "flower-superlink.exe", "kind": "target"}]]
    assert not any("powershell" in str(c).lower() for c in calls)            # 명령줄 부분 문자열로 내리던 길이 없다


def test_검출의_cp949_재시도는_신원이_불명확하면_내리지_않고_멈춘다(tmp_path, monkeypatch):
    import fl.flwr_procs as FP
    import scripts.main_det as MD

    calls, stopped = [], []
    monkeypatch.setattr(MD.subprocess, "run", _flwr_fake(calls))
    monkeypatch.setattr(MD.time, "sleep", lambda s: None)
    monkeypatch.setattr(FP, "flwr_processes", lambda **k: [{"pid": 2, "name": "flower-superlink.exe", "kind": "unclear"}])
    monkeypatch.setattr(FP, "stop_processes", lambda found: stopped.append(found))
    with pytest.raises(SystemExit, match="신원이 불명확"):
        MD._flwr_run("cell=\"x\"", tmp_path / "flwr.log")
    assert len(calls) == 1 and stopped == []
