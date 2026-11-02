"""연합 실행 파일 찾기(`scripts/main_det.py::_flwr_exe` · `fl.flwr_procs.venv_scripts`) — 서버 선제 발견(2026-10-17).

Linux venv 의 `bin/python` 은 base 인터프리터로 가는 심볼릭 링크라, 앞 판의 `Path(sys.executable).resolve().parent` 가 venv 밖을 보아
`venv/bin/flwr` 를 못 찾았다. 이제 `sys.prefix` 의 실행 파일 폴더를 먼저 보고, 인터프리터 경로의 링크를 풀지 않으며, `PATH` 의 flwr 는
`sys.prefix` 아래의 것만 받는다. 셋 다 없으면 `[flwr_missing]` 으로 멈춘다. 실제 flwr 를 띄우지 않는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fl.flwr_procs as FP
import scripts.main_det as MD

BIN = "Scripts" if os.name == "nt" else "bin"
EXE = "flwr.exe" if os.name == "nt" else "flwr"
PY = "python.exe" if os.name == "nt" else "python"


def _venv(tmp: Path, *, with_flwr: bool = True) -> tuple[Path, Path]:
    """(venv 뿌리, base 인터프리터) — venv 의 실행 파일 폴더에 flwr 가 있고, base 는 venv 밖이다."""
    venv, base = tmp / "venv", tmp / "base"
    (venv / BIN).mkdir(parents=True)
    base.mkdir()
    (base / PY).write_bytes(b"")
    if with_flwr:
        (venv / BIN / EXE).write_bytes(b"")
    return venv, base / PY


def test_인터프리터의_실제_경로가_venv_밖이어도_sys_prefix_의_flwr_를_찾는다(tmp_path, monkeypatch):
    """링크를 푼 인터프리터 경로(= base)를 그대로 준 모사 — 앞 판은 base 폴더에서 찾아 실패했다."""
    venv, base_py = _venv(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "executable", str(base_py))
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert MD._flwr_exe() == venv / BIN / EXE
    assert FP.venv_scripts() == venv / BIN
    assert not (base_py.parent / EXE).exists()                          # 앞 판이 보던 자리에는 없다


def test_심볼릭_링크_venv_의_python_을_풀지_않는다(tmp_path, monkeypatch):
    venv, base_py = _venv(tmp_path)
    link = venv / BIN / PY
    try:
        os.symlink(base_py, link)
    except (OSError, NotImplementedError):
        pytest.skip("이 환경은 심볼릭 링크를 만들 권한이 없다 — 위 모사가 같은 경우를 본다")
    assert link.resolve().parent == base_py.parent                       # 링크를 풀면 venv 밖이다(서버의 꼴)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "executable", str(link))
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert MD._flwr_exe() == venv / BIN / EXE


def test_어디에도_없으면_사유_코드로_멈추고_다른_환경의_flwr_는_쓰지_않는다(tmp_path, monkeypatch):
    venv, base_py = _venv(tmp_path, with_flwr=False)
    other = tmp_path / "other_env" / BIN / EXE
    other.parent.mkdir(parents=True)
    other.write_bytes(b"")
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "executable", str(base_py))
    monkeypatch.setattr("shutil.which", lambda name: str(other))          # PATH 에는 다른 환경의 flwr 가 있다
    with pytest.raises(SystemExit, match=r"^\[flwr_missing\]"):
        MD._flwr_exe()


def test_PATH_의_flwr_는_sys_prefix_아래면_받는다(tmp_path, monkeypatch):
    venv, base_py = _venv(tmp_path, with_flwr=False)
    inside = venv / "tools" / EXE
    inside.parent.mkdir()
    inside.write_bytes(b"")
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "executable", str(base_py))
    monkeypatch.setattr("shutil.which", lambda name: str(inside))
    assert MD._flwr_exe() == inside


def test_venv_실행_파일_폴더가_없으면_인터프리터_경로의_폴더를_링크를_풀지_않고_쓴다(tmp_path, monkeypatch):
    py = tmp_path / "x" / PY
    py.parent.mkdir()
    py.write_bytes(b"")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "no_such_prefix"))
    monkeypatch.setattr(sys, "executable", str(py))
    assert FP.venv_scripts() == py.parent


def test_flwr_run_은_찾은_실행_파일로_띄우고_그_폴더를_PATH_앞에_붙인다(tmp_path, monkeypatch):
    venv, base_py = _venv(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "executable", str(base_py))
    monkeypatch.setattr("shutil.which", lambda name: None)
    seen = {}

    class Done:
        returncode = 0

    def run(cmd, **kw):
        seen["cmd"], seen["path"] = cmd, kw["env"]["PATH"]
        return Done()

    monkeypatch.setattr(MD.subprocess, "run", run)
    MD._flwr_run('cell="x"', tmp_path / "flwr.log")
    assert seen["cmd"][0] == str(venv / BIN / EXE) and seen["cmd"][1:4] == ["run", ".", "local-sim"]
    assert seen["path"].split(os.pathsep)[0] == str(venv / BIN)
