"""독립 구현 스크립트의 출력 경로 보호(`scripts.crosscheck.xc_out_guard`) — 새 출력만, `_workspace/` 아래만."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.crosscheck.xc_out_guard import guard_new_output, recheck_inside


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "_workspace").mkdir(parents=True)
    return r


def test_새_경로는_받는다(repo):
    got = guard_new_output(repo / "_workspace" / "run" / "out", repo=repo, inputs=[repo / "in.csv"])
    assert got.name == "out"


def test_이미_있는_출력은_멈춘다(repo):
    (repo / "_workspace" / "old").mkdir()
    with pytest.raises(SystemExit, match="이미 있다"):
        guard_new_output(repo / "_workspace" / "old", repo=repo)
    (repo / "_workspace" / "f.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="이미 있다"):
        guard_new_output(repo / "_workspace" / "f.json", repo=repo)


def test_workspace_밖은_멈춘다(repo, tmp_path):
    for p in (repo / "data" / "interim" / "x", tmp_path / "elsewhere", repo / "_workspace" / ".." / "x"):
        with pytest.raises(SystemExit, match="아래가 아니다"):
            guard_new_output(p, repo=repo)


def test_입력과_겹치면_멈춘다(repo):
    out = repo / "_workspace" / "cand"
    with pytest.raises(SystemExit, match="겹친다"):                 # 출력이 입력과 같은 자리
        guard_new_output(out, repo=repo, inputs=[out])
    with pytest.raises(SystemExit, match="겹친다"):                 # 입력이 출력 아래
        guard_new_output(out, repo=repo, inputs=[out / "a.csv"])
    (repo / "_workspace" / "inp").mkdir()
    with pytest.raises(SystemExit, match="겹친다"):                 # 출력이 입력 아래
        guard_new_output(repo / "_workspace" / "inp" / "new", repo=repo, inputs=[repo / "_workspace" / "inp"])


def test_동결_디렉터리_아래는_멈춘다(repo):
    sealed = repo / "_workspace" / "sealed"
    sealed.mkdir()
    (sealed / "SNAPSHOT.sha256").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="동결"):
        guard_new_output(sealed / "sub" / "out", repo=repo)


@pytest.mark.skipif(sys.platform != "win32", reason="정션은 Windows 에만 있다")
def test_정션을_풀어_밖으로_나가면_멈춘다(repo, tmp_path):
    """`_workspace` 안의 정션이 밖을 가리키면 그 아래는 받지 않는다. 붙인 정션은 시험이 링크만 뗀다."""
    target = tmp_path / "protected"
    target.mkdir()
    link = repo / "_workspace" / "link"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    try:
        with pytest.raises(SystemExit, match="아래가 아니다"):
            guard_new_output(link / "out", repo=repo)
    finally:
        os.rmdir(link)                                    # 링크만 뗀다 — 대상은 그대로다
    assert target.is_dir()


@pytest.mark.skipif(sys.platform != "win32", reason="정션은 Windows 에만 있다")
def test_workspace_가_링크면_멈춘다(tmp_path):
    r = tmp_path / "repo2"
    r.mkdir()
    target = tmp_path / "ws_target"
    target.mkdir()
    subprocess.run(["cmd", "/c", "mklink", "/J", str(r / "_workspace"), str(target)], check=True,
                   capture_output=True)
    try:
        with pytest.raises(SystemExit, match="링크"):
            guard_new_output(r / "_workspace" / "out", repo=r)
    finally:
        os.rmdir(r / "_workspace")
    assert target.is_dir()


def test_만든_뒤_다시_본다(repo, tmp_path):
    out = repo / "_workspace" / "made"
    out.mkdir()
    recheck_inside(out, repo=repo)
    with pytest.raises(SystemExit, match="밖이다"):
        recheck_inside(tmp_path, repo=repo)
