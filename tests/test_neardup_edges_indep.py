"""간선 판본의 독립 구현(`scripts/neardup/nd_stage16b_edges_v1_indep.py`)의 출력 경로 보호 — 외부 검토 §49.

새 폴더만 쓰고 동결 경로에는 쓰지 않는다. 검사는 디렉터리를 만들고 입력을 읽기 **전에** 돈다 — 그래서 이 시험은 실물
입력 없이 돈다(없는 입력 경로를 줘도 출력 검사에서 먼저 멈춘다).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "neardup" / "nd_stage16b_edges_v1_indep.py"


def _run(out: Path, tmp: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), str(tmp / "no_inputs"), str(tmp / "no.csv"),
                           str(out)], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=REPO,
                          check=False)


def test_이미_있는_출력_폴더는_덮어쓰지_않는다(tmp_path):
    out = tmp_path / "prev_run"
    out.mkdir()
    (out / "edges.csv").write_text("a_id,b_id\nx,y\n", encoding="utf-8")
    r = _run(out, tmp_path)
    assert r.returncode != 0 and "이미 있다" in r.stderr
    assert (out / "edges.csv").read_text(encoding="utf-8") == "a_id,b_id\nx,y\n"
    assert sorted(p.name for p in out.iterdir()) == ["edges.csv"]


def test_동결_명부의_자리_아래에는_쓰지_않고_폴더도_만들지_않는다(tmp_path):
    from data.frozen_guard import EXPECTED_SEALED
    rel = "data/interim/neardup_edges_v1"
    assert rel in EXPECTED_SEALED
    out = REPO / rel / "rerun_should_not_exist"
    r = _run(out, tmp_path)
    assert r.returncode != 0 and "동결 명부" in r.stderr
    assert not out.exists()


def test_계약서가_있는_조상_폴더_아래에는_쓰지_않는다(tmp_path):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "SNAPSHOT.sha256").write_text("x  edges.csv\n", encoding="utf-8")
    out = sealed / "sub" / "run"
    r = _run(out, tmp_path)
    assert r.returncode != 0 and "FrozenDirectoryError" in r.stderr
    assert not (sealed / "sub").exists()


def test_입력은_출력_검사_뒤에_읽는다(tmp_path):
    """없는 새 출력 폴더면 출력 검사를 지나 입력에서 멈춘다 — 그 전에 폴더를 만들고 입력을 읽는다는 뜻이다."""
    out = tmp_path / "fresh"
    r = _run(out, tmp_path)
    assert r.returncode != 0 and "stage12_frozen_rule.json" in r.stderr
    assert out.is_dir() and not any(out.iterdir())
