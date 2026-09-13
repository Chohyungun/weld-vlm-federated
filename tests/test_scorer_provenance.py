"""채점기 코드 해시 — 여러 시드가 같은 코드로 채점됐는지 확인할 수단 (27번 §12-1).

이 파일이 지키는 것.

1. **내용이 바뀌면 해시가 바뀐다.** 안 바뀌면 지표가 아무것도 안 재는 것이다.
2. **이름이 바뀌어도 해시가 바뀐다.** 내용만 해싱하면 파일을 옮겨 담아 같은 지문을 낼 수 있다.
3. **결정론적이다.** 같은 파일 집합에서 언제 돌려도 같은 값이다 — 아니면 대조가 성립하지 않는다.
4. **git 상태는 판정 근거가 아니다.** 더러운 트리에서 커밋 해시는 코드를 대표하지 않는다.
"""

from __future__ import annotations

from evaluation.provenance import (
    RULE,
    combined_digest,
    scorer_code_digest,
    scorer_source_files,
)


def _tree(root, files: dict[str, str]) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")


def _repo(tmp_path, body: str = "x = 1\n"):
    _tree(tmp_path, {
        "evaluation/__init__.py": "",
        "evaluation/score.py": body,
        "evaluation/metrics/detection.py": "y = 2\n",
        "scripts/probe/score_cells.py": "z = 3\n",
    })
    return tmp_path


# --------------------------------------------------------------------------------------
# 1·2 — 무엇이 해시를 바꾸는가
# --------------------------------------------------------------------------------------

def test_내용이_바뀌면_해시가_바뀐다(tmp_path):
    a = scorer_code_digest(_repo(tmp_path))["combined"]
    (tmp_path / "evaluation/score.py").write_text("x = 2\n", encoding="utf-8")
    assert scorer_code_digest(tmp_path)["combined"] != a


def test_한_글자_차이도_잡는다(tmp_path):
    a = scorer_code_digest(_repo(tmp_path, "x = 1\n"))["combined"]
    (tmp_path / "evaluation/score.py").write_text("x = 1 \n", encoding="utf-8")
    assert scorer_code_digest(tmp_path)["combined"] != a


def test_이름이_바뀌면_해시가_바뀐다(tmp_path):
    """내용만 해싱하면 파일을 옮겨 담아 같은 지문을 낼 수 있다 — 경로도 넣는 이유다."""
    a = scorer_code_digest(_repo(tmp_path))["combined"]
    (tmp_path / "evaluation/score.py").rename(tmp_path / "evaluation/scoring.py")
    assert scorer_code_digest(tmp_path)["combined"] != a


def test_파일이_늘면_해시가_바뀐다(tmp_path):
    a = scorer_code_digest(_repo(tmp_path))["combined"]
    (tmp_path / "evaluation/extra.py").write_text("w = 4\n", encoding="utf-8")
    d = scorer_code_digest(tmp_path)
    assert d["combined"] != a and d["n_files"] == 5


def test_채점과_무관한_파일은_안_본다(tmp_path):
    """`evaluation/` 와 진입점 밖은 규칙에 없다 — 그 사실을 rule 이 밝힌다."""
    a = scorer_code_digest(_repo(tmp_path))["combined"]
    _tree(tmp_path, {"detection/train_cell.py": "무관\n", "docs/메모.md": "무관\n"})
    assert scorer_code_digest(tmp_path)["combined"] == a
    assert "evaluation/**/*.py" in RULE


# --------------------------------------------------------------------------------------
# 3 — 결정론
# --------------------------------------------------------------------------------------

def test_같은_트리에서_언제나_같은_값(tmp_path):
    root = _repo(tmp_path)
    assert scorer_code_digest(root)["combined"] == scorer_code_digest(root)["combined"]


def test_파일_순서에_좌우되지_않는다(tmp_path):
    """`combined_digest` 는 정렬된 목록을 받기로 돼 있다 — 뒤섞어 넣어도 같아야 한다."""
    root = _repo(tmp_path)
    files = scorer_source_files(root)
    a, _ = combined_digest(files, root)
    b, _ = combined_digest(sorted(files, reverse=True), root)
    assert a != b, "정렬을 호출부가 보장한다 — 이 성질이 깨지면 scorer_source_files 를 거쳐야 한다"
    assert scorer_source_files(root) == files


def test_바이트코드_캐시는_제외한다(tmp_path):
    root = _repo(tmp_path)
    a = scorer_code_digest(root)["combined"]
    cache = root / "evaluation/__pycache__"
    cache.mkdir()
    (cache / "score.cpython-311.pyc").write_bytes(b"\x00\x01")
    (cache / "stale.py").write_text("캐시 안의 파이썬 파일\n", encoding="utf-8")
    assert scorer_code_digest(root)["combined"] == a


def test_진입점이_없어도_죽지_않는다(tmp_path):
    """트리 밖 파일은 없을 수 있다 — 목록에서 빠지고 그만큼 해시가 달라진다."""
    root = _repo(tmp_path)
    a = scorer_code_digest(root)
    (root / "scripts/probe/score_cells.py").unlink()
    b = scorer_code_digest(root)
    assert b["n_files"] == a["n_files"] - 1 and b["combined"] != a["combined"]


# --------------------------------------------------------------------------------------
# 4 — git 은 참고
# --------------------------------------------------------------------------------------

def test_git_정보는_판정_근거가_아니라고_적혀_있다(tmp_path):
    d = scorer_code_digest(_repo(tmp_path))
    assert "참고" in d["git"]["role"]
    assert set(d["git"]) >= {"head", "dirty", "branch"}


def test_소급_금지를_필드가_말한다(tmp_path):
    assert "소급 계산해" in scorer_code_digest(_repo(tmp_path))["note"]


def test_파일별_해시는_끌_수_있다(tmp_path):
    root = _repo(tmp_path)
    assert "files" in scorer_code_digest(root)
    d = scorer_code_digest(root, include_files=False)
    assert "files" not in d and d["combined"]


def test_실제_저장소에서_돈다():
    """기본 인자 — 이 저장소의 채점기 소스를 실제로 해싱한다."""
    d = scorer_code_digest()
    assert d["n_files"] > 20
    assert "evaluation/score.py" in d["files"]
    assert "scripts/probe/score_cells.py" in d["files"]
    assert len(d["combined"]) == 64
