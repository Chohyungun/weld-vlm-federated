"""채점기 자신의 코드 해시 — 여러 시드가 같은 코드로 채점됐는지 확인할 수단.

## 왜 필요한가

27번 §1-0 이 적은 구멍이다. 시드 1·2·3 산출물의 채점 파라미터가 전부 같다는 것은 확인되는데,
**같은 채점기 코드로 나왔는지는 확인되지 않았다** — 산출물 어디에도 코드의 지문이 없었다.
그래서 "세 시드가 한 기준" 은 보고서의 고정 문자열, 즉 작성자의 진술이었다. 고정 문자열은
입력 검증을 대신하지 못한다. 이 모듈이 그 자리를 채운다.

## 무엇을 해싱하나 — 그리고 왜 그 방식인가

**디렉터리 + 명시 목록 규칙으로 고정한다.** `evaluation/**/*.py` 전부 + 채점 값을 결정하는
트리 밖 모듈(`SCORER_FILES`: 진입점·레코드 생성기·대조선 생성기·클래스 사상·층화 절단점·
매니페스트 읽기·직렬화).

임포트된 모듈(`sys.modules`)을 훑는 방식을 **쓰지 않는다.** 이 채점기는 함수 안에서 지연
임포트를 하므로(`--cells det` 인지, 통합형이 있는지, P9 맥락이 있는지에 따라 다르다) 모듈
목록이 실행 경로마다 달라진다. 그러면 **같은 코드인데 해시가 갈리고**, 해시가 갈리는 것이
코드 차이를 뜻한다는 이 지표의 전제가 무너진다.

디렉터리 규칙은 반대 방향으로 보수적이다. 채점과 무관한 `evaluation/` 파일이 바뀌어도 해시가
바뀐다. 그것은 오탐이 아니라 **안전한 쪽의 오차**다 — 집계기가 "다르다" 고 멈추는 쪽이,
다른 코드를 같다고 통과시키는 쪽보다 낫다.

## 해시의 성질

- **내용 기반.** 파일 바이트를 읽는다. 경로·수정 시각·git 상태가 아니다.
- **결정론적.** 저장소 상대 경로를 정렬하고 `경로\\0파일해시\\n` 를 이어 붙여 다시 해싱한다.
  같은 파일 집합이면 언제 어디서 돌려도 같은 값이 나온다.
- **git 정보는 참고다.** 커밋 해시는 작업 트리가 더러우면 거짓말을 한다. `dirty` 를 함께
  남기되 판정 근거는 파일 해시 쪽이다.

## 소급하지 않는다

**이미 나온 산출물에 이 필드를 덧붙이지 않는다.** 나중에 계산해 끼워 넣으면 그 값은 "그때 그
코드의 지문" 이 아니라 "지금 코드의 지문" 이고, 그것을 당시 기록인 척 싣는 것이 사후 조작이다.
필드가 없는 산출물은 **없는 대로** 두고, 집계기가 "해시 없음 — 독립 확인 불가" 로 보고한다.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SCORER_TREES: tuple[str, ...] = ("evaluation",)
"""통째로 해싱하는 디렉터리. 하위 전부의 `.py` 를 넣는다."""

SCORER_FILES: tuple[str, ...] = (
    "scripts/probe/score_cells.py",             # 진입점
    "scripts/probe/adapt_main_detections.py",   # 채점 레코드 생성 — 레코드 자체의 해시는 산출물 `input_records` 에
    "scripts/probe/content_free_baselines.py",  # 무내용 대조선 생성 — 집계기 "동결본 불변" 판정의 근거
    "data/label_map.py",                        # 클래스 사상 (원본 라벨 ↔ ISO 6520-1)
    "data/id_strata.py",                        # id 구간 층화 절단점
    "data/manifest_io.py",                      # 매니페스트 읽기
    "detection/serialize.py",                   # 레코드 직렬화
)
"""`evaluation/` 밖에 있지만 채점 값을 결정하는 모듈. C 34번 Important 3 이 짚은 구멍이다 —
"같은 코드로 채점됐는가" 를 물으면서 클래스 사상·층화 절단점·레코드 생성기를 지문 밖에
두면 그 모듈이 바뀌어도 지문이 같다. 여기 없는 파일은 해시에 안 들어간다 — `rule` 이 밝힌다.
존재하지 않는 항목은 건너뛰고 `n_files` 가 줄어든다(같은 트리면 같은 수).

**데이터 파일은 넣지 않는다.** 스냅샷은 `verify_snapshot` 을 지나야 열리고 그 digest 가 게이트
결과에 실리며, 사상표가 바뀌면 무내용 대조선 값이 함께 움직여 집계기가 멈춘다(C 34번 Minor 5).
"""

RULE = (
    "evaluation/**/*.py 전부 + 채점 경로의 트리 밖 모듈 " + ", ".join(SCORER_FILES) + ". "
    "sys.modules 가 아니라 디렉터리·명시 목록 규칙으로 고정한다 — 지연 임포트 때문에 실행 "
    "경로마다 모듈 목록이 달라지면 같은 코드에서 다른 해시가 나온다"
)


def scorer_source_files(repo: Path | None = None) -> list[Path]:
    """해시 대상 파일 목록. 정렬된 저장소 상대 경로 순서다."""
    root = Path(repo or REPO)
    out: set[Path] = set()
    for tree in SCORER_TREES:
        out.update(
            p for p in (root / tree).rglob("*.py")
            if "__pycache__" not in p.parts
        )
    for rel in SCORER_FILES:
        p = root / rel
        if p.exists():
            out.add(p)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def combined_digest(files: Iterable[Path], repo: Path | None = None) -> tuple[str, dict]:
    """`(합산 해시, 파일별 해시)`. 경로와 내용을 함께 넣어 **파일 이름 변경도 잡는다.**"""
    root = Path(repo or REPO)
    per_file: dict[str, str] = {}
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        digest = _file_sha256(p)
        per_file[rel] = digest
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest(), per_file


def _git_state(repo: Path) -> dict:
    """참고용. **판정 근거가 아니다** — 더러운 트리에서 커밋 해시는 코드를 대표하지 않는다."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True,
                encoding="utf-8", timeout=10, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    porcelain = run("status", "--porcelain")
    return {
        "head": head,
        "dirty": None if porcelain is None else bool(porcelain.strip()),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "role": "참고. 판정 근거는 combined 파일 해시다 — 더러운 트리에서 커밋은 코드를 대표하지 않는다",
    }


def scorer_code_digest(repo: Path | None = None, *, include_files: bool = True) -> dict:
    """산출물에 실을 채점기 코드 지문.

    Args:
        include_files: 파일별 해시를 함께 실을지. 끄면 합산 해시만 남는다 — 파일 수는
            `n_files` 가 말한다(고정 상수가 아니다).

    Returns:
        `combined` 이 판정 근거다. 두 산출물의 `combined` 이 같으면 같은 채점기 소스이고,
        다르면 어느 파일이 달라졌는지 `files` 로 짚을 수 있다.
    """
    root = Path(repo or REPO)
    files = scorer_source_files(root)
    combined, per_file = combined_digest(files, root)
    out = {
        "combined": combined,
        "n_files": len(files),
        "rule": RULE,
        "algorithm": "sha256(경로 + '\\0' + sha256(파일 바이트) + '\\n' 를 정렬 순서로 이어 붙임)",
        "git": _git_state(root),
        "note": (
            "이 필드가 없는 산출물은 이 규칙이 생기기 전에 나온 것이다. **소급 계산해 "
            "끼워 넣지 않는다** — 지금 코드의 지문을 당시 기록인 척 싣는 것이 되기 때문이다"
        ),
    }
    if include_files:
        out["files"] = per_file
    return out


def relpath(path: str | Path, repo: Path | None = None) -> str:
    """산출물에 싣는 경로는 **저장소 상대**로 — CLI 에 절대경로를 줘도 산출물에 들어가지 않게
    (규약 2-6, C 34번 Minor 15). 저장소 밖이면 이름만 남긴다."""
    import os

    root = Path(repo or REPO).resolve()
    # `resolve()` 는 정션을 따라간다 — `outputs/` 는 본체로 가는 정션이라 저장소 밖으로
    # 튀어 이름만 남는다. 정션을 따라가지 않는 절대화로 상대경로를 낸다.
    p = Path(os.path.abspath(str(path)))
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.name


def hash_files(paths: Iterable[str | Path], repo: Path | None = None) -> dict[str, str | None]:
    """입력 파일(채점 레코드 등)의 sha256. 없는 파일은 None — 있는 척 적지 않는다."""
    root = Path(repo or REPO)
    out: dict[str, str | None] = {}
    for p in paths:
        q = Path(p)
        out[relpath(q, root)] = _file_sha256(q) if q.exists() else None
    return out


def stable_digest(start: dict, end: dict) -> dict:
    """긴 채점의 **시작과 끝**에 두 번 계산한 지문을 맞댄다(C 34번 Minor 15).

    채점 중에 트리가 바뀌면 끝 시점 지문은 시작 시점 코드를 대표하지 않는다. 둘이 다르면
    `stable=False` 로 남기고 어느 파일이 갈렸는지 적는다 — 그 산출물은 한 코드 상태로 나온
    것이 아니다.
    """
    same = start["combined"] == end["combined"]
    changed = sorted(
        k for k in set(start.get("files", {})) | set(end.get("files", {}))
        if start.get("files", {}).get(k) != end.get("files", {}).get(k)
    )
    return {**end, "combined_at_start": start["combined"], "stable": same,
            "files_changed_during_run": changed,
            "stability_note": ("시작·끝 지문 일치 — 한 코드 상태로 채점됐다" if same else
                               "**시작·끝 지문 불일치** — 채점 중 트리가 바뀌었다. 한 코드 상태가 아니다")}
